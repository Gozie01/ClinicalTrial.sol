import statistics
import time
from dataclasses import dataclass

import pandas as pd
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

from blockchain_contract_runtime import compile_contracts, deploy_local_architecture, prepare_canonical_flow, tx_success


PURECHAIN_RPC = "https://purechainnode.com:8547"
SEPOLIA_RPC = "https://ethereum-sepolia-rpc.publicnode.com"
DUMMY_FROM = Web3.to_checksum_address("0x000000000000000000000000000000000000dEaD")
DUMMY_COORDINATOR = Web3.to_checksum_address("0x0000000000000000000000000000000000000A11")
DUMMY_VALIDATORS = [
    Web3.to_checksum_address("0x0000000000000000000000000000000000000B11"),
    Web3.to_checksum_address("0x0000000000000000000000000000000000000B12"),
    Web3.to_checksum_address("0x0000000000000000000000000000000000000B13"),
]
DUMMY_TRIAL_MANAGER = Web3.to_checksum_address("0x0000000000000000000000000000000000000C11")
DUMMY_TOKEN = Web3.to_checksum_address("0x0000000000000000000000000000000000000C12")


@dataclass(frozen=True)
class NetworkProfile:
    name: str
    rpc_url: str
    native_symbol: str
    expected_chain_id: int | None = None


NETWORKS = [
    NetworkProfile("PureChain", PURECHAIN_RPC, "PURE", 900520900520),
    NetworkProfile("Ethereum Sepolia", SEPOLIA_RPC, "ETH", 11155111),
]


def _connect(profile: NetworkProfile):
    w3 = Web3(Web3.HTTPProvider(profile.rpc_url, request_kwargs={"timeout": 20}))
    if profile.name == "PureChain":
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    if not w3.is_connected():
        raise RuntimeError(f"Unable to connect to {profile.name} RPC")
    return w3


def _block_query_latency_ms(w3: Web3, samples: int = 5):
    latencies = []
    for _ in range(samples):
        started = time.perf_counter()
        w3.eth.get_block("latest")
        latencies.append((time.perf_counter() - started) * 1000.0)
    return latencies


def _average_block_time_sec(w3: Web3, depth: int = 20):
    latest = int(w3.eth.block_number)
    if latest <= 1:
        return 0.0
    start = max(1, latest - depth)
    timestamps = []
    for block_number in range(start, latest + 1):
        block = w3.eth.get_block(block_number)
        timestamps.append(int(block["timestamp"]))
    if len(timestamps) < 2:
        return 0.0
    deltas = [right - left for left, right in zip(timestamps[:-1], timestamps[1:]) if right >= left]
    return float(statistics.mean(deltas)) if deltas else 0.0


def _estimate_total_deployment_gas(w3: Web3, evm_version: str):
    artifacts = compile_contracts(evm_version=evm_version)

    trial_manager = w3.eth.contract(
        abi=artifacts["TrialManager"]["abi"],
        bytecode=artifacts["TrialManager"]["bin"],
    )
    ct_token = w3.eth.contract(
        abi=artifacts["CTToken"]["abi"],
        bytecode=artifacts["CTToken"]["bin"],
    )
    incentive_manager = w3.eth.contract(
        abi=artifacts["IncentiveManager"]["abi"],
        bytecode=artifacts["IncentiveManager"]["bin"],
    )

    gas_trial_manager = int(
        trial_manager.constructor([DUMMY_COORDINATOR]).estimate_gas({"from": DUMMY_FROM})
    )
    gas_ct_token = int(
        ct_token.constructor(DUMMY_FROM).estimate_gas({"from": DUMMY_FROM})
    )
    gas_incentive_manager = int(
        incentive_manager.constructor(
            DUMMY_TRIAL_MANAGER,
            DUMMY_TOKEN,
            DUMMY_COORDINATOR,
            DUMMY_VALIDATORS,
            2,
            3,
        ).estimate_gas({"from": DUMMY_FROM})
    )
    return {
        "TrialManager": gas_trial_manager,
        "CTToken": gas_ct_token,
        "IncentiveManager": gas_incentive_manager,
        "total": gas_trial_manager + gas_ct_token + gas_incentive_manager,
    }


def _supports_cancun_bytecode(w3: Web3):
    try:
        _estimate_total_deployment_gas(w3, evm_version="cancun")
        return True
    except Exception:
        return False


def _measure_local_reward_execution_gas():
    architecture = deploy_local_architecture()
    flow = prepare_canonical_flow(architecture, trial_id=77, visit_id=9, reward_amount=25, incentive_level=12)
    tx_success(architecture.incentive_manager.functions.approveReward(flow["claim_id"]), architecture.actors.validator_a)
    tx_success(architecture.incentive_manager.functions.approveReward(flow["claim_id"]), architecture.actors.validator_b)
    architecture.w3.provider.ethereum_tester.mine_blocks(4)
    execute_receipt = tx_success(
        architecture.incentive_manager.functions.executeReward(flow["claim_id"]),
        architecture.actors.owner,
    )
    return int(execute_receipt.gasUsed)


def benchmark_networks():
    reward_execution_gas = _measure_local_reward_execution_gas()
    raw_rows = []
    summary_rows = []

    for profile in NETWORKS:
        try:
            w3 = _connect(profile)
            block_latency_samples = _block_query_latency_ms(w3, samples=5)
            avg_block_time = _average_block_time_sec(w3, depth=20)
            gas_price_wei = int(w3.eth.gas_price)
            latest_block = int(w3.eth.block_number)
            observed_chain_id = int(w3.eth.chain_id)
            client_version = None
            try:
                client_version = w3.client_version
            except Exception:
                client_version = None

            supports_cancun = _supports_cancun_bytecode(w3)
            paris_deployment = _estimate_total_deployment_gas(w3, evm_version="paris")

            for sample_id, latency_ms in enumerate(block_latency_samples, start=1):
                raw_rows.append(
                    {
                        "network": profile.name,
                        "sample_id": sample_id,
                        "rpc_url": profile.rpc_url,
                        "native_symbol": profile.native_symbol,
                        "expected_chain_id": profile.expected_chain_id,
                        "observed_chain_id": observed_chain_id,
                        "latest_block": latest_block,
                        "gas_price_wei": gas_price_wei,
                        "block_latency_ms": latency_ms,
                        "avg_block_time_sec": avg_block_time,
                        "deployment_gas_trial_manager_paris": paris_deployment["TrialManager"],
                        "deployment_gas_ct_token_paris": paris_deployment["CTToken"],
                        "deployment_gas_incentive_manager_paris": paris_deployment["IncentiveManager"],
                        "deployment_gas_paris": paris_deployment["total"],
                        "local_reward_execution_gas": reward_execution_gas,
                        "supports_cancun_bytecode": supports_cancun,
                        "client_version": client_version,
                        "status": "ok",
                    }
                )

            summary_rows.append(
                {
                    "network": profile.name,
                    "rpc_url": profile.rpc_url,
                    "native_symbol": profile.native_symbol,
                    "expected_chain_id": profile.expected_chain_id,
                    "observed_chain_id": observed_chain_id,
                    "latest_block": latest_block,
                    "gas_price_wei": gas_price_wei,
                    "block_latency_ms": float(statistics.mean(block_latency_samples)),
                    "avg_block_time_sec": avg_block_time,
                    "deployment_gas_trial_manager_paris": paris_deployment["TrialManager"],
                    "deployment_gas_ct_token_paris": paris_deployment["CTToken"],
                    "deployment_gas_incentive_manager_paris": paris_deployment["IncentiveManager"],
                    "deployment_gas_paris": paris_deployment["total"],
                    "deployment_cost_paris_native": (paris_deployment["total"] * gas_price_wei) / 1e18,
                    "projected_reward_execution_gas": reward_execution_gas,
                    "projected_reward_execution_cost_native": (reward_execution_gas * gas_price_wei) / 1e18,
                    "supports_cancun_bytecode": supports_cancun,
                    "client_version": client_version,
                    "status": "ok",
                }
            )
        except Exception as exc:
            raw_rows.append(
                {
                    "network": profile.name,
                    "sample_id": 0,
                    "rpc_url": profile.rpc_url,
                    "native_symbol": profile.native_symbol,
                    "expected_chain_id": profile.expected_chain_id,
                    "observed_chain_id": None,
                    "latest_block": None,
                    "gas_price_wei": None,
                    "block_latency_ms": None,
                    "avg_block_time_sec": None,
                    "deployment_gas_trial_manager_paris": None,
                    "deployment_gas_ct_token_paris": None,
                    "deployment_gas_incentive_manager_paris": None,
                    "deployment_gas_paris": None,
                    "local_reward_execution_gas": reward_execution_gas,
                    "supports_cancun_bytecode": False,
                    "client_version": None,
                    "status": f"failed:{exc.__class__.__name__}",
                }
            )
            summary_rows.append(
                {
                    "network": profile.name,
                    "rpc_url": profile.rpc_url,
                    "native_symbol": profile.native_symbol,
                    "expected_chain_id": profile.expected_chain_id,
                    "observed_chain_id": None,
                    "latest_block": None,
                    "gas_price_wei": None,
                    "block_latency_ms": None,
                    "avg_block_time_sec": None,
                    "deployment_gas_trial_manager_paris": None,
                    "deployment_gas_ct_token_paris": None,
                    "deployment_gas_incentive_manager_paris": None,
                    "deployment_gas_paris": None,
                    "deployment_cost_paris_native": None,
                    "projected_reward_execution_gas": reward_execution_gas,
                    "projected_reward_execution_cost_native": None,
                    "supports_cancun_bytecode": False,
                    "client_version": None,
                    "status": f"failed:{exc.__class__.__name__}",
                }
            )

    return pd.DataFrame(raw_rows), pd.DataFrame(summary_rows)


if __name__ == "__main__":
    raw, summary = benchmark_networks()
    print(summary.to_string(index=False))
