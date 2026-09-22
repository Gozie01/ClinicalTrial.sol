import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from eth_tester.exceptions import TransactionFailed
from solcx import compile_files, get_installed_solc_versions, install_solc, set_solc_version
from web3 import Web3
from web3.exceptions import ContractLogicError
from web3.providers.eth_tester import EthereumTesterProvider

from ipfs_storage import add_json_to_ipfs


ROOT = Path(__file__).resolve().parent
CONTRACT_FILES = [
    ROOT / "contracts" / "CTToken.sol",
    ROOT / "contracts" / "TrialManager.sol",
    ROOT / "contracts" / "IncentiveManager.sol",
]
SOLC_VERSION = "0.8.24"


@dataclass
class LocalActors:
    owner: str
    coordinator: str
    validator_a: str
    validator_b: str
    validator_c: str
    participant_a: str
    participant_b: str


@dataclass
class LocalArchitecture:
    w3: Web3
    ct_token: any
    trial_manager: any
    incentive_manager: any
    actors: LocalActors


def ensure_solc(version: str = SOLC_VERSION):
    versions = {str(item) for item in get_installed_solc_versions()}
    if version not in versions:
        install_solc(version)
    set_solc_version(version)


def compile_contracts(evm_version: str | None = None):
    ensure_solc()
    compiled = compile_files(
        [str(path) for path in CONTRACT_FILES],
        output_values=["abi", "bin"],
        optimize=True,
        optimize_runs=200,
        evm_version=evm_version,
    )

    def artifact(path: Path, name: str):
        key = f"{path.as_posix()}:{name}"
        if key in compiled:
            return compiled[key]
        fallback = f"contracts/{path.name}:{name}"
        return compiled[fallback]

    return {
        "CTToken": artifact(ROOT / "contracts" / "CTToken.sol", "CTToken"),
        "TrialManager": artifact(ROOT / "contracts" / "TrialManager.sol", "TrialManager"),
        "IncentiveManager": artifact(ROOT / "contracts" / "IncentiveManager.sol", "IncentiveManager"),
    }


def contract_abi_bytecode(name: str, evm_version: str | None = None):
    artifacts = compile_contracts(evm_version=evm_version)
    artifact = artifacts[name]
    return artifact["abi"], artifact["bin"]


def deploy_local_architecture():
    artifacts = compile_contracts()
    provider = EthereumTesterProvider()
    w3 = Web3(provider)
    accounts = w3.eth.accounts
    actors = LocalActors(
        owner=accounts[0],
        coordinator=accounts[1],
        validator_a=accounts[2],
        validator_b=accounts[3],
        validator_c=accounts[4],
        participant_a=accounts[5],
        participant_b=accounts[6],
    )

    trial_manager_factory = w3.eth.contract(
        abi=artifacts["TrialManager"]["abi"],
        bytecode=artifacts["TrialManager"]["bin"],
    )
    tx_hash = trial_manager_factory.constructor([actors.coordinator]).transact({"from": actors.owner})
    trial_manager_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    trial_manager = w3.eth.contract(address=trial_manager_receipt.contractAddress, abi=artifacts["TrialManager"]["abi"])
    trial_manager.functions.registerParticipant(actors.participant_a, True).transact({"from": actors.owner})
    trial_manager.functions.registerParticipant(actors.participant_b, True).transact({"from": actors.owner})

    token_factory = w3.eth.contract(
        abi=artifacts["CTToken"]["abi"],
        bytecode=artifacts["CTToken"]["bin"],
    )
    tx_hash = token_factory.constructor(actors.owner).transact({"from": actors.owner})
    token_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    ct_token = w3.eth.contract(address=token_receipt.contractAddress, abi=artifacts["CTToken"]["abi"])

    incentive_factory = w3.eth.contract(
        abi=artifacts["IncentiveManager"]["abi"],
        bytecode=artifacts["IncentiveManager"]["bin"],
    )
    tx_hash = incentive_factory.constructor(
        trial_manager.address,
        ct_token.address,
        actors.coordinator,
        [actors.validator_a, actors.validator_b, actors.validator_c],
        2,
        3,
    ).transact({"from": actors.owner})
    incentive_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    incentive_manager = w3.eth.contract(
        address=incentive_receipt.contractAddress,
        abi=artifacts["IncentiveManager"]["abi"],
    )

    ct_token.functions.setMinter(incentive_manager.address).transact({"from": actors.owner})

    return LocalArchitecture(
        w3=w3,
        ct_token=ct_token,
        trial_manager=trial_manager,
        incentive_manager=incentive_manager,
        actors=actors,
    )


def tx_success(fn_call, sender):
    tx_hash = fn_call.transact({"from": sender})
    return fn_call.w3.eth.wait_for_transaction_receipt(tx_hash)


def tx_reverts(fn_call, sender):
    try:
        fn_call.transact({"from": sender})
    except (TransactionFailed, ValueError, ContractLogicError):
        return True
    return False


def store_ipfs_json(artifact_name: str, payload: dict):
    result = add_json_to_ipfs(artifact_name=artifact_name, payload=payload)
    return result.cid, Web3.to_bytes(hexstr=result.cid_keccak256)


def prepare_canonical_flow(architecture: LocalArchitecture, trial_id: int, visit_id: int, reward_amount: int, incentive_level: int):
    w3 = architecture.w3
    actors = architecture.actors
    trial_manager = architecture.trial_manager
    incentive_manager = architecture.incentive_manager

    model_cid, model_hash = store_ipfs_json(
        f"model_update_trial{trial_id}_visit{visit_id}.json",
        {
            "artifact_type": "model_update",
            "trial_id": trial_id,
            "visit_id": visit_id,
            "model_hash_sha256": "sha256-demo-model-hash",
            "round": 1,
        },
    )
    model_receipt = tx_success(
        trial_manager.functions.anchorModelUpdate(
            model_cid,
            model_hash,
            Web3.keccak(text=f"model-round-{trial_id}-{visit_id}"),
            1,
            10_000 + visit_id,
        ),
        actors.coordinator,
    )

    attendance_cid, attendance_hash = store_ipfs_json(
        f"attendance_trial{trial_id}_visit{visit_id}.json",
        {
            "artifact_type": "attendance",
            "trial_id": trial_id,
            "visit_id": visit_id,
            "participant": actors.participant_a,
            "status": "attended",
        },
    )
    tx_success(
        trial_manager.functions.logAttendance(
            actors.participant_a,
            trial_id,
            visit_id,
            20_000 + visit_id,
            attendance_cid,
            attendance_hash,
        ),
        actors.coordinator,
    )

    incentive_cid, incentive_hash = store_ipfs_json(
        f"incentive_trial{trial_id}_visit{visit_id}.json",
        {
            "artifact_type": "incentive_decision",
            "trial_id": trial_id,
            "visit_id": visit_id,
            "incentive_level": incentive_level,
            "policy": "integrated_runtime_demo",
        },
    )
    decision_receipt = tx_success(
        trial_manager.functions.logIncentiveDecision(
            visit_id,
            incentive_level,
            incentive_cid,
            incentive_hash,
            30_000 + visit_id,
        ),
        actors.coordinator,
    )

    propose_receipt = tx_success(
        incentive_manager.functions.proposeReward(
            actors.participant_a,
            trial_id,
            visit_id,
            reward_amount,
            incentive_level,
            incentive_cid,
            incentive_hash,
            40_000 + visit_id,
        ),
        actors.coordinator,
    )
    claim_id = int(incentive_manager.functions.claimCount().call())

    return {
        "claim_id": claim_id,
        "model_anchor_gas": int(model_receipt.gasUsed),
        "decision_anchor_gas": int(decision_receipt.gasUsed),
        "reward_proposal_gas": int(propose_receipt.gasUsed),
        "model_cid": model_cid,
        "attendance_cid": attendance_cid,
        "incentive_cid": incentive_cid,
    }


def run_contract_security_matrix():
    rows = []

    # Canonical successful execution
    architecture = deploy_local_architecture()
    flow = prepare_canonical_flow(architecture, trial_id=1, visit_id=1, reward_amount=25, incentive_level=12)
    tx_success(architecture.incentive_manager.functions.approveReward(flow["claim_id"]), architecture.actors.validator_a)
    tx_success(architecture.incentive_manager.functions.approveReward(flow["claim_id"]), architecture.actors.validator_b)
    architecture.w3.provider.ethereum_tester.mine_blocks(4)
    execute_receipt = tx_success(architecture.incentive_manager.functions.executeReward(flow["claim_id"]), architecture.actors.owner)
    token_balance = int(architecture.ct_token.functions.balanceOf(architecture.actors.participant_a).call())
    rows.append(
        {
            "scenario": "canonical_reward_execution",
            "blocked": 0,
            "status": "succeeded",
            "gas_used": int(execute_receipt.gasUsed),
            "extra_metric": token_balance / 1e18,
        }
    )
    rows.append(
        {
            "scenario": "model_update_anchor",
            "blocked": 0,
            "status": "succeeded",
            "gas_used": flow["model_anchor_gas"],
            "extra_metric": 1,
        }
    )
    rows.append(
        {
            "scenario": "incentive_decision_anchor",
            "blocked": 0,
            "status": "succeeded",
            "gas_used": flow["decision_anchor_gas"],
            "extra_metric": 1,
        }
    )

    # Replay attendance attack
    architecture = deploy_local_architecture()
    attendance_cid, attendance_hash = store_ipfs_json(
        "replay_attendance.json",
        {"artifact_type": "attendance", "participant": architecture.actors.participant_a, "status": "attended"},
    )
    tx_success(
        architecture.trial_manager.functions.logAttendance(
            architecture.actors.participant_a,
            1,
            2,
            200,
            attendance_cid,
            attendance_hash,
        ),
        architecture.actors.coordinator,
    )
    rows.append(
        {
            "scenario": "replay_attendance",
            "blocked": int(
                tx_reverts(
                    architecture.trial_manager.functions.logAttendance(
                        architecture.actors.participant_a,
                        1,
                        2,
                        200,
                        attendance_cid,
                        attendance_hash,
                    ),
                    architecture.actors.coordinator,
                )
            ),
            "status": "blocked",
            "gas_used": None,
            "extra_metric": None,
        }
    )

    # Invalid CID hash blocked
    architecture = deploy_local_architecture()
    rows.append(
        {
            "scenario": "invalid_cid_hash",
            "blocked": int(
                tx_reverts(
                    architecture.trial_manager.functions.logAttendance(
                        architecture.actors.participant_a,
                        1,
                        3,
                        210,
                        "bafy-invalid-attendance",
                        Web3.keccak(text="wrong-cid"),
                    ),
                    architecture.actors.coordinator,
                )
            ),
            "status": "blocked",
            "gas_used": None,
            "extra_metric": None,
        }
    )

    # Unauthorized participant manipulation
    architecture = deploy_local_architecture()
    attendance_cid, attendance_hash = store_ipfs_json(
        "forgery_attendance.json",
        {"artifact_type": "attendance", "participant": architecture.actors.participant_b, "status": "attended"},
    )
    rows.append(
        {
            "scenario": "malicious_participant_forgery",
            "blocked": int(
                tx_reverts(
                    architecture.trial_manager.functions.logAttendance(
                        architecture.actors.participant_b,
                        1,
                        4,
                        220,
                        attendance_cid,
                        attendance_hash,
                    ),
                    architecture.actors.participant_a,
                )
            ),
            "status": "blocked",
            "gas_used": None,
            "extra_metric": None,
        }
    )

    # Reward proposal without attendance
    architecture = deploy_local_architecture()
    incentive_cid, incentive_hash = store_ipfs_json(
        "missing_attendance_incentive.json",
        {"artifact_type": "incentive_decision", "visit_id": 5, "incentive_level": 8},
    )
    rows.append(
        {
            "scenario": "attendance_missing_reward_proposal",
            "blocked": int(
                tx_reverts(
                    architecture.incentive_manager.functions.proposeReward(
                        architecture.actors.participant_a,
                        1,
                        5,
                        10,
                        8,
                        incentive_cid,
                        incentive_hash,
                        230,
                    ),
                    architecture.actors.coordinator,
                )
            ),
            "status": "blocked",
            "gas_used": None,
            "extra_metric": None,
        }
    )

    # Finality not reached
    architecture = deploy_local_architecture()
    tx_success(architecture.incentive_manager.functions.setFinalityDelay(8), architecture.actors.owner)
    flow = prepare_canonical_flow(architecture, trial_id=1, visit_id=6, reward_amount=18, incentive_level=12)
    tx_success(architecture.incentive_manager.functions.approveReward(flow["claim_id"]), architecture.actors.validator_a)
    tx_success(architecture.incentive_manager.functions.approveReward(flow["claim_id"]), architecture.actors.validator_b)
    rows.append(
        {
            "scenario": "pre_finality_execution",
            "blocked": int(
                tx_reverts(
                    architecture.incentive_manager.functions.executeReward(flow["claim_id"]),
                    architecture.actors.owner,
                )
            ),
            "status": "blocked",
            "gas_used": None,
            "extra_metric": None,
        }
    )

    # Quorum not reached
    architecture = deploy_local_architecture()
    flow = prepare_canonical_flow(architecture, trial_id=1, visit_id=7, reward_amount=19, incentive_level=8)
    tx_success(architecture.incentive_manager.functions.approveReward(flow["claim_id"]), architecture.actors.validator_a)
    architecture.w3.provider.ethereum_tester.mine_blocks(4)
    rows.append(
        {
            "scenario": "insufficient_quorum_execution",
            "blocked": int(
                tx_reverts(
                    architecture.incentive_manager.functions.executeReward(flow["claim_id"]),
                    architecture.actors.owner,
                )
            ),
            "status": "blocked",
            "gas_used": None,
            "extra_metric": None,
        }
    )

    # Double approval blocked
    architecture = deploy_local_architecture()
    flow = prepare_canonical_flow(architecture, trial_id=1, visit_id=8, reward_amount=20, incentive_level=12)
    tx_success(architecture.incentive_manager.functions.approveReward(flow["claim_id"]), architecture.actors.validator_a)
    rows.append(
        {
            "scenario": "duplicate_validator_approval",
            "blocked": int(
                tx_reverts(
                    architecture.incentive_manager.functions.approveReward(flow["claim_id"]),
                    architecture.actors.validator_a,
                )
            ),
            "status": "blocked",
            "gas_used": None,
            "extra_metric": None,
        }
    )

    # Conflicting reward execution blocked
    architecture = deploy_local_architecture()
    flow_a = prepare_canonical_flow(architecture, trial_id=1, visit_id=9, reward_amount=10, incentive_level=4)
    incentive_cid, incentive_hash = store_ipfs_json(
        "conflict_incentive.json",
        {"artifact_type": "incentive_decision", "visit_id": 9, "incentive_level": 12},
    )
    tx_success(
        architecture.trial_manager.functions.logIncentiveDecision(9, 12, incentive_cid, incentive_hash, 90_999),
        architecture.actors.coordinator,
    )
    tx_success(
        architecture.incentive_manager.functions.proposeReward(
            architecture.actors.participant_a,
            1,
            9,
            12,
            12,
            incentive_cid,
            incentive_hash,
            91_000,
        ),
        architecture.actors.coordinator,
    )
    claim_a = flow_a["claim_id"]
    claim_b = int(architecture.incentive_manager.functions.claimCount().call())
    for claim_id in (claim_a, claim_b):
        tx_success(architecture.incentive_manager.functions.approveReward(claim_id), architecture.actors.validator_a)
        tx_success(architecture.incentive_manager.functions.approveReward(claim_id), architecture.actors.validator_b)
    architecture.w3.provider.ethereum_tester.mine_blocks(4)
    tx_success(architecture.incentive_manager.functions.executeReward(claim_a), architecture.actors.owner)
    rows.append(
        {
            "scenario": "conflicting_reward_execution",
            "blocked": int(
                tx_reverts(
                    architecture.incentive_manager.functions.executeReward(claim_b),
                    architecture.actors.owner,
                )
            ),
            "status": "blocked",
            "gas_used": None,
            "extra_metric": None,
        }
    )

    # Cancelled claim blocked
    architecture = deploy_local_architecture()
    flow = prepare_canonical_flow(architecture, trial_id=1, visit_id=10, reward_amount=14, incentive_level=8)
    tx_success(architecture.incentive_manager.functions.approveReward(flow["claim_id"]), architecture.actors.validator_a)
    tx_success(architecture.incentive_manager.functions.approveReward(flow["claim_id"]), architecture.actors.validator_b)
    tx_success(architecture.incentive_manager.functions.cancelClaim(flow["claim_id"]), architecture.actors.coordinator)
    architecture.w3.provider.ethereum_tester.mine_blocks(4)
    rows.append(
        {
            "scenario": "cancelled_claim_execution",
            "blocked": int(
                tx_reverts(
                    architecture.incentive_manager.functions.executeReward(flow["claim_id"]),
                    architecture.actors.owner,
                )
            ),
            "status": "blocked",
            "gas_used": None,
            "extra_metric": None,
        }
    )

    raw_df = pd.DataFrame(rows)
    summary_df = (
        raw_df.groupby("scenario")[["blocked", "gas_used", "extra_metric"]]
        .agg({"blocked": "mean", "gas_used": "mean", "extra_metric": "mean"})
        .reset_index()
        .rename(columns={"blocked": "blocked_rate", "gas_used": "mean_gas_used", "extra_metric": "mean_extra_metric"})
    )
    return raw_df, summary_df


def export_contract_artifact(output_dir: Path):
    artifacts = compile_contracts()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {}
    for name, artifact in artifacts.items():
        artifact_path = output_dir / f"{name}.local_artifact.json"
        artifact_path.write_text(json.dumps({"abi": artifact["abi"], "bytecode": artifact["bin"]}, indent=2), encoding="utf-8")
        artifact_paths[name] = artifact_path
    return artifact_paths
