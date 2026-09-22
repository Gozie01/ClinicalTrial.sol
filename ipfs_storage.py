import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests
from web3 import Web3


ROOT = Path(__file__).resolve().parent
KUBO_EXE = ROOT / ".tools" / "kubo" / "v0.41.0" / "kubo" / "ipfs.exe"
KUBO_REPO = ROOT / ".tools" / "kubo" / "repo"
DEFAULT_API_URL = "http://127.0.0.1:5001"
DEFAULT_GATEWAY_URL = "http://127.0.0.1:8080"
CACHE_DIR = ROOT / "cache" / "ipfs_artifacts"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class IPFSStoreResult:
    artifact_name: str
    cid: str
    cid_keccak256: str
    size_bytes: int
    api_url: str
    gateway_url: str
    local_cache_path: str
    manifest_path: str
    content_sha256: str


def _sha256_bytes(payload: bytes):
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _keccak_cid(cid: str):
    return Web3.keccak(text=cid).hex()


def ipfs_api_available(api_url: str = DEFAULT_API_URL, timeout: float = 3.0):
    try:
        response = requests.post(f"{api_url}/api/v0/version", timeout=timeout)
        response.raise_for_status()
        return True
    except Exception:
        return False


def ensure_local_kubo_daemon(
    api_url: str = DEFAULT_API_URL,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    startup_timeout: float = 20.0,
):
    if ipfs_api_available(api_url=api_url):
        return {"started": False, "api_url": api_url, "gateway_url": gateway_url}

    if not KUBO_EXE.exists():
        raise FileNotFoundError(f"Kubo executable not found at {KUBO_EXE}")
    if not KUBO_REPO.exists():
        raise FileNotFoundError(f"Kubo repo not found at {KUBO_REPO}")

    env = os.environ.copy()
    env["IPFS_PATH"] = str(KUBO_REPO)
    env["IPFS_TELEMETRY"] = "off"

    subprocess.Popen(
        [str(KUBO_EXE), "daemon", "--init=false"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if ipfs_api_available(api_url=api_url, timeout=2.0):
            return {"started": True, "api_url": api_url, "gateway_url": gateway_url}
        time.sleep(0.5)

    raise RuntimeError("Kubo daemon did not become ready within the startup timeout")


def _write_manifest(result: IPFSStoreResult, extra: dict | None = None):
    payload = asdict(result)
    if extra:
        payload["extra"] = extra
    manifest_path = Path(result.manifest_path)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def add_bytes_to_ipfs(
    artifact_name: str,
    payload: bytes,
    api_url: str = DEFAULT_API_URL,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    extra_metadata: dict | None = None,
):
    ensure_local_kubo_daemon(api_url=api_url, gateway_url=gateway_url)

    files = {"file": (artifact_name, payload)}
    response = requests.post(
        f"{api_url}/api/v0/add",
        params={"pin": "true", "cid-version": "1"},
        files=files,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    cid = data["Hash"]

    cache_path = CACHE_DIR / artifact_name
    cache_path.write_bytes(payload)

    manifest_path = CACHE_DIR / f"{artifact_name}.manifest.json"
    result = IPFSStoreResult(
        artifact_name=artifact_name,
        cid=cid,
        cid_keccak256=_keccak_cid(cid),
        size_bytes=len(payload),
        api_url=api_url,
        gateway_url=f"{gateway_url}/ipfs/{cid}",
        local_cache_path=str(cache_path),
        manifest_path=str(manifest_path),
        content_sha256=_sha256_bytes(payload),
    )
    _write_manifest(result, extra=extra_metadata)
    return result


def add_json_to_ipfs(
    artifact_name: str,
    payload: dict,
    api_url: str = DEFAULT_API_URL,
    gateway_url: str = DEFAULT_GATEWAY_URL,
):
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    return add_bytes_to_ipfs(
        artifact_name=artifact_name,
        payload=body,
        api_url=api_url,
        gateway_url=gateway_url,
        extra_metadata={"content_type": "application/json"},
    )


def fetch_ipfs_text(cid: str, api_url: str = DEFAULT_API_URL):
    ensure_local_kubo_daemon(api_url=api_url)
    response = requests.post(
        f"{api_url}/api/v0/cat",
        params={"arg": cid},
        timeout=30,
    )
    response.raise_for_status()
    return response.content.decode("utf-8")


def verify_ipfs_artifact(cid: str, expected_sha256: str, api_url: str = DEFAULT_API_URL):
    response = requests.post(
        f"{api_url}/api/v0/cat",
        params={"arg": cid},
        timeout=30,
    )
    response.raise_for_status()
    observed = _sha256_bytes(response.content)
    return observed == expected_sha256, observed
