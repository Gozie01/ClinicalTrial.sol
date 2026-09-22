"""
TRUST-CT Governance — Signed oracle attestation model.

Each clinical event submitted to the ledger is wrapped in a signed attestation:

    o_i = (subjectHash, eventType, value, timestamp, nonce, sourceID, signature)

The ledger establishes:
  - integrity   (hash-chain tamper evidence)
  - ordering    (monotonic block timestamps)
  - authorization (sourceID verified against registered oracle registry)
  - non-repudiation (signature by private key of sourceID)

The ledger CANNOT independently verify:
  - Whether the original off-chain observation was clinically correct
  - Device authenticity or sensor calibration
  - Oracle correctness (the oracle may faithfully record a wrong measurement)

This module provides:
  - Attestation dataclass
  - Simulated oracle registry with key management
  - Verification function (signature check)
  - Audit log builder (ordered event stream)
"""

import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Attestation:
    subject_hash: str       # SHA-256 of patient pseudonym (not PID directly)
    event_type:   str       # e.g. "REFILL_CLAIM", "VISIT_ATTENDED", "LAB_RESULT"
    value:        str       # JSON-encoded event payload
    timestamp:    int       # Unix epoch (seconds)
    nonce:        str       # random hex string (replay prevention)
    source_id:    str       # oracle identifier (must be in registry)
    signature:    str = ""  # HMAC-SHA256(key, canonical_msg) — filled by sign()

    def canonical(self) -> str:
        """Deterministic message string for signing."""
        return "|".join([
            self.subject_hash,
            self.event_type,
            self.value,
            str(self.timestamp),
            self.nonce,
            self.source_id,
        ])


class OracleRegistry:
    """
    Simulated oracle registry: maps source_id → symmetric key.
    In production this is an on-chain mapping of oracle address → public key.
    """

    def __init__(self):
        self._keys: dict[str, bytes] = {}

    def register(self, source_id: str, key: Optional[bytes] = None) -> bytes:
        if key is None:
            key = secrets.token_bytes(32)
        self._keys[source_id] = key
        return key

    def is_registered(self, source_id: str) -> bool:
        return source_id in self._keys

    def _key(self, source_id: str) -> bytes:
        if source_id not in self._keys:
            raise KeyError(f"Unknown oracle: {source_id!r}")
        return self._keys[source_id]

    def sign(self, attestation: Attestation) -> Attestation:
        key = self._key(attestation.source_id)
        sig = hmac.new(key, attestation.canonical().encode(), hashlib.sha256).hexdigest()
        return Attestation(**{**asdict(attestation), "signature": sig})

    def verify(self, attestation: Attestation) -> bool:
        if not self.is_registered(attestation.source_id):
            return False
        key = self._key(attestation.source_id)
        expected = hmac.new(key, attestation.canonical().encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, attestation.signature)


def make_attestation(
    registry: OracleRegistry,
    source_id: str,
    subject_hash: str,
    event_type: str,
    value: dict,
    timestamp: Optional[int] = None,
) -> Attestation:
    """Create and sign a new attestation."""
    att = Attestation(
        subject_hash = subject_hash,
        event_type   = event_type,
        value        = json.dumps(value, sort_keys=True),
        timestamp    = timestamp or int(time.time()),
        nonce        = secrets.token_hex(16),
        source_id    = source_id,
    )
    return registry.sign(att)


class AuditLog:
    """
    Ordered, append-only event log (simulates on-chain ordering).
    Entries are hash-linked (each block stores hash of previous block).
    """

    def __init__(self):
        self._blocks: list[dict] = []
        self._prev_hash = "0" * 64

    def append(self, attestation: Attestation, verified: bool) -> str:
        entry = {
            "index":      len(self._blocks),
            "prev_hash":  self._prev_hash,
            "attestation": asdict(attestation),
            "verified":   verified,
        }
        entry_str   = json.dumps(entry, sort_keys=True)
        block_hash  = hashlib.sha256(entry_str.encode()).hexdigest()
        entry["block_hash"] = block_hash
        self._blocks.append(entry)
        self._prev_hash = block_hash
        return block_hash

    def verify_chain(self) -> bool:
        """Verify hash-chain integrity from genesis."""
        prev = "0" * 64
        for block in self._blocks:
            check = dict(block)
            bh = check.pop("block_hash")
            if hashlib.sha256(json.dumps(check, sort_keys=True).encode()).hexdigest() != bh:
                return False
            if check["prev_hash"] != prev:
                return False
            prev = bh
        return True

    def to_dataframe(self):
        import pandas as pd
        rows = []
        for b in self._blocks:
            att = b["attestation"]
            rows.append({
                "index":        b["index"],
                "source_id":    att["source_id"],
                "event_type":   att["event_type"],
                "timestamp":    att["timestamp"],
                "verified":     b["verified"],
                "block_hash":   b["block_hash"],
            })
        return pd.DataFrame(rows)
