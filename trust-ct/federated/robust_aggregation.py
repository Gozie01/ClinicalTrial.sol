"""
TRUST-CT — Robust aggregation rules (re-exported from security/attacks.py).

Convenience module so federated experiments can import:
    from federated.robust_aggregation import coordinate_median, trimmed_mean, krum
"""

from security.attacks import coordinate_median, trimmed_mean, krum  # noqa: F401
