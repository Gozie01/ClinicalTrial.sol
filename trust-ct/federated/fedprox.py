"""
FedProx wrapper — convenience import.

FedProx is implemented in fedavg.py as run_federated(..., mu=0.01).
This module re-exports it with a mu-specific default for clarity.
"""

from .fedavg import run_federated as _run


def run_fedprox(client_data, n_rounds=20, mu=0.01, **kwargs):
    """FedProx: FedAvg with proximal penalty mu (default=0.01)."""
    return _run(client_data, n_rounds=n_rounds, mu=mu, **kwargs)
