"""
activetextclassification.domain
================================
Camada de domínio: contratos (interfaces), entidades e métricas puras.
Sem dependências de I/O ou infraestrutura.
"""

from .interfaces import (
    IClassifier,
    IEmbedder,
    IOracle,
    IQueryStrategy,
    IColdStart,
)
from .entities import (
    Budget,
    IterationRecord,
    ExperimentResult,
)
from .metrics import compute_accuracy, compute_f1_macro, compute_lce

__all__ = [
    # Interfaces
    "IClassifier",
    "IEmbedder",
    "IOracle",
    "IQueryStrategy",
    "IColdStart",
    # Entities
    "Budget",
    "IterationRecord",
    "ExperimentResult",
    # Metrics
    "compute_accuracy",
    "compute_f1_macro",
    "compute_lce",
]
