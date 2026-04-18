"""
activetextclassification.infrastructure
=========================================
Camada de infraestrutura: I/O de dados, cache, serialização de histórico.
"""

from .data_loader import load_and_prepare_data
from .history_store import HistoryStore

__all__ = [
    "load_and_prepare_data",
    "HistoryStore",
]
