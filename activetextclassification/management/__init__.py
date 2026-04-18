"""
activetextclassification.management
=====================================
Gerenciamento de execução de múltiplos experimentos.
"""

from .experiment_manager import ExperimentManager
from .history_manager import HistoryManager

__all__ = [
    "ExperimentManager",
    "HistoryManager",
]
