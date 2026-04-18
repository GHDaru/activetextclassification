"""
activetextclassification.application
======================================
Camada de aplicação: orquestra o domínio sem tocar em I/O diretamente.
"""

from .config import (
    ComponentConfig,
    DataConfig,
    ALConfig,
    BaselineConfig,
    ExperimentConfig,
)
from .active_learner import ActiveLearner
from .experiment_runner import ExperimentRunner

__all__ = [
    "ComponentConfig",
    "DataConfig",
    "ALConfig",
    "BaselineConfig",
    "ExperimentConfig",
    "ActiveLearner",
    "ExperimentRunner",
]
