"""
activetextclassification.domain.entities
=========================================
Objetos de valor e entidades imutáveis do domínio.
Sem dependências de I/O ou infraestrutura.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Budget — define os critérios de parada do loop AL
# ---------------------------------------------------------------------------

@dataclass
class Budget:
    """Encapsula todos os critérios de parada do loop de Aprendizado Ativo."""

    max_iterations: int = 1000
    """Número máximo de iterações do loop AL."""

    target_budget_pct: float = 1.0
    """
    Proporção máxima do pool U original que pode ser rotulada.
    Ex: 0.30 = rotular até 30% do U inicial.
    """

    early_stopping_metric: Optional[str] = None
    """
    Métrica monitorada para parada antecipada.
    Valores válidos: 'external_acc', 'external_f1', 'internal_acc', 'internal_f1'.
    None desativa a parada antecipada.
    """

    early_stopping_patience: Optional[int] = None
    """
    Número de iterações sem melhora antes de parar.
    None desativa a parada antecipada.
    """

    early_stopping_tolerance: float = 0.001
    """Variação mínima considerada como melhora."""


# ---------------------------------------------------------------------------
# IterationRecord — resultado de uma única iteração
# ---------------------------------------------------------------------------

@dataclass
class IterationRecord:
    """Registra as métricas e tempos de uma única iteração do loop AL."""

    iteration: int
    l_size: int
    u_size: int
    status: str = "COMPLETED_ITERATION"

    # Métricas de avaliação
    internal_acc: float = float("nan")
    internal_f1: float = float("nan")
    external_acc: float = float("nan")
    external_f1: float = float("nan")

    # Durações (segundos)
    iteration_duration_sec: float = float("nan")
    train_duration_sec: float = float("nan")
    eval_duration_sec: float = float("nan")
    query_duration_sec: float = float("nan")
    update_duration_sec: float = float("nan")

    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "L_size": self.l_size,
            "U_size": self.u_size,
            "status": self.status,
            "internal_acc": self.internal_acc,
            "internal_f1": self.internal_f1,
            "external_acc": self.external_acc,
            "external_f1": self.external_f1,
            "iteration_duration_sec": self.iteration_duration_sec,
            "train_duration_sec": self.train_duration_sec,
            "eval_duration_sec": self.eval_duration_sec,
            "query_duration_sec": self.query_duration_sec,
            "update_duration_sec": self.update_duration_sec,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# ExperimentResult — resultado completo de um experimento
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """
    Resultado completo de um experimento de Aprendizado Ativo.
    Retornado por ``ActiveLearner.run()``.
    """

    experiment_name: str
    status: str = "INITIALIZED"
    history: List[IterationRecord] = field(default_factory=list)
    baseline_metrics: Dict[str, Any] = field(default_factory=dict)
    total_duration_sec: Optional[float] = None
    error_message: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def to_dataframe(self) -> pd.DataFrame:
        """Converte o histórico de iterações em um DataFrame."""
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.history])

    def to_summary_dict(self) -> Dict[str, Any]:
        """Retorna um dicionário resumido apto para serialização JSON."""
        return {
            "experiment_name": self.experiment_name,
            "status": self.status,
            "total_duration_sec": self.total_duration_sec,
            "error_message": self.error_message,
            "baseline_metrics": self.baseline_metrics,
            "history_data": [r.to_dict() for r in self.history],
            "config": self.config,
        }
