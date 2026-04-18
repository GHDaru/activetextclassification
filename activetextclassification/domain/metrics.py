"""
activetextclassification.domain.metrics
========================================
Funções puras de avaliação.  Sem estado, sem I/O.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score as _sk_f1


def compute_accuracy(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
) -> float:
    """
    Acurácia de classificação.

    Returns:
        float entre 0 e 1, ou ``float('nan')`` se os arrays forem inválidos.
    """
    try:
        return float(accuracy_score(y_true, y_pred))
    except Exception:
        return float("nan")


def compute_f1_macro(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    labels: Optional[List[str]] = None,
    zero_division: float = 0.0,
) -> float:
    """
    F1-score macro.

    Args:
        y_true:         Rótulos verdadeiros.
        y_pred:         Rótulos preditos.
        labels:         Lista completa de rótulos possíveis.  Quando fornecida,
                        inclui classes ausentes na avaliação (útil para F1 macro
                        globalizado).
        zero_division:  Valor para classes sem suporte (padrão 0).

    Returns:
        float entre 0 e 1, ou ``float('nan')`` se os arrays forem inválidos.
    """
    try:
        return float(
            _sk_f1(
                y_true,
                y_pred,
                average="macro",
                labels=labels,
                zero_division=zero_division,
            )
        )
    except Exception:
        return float("nan")


def compute_lce(
    l_sizes: Union[List[int], np.ndarray, "pd.Series"],
    performance_scores: Union[List[float], np.ndarray, "pd.Series"],
    baseline_performance: float,
) -> float:
    """
    Learning Curve Efficiency (LCE).

    LCE = Área sob a curva de aprendizado real /
          Área sob a curva ideal (retângulo de baseline × intervalo de L).

    Args:
        l_sizes:             Tamanhos do conjunto rotulado (eixo X), ordenados.
        performance_scores:  Métricas correspondentes (eixo Y).
        baseline_performance: Valor de referência (baseline).

    Returns:
        LCE (float), ou ``float('nan')`` em caso de dados inválidos.
    """
    if baseline_performance is None or pd.isna(baseline_performance):
        return float("nan")
    if baseline_performance <= 0:
        return float("nan")

    try:
        x = np.asarray(l_sizes, dtype=float)
        y = np.asarray(performance_scores, dtype=float)
        mask = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[mask], y[mask]

        if len(x) < 2 or len(x) != len(y):
            return float("nan")

        order = np.argsort(x)
        x, y = x[order], y[order]

        delta_x = x[-1] - x[0]
        if delta_x <= 0:
            return float("nan")

        area_ideal = baseline_performance * delta_x
        if area_ideal <= 0:
            return float("nan")

        area_actual = float(np.trapezoid(y=y, x=x) if hasattr(np, "trapezoid") else np.trapz(y=y, x=x))
        return area_actual / area_ideal

    except Exception:
        return float("nan")
