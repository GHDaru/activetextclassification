"""
activetextclassification.optimization
=======================================
Otimização do conjunto inicial L0 via Algoritmos Genéticos.

A versão canônica e mantida é ``genetic_optimizer.py`` (v4).
Versões anteriores foram movidas para ``archive/``.
"""

from .genetic_optimizer import GeneticL0Optimizer

__all__ = [
    "GeneticL0Optimizer",
]
