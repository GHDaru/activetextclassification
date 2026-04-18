"""
activetextclassification.oracle
================================
Oráculos de rotulação para o ciclo de aprendizado ativo.
"""

from .oracle import BaseOracle, SimulatedOracle, get_oracle

__all__ = [
    "BaseOracle",
    "SimulatedOracle",
    "get_oracle",
]
