"""
activetextclassification.oracle
================================
Oráculos de rotulação para o ciclo de aprendizado ativo.
"""

from .oracles import BaseOracle, get_oracle

__all__ = [
    "BaseOracle",
    "get_oracle",
]
