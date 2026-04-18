"""Estratégia híbrida: fração por entropia + fração aleatória."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..domain.interfaces import IQueryStrategy
from .entropy_strategy import EntropyStrategy
from .random_strategy import RandomStrategy


class HybridStrategy(IQueryStrategy):
    """
    Seleciona ``entropy_fraction`` do lote por máxima entropia e o restante
    aleatoriamente, garantindo que não haja sobreposição.

    Parâmetros
    ----------
    batch_size:       Total de instâncias por seleção.
    entropy_fraction: Proporção selecionada por entropia (padrão 0.5).
    """

    def __init__(self, batch_size: int, entropy_fraction: float = 0.5):
        if batch_size < 1:
            raise ValueError("batch_size deve ser >= 1.")
        if not 0.0 <= entropy_fraction <= 1.0:
            raise ValueError("entropy_fraction deve estar em [0, 1].")
        self._batch_size = batch_size
        self._entropy_fraction = entropy_fraction

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def entropy_fraction(self) -> float:
        return self._entropy_fraction

    def select(
        self,
        pool_size: int,
        probabilities: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Args:
            pool_size:     Tamanho do pool.
            probabilities: Array (pool_size, n_classes) — obrigatório.
            rng:           Gerador de números aleatórios para a parte aleatória.

        Returns:
            Índices selecionados (sem repetição).
        """
        if probabilities is None:
            raise ValueError("HybridStrategy requer 'probabilities'.")
        if pool_size <= 0 or self._batch_size <= 0:
            return np.array([], dtype=int)

        if rng is None:
            rng = np.random.default_rng()

        n = min(self._batch_size, pool_size)
        n_entropy = int(round(n * self._entropy_fraction))
        n_random = n - n_entropy

        entropy_indices = np.array([], dtype=int)
        if n_entropy > 0:
            entropy_indices = EntropyStrategy(n_entropy).select(
                pool_size, probabilities
            )

        remaining = np.setdiff1d(np.arange(pool_size), entropy_indices, assume_unique=True)

        random_indices = np.array([], dtype=int)
        if n_random > 0 and len(remaining) > 0:
            k = min(n_random, len(remaining))
            random_indices = rng.choice(remaining, size=k, replace=False)

        return np.concatenate([entropy_indices, random_indices]).astype(int)
