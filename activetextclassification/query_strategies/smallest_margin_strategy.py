"""Estratégia de seleção por menor margem."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..domain.interfaces import IQueryStrategy
from .random_strategy import RandomStrategy


class SmallestMarginStrategy(IQueryStrategy):
    """
    Seleciona instâncias em que a margem entre as duas classes mais
    prováveis é a menor (menor margem = maior incerteza binária).
    """

    def __init__(self, batch_size: int):
        if batch_size < 1:
            raise ValueError("batch_size deve ser >= 1.")
        self._batch_size = batch_size

    @property
    def batch_size(self) -> int:
        return self._batch_size

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
                           Se n_classes < 2, cai em seleção aleatória.
            rng:           Gerador de números aleatórios (usado apenas no fallback aleatório).

        Returns:
            Índices com menor margem.
        """
        if probabilities is None:
            raise ValueError("SmallestMarginStrategy requer 'probabilities'.")
        if pool_size <= 0 or self._batch_size <= 0:
            return np.array([], dtype=int)

        n = min(self._batch_size, pool_size)

        if probabilities.shape[1] < 2:
            # Fallback: aleatório quando há apenas uma classe
            return RandomStrategy(n).select(pool_size, rng=rng)

        sorted_probs = np.sort(probabilities, axis=1)
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
        return np.argsort(margins)[:n]
