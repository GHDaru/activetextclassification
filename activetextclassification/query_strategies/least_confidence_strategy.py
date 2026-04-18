"""Estratégia de seleção por menor confiança."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..domain.interfaces import IQueryStrategy


class LeastConfidenceStrategy(IQueryStrategy):
    """
    Seleciona instâncias em que a probabilidade da classe mais provável
    é a menor (menor confiança = maior incerteza).
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
            rng:           Não utilizado.

        Returns:
            Índices com menor confiança.
        """
        if probabilities is None:
            raise ValueError("LeastConfidenceStrategy requer 'probabilities'.")
        if pool_size <= 0 or self._batch_size <= 0:
            return np.array([], dtype=int)

        n = min(self._batch_size, pool_size)
        confidence = np.max(probabilities, axis=1)
        # Os n índices com menor confiança (= início do argsort crescente)
        return np.argsort(confidence)[:n]
