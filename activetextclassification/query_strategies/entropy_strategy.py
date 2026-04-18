"""Estratégia de seleção por máxima entropia."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..domain.interfaces import IQueryStrategy


class EntropyStrategy(IQueryStrategy):
    """Seleciona instâncias com maior entropia de probabilidade (maior incerteza)."""

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
            probabilities: Array (pool_size, n_classes) de probabilidades — obrigatório.
            rng:           Não utilizado nesta estratégia (determinística).

        Returns:
            Índices com maior entropia.

        Raises:
            ValueError: Se ``probabilities`` for None.
        """
        if probabilities is None:
            raise ValueError("EntropyStrategy requer 'probabilities'.")
        if pool_size <= 0 or self._batch_size <= 0:
            return np.array([], dtype=int)

        n = min(self._batch_size, pool_size)
        probs = np.clip(probabilities, 1e-9, 1 - 1e-9)
        entropy = -np.sum(probs * np.log2(probs), axis=1)
        # Os n índices com maior entropia
        return np.argsort(entropy)[-n:]
