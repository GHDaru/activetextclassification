"""Estratégia de seleção aleatória."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..domain.interfaces import IQueryStrategy


class RandomStrategy(IQueryStrategy):
    """Seleciona instâncias aleatoriamente do pool."""

    def __init__(self, batch_size: int):
        """
        Args:
            batch_size: Número de instâncias a selecionar por chamada.
        """
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
            probabilities: Ignorado nesta estratégia.
            rng:           Gerador de números aleatórios.  Se None, cria um
                           gerador sem semente (não reprodutível).

        Returns:
            Índices selecionados.
        """
        if pool_size <= 0 or self._batch_size <= 0:
            return np.array([], dtype=int)

        if rng is None:
            rng = np.random.default_rng()

        n = min(self._batch_size, pool_size)
        return rng.choice(pool_size, size=n, replace=False)
