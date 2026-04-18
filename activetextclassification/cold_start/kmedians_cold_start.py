"""KMediansColdStart — seleção por K-Medoids nos embeddings."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..domain.interfaces import IColdStart

logger = logging.getLogger(__name__)


class KMediansColdStart(IColdStart):
    """
    Seleciona o L0 usando K-Medoids sobre os embeddings do pool.

    Args:
        method: Método do KMedoids (``'pam'`` ou ``'alternate'``).
    """

    def __init__(self, method: str = "pam"):
        self.method = method

    def select(
        self,
        U_df: pd.DataFrame,
        n_initial: int,
        embeddings: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if embeddings is None:
            raise ValueError("KMediansColdStart requer 'embeddings'.")
        if n_initial <= 0 or len(U_df) == 0:
            return np.array([], dtype=int)

        n = min(n_initial, len(U_df))
        if n > len(U_df):
            return np.arange(len(U_df))

        try:
            from sklearn_extra.cluster import KMedoids  # type: ignore
        except ImportError:
            raise ImportError(
                "KMediansColdStart requer scikit-learn-extra.  "
                "Execute: pip install scikit-learn-extra"
            )

        seed = int(rng.integers(0, 2**31)) if rng is not None else 42
        km = KMedoids(n_clusters=n, random_state=seed, method=self.method)
        try:
            km.fit(embeddings)
            return km.medoid_indices_
        except Exception as exc:
            logger.error("KMedoids falhou: %s.  Retornando seleção vazia.", exc)
            return np.array([], dtype=int)
