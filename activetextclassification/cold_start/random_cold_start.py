"""RandomColdStart — seleção aleatória inicial."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..domain.interfaces import IColdStart


class RandomColdStart(IColdStart):
    """Seleciona amostras aleatoriamente do pool para o L0 inicial."""

    def select(
        self,
        U_df: pd.DataFrame,
        n_initial: int,
        embeddings: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        n = max(0, min(n_initial, len(U_df)))
        if n == 0:
            return np.array([], dtype=int)
        if rng is None:
            rng = np.random.default_rng()
        return rng.choice(len(U_df), size=n, replace=False)
