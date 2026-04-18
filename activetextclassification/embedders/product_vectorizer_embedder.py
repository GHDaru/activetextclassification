"""
ProductVectorizerEmbedder — implementação de IEmbedder com cache em disco.

Cache de embeddings: evita re-computar embeddings quando os dados e os
parâmetros não mudam entre experimentos.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np

from ..domain.interfaces import IEmbedder
from ..vectorizers import ProductVectorizer

logger = logging.getLogger(__name__)


class ProductVectorizerEmbedder(IEmbedder):
    """
    Usa ``ProductVectorizer`` para gerar vetores de probabilidade como embeddings.

    O cache (baseado em hash de textos + labels + parâmetros) evita re-computar
    quando os mesmos dados são usados em experimentos diferentes.

    Args:
        vectorizer_params: Parâmetros do ``ProductVectorizer``.
        cache_dir:         Diretório para cache em disco (None desativa).
    """

    _DEFAULT_PARAMS: Dict[str, Any] = {
        "method": "tfidf",
        "query": "tfidf",
        "norm": "l2",
        "query_norm": "l2",
    }

    def __init__(
        self,
        vectorizer_params: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str] = ".pv_embedder_cache",
    ):
        self._vectorizer_params = vectorizer_params or dict(self._DEFAULT_PARAMS)
        # ngram_range JSON → tuple
        if "ngram_range" in self._vectorizer_params and isinstance(
            self._vectorizer_params["ngram_range"], list
        ):
            self._vectorizer_params["ngram_range"] = tuple(
                self._vectorizer_params["ngram_range"]
            )

        self._pv: Optional[ProductVectorizer] = None
        self._embedding_dim: Optional[int] = None
        self.cache_dir = cache_dir

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  IEmbedder                                                           #
    # ------------------------------------------------------------------ #

    def fit(self, texts: List[str], labels: List[str]) -> "ProductVectorizerEmbedder":
        """
        Ajusta o ``ProductVectorizer`` interno.

        Args:
            texts:  Lista de textos.
            labels: Rótulos (obrigatório para ``ProductVectorizer``).
        """
        if labels is None:
            raise ValueError(
                "ProductVectorizerEmbedder requer 'labels' no método fit()."
            )

        texts_hash = _hash(texts)
        labels_hash = _hash(labels)
        cache_file = self._cache_path("fit_state", texts_hash, labels_hash)

        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
                self._pv = cached["pv_instance"]
                self._embedding_dim = cached["embedding_dim"]
                logger.debug("Fit: estado carregado do cache %s.", cache_file)
                return self
            except Exception as exc:
                logger.warning("Fit: falha ao carregar cache %s: %s.", cache_file, exc)
                self._pv = None

        self._pv = ProductVectorizer(**self._vectorizer_params)
        self._pv.fit(texts, labels)
        self._embedding_dim = (
            len(self._pv.category_index) if self._pv.category_index else 0
        )

        if cache_file and self._embedding_dim > 0:
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(
                        {"pv_instance": self._pv, "embedding_dim": self._embedding_dim}, f
                    )
            except Exception as exc:
                logger.warning("Fit: falha ao salvar cache %s: %s.", cache_file, exc)

        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        if self._pv is None or self._embedding_dim is None:
            raise RuntimeError(
                "ProductVectorizerEmbedder não ajustado.  Chame fit() primeiro."
            )
        if self._embedding_dim == 0:
            return np.empty((len(texts), 0))
        if not texts:
            return np.empty((0, self._embedding_dim))

        texts_hash = _hash(texts)
        cache_file = self._cache_path("transform_output", texts_hash)

        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    embeddings = pickle.load(f)
                if (
                    isinstance(embeddings, np.ndarray)
                    and embeddings.shape[1] == self._embedding_dim
                ):
                    return embeddings
            except Exception as exc:
                logger.warning(
                    "Transform: falha ao carregar cache %s: %s.", cache_file, exc
                )

        proba_raw = self._pv.predict_proba(texts)
        embeddings = proba_raw.T  # (n_samples, n_classes)

        if cache_file:
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(embeddings, f)
            except Exception as exc:
                logger.warning(
                    "Transform: falha ao salvar cache %s: %s.", cache_file, exc
                )

        return embeddings

    def get_embedding_dimension(self) -> Optional[int]:
        return self._embedding_dim

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _cache_path(
        self,
        operation: str,
        texts_hash: str,
        labels_hash: Optional[str] = None,
    ) -> Optional[str]:
        if not self.cache_dir:
            return None
        params_hash = _hash(self._vectorizer_params)
        if operation == "fit_state" and labels_hash:
            fname = (
                f"{operation}_p{params_hash}_t{texts_hash}_l{labels_hash}.pkl"
            )
        else:
            fname = f"{operation}_p{params_hash}_t{texts_hash}.pkl"
        return os.path.join(self.cache_dir, fname)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _hash(data: Any) -> str:
    if isinstance(data, (list, tuple)):
        s = "<SEP>".join(str(item) for item in data)
    elif isinstance(data, dict):
        s = json.dumps(data, sort_keys=True)
    else:
        s = str(data)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]
