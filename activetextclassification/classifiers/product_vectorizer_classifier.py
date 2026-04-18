"""ProductVectorizerClassifier — wrapper IClassifier para ProductVectorizer."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..domain.interfaces import IClassifier
from ..vectorizers import ProductVectorizer

logger = logging.getLogger(__name__)


class ProductVectorizerClassifier(IClassifier):
    """
    Classificador de texto que usa o ``ProductVectorizer`` como motor interno.

    Opera diretamente sobre listas de texto (sem embedding externo), tornando-o
    um ``IClassifier`` baseado em texto.

    Args:
        params: Parâmetros passados ao construtor de ``ProductVectorizer``
                (ex: ``method``, ``ngram_range``, ``norm``).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self._params = params or {}
        # Converter ngram_range de lista → tupla (necessário para JSON round-trip)
        if "ngram_range" in self._params and isinstance(
            self._params["ngram_range"], list
        ):
            self._params["ngram_range"] = tuple(self._params["ngram_range"])

        self._pv: Optional[ProductVectorizer] = None
        self._classes: Optional[List[str]] = None
        self._label_to_id: Optional[Dict[str, int]] = None
        self._id_to_label: Optional[Dict[int, str]] = None

    # ------------------------------------------------------------------ #
    #  IClassifier                                                         #
    # ------------------------------------------------------------------ #

    def fit(self, X: List[str], y_labels: List[str]) -> "ProductVectorizerClassifier":
        logger.debug("Fitting ProductVectorizerClassifier com %d amostras.", len(X))
        self._pv = ProductVectorizer(**self._params)
        self._pv.fit(X, y_labels)
        # Expor mapeamento de labels no contrato da interface
        self._label_to_id = self._pv.category_index
        self._id_to_label = self._pv.index_category
        self._classes = sorted(list(self._id_to_label.values())) if self._id_to_label else []
        return self

    def predict(self, X: List[str]) -> np.ndarray:
        self._check_fitted()
        return self._pv.predict(X, out="category")

    def predict_proba(self, X: List[str]) -> np.ndarray:
        self._check_fitted()
        # ProductVectorizer.predict_proba retorna shape (n_classes, n_samples)
        proba_raw = self._pv.predict_proba(X)  # (n_classes_pv, n_samples)

        internal_classes = self._pv.get_category_from_index(
            np.arange(len(self._pv.category_index))
        )
        internal_map = {lbl: i for i, lbl in enumerate(internal_classes)}

        n_samples = len(X)
        output = np.zeros((n_samples, len(self._classes)))
        for i, target_lbl in enumerate(self._classes):
            if target_lbl in internal_map:
                output[:, i] = proba_raw[internal_map[target_lbl], :]

        return output  # (n_samples, n_classes)

    def get_classes(self) -> Optional[List[str]]:
        return self._classes

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _check_fitted(self) -> None:
        if self._pv is None or self._classes is None:
            raise RuntimeError(
                "ProductVectorizerClassifier não treinado.  Chame fit() primeiro."
            )
