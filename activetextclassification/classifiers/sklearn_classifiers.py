"""Wrappers scikit-learn que implementam IClassifier."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from ..domain.interfaces import IClassifier

logger = logging.getLogger(__name__)


class SklearnClassifier(IClassifier):
    """
    Wrapper genérico para classificadores sklearn baseados em features numéricas.

    Recebe a *classe* sklearn (não instância) e os parâmetros, permitindo
    reutilização com qualquer estimador compatível.

    Args:
        sklearn_class: Classe do estimador sklearn (ex: ``LogisticRegression``).
        params:        Parâmetros passados ao construtor do estimador.
    """

    def __init__(
        self,
        sklearn_class: Type,
        params: Optional[Dict[str, Any]] = None,
    ):
        self._sklearn_class = sklearn_class
        self._params = params or {}
        self._model = None
        self._classes: Optional[List[str]] = None
        self._label_to_id: Optional[Dict[str, int]] = None
        self._id_to_label: Optional[Dict[int, str]] = None

        try:
            self._model = sklearn_class(**self._params)
        except TypeError as exc:
            raise ValueError(
                f"Erro ao criar {sklearn_class.__name__} com params={self._params}: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    #  Label mapping helpers                                               #
    # ------------------------------------------------------------------ #

    def _build_label_mapping(self, y_labels: List[str]) -> None:
        self._classes = sorted(list(np.unique(y_labels)))
        self._label_to_id = {lbl: i for i, lbl in enumerate(self._classes)}
        self._id_to_label = {i: lbl for lbl, i in self._label_to_id.items()}

    def _to_ids(self, labels: List[str]) -> np.ndarray:
        if self._label_to_id is None:
            raise RuntimeError("Modelo não treinado.  Chame fit() primeiro.")
        try:
            return np.array([self._label_to_id[lbl] for lbl in labels])
        except KeyError as exc:
            raise ValueError(f"Label '{exc}' não visto no fit.") from exc

    def _to_labels(self, ids: np.ndarray) -> np.ndarray:
        if self._id_to_label is None:
            raise RuntimeError("Modelo não treinado.  Chame fit() primeiro.")
        return np.array([self._id_to_label[i] for i in ids])

    # ------------------------------------------------------------------ #
    #  IClassifier                                                         #
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray, y_labels: List[str]) -> "SklearnClassifier":
        logger.debug(
            "Fitting %s com %d amostras.", type(self._model).__name__, len(y_labels)
        )
        self._build_label_mapping(y_labels)
        y_ids = self._to_ids(y_labels)
        # Reinicializar modelo para treino limpo a cada fit
        self._model = self._sklearn_class(**self._params)
        self._model.fit(X, y_ids)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or self._classes is None:
            raise RuntimeError("Modelo não treinado.")
        return self._to_labels(self._model.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or self._classes is None:
            raise RuntimeError("Modelo não treinado.")
        if not hasattr(self._model, "predict_proba"):
            logger.warning(
                "%s não suporta predict_proba — retornando NaNs.",
                type(self._model).__name__,
            )
            return np.full((X.shape[0], len(self._classes)), np.nan)
        return self._model.predict_proba(X)

    def get_classes(self) -> Optional[List[str]]:
        return self._classes


# ---------------------------------------------------------------------------
# Aliases tipados (conveniência)
# ---------------------------------------------------------------------------

class GNBClassifier(SklearnClassifier):
    """Gaussian Naïve Bayes via IClassifier."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(GaussianNB, params)


class LSVCClassifier(SklearnClassifier):
    """LinearSVC via IClassifier.  Nota: não suporta predict_proba."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(LinearSVC, params)


class LRClassifier(SklearnClassifier):
    """Logistic Regression via IClassifier."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(LogisticRegression, params)


class SGDClassifier_(SklearnClassifier):
    """SGD Classifier via IClassifier."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(SGDClassifier, params)
