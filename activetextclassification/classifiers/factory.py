"""Fábrica e registro de classificadores."""

from __future__ import annotations

from typing import Any, Callable, Dict

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from ..domain.interfaces import IClassifier
from .sklearn_classifiers import SklearnClassifier
from .product_vectorizer_classifier import ProductVectorizerClassifier


# ---------------------------------------------------------------------------
# Registro público
# ---------------------------------------------------------------------------

REGISTRY: Dict[str, Callable[..., IClassifier]] = {
    "GNB": lambda p: SklearnClassifier(GaussianNB, p),
    "LSVC": lambda p: SklearnClassifier(LinearSVC, p),
    "LR": lambda p: SklearnClassifier(LogisticRegression, p),
    "SGD": lambda p: SklearnClassifier(SGDClassifier, p),
    "PVBin": lambda p: ProductVectorizerClassifier(p),
}


def get_classifier(config) -> IClassifier:
    """
    Instancia um classificador a partir de uma configuração.

    Args:
        config: ``ComponentConfig`` ou dicionário com 'type' e 'params'.

    Returns:
        Instância de ``IClassifier``.

    Raises:
        ValueError: Se o tipo não estiver no registro.
    """
    if hasattr(config, "type"):
        clf_type = config.type
        params = dict(config.params or {})
    else:
        clf_type = config.get("type")
        params = dict(config.get("params", {}))

    # ngram_range JSON → tuple
    if "ngram_range" in params and isinstance(params["ngram_range"], list):
        params["ngram_range"] = tuple(params["ngram_range"])

    factory = REGISTRY.get(clf_type)
    if factory is None:
        raise ValueError(
            f"Classificador desconhecido: '{clf_type}'. Disponíveis: {list(REGISTRY)}"
        )
    return factory(params)
