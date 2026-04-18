"""
activetextclassification.classifiers
======================================
Implementações concretas de ``IClassifier``.

Registro rápido::

    from activetextclassification.classifiers import get_classifier, REGISTRY

    clf = get_classifier(ComponentConfig(type="LR", params={"max_iter": 500}))
"""

from .sklearn_classifiers import (
    SklearnClassifier,
    GNBClassifier,
    LSVCClassifier,
    LRClassifier,
    SGDClassifier_,
)
from .product_vectorizer_classifier import ProductVectorizerClassifier
from .factory import get_classifier, REGISTRY

__all__ = [
    "SklearnClassifier",
    "GNBClassifier",
    "LSVCClassifier",
    "LRClassifier",
    "SGDClassifier_",
    "ProductVectorizerClassifier",
    "get_classifier",
    "REGISTRY",
]
