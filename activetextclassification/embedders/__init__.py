"""
activetextclassification.embedders
=====================================
Implementações concretas de ``IEmbedder``.
"""

from .product_vectorizer_embedder import ProductVectorizerEmbedder
from .factory import get_embedder, REGISTRY as EMBEDDER_REGISTRY

__all__ = [
    "ProductVectorizerEmbedder",
    "get_embedder",
    "EMBEDDER_REGISTRY",
]
