"""Fábrica e registro de embedders."""

from __future__ import annotations

from typing import Callable, Dict

from ..domain.interfaces import IEmbedder
from .product_vectorizer_embedder import ProductVectorizerEmbedder


# ---------------------------------------------------------------------------
# Registro público
# ---------------------------------------------------------------------------

REGISTRY: Dict[str, Callable[..., IEmbedder]] = {
    "PVProb": lambda p: ProductVectorizerEmbedder(
        vectorizer_params=p.get("vectorizer_params", {}),
        cache_dir=p.get("cache_dir", ".pv_embedder_cache"),
    ),
}


def get_embedder(config) -> IEmbedder:
    """
    Instancia um embedder a partir de uma configuração.

    Args:
        config: ``ComponentConfig`` ou dicionário com 'type' e 'params'.
                Para 'ST' (SentenceTransformer) o modelo é criado inline
                usando a biblioteca ``sentence-transformers``.

    Returns:
        Instância de ``IEmbedder``.

    Raises:
        ValueError: Se o tipo não estiver no registro.
    """
    if hasattr(config, "type"):
        emb_type = config.type
        params = dict(config.params or {})
    else:
        emb_type = config.get("type")
        params = dict(config.get("params", {}))

    if emb_type in REGISTRY:
        return REGISTRY[emb_type](params)

    if emb_type == "ST":
        # Importação lazy — não exige sentence-transformers em tempo de importação
        from sentence_transformers import SentenceTransformer  # type: ignore
        from ..domain.interfaces import IEmbedder as _IEmbedder

        model_name = params.get(
            "model_name", "paraphrase-multilingual-mpnet-base-v2"
        )

        class _STEmbedder(_IEmbedder):
            def __init__(self, name: str):
                self._model = SentenceTransformer(name)
                self._embedding_dim = self._model.get_sentence_embedding_dimension()

            def fit(self, texts, labels=None):
                return self  # pré-treinado

            def transform(self, texts):
                return self._model.encode(
                    texts, convert_to_numpy=True, show_progress_bar=False
                )

        return _STEmbedder(model_name)

    raise ValueError(
        f"Embedder desconhecido: '{emb_type}'. Disponíveis: {list(REGISTRY) + ['ST']}"
    )
