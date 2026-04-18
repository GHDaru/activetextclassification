"""Fábrica e registro de estratégias de cold start."""

from __future__ import annotations

from typing import Callable, Dict

from ..domain.interfaces import IColdStart
from .random_cold_start import RandomColdStart
from .kmedians_cold_start import KMediansColdStart


REGISTRY: Dict[str, Callable[..., IColdStart]] = {
    "RND": lambda p: RandomColdStart(),
    "KM": lambda p: KMediansColdStart(method=p.get("method", "pam")),
}


def get_cold_start(config) -> IColdStart:
    """
    Instancia uma estratégia de cold start a partir de uma configuração.

    Args:
        config: ``ComponentConfig`` ou dicionário com 'type' e 'params'.

    Returns:
        Instância de ``IColdStart``.
    """
    if hasattr(config, "type"):
        cs_type = config.type
        params = dict(config.params or {})
    else:
        cs_type = config.get("type", "RND")
        params = dict(config.get("params", {}))

    factory = REGISTRY.get(cs_type)
    if factory is None:
        # DRI-Cluster (opcional, requer dependências pesadas)
        if cs_type == "DRI":
            from .dri_cluster import DRIClusterColdStart  # type: ignore
            return DRIClusterColdStart(
                i_target=params.get("i_target", 10),
                semantic_embedder=params.get("semantic_embedder"),
                n_clusters_semantic=params.get("n_clusters_semantic", 5),
                random_seed=params.get("random_seed", 42),
            )
        raise ValueError(
            f"Cold start desconhecido: '{cs_type}'.  Disponíveis: {list(REGISTRY) + ['DRI']}"
        )
    return factory(params)
