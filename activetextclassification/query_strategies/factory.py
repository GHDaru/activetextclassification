"""Fábrica e registro de estratégias de seleção."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

import numpy as np

from ..domain.interfaces import IQueryStrategy
from .random_strategy import RandomStrategy
from .entropy_strategy import EntropyStrategy
from .least_confidence_strategy import LeastConfidenceStrategy
from .smallest_margin_strategy import SmallestMarginStrategy
from .hybrid_strategy import HybridStrategy


# ---------------------------------------------------------------------------
# Registro público — mapeamento: código (str) → factory callable
# ---------------------------------------------------------------------------

REGISTRY: Dict[str, Callable[..., IQueryStrategy]] = {
    "RND": lambda params: RandomStrategy(batch_size=params.get("batch_size", 10)),
    "ENT": lambda params: EntropyStrategy(batch_size=params.get("batch_size", 10)),
    "LCO": lambda params: LeastConfidenceStrategy(batch_size=params.get("batch_size", 10)),
    "SMA": lambda params: SmallestMarginStrategy(batch_size=params.get("batch_size", 10)),
    "HYB": lambda params: HybridStrategy(
        batch_size=params.get("batch_size", 10),
        entropy_fraction=params.get("entropy_fraction", 0.5),
    ),
}


def get_query_strategy(
    config,
    rng: Optional[np.random.Generator] = None,
) -> IQueryStrategy:
    """
    Instancia uma estratégia de seleção a partir de uma configuração.

    Args:
        config: ``ComponentConfig`` ou dicionário com 'type' e 'params'.
        rng:    Gerador de números aleatórios (passado para o ``select``
                de estratégias que precisam de aleatoriedade).

    Returns:
        Instância de ``IQueryStrategy``.

    Raises:
        ValueError: Se o tipo não estiver no registro.
    """
    # Aceita tanto ComponentConfig quanto dict para compatibilidade
    if hasattr(config, "type"):
        strategy_type = config.type
        params = config.params or {}
    else:
        strategy_type = config.get("type", "RND")
        params = config.get("params", {})

    factory = REGISTRY.get(strategy_type)
    if factory is None:
        raise ValueError(
            f"Estratégia desconhecida: '{strategy_type}'. "
            f"Disponíveis: {list(REGISTRY)}"
        )

    return factory(params)
