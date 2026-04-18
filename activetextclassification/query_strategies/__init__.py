"""
activetextclassification.query_strategies
==========================================
Estratégias de seleção do próximo lote de consulta (query strategies).

Cada estratégia implementa ``IQueryStrategy`` e encapsula seu estado
(batch_size, rng) evitando o antipadrão ``np.random.seed()`` global.

Uso::

    from activetextclassification.query_strategies import (
        RandomStrategy, EntropyStrategy, HybridStrategy, get_query_strategy
    )

    rng = np.random.default_rng(seed=42)
    strategy = EntropyStrategy(batch_size=10)
    indices = strategy.select(pool_size=200, probabilities=probs, rng=rng)

    # Via fábrica (compatível com ComponentConfig):
    from activetextclassification.application.config import ComponentConfig
    cfg = ComponentConfig(type="ENT", params={"batch_size": 10})
    strategy = get_query_strategy(cfg, rng=rng)
"""

from .random_strategy import RandomStrategy
from .entropy_strategy import EntropyStrategy
from .least_confidence_strategy import LeastConfidenceStrategy
from .smallest_margin_strategy import SmallestMarginStrategy
from .hybrid_strategy import HybridStrategy
from .factory import get_query_strategy, REGISTRY as STRATEGY_REGISTRY

__all__ = [
    "RandomStrategy",
    "EntropyStrategy",
    "LeastConfidenceStrategy",
    "SmallestMarginStrategy",
    "HybridStrategy",
    "get_query_strategy",
    "STRATEGY_REGISTRY",
]
