"""
activetextclassification.oracles
===================================
Implementações concretas de ``IOracle``.

Uso rápido::

    from activetextclassification.oracles import SimulatedOracle, get_oracle
    from activetextclassification.application.config import ComponentConfig

    oracle = SimulatedOracle(label_column="category")
    labels = oracle.query(df_batch)

    # Via fábrica
    cfg = ComponentConfig(type="Simulated", params={"label_column": "category"})
    oracle = get_oracle(cfg)
"""

from .simulated_oracle import SimulatedOracle
from .llm_oracles import BaseLLMOracle, OllamaOracle, GoogleOracle, OpenaiOracle, AnthropicOracle
from .factory import get_oracle, REGISTRY as ORACLE_REGISTRY

__all__ = [
    "SimulatedOracle",
    "BaseLLMOracle",
    "OllamaOracle",
    "GoogleOracle",
    "OpenaiOracle",
    "AnthropicOracle",
    "get_oracle",
    "ORACLE_REGISTRY",
]
