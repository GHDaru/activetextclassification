"""Fábrica e registro de oráculos."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ..domain.interfaces import IOracle
from .simulated_oracle import SimulatedOracle


# ---------------------------------------------------------------------------
# Registro público
# ---------------------------------------------------------------------------

REGISTRY: Dict[str, Callable[..., IOracle]] = {
    "Simulated": lambda p: SimulatedOracle(
        label_column=p.get("label_column", "label")
    ),
}


def get_oracle(
    config,
    all_possible_labels: Optional[List[str]] = None,
    prompt_template: Optional[str] = None,
) -> IOracle:
    """
    Instancia um oráculo a partir de uma configuração.

    Args:
        config:               ``ComponentConfig`` ou dicionário.
        all_possible_labels:  Necessário para oráculos LLM (monta a string de labels).
        prompt_template:      Template de prompt.  Se None, tenta carregar do
                              módulo ``oracle.prompts``.

    Returns:
        Instância de ``IOracle``.

    Raises:
        ValueError: Se o tipo for desconhecido ou parâmetros obrigatórios estiverem ausentes.
    """
    if hasattr(config, "type"):
        oracle_type = config.type
        params = dict(config.params or {})
    else:
        oracle_type = config.get("type", "Simulated")
        params = dict(config.get("params", {}))

    # ── Simulated ──────────────────────────────────────────────────────
    if oracle_type == "Simulated":
        label_col = params.get("label_column")
        if not label_col:
            raise ValueError(
                "SimulatedOracle requer 'label_column' nos params do ComponentConfig."
            )
        return SimulatedOracle(label_column=label_col)

    # ── LLM Oracles ────────────────────────────────────────────────────
    from .llm_oracles import OllamaOracle, GoogleOracle, OpenaiOracle, AnthropicOracle

    model_name: str = params.get("model_name", "")

    # Determinar provedor pelo nome do modelo
    _PROVIDER_MAP = {
        "openai": OpenaiOracle,
        "anthropic": AnthropicOracle,
        "google": GoogleOracle,
        "ollama": OllamaOracle,
    }

    provider_key = oracle_type.lower()
    if provider_key not in _PROVIDER_MAP:
        # Tentar inferir pelo nome do modelo
        if model_name.startswith(("gpt-4", "gpt-3.5")):
            provider_key = "openai"
        elif model_name.startswith("claude"):
            provider_key = "anthropic"
        elif model_name.startswith("gemini"):
            provider_key = "google"
        else:
            provider_key = "ollama"

    OracleClass = _PROVIDER_MAP.get(provider_key)
    if OracleClass is None:
        raise ValueError(
            f"Oráculo desconhecido: '{oracle_type}'. Disponíveis: {list(_PROVIDER_MAP) + ['Simulated']}"
        )

    # Montar string de labels para o prompt
    if all_possible_labels:
        import json
        labels_str = json.dumps(all_possible_labels, ensure_ascii=False)
    else:
        labels_str = params.get("labels_str", "[]")

    # Resolver template de prompt
    if prompt_template is None:
        try:
            from ..oracle.prompts import PROMPTS_ORACULO  # type: ignore
            prompt_key = params.get("prompt_version_key", "v3_universal_batch")
            prompt_template = PROMPTS_ORACULO.get(prompt_key, "")
        except ImportError:
            prompt_template = ""

    return OracleClass(
        model_name=model_name,
        temperature=params.get("temperature", 0.2),
        prompt_template=prompt_template,
        labels_str=labels_str,
        retries=params.get("retries", 3),
    )
