"""
activetextclassification.application.config
=============================================
Configuração tipada via dataclasses para substituir os dicionários livres.

Vantagens:
- Erros detectados na construção, não em runtime.
- Autocompletar em IDEs.
- Serializável para/de JSON de forma determinística.

Exemplo::

    from activetextclassification.application.config import (
        DataConfig, ALConfig, ComponentConfig, ExperimentConfig
    )

    cfg = ExperimentConfig(
        name="exp_001",
        data=DataConfig(
            file_path="data/products.csv",
            text_column="description",
            label_column="category",
        ),
        al=ALConfig(
            cold_start=ComponentConfig(type="RND"),
            query_strategy=ComponentConfig(type="ENT", params={"batch_size": 10}),
            classifier=ComponentConfig(type="PVBin"),
        ),
    )
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# ComponentConfig — configuração genérica de qualquer componente plugável
# ---------------------------------------------------------------------------

@dataclass
class ComponentConfig:
    """
    Configuração genérica para componentes instanciáveis via fábrica
    (classificadores, embedders, estratégias, oráculos, cold-start).

    Attributes:
        type:   Identificador do componente (ex: 'GNB', 'ENT', 'Simulated').
        params: Parâmetros específicos passados ao construtor.
    """

    type: str
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.type:
            raise ValueError("ComponentConfig.type não pode ser vazio.")
        if not isinstance(self.params, dict):
            raise TypeError("ComponentConfig.params deve ser um dict.")


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Parâmetros de carregamento e divisão dos dados."""

    file_path: str
    text_column: str
    label_column: str
    population_size: float = 0.50
    min_samples_per_class: int = 2
    rare_group_label: str = "_RARE_GROUP_"
    random_seed: int = 42
    sheet_name: int = 0

    def __post_init__(self):
        if not self.file_path:
            raise ValueError("DataConfig.file_path não pode ser vazio.")
        if not self.text_column:
            raise ValueError("DataConfig.text_column não pode ser vazio.")
        if not self.label_column:
            raise ValueError("DataConfig.label_column não pode ser vazio.")
        if not 0 < self.population_size < 1:
            raise ValueError(
                f"DataConfig.population_size deve estar em (0, 1), recebido: {self.population_size}"
            )


# ---------------------------------------------------------------------------
# BaselineConfig
# ---------------------------------------------------------------------------

@dataclass
class BaselineConfig:
    """Configuração do classificador de baseline."""

    classifier: ComponentConfig
    test_size: float = 0.05

    def __post_init__(self):
        if not 0 < self.test_size < 1:
            raise ValueError(
                f"BaselineConfig.test_size deve estar em (0, 1), recebido: {self.test_size}"
            )


# ---------------------------------------------------------------------------
# ALConfig
# ---------------------------------------------------------------------------

@dataclass
class ALConfig:
    """Parâmetros do loop de Aprendizado Ativo."""

    cold_start: ComponentConfig
    query_strategy: ComponentConfig
    classifier: ComponentConfig
    embedder: Optional[ComponentConfig] = None
    oracle: ComponentConfig = field(
        default_factory=lambda: ComponentConfig(type="Simulated", params={})
    )
    budget_pct: float = 0.30
    max_iterations: int = 100
    internal_test_size: float = 0.20
    early_stopping_metric: Optional[str] = None
    early_stopping_patience: Optional[int] = None
    early_stopping_tolerance: float = 0.001

    def __post_init__(self):
        if not 0 < self.budget_pct <= 1:
            raise ValueError(
                f"ALConfig.budget_pct deve estar em (0, 1], recebido: {self.budget_pct}"
            )
        if self.max_iterations < 1:
            raise ValueError("ALConfig.max_iterations deve ser >= 1.")
        valid_es = {None, "external_acc", "external_f1", "internal_acc", "internal_f1"}
        if self.early_stopping_metric not in valid_es:
            raise ValueError(
                f"ALConfig.early_stopping_metric inválido: '{self.early_stopping_metric}'. "
                f"Válidos: {valid_es}"
            )


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Configuração completa de um experimento de Aprendizado Ativo."""

    name: str
    data: DataConfig
    al: ALConfig
    baseline: Optional[BaselineConfig] = None
    active: bool = True
    random_seed: int = 42
    verbose: bool = False

    def __post_init__(self):
        if not self.name:
            raise ValueError("ExperimentConfig.name não pode ser vazio.")

    # ------------------------------------------------------------------ #
    #  Serialização                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário serializável em JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """
        Reconstrói a partir de um dicionário (ex: carregado de JSON).

        Suporta tanto o formato novo (dataclass) quanto o formato legado
        (chaves ``experiment_name``, ``data_params``, ``al_params``, etc.)
        para migração gradual.
        """
        # ── Formato novo ────────────────────────────────────────────────
        if "name" in d and "data" in d and "al" in d:
            return cls(
                name=d["name"],
                data=DataConfig(**d["data"]),
                al=_al_config_from_dict(d["al"]),
                baseline=(
                    _baseline_config_from_dict(d["baseline"])
                    if d.get("baseline")
                    else None
                ),
                active=d.get("active", True),
                random_seed=d.get("random_seed", 42),
                verbose=d.get("verbose", False),
            )

        # ── Formato legado (data_params / al_params) ────────────────────
        data_p = d.get("data_params", {})
        al_p = d.get("al_params", {})
        gen_p = d.get("general_params", {})

        data = DataConfig(
            file_path=data_p.get("file_path", ""),
            text_column=data_p.get("text_column", ""),
            label_column=data_p.get("label_column", ""),
            population_size=data_p.get("population_size", 0.50),
            min_samples_per_class=data_p.get("min_samples_per_class", 2),
            rare_group_label=data_p.get("rare_group_label", "_RARE_GROUP_"),
            random_seed=gen_p.get("random_seed", 42),
            sheet_name=data_p.get("sheet_name", 0),
        )

        al = ALConfig(
            cold_start=ComponentConfig(
                **al_p.get("cold_start_config", {"type": "RND"})
            ),
            query_strategy=ComponentConfig(
                **al_p.get("query_strategy_config", {"type": "RND", "params": {"batch_size": 10}})
            ),
            classifier=ComponentConfig(
                **al_p.get("classifier_config", {"type": "PVBin"})
            ),
            embedder=(
                ComponentConfig(**d["embedder_global_config"])
                if d.get("embedder_global_config")
                else None
            ),
            oracle=ComponentConfig(
                **al_p.get("oracle_config", {"type": "Simulated"})
            ),
            budget_pct=al_p.get("target_budget_pct", 0.30),
            max_iterations=al_p.get("max_iterations", 100),
            internal_test_size=al_p.get("internal_test_size", 0.20),
            early_stopping_metric=al_p.get("early_stopping_metric"),
            early_stopping_patience=al_p.get("early_stopping_patience"),
            early_stopping_tolerance=al_p.get("early_stopping_tolerance", 0.001),
        )

        baseline_cfg_raw = d.get("baseline_classifier_config")
        baseline = (
            BaselineConfig(
                classifier=ComponentConfig(type=baseline_cfg_raw.get("type", "PVBin"),
                                           params=baseline_cfg_raw.get("params", {})),
                test_size=baseline_cfg_raw.get("test_size", 0.05),
            )
            if baseline_cfg_raw
            else None
        )

        return cls(
            name=d.get("experiment_name", "unknown"),
            data=data,
            al=al,
            baseline=baseline,
            active=d.get("active", True),
            random_seed=gen_p.get("random_seed", 42),
            verbose=gen_p.get("verbose", False),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ExperimentConfig":
        """Desserializa a partir de uma string JSON."""
        return cls.from_dict(json.loads(json_str))

    def to_json(self) -> str:
        """Serializa para string JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Helpers privados de desserialização
# ---------------------------------------------------------------------------

def _al_config_from_dict(d: Dict[str, Any]) -> ALConfig:
    return ALConfig(
        cold_start=ComponentConfig(**d["cold_start"]),
        query_strategy=ComponentConfig(**d["query_strategy"]),
        classifier=ComponentConfig(**d["classifier"]),
        embedder=ComponentConfig(**d["embedder"]) if d.get("embedder") else None,
        oracle=ComponentConfig(**d.get("oracle", {"type": "Simulated"})),
        budget_pct=d.get("budget_pct", 0.30),
        max_iterations=d.get("max_iterations", 100),
        internal_test_size=d.get("internal_test_size", 0.20),
        early_stopping_metric=d.get("early_stopping_metric"),
        early_stopping_patience=d.get("early_stopping_patience"),
        early_stopping_tolerance=d.get("early_stopping_tolerance", 0.001),
    )


def _baseline_config_from_dict(d: Dict[str, Any]) -> BaselineConfig:
    return BaselineConfig(
        classifier=ComponentConfig(**d["classifier"]),
        test_size=d.get("test_size", 0.05),
    )


# ---------------------------------------------------------------------------
# load_experiment_configs — utilitário de carregamento em lote
# ---------------------------------------------------------------------------

def load_experiment_configs(config_path: str) -> List[ExperimentConfig]:
    """
    Carrega uma lista de ExperimentConfig a partir de um arquivo JSON.

    O arquivo pode conter tanto o formato novo quanto o formato legado.

    Args:
        config_path: Caminho para o arquivo JSON.

    Returns:
        Lista de ``ExperimentConfig``.  Retorna lista vazia em caso de erro.
    """
    import os
    import logging

    logger = logging.getLogger(__name__)

    if not os.path.exists(config_path):
        logger.error("Arquivo de configuração não encontrado: %s", config_path)
        return []

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            logger.error("Conteúdo de %s não é uma lista JSON.", config_path)
            return []

        configs: List[ExperimentConfig] = []
        for i, item in enumerate(raw):
            try:
                configs.append(ExperimentConfig.from_dict(item))
            except Exception as exc:
                logger.warning(
                    "Ignorando configuração #%d ('%s'): %s",
                    i,
                    item.get("name", item.get("experiment_name", "?")),
                    exc,
                )

        logger.info("%d configurações carregadas de %s.", len(configs), config_path)
        return configs

    except json.JSONDecodeError as exc:
        logger.error("Falha ao decodificar JSON em %s: %s", config_path, exc)
        return []
    except Exception as exc:
        logger.error("Erro inesperado ao carregar %s: %s", config_path, exc)
        return []
