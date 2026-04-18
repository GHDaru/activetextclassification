"""
activetextclassification
========================

Biblioteca Python para classificação de texto com Aprendizado Ativo.

Nova API (DDD)::

    from activetextclassification.application import ActiveLearner, ExperimentConfig
    from activetextclassification.classifiers import get_classifier
    from activetextclassification.oracles import SimulatedOracle
    from activetextclassification.query_strategies import EntropyStrategy
    from activetextclassification.domain.entities import Budget
    import numpy as np

    rng = np.random.default_rng(seed=42)
    learner = ActiveLearner(
        P_df=P_df,
        U_df=U_df,
        text_column="description",
        label_column="category",
        all_possible_labels=labels,
        classifier=get_classifier(clf_cfg),
        oracle=SimulatedOracle("category"),
        query_strategy=EntropyStrategy(batch_size=10),
        budget=Budget(target_budget_pct=0.30, max_iterations=50),
        rng=rng,
    )
    result = learner.run()

API legada (ainda suportada para compatibilidade)::

    from activetextclassification import ActiveLearner   # legado, config-based
    learner = ActiveLearner(config)
    learner.setup()
    learner.run()
"""

# ── Legacy API (backward compat) ───────────────────────────────────────────
from .active_learner import ActiveLearner as _LegacyActiveLearner
from .data_preparation import load_and_prepare_data, load_split_and_preprocess_data
from .embeddings import get_embedder, BaseEmbedder
from .models import get_model, BaseTextClassifier, BaseFeatureClassifier
from .selection import select_query_batch
from .cold_start import select_initial_batch
from .oracle import get_oracle, BaseOracle
from .config import load_experiments_config, validate_experiment_config
from .utils import load_and_flatten_experiment_history, calculate_lce

# Keep ActiveLearner importable from the top level (legacy alias)
ActiveLearner = _LegacyActiveLearner

# ── New DDD API ────────────────────────────────────────────────────────────
from .domain import (
    IClassifier,
    IEmbedder,
    IOracle,
    IQueryStrategy,
    IColdStart,
    Budget,
    ExperimentResult,
    compute_accuracy,
    compute_f1_macro,
    compute_lce,
)
from .application.config import (
    ComponentConfig,
    DataConfig,
    ALConfig,
    BaselineConfig,
    ExperimentConfig,
    load_experiment_configs,
)
from .application.active_learner import ActiveLearner as NewActiveLearner
from .application.experiment_runner import ExperimentRunner
from .classifiers import get_classifier
from .embedders import get_embedder as get_embedder_new
from .query_strategies import get_query_strategy
from .oracles import get_oracle as get_oracle_new, SimulatedOracle
from .infrastructure import HistoryStore

try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("activetextclassification")
    except PackageNotFoundError:
        __version__ = "0.0.1"
except ImportError:
    __version__ = "0.0.1"

__author__ = "Gilsiley Henrique Darú"

__all__ = [
    # ── Legacy ──────────────────────────────────────────────────────────
    "ActiveLearner",
    "load_and_prepare_data",
    "load_split_and_preprocess_data",
    "get_embedder",
    "BaseEmbedder",
    "get_model",
    "BaseTextClassifier",
    "BaseFeatureClassifier",
    "select_query_batch",
    "select_initial_batch",
    "get_oracle",
    "BaseOracle",
    "load_experiments_config",
    "validate_experiment_config",
    "load_and_flatten_experiment_history",
    "calculate_lce",
    # ── New DDD API ──────────────────────────────────────────────────────
    # Domain interfaces
    "IClassifier",
    "IEmbedder",
    "IOracle",
    "IQueryStrategy",
    "IColdStart",
    # Domain entities
    "Budget",
    "ExperimentResult",
    # Domain metrics
    "compute_accuracy",
    "compute_f1_macro",
    "compute_lce",
    # Application config
    "ComponentConfig",
    "DataConfig",
    "ALConfig",
    "BaselineConfig",
    "ExperimentConfig",
    "load_experiment_configs",
    # Application AL
    "NewActiveLearner",
    "ExperimentRunner",
    # Classifiers
    "get_classifier",
    # Embedders
    "get_embedder_new",
    # Query strategies
    "get_query_strategy",
    # Oracles
    "get_oracle_new",
    "SimulatedOracle",
    # Infrastructure
    "HistoryStore",
]
