"""
activetextclassification
========================

Biblioteca Python para classificação de texto com Aprendizado Ativo.

Expõe a API pública principal:

    ActiveLearner          — orquestra o ciclo completo de aprendizado ativo
    load_and_prepare_data  — carregamento e preparação de dados
    get_embedder           — fábrica de embedders
    get_model              — fábrica de classificadores
    select_query_batch     — seleção de lote de consulta (query strategy)
    select_initial_batch   — seleção do lote inicial (cold start)
    get_oracle             — fábrica de oráculos de rotulação

Exemplo de uso::

    from activetextclassification import ActiveLearner

    learner = ActiveLearner(config)
    learner.setup()
    learner.run()
"""

from .active_learner import ActiveLearner
from .data_preparation import load_and_prepare_data, load_split_and_preprocess_data
from .embeddings import get_embedder, BaseEmbedder
from .models import get_model, BaseTextClassifier, BaseFeatureClassifier
from .selection import select_query_batch
from .cold_start import select_initial_batch
from .oracle import get_oracle, BaseOracle, SimulatedOracle
from .config import load_experiments_config, validate_experiment_config
from .utils import load_and_flatten_experiment_history, calculate_lce

try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("activetextclassification")
    except PackageNotFoundError:
        __version__ = "0.0.1"  # fallback durante desenvolvimento
except ImportError:
    __version__ = "0.0.1"  # Python < 3.8 fallback

__author__ = "Gilsiley Henrique Darú"

__all__ = [
    # Core
    "ActiveLearner",
    # Data
    "load_and_prepare_data",
    "load_split_and_preprocess_data",
    # Embeddings
    "get_embedder",
    "BaseEmbedder",
    # Models
    "get_model",
    "BaseTextClassifier",
    "BaseFeatureClassifier",
    # Selection / Query Strategies
    "select_query_batch",
    # Cold Start
    "select_initial_batch",
    # Oracle
    "get_oracle",
    "BaseOracle",
    "SimulatedOracle",
    # Config
    "load_experiments_config",
    "validate_experiment_config",
    # Utils / Analysis
    "load_and_flatten_experiment_history",
    "calculate_lce",
]
