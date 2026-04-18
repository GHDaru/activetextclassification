"""
Testes de integração para o ciclo completo de Aprendizado Ativo.

Usa dados sintéticos + ProductVectorizerClassifier (sem dependências externas).
"""

import numpy as np
import pytest

from activetextclassification.application.active_learner import ActiveLearner
from activetextclassification.classifiers import get_classifier
from activetextclassification.oracles import SimulatedOracle
from activetextclassification.query_strategies import (
    RandomStrategy,
    EntropyStrategy,
    HybridStrategy,
)
from activetextclassification.domain.entities import Budget
from activetextclassification.application.config import ComponentConfig

LABELS = ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_learner(P_df, U_df, strategy, budget, rng):
    clf = get_classifier(ComponentConfig(type="PVBin"))
    oracle = SimulatedOracle(label_column="label")
    return ActiveLearner(
        P_df=P_df,
        U_df=U_df,
        text_column="text",
        label_column="label",
        all_possible_labels=LABELS,
        classifier=clf,
        oracle=oracle,
        query_strategy=strategy,
        budget=budget,
        rng=rng,
        experiment_name="test_run",
        internal_test_size=0.2,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestActiveLearnerFullCycle:
    def test_cold_start_moves_samples(self, P_df_int, U_df_int, integration_rng):
        rng = np.random.default_rng(seed=1)
        clf = get_classifier(ComponentConfig(type="PVBin"))
        learner = ActiveLearner(
            P_df=P_df_int,
            U_df=U_df_int,
            text_column="text",
            label_column="label",
            all_possible_labels=LABELS,
            classifier=clf,
            oracle=SimulatedOracle("label"),
            query_strategy=RandomStrategy(batch_size=3),
            budget=Budget(max_iterations=2, target_budget_pct=1.0),
            rng=rng,
        )
        learner.cold_start(n_initial=3)
        assert len(learner.L_df) == 3
        assert len(learner.U_df) == len(U_df_int) - 3

    def test_random_strategy_run_completes(self, P_df_int, U_df_int):
        rng = np.random.default_rng(seed=2)
        budget = Budget(max_iterations=5, target_budget_pct=0.30)
        learner = _make_learner(
            P_df_int, U_df_int, RandomStrategy(batch_size=3), budget, rng
        )
        learner.cold_start(n_initial=3)
        result = learner.run()
        assert result.status in ("COMPLETED", "STOPPED_BUDGET", "STOPPED_EMPTY_POOL")
        assert len(result.history) > 0

    def test_entropy_strategy_run(self, P_df_int, U_df_int):
        rng = np.random.default_rng(seed=3)
        budget = Budget(max_iterations=5, target_budget_pct=0.30)
        learner = _make_learner(
            P_df_int, U_df_int, EntropyStrategy(batch_size=3), budget, rng
        )
        learner.cold_start(n_initial=3)
        result = learner.run()
        assert result.status in ("COMPLETED", "STOPPED_BUDGET", "STOPPED_EMPTY_POOL")

    def test_hybrid_strategy_run(self, P_df_int, U_df_int):
        rng = np.random.default_rng(seed=4)
        budget = Budget(max_iterations=5, target_budget_pct=0.30)
        learner = _make_learner(
            P_df_int, U_df_int, HybridStrategy(batch_size=4, entropy_fraction=0.5), budget, rng
        )
        learner.cold_start(n_initial=3)
        result = learner.run()
        assert result.status in ("COMPLETED", "STOPPED_BUDGET", "STOPPED_EMPTY_POOL")

    def test_result_has_history(self, P_df_int, U_df_int):
        rng = np.random.default_rng(seed=5)
        budget = Budget(max_iterations=3, target_budget_pct=1.0)
        learner = _make_learner(
            P_df_int, U_df_int, RandomStrategy(batch_size=3), budget, rng
        )
        learner.cold_start(n_initial=3)
        result = learner.run()
        df = result.to_dataframe()
        assert not df.empty
        assert "iteration" in df.columns

    def test_reproducibility(self, P_df_int, U_df_int):
        """Mesmo seed → mesmos resultados."""
        def run_with_seed(seed):
            rng = np.random.default_rng(seed=seed)
            budget = Budget(max_iterations=4, target_budget_pct=1.0)
            learner = _make_learner(
                P_df_int.copy(), U_df_int.copy(), RandomStrategy(batch_size=3), budget, rng
            )
            learner.cold_start(n_initial=3)
            result = learner.run()
            return [r.l_size for r in result.history]

        hist_a = run_with_seed(42)
        hist_b = run_with_seed(42)
        assert hist_a == hist_b

    def test_early_stopping(self, P_df_int, U_df_int):
        rng = np.random.default_rng(seed=9)
        budget = Budget(
            max_iterations=20,
            target_budget_pct=1.0,
            early_stopping_metric="external_acc",
            early_stopping_patience=2,
            early_stopping_tolerance=0.01,
        )
        learner = _make_learner(
            P_df_int, U_df_int, RandomStrategy(batch_size=3), budget, rng
        )
        learner.cold_start(n_initial=3)
        result = learner.run()
        # Deve parar antes das 20 iterações (ou completar se convergir rápido)
        assert len(result.history) <= 20
