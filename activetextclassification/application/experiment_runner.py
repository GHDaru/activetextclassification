"""
activetextclassification.application.experiment_runner
=======================================================
Executa múltiplos experimentos a partir de uma lista de ``ExperimentConfig``.
Salva resultados via ``HistoryStore``.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

from ..domain.entities import Budget, ExperimentResult
from ..infrastructure.data_loader import load_and_prepare_data
from ..infrastructure.history_store import HistoryStore
from .config import ExperimentConfig
from .active_learner import ActiveLearner

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Executa múltiplos experimentos sequencialmente.

    Args:
        history_store:  Instância de ``HistoryStore`` para persistência.
                        Se None, usa ``HistoryStore("history_log.jsonl")``.
        skip_completed: Se True, pula experimentos já concluídos no log.
    """

    def __init__(
        self,
        history_store: Optional[HistoryStore] = None,
        skip_completed: bool = True,
    ):
        self.history_store = history_store or HistoryStore()
        self.skip_completed = skip_completed

    def run_all(
        self, configs: List[ExperimentConfig]
    ) -> List[ExperimentResult]:
        """
        Executa todos os experimentos da lista.

        Args:
            configs: Lista de ``ExperimentConfig``.

        Returns:
            Lista de ``ExperimentResult``.
        """
        completed = (
            self.history_store.get_completed_names() if self.skip_completed else set()
        )
        results: List[ExperimentResult] = []

        for i, cfg in enumerate(configs):
            logger.info(
                "[%d/%d] Iniciando experimento: '%s'", i + 1, len(configs), cfg.name
            )

            if not cfg.active:
                logger.info("Experimento '%s' marcado como inativo — pulando.", cfg.name)
                continue

            if cfg.name in completed:
                logger.info(
                    "Experimento '%s' já concluído — pulando.", cfg.name
                )
                continue

            result = self._run_one(cfg)
            results.append(result)

            # Persist
            summary = result.to_summary_dict()
            summary["execution_timestamp"] = datetime.now(timezone.utc).isoformat()
            self.history_store.append(summary)

        logger.info(
            "ExperimentRunner concluído: %d/%d experimentos executados.",
            len(results),
            len(configs),
        )
        return results

    def _run_one(self, cfg: ExperimentConfig) -> ExperimentResult:
        """Executa um único experimento a partir de uma ``ExperimentConfig``."""
        exp_start = time.time()
        result = ExperimentResult(
            experiment_name=cfg.name,
            config=cfg.to_dict(),
        )

        try:
            rng = np.random.default_rng(seed=cfg.random_seed)

            # ── Carregar dados ──────────────────────────────────────────
            P_df, U_df, label_to_id, id_to_label, all_labels = load_and_prepare_data(
                file_path=cfg.data.file_path,
                text_column=cfg.data.text_column,
                label_column=cfg.data.label_column,
                min_samples_per_class=cfg.data.min_samples_per_class,
                rare_group_label=cfg.data.rare_group_label,
                population_size=cfg.data.population_size,
                random_seed=cfg.data.random_seed,
                sheet_name=cfg.data.sheet_name,
            )

            # ── Instanciar componentes ──────────────────────────────────
            from ..classifiers.factory import get_classifier
            from ..embedders.factory import get_embedder
            from ..oracles.factory import get_oracle
            from ..query_strategies.factory import get_query_strategy

            embedder = (
                get_embedder(cfg.al.embedder) if cfg.al.embedder else None
            )

            # Ajustar embedder (necessário para PVProb etc.)
            if embedder is not None:
                all_texts = pd.concat(
                    [P_df[cfg.data.text_column], U_df[cfg.data.text_column]],
                    ignore_index=True,
                ).tolist()
                all_labels_fit = pd.concat(
                    [P_df[cfg.data.label_column], U_df[cfg.data.label_column]],
                    ignore_index=True,
                ).tolist()
                embedder.fit(all_texts, all_labels_fit)

            classifier = get_classifier(cfg.al.classifier)

            oracle_params = dict(cfg.al.oracle.params or {})
            if cfg.al.oracle.type == "Simulated" and "label_column" not in oracle_params:
                oracle_params["label_column"] = cfg.data.label_column
            from ..application.config import ComponentConfig
            oracle_cfg_with_label = ComponentConfig(
                type=cfg.al.oracle.type, params=oracle_params
            )
            oracle = get_oracle(oracle_cfg_with_label, all_possible_labels=all_labels)

            strategy = get_query_strategy(cfg.al.query_strategy, rng=rng)

            # ── Baseline ───────────────────────────────────────────────
            if cfg.baseline:
                try:
                    baseline_metrics = _compute_baseline(
                        P_df, U_df, cfg, embedder, all_labels, rng
                    )
                    result.baseline_metrics = baseline_metrics
                except Exception as exc:
                    logger.warning("Falha no baseline: %s", exc)
                    result.baseline_metrics = {"error": str(exc)}

            # ── Build ActiveLearner ─────────────────────────────────────
            budget = Budget(
                max_iterations=cfg.al.max_iterations,
                target_budget_pct=cfg.al.budget_pct,
                early_stopping_metric=cfg.al.early_stopping_metric,
                early_stopping_patience=cfg.al.early_stopping_patience,
                early_stopping_tolerance=cfg.al.early_stopping_tolerance,
            )

            learner = ActiveLearner(
                P_df=P_df,
                U_df=U_df,
                text_column=cfg.data.text_column,
                label_column=cfg.data.label_column,
                all_possible_labels=all_labels,
                classifier=classifier,
                oracle=oracle,
                query_strategy=strategy,
                budget=budget,
                rng=rng,
                embedder=embedder,
                experiment_name=cfg.name,
                internal_test_size=cfg.al.internal_test_size,
            )

            # ── Cold Start ─────────────────────────────────────────────
            n_classes = len(all_labels)
            from ..cold_start.factory import get_cold_start
            cs = get_cold_start(cfg.al.cold_start)

            cs_embeddings = None
            if embedder is not None and cfg.al.cold_start.type == "KM":
                cs_embeddings = embedder.transform(U_df[cfg.data.text_column].tolist())

            cs_params = cfg.al.cold_start.params or {}
            n_initial = cs_params.get("n_samples", n_classes)
            learner.cold_start(n_initial=n_initial, strategy=cs, embeddings=cs_embeddings)

            # ── Run ────────────────────────────────────────────────────
            al_result = learner.run()
            result.history = al_result.history
            result.status = al_result.status
            result.error_message = al_result.error_message

        except Exception as exc:
            logger.error(
                "Erro crítico no experimento '%s': %s", cfg.name, exc, exc_info=True
            )
            result.status = "FAILED"
            result.error_message = f"{type(exc).__name__}: {exc}"

        result.total_duration_sec = round(time.time() - exp_start, 2)
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_baseline(
    P_df: pd.DataFrame,
    U_df: pd.DataFrame,
    cfg: ExperimentConfig,
    embedder,
    all_labels: List[str],
    rng: np.random.Generator,
) -> dict:
    """Calcula métricas de baseline (treino em todo dataset, avalia em subconjunto)."""
    from sklearn.model_selection import train_test_split
    from ..classifiers.factory import get_classifier
    from ..domain.metrics import compute_accuracy, compute_f1_macro

    if cfg.baseline is None:
        return {}

    df_full = pd.concat([P_df, U_df], ignore_index=True)
    text_col = cfg.data.text_column
    label_col = cfg.data.label_column
    test_size = cfg.baseline.test_size

    rng_state = int(rng.integers(0, 2**31))
    train_df, test_df = train_test_split(
        df_full, test_size=test_size, random_state=rng_state, stratify=df_full[label_col]
    )

    clf = get_classifier(cfg.baseline.classifier)

    t0 = time.time()
    if embedder is not None:
        X_train = embedder.transform(train_df[text_col].tolist())
        X_test = embedder.transform(test_df[text_col].tolist())
    else:
        X_train = train_df[text_col].tolist()
        X_test = test_df[text_col].tolist()

    clf.fit(X_train, train_df[label_col].tolist())
    train_time = time.time() - t0

    y_pred = clf.predict(X_test)
    y_true = test_df[label_col].tolist()

    return {
        "baseline_acc": compute_accuracy(y_true, y_pred),
        "baseline_f1": compute_f1_macro(y_true, y_pred, labels=all_labels),
        "baseline_train_time_sec": round(train_time, 2),
    }
