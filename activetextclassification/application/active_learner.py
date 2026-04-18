"""
activetextclassification.application.active_learner
=====================================================
Orquestrador puro do ciclo de Aprendizado Ativo.

O ``ActiveLearner`` desta camada **não** realiza I/O.  Recebe objetos já
instanciados e um ``np.random.Generator`` explícito para reprodutibilidade
total.

Exemplo::

    import numpy as np
    from activetextclassification.application.active_learner import ActiveLearner
    from activetextclassification.classifiers import get_classifier
    from activetextclassification.oracles import SimulatedOracle
    from activetextclassification.query_strategies import EntropyStrategy
    from activetextclassification.domain.entities import Budget

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
"""

from __future__ import annotations

import logging
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from ..domain.entities import Budget, ExperimentResult, IterationRecord
from ..domain.interfaces import (
    IClassifier,
    IColdStart,
    IEmbedder,
    IOracle,
    IQueryStrategy,
)
from ..domain.metrics import compute_accuracy, compute_f1_macro

logger = logging.getLogger(__name__)


class ActiveLearner:
    """
    Orquestrador puro do ciclo de Aprendizado Ativo.

    Não realiza I/O.  Toda configuração é injetada via construtor
    (padrão de Injeção de Dependência).

    Args:
        P_df:               DataFrame da população (conjunto de teste externo).
        U_df:               DataFrame do pool não rotulado.
        text_column:        Nome da coluna de texto.
        label_column:       Nome da coluna de rótulos.
        all_possible_labels: Lista de todos os rótulos possíveis.
        classifier:         Instância de ``IClassifier``.
        oracle:             Instância de ``IOracle``.
        query_strategy:     Instância de ``IQueryStrategy``.
        budget:             Critérios de parada (``Budget``).
        rng:                ``np.random.Generator`` para reprodutibilidade.
                            Se None, cria gerador sem semente.
        embedder:           Instância opcional de ``IEmbedder`` (necessário
                            para classificadores baseados em features).
        experiment_name:    Nome do experimento (para relatórios).
        internal_test_size: Fração de L reservada para avaliação interna.
    """

    def __init__(
        self,
        P_df: pd.DataFrame,
        U_df: pd.DataFrame,
        text_column: str,
        label_column: str,
        all_possible_labels: List[str],
        classifier: IClassifier,
        oracle: IOracle,
        query_strategy: IQueryStrategy,
        budget: Optional[Budget] = None,
        rng: Optional[np.random.Generator] = None,
        embedder: Optional[IEmbedder] = None,
        experiment_name: str = "experiment",
        internal_test_size: float = 0.20,
    ):
        # ── State ──────────────────────────────────────────────────────
        self.P_df = P_df.reset_index(drop=True).copy()
        self.U_df = U_df.reset_index(drop=True).copy()
        self.L_df: pd.DataFrame = pd.DataFrame(columns=U_df.columns)

        self.text_column = text_column
        self.label_column = label_column
        self.all_possible_labels = list(all_possible_labels)
        self.experiment_name = experiment_name
        self.internal_test_size = internal_test_size

        # ── Injected components ────────────────────────────────────────
        self.classifier = classifier
        self.oracle = oracle
        self.query_strategy = query_strategy
        self.embedder = embedder

        # ── Budget / stopping criteria ─────────────────────────────────
        self.budget = budget or Budget()

        # ── Reproducibility ────────────────────────────────────────────
        self.rng = rng if rng is not None else np.random.default_rng()

        # ── Internal state ─────────────────────────────────────────────
        self._original_u_size = len(U_df)
        self._current_iteration = 0
        self._best_metric: float = -np.inf
        self._patience_counter: int = 0
        self._active_classifier: Optional[IClassifier] = None
        self.status: str = "READY"

    # ------------------------------------------------------------------ #
    #  Cold Start                                                          #
    # ------------------------------------------------------------------ #

    def cold_start(
        self,
        n_initial: int,
        strategy: Optional[IColdStart] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> None:
        """
        Seleciona o lote inicial L0 e o move de U para L.

        Args:
            n_initial:  Número de amostras para o lote inicial.
            strategy:   Instância de ``IColdStart``.  Se None, usa seleção aleatória.
            embeddings: Embeddings de U (necessário para estratégias baseadas em
                        clustering).
        """
        if strategy is not None:
            indices = strategy.select(
                self.U_df, n_initial, embeddings=embeddings, rng=self.rng
            )
        else:
            # Fallback aleatório
            n = min(n_initial, len(self.U_df))
            indices = self.rng.choice(len(self.U_df), size=n, replace=False)

        if len(indices) == 0:
            logger.warning("Cold start retornou zero índices — L0 estará vazio.")
            return

        self.L_df = self.U_df.iloc[indices].copy()
        self.U_df = self.U_df.drop(self.U_df.index[indices]).reset_index(drop=True)
        logger.info("Cold start: L0=%d | U restante=%d.", len(self.L_df), len(self.U_df))

    # ------------------------------------------------------------------ #
    #  Run                                                                 #
    # ------------------------------------------------------------------ #

    def run(self) -> ExperimentResult:
        """
        Executa o loop completo de Aprendizado Ativo.

        Returns:
            ``ExperimentResult`` com histórico de iterações e métricas.
        """
        result = ExperimentResult(experiment_name=self.experiment_name)
        self.status = "RUNNING"
        run_start = time.time()

        logger.info(
            "Iniciando run '%s' | max_iter=%d | budget=%.0f%%",
            self.experiment_name,
            self.budget.max_iterations,
            self.budget.target_budget_pct * 100,
        )

        try:
            with tqdm(
                total=self.budget.max_iterations,
                desc=f"AL [{self.experiment_name}]",
            ) as pbar:
                while self._current_iteration < self.budget.max_iterations:
                    record, should_continue = self._step()
                    result.history.append(record)
                    pbar.update(1)

                    if not should_continue:
                        pbar.total = self._current_iteration
                        pbar.refresh()
                        break

        except Exception as exc:
            logger.error("Erro durante o run: %s", exc, exc_info=True)
            self.status = "FAILED"
            result.status = "FAILED"
            result.error_message = f"{type(exc).__name__}: {exc}"
        else:
            self.status = "COMPLETED"
            result.status = "COMPLETED"

        result.total_duration_sec = round(time.time() - run_start, 2)
        logger.info(
            "Run concluído: status=%s | duração=%.2fs | iterações=%d",
            result.status,
            result.total_duration_sec,
            self._current_iteration,
        )
        return result

    def step(self) -> IterationRecord:
        """Executa UMA iteração e retorna o registro."""
        record, _ = self._step()
        return record

    # ------------------------------------------------------------------ #
    #  Private: single iteration                                           #
    # ------------------------------------------------------------------ #

    def _step(self):
        """Executa uma iteração.  Retorna (IterationRecord, should_continue)."""
        iter_start = time.time()
        n_labeled = len(self.L_df)
        labeled_pct = n_labeled / self._original_u_size if self._original_u_size else 1.0

        # ── Stopping criteria ─────────────────────────────────────────
        if labeled_pct >= self.budget.target_budget_pct:
            logger.info("Critério de parada: budget %.1f%% atingido.", labeled_pct * 100)
            return self._empty_record("STOPPED_BUDGET"), False

        if len(self.U_df) == 0:
            logger.info("Critério de parada: pool U esgotado.")
            return self._empty_record("STOPPED_EMPTY_POOL"), False

        fit_dur = eval_dur = query_dur = update_dur = float("nan")
        int_acc = int_f1 = ext_acc = ext_f1 = float("nan")

        try:
            # 1. Split interno de L
            train_df, test_df = self._split_l()
            if train_df.empty:
                self._current_iteration += 1
                return IterationRecord(
                    iteration=self._current_iteration,
                    l_size=n_labeled,
                    u_size=len(self.U_df),
                    status="SKIPPED_EMPTY_TRAIN",
                    iteration_duration_sec=time.time() - iter_start,
                ), True

            # 2. Preparar inputs
            X_train, X_test, X_P, X_U = self._prepare_inputs(train_df, test_df)

            # 3. Treinar
            fit_dur = self._train(X_train, train_df[self.label_column].tolist())

            # 4. Avaliar
            int_acc, int_f1, ext_acc, ext_f1, eval_dur = self._evaluate(
                X_test, test_df, X_P
            )

            # 5. Query
            query_indices, query_dur = self._query(X_U)

            # 6. Update L/U via oracle
            update_dur = self._update(query_indices)

        except Exception as exc:
            logger.error("Erro na iteração %d: %s", self._current_iteration + 1, exc, exc_info=True)
            self.status = "FAILED"
            record = IterationRecord(
                iteration=self._current_iteration + 1,
                l_size=n_labeled,
                u_size=len(self.U_df),
                status="FAILED_ITERATION",
                internal_acc=int_acc,
                internal_f1=int_f1,
                external_acc=ext_acc,
                external_f1=ext_f1,
                iteration_duration_sec=time.time() - iter_start,
                train_duration_sec=fit_dur,
                eval_duration_sec=eval_dur,
                query_duration_sec=query_dur,
                update_duration_sec=update_dur,
                error=str(exc),
            )
            return record, False

        # ── Record ────────────────────────────────────────────────────
        record = IterationRecord(
            iteration=self._current_iteration + 1,
            l_size=n_labeled,
            u_size=len(self.U_df),
            status="COMPLETED_ITERATION",
            internal_acc=int_acc,
            internal_f1=int_f1,
            external_acc=ext_acc,
            external_f1=ext_f1,
            iteration_duration_sec=round(time.time() - iter_start, 4),
            train_duration_sec=fit_dur,
            eval_duration_sec=eval_dur,
            query_duration_sec=query_dur,
            update_duration_sec=update_dur,
        )
        logger.debug(
            "Iter %d | L=%d U=%d | ext_acc=%.4f ext_f1=%.4f | %.2fs",
            record.iteration,
            n_labeled,
            len(self.U_df),
            ext_acc,
            ext_f1,
            record.iteration_duration_sec,
        )

        # ── Early stopping ─────────────────────────────────────────────
        if self._check_early_stopping(ext_acc, ext_f1):
            self._current_iteration += 1
            return record, False

        self._current_iteration += 1
        return record, True

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _split_l(self):
        """Divide L em treino interno / teste interno."""
        if (
            len(self.L_df) < 10
            or self.internal_test_size <= 0
        ):
            return self.L_df, pd.DataFrame(columns=self.L_df.columns)

        try:
            return train_test_split(
                self.L_df,
                test_size=self.internal_test_size,
                random_state=self._rng_int(),
                stratify=self.L_df[self.label_column],
            )
        except ValueError:
            return train_test_split(
                self.L_df,
                test_size=self.internal_test_size,
                random_state=self._rng_int(),
            )

    def _prepare_inputs(self, train_df, test_df):
        """Retorna (X_train, X_test, X_P, X_U) prontos para o classificador."""
        if self.embedder is not None:
            X_train = self.embedder.transform(train_df[self.text_column].tolist())
            X_test = (
                self.embedder.transform(test_df[self.text_column].tolist())
                if not test_df.empty
                else np.empty((0, self.embedder.get_embedding_dimension() or 0))
            )
            X_P = self.embedder.transform(self.P_df[self.text_column].tolist())
            X_U = (
                self.embedder.transform(self.U_df[self.text_column].tolist())
                if not self.U_df.empty
                else np.empty((0, self.embedder.get_embedding_dimension() or 0))
            )
        else:
            X_train = train_df[self.text_column].tolist()
            X_test = test_df[self.text_column].tolist() if not test_df.empty else []
            X_P = self.P_df[self.text_column].tolist()
            X_U = self.U_df[self.text_column].tolist() if not self.U_df.empty else []
        return X_train, X_test, X_P, X_U

    def _train(self, X_train, y_labels: List[str]) -> float:
        start = time.time()
        self._active_classifier = self.classifier
        self._active_classifier.fit(X_train, y_labels)
        return round(time.time() - start, 4)

    def _evaluate(self, X_test, test_df, X_P):
        start = time.time()
        int_acc = int_f1 = ext_acc = ext_f1 = float("nan")

        clf = self._active_classifier
        if clf is None:
            return int_acc, int_f1, ext_acc, ext_f1, 0.0

        try:
            # Internal
            if X_test is not None and len(X_test) > 0 and not test_df.empty:
                y_pred = clf.predict(X_test)
                y_true = test_df[self.label_column].tolist()
                int_acc = compute_accuracy(y_true, y_pred)
                int_f1 = compute_f1_macro(y_true, y_pred, labels=self.all_possible_labels)

            # External
            if X_P is not None and len(X_P) > 0:
                y_pred_ext = clf.predict(X_P)
                y_true_ext = self.P_df[self.label_column].tolist()
                ext_acc = compute_accuracy(y_true_ext, y_pred_ext)
                ext_f1 = compute_f1_macro(
                    y_true_ext, y_pred_ext, labels=self.all_possible_labels
                )
        except Exception as exc:
            logger.warning("Erro durante avaliação: %s", exc)

        return int_acc, int_f1, ext_acc, ext_f1, round(time.time() - start, 4)

    def _query(self, X_U):
        """Seleciona próximo lote de U.  Retorna (indices, duration)."""
        start = time.time()

        if not self.U_df.empty and len(X_U) > 0:
            pool_size = len(self.U_df)
            probs = None

            needs_proba = not isinstance(self.query_strategy.__class__.__name__, str) or \
                "Random" not in type(self.query_strategy).__name__

            # More precise check
            from ..query_strategies.random_strategy import RandomStrategy
            needs_proba = not isinstance(self.query_strategy, RandomStrategy)

            if needs_proba and self._active_classifier is not None:
                try:
                    probs = self._active_classifier.predict_proba(X_U)
                    if probs is None or probs.shape[0] != pool_size:
                        probs = None
                except Exception as exc:
                    logger.warning("predict_proba falhou, usando Random: %s", exc)
                    from ..query_strategies.random_strategy import RandomStrategy
                    _fallback = RandomStrategy(self.query_strategy.batch_size)
                    indices = _fallback.select(pool_size, rng=self.rng)
                    return indices.astype(int), round(time.time() - start, 4)

            indices = self.query_strategy.select(pool_size, probabilities=probs, rng=self.rng)
        else:
            indices = np.array([], dtype=int)

        return indices.astype(int), round(time.time() - start, 4)

    def _update(self, query_indices: np.ndarray) -> float:
        """Consulta oráculo, atualiza L e remove de U.  Retorna duração."""
        start = time.time()

        if len(query_indices) == 0 or len(query_indices) > len(self.U_df):
            return round(time.time() - start, 4)

        queried_df = self.U_df.iloc[query_indices].copy()
        labels = self.oracle.query(queried_df)

        if not isinstance(labels, (list, np.ndarray)) or len(labels) != len(queried_df):
            logger.warning(
                "Oráculo retornou dados inesperados (tipo=%s, tamanho=%s).",
                type(labels),
                len(labels) if hasattr(labels, "__len__") else "?",
            )
            return round(time.time() - start, 4)

        # Filter None labels
        valid_mask = [lbl is not None for lbl in labels]
        valid_df = queried_df[valid_mask].copy()
        valid_labels = [lbl for lbl, ok in zip(labels, valid_mask) if ok]

        if valid_labels:
            valid_df[self.label_column] = valid_labels
            self.L_df = pd.concat([self.L_df, valid_df], ignore_index=True)

        # Remove queried items from U (even if oracle failed)
        self.U_df = self.U_df.drop(self.U_df.index[query_indices]).reset_index(drop=True)

        logger.debug(
            "Update: %d rotulados | L=%d | U=%d",
            len(valid_labels),
            len(self.L_df),
            len(self.U_df),
        )
        return round(time.time() - start, 4)

    def _check_early_stopping(self, ext_acc: float, ext_f1: float) -> bool:
        metric = self.budget.early_stopping_metric
        patience = self.budget.early_stopping_patience
        tol = self.budget.early_stopping_tolerance

        if not metric or not patience:
            return False

        value_map = {
            "external_acc": ext_acc,
            "external_f1": ext_f1,
        }
        current = value_map.get(metric, float("nan"))

        if np.isnan(current):
            return False

        if current > self._best_metric + tol:
            self._best_metric = current
            self._patience_counter = 0
        else:
            self._patience_counter += 1
            if self._patience_counter >= patience:
                logger.info(
                    "Early stopping atingido após %d iterações sem melhora.", patience
                )
                return True
        return False

    def _rng_int(self) -> int:
        """Gera um inteiro aleatório para uso como random_state."""
        return int(self.rng.integers(0, 2**31))

    def _empty_record(self, status: str) -> IterationRecord:
        return IterationRecord(
            iteration=self._current_iteration,
            l_size=len(self.L_df),
            u_size=len(self.U_df),
            status=status,
        )

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def current_model(self) -> Optional[IClassifier]:
        return self._active_classifier

    def get_history_dataframe(self) -> pd.DataFrame:
        """Compatibilidade com o ActiveLearner legado."""
        return self.L_df  # Retorna estado atual de L; history é retornado pelo run()
