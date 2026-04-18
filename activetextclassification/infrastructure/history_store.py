"""
activetextclassification.infrastructure.history_store
======================================================
Armazenamento e leitura do histórico de experimentos em formato JSON Lines.

Usa ``logging`` em vez de ``print()``.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HistoryStore:
    """
    Gerencia leitura e escrita do log de execuções em formato JSON Lines (.jsonl).

    Cada linha do arquivo é um objeto JSON completo representando uma execução
    de experimento (resultado de ``ExperimentResult.to_summary_dict()``).

    Args:
        log_file_path: Caminho para o arquivo ``.jsonl``.
    """

    def __init__(self, log_file_path: str = "history_log.jsonl"):
        self.log_file_path = log_file_path
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        logger.info("HistoryStore em: %s", os.path.abspath(log_file_path))

    # ------------------------------------------------------------------ #
    #  Write                                                               #
    # ------------------------------------------------------------------ #

    def append(self, result_dict: dict) -> None:
        """
        Anexa um dicionário de resultado ao log JSONL.

        Args:
            result_dict: Dicionário serializável em JSON (resultado de
                         ``ExperimentResult.to_summary_dict()``).
        """
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_dict, ensure_ascii=False, default=str) + "\n")
            logger.info(
                "Histórico de '%s' salvo em %s.",
                result_dict.get("experiment_name", "?"),
                self.log_file_path,
            )
        except Exception as exc:
            logger.error("Falha ao salvar histórico: %s", exc)

    # ------------------------------------------------------------------ #
    #  Read                                                                #
    # ------------------------------------------------------------------ #

    def get_completed_names(self) -> Set[str]:
        """
        Retorna o conjunto de nomes de experimentos com status ``'COMPLETED'``.
        """
        completed: Set[str] = set()
        if not os.path.exists(self.log_file_path):
            return completed
        malformed = 0
        with open(self.log_file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("status") == "COMPLETED" and entry.get(
                        "experiment_name"
                    ):
                        completed.add(entry["experiment_name"])
                except json.JSONDecodeError:
                    malformed += 1
        if malformed:
            logger.warning("%d linhas mal formadas ignoradas.", malformed)
        return completed

    def load_flat_dataframe(self) -> pd.DataFrame:
        """
        Carrega todas as entradas ``'COMPLETED'`` e achata o histórico de
        iterações em um único ``DataFrame``.

        Equivalente ao ``load_and_flatten_experiment_history()`` de ``utils.py``,
        mas sem ``print()``.

        Returns:
            ``DataFrame`` com uma linha por iteração de cada experimento concluído.
            Retorna DataFrame vazio se não houver dados.
        """
        if not os.path.exists(self.log_file_path):
            logger.warning("Arquivo de log não encontrado: %s", self.log_file_path)
            return pd.DataFrame()

        records: List[dict] = []
        processed = 0
        malformed = 0

        with open(self.log_file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue

                if entry.get("status") != "COMPLETED":
                    continue
                history_data = entry.get("history_data") or []
                if not history_data:
                    continue

                processed += 1
                meta = _extract_meta(entry)

                for iter_data in history_data:
                    if isinstance(iter_data, dict):
                        records.append({**meta, **iter_data})

        if not records:
            logger.info("Nenhum dado válido encontrado em %s.", self.log_file_path)
            return pd.DataFrame()

        df = pd.DataFrame(records)
        _coerce_numeric(df)
        if malformed:
            logger.warning("%d linhas mal formadas ignoradas.", malformed)
        logger.info(
            "%d registros de %d experimentos carregados.", len(records), processed
        )
        return df

    def load_entries(self) -> List[dict]:
        """Carrega todas as entradas do log como lista de dicionários."""
        entries: List[dict] = []
        if not os.path.exists(self.log_file_path):
            return entries
        with open(self.log_file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return entries


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_meta(entry: dict) -> dict:
    config = entry.get("config") or {}
    al_params = config.get("al_params") or {}
    cs = al_params.get("cold_start_config") or {}
    clf = al_params.get("classifier_config") or {}
    qs = al_params.get("query_strategy_config") or {}
    qs_params = qs.get("params") or {}
    data_p = config.get("data_params") or {}
    gen_p = config.get("general_params") or {}
    baseline = entry.get("baseline_metrics") or {}
    return {
        "experiment_name": entry.get("experiment_name", ""),
        "execution_timestamp": entry.get("execution_timestamp", ""),
        "overall_experiment_duration_sec": entry.get("overall_experiment_duration_sec"),
        "loop_run_duration_sec": entry.get("total_duration_sec"),
        "baseline_acc": baseline.get("baseline_acc"),
        "baseline_f1": baseline.get("baseline_f1"),
        "cold_start_type": cs.get("type", "N/A"),
        "classifier_type": clf.get("type", "N/A"),
        "classifier_params_str": str(clf.get("params", {})),
        "query_strategy_type": qs.get("type", "N/A"),
        "query_batch_size": qs_params.get("batch_size", float("nan")),
        "query_entropy_fraction": qs_params.get("entropy_fraction", float("nan")),
        "min_samples_per_class": data_p.get("min_samples_per_class"),
        "population_size_pct": data_p.get("population_size"),
        "random_seed": gen_p.get("random_seed"),
    }


_NUMERIC_COLS = [
    "baseline_acc", "baseline_f1", "overall_experiment_duration_sec",
    "loop_run_duration_sec", "iteration", "L_size", "l_size",
    "internal_acc", "internal_f1", "external_acc", "external_f1",
    "iteration_duration_sec", "train_duration_sec", "eval_duration_sec",
    "query_duration_sec", "update_duration_sec", "U_size", "u_size",
    "query_batch_size", "query_entropy_fraction",
    "min_samples_per_class", "population_size_pct", "random_seed",
]


def _coerce_numeric(df: pd.DataFrame) -> None:
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
