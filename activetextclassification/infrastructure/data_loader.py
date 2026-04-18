"""
activetextclassification.infrastructure.data_loader
=====================================================
Carregamento e preparação de dados com logging em vez de print().
Envolve a função legada ``data_preparation.load_and_prepare_data`` mas usa
``logging`` para todas as mensagens.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def _preprocess_label(label: str) -> str:
    """Normaliza um rótulo: minúsculas, sem acentos, espaços comprimidos."""
    import re
    try:
        from unidecode import unidecode
    except ImportError:
        unidecode = lambda x: x  # type: ignore
    if not isinstance(label, str):
        label = str(label)
    label = label.lower()
    label = unidecode(label)
    label = re.sub(r"\s+", " ", label).strip()
    return label


def load_and_prepare_data(
    file_path: str,
    text_column: str,
    label_column: str,
    min_samples_per_class: int = 2,
    rare_group_label: str = "_RARE_GROUP_",
    population_size: float = 0.50,
    random_seed: int = 42,
    sheet_name: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[int, str], List[str]]:
    """
    Carrega dados de CSV ou Excel, pré-processa rótulos, agrupa classes raras
    e divide em População (P) e Pool não rotulado (U).

    Args:
        file_path:              Caminho para o arquivo CSV ou Excel.
        text_column:            Nome da coluna de texto.
        label_column:           Nome da coluna de rótulos.
        min_samples_per_class:  Mínimo de amostras para uma classe não ser agrupada.
        rare_group_label:       Rótulo para classes raras.
        population_size:        Fração usada como P (restante vai para U).
        random_seed:            Semente para reprodutibilidade.
        sheet_name:             Planilha para arquivos Excel.

    Returns:
        Tupla (P_df, U_df, label_to_id, id_to_label, all_possible_labels).

    Raises:
        FileNotFoundError: Se o arquivo não for encontrado.
        ValueError:         Se os dados forem inválidos ou colunas inexistentes.
    """
    logger.info("Carregando dados de: %s", file_path)

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        raise ValueError(f"Extensão não suportada: '{ext}'.  Use .csv, .xlsx ou .xls.")

    logger.info("Shape inicial: %s", df.shape)

    if text_column not in df.columns:
        raise ValueError(f"Coluna de texto '{text_column}' não encontrada.")
    if label_column not in df.columns:
        raise ValueError(f"Coluna de label '{label_column}' não encontrada.")

    before = len(df)
    df.dropna(subset=[text_column, label_column], inplace=True)
    logger.info("Shape após remover NaNs: %s (%d removidas).", df.shape, before - len(df))

    if df.empty:
        raise ValueError("DataFrame vazio após remover NaNs.")

    # ── Pré-processar rótulos ──────────────────────────────────────────
    df = df.copy()
    df[label_column] = df[label_column].apply(_preprocess_label).astype(str)

    # ── Agrupar classes raras ──────────────────────────────────────────
    if min_samples_per_class and min_samples_per_class > 1:
        counts = df[label_column].value_counts()
        rare = counts[counts < min_samples_per_class].index.tolist()
        if rare:
            logger.info(
                "Agrupando %d classes raras em '%s': %s%s",
                len(rare),
                rare_group_label,
                rare[:5],
                "..." if len(rare) > 5 else "",
            )
            df[label_column] = df[label_column].replace(rare, rare_group_label)

    # ── Mapeamento de rótulos ──────────────────────────────────────────
    all_possible_labels: List[str] = sorted(pd.unique(df[label_column]).tolist())
    label_to_id: Dict[str, int] = {lbl: i for i, lbl in enumerate(all_possible_labels)}
    id_to_label: Dict[int, str] = {i: lbl for lbl, i in label_to_id.items()}

    df["label_id"] = df[label_column].map(label_to_id)

    # ── Divisão P / U ─────────────────────────────────────────────────
    logger.info(
        "Dividindo em P=%.0f%% / U=%.0f%%.",
        population_size * 100,
        (1 - population_size) * 100,
    )

    # Estratificar apenas se todas as classes tiverem ≥ 2 amostras
    final_counts = df["label_id"].value_counts()
    stratify = df["label_id"] if (final_counts >= 2).all() else None
    if stratify is None:
        few = final_counts[final_counts < 2].index.map(id_to_label).tolist()
        logger.warning(
            "Divisão não estratificada — classes com < 2 amostras: %s", few
        )

    P_df, U_df = train_test_split(
        df,
        test_size=1.0 - population_size,
        random_state=random_seed,
        stratify=stratify,
    )

    logger.info("P: %d amostras | U: %d amostras | %d classes.", len(P_df), len(U_df), len(all_possible_labels))

    return (
        P_df.reset_index(drop=True),
        U_df.reset_index(drop=True),
        label_to_id,
        id_to_label,
        all_possible_labels,
    )
