"""
Fixtures compartilhadas para todos os testes.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Dados sintéticos reutilizáveis
# ---------------------------------------------------------------------------

LABELS = ["catA", "catB", "catC"]
N_SAMPLES = 60  # 20 por classe


@pytest.fixture(scope="session")
def rng_seed42():
    """np.random.Generator com seed fixo para testes reprodutíveis."""
    return np.random.default_rng(seed=42)


@pytest.fixture(scope="session")
def synthetic_texts():
    """Lista de textos sintéticos (20 por classe = 60 total)."""
    texts = []
    for label in LABELS:
        texts += [f"texto de exemplo para classe {label} item {i}" for i in range(N_SAMPLES // len(LABELS))]
    return texts


@pytest.fixture(scope="session")
def synthetic_labels():
    """Lista de rótulos correspondentes aos synthetic_texts."""
    labels = []
    for label in LABELS:
        labels += [label] * (N_SAMPLES // len(LABELS))
    return labels


@pytest.fixture(scope="session")
def synthetic_df(synthetic_texts, synthetic_labels):
    """DataFrame sintético com colunas 'text' e 'label'."""
    rng = np.random.default_rng(seed=0)
    idx = rng.permutation(len(synthetic_texts))
    texts = [synthetic_texts[i] for i in idx]
    labels = [synthetic_labels[i] for i in idx]
    return pd.DataFrame({"text": texts, "label": labels})


@pytest.fixture(scope="session")
def P_df(synthetic_df):
    """30 amostras como população (avaliação externa)."""
    return synthetic_df.iloc[:30].reset_index(drop=True)


@pytest.fixture(scope="session")
def U_df(synthetic_df):
    """30 amostras como pool não rotulado."""
    return synthetic_df.iloc[30:].reset_index(drop=True)


@pytest.fixture(scope="session")
def simple_probs():
    """Array de probabilidades sintético (30 amostras × 3 classes)."""
    rng = np.random.default_rng(seed=7)
    raw = rng.dirichlet(alpha=[1, 1, 1], size=30)
    return raw.astype(np.float64)
