"""Fixtures de integração e conftest para testes de integração."""

import numpy as np
import pandas as pd
import pytest


LABELS = ["alpha", "beta", "gamma"]
N_PER_CLASS = 15


@pytest.fixture(scope="module")
def integration_rng():
    return np.random.default_rng(seed=123)


@pytest.fixture(scope="module")
def integration_df():
    """Dataset sintético balanceado (45 amostras × 3 classes)."""
    records = []
    for label in LABELS:
        for i in range(N_PER_CLASS):
            records.append({
                "text": f"texto representativo da classe {label} exemplo {i}",
                "label": label,
            })
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(records)
    idx = rng.permutation(len(df))
    return df.iloc[idx].reset_index(drop=True)


@pytest.fixture(scope="module")
def P_df_int(integration_df):
    return integration_df.iloc[:20].reset_index(drop=True)


@pytest.fixture(scope="module")
def U_df_int(integration_df):
    return integration_df.iloc[20:].reset_index(drop=True)
