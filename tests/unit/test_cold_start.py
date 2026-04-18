"""Testes unitários para activetextclassification.cold_start."""

import numpy as np
import pandas as pd
import pytest

from activetextclassification.cold_start import RandomColdStart, KMediansColdStart


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def small_df():
    return pd.DataFrame({"text": [f"texto {i}" for i in range(20)], "label": ["a"] * 10 + ["b"] * 10})


@pytest.fixture
def simple_embeddings():
    rng = np.random.default_rng(3)
    return rng.standard_normal((20, 8)).astype(np.float32)


class TestRandomColdStart:
    def test_returns_correct_count(self, small_df, rng):
        cs = RandomColdStart()
        indices = cs.select(small_df, n_initial=5, rng=rng)
        assert len(indices) == 5

    def test_no_duplicates(self, small_df, rng):
        cs = RandomColdStart()
        indices = cs.select(small_df, n_initial=10, rng=rng)
        assert len(set(indices)) == 10

    def test_clipped_to_available(self, small_df, rng):
        cs = RandomColdStart()
        indices = cs.select(small_df, n_initial=100, rng=rng)
        assert len(indices) == len(small_df)

    def test_zero_returns_empty(self, small_df, rng):
        cs = RandomColdStart()
        indices = cs.select(small_df, n_initial=0, rng=rng)
        assert len(indices) == 0

    def test_reproducible(self, small_df):
        cs = RandomColdStart()
        rng_a = np.random.default_rng(seed=7)
        rng_b = np.random.default_rng(seed=7)
        a = cs.select(small_df, n_initial=5, rng=rng_a)
        b = cs.select(small_df, n_initial=5, rng=rng_b)
        np.testing.assert_array_equal(a, b)


class TestKMediansColdStart:
    def test_returns_correct_count(self, small_df, simple_embeddings, rng):
        pytest.importorskip("sklearn_extra")
        cs = KMediansColdStart()
        indices = cs.select(small_df, n_initial=3, embeddings=simple_embeddings, rng=rng)
        assert len(indices) == 3

    def test_requires_embeddings(self, small_df, rng):
        cs = KMediansColdStart()
        with pytest.raises(ValueError, match="embeddings"):
            cs.select(small_df, n_initial=3, rng=rng)

    def test_no_duplicates(self, small_df, simple_embeddings, rng):
        pytest.importorskip("sklearn_extra")
        cs = KMediansColdStart()
        indices = cs.select(small_df, n_initial=5, embeddings=simple_embeddings, rng=rng)
        assert len(set(indices)) == len(indices)
