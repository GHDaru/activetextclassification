"""Testes unitários para activetextclassification.query_strategies."""

import numpy as np
import pytest

from activetextclassification.query_strategies import (
    RandomStrategy,
    EntropyStrategy,
    LeastConfidenceStrategy,
    SmallestMarginStrategy,
    HybridStrategy,
    get_query_strategy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def uniform_probs():
    """Probabilidades uniformes — não há preferência."""
    n = 20
    return np.full((n, 3), 1 / 3)


@pytest.fixture
def peaked_probs():
    """Probabilidades com alta confiança na classe 0 para todos."""
    n = 20
    p = np.zeros((n, 3))
    p[:, 0] = 0.9
    p[:, 1] = 0.05
    p[:, 2] = 0.05
    return p


@pytest.fixture
def mixed_probs():
    """Metade alta confiança, metade incerta."""
    rng = np.random.default_rng(0)
    probs = np.zeros((20, 3))
    # Primeiros 10: muito confiantes
    probs[:10, 0] = 0.95
    probs[:10, 1] = 0.025
    probs[:10, 2] = 0.025
    # Últimos 10: incertos
    probs[10:] = rng.dirichlet([1, 1, 1], size=10)
    return probs


# ---------------------------------------------------------------------------
# RandomStrategy
# ---------------------------------------------------------------------------

class TestRandomStrategy:
    def test_returns_correct_count(self, rng):
        strategy = RandomStrategy(batch_size=5)
        indices = strategy.select(pool_size=20, rng=rng)
        assert len(indices) == 5

    def test_no_duplicates(self, rng):
        strategy = RandomStrategy(batch_size=10)
        indices = strategy.select(pool_size=20, rng=rng)
        assert len(indices) == len(set(indices))

    def test_clipped_to_pool(self, rng):
        strategy = RandomStrategy(batch_size=100)
        indices = strategy.select(pool_size=5, rng=rng)
        assert len(indices) == 5

    def test_empty_pool(self, rng):
        strategy = RandomStrategy(batch_size=5)
        indices = strategy.select(pool_size=0, rng=rng)
        assert len(indices) == 0

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError):
            RandomStrategy(batch_size=0)

    def test_reproducible(self):
        s = RandomStrategy(batch_size=5)
        rng_a = np.random.default_rng(seed=99)
        rng_b = np.random.default_rng(seed=99)
        a = s.select(20, rng=rng_a)
        b = s.select(20, rng=rng_b)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# EntropyStrategy
# ---------------------------------------------------------------------------

class TestEntropyStrategy:
    def test_selects_uncertain(self, mixed_probs, rng):
        """Os 5 mais incertos devem vir dos últimos 10 (probabilidades ~uniformes)."""
        strategy = EntropyStrategy(batch_size=5)
        indices = strategy.select(pool_size=20, probabilities=mixed_probs, rng=rng)
        assert len(indices) == 5
        # A maioria dos selecionados deve ser dos índices 10-19 (incertos)
        high_entropy_count = sum(idx >= 10 for idx in indices)
        assert high_entropy_count >= 4, f"Esperado >= 4 incertos, obteve {high_entropy_count}"

    def test_requires_probabilities(self, rng):
        strategy = EntropyStrategy(batch_size=5)
        with pytest.raises(ValueError):
            strategy.select(pool_size=10, probabilities=None, rng=rng)


# ---------------------------------------------------------------------------
# LeastConfidenceStrategy
# ---------------------------------------------------------------------------

class TestLeastConfidenceStrategy:
    def test_selects_low_confidence(self, mixed_probs, rng):
        strategy = LeastConfidenceStrategy(batch_size=5)
        indices = strategy.select(pool_size=20, probabilities=mixed_probs, rng=rng)
        assert len(indices) == 5
        low_conf_count = sum(idx >= 10 for idx in indices)
        assert low_conf_count >= 4

    def test_requires_probabilities(self):
        strategy = LeastConfidenceStrategy(batch_size=5)
        with pytest.raises(ValueError):
            strategy.select(pool_size=10, probabilities=None)


# ---------------------------------------------------------------------------
# SmallestMarginStrategy
# ---------------------------------------------------------------------------

class TestSmallestMarginStrategy:
    def test_selects_small_margin(self, mixed_probs, rng):
        strategy = SmallestMarginStrategy(batch_size=5)
        indices = strategy.select(pool_size=20, probabilities=mixed_probs, rng=rng)
        assert len(indices) == 5

    def test_requires_probabilities(self):
        strategy = SmallestMarginStrategy(batch_size=5)
        with pytest.raises(ValueError):
            strategy.select(pool_size=10, probabilities=None)


# ---------------------------------------------------------------------------
# HybridStrategy
# ---------------------------------------------------------------------------

class TestHybridStrategy:
    def test_total_count(self, mixed_probs, rng):
        strategy = HybridStrategy(batch_size=10, entropy_fraction=0.5)
        indices = strategy.select(pool_size=20, probabilities=mixed_probs, rng=rng)
        assert len(indices) == 10

    def test_no_duplicates(self, mixed_probs, rng):
        strategy = HybridStrategy(batch_size=8, entropy_fraction=0.5)
        indices = strategy.select(pool_size=20, probabilities=mixed_probs, rng=rng)
        assert len(set(indices)) == len(indices)

    def test_pure_entropy(self, mixed_probs, rng):
        strategy = HybridStrategy(batch_size=5, entropy_fraction=1.0)
        indices = strategy.select(pool_size=20, probabilities=mixed_probs, rng=rng)
        assert len(indices) == 5

    def test_pure_random(self, mixed_probs, rng):
        strategy = HybridStrategy(batch_size=5, entropy_fraction=0.0)
        indices = strategy.select(pool_size=20, probabilities=mixed_probs, rng=rng)
        assert len(indices) == 5

    def test_invalid_entropy_fraction(self):
        with pytest.raises(ValueError):
            HybridStrategy(batch_size=5, entropy_fraction=1.5)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestGetQueryStrategy:
    @pytest.mark.parametrize("type_", ["RND", "ENT", "LCO", "SMA", "HYB"])
    def test_all_types_instantiate(self, type_):
        from activetextclassification.application.config import ComponentConfig
        cfg = ComponentConfig(type=type_, params={"batch_size": 5})
        strategy = get_query_strategy(cfg)
        assert strategy.batch_size == 5

    def test_unknown_type_raises(self):
        from activetextclassification.application.config import ComponentConfig
        with pytest.raises(ValueError, match="desconhecida"):
            get_query_strategy(ComponentConfig(type="UNKNOWN"))
