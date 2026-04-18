"""Testes unitários para activetextclassification.domain.metrics."""

import math
import numpy as np
import pytest

from activetextclassification.domain.metrics import (
    compute_accuracy,
    compute_f1_macro,
    compute_lce,
)


class TestComputeAccuracy:
    def test_perfect(self):
        y_true = ["a", "b", "c"]
        y_pred = ["a", "b", "c"]
        assert compute_accuracy(y_true, y_pred) == pytest.approx(1.0)

    def test_all_wrong(self):
        y_true = ["a", "a", "a"]
        y_pred = ["b", "b", "b"]
        assert compute_accuracy(y_true, y_pred) == pytest.approx(0.0)

    def test_partial(self):
        y_true = ["a", "b", "a", "b"]
        y_pred = ["a", "b", "b", "a"]
        assert compute_accuracy(y_true, y_pred) == pytest.approx(0.5)

    def test_empty_returns_nan(self):
        result = compute_accuracy([], [])
        assert math.isnan(result)


class TestComputeF1Macro:
    def test_perfect(self):
        y_true = ["a", "b", "c"] * 4
        y_pred = ["a", "b", "c"] * 4
        assert compute_f1_macro(y_true, y_pred) == pytest.approx(1.0)

    def test_all_wrong_zero_division(self):
        y_true = ["a", "a"]
        y_pred = ["b", "b"]
        result = compute_f1_macro(y_true, y_pred, zero_division=0.0)
        assert result == pytest.approx(0.0)

    def test_with_labels(self):
        y_true = ["a", "b"]
        y_pred = ["a", "b"]
        # Adding unseen label "c" should not raise
        result = compute_f1_macro(y_true, y_pred, labels=["a", "b", "c"], zero_division=0.0)
        assert 0.0 <= result <= 1.0

    def test_empty_returns_nan(self):
        result = compute_f1_macro([], [])
        assert math.isnan(result)


class TestComputeLCE:
    def test_perfect_linear_improvement(self):
        l_sizes = [10, 20, 30]
        scores = [0.9, 0.9, 0.9]
        baseline = 0.9
        lce = compute_lce(l_sizes, scores, baseline)
        assert lce == pytest.approx(1.0, rel=1e-3)

    def test_above_baseline(self):
        l_sizes = [10, 20, 30]
        scores = [0.95, 0.95, 0.95]
        baseline = 0.90
        lce = compute_lce(l_sizes, scores, baseline)
        assert lce > 1.0

    def test_below_baseline(self):
        l_sizes = [10, 20, 30]
        scores = [0.5, 0.5, 0.5]
        baseline = 0.90
        lce = compute_lce(l_sizes, scores, baseline)
        assert lce < 1.0

    def test_nan_baseline_returns_nan(self):
        result = compute_lce([1, 2], [0.5, 0.6], float("nan"))
        assert math.isnan(result)

    def test_too_few_points_returns_nan(self):
        result = compute_lce([10], [0.8], 0.9)
        assert math.isnan(result)

    def test_mismatched_lengths_returns_nan(self):
        result = compute_lce([10, 20], [0.8], 0.9)
        assert math.isnan(result)

    def test_zero_baseline_returns_nan(self):
        result = compute_lce([10, 20], [0.5, 0.6], 0.0)
        assert math.isnan(result)
