"""Testes unitários para activetextclassification.classifiers."""

import numpy as np
import pytest

from activetextclassification.classifiers import (
    GNBClassifier,
    LRClassifier,
    get_classifier,
)
from activetextclassification.application.config import ComponentConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def feature_data():
    """X (float array) e y (labels string) para classificadores de features."""
    rng = np.random.default_rng(seed=0)
    X_a = rng.standard_normal((20, 4)) + np.array([2, 2, 2, 2])
    X_b = rng.standard_normal((20, 4)) + np.array([-2, -2, -2, -2])
    X = np.vstack([X_a, X_b]).astype(np.float32)
    y = ["classA"] * 20 + ["classB"] * 20
    return X, y


@pytest.fixture
def text_data():
    """Textos e rótulos para classificadores de texto."""
    texts = (
        ["gato felino miau"] * 20
        + ["cachorro latir"] * 20
    )
    labels = ["animal_gato"] * 20 + ["animal_cachorro"] * 20
    return texts, labels


# ---------------------------------------------------------------------------
# SklearnClassifier (GNB / LR)
# ---------------------------------------------------------------------------

class TestGNBClassifier:
    def test_fit_predict(self, feature_data):
        X, y = feature_data
        clf = GNBClassifier()
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset(set(y))

    def test_predict_proba_shape(self, feature_data):
        X, y = feature_data
        clf = GNBClassifier()
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 2)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(y)), atol=1e-5)

    def test_get_classes(self, feature_data):
        X, y = feature_data
        clf = GNBClassifier()
        clf.fit(X, y)
        classes = clf.get_classes()
        assert set(classes) == {"classA", "classB"}

    def test_not_fitted_raises(self):
        clf = GNBClassifier()
        with pytest.raises(RuntimeError):
            clf.predict(np.zeros((5, 4)))


class TestLRClassifier:
    def test_high_accuracy_on_linearly_separable(self, feature_data):
        X, y = feature_data
        clf = LRClassifier({"max_iter": 500})
        clf.fit(X, y)
        acc = np.mean(clf.predict(X) == np.array(y))
        assert acc > 0.95


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestGetClassifier:
    @pytest.mark.parametrize(
        "type_,params",
        [
            ("GNB", {}),
            ("LR", {"max_iter": 200}),
        ],
    )
    def test_factory_creates_instances(self, type_, params, feature_data):
        X, y = feature_data
        cfg = ComponentConfig(type=type_, params=params)
        clf = get_classifier(cfg)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(y)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="desconhecido"):
            get_classifier(ComponentConfig(type="INVALID"))
