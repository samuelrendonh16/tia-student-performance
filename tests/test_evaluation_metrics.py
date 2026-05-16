"""Tests del módulo de métricas."""
import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.evaluation.metrics import (
    compute_basic_metrics,
    compute_classification_report_dict,
    compute_confusion_matrix,
    compute_feature_importances,
    compute_roc_metrics,
)


@pytest.fixture
def y_true_pred():
    """Etiquetas con un caso conocido para verificar números."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1])
    return y_true, y_pred


@pytest.fixture
def y_true_proba():
    """y_true y probabilidades para tests de ROC."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 0.4, 0.3, 0.85])
    return y_true, y_proba


# ─── Tests de compute_basic_metrics ───────────────────────────────────


def test_metricas_basicas_devuelve_diccionario(y_true_pred):
    y_true, y_pred = y_true_pred
    m = compute_basic_metrics(y_true, y_pred)
    assert set(m.keys()) == {"accuracy", "precision", "recall", "f1"}


def test_metricas_basicas_valores_en_rango_valido(y_true_pred):
    y_true, y_pred = y_true_pred
    m = compute_basic_metrics(y_true, y_pred)
    for valor in m.values():
        assert 0 <= valor <= 1


def test_accuracy_perfecto_da_1():
    y = np.array([0, 1, 0, 1])
    m = compute_basic_metrics(y, y.copy())
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_accuracy_calculado_correctamente(y_true_pred):
    """y_true_pred tiene 7 aciertos de 10 → accuracy = 0.7"""
    y_true, y_pred = y_true_pred
    m = compute_basic_metrics(y_true, y_pred)
    assert abs(m["accuracy"] - 0.7) < 1e-9


# ─── Tests de compute_confusion_matrix ────────────────────────────────


def test_matriz_confusion_devuelve_estructura_correcta(y_true_pred):
    y_true, y_pred = y_true_pred
    cm = compute_confusion_matrix(y_true, y_pred)
    assert "matrix" in cm
    assert "tn" in cm and "fp" in cm and "fn" in cm and "tp" in cm


def test_matriz_confusion_suma_total(y_true_pred):
    """tn + fp + fn + tp debe ser igual al total de muestras."""
    y_true, y_pred = y_true_pred
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(y_true)


def test_matriz_confusion_valores_correctos(y_true_pred):
    """
    y_true: [0,0,0,0,1,1,1,1,1,1]
    y_pred: [0,0,0,1,1,1,1,0,0,1]
    TN=3 (0→0), FP=1 (0→1), FN=2 (1→0), TP=4 (1→1)
    """
    y_true, y_pred = y_true_pred
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm["tn"] == 3
    assert cm["fp"] == 1
    assert cm["fn"] == 2
    assert cm["tp"] == 4


# ─── Tests de compute_roc_metrics ─────────────────────────────────────


def test_roc_auc_en_rango_valido(y_true_proba):
    y_true, y_proba = y_true_proba
    roc = compute_roc_metrics(y_true, y_proba)
    assert 0 <= roc["auc"] <= 1


def test_roc_separacion_perfecta_da_auc_1():
    """Si las probabilidades separan perfectamente las clases, AUC = 1."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    roc = compute_roc_metrics(y_true, y_proba)
    assert roc["auc"] == 1.0


def test_roc_estructura_fpr_tpr(y_true_proba):
    """fpr y tpr deben tener la misma longitud."""
    y_true, y_proba = y_true_proba
    roc = compute_roc_metrics(y_true, y_proba)
    assert len(roc["fpr"]) == len(roc["tpr"])


# ─── Tests de compute_classification_report_dict ──────────────────────


def test_classification_report_es_diccionario(y_true_pred):
    y_true, y_pred = y_true_pred
    rep = compute_classification_report_dict(
        y_true, y_pred, class_names=["Neg", "Pos"]
    )
    assert isinstance(rep, dict)
    assert "Neg" in rep
    assert "Pos" in rep
    assert "accuracy" in rep


# ─── Tests de compute_feature_importances ─────────────────────────────


def test_importancias_ordenadas_descendente():
    """Entrenamos un árbol pequeño y verificamos el orden."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "a": rng.normal(0, 1, 200),
        "b": rng.normal(0, 1, 200),
        "c": rng.normal(0, 1, 200),
    })
    # b es la feature que decide y
    y = (X["b"] > 0).astype(int)

    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X, y)

    # Simular un pipeline con el tree como "classifier"
    class FakePipeline:
        named_steps = {"classifier": tree}

    importances = compute_feature_importances(
        FakePipeline(), ["a", "b", "c"]
    )

    # Verificar orden descendente
    valores = [d["importance"] for d in importances]
    assert valores == sorted(valores, reverse=True)
    # b debería ser la más importante
    assert importances[0]["feature"] == "b"


def test_importancias_suman_1_o_menos():
    """feature_importances_ de sklearn suma 1."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "a": rng.normal(0, 1, 100),
        "b": rng.normal(0, 1, 100),
    })
    y = (X["a"] > 0).astype(int)

    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X, y)

    class FakePipeline:
        named_steps = {"classifier": tree}

    importances = compute_feature_importances(FakePipeline(), ["a", "b"])
    total = sum(d["importance"] for d in importances)
    assert abs(total - 1.0) < 1e-9


def test_importancias_top_n_limita_resultado():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "a": rng.normal(0, 1, 100),
        "b": rng.normal(0, 1, 100),
        "c": rng.normal(0, 1, 100),
    })
    y = (X["a"] > 0).astype(int)
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X, y)

    class FakePipeline:
        named_steps = {"classifier": tree}

    top = compute_feature_importances(FakePipeline(), ["a", "b", "c"], top_n=2)
    assert len(top) == 2


def test_importancias_falla_con_estimador_sin_importances():
    """Algunos estimadores (ej. SVC con kernel rbf) no exponen importances."""
    from sklearn.svm import SVC

    class FakePipeline:
        named_steps = {"classifier": SVC()}

    with pytest.raises(AttributeError, match="feature_importances_"):
        compute_feature_importances(FakePipeline(), ["a", "b"])