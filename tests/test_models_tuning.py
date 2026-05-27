"""Tests del modulo de ajuste de hiperparametros."""
import numpy as np
import pandas as pd
import pytest

from src.models.tuning import TuningResult, extract_tree_params, tune_decision_tree


@pytest.fixture
def model_config():
    return {
        "decision_tree": {
            "max_depth": 5,
            "criterion": "gini",
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        },
        "smote": {
            "enabled": True,
            "sampling_strategy": "auto",
            "k_neighbors": 5,
        },
    }


@pytest.fixture
def numeric_features():
    return ["feat_a", "feat_b", "feat_c"]


@pytest.fixture
def datos_entrenamiento(numeric_features):
    """Dataset sintetico pequeno para tests rapidos."""
    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame({
        "feat_a": rng.normal(0, 1, n),
        "feat_b": rng.normal(5, 2, n),
        "feat_c": rng.normal(-2, 0.5, n),
    })
    # Target que depende de feat_a (para que el modelo aprenda algo)
    y = pd.Series((X["feat_a"] > 0).astype(int))
    return X, y


@pytest.fixture
def param_grid_pequeno():
    """Grilla minima para tests rapidos."""
    return {
        "classifier__max_depth": [2, 3],
        "classifier__criterion": ["gini"],
    }


# ─── Tests de extract_tree_params ─────────────────────────────────────


def test_extract_quita_prefijo_classifier():
    entrada = {
        "classifier__max_depth": 5,
        "classifier__criterion": "gini",
    }
    resultado = extract_tree_params(entrada)
    assert resultado == {"max_depth": 5, "criterion": "gini"}


def test_extract_ignora_claves_sin_prefijo():
    entrada = {
        "classifier__max_depth": 5,
        "smote__k_neighbors": 3,  # no es del classifier
    }
    resultado = extract_tree_params(entrada)
    assert "max_depth" in resultado
    assert "k_neighbors" not in resultado


def test_extract_diccionario_vacio():
    assert extract_tree_params({}) == {}


# ─── Tests de tune_decision_tree ──────────────────────────────────────


def test_tuning_retorna_tuning_result(
    datos_entrenamiento, numeric_features, model_config, param_grid_pequeno
):
    X, y = datos_entrenamiento
    resultado = tune_decision_tree(
        X, y, numeric_features, model_config, param_grid_pequeno,
        cv_folds=3,
    )
    assert isinstance(resultado, TuningResult)


def test_tuning_best_params_estan_en_la_grilla(
    datos_entrenamiento, numeric_features, model_config, param_grid_pequeno
):
    X, y = datos_entrenamiento
    resultado = tune_decision_tree(
        X, y, numeric_features, model_config, param_grid_pequeno,
        cv_folds=3,
    )
    # max_depth elegido debe estar en la grilla [2, 3]
    assert resultado.best_params["classifier__max_depth"] in [2, 3]
    assert resultado.best_params["classifier__criterion"] == "gini"


def test_tuning_cuenta_combinaciones_correctamente(
    datos_entrenamiento, numeric_features, model_config, param_grid_pequeno
):
    X, y = datos_entrenamiento
    resultado = tune_decision_tree(
        X, y, numeric_features, model_config, param_grid_pequeno,
        cv_folds=3,
    )
    # Grilla: 2 max_depth x 1 criterion = 2 combinaciones
    assert resultado.n_combinations == 2


def test_tuning_score_en_rango_valido(
    datos_entrenamiento, numeric_features, model_config, param_grid_pequeno
):
    X, y = datos_entrenamiento
    resultado = tune_decision_tree(
        X, y, numeric_features, model_config, param_grid_pequeno,
        cv_folds=3,
    )
    assert 0 <= resultado.best_score <= 1


def test_tuning_cv_results_no_vacio(
    datos_entrenamiento, numeric_features, model_config, param_grid_pequeno
):
    X, y = datos_entrenamiento
    resultado = tune_decision_tree(
        X, y, numeric_features, model_config, param_grid_pequeno,
        cv_folds=3,
    )
    assert len(resultado.cv_results) == resultado.n_combinations


def test_tuning_es_reproducible(
    datos_entrenamiento, numeric_features, model_config, param_grid_pequeno
):
    """Misma semilla -> mismos mejores parametros."""
    X, y = datos_entrenamiento
    r1 = tune_decision_tree(
        X, y, numeric_features, model_config, param_grid_pequeno,
        cv_folds=3, random_state=42,
    )
    r2 = tune_decision_tree(
        X, y, numeric_features, model_config, param_grid_pequeno,
        cv_folds=3, random_state=42,
    )
    assert r1.best_params == r2.best_params