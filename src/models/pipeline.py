"""Construccion del pipeline de modelado."""
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def build_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    """Construye el preprocesador de columnas."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(
    numeric_features: list[str],
    model_config: dict,
    random_state: int = 42,
) -> ImbPipeline:
    """Construye el pipeline completo: preproc -> SMOTE -> DecisionTree."""
    preprocessor = build_preprocessor(numeric_features)

    tree_params = model_config["decision_tree"]
    tree = DecisionTreeClassifier(
        max_depth=tree_params["max_depth"],
        criterion=tree_params.get("criterion", "gini"),
        min_samples_split=tree_params.get("min_samples_split", 2),
        min_samples_leaf=tree_params.get("min_samples_leaf", 1),
        random_state=random_state,
    )

    smote_cfg = model_config["smote"]
    steps = [("preproc", preprocessor)]

    if smote_cfg.get("enabled", True):
        smote = SMOTE(
            sampling_strategy=smote_cfg.get("sampling_strategy", "auto"),
            k_neighbors=smote_cfg.get("k_neighbors", 5),
            random_state=random_state,
        )
        steps.append(("smote", smote))

    steps.append(("classifier", tree))

    return ImbPipeline(steps=steps)
