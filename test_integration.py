from src.data.loader import load_raw_data
from src.features.selection import drop_irrelevant_columns
from src.features.target import compute_threshold, binarize_score
from src.utils.config import load_config

cfg = load_config()

# 1. Cargar datos crudos validados
df = load_raw_data(cfg["paths"]["data_raw"])
print(f"Crudo: {df.shape}")

# 2. Eliminar columnas irrelevantes
df = drop_irrelevant_columns(df, cfg["features"]["drop_columns"])
print(f"Sin columnas irrelevantes: {df.shape}")

# 3. Calcular umbral y crear target
umbral = compute_threshold(
    df[cfg["target"]["source"]],
    strategy=cfg["target"]["strategy"],
)
df = binarize_score(
    df,
    source_column=cfg["target"]["source"],
    target_column=cfg["target"]["name"],
    threshold=umbral,
    drop_source=True,
)
print(f"Final: {df.shape}")
print(f"Balance del target: {df['aprobado'].value_counts(normalize=True)}")