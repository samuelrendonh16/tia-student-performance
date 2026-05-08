"""Utilidades para cargar configuración del proyecto."""
from pathlib import Path
import yaml


def load_config(config_path: str | Path = "config/config.yaml") -> dict:
    """
    Carga el archivo de configuración YAML.

    Parameters
    ----------
    config_path : str | Path
        Ruta al archivo de configuración.

    Returns
    -------
    dict
        Diccionario con la configuración del proyecto.

    Raises
    ------
    FileNotFoundError
        Si el archivo de configuración no existe.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de configuración: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """
    Retorna la raíz del proyecto (carpeta que contiene 'src/').

    Útil para construir rutas absolutas independientes de dónde se ejecute.
    """
    # Subimos dos niveles: src/utils/config.py -> src/utils -> src -> raíz
    return Path(__file__).resolve().parent.parent.parent