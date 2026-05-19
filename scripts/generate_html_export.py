"""
Generador del HTML de entrega.

Recorre todos los archivos .py del proyecto, los formatea con
syntax highlighting via Pygments, y produce un HTML autocontenido
con indice navegable.

Uso:
    python scripts/generate_html_export.py

Salida:
    reports/codigo_proyecto.html
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import (
    JsonLexer,
    PythonLexer,
    TextLexer,
    YamlLexer,
    get_lexer_for_filename,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SECCIONES = [
    {
        "titulo": "1. Configuracion del proyecto",
        "descripcion": "Archivos de configuracion central, dependencias y entorno.",
        "archivos": [
            "config/config.yaml",
            "requirements.in",
            "requirements-dev.in",
            "requirements.txt",
            ".python-version",
            "pyproject.toml",
        ],
    },
    {
        "titulo": "2. Modulo de datos (src/data/)",
        "descripcion": "Carga del CSV y validacion de schema con pandera.",
        "archivos": [
            "src/data/__init__.py",
            "src/data/loader.py",
            "src/data/schema.py",
        ],
    },
    {
        "titulo": "3. Modulo de features (src/features/)",
        "descripcion": "Seleccion de variables y creacion del target binario.",
        "archivos": [
            "src/features/__init__.py",
            "src/features/selection.py",
            "src/features/target.py",
        ],
    },
    {
        "titulo": "4. Modulo de modelado (src/models/)",
        "descripcion": "Pipeline de entrenamiento con SMOTE y arbol de decision.",
        "archivos": [
            "src/models/__init__.py",
            "src/models/splitter.py",
            "src/models/pipeline.py",
            "src/models/train.py",
        ],
    },
    {
        "titulo": "5. Modulo de evaluacion (src/evaluation/)",
        "descripcion": "Calculo de metricas, generacion de graficas y reportes.",
        "archivos": [
            "src/evaluation/__init__.py",
            "src/evaluation/metrics.py",
            "src/evaluation/plots.py",
            "src/evaluation/report.py",
            "src/evaluation/evaluate.py",
        ],
    },
    {
        "titulo": "6. Utilidades (src/utils/)",
        "descripcion": "Helpers compartidos (carga de configuracion, paths).",
        "archivos": [
            "src/utils/__init__.py",
            "src/utils/config.py",
        ],
    },
    {
        "titulo": "7. Modulo prescriptivo (src/prescriptive/)",
        "descripcion": "Reglas de negocio RN-01 a RN-05 y motor de recomendaciones.",
        "archivos": [
            "src/prescriptive/__init__.py",
            "src/prescriptive/rules.py",
            "src/prescriptive/recommender.py",
            "src/prescriptive/analyze.py",
        ],
    },
    {
        "titulo": "8. Pruebas unitarias (tests/)",
        "descripcion": "Suite de pytest cubriendo todos los modulos.",
        "archivos": [
            "tests/__init__.py",
            "tests/conftest.py",
            "tests/test_data_loader.py",
            "tests/test_features_selection.py",
            "tests/test_features_target.py",
            "tests/test_models_splitter.py",
            "tests/test_models_pipeline.py",
            "tests/test_evaluation_metrics.py",
            "tests/test_prescriptive_rules.py",
            "tests/test_prescriptive_recommender.py",
        ],
    },
    {
        "titulo": "9. Documentacion adicional",
        "descripcion": "Documento de negocio, README y archivos de control de versiones.",
        "archivos": [
            "docs/business_context.md",
            "README.md",
            ".gitignore",
        ],
    },
]


def get_lexer(filepath: Path):
    """Devuelve el lexer adecuado segun la extension."""
    name = filepath.name.lower()
    if name.endswith((".yaml", ".yml")):
        return YamlLexer()
    if name.endswith(".json"):
        return JsonLexer()
    if name.endswith(".py"):
        return PythonLexer()
    try:
        return get_lexer_for_filename(filepath.name)
    except Exception:
        return TextLexer()


def count_lines(content: str) -> int:
    return sum(1 for line in content.splitlines() if line.strip())


def render_file(filepath: Path) -> dict:
    rel_path = filepath.relative_to(PROJECT_ROOT)

    if not filepath.exists():
        return {
            "path": str(rel_path).replace("\\", "/"),
            "exists": False,
            "html": "",
            "lines": 0,
            "anchor": "",
        }

    try:
        content = filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = filepath.read_text(encoding="latin-1")

    formatter = HtmlFormatter(
        style="monokai",
        linenos="inline",
        cssclass="codehilite",
    )
    lexer = get_lexer(filepath)
    highlighted = highlight(content, lexer, formatter)

    anchor = str(rel_path).replace("\\", "/").replace("/", "-").replace(".", "-")

    return {
        "path": str(rel_path).replace("\\", "/"),
        "exists": True,
        "html": highlighted,
        "lines": count_lines(content),
        "size_bytes": filepath.stat().st_size,
        "anchor": anchor,
    }


def get_project_metadata() -> dict:
    try:
        import imblearn
        import numpy
        import pandas
        import sklearn
        versions = {
            "Python": sys.version.split()[0],
            "scikit-learn": sklearn.__version__,
            "imbalanced-learn": imblearn.__version__,
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
        }
    except ImportError as e:
        versions = {"error": str(e)}

    return {
        "fecha_generacion": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "versiones": versions,
    }


def build_html(secciones_renderizadas: list, metadata: dict) -> str:
    pygments_css = HtmlFormatter(style="monokai").get_style_defs(".codehilite")

    toc_items = []
    for i, sec in enumerate(secciones_renderizadas, 1):
        archivos_validos = [f for f in sec["archivos_render"] if f["exists"]]
        if not archivos_validos:
            continue
        sec_anchor = f"seccion-{i}"
        toc_items.append(
            f'<li><a href="#{sec_anchor}"><strong>{sec["titulo"]}</strong></a>'
            f'<ul>'
            + "".join(
                f'<li><a href="#{f["anchor"]}"><code>{f["path"]}</code></a></li>'
                for f in archivos_validos
            )
            + '</ul></li>'
        )
    toc_html = "<ul class='toc'>" + "".join(toc_items) + "</ul>"

    body_parts = []
    for i, sec in enumerate(secciones_renderizadas, 1):
        archivos_validos = [f for f in sec["archivos_render"] if f["exists"]]
        if not archivos_validos:
            continue
        sec_anchor = f"seccion-{i}"
        body_parts.append(
            f'<section id="{sec_anchor}" class="seccion">'
            f'<h2>{sec["titulo"]}</h2>'
            f'<p class="descripcion">{sec["descripcion"]}</p>'
        )
        for f in archivos_validos:
            body_parts.append(
                f'<article id="{f["anchor"]}" class="archivo">'
                f'<header class="archivo-header">'
                f'<h3><code>{f["path"]}</code></h3>'
                f'<span class="meta">{f["lines"]} lineas - {f["size_bytes"]} bytes</span>'
                f'</header>'
                f'{f["html"]}'
                f'<a href="#top" class="back-to-top">Volver al indice</a>'
                f'</article>'
            )
        body_parts.append('</section>')
    body_html = "\n".join(body_parts)

    versiones_html = "".join(
        f'<tr><td><code>{k}</code></td><td>{v}</td></tr>'
        for k, v in metadata["versiones"].items()
    )

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>TIA - Student Performance - Codigo del Proyecto</title>
<style>
:root {{
    --bg: #1e1e1e;
    --bg-card: #252526;
    --bg-soft: #2d2d30;
    --text: #d4d4d4;
    --text-muted: #858585;
    --accent: #4ec9b0;
    --accent-soft: #569cd6;
    --border: #3e3e42;
}}

* {{ box-sizing: border-box; }}

body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 0;
    line-height: 1.6;
}}

.container {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}}

header.principal {{
    border-bottom: 2px solid var(--accent);
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
}}

h1 {{
    color: var(--accent);
    margin: 0 0 0.5rem 0;
    font-size: 2rem;
}}

h2 {{
    color: var(--accent);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-top: 3rem;
}}

h3 {{
    color: var(--accent-soft);
    margin: 0;
}}

h3 code {{
    background: transparent;
    color: var(--accent-soft);
    font-size: 1.1rem;
    padding: 0;
}}

.subtitulo {{
    color: var(--text-muted);
    margin-top: 0;
}}

.metadata-box {{
    background: var(--bg-card);
    border-left: 3px solid var(--accent);
    padding: 1rem 1.5rem;
    margin: 1.5rem 0;
    border-radius: 4px;
}}

.metadata-box table {{
    border-collapse: collapse;
    margin-top: 0.5rem;
}}

.metadata-box td {{
    padding: 0.25rem 1rem 0.25rem 0;
}}

.metadata-box td:first-child {{
    color: var(--text-muted);
    width: 200px;
}}

.toc {{
    background: var(--bg-card);
    padding: 1.5rem 2rem;
    border-radius: 6px;
    list-style: none;
}}

.toc ul {{
    list-style: none;
    padding-left: 1.5rem;
}}

.toc > li {{
    margin-bottom: 0.8rem;
}}

.toc a {{
    color: var(--text);
    text-decoration: none;
    transition: color 0.15s;
}}

.toc a:hover {{
    color: var(--accent);
}}

.toc code {{
    font-size: 0.9rem;
    background: transparent;
    padding: 0;
    color: inherit;
}}

.descripcion {{
    color: var(--text-muted);
    font-style: italic;
    margin-bottom: 1.5rem;
}}

.seccion {{
    margin-bottom: 3rem;
}}

.archivo {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 1.5rem;
    overflow: hidden;
}}

.archivo-header {{
    background: var(--bg-soft);
    padding: 0.75rem 1.25rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
}}

.meta {{
    color: var(--text-muted);
    font-size: 0.85rem;
    font-family: monospace;
}}

code {{
    background: var(--bg-soft);
    padding: 0.15rem 0.4rem;
    border-radius: 3px;
    font-family: "Cascadia Code", "Fira Code", Consolas, monospace;
    font-size: 0.9em;
}}

.codehilite {{
    margin: 0;
    padding: 1.25rem;
    overflow-x: auto;
    font-family: "Cascadia Code", "Fira Code", Consolas, monospace;
    font-size: 0.85rem;
    line-height: 1.5;
}}

.codehilite pre {{
    margin: 0;
}}

.linenos {{
    color: var(--text-muted);
    padding-right: 1rem;
    user-select: none;
    border-right: 1px solid var(--border);
    margin-right: 1rem;
}}

.back-to-top {{
    display: inline-block;
    padding: 0.5rem 1rem;
    margin: 0.5rem 1.25rem 1rem;
    background: var(--bg-soft);
    color: var(--text-muted);
    text-decoration: none;
    border-radius: 4px;
    font-size: 0.85rem;
    transition: all 0.15s;
}}

.back-to-top:hover {{
    background: var(--accent);
    color: var(--bg);
}}

footer {{
    margin-top: 4rem;
    padding-top: 2rem;
    border-top: 1px solid var(--border);
    color: var(--text-muted);
    text-align: center;
    font-size: 0.9rem;
}}

{pygments_css}
</style>
</head>
<body>
<a id="top"></a>
<div class="container">

<header class="principal">
    <h1>TIA - Student Performance</h1>
    <p class="subtitulo">
        Modelo predictivo de aprobacion estudiantil -
        Curso de Machine Learning - IU Pascual Bravo
    </p>
    <p class="subtitulo"><strong>Autor:</strong> Samuel Rendon Hincapie</p>
</header>

<div class="metadata-box">
    <strong>Metadata del proyecto</strong>
    <table>
        <tr><td>Fecha de generacion</td><td>{metadata["fecha_generacion"]}</td></tr>
        {versiones_html}
        <tr><td>Modelo final</td><td>DecisionTreeClassifier (max_depth=5) + SMOTE</td></tr>
    </table>
</div>

<h2>Indice del proyecto</h2>
{toc_html}

{body_html}

<footer>
    Generado automaticamente - {metadata["fecha_generacion"]}<br>
    Codigo migrado desde Google Colab a un entorno modular reproducible en VS Code.
</footer>

</div>
</body>
</html>
"""


def main():
    print(f"[INFO] Raiz del proyecto: {PROJECT_ROOT}")

    secciones_renderizadas = []
    total_archivos = 0
    archivos_faltantes = []

    for sec in SECCIONES:
        archivos_render = []
        for ruta in sec["archivos"]:
            filepath = PROJECT_ROOT / ruta
            resultado = render_file(filepath)
            archivos_render.append(resultado)
            if resultado["exists"]:
                total_archivos += 1
                print(f"  + {ruta} ({resultado['lines']} lineas)")
            else:
                archivos_faltantes.append(ruta)
                print(f"  - {ruta} (no existe, se omite)")

        secciones_renderizadas.append({
            **sec,
            "archivos_render": archivos_render,
        })

    metadata = get_project_metadata()
    html = build_html(secciones_renderizadas, metadata)

    output_dir = PROJECT_ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "codigo_proyecto.html"
    output_path.write_text(html, encoding="utf-8")

    size_kb = output_path.stat().st_size / 1024
    print(f"\n[OK] HTML generado: {output_path}")
    print(f"     Tamano: {size_kb:.1f} KB")
    print(f"     Archivos incluidos: {total_archivos}")
    if archivos_faltantes:
        print(f"     Archivos omitidos: {len(archivos_faltantes)}")


if __name__ == "__main__":
    main()
