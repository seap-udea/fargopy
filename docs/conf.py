# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath(".."))  # Para que encuentre el paquete fargopy

# -- Project information -----------------------------------------------------
project = "FARGOpy"
copyright = "2025, Jorge Zuluaga, Alejandro Murillo-González, Matias Montesinos"
author = "Jorge Zuluaga, Alejandro Murillo-González, Matias Montesinos"
release = "0.4.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",     # Lee docstrings automáticamente
    "sphinx.ext.napoleon",    # Soporta docstrings estilo Google y NumPy
    "sphinx.ext.viewcode",    # Agrega enlaces al código fuente
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "en"  # O "es" si prefieres en español

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Autodoc settings --------------------------------------------------------
autodoc_default_options = {
    "members": True,            # Documenta miembros públicos
    "undoc-members": False,     # No muestra miembros sin docstring
    "private-members": False,   # Oculta métodos privados (_internos)
    "special-members": "__init__",  # Incluye constructores
    "inherited-members": False,
    "show-inheritance": True,
}

