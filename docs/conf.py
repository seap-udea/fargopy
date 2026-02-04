# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
import fargopy

project = "fargopy"
copyright = "2025, Jorge I. Zuluaga, Alejandro Murillo-González, Matias Montesinos"
author = "Jorge I. Zuluaga, Alejandro Murillo-González, Matias Montesinos"
release = fargopy.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_rtd_theme",
    "sphinx_mdinclude",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "../gallery/fargopy_logo_white.png"
html_theme_options = {
    "logo_only": True,
    "display_version": True,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}
