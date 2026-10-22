"""Sphinx configuration for the two-timescale documentation."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "two-timescale"
author = "two-timescale contributors"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "two-timescale"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2f6f73",
        "color-brand-content": "#24585c",
        "color-api-name": "#214c50",
        "font-stack": "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
        "font-stack--monospace": "JetBrains Mono, SFMono-Regular, Consolas, Liberation Mono, monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#8fd0d4",
        "color-brand-content": "#9bd8dc",
    },
}
