# Configuration file for the Sphinx documentation builder.
# See https://www.sphinx-doc.org/en/master/usage/configuration.html

import logging
import os
import sys

# -- Path setup ----------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -------------------------------------------------------
project = "marapendi"
copyright = "2024-2026, Pedro Affonso Nobrega"
author = "Pedro Affonso Nobrega"
release = "0.5.0"

# -- General configuration -----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True

# -- Intersphinx ---------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy":  ("https://numpy.org/doc/stable", None),
    "scipy":  ("https://docs.scipy.org/doc/scipy", None),
}

# -- autodoc options -----------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"

# -- Napoleon (NumPy-style docstrings) -----------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- copybutton ----------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- HTML output ---------------------------------------------------------------
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {
        "image_light": "_static/marapendi-full-bleu.png",
        "image_dark":  "_static/marapendi-full-bleu.png",
        "alt_text": "marapendi",
    },
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc"],
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "pygments_light_style": "tango",
    "footer_start": ["copyright"],
    "footer_end": [],
}

# Force light mode via the Jinja2 template variable used by the theme layout.
html_context = {
    "default_mode": "light",
}

# -- Sphinx-Gallery ------------------------------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": "../examples",               # location of gallery scripts
    "gallery_dirs": "auto_examples",            # generated HTML output
    "filename_pattern": r"/plot_",              # only files named plot_*.py
    "within_subsection_order": "FileNameSortKey",
    "download_all_examples": True,
    "remove_config_comments": True,
    "plot_gallery": True,
    "thumbnail_size": (400, 280),
    "image_scrapers": ("matplotlib",),
    "reset_modules": ("matplotlib",),
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Duplicate-object warning filter -------------------------------------------

def _filter_duplicate_object_warnings() -> None:
    """Suppress benign duplicate-object warnings from star-import re-exports."""

    class _DupFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "duplicate object description" not in record.getMessage()

    logging.getLogger("sphinx").addFilter(_DupFilter())


def setup(app) -> None:
    _filter_duplicate_object_warnings()
