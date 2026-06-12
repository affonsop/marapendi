# Configuration file for the Sphinx documentation builder.
# See https://www.sphinx-doc.org/en/master/usage/configuration.html

import logging
import os
import sys

# -- Path setup ----------------------------------------------------------------
# Point Sphinx at the src/ layout so `import marapendi` resolves.
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -------------------------------------------------------
project = "marapendi"
copyright = "2024, Pedro Affonso Nobrega"
author = "Pedro Affonso Nobrega"
release = "0.1.0"

# -- General configuration -----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Automatically generate stub pages for autosummary directives.
autosummary_generate = True

# -- autodoc options -----------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# The top-level marapendi/__init__.py re-exports every submodule via star
# imports.  When Sphinx imports one submodule it ends up importing the whole
# package, which triggers "duplicate object description" warnings for every
# attribute that appears under both the subpackage page and the parent package
# namespace.  These are harmless — the filter below silences them so that
# `-W` builds stay clean without modifying any source file.


def _filter_duplicate_object_warnings() -> None:
    """Install a log filter that drops Sphinx duplicate-object warnings.

    The warning is emitted by ``sphinx.domains.python`` without a structured
    type tag, so it cannot be suppressed via ``suppress_warnings``.  Instead we
    attach a ``logging.Filter`` to the ``sphinx`` logger that drops the specific
    message text before it reaches the warning handler.
    """

    class _DupFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
            msg = record.getMessage()
            return "duplicate object description" not in msg

    logging.getLogger("sphinx").addFilter(_DupFilter())

# -- Napoleon (NumPy-style docstrings) -----------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- HTML output ---------------------------------------------------------------
# Try sphinx_rtd_theme; fall back gracefully to alabaster.
try:
    import sphinx_rtd_theme  # noqa: F401

    html_theme = "sphinx_rtd_theme"
except ImportError:
    html_theme = "alabaster"

html_static_path = ["_static"]


# -- Setup hook ----------------------------------------------------------------

def setup(app) -> None:  # noqa: ANN001
    """Install log filter to suppress benign duplicate-object warnings."""
    _filter_duplicate_object_warnings()
