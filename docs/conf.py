# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
html_theme = "furo"

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MicroFrame"
copyright = "2023, Cameron Olson"
author = "Cameron Olson"
release = "0.1.0"

html_title = f"{project} {release} Documentation"
html_favicon = "_static/Logo.png"
html_logo = "_static/Logo.png"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", 'm2r2']
source_suffix = ".rst"


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_static_path = ["_static"]

sys.path.insert(0, os.path.abspath("."))


def setup(app):
    from partialmdinclude import setup as partialmdinclude_setup
    partialmdinclude_setup(app)
