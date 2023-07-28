# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "Neural Circuit Policies"
copyright = "2023, Mathias Lechner"
author = "Mathias Lechner"

# The short X.Y version
version = ""
# The full version, including alpha/beta/rc tags
release = "0.0.1"

html_favicon = "img/ncp_32.ico"
needs_sphinx = "4.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    # "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    # "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"
# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "friendly"
autodoc_typehints = "description"
autoclass_content = "init"
autodoc_inherit_docstrings = False
autodoc_default_options = {
    # "members": True,
    "undoc-members": False,
    "member-order": "bysource",
    "show-inheritance": False,
}
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_logo = "img/banner.png"
# text_fonts = "FreightSans, Helvetica Neue, Helvetica, Arial, sans-serif"
html_theme_options = {
    "sidebar_hide_name": True,
    # "page_width": "1140px",
    # "fixed_sidebar": "true",
    # "logo": "banner_64.png",
    # "description_font_style": "Quicksand",
    # "font_family": text_fonts,
    # "caption_font_family": "Quicksand",
    # "head_font_family": "Quicksand",
    # "sidebar_collapse": True,
    # "sidebar_includehidden": False,
}