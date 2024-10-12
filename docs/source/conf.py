# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import datetime

# sys.path.insert(0, os.path.abspath('./Garfield'))
sys.path.insert(0, os.path.abspath('../Garfield'))
sys.path.insert(0, os.path.abspath('_ext'))

import Garfield  # noqa: E402

# -- Project information -----------------------------------------------------
project = 'Garfield'
author = 'Weige Zhou'
copyright = f'{datetime.now():%Y}, {author}.'

# The full version, including alpha/beta/rc tags
release = Garfield.__version__

# -- General configuration ---------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

needs_sphinx = "3.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "nbsphinx",
    'sphinx_copybutton',
    "edit_on_github",
    ]

# Generate the API documentation when building
autosummary_generate = True
# autodoc_member_order = 'bysource'
# autodoc_mock_imports = ['Garfield']

# Napoleon settings
napoleon_google_docstring = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "navigation_depth": 1,
    "titles_only": True,
    'logo_only': True,
}
html_context = dict(
    github_user='zhou-1314',  # Username
    github_repo='Garfield',  # Repo name
    github_version='main',  # Version
    conf_py_path='/docs/',  # Path in the checkout to the docs root
)
html_show_sphinx = False
html_logo = '_static/img/logo_garfield.png'
html_favicon = '_static/img/garfield_icon.svg'
# github_repo = 'Garfield'
# github_nb_repo = 'Garfield_tutorials'


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_static_path = ['_static']
