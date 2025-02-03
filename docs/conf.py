# docs/conf.py

import os
import sys

# Add the repository root and the 'projektor' folder to the path for autodoc.
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../projektor'))
sys.path.insert(0, os.path.abspath('../otdd'))

# -- Project information -----------------------------------------------------

project = 'Projektor Generalized'
author = 'Your Name'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',      # Automatically document modules.
    'sphinx.ext.napoleon',     # Support for NumPy and Google style docstrings.
    'sphinx.ext.viewcode',     # Add links to highlighted source code.
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'
html_static_path = ['_static']

