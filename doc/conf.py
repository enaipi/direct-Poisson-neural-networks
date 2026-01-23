# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Direct Poisson Neural Networks'
copyright = '2023, Martin Šípka, Michal Pavelka, Oğul Esen, and Miroslav Grmela'
author = 'Martin Šípka, Michal Pavelka, Oğul Esen, and Miroslav Grmela'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # ...
    'sphinx.ext.githubpages',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage', 
    'sphinx.ext.napoleon',
    'sphinxarg.ext',
]

# Document all members (methods, classes, functions) by default
autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autodoc_member_order = 'bysource'



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

import os
import sys

# Add the path to your Python code directory to sys.path
sys.path.insert(0, os.path.abspath('..'))
print(sys.path)

