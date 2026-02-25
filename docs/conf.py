# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AscendNPU IR'
copyright = '2026, Huawei'
author = 'Huawei'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

extensions = [
    "myst_parser",
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_theme = 'pydata_sphinx_theme'
# html_theme = 'shibuya'
# html_theme = 'nvidia_sphinx_theme'
# html_theme = 'furo'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


def setup(app):
    """Register Pygments lexer aliases for MLIR and plaintext to avoid unknown-lexer warnings."""
    from sphinx.highlighting import lexers
    from pygments.lexers import get_lexer_by_name
    lexers['mlir'] = get_lexer_by_name('text')
    lexers['plaintext'] = get_lexer_by_name('text')
