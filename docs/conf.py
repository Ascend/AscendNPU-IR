# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os

# -- Project information -----------------------------------------------------
project = 'AscendNPU IR'
copyright = '2026, Huawei'
author = 'Huawei'

# -- I18n: detect language and root doc ---------------------------------------
_readthedocs_lang = os.environ.get('READTHEDOCS_LANGUAGE')
_is_build_by_readthedocs = _readthedocs_lang is not None

if _readthedocs_lang:
    _build_lang = _readthedocs_lang.strip().lower().replace('_', '-')
else:
    _build_lang = (os.environ.get('LANGUAGE') or 'en').strip().lower().replace('_', '-')

_is_zh = _build_lang in ('zh-cn', 'zh') or _build_lang.startswith('zh-')
language = 'zh_CN' if _is_zh else 'en'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
if _is_zh:
    exclude_patterns.extend(['source/en'])
else:
    exclude_patterns.extend(['source/zh_cn'])

# -- General configuration ---------------------------------------------------
templates_path = ['_templates']

extensions = [
    "myst_parser",
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
pygments_style = "friendly"
html_last_updated_fmt = "%b %d, %Y"

def setup(app):
    """Register Pygments lexer aliases."""
    from sphinx.highlighting import lexers
    from pygments.lexers import get_lexer_by_name
    import shutil
    lexers['mlir'] = get_lexer_by_name('text')
    lexers['plaintext'] = get_lexer_by_name('text')
 
    if not _is_build_by_readthedocs:
        app.add_js_file('lang-switcher.js')
        app.add_css_file('lang-switcher.css')
    return {'version': '0.1', 'parallel_read_safe': True}
