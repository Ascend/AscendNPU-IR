# Configuration file for the Sphinx documentation builder.
# i18n: .md = English, *_zh.md = Chinese.
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
root_doc = 'index_zh' if _is_zh else 'index'

# Tblgen-generated docs (dialects, passes) have only .md (no _zh.md); both en and zh builds use the same file.

# -- Auto-scan sources/ for .md and *_zh.md ---------
_docs_dir = os.path.dirname(os.path.abspath(__file__))
_sources_dir = os.path.join(_docs_dir, 'sources')

def _scan_sources_md():
    """Scan sources/ recursively. Return (has_en, has_zh) sets of doc names (path without suffix)."""
    has_en = set()
    has_zh = set()
    for root, _dirs, files in os.walk(_sources_dir):
        for f in files:
            if not f.endswith('.md'):
                continue
            rel_root = os.path.relpath(root, _docs_dir)
            rel_path = os.path.join(rel_root, f).replace('\\', '/')
            if f.endswith('_zh.md'):
                base = rel_path[:-6]  # strip _zh.md
                has_zh.add(base)
            else:
                base = rel_path[:-3]  # strip .md
                has_en.add(base)
    return has_en, has_zh

_has_en, _has_zh = _scan_sources_md()
_both = _has_en & _has_zh

# Canonical order (add new docs here; scanned docs not in list can be appended for ordering).
_main_doc_order = [
    'sources/introduction/introduction',
    'sources/introduction/quick_start/index',
    'sources/introduction/programming_model',
    'sources/introduction/architecture',
    'sources/user_guide/compile_option',
    'sources/user_guide/debug_option',
    'sources/user_guide/best_practice',
    'sources/developer_guide/conversion/index',
    'sources/developer_guide/dialects/index',
    'sources/developer_guide/passes/index',
    'sources/developer_guide/features/index',
    'sources/contributing_guide/contribute',
    'sources/user_of_npuir/users',
    'sources/faq/faq',
    'sources/reference/thanks',
    'sources/reference/talk_and_course',
]

# When both .md and *_zh.md exist, exclude the other language so each build sees one version.
# Also exclude the other language's index to avoid "unknown document" refs.
_common_excludes = ['_build', 'Thumbs.db', '.DS_Store']
if _is_zh:
    exclude_patterns = _common_excludes + [d + '.md' for d in _both] + ['index.rst']
else:
    exclude_patterns = _common_excludes + [d + '_zh.md' for d in _both] + ['index_zh.rst']

# -- General configuration ---------------------------------------------------
extensions = []

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

# html_theme = 'alabaster'
# html_theme = 'pydata_sphinx_theme'
# html_theme = 'shibuya'
# html_theme = 'nvidia_sphinx_theme'
# html_theme = 'furo'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# 加载自定义CSS文件
html_css_files = ['custom.css']

def setup(app):
    """Register Pygments lexer aliases; for zh build copy index_zh.html to index.html (RTD + local)."""
    from sphinx.highlighting import lexers
    from pygments.lexers import get_lexer_by_name
    import shutil
    lexers['mlir'] = get_lexer_by_name('text')
    lexers['plaintext'] = get_lexer_by_name('text')

    def _copy_index_zh_to_index(app, exception):
        if exception or not _is_zh:
            return
        out = getattr(app, 'outdir', None)
        if not out:
            return
        src = os.path.join(out, 'index_zh.html')
        dst = os.path.join(out, 'index.html')
        try:
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        except OSError:
            pass

    app.connect('build-finished', _copy_index_zh_to_index)
    if not _is_build_by_readthedocs:
        app.add_js_file('lang-switcher.js')
        app.add_css_file('lang-switcher.css')
    return {'version': '0.1', 'parallel_read_safe': True}
