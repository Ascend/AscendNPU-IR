# AscendNPU IR Documentation

This directory is the Sphinx documentation project for AscendNPU IR. It supports **English and Chinese** via dual source files.

**Language:** [English](README.md) · [Chinese](README_zh.md)

## Documentation scheme

| Item | Convention |
|------|------------|
| **English** | The entry page is `docs/source/en/index.rst`; content under `docs/source/en` directory |
| **Chinese** | The entry page is `docs/source/zh_cn/index.rst`; content under `docs/source/zh_ch` directory. |
| **Read the Docs** | Main project (en) + Translation project (zh); link in Admin → Translations for language switcher. |

### Naming convention (snake_case)

Directory names and document file names under `docs/` (including `docs/source/`) adopt a unified lowercase **snake_case** style, for example: `quick_start/`, `installing_guide.md`, `user_guide/`，to keep consistent paths and URLs.

## How to build

From the **repository root**:

```bash
make -C docs html      # English only → docs/_build/en
make -C docs html-zh   # Chinese only → docs/_build/zh_cn
make -C docs html-all  # Both
```

Or from `docs/`:

```bash
make html
make html-zh
make html-all
```

## Local preview

```bash
# English
open docs/_build/en/index.html

# Chinese
open docs/_build/zh_cn/index_zh.html

# Or serve with HTTP (e.g. port 8080 for English, 8081 for Chinese)
cd docs/_build/en && python3 -m http.server 8080
cd docs/_build/zh_cn && python3 -m http.server 8081
```

## Deploy on Read the Docs

1. Import the repository on [Read the Docs](https://readthedocs.org/).
2. **Main project (English)**  
   - In **Admin → Environment variables**, add `READTHEDOCS_LANGUAGE` = `en`.  
   - Builds will use `index.rst` and `.md`; URL like `https://<slug>.readthedocs.io/en/latest/`.
3. **Translation project (Chinese)**  
   - Create a **new** RTD project for the same repo.  
   - Set **Language** to Chinese (Simplified).  
   - Add `READTHEDOCS_LANGUAGE` = `zh_CN` (or leave unset; default is zh_CN).  
   - In the **main** project, go to **Admin → Translations** and add this project.  
   - URL like `https://<slug>.readthedocs.io/zh_CN/latest/`.
4. The Translations flyout in the RTD theme will let users switch between English and Chinese.

## Adding a new document

1. Add the **English** file, e.g. `source/en/introduction/new_doc.md`.
2. Add the **Chinese** file, e.g. `source/zh_cn/introduction/new_doc.md`.
3. Add the doc to the right toctree:
   - In **English**: `docs/source/en/index.rst` or the relevant subdir `index.rst`(e.g. `source/en/introduction/quick_start/index.rst`) with entry `new_doc` (or path like `section/new_doc`).
   - In **Chinese**: `docs/source/zh_cn/index.rst` or the subdir `index.rst`(e.g. `source/zh_cn/introduction/quick_start/index.rst`) with entry `new_doc` (or `section/new_doc`).
4. Optionally append the doc to `_main_doc_order` in `conf.py` if you use it for ordering or tooling.

## Doc layout

| Path | Description |
|------|-------------|
| conf.py | Sphinx config; `language` and `root_doc` from `READTHEDOCS_LANGUAGE`; `_main_doc_order` for canonical order |
| source/en/index.rst | English home and toctrees |
| source/zh_cn/index.rst | Chinese home and toctrees |
| source/en/**/\*.md | English content |
| source/zh_cn/**/\*.md | Chinese content |
| source/en/**/index.rst| English section toctrees |
| source/zh_cn/**/index.rst | Chinese section toctrees |
| Makefile | `html` (en), `html-zh` (zh), `html-all` |
| requirements.txt | Sphinx, myst-parser, furo |
