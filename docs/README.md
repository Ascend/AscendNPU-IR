# AscendNPU IR Documentation

This directory is the Sphinx documentation project for AscendNPU IR. It supports **English and Chinese** via dual source files.

**Language / 语言:** [English](README.md) · [中文](README_zh.md)

---

## Documentation scheme

| Item | Convention |
|------|------------|
| **English** | Default `.md` files; entry page `index.rst`. |
| **Chinese** | `*_zh.md` files; entry page `index_zh.rst`. |
| **Read the Docs** | Main project (en) + Translation project (zh); link in Admin → Translations for language switcher. |

- **English**: `index.rst` and all `sources/**/*.md` (e.g. `introduction.md`, `installing_guide.md`).
- **Chinese**: `index_zh.rst` and all `sources/**/*_zh.md` (e.g. `introduction_zh.md`). Subdirs have `index_zh.rst` where they have a toctree.

### Naming convention (snake_case)

Directory names and document file names under `docs/` (including `sources/`) should use **snake_case** by default. Examples: `quick_start/`, `installing_guide.md`, `installing_guide_zh.md`, `user_guide/`, `developer_guide/`, `contributing_guide/`. This keeps URLs and paths consistent and avoids mixed styles (e.g. no camelCase like `installingGuide.md`).

---

## How to build

From the **repository root**:

```bash
make -C docs html      # English only → docs/_build/html
make -C docs html-zh   # Chinese only → docs/_build/html_zh
make -C docs html-all  # Both
```

Or from `docs/`:

```bash
make html
make html-zh
make html-all
```

---

## Local preview

```bash
# English
open docs/_build/html/index.html

# Chinese
open docs/_build/html_zh/index_zh.html

# Or serve with HTTP (e.g. port 8080 for English, 8081 for Chinese)
cd docs/_build/html && python3 -m http.server 8080
cd docs/_build/html_zh && python3 -m http.server 8081
```

---

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

---

## Adding a new document

1. Add the **English** file, e.g. `sources/introduction/NewDoc.md`.
2. Add the **Chinese** file, e.g. `sources/introduction/NewDoc_zh.md`.
3. Add the doc to the right toctree:
   - In **English**: `docs/index.rst` or the relevant subdir `index.rst` (e.g. `sources/introduction/quick_start/index.rst`) with entry `NewDoc` (or path like `section/NewDoc`).
   - In **Chinese**: `docs/index_zh.rst` or the subdir `index_zh.rst` with entry `NewDoc_zh` (or `section/NewDoc_zh`).
4. Optionally append the doc to `_main_doc_order` in `conf.py` if you use it for ordering or tooling.

---

## Doc layout

| Path | Description |
|------|-------------|
| **conf.py** | Sphinx config; `language` and `root_doc` from `READTHEDOCS_LANGUAGE`; `_main_doc_order` for canonical order |
| **index.rst** / **index_zh.rst** | English / Chinese home and toctrees |
| **sources/**/*.md** | English content |
| **sources/**/*_zh.md** | Chinese content |
| **sources/**/index.rst** | English section toctrees |
| **sources/**/index_zh.rst** | Chinese section toctrees (quick_start, conversion, dialects, passes, features) |
| **Makefile** | `html` (en), `html-zh` (zh), `html-all` |
| **requirements.txt** | Sphinx, myst-parser, sphinx-rtd-theme |
