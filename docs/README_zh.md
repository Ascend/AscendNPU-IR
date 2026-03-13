# AscendNPU IR 文档

本目录为 AscendNPU IR 的 Sphinx 文档工程，通过双语文档源文件支持**英文与中文**。

**Language / 语言:** [English](README.md) · [中文](README_zh.md)

---

## 文档方案

| 项 | 约定 |
|------|------------|
| **英文** | 默认 `.md` 文件；入口页 `index.rst`。 |
| **中文** | `*_zh.md` 文件；入口页 `index_zh.rst`。 |
| **Read the Docs** | 主项目（英文）+ 翻译项目（中文）；在 Admin → Translations 中关联，用于语言切换。 |

- **英文**：`index.rst` 及所有 `sources/**/*.md`（如 `introduction.md`、`installing_guide.md`）。
- **中文**：`index_zh.rst` 及所有 `sources/**/*_zh.md`（如 `introduction_zh.md`）。含 toctree 的子目录中配有 `index_zh.rst`。

### 命名规范（snake_case）

`docs/` 下（含 `sources/`）的**目录名**与**文档文件名**默认使用 **snake_case**（单词下划线）。示例：`quick_start/`、`installing_guide.md`、`installing_guide_zh.md`、`user_guide/`、`developer_guide/`、`contributing_guide/`。便于路径与 URL 统一，避免与 camelCase 等混用。

---

## 如何构建

在**仓库根目录**下执行：

```bash
make -C docs html      # 仅英文 → docs/_build/en
make -C docs html-zh   # 仅中文 → docs/_build/zh_cn
make -C docs html-all  # 中英文均构建
```

或在 `docs/` 目录下执行：

```bash
make html
make html-zh
make html-all
```

---

## 本地预览

```bash
# 英文
open docs/_build/en/index.html

# 中文
open docs/_build/zh_cn/index_zh.html

# 或用 HTTP 服务（例如英文 8080 端口，中文 8081 端口）
cd docs/_build/en && python3 -m http.server 8080
cd docs/_build/zh_cn && python3 -m http.server 8081
```

---

## 在 Read the Docs 上部署

1. 在 [Read the Docs](https://readthedocs.org/) 上导入本仓库。
2. **主项目（英文）**  
   - 在 **Admin → Environment variables** 中添加 `READTHEDOCS_LANGUAGE` = `en`。  
   - 构建将使用 `index.rst` 与 `.md`；URL 形如 `https://<slug>.readthedocs.io/en/latest/`。
3. **翻译项目（中文）**  
   - 为同一仓库**新建**一个 RTD 项目。  
   - 将 **Language** 设为 Chinese (Simplified)。  
   - 添加 `READTHEDOCS_LANGUAGE` = `zh_CN`（也可不设置，默认为 zh_CN）。  
   - 在**主项目**中进入 **Admin → Translations**，添加该翻译项目。  
   - URL 形如 `https://<slug>.readthedocs.io/zh_CN/latest/`。
4. RTD 主题中的 Translations 下拉可让读者在英文与中文之间切换。

---

## 添加新文档

1. 添加**英文**文件，如 `sources/introduction/NewDoc.md`。
2. 添加**中文**文件，如 `sources/introduction/NewDoc_zh.md`。
3. 将文档加入对应 toctree：
   - **英文**：在 `docs/index.rst` 或相应子目录的 `index.rst`（如 `sources/introduction/quick_start/index.rst`）中加入 `NewDoc`（或路径如 `section/NewDoc`）。
   - **中文**：在 `docs/index_zh.rst` 或子目录的 `index_zh.rst` 中加入 `NewDoc_zh`（或 `section/NewDoc_zh`）。
4. 若需参与排序或工具处理，可在 `conf.py` 的 `_main_doc_order` 中追加该文档路径。

---

## 文档结构

| 路径 | 说明 |
|------|-------------|
| **conf.py** | Sphinx 配置；`language`、`root_doc` 由 `READTHEDOCS_LANGUAGE` 决定；`_main_doc_order` 用于规范顺序 |
| **index.rst** / **index_zh.rst** | 英文 / 中文首页及 toctree |
| **sources/**/*.md** | 英文正文 |
| **sources/**/*_zh.md** | 中文正文 |
| **sources/**/index.rst** | 英文各章节 toctree |
| **sources/**/index_zh.rst** | 中文各章节 toctree（quick_start、conversion、dialects、passes、features） |
| **Makefile** | `html`（英文）、`html-zh`（中文）、`html-all` |
| **requirements.txt** | Sphinx、myst-parser、sphinx-rtd-theme |
