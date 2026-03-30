# AscendNPU IR 文档

本目录为 AscendNPU IR 的 Sphinx 文档工程，通过双语文档源文件支持**英文与中文**。

**语言:** [English](README.md) · [中文](README_zh.md)

---

## 文档方案

| 项 | 约定 |
|------|------------|
| **英文** | 入口页为 `docs/source/en/index.rst`，正文位于 `docs/source/en` 目录下 |
| **中文** | 入口页为 `docs/source/zh_cn/index.rst`，正文位于 `docs/source/zh_cn` 目录下 |
| **Read the Docs** | 主项目（英文）+ 翻译项目（中文）；在 Admin → Translations 中关联，用于语言切换。 |

### 命名规范（snake_case）

`docs/`（包含 `docs/source/`）下的**目录名**与**文档文件名**统一采用 **snake_case** 小写下划线风格，例如：`quick_start/`、`installing_guide.md`、`user_guide/`，以保持路径与 URL 风格一致。

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
open docs/_build/zh_cn/index.html

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

1. 添加**英文**文件，如 `source/en/introduction/new_doc.md`。
2. 添加**中文**文件，如 `source/zh_cn/introduction/new_doc.md`。
3. 将文档加入对应 toctree：
   - **英文**：在 `docs/source/en/index.rst` 或相应子目录的 `index.rst`（如 `source/en/introduction/quick_start/index.rst`）中加入 `new_doc`（或路径如 `section/new_doc`）。
   - **中文**：在 `docs/source/zh_cn/index.rst` 或子目录的 `index.rst`（如 `source/en/introduction/quick_start/index.rst`）中加入 `new_doc`（或 `section/new_doc`）。
4. 若需参与排序或工具处理，可在 `conf.py` 的 `_main_doc_order` 中追加该文档路径。

---

## 文档结构

| 路径 | 说明 |
|------|-------------|
| conf.py| Sphinx 配置；`language`、`root_doc` 由 `READTHEDOCS_LANGUAGE` 决定；`_main_doc_order` 用于规范顺序 |
| source/en/index.rst | 英文首页及 toctree |
| source/zh_cn/index.rst | 中文首页及 toctree |
| source/en/**/\*.md | 英文正文 |
| source/zh_cn/**/\*.md | 中文正文 |
| source/en/**/index.rst | 英文各章节 toctree |
| source/zh_cn/**/index.rst | 中文各章节 toctree |
| Makefile | `html`（英文）、`html-zh`（中文）、`html-all` |
| requirements.txt | Sphinx、myst-parser、furo |
