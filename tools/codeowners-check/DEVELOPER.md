# codeowners-check — 开发者工作流与迭代指南

> English: [DEVELOPER.en.md](DEVELOPER.en.md)

本目录的 `codeowners-check.py` 是 CODEOWNERS 的**本地**影响检查工具。命令用法见
[TLDR.md](TLDR.md)。本文档面向仓库开发者，覆盖三个工作流，以及"本地判定与 GitCode 线上
不一致"时的定位与迭代方法。

## 三个工作流

### 1. 改动涉及 CODEOWNERS → 本地确认 diff 符合预期

改完仓库根 `CODEOWNERS` 后、提交前，本地看影响面，确认 owner 增删只落在预期文件：

```bash
codeowners-check.py diff --before ref:master
```

- `-@旧` / `+@新` 逐文件标注 owner 增删（同向变化可折叠）
- 只动文件、不改规则时恒为 0 —— 文件集由 `--on` 单独决定（默认 index）
- `--tree` 看目录级折叠，`--flat` 逐文件不折叠

判断标准：diff 里出现的文件 == 你本意要重划的文件。出现意外路径，说明某条规则写宽了
（常见：裸 `*` 规则、未锚定模式），回到 CODEOWNERS 收窄。

### 2. owner 自查：某路径是否列了我

作为 owner，改动前确认某路径确实归你（判断之前的 CODEOWNERS 是否加错 / 漏加）：

```bash
# 单文件
codeowners-check.py owners --at ref:master bishengir/lib/Dialect/HFusion/Transforms/AutoVectorizeV2.cpp
# 自查本次碰过的所有文件里有没有你
git diff --name-only $(git merge-base master HEAD) | codeowners-check.py owners | grep '@你的handle'
```

- 输出 `path @u1 @u2`；你的 handle 不在其中 = 这条路径没覆盖到你
- `--at ref:master` 用线上版本判定，排除本地未提交编辑的干扰

### 3. 本地与线上不一致 → 迭代这个工具

发现本地判定的 owner 与 GitCode MR 页展示不一致，按以下顺序定位，再决定改 CODEOWNERS
还是改工具：

1. 先核对语义差异（见下节"语义如何编码"）——多数不一致是语义理解差异，不是 bug。
2. 确认是工具 bug 后，直接在本目录迭代：单文件、纯 stdlib + pathspec，改完无需编译。
3. 核心逻辑只在这三处：`parse_codeowners`（规则解析）、`Codeowners.owners`
   （last-match-wins 判定）、`compute_diff`（diff 计算）。
4. 无内置自测：在 python REPL 里喂文本验证，或临时造一个小 git 仓跑真实命令。

## 语义如何编码（对齐 GitCode）

工具刻意把 GitCode 的 CODEOWNERS 语义固化在代码里。本地 vs 线上不一致时，逐条对照：

| 语义 | 工具实现位置 | 备注 |
|------|-------------|------|
| last-match-wins | `Codeowners.owners` | 命中多条取最后一条 |
| `*` 不跨 `/`，`**` 跨目录 | pathspec `gitignore` factory | 已从弃用的 `gitwildmatch` 迁来 |
| 前导 `/` 根锚定 | `parse_codeowners` | 未锚定模式全树匹配并告警 |
| 尾 `/` 目录模式 | pathspec | 匹配目录下文件，不误配同名 `.cpp` |
| owner token：`@user` / `@group/sub` / `user@example.com` | `_valid_owner` | 裸词非法并告警 |
| 跳过空行 / `#` 注释 / 功能性分节 | `parse_codeowners` | 本项目分节标题是 `#` 注释 |
| 只读仓库根 `CODEOWNERS` | `load_codeowners` | `.github/CODEOWNERS` 等不查 |

## 迭代守则

- 改语义：先补一个 REPL 验证，再改 `Codeowners.owners`，最后同步本文档 + TLDR 的"匹配语义"节。
- 加子命令 / flag：同步 TLDR 用法表、场景表、退出码表，以及根 `AGENTS.md` 的索引。
- 依赖升级：guard 在 `main` 开头（`pathspec>=1.0` 探针），升级前确认目标版本 `gitignore`
  factory 仍可用。
- 提交格式：走仓库 AGENTS.md 的 `[Tools]` 模块、`Assisted-by: AI` trailer。
