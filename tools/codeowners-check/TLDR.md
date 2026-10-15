# codeowners-check.py — 用户手册

> English: [TLDR.en.md](TLDR.en.md)

前置：`pip install 'pathspec>=1.0'`。**全程本地 git**，默认不碰远端、无需 `git fetch`（只有显式传远端 ref 才碰）。

两个子命令 → 两类问题：

| 我想要的 | 命令 |
|---------|------|
| 我改 CODEOWNERS 让哪些文件 owner 变了 | `codeowners-check.py diff` |
| 这些路径归谁 | `codeowners-check.py owners <path>...` 或管道喂 stdin |

`diff` 的含义是收窄的：**在给定的文件集上，两份 CODEOWNERS 的判定差异**。
`--before` / `--after` 只选规则版本，文件集由 `--on` 单独决定（默认 `INDEX`，即 `git ls-files`，含
`git add` 了没 commit 的文件、不含 `??`）——所以只动文件不改规则时 `diff` 恒为 0。

`owners` 从 stdin 逐行读路径，所以「我碰过的文件归谁」是一条管道，不是一个子命令。

## 用法

```
codeowners-check.py diff   [--before <SRC>] [--after <SRC>] [--on <SRC>] [--flat] [--tree]
codeowners-check.py owners [--at <SRC>] [--no-resolve] [<path> ...]   # 省略 path 或传 `-` 时从 stdin 逐行读
```

- `diff` 默认：`--before ref:master` · `--after worktree` · `--on index`
- `owners` 默认：`--at worktree`
- `--flat` / `--tree` 是独立两轴，可组合：`--flat --tree` = 不折叠的完整目录树

> `SRC` 统一值域：`worktree`（磁盘文件）/ `index`（暂存，别名 `staged`）/ `ref:<name>`（git ref，
> 大小写敏感）。文件状态是裸词，ref 用 `ref:` 前缀，所以叫 `worktree`/`index` 的分支要写成
> `ref:worktree`/`ref:index`，不会被误判。

## 常见场景

| # | 场景 | 命令 | 效果 |
|---|------|------|------|
| 1 | 提交前自查：我碰了哪些文件、谁该看 | `git diff --name-only $(git merge-base master HEAD) \| codeowners-check.py owners` | 按 owner 折叠列出，无主的标 `NO OWNER` |
| 2 | 新文件还没 `git add` 也要算进来 | `{ git diff --name-only $(git merge-base master HEAD); git ls-files --others --exclude-standard; } \| codeowners-check.py owners` | `git diff` 不列 `??`，得并一路 |
| 3 | 只看未提交改动 | `git diff --name-only HEAD \| codeowners-check.py owners` | HEAD → 工作区 |
| 4 | 某个 commit 动了谁的文件 | `git show --name-only --format= HEAD \| codeowners-check.py owners` | 任意文件列表都能喂 |
| 5 | CI 拦截无主文件 | `... \| codeowners-check.py owners \| grep -q 'NO OWNER' && exit 1` | 工具本身不做 gate，自己判 |
| 6 | 改了 CODEOWNERS 看影响面 | `codeowners-check.py diff` | `-`/`+` 标注 owner 增删，同向变化的目录折叠 |
| 7 | 只看未提交改动的影响 | `codeowners-check.py diff --before ref:HEAD` | 暂存 + 未暂存，相对上次 commit |
| 7a | 只看**已 add** 的规则改动 | `codeowners-check.py diff --before ref:HEAD --after index` | HEAD → 暂存区 |
| 7b | 只看**还没 add** 的规则改动 | `codeowners-check.py diff --before index --after worktree` | 暂存区 → 磁盘 |
| 8 | 手动拉远端后看相对上游 | `codeowners-check.py diff --before ref:upstream/master` | 相对上游的完整差异（需先 `git fetch upstream`） |
| 9 | 看目录树结构 | `codeowners-check.py diff --tree` | 缩进树，owner 变化内联 `@旧 → @新` |
| 10 | 逐个文件看（不折叠） | `codeowners-check.py diff --flat` | 每个文件一行 |
| 11 | 查单个文件 | `codeowners-check.py owners bishengir/lib/Dialect/HFusion/Transforms/AutoVectorizeV2.cpp` | `path @u1 @u2` 或 `NO OWNER` |
| 12 | 查目录的 owner 分布 | `codeowners-check.py owners bishengir/lib/Dialect/HFusion/` | 枚举子文件，同 owner 折叠、不同 owner 分列 |
| 13 | 让 diff 算上未跟踪（`??`）新文件 | `codeowners-check.py diff --on worktree` | 文件集 = tracked + untracked |
| 14 | 在上游文件树上评估 | `codeowners-check.py diff --on ref:origin/master` | 需先 `git fetch origin` |
| 15 | 查某 commit/branch 上文件的归属（不含本地编辑） | `codeowners-check.py owners --at ref:master <path>` | `--at` 与 `--before`/`--after` 同值域 |
| 16 | 查 MR 改动文件里谁该 review（含新增/删除，文件无需已在本地） | `git diff --name-only $(git merge-base master HEAD) \| codeowners-check.py owners --no-resolve` | 按 pattern 报 owner，缺路径不再报 `(no matching files)` |

## 匹配语义（对齐 GitCode）

- **last-match-wins**：一个文件命中多条规则，取**最后一条**的 owner（同 GitLab / GitCode）。
- 支持根锚定 `/`、`**`（跨目录）、`*`（不跨 `/`）、尾 `/` 目录模式（匹配目录下文件，不误配同名 `.cpp`）。
- owner 支持 `@user`、`@group/sub`、`user@example.com`；裸词（无 `@`）非法并告警。
- **未锚定模式告警**：缺前导 `/` 的模式在 gitignore 下全树匹配，可能偏离 GitCode——保留但打 warning，建议统一加 `/`。
- 跳过空行、`#` 注释、功能性 `[Section]` / `^[Section]` 分节；本项目分节标题是 `#` 注释。

## 容易踩的坑

| 坑 | 说明 |
|----|------|
| 管道用错 diff 范围 | `git diff --name-only master`（**两点**）在 master 前进后把别人的改动算成你的；`master...HEAD`（**三点**）只含已提交、漏未提交。用 `$(git merge-base master HEAD)` |
| `owner changes: N` 当成行数 | N 是**变化的文件数**；一个 A→B 算 1 个文件、打 `-`/`+` 两行 |

## 边界行为（契约）

| 情形 | 行为 |
|------|------|
| 不带子命令运行 | 打印 help |
| 自测 | 无内置自测：工具纯逻辑见 `parse_codeowners` / `Codeowners.owners`，可直接在 python REPL 里喂文本验证 |
| 来源不存在（`--before`/`--after`/`--on`/`--at`） | **提前**校验：exit 2，报错里列出合法来源（`worktree` / `index` / `ref:<name>`） |
| 某 ref 上没有 CODEOWNERS | 不报错，规则集为空 → 该状态下所有文件无主 |
| 只读仓库根 `CODEOWNERS` | `.github/CODEOWNERS` / `docs/CODEOWNERS` 不查，放别处会误判全无主 |
| 规则行有非法 owner token（裸词、无 `@`） | stderr 告警；还有合法 owner 就保留，全非法则跳过该行 |
| 未锚定模式（缺前导 `/`） | stderr 告警，规则仍生效 |
| `owners` 空 stdin（无 path 且管道为空） | exit 2「empty input on stdin」，**不再**枚举全仓库 |
| `owners /` / `owners .`（查仓库根） | 归一化为全部文件，等价于列全量 |
| `owners` 路径查不到（打错 / 空目录 / 该来源下不存在） | `path  (no matching files)` |
| `owners --no-resolve` 路径不存在 | 跳过文件解析，仍按 pattern 报 owner（无规则 → `NO OWNER`），用于 MR 改动文件列表 |
| `owners` 的文件集跟随 `--at` | `--at worktree` 含未跟踪文件；`--at index` 或 ref 只含该状态下已跟踪的 |
| `owners` 加 `--flat` / `--tree` | 未注册这些 flag，argparse 报错 |
| 在子目录运行 | 自动切到仓库根；**所有路径按仓库根相对解析** |

## 退出码

| 码 | 含义 |
|----|------|
| 0 | 正常（含无子命令、`owners` 查不到路径、`diff` 无变化） |
| 2 | 参数错误、ref 不存在、缺 `pathspec` 依赖、不在 git 仓库内、或 git 子命令失败 |
