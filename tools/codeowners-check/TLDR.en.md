# codeowners-check.py — User Manual

> 中文手册见 [TLDR.md](TLDR.md)

Prerequisite: `pip install 'pathspec>=1.0'`. **Local git only** — no remote, no `git fetch` by default
(only when you pass a remote ref explicitly).

Two subcommands → two questions:

| I want | command |
|--------|---------|
| which files my CODEOWNERS edit changed the owner of | `codeowners-check.py diff` |
| who owns these paths | `codeowners-check.py owners <path>...` or pipe into stdin |

`diff` has a deliberately narrow meaning: **the owner-assignment difference between two CODEOWNERS,
judged on a given file set**. `--before` / `--after` pick only the rule versions; `--on` alone decides
the file set (default `INDEX`, i.e. `git ls-files` — includes `git add`ed-but-uncommitted files,
excludes `??`) — so a branch that only touches files, not rules, always reports 0.

`owners` reads paths one per line from stdin, so "who owns the files I touched" is a pipe, not a
subcommand.

## Usage

```
codeowners-check.py diff   [--before <SRC>] [--after <SRC>] [--on <SRC>] [--flat] [--tree]
codeowners-check.py owners [--at <SRC>] [--no-resolve] [<path> ...]      # omit path (or pass `-`) to read stdin
```

- `diff` defaults: `--before ref:master` · `--after worktree` · `--on index`
- `owners` defaults: `--at worktree`
- `--flat` / `--tree` are two independent axes and combine: `--flat --tree` = the full tree, uncollapsed

> `SRC` is one value domain: `worktree` (the file on disk) / `index` (the staged copy, alias
> `staged`) / `ref:<name>` (a git ref, case-sensitive). File states are bare words, a ref is
> prefixed `ref:` — so a ref literally named `worktree`/`index` is spelled
> `ref:worktree`/`ref:index` and can't collide with the file states.

## Common scenarios

| # | scenario | command | result |
|---|----------|---------|--------|
| 1 | pre-submit self-check: which files did I touch, who should look | `git diff --name-only $(git merge-base master HEAD) \| codeowners-check.py owners` | folds by owner, unowned files marked `NO OWNER` |
| 2 | count new files that aren't `git add`ed yet | `{ git diff --name-only $(git merge-base master HEAD); git ls-files --others --exclude-standard; } \| codeowners-check.py owners` | `git diff` doesn't list `??`, so add a second source |
| 3 | uncommitted changes only | `git diff --name-only HEAD \| codeowners-check.py owners` | HEAD → worktree |
| 4 | which files a commit touched | `git show --name-only --format= HEAD \| codeowners-check.py owners` | any file list feeds in |
| 5 | CI gate: block unowned files | `... \| codeowners-check.py owners \| grep -q 'NO OWNER' && exit 1` | the tool itself doesn't gate; you judge |
| 6 | I changed CODEOWNERS — what's the blast radius | `codeowners-check.py diff` | `-`/`+` mark owner add/remove, same-direction dirs fold |
| 7 | uncommitted changes' impact | `codeowners-check.py diff --before ref:HEAD` | staged + unstaged, vs last commit |
| 7a | **staged** rule change only | `codeowners-check.py diff --before ref:HEAD --after index` | HEAD → staging area |
| 7b | **unstaged** rule change only | `codeowners-check.py diff --before index --after worktree` | staging area → disk |
| 8 | after a manual fetch, vs upstream | `codeowners-check.py diff --before ref:upstream/master` | full diff vs upstream (needs `git fetch upstream` first) |
| 9 | see the directory tree | `codeowners-check.py diff --tree` | indented tree, owner change inline as `@old → @new` |
| 10 | one file per line (no folding) | `codeowners-check.py diff --flat` | one line per file |
| 11 | query one file | `codeowners-check.py owners bishengir/lib/Dialect/HFusion/Transforms/AutoVectorizeV2.cpp` | `path @u1 @u2` or `NO OWNER` |
| 12 | query a directory's owner distribution | `codeowners-check.py owners bishengir/lib/Dialect/HFusion/` | enumerate its files, same-owner fold, different-owner split |
| 13 | make diff count untracked (`??`) new files | `codeowners-check.py diff --on worktree` | file set = tracked + untracked |
| 14 | evaluate on an upstream file tree | `codeowners-check.py diff --on ref:origin/master` | needs `git fetch origin` first |
| 15 | query ownership on a commit/branch (no local edits) | `codeowners-check.py owners --at ref:master <path>` | `--at` shares the same value domain as `--before`/`--after` |
| 16 | who reviews a MR's changed files (incl. added/deleted, not yet local) | `git diff --name-only $(git merge-base master HEAD) \| codeowners-check.py owners --no-resolve` | owners by pattern; absent paths no longer print `(no matching files)` |

## Matching semantics (aligned with GitCode)

- **last-match-wins**: when a file matches several rules, the **last** rule's owners win (same as
  GitLab / GitCode).
- Supports a root anchor `/`, `**` (crosses directories), `*` (does not cross `/`), and a trailing
  `/` directory pattern (matches files under the dir, not a sibling `.cpp`).
- Owners may be `@user`, `@group/sub` or `user@example.com`; a bare word (no `@`) is invalid and warned.
- **Unanchored patterns warn**: a pattern without a leading `/` matches at any depth under
  gitignore, which may diverge from GitCode — it is kept but flagged; prefer a leading `/`.
- Skips blank lines, `#` comments, and functional `[Section]` / `^[Section]` sections; this repo's
  section headings are `#` comments, not functional sections.

## Gotchas

| gotcha | explanation |
|--------|-------------|
| pipe uses a wrong diff range | `git diff --name-only master` (two-dot) counts others' changes as yours once master advances; `master...HEAD` (three-dot) is committed-only, misses uncommitted. Use `$(git merge-base master HEAD)` |
| read `owner changes: N` as a line count | N is the number of **changed files**; one A→B is 1 file, printed as `-`/`+` two lines |

## Edge behavior (the contract)

| situation | behavior |
|-----------|----------|
| run with no subcommand | print help |
| self-test | none built in: the core logic is `parse_codeowners` / `Codeowners.owners`, which you can feed text to in a python REPL |
| source not found (`--before`/`--after`/`--on`/`--at`) | validated **up front**: exit 2, naming the valid sources (`worktree` / `index` / `ref:<name>`) |
| a ref has no CODEOWNERS | not an error — empty ruleset, so every file is unowned in that state |
| only reads the repo-root `CODEOWNERS` | `.github/CODEOWNERS` / `docs/CODEOWNERS` are not consulted — anywhere else reads as "no owner" |
| a rule line has an invalid owner token (bare word, no `@`) | stderr warning; the rule survives if any valid owner remains, else it's skipped |
| an unanchored pattern (no leading `/`) | stderr warning, the rule still applies |
| `owners` with empty stdin (no path, empty pipe) | exit 2 "empty input on stdin"; does **not** enumerate the whole repo |
| `owners /` / `owners .` (the repo root) | normalized to every file, same as listing all |
| `owners` path not found (typo / empty dir / absent at that source) | `path  (no matching files)` |
| `owners --no-resolve` with a path absent at `--at` | skips tree resolution; reports owner by pattern (`NO OWNER` if no rule) — for MR changed-file lists |
| `owners` file set follows `--at` | `--at worktree` includes untracked files; `--at index` or a ref only what is tracked there |
| `owners` with `--flat` / `--tree` | those flags aren't registered for it — argparse errors |
| run from a subdirectory | auto-`cd`s to the repo root; **all paths resolve repo-root-relative** |

## Exit codes

| code | meaning |
|------|---------|
| 0 | success (incl. no subcommand, `owners` finding nothing, `diff` with no changes) |
| 2 | argument error, ref not found, `pathspec` dependency missing, not inside a git repository, or a git subcommand failed |
