# codeowners-check — Developer Workflows & Iteration Guide

> 中文见 [DEVELOPER.md](DEVELOPER.md)

`codeowners-check.py` is the local CODEOWNERS impact checker. Command reference:
[TLDR.en.md](TLDR.en.md). This guide covers three developer workflows and how to
triage a "local ≠ GitCode" mismatch.

## Three workflows

### 1. I changed CODEOWNERS → confirm the diff locally

```bash
codeowners-check.py diff --before ref:master
```

`-@old` / `+@new` per file; a branch that only touches files, not rules, always
reports 0. `--tree` for a folded tree, `--flat` for one file per line. Expected:
the listed files == the files you meant to reassign. An unexpected path means a
rule is too broad (a bare `*` rule or an unanchored pattern).

### 2. owner self-check → is this path assigned to me

```bash
codeowners-check.py owners --at ref:master <path>
git diff --name-only $(git merge-base master HEAD) | codeowners-check.py owners | grep '@you'
```

`path @u1 @u2`; your handle absent = the path does not reach you. `--at ref:master`
judges by the online rule set, ignoring local edits.

### 3. local ≠ online → iterate the tool

1. Check the semantics table below — most mismatches are understanding gaps.
2. If it is a tool bug, edit in place: single file, stdlib + pathspec, no build.
3. Core logic lives in `parse_codeowners`, `Codeowners.owners`, `compute_diff`.
4. No built-in self-test: verify by feeding text to the functions in a python
   REPL, or a throwaway git repo for real commands.

## How semantics are encoded (GitCode-aligned)

| semantics | where | note |
|-----------|-------|------|
| last-match-wins | `Codeowners.owners` | last matching rule wins |
| `*` not across `/`, `**` across dirs | pathspec `gitignore` factory | migrated from deprecated `gitwildmatch` |
| leading `/` root anchor | `parse_codeowners` | unanchored patterns warn |
| trailing `/` directory pattern | pathspec | dir contents, not a sibling `.cpp` |
| `@user` / `@group/sub` / `user@example.com` | `_valid_owner` | bare word invalid, warned |
| blank lines / `#` / functional sections skipped | `parse_codeowners` | repo sections are `#` comments |
| repo-root `CODEOWNERS` only | `load_codeowners` | `.github/CODEOWNERS` etc. ignored |

## Iteration rules

- Semantics: add a REPL check first, then edit `Codeowners.owners`, then sync this
  doc + TLDR's "Matching semantics".
- New subcommand/flag: sync TLDR usage/scenarios/exit-code tables and the root
  `AGENTS.md` index.
- Dependency bump: the `pathspec>=1.0` probe guards `main`; confirm the target
  version still provides the `gitignore` factory.
- Commit: repo `AGENTS.md` `[Tools]` module + `Assisted-by: AI` trailer.
