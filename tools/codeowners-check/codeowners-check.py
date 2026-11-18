#!/usr/bin/env python3
"""Local CODEOWNERS impact checker — two subcommands, all local git.

  diff    — owner changes a CODEOWNERS edit makes; `--flat`/`--tree` shape the output
  owners  — who owns path(s), from args or stdin

Item = (file → owner set | none), built by compute → coalesce → render.
"""

import argparse
import os
import subprocess
import sys
from collections import defaultdict
from functools import cache
from operator import attrgetter
from pathlib import Path
from typing import NamedTuple


def _fail(msg):
    # 2 is argparse's own usage-error code, so every bad invocation exits alike.
    sys.stderr.write(msg + "\n")
    sys.exit(2)


try:
    import pathspec
    # `gitignore` is the documented-gitignore factory added in pathspec 1.0;
    # `gitwildmatch` (v0.12's only option) is deprecated as of 1.0 and slated
    # for removal. Compiling a probe fails identically on a too-old pathspec
    # (KeyError: pattern not registered) and on a missing one (ImportError).
    pathspec.PathSpec.from_lines("gitignore", ["x"])
except (ImportError, KeyError):
    _fail("error: pathspec >= 1.0 missing, run `pip install -U pathspec`")

# A "source" names where CODEOWNERS text or a file list comes from: the on-disk
# file (WORKTREE), the staged copy (INDEX), or any git ref — git's three file
# states. On the CLI a bare `worktree` / `index` (alias `staged`) resolves to one
# of these two sentinel tuples; `ref:<name>` is a git ref, kept as a string. The
# tuple-vs-string split keeps the domain injective — a ref literally named
# `worktree` is spelled `ref:worktree` and stays a string, never a tuple.
WORKTREE = ("worktree",)
INDEX = ("index",)


# ---------------------------------------------------------------------------
# CODEOWNERS: parse rules, then resolve owners
# ---------------------------------------------------------------------------
class Rule(NamedTuple):
    """One CODEOWNERS line: a gitignore pattern and its owners."""
    pattern: str
    owners: list[str]  # keep '@'


def _valid_owner(tok):
    """A GitCode owner is `@user`, `@group/sub` or an email; a bare word (no
    `@`) is not an owner."""
    if len(tok) <= 1:
        return False
    if tok.startswith("@"):
        return True
    return tok.count("@") == 1 and not tok.endswith("@")


def parse_codeowners(text):
    """Return (rules, warnings). Pure — text in, rules out.

    Skips blank/`#` lines and `[Section]`/`^[Section]` blocks; owners are sorted
    into a canonical set (order is not meaning). An unanchored pattern (no
    leading `/`) is kept but warned: gitignore matches it at any depth, which
    may diverge from GitCode.
    """
    rules = []
    warnings = []
    # A UTF-8 BOM (U+FEFF) survives both `Path.read_text(encoding="utf-8")` and
    # git's `text=True` output; without stripping it, the first line parses as a
    # bogus pattern. Strip once here so every read path is covered.
    text = text.removeprefix("\ufeff")
    for lineno, raw in enumerate(text.splitlines(), 1):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith(("[", "^[")) and s.endswith("]"):
            continue
        parts = s.split()
        pattern, owner_tokens = parts[0], parts[1:]
        if not pattern.startswith("/"):
            warnings.append(
                f"CODEOWNERS line {lineno}: unanchored pattern {pattern!r} "
                "matches at any depth under gitignore; verify against GitCode"
            )
        owners = []
        for tok in owner_tokens:
            if _valid_owner(tok):
                owners.append(tok)
            else:
                warnings.append(
                    f"CODEOWNERS line {lineno}: invalid owner {tok!r} for {pattern!r}"
                )
        if not owners:
            warnings.append(f"CODEOWNERS line {lineno}: no valid owners for {pattern!r}")
            continue
        rules.append(Rule(pattern, sorted(owners)))
    return rules, warnings


class Codeowners:
    """Compiled rules → {path: owners}."""

    def __init__(self, rules):
        self.rules = rules
        # `check_file` reports the index of the *last* pattern that matched,
        # which is exactly CODEOWNERS' last-match-wins rule — so one combined
        # PathSpec replaces a per-rule spec plus a reverse scan, and pathspec
        # normalizes each path once instead of once per rule.
        # Must be fed parsed patterns, never raw CODEOWNERS text: `from_lines`
        # keeps comment lines, which would shift every index into `rules`.
        self._spec = pathspec.PathSpec.from_lines(
            "gitignore", [r.pattern for r in rules])

    def owners(self, files):
        """`{path: owners}` for every file in `files`; `[]` where no rule matches.

        `check_files` yields one result per input file, unmatched included —
        `match_files` would drop those and lose the unowned files entirely.
        """
        return {
            r.file: (self.rules[r.index].owners if r.include and r.index is not None else [])
            for r in self._spec.check_files(files)
        }

    def __eq__(self, other):
        """Behavioral equality (same pattern→owners), so `diff` skips a no-op edit."""
        if not isinstance(other, Codeowners):
            return NotImplemented
        return self.rules == other.rules


# ---------------------------------------------------------------------------
# git helpers
# ---------------------------------------------------------------------------
def _git(args):
    # `text=True` alone decodes with the locale (ASCII under C/POSIX), so a repo
    # with non-ASCII filenames or CODEOWNERS content raises UnicodeDecodeError.
    # Pin UTF-8 and degrade gracefully on undecodable bytes instead of crashing.
    # `core.quotepath=false` makes git print pathnames verbatim (no `"\"\303\251"`)
    # so a non-ASCII filename round-trips through every listing unchanged; `-z`
    # additionally splits on NUL, so no name quoting survives anywhere.
    return subprocess.run(
        ["git", "-c", "core.quotepath=false"] + args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _chdir_to_repo_root():
    root = _git_stdout(["rev-parse", "--show-toplevel"])
    if root is None:
        _fail("error: not inside a git repository")
    os.chdir(root)


def _git_lines(args):
    r = _git(args)
    if r.returncode != 0:
        _fail(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return [l for l in r.stdout.splitlines() if l]


def _git_stdout(args):
    r = _git(args)
    return r.stdout.strip() if r.returncode == 0 else None


@cache
def _ref_short_sha(ref):
    """Short sha or None; doubles as the existence check in `_src_validate` (cached: validate + describe each resolve it)."""
    return _git_stdout(["rev-parse", "--short", ref])


@cache
def _head_location():
    """Current HEAD as `branch@short-sha` (e.g. `feature-x@a1b2c3d`)."""
    branch = _git_stdout(["rev-parse", "--abbrev-ref", "HEAD"])
    return f"{branch}@{_ref_short_sha('HEAD') or '?'}"


# ---------------------------------------------------------------------------
# Source: git's three usable locations, one value domain
# ---------------------------------------------------------------------------
def _src_read_file(src, name):
    """Content of `name` at `src`, or "" when absent."""
    if src == WORKTREE:
        try:
            return Path(name).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return ""
    # An empty rev makes `git show :<name>` read the index. Raw stdout, not
    # `_git_stdout`: its strip() would eat leading blank lines and shift every
    # line number `parse_codeowners` reports.
    rev = "" if src == INDEX else src
    r = _git(["show", f"{rev}:{name}"])
    return r.stdout if r.returncode == 0 else ""


def _src_list_files(src, paths=None):
    """Repo-root-relative paths at `src`. `paths=None` lists every file;
    `paths=[]` lists none; otherwise narrowed to `paths`.

    git expands a directory argument recursively and dedups, so a caller can
    pass a mix of files, directories and duplicates.
    """
    # These are user-supplied file or directory names, not Git pathspecs.
    # Keep metacharacters such as `[]`, `*` and `:(glob)` literal while letting
    # Git continue to expand a literal directory path recursively. `--literal-pathspecs`
    # is a global option and must precede the subcommand.
    if src == WORKTREE:
        # `--cached` must be spelled out: passing any listing option replaces
        # the implicit `--cached` default, so `--others` alone would drop
        # tracked files.
        args = ["ls-files", "--cached", "--others", "--exclude-standard"]
    elif src == INDEX:
        args = ["ls-files", "--cached"]
    else:
        args = ["ls-tree", "-r", "--name-only", src]

    if paths is not None:
        args = ["--literal-pathspecs", *args, "--", *paths]
    return _git_lines(args)


def _strip_redundant_prefix(p):
    """Drop a redundant leading `/` and `./` (shell globs and `find .` emit `./`);
    `.` (the repo root) collapses to `""`.

    `..` is preserved: `../foo` is outside the repo and should fail at git rather
    than be silently rewritten.
    """
    p = p.lstrip("/")
    while p.startswith("./"):
        p = p.removeprefix("./")
    return "" if p == "." else p


def _src_resolve_paths(src, paths):
    """Expand `paths` at `src`; return (files, paths that matched nothing).

    A leading `/` or `./` is redundant on repo-root-relative paths, so it's
    stripped and the empty "root" entries dropped: `owners /` lists every file,
    `owners /foo` means `foo`. A trailing `/` (`dir/`) is equivalent to `dir`.
    """
    paths = [_strip_redundant_prefix(p) for p in paths]
    paths = [p for p in paths if p]
    if not paths:
        # everything collapsed to the root, e.g. `owners /`
        return _src_list_files(src), []
    files = _src_list_files(src, paths)
    found = set(files)
    # `+ "/"` makes this a directory-boundary test, so `a` won't match a sibling
    # file like `ab/c.cpp`; rstrip first, or the `dir/` spelling the manuals use
    # would build the prefix `dir//` and match nothing.
    missing = []
    for p in paths:
        if p in found:
            continue
        prefix = p.rstrip("/") + "/"
        if not any(f.startswith(prefix) for f in files):
            missing.append(p)
    return files, missing


def _src_describe(src):
    """Human-readable name, e.g. `master (a1b2c3d)`."""
    if src == WORKTREE:
        return f"WORKTREE ({_head_location()})"
    if src == INDEX:
        return "INDEX (staged)"
    return f"{src} ({_ref_short_sha(src)})"


def _src_validate(src, flag):
    """Fail with one clear message when `src` is unusable. The two file states
    always work; a ref has to resolve."""
    if src not in (WORKTREE, INDEX) and _ref_short_sha(src) is None:
        _fail(f"error: {flag} {src!r} is not a git ref or a file state\n"
              f"       file states: worktree (the file on disk), "
              "index / staged (the staged copy);\n"
              "       a ref is spelled ref:<name>; a remote branch needs `git fetch` first")


def _source(value):
    """argparse type: a bare `worktree` / `index` (alias `staged`) names a file
    state (case-insensitive); `ref:<name>` is a git ref, passed through
    case-sensitively.

    Runs before `_chdir_to_repo_root`, so it can't consult git — `_src_validate`
    checks a ref's existence later.
    """
    if value.startswith("ref:"):
        return value.removeprefix("ref:")
    up = value.upper()
    if up == "WORKTREE":
        return WORKTREE
    if up in ("INDEX", "STAGED"):
        return INDEX
    _fail(f"error: unknown source {value!r} "
          "(expected worktree, index, staged, or ref:<name>)")


def _add_src_arg(parser, flag, default, what):
    """Register a source argument: worktree / index / staged, or ref:<name>."""
    parser.add_argument(flag, default=default, metavar="SRC", type=_source,
                        help=f"{what}: worktree / index / staged, or ref:<name> "
                             f"(default {default})")


def load_codeowners(src):
    """Read + parse CODEOWNERS at `src`, print warnings, return a compiled `Codeowners`."""
    rules, warnings = parse_codeowners(_src_read_file(src, "CODEOWNERS"))
    # Label the source: `diff` loads two, and unlabelled warnings from both are
    # indistinguishable (and look like each one was printed twice).
    where = _src_describe(src)
    for w in warnings:
        sys.stderr.write(f"{where}: {w}\n")
    return Codeowners(rules)


# ---------------------------------------------------------------------------
# compute: items, diff, and the directory fold
# ---------------------------------------------------------------------------
class Item(NamedTuple):
    """One file, or one folded directory whose files all moved the same way."""
    path: str
    owners: tuple[str, ...]          # the resulting set
    before: tuple[str, ...] = ()     # the prior set; empty outside `diff`
    count: int = 1                   # how many files this entry stands for


def _item(path, owners, before=()):
    """Build an `Item`, normalizing both owner sets to tuples."""
    return Item(path, tuple(owners), tuple(before))


def compute_diff(before_map, after_map):
    """One record per changed file; takes `{path: owners}` per side.

    A file whose owner set is unchanged is skipped — unowned on both sides
    included, which is why a rule edit can legitimately report 0 changes.
    """
    results = []
    for path in before_map.keys() | after_map.keys():
        before = sorted(before_map.get(path, ()))
        after = sorted(after_map.get(path, ()))
        if before == after:
            continue
        results.append(_item(path, after, before=before))
    return results


class _Node:
    __slots__ = ("children", "result")

    def __init__(self):
        self.children = defaultdict(_Node)
        self.result = None


def _collapse_key(e):
    """Folding identity: the whole before→after transition, so a dir folds only
    when all its files moved the same way. Every `Item` field but `path` and
    `count` belongs here — the fold copies the rest from one sibling — and owner
    order must be canonical, since tuples compare positionally."""
    return (e.owners, e.before)


# Entries are presented in full-path order.
_PATH_KEY = attrgetter("path")


def _build_trie(entries):
    """The entry list as a path trie: a `_Node` per path segment, the entry
    itself on the terminal node. Shared by `coalesce` (fold) and `_render_tree`
    (draw)."""
    root = _Node()
    for e in entries:
        node = root
        for part in e.path.split("/"):
            node = node.children[part]
        node.result = e
    return root


def coalesce(results):
    """Fold siblings sharing a `_collapse_key` into one directory entry."""
    root = _build_trie(results)

    def walk(node, path):
        if node.result is not None:  # file leaf
            return [node.result]
        flat = []
        for name, child in node.children.items():
            # These are git paths: always `/`, never the platform separator.
            flat.extend(walk(child, f"{path}/{name}" if path else name))
        # Fold only when >=2 siblings agree; a lone entry stays as itself so a
        # single-child chain is not relabelled as its parent directory. Note the
        # root participates: if every top-level entry agrees, the whole set folds
        # to one entry whose path is "".
        if len(flat) >= 2:
            key0 = _collapse_key(flat[0])
            if all(_collapse_key(e) == key0 for e in flat):
                total = sum(e.count for e in flat)
                return [flat[0]._replace(path=path, count=total)]
        return flat

    return walk(root, "")


# ---------------------------------------------------------------------------
# render: terminal consumers
# ---------------------------------------------------------------------------
def _label(base, e):
    """`base`, or `base/ (N files)` when `e` is a folded directory. `base` is
    `""` when the whole set folded to the root."""
    if not base:
        return f"(whole repo, {e.count} files)" if e.count > 1 else "(whole repo)"
    return f"{base}/ ({e.count} files)" if e.count > 1 else base


def _join_owners(owners, empty):
    """`owners` joined as `@a @b`, or `empty` when there are none."""
    return " ".join(owners) or empty


def _render_terminal_diff(entries):
    entries = sorted(entries, key=_PATH_KEY)
    total = sum(e.count for e in entries)
    lines = [f"owner changes: {total}"]
    for e in entries:
        path = _label(e.path, e)
        if e.before:
            lines.append(f"- {path}  {_join_owners(e.before, '')}")
        if e.owners:
            lines.append(f"+ {path}  {_join_owners(e.owners, '')}")
    return "\n".join(lines)


def _render_terminal_owners(entries):
    entries = sorted(entries, key=_PATH_KEY)
    return "\n".join(
        f"{_label(e.path, e)}  {_join_owners(e.owners, 'NO OWNER')}" for e in entries
    )


_TREE_BRANCH = "├── "
_TREE_LAST = "└── "
_TREE_CONT = "│   "
_TREE_GAP = "    "


def _tree_owner(e):
    """`@old → @new` on one line; `∅` marks a side with no owner."""
    return f"{_join_owners(e.before, '∅')} → {_join_owners(e.owners, '∅')}"


def _render_tree(entries):
    """Collapsed entries as an indented box-drawing directory tree (owner change inline)."""
    root = _build_trie(entries)

    lines = []

    def walk(node, prefix, top):
        names = sorted(node.children)
        for i, name in enumerate(names):
            last = i == len(names) - 1
            child = node.children[name]
            box = "" if top else (_TREE_LAST if last else _TREE_BRANCH)
            cont = "" if top else (_TREE_GAP if last else _TREE_CONT)
            if child.result is None:
                lines.append(f"{prefix}{box}{name}")
                walk(child, prefix + cont, False)
            else:
                e = child.result
                label = _label(name, e)
                lines.append(f"{prefix}{box}{label}  {_tree_owner(e)}")

    walk(root, "", True)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------
def cmd_diff(args):
    _src_validate(args.before, "--before")
    _src_validate(args.after, "--after")
    _src_validate(args.on, "--on")
    before = load_codeowners(args.before)
    after = load_codeowners(args.after)
    if before == after:
        # Same rules can't re-assign anything; skip the (common) whole-tree
        # resolve — and its file listing.
        results = []
    else:
        # One file list, from --on only: a path that exists in one revision but
        # not the other never enters the comparison. diff reports rule
        # re-assignment, never file adds or deletes.
        files = _src_list_files(args.on)
        ctx = (f"CODEOWNERS: {_src_describe(args.before)} → {_src_describe(args.after)}"
               f"   files: {_src_describe(args.on)} ({len(files)} files)")
        sys.stderr.write(ctx + "\n")
        results = compute_diff(before.owners(files), after.owners(files))

    entries = results if args.flat else coalesce(results)
    if args.tree:
        print(_render_tree(entries))
    else:
        print(_render_terminal_diff(entries))
    return 0


def _owners_paths(args):
    """Paths from argv, or stdin (one per line) when argv is empty or `-`."""
    if args.path and args.path != ["-"]:
        return args.path
    if sys.stdin.isatty():
        _fail("error: no paths given (pass them as arguments, or pipe one per line)")
    paths = [s for line in sys.stdin if (s := line.strip())]
    if not paths:
        _fail("error: no paths given (empty input on stdin)")
    return paths


def cmd_owners(args):
    _src_validate(args.at, "--at")
    co = load_codeowners(args.at)
    paths = _owners_paths(args)
    if args.no_resolve:
        # Literal mode: match owners by pattern only, no tree resolution — so a
        # path absent at --at (an MR's added/renamed/deleted file) still gets an
        # owner verdict instead of "(no matching files)".
        paths = [_strip_redundant_prefix(p) for p in paths]
        paths = [p for p in paths if p]
        if not paths:
            return 0
        items = [_item(p, o) for p, o in co.owners(paths).items()]
        if items:
            # No coalesce here: literal paths may include a directory name that is
            # also a strict prefix of another (e.g. `docs` and `docs/a.md`), and
            # `coalesce`'s trie fold drops the descendant. One line per input path.
            print(_render_terminal_owners(items))
        return 0
    # Resolve as one set so a piped list folds the same way a directory does.
    files, missing = _src_resolve_paths(args.at, paths)
    items = [_item(p, o) for p, o in co.owners(files).items()]
    if items:
        print(_render_terminal_owners(coalesce(items)))
    for path in missing:
        print(f"{path}  (no matching files)")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Local CODEOWNERS impact checker (run inside a git repository)",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_diff = sub.add_parser(
        "diff",
        help="what a CODEOWNERS edit re-assigns, judged on your current files",
        description="Resolve the files at --on under two CODEOWNERS versions and report where "
                    "the owners differ. --before/--after pick which CODEOWNERS text to compare, "
                    "--on picks the file tree to judge on — e.g. "
                    "`--before ref:HEAD --after index` shows what staging your edit did.",
    )
    _add_src_arg(p_diff, "--before", "ref:master", "OLD CODEOWNERS")
    _add_src_arg(p_diff, "--after", "worktree", "NEW CODEOWNERS")
    _add_src_arg(p_diff, "--on", "index", "file tree to evaluate on")
    p_diff.add_argument("--flat", action="store_true",
                        help="disable directory collapsing (every file is its own entry)")
    p_diff.add_argument("--tree", action="store_true",
                        help="render as an indented box-drawing tree")
    p_diff.set_defaults(func=cmd_diff)

    p_owners = sub.add_parser(
        "owners",
        help="who owns these paths (args, or one per line on stdin)",
        description="Resolve each path under a CODEOWNERS chosen by --at. "
                    "Files, directories and piped file lists all work; same-owner siblings fold "
                    "into one directory line.",
    )
    _add_src_arg(p_owners, "--at", "worktree", "which CODEOWNERS to query")
    p_owners.add_argument(
        "path", nargs="*",
        help="repo-root-relative path(s); omit (or pass -) to read them from stdin, one per line",
    )
    p_owners.add_argument(
        "--no-resolve", action="store_true",
        help="match owners by pattern only — don't require each path to exist at --at "
             "(for MR file lists that include added/renamed/deleted paths)",
    )
    p_owners.set_defaults(func=cmd_owners)

    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 0

    _chdir_to_repo_root()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
