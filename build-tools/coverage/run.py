#!/usr/bin/env python3
"""Run gcov-compatible coverage workloads and generate lcov reports."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from blacklist import BLACKLIST

# Keep the runner project-scoped. If the coverage scope changes, update this
# constant and blacklist.py instead of adding runtime compatibility switches.
BISHENGIR_EXTRACT_PATTERN = "*/bishengir/*"
LCOV_IGNORE_ERRORS = "gcov,source,graph,inconsistent,unused"


class CoverageError(RuntimeError):
    """Raised for user-facing coverage failures."""


def log(message: str) -> None:
    print(f"[coverage] {message}")


def warn(message: str) -> None:
    print(f"[coverage] {message}", file=sys.stderr)


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def default_jobs() -> int:
    cpus = os.cpu_count() or 1
    return max(1, min(8, cpus // 4))


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def run_checked(command: list[str], *, cwd: Path | None = None) -> None:
    try:
        subprocess.run(command, check=True, cwd=cwd, text=True)
    except OSError as exc:
        raise CoverageError(f"failed to execute: {shlex.join(command)}") from exc
    except subprocess.CalledProcessError as exc:
        raise CoverageError(f"command failed ({exc.returncode}): {shlex.join(command)}") from exc


def run_workload(command: list[str], *, cwd: Path) -> int:
    try:
        result = subprocess.run(command, cwd=cwd, text=True)
    except OSError as exc:
        raise CoverageError(f"failed to execute: {shlex.join(command)}") from exc
    return result.returncode


def existing_path(candidate: str | Path, *roots: Path) -> Path | None:
    raw = Path(candidate).expanduser()
    candidates = [raw] if raw.is_absolute() else [raw, *(root / raw for root in roots)]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def planned_path(candidate: str | Path, *, default_root: Path, extra_roots: tuple[Path, ...] = ()) -> Path:
    raw = Path(candidate).expanduser()
    if raw.is_absolute():
        return raw.resolve(strict=False)

    preferred = (default_root / raw).resolve(strict=False)
    fallbacks = [preferred, *(root / raw for root in extra_roots)]
    for path in fallbacks:
        if path.exists():
            return path.resolve()
    return preferred


def discover_repo_root(build_dir: Path) -> Path:
    commands = (
        ["git", "-C", str(build_dir), "rev-parse", "--show-toplevel"],
        ["git", "rev-parse", "--show-toplevel"],
    )
    for command in commands:
        try:
            root = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            continue
        if root:
            return Path(root)
    return Path.cwd()


@dataclass(frozen=True)
class CommonOptions:
    build_dir: Path
    out_dir: Path
    jobs: int
    diff_range: str | None = None
    llvm_cov_gcov_flags: str = ""

    @classmethod
    def from_namespace(cls, args: argparse.Namespace, repo_root: Path) -> "CommonOptions":
        build_dir = resolve_path(args.build_dir)
        out_dir = resolve_path(args.out_dir) if args.out_dir else resolve_path(build_dir / "coverage")
        return cls(
            build_dir=build_dir,
            out_dir=out_dir,
            jobs=args.jobs,
            diff_range=args.diff_range,
            llvm_cov_gcov_flags=args.llvm_cov_gcov_flags,
        )


@dataclass(frozen=True)
class OutputLayout:
    root: Path
    capture_info: Path
    extracted_info: Path
    filtered_info: Path
    html_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> "OutputLayout":
        return cls(
            root=root,
            capture_info=root / "coverage.info",
            extracted_info=root / "coverage.filtered.info",
            filtered_info=root / "coverage.cleaned.info",
            html_dir=root / "html",
        )

    def reset(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for path in (self.capture_info, self.extracted_info, self.filtered_info):
            if path.exists():
                path.unlink()
        if self.html_dir.exists():
            shutil.rmtree(self.html_dir)


@dataclass(frozen=True)
class CommandPlan:
    mode: str
    cwd: Path
    command: tuple[str, ...]
    summary_lines: tuple[str, ...] = ()
    prepare_commands: tuple[tuple[str, ...], ...] = ()


def require_tool(binary_name: str) -> Path:
    resolved = shutil.which(binary_name)
    if not resolved:
        raise CoverageError(f"missing tool in PATH: {binary_name}")
    return Path(resolved)


def read_blacklist_patterns() -> tuple[str, ...]:
    patterns = BLACKLIST
    if not isinstance(patterns, (list, tuple)):
        raise CoverageError("BLACKLIST must be a list or tuple")
    return tuple(str(pattern) for pattern in patterns if str(pattern).strip())


class CoverageSession:
    def __init__(self, common: CommonOptions, repo_root: Path) -> None:
        self.common = common
        self.repo_root = repo_root
        self.lcov = require_tool("lcov")
        self.genhtml = require_tool("genhtml")
        self.llvm_cov = require_tool("llvm-cov")
        self.outputs = OutputLayout.from_root(common.out_dir)
        self.blacklist_patterns = read_blacklist_patterns()
        self._gcov_tool: Path | None = None
        self._llvm_cov_gcov_flags = common.llvm_cov_gcov_flags

    def run(self, plan: CommandPlan) -> None:
        self._warn_if_llvm_cov_version_mismatch()
        self.outputs.reset()
        self._reset_gcda_files()

        for command in plan.prepare_commands:
            log(f"preparing: {shlex.join(command)}")
            run_checked(list(command), cwd=self.common.build_dir)

        log(f"mode: {plan.mode}")
        for line in plan.summary_lines:
            log(line)
        log(f"lcov: {self.lcov}")
        log(f"llvm-cov: {self.llvm_cov}")
        log(f"genhtml: {self.genhtml}")
        log(f"command: {shlex.join(plan.command)}")

        workload_returncode = run_workload(list(plan.command), cwd=plan.cwd)
        if workload_returncode != 0:
            warn(
                f"workload failed ({workload_returncode}); "
                "continuing coverage collection from generated .gcda files"
            )
        if not self._find_gcda_files():
            raise CoverageError(
                f"no .gcda files found under {self.common.build_dir}; "
                "ensure the build was configured with build-tools/build.sh --coverage "
                "and the workload executed instrumented binaries"
            )

        with tempfile.TemporaryDirectory(prefix="coverage-gcov-") as wrapper_dir:
            self._gcov_tool = self._write_gcov_wrapper(Path(wrapper_dir))
            self._capture()
            self._extract()
            self._remove()
            if self.common.diff_range is not None:
                self._filter_by_diff()
            self._generate_html()
        self._print_outputs()
        if workload_returncode != 0:
            warn(
                f"workload failed ({workload_returncode}); coverage report was generated"
            )

    def _write_gcov_wrapper(self, wrapper_dir: Path) -> Path:
        # lcov --gcov-tool accepts an executable path, not argv like
        # "llvm-cov gcov", so generate the smallest possible adapter at runtime.
        wrapper = wrapper_dir / "llvm-cov-gcov"
        wrapper.write_text(
            "#!/usr/bin/env sh\n"
            f"exec {shlex.quote(str(self.llvm_cov))} gcov {self._llvm_cov_gcov_flags} \"$@\"\n",
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
        return wrapper

    def _warn_if_llvm_cov_version_mismatch(self) -> None:
        try:
            result = subprocess.run(
                [str(self.llvm_cov), "--version"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            warn(f"failed to check llvm-cov version: {self.llvm_cov}")
            return

        output = f"{result.stdout}\n{result.stderr}"
        match = re.search(r"LLVM version\s+(\d+)(?:\.\d+)*", output)
        if match and match.group(1) == "18":
            return

        first_line = next((line.strip() for line in output.splitlines() if line.strip()), "unknown version")
        warn(f"llvm-cov 18 is recommended; current llvm-cov is {first_line}")

    def _reset_gcda_files(self) -> None:
        for gcda in self.common.build_dir.rglob("*.gcda"):
            gcda.unlink()

    def _find_gcda_files(self) -> list[Path]:
        return sorted(self.common.build_dir.rglob("*.gcda"))

    def _lcov_base_command(self) -> list[str]:
        if self._gcov_tool is None:
            raise CoverageError("internal error: gcov wrapper is not initialized")
        return [
            str(self.lcov),
            "--gcov-tool",
            str(self._gcov_tool),
            "--rc",
            "branch_coverage=1",
            "--rc",
            "geninfo_unexecuted_blocks=1",
        ]

    def _capture(self) -> None:
        log(f"capturing lcov data to {self.outputs.capture_info}")
        run_checked(
            [
                *self._lcov_base_command(),
                "--capture",
                "--directory",
                str(self.common.build_dir),
                "--output-file",
                str(self.outputs.capture_info),
                "--ignore-errors",
                LCOV_IGNORE_ERRORS,
            ]
        )

    def _extract(self) -> None:
        log(f"extracting focused paths to {self.outputs.extracted_info}")
        run_checked(
            [
                *self._lcov_base_command(),
                "--extract",
                str(self.outputs.capture_info),
                BISHENGIR_EXTRACT_PATTERN,
                "--output-file",
                str(self.outputs.extracted_info),
                "--ignore-errors",
                LCOV_IGNORE_ERRORS,
            ]
        )

    def _remove(self) -> None:
        log(f"removing excluded paths to {self.outputs.filtered_info}")
        run_checked(
            [
                *self._lcov_base_command(),
                "--remove",
                str(self.outputs.extracted_info),
                *self.blacklist_patterns,
                "--output-file",
                str(self.outputs.filtered_info),
                "--ignore-errors",
                LCOV_IGNORE_ERRORS,
            ]
        )

    def _resolve_repo_path(self, path: str) -> str:
        """Normalize an absolute or repo-relative file path to an absolute string."""
        p = Path(path)
        if p.is_absolute():
            return str(p.resolve(strict=False))
        return str((self.repo_root / p).resolve(strict=False))

    def _get_added_lines(self, diff_range: str) -> dict[str, set[int]]:
        """Get added/modified line numbers per file for the given diff range.

        Returns a dict mapping absolute file paths to sets of added line numbers.
        Only lines that were added (not deleted or unchanged) are returned.
        """
        added_lines: dict[str, set[int]] = {}
        try:
            result = subprocess.run(
                ["git", "-C", str(self.repo_root), "diff", "--unified=0",
                 "--diff-filter=ACMRT", diff_range],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise CoverageError(
                f"git diff --unified=0 {diff_range} failed: {exc.stderr.strip()}"
            ) from exc

        import re
        current_file: str | None = None
        current_lines: set[int] = set()
        in_hunk = False

        for line in result.stdout.splitlines():
            if line.startswith("--- a/") or line.startswith("diff --git"):
                if current_file and current_lines:
                    abs_path = self._resolve_repo_path(current_file)
                    added_lines[abs_path] = current_lines
                if line.startswith("--- a/"):
                    current_file = line[4:].strip()
                    if current_file.startswith("a/"):
                        current_file = current_file[2:]
                current_lines = set()
                in_hunk = False
            elif line.startswith("@@"):
                in_hunk = True
                match = re.search(r'\+(\d+)(?:,(\d+))?', line)
                if match:
                    start = int(match.group(1))
                    count = int(match.group(2)) if match.group(2) else 1
                    for i in range(start, start + count):
                        current_lines.add(i)
            elif in_hunk and line.startswith("+") and not line.startswith("+++"):
                pass
            elif in_hunk and line.startswith(" "):
                pass

        if current_file and current_lines:
            abs_path = self._resolve_repo_path(current_file)
            added_lines[abs_path] = current_lines

        return added_lines

    def _filter_by_diff(self) -> None:
        """Filter coverage.cleaned.info to only keep coverage for changed lines.

        Handles FN, FNDA, DA, and BRDA records. For diff-matched files:
        - FN/FNDA: kept only for functions whose start line is in the diff
        - DA: kept only for lines in the diff
        - BRDA: kept only for branches on lines in the diff
        """
        diff_range = self.common.diff_range
        if diff_range is None:
            return

        added_lines_per_file = self._get_added_lines(diff_range)
        if not added_lines_per_file:
            raise CoverageError(
                f"git diff --unified=0 --diff-filter=ACMRT {diff_range} returned no changes; "
                "the range may be empty or invalid"
            )

        # Build a lookup that maps resolved absolute paths to added lines,
        # and also a fallback lookup by repo-relative path suffix for robustness.
        diff_by_abs: dict[str, set[int]] = {}
        diff_by_suffix: dict[str, set[int]] = {}
        for abs_path, lines in added_lines_per_file.items():
            diff_by_abs[abs_path] = lines
            try:
                rel = str(Path(abs_path).relative_to(self.repo_root))
                diff_by_suffix[rel] = lines
            except ValueError:
                pass

        diff_files = list(added_lines_per_file.keys())
        src = self.outputs.filtered_info
        dst = src.with_name(f"{src.name}.diff.tmp")

        # Phase 1: Parse all records from source file
        # Each record stores: sf_path, fn_list, fnda_list, da_list, brda_list
        # FN:  (line_number, function_name)
        # FNDA: (count, function_name)
        # DA:   (line_number, hit_count)
        # BRDA: (line_number, block_number, branch_number, taken)
        @dataclass
        class CovRecord:
            sf: str
            fn: list[tuple[int, str]] = None  # type: ignore[assignment]
            fnda: list[tuple[int, str]] = None  # type: ignore[assignment]
            da: list[tuple[int, int]] = None  # type: ignore[assignment]
            brda: list[tuple[int, int, int, str]] = None  # type: ignore[assignment]

            def __post_init__(self):
                self.fn = []
                self.fnda = []
                self.da = []
                self.brda = []

        records: list[CovRecord] = []
        current: CovRecord | None = None
        with open(src, encoding="utf-8") as fin:
            for line in fin:
                if line.startswith("SF:"):
                    if current is not None:
                        records.append(current)
                    current = CovRecord(sf=line[3:].strip())
                elif current is not None:
                    stripped = line.strip()
                    if stripped.startswith("FN:"):
                        m = re.match(r'FN:(\d+),(.+)', stripped)
                        if m:
                            current.fn.append((int(m.group(1)), m.group(2)))
                    elif stripped.startswith("FNDA:"):
                        m = re.match(r'FNDA:(\d+),(.+)', stripped)
                        if m:
                            current.fnda.append((int(m.group(1)), m.group(2)))
                    elif stripped.startswith("DA:"):
                        m = re.match(r'DA:(\d+),(\d+)', stripped)
                        if m:
                            current.da.append((int(m.group(1)), int(m.group(2))))
                    elif stripped.startswith("BRDA:"):
                        m = re.match(r'BRDA:(\d+),(\d+),(\d+),(.+)', stripped)
                        if m:
                            current.brda.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)))
            if current is not None:
                records.append(current)

        # Phase 2: Match and filter, write output
        def _match_sf(sf: str) -> tuple[bool, set[int]]:
            if sf in diff_by_abs:
                return True, diff_by_abs[sf]
            resolved = str(Path(sf).resolve(strict=False))
            if resolved in diff_by_abs:
                return True, diff_by_abs[resolved]
            try:
                sf_rel = str(Path(sf).relative_to(self.repo_root))
                if sf_rel in diff_by_suffix:
                    return True, diff_by_suffix[sf_rel]
            except ValueError:
                pass
            return False, set()

        kept = 0
        total_lines = 0
        hit_lines = 0
        total_branches = 0
        hit_branches = 0
        total_functions = 0
        hit_functions = 0
        sf_matched = 0

        with open(dst, "w", encoding="utf-8") as fout:
            for rec in records:
                matched, added_lines = _match_sf(rec.sf)
                if not matched:
                    continue
                sf_matched += 1

                # Filter FN/FNDA: keep only functions whose start line is in diff
                fn_names_in_diff: set[str] = set()
                for fn_line, fn_name in rec.fn:
                    if fn_line in added_lines:
                        fn_names_in_diff.add(fn_name)
                filtered_fnda = [(cnt, name) for cnt, name in rec.fnda if name in fn_names_in_diff]

                # Filter DA: keep only lines in diff
                filtered_da = [(ln, cnt) for ln, cnt in rec.da if ln in added_lines]

                # Filter BRDA: keep only branches on lines in diff
                filtered_brda = [(ln, blk, br, taken) for ln, blk, br, taken in rec.brda if ln in added_lines]

                # Skip file if nothing to report (no diff lines and no data)
                if not filtered_da and not added_lines and not filtered_fnda and not filtered_brda:
                    continue
                kept += 1
                fout.write(f"SF:{rec.sf}\n")

                # Write filtered FN
                for fn_line, fn_name in rec.fn:
                    if fn_line in added_lines:
                        fout.write(f"FN:{fn_line},{fn_name}\n")
                        total_functions += 1

                # Write filtered FNDA
                for cnt, name in filtered_fnda:
                    fout.write(f"FNDA:{cnt},{name}\n")
                    if cnt > 0:
                        hit_functions += 1

                # Write filtered DA
                for ln, cnt in filtered_da:
                    fout.write(f"DA:{ln},{cnt}\n")
                    if cnt > 0:
                        hit_lines += 1

                # For files with no DA data but in diff, write DA:0 for each
                # added line so genhtml recognizes the record and generates HTML.
                if not filtered_da and added_lines:
                    for ln in sorted(added_lines):
                        fout.write(f"DA:{ln},0\n")

                # Write filtered BRDA
                for ln, blk, br, taken in filtered_brda:
                    fout.write(f"BRDA:{ln},{blk},{br},{taken}\n")
                    total_branches += 1
                    if taken != "0" and taken != "-":
                        hit_branches += 1

                # LH/LF: use added_lines count when no DA data exists for the file,
                # so that diff-only files (e.g. headers with no executable lines)
                # still appear in the HTML report and count toward overall coverage.
                lh_value = sum(1 for ln, cnt in filtered_da if cnt > 0)
                lf_value = len(filtered_da) if filtered_da else len(added_lines)
                total_lines += lf_value
                fnh_value = sum(1 for cnt, _ in filtered_fnda if cnt > 0)
                fnf_value = len(filtered_fnda)
                brh_value = sum(1 for _, _, _, taken in filtered_brda if taken != "0" and taken != "-")
                brf_value = len(filtered_brda)
                fout.write(f"LH:{lh_value}\n")
                fout.write(f"LF:{lf_value}\n")
                fout.write(f"FNH:{fnh_value}\n")
                fout.write(f"FNF:{fnf_value}\n")
                fout.write(f"BRH:{brh_value}\n")
                fout.write(f"BRF:{brf_value}\n")
                fout.write("end_of_record\n")

        if kept == 0:
            dst.unlink(missing_ok=True)
            log(f"diff files from git: {diff_files}")
            log(f"SF matched: {sf_matched}, kept files with data: {kept}")
            # Still generate incremental info with 0% coverage for visibility
            with open(src, "w", encoding="utf-8") as f:
                f.write("TN:incremental\n")
                for abs_path in diff_files:
                    try:
                        rel = str(Path(abs_path).relative_to(self.repo_root))
                    except ValueError:
                        rel = abs_path
                    f.write(f"SF:{abs_path}\n")
                    f.write("LH:0\nLF:0\nFNH:0\nFNF:0\nBRH:0\nBRF:0\n")
                    f.write("end_of_record\n")
            pct = 0.0
            print(f"INCREMENTAL_COVERAGE: {hit_lines}/{total_lines} lines hit ({pct:.1f}%)")
            return

        src.unlink()
        dst.rename(src)
        pct = (hit_lines / total_lines * 100) if total_lines > 0 else 0.0
        branch_pct = (hit_branches / total_branches * 100) if total_branches > 0 else 0.0
        func_pct = (hit_functions / total_functions * 100) if total_functions > 0 else 0.0
        log(f"line-level diff filter: kept {kept} file(s), {total_lines} changed line(s), {hit_lines} hit")
        log(f"branch diff filter: {hit_branches}/{total_branches} branches hit")
        log(f"function diff filter: {hit_functions}/{total_functions} functions hit")
        print(f"INCREMENTAL_COVERAGE: {hit_lines}/{total_lines} lines hit ({pct:.1f}%)")
        print(f"INCREMENTAL_BRANCH_COVERAGE: {hit_branches}/{total_branches} branches hit ({branch_pct:.1f}%)")
        print(f"INCREMENTAL_FUNCTION_COVERAGE: {hit_functions}/{total_functions} functions hit ({func_pct:.1f}%)")

    def _generate_html(self) -> None:
        self.outputs.html_dir.mkdir(parents=True, exist_ok=True)
        log(f"generating html report at {self.outputs.html_dir / 'index.html'}")
        run_checked(
            [
                str(self.genhtml),
                "--output-directory",
                str(self.outputs.html_dir),
                "--legend",
                "--branch-coverage",
                "--ignore-errors",
                "source,inconsistent,unmapped,category",
                str(self.outputs.filtered_info),
            ]
        )

    def _print_outputs(self) -> None:
        log("done:")
        print(f"  capture:  {self.outputs.capture_info}")
        print(f"  extract:  {self.outputs.extracted_info}")
        print(f"  filtered: {self.outputs.filtered_info}")
        print(f"  html:     {self.outputs.html_dir / 'index.html'}")
        # Write incremental info file if diff filter was applied
        if self.common.diff_range is not None and self.outputs.filtered_info.exists():
            incremental_info = self.outputs.filtered_info.with_name("coverage.incremental.info")
            import shutil
            shutil.copy2(self.outputs.filtered_info, incremental_info)
            print(f"  incremental: {incremental_info}")


def build_lit_plan(args: argparse.Namespace, common: CommonOptions, repo_root: Path) -> CommandPlan:
    lit = common.build_dir / "bin" / "llvm-lit"
    if not lit.is_file() or not os.access(lit, os.X_OK):
        raise CoverageError(f"missing tool: {lit}")

    lit_test_root = existing_path(args.lit_test_root, repo_root)
    if lit_test_root is None or not lit_test_root.is_dir():
        raise CoverageError(f"lit test root not found: {args.lit_test_root}")

    targets: list[Path] = []
    if args.test:
        for test in args.test:
            resolved = existing_path(test, lit_test_root, repo_root)
            if resolved is None:
                raise CoverageError(f"test target not found: {test}")
            targets.append(resolved)
    else:
        targets.append(lit_test_root)

    command = (
        str(lit),
        "-sv",
        "-j",
        str(common.jobs),
        *(str(target) for target in targets),
    )
    return CommandPlan(
        mode="lit",
        cwd=common.build_dir,
        command=command,
        summary_lines=tuple(f"target: {target}" for target in targets),
    )


def build_gtest_plan(args: argparse.Namespace, common: CommonOptions, repo_root: Path) -> CommandPlan:
    binary = planned_path(
        args.binary,
        default_root=common.build_dir,
        extra_roots=(repo_root,),
    )
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise CoverageError(f"gtest binary not found or not executable: {binary}")

    command = [str(binary)]
    if args.gtest_filter:
        command.append(f"--gtest_filter={args.gtest_filter}")

    prepare_commands: list[tuple[str, ...]] = []
    if args.build_target:
        prepare_commands.append(
            (
                "ninja",
                "-C",
                str(common.build_dir),
                "-j",
                str(common.jobs),
                *args.build_target,
            )
        )

    summary = [f"binary: {binary}"]
    if args.build_target:
        summary.append(f"build target: {' '.join(args.build_target)}")
    if args.gtest_filter:
        summary.append(f"gtest filter: {args.gtest_filter}")

    return CommandPlan(
        mode="gtest",
        cwd=binary.parent,
        command=tuple(command),
        prepare_commands=tuple(prepare_commands),
        summary_lines=tuple(summary),
    )


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--build-dir", required=True, help="Build directory containing coverage-enabled binaries")
    parser.add_argument("--out-dir", default="", help="Coverage output directory (default: <repo-root>/coverage)")
    parser.add_argument("--jobs", type=positive_int, default=default_jobs())
    parser.add_argument(
        "--diff-range",
        default=None,
        help="Git diff range (e.g., 'main...HEAD') to filter coverage to changed files only",
    )
    parser.add_argument(
        "--llvm-cov-gcov-flags",
        default="",
        help="Extra flags to pass to llvm-cov gcov (e.g., '--show-branches=count')",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 build-tools/coverage/run.py",
        description="Run a gcov-compatible workload and generate lcov reports.",
    )
    subparsers = parser.add_subparsers(dest="runner", required=True)

    lit_parser = subparsers.add_parser("lit", help="Run llvm-lit based tests")
    add_common_options(lit_parser)
    lit_parser.add_argument("--lit-test-root", required=True)
    lit_parser.add_argument("--test", action="append", default=[])

    gtest_parser = subparsers.add_parser("gtest", help="Run a gtest binary")
    add_common_options(gtest_parser)
    gtest_parser.add_argument(
        "--binary",
        required=True,
        help="Path to the gtest executable, usually relative to --build-dir",
    )
    gtest_parser.add_argument(
        "--build-target",
        action="append",
        default=[],
        help="Optional ninja target to build before running the gtest binary",
    )
    gtest_parser.add_argument("--gtest-filter", default="")
    return parser


def main(argv: list[str]) -> int:
    try:
        repo_root = discover_repo_root(Path.cwd())
        args = build_parser().parse_args(argv)
        common = CommonOptions.from_namespace(args, repo_root)
        session = CoverageSession(common, repo_root)
        if args.runner == "lit":
            plan = build_lit_plan(args, common, repo_root)
        else:
            plan = build_gtest_plan(args, common, repo_root)
        session.run(plan)
        return 0
    except CoverageError as exc:
        warn(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
