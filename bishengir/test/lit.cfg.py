# -*- Python -*-

import os
import platform
import re
import shutil
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.
# NOTE: This file must be loaded via lit.site.cfg.py (generated in build dir).
# Run tests from the build directory, e.g.:
#   cd build && python bin/llvm-lit test/bishengir-compile/commandline.mlir

if llvm_config is None:
    raise SystemExit(
        "lit.llvm.initialize() was not called. Run lit from the build "
        "directory so that lit.site.cfg.py is loaded first, e.g.:\n"
        "  cd build && python bin/llvm-lit test/bishengir-compile/commandline.mlir"
    )

# name: The name of this test suite.
config.name = "bishengir"

# Host architecture feature for architecture-specific test gating.
if platform.machine().lower() in ("aarch64", "arm64"):
    config.available_features.add("host-aarch64")

# use_lit_shell may be None in some lit versions; use getattr for compatibility.
use_lit_shell = getattr(llvm_config, "use_lit_shell", False)
config.test_format = lit.formats.ShTest(not use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [
    ".td",
    ".mlir",
    ".toy",
    ".ll",
    ".tc",
    ".py",
    ".yaml",
    ".test",
    ".pdll",
    ".c",
]

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%host_cxx", config.host_cxx))
config.substitutions.append(("%host_cc", config.host_cc))
config.substitutions.append(("%bishengir_src_root", config.bishengir_src_root))
config.substitutions.append(("%mlir_src_root", config.mlir_src_root))
config.substitutions.append(("%llvm_src_root", config.llvm_src_root))
config.substitutions.append(("%bishengir_lib_dir", config.llvm_shlib_dir))
config.substitutions.append(("%bisheng_compiler", config.bisheng_compiler_executable))
config.substitutions.append(("%shmem_rdma_backend", config.shmem_rdma_backend))
config.substitutions.append(("%shmem_relay_support", config.shmem_relay_support))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    "Inputs",
    "CMakeLists.txt",
    "README.txt",
    "LICENSE.txt",
    "lit.cfg.py",
    "lit.site.cfg.py",
]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.mlir_obj_root, "test")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.mlir_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.bishengir_tools_dir, append_path=True)

tool_dirs = [config.mlir_tools_dir, config.llvm_tools_dir, config.bishengir_tools_dir]


def find_runtime(name):
    for prefix in ["", "lib"]:
        path = os.path.join(
            config.llvm_shlib_dir, f"{prefix}{name}{config.llvm_shlib_ext}"
        )
        if os.path.isfile(path):
            return path
    return ""


# Searches for a runtime library with the given name and returns a tool
# substitution of the same name and the found path.
def add_runtime(name):
    return ToolSubst(f"%{name}", find_runtime(name))


tools = [
    "bishengir-compile",
    "bishengir-opt",
    "bishengir-options-tblgen",
    "bishengir-target-spec-tblgen",
    "bishengir-capi-ir-test",
    "bishengir-capi-pass-test",
    "mlir-cpu-runner",
    add_runtime("mlir_runner_utils"),
    add_runtime("mlir_c_runner_utils"),
]

llvm_config.add_tool_substitutions(tools, tool_dirs)


# FileCheck -enable-var-scope is enabled by default in MLIR test
# This option avoids to accidentally reuse variable across -LABEL match,
# it can be explicitly opted-in by prefixing the variable name with $
config.environment["FILECHECK_OPTS"] = "-enable-var-scope --allow-unused-prefixes=false"

if config.enable_assertions:
    config.available_features.add("asserts")
else:
    config.available_features.add("noasserts")


# The execution-engine feature additionally requires the mlir-cpu-runner tool
# and the runner runtime libraries to be present in the build tree, so tests
# using them are skipped (instead of failing) in builds where they are absent.
def find_tool(name):
    for d in tool_dirs:
        if not d:
            continue
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return shutil.which(name)


if (
    config.enable_execution_engine
    and find_tool("mlir-cpu-runner")
    and find_runtime("mlir_runner_utils")
    and find_runtime("mlir_c_runner_utils")
):
    config.available_features.add("execution-engine")

if config.bishengir_published:
    config.available_features.add("bishengir_published")

if config.enable_bindings_python:
    llvm_config.with_environment(
        "PYTHONPATH",
        [
            os.path.join(config.bishengir_obj_root, "python_packages", "bishengir"),
        ],
        append_path=True,
    )

if "bisheng" in config.bisheng_compiler_executable and os.path.isfile(
    config.bisheng_compiler_executable
):
    config.available_features.add("enable-lir-compile")
    llvm_config.with_environment(
        "BISHENG_INSTALL_PATH",
        os.path.dirname(config.bisheng_compiler_executable),
        append_path=True,
    )

# Detect whether the build compiled libbisheng-llc support (BISHENGIR_HAVE_BISHENG_LLC=1)
# AND the vendor shared library is reachable AND the host glibc is new enough for the
# dual-namespace dlmopen the Native path uses.  When all hold, tests can exercise the
# in-process Native backend (BISHENG_BACKEND=native) in addition to the default
# external-process backend.
#
# This mirrors the C++ gate in bishengir-compile.cpp (runBishengirCompileA5) and
# BishengLlcBridge.cpp so that the lit feature is registered if-and-only-if the
# driver would actually honour BISHENG_BACKEND=native instead of silently falling
# back to Process mode.  The driver's conditions for staying Native are:
#
#   1. BISHENGIR_HAVE_BISHENG_LLC == 1  (compile-time macro)
#   2. No debug/sanitizer options        (not controllable from lit, always off)
#   3. glibc >= 2.35 OR BISHENG_NATIVE_DLMOPEN_FORCE is set
#   4. resolveBishengLlcLibrary() returns a path that exists on disk
#
# Condition 2 is always satisfied for these tests (they pass none of those flags),
# so only 1, 3, and 4 are checked here.

# --- Condition 1: BISHENGIR_HAVE_BISHENG_LLC ---
_bisheng_llc_config_h = os.path.join(
    config.bishengir_obj_root,
    "bishengir",
    "include",
    "bishengir",
    "Tools",
    "hivmc",
    "BishengLlcConfig.h",
)
_bisheng_have_llc = False
if os.path.isfile(_bisheng_llc_config_h):
    with open(_bisheng_llc_config_h) as _f:
        if "BISHENGIR_HAVE_BISHENG_LLC 1" in _f.read():
            _bisheng_have_llc = True


# --- Condition 3: glibc >= 2.35 or BISHENG_NATIVE_DLMOPEN_FORCE ---
# Mirrors getRuntimeGlibcVersion() in bishengir-compile.cpp: query
# gnu_get_libc_version() at RUNTIME (not __GLIBC__ compile-time macros).
# The C++ uses llvm::VersionTuple::tryParse which accepts "M", "M.m", or
# "M.m.p"; a parse failure is treated as "too old" (conservative).
def _parse_glibc_version():
    """Return a (major, minor) tuple or None if it cannot be determined.

    Mirrors the C++ getRuntimeGlibcVersion() + VersionTuple::tryParse:
    returns None on non-glibc platforms or when the version string cannot
    be parsed, matching the C++ ``std::nullopt`` outcome.
    """
    try:
        import ctypes
    except ImportError:
        return None
    try:
        libc = ctypes.CDLL("libc.so.6")
    except OSError:
        return None  # non-glibc platform
    try:
        getversion = libc.gnu_get_libc_version
        getversion.restype = ctypes.c_char_p
        ver_str = getversion().decode("ascii")  # e.g. "2.35" or "2.35.1"
        parts = ver_str.split(".")
        # VersionTuple requires at least major; minor/patch default to 0.
        # A parse failure (non-numeric) maps to std::nullopt in C++.
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return (major, minor)
    except Exception:
        return None


# BISHENG_NATIVE_DLMOPEN_FORCE: getenv() in C returns non-NULL for an empty
# string, but os.environ.get() in Python returns "" which is falsy.  Use
# "in os.environ" to match the C++ getenv semantics exactly.
if "BISHENG_NATIVE_DLMOPEN_FORCE" in os.environ:
    _glibc_ok = True
else:
    _glibc_ver = _parse_glibc_version()
    # C++: glibcTooOld = !glibcVersion || *glibcVersion < (2, 35)
    _glibc_ok = _glibc_ver is not None and _glibc_ver >= (2, 35)


# --- Condition 4: resolveBishengLlcLibrary() ---
# Mirrors resolveBishengLlcLibrary() + defaultBishengLlcLibraryPath() +
# getBiShengCompilerInstallPath() in BishengLlcBridge.cpp / Utils.cpp.
#
# resolveBishengLlcLibrary():
#   1. BISHENG_LLC_LIBRARY env var (explicit override, highest priority)
#   2. defaultBishengLlcLibraryPath():
#      getBiShengCompilerInstallPath() resolves the bisheng bin directory:
#        a. BISHENG_INSTALL_PATH env var  (already set by lit.cfg above)
#        b. PATH lookup of "bisheng" binary
#        c. ASCEND_HOME_PATH/bin
#      Then: <bisheng-real-bin>/../lib/libbisheng-llc.so
#      (symlink-resolved, matching the C++ llvm::sys::fs::real_path)
def _resolve_bisheng_llc_library():
    """Return the resolved libbisheng-llc.so path, or '' if not found.

    Mirrors the C++ resolveBishengLlcLibrary() → defaultBishengLlcLibraryPath()
    → getBiShengCompilerInstallPath() chain.
    """
    # 1. BISHENG_LLC_LIBRARY env var
    _env = os.environ.get("BISHENG_LLC_LIBRARY", "")
    if _env and os.path.isfile(_env):
        return os.path.realpath(_env)

    # 2. getBiShengCompilerInstallPath(): resolve the bisheng bin directory
    _bisheng_bin = ""
    # 2a. BISHENG_INSTALL_PATH
    _install = os.environ.get("BISHENG_INSTALL_PATH", "")
    if _install:
        _bisheng_bin = os.path.abspath(_install)
    # 2b. PATH lookup of "bisheng"
    if not _bisheng_bin:
        _bisheng_exe = shutil.which("bisheng")
        if _bisheng_exe:
            _bisheng_bin = os.path.dirname(_bisheng_exe)
    # 2c. ASCEND_HOME_PATH/bin
    if not _bisheng_bin:
        _ascend = os.environ.get("ASCEND_HOME_PATH", "")
        if _ascend:
            _candidate_bin = os.path.join(_ascend, "bin")
            if os.path.isfile(os.path.join(_candidate_bin, "bisheng")):
                _bisheng_bin = _candidate_bin

    if not _bisheng_bin:
        return ""

    # defaultBishengLlcLibraryPath(): resolve symlink of bisheng binary,
    # then take <real-bin>/../lib/libbisheng-llc.so
    _bisheng_file = os.path.join(_bisheng_bin, "bisheng")
    _real_bin = os.path.realpath(_bisheng_file)
    if _real_bin and os.path.isfile(_real_bin):
        _bisheng_bin = os.path.dirname(_real_bin)
    _so_candidate = os.path.join(_bisheng_bin, "..", "lib", "libbisheng-llc.so")
    if os.path.isfile(_so_candidate):
        return os.path.realpath(_so_candidate)
    return ""


if _bisheng_have_llc and _glibc_ok:
    _bisheng_llc_so = _resolve_bisheng_llc_library()
    if _bisheng_llc_so:
        config.available_features.add("bisheng-native")
        llvm_config.with_environment(
            "BISHENG_LLC_LIBRARY",
            _bisheng_llc_so,
            append_path=False,
        )

if config.bishengir_enable_triton_compile:
    config.available_features.add("enable_triton_ir_compiless")

if shutil.which("hivmc"):
    config.available_features.add("hivmc")

if config.shmem_build_template:
    config.available_features.add("shmem")

# Reading all linked SHMEM template variants is expensive. Run this ABI audit
# only when requested, independently of the normal SHMEM compile tests.
if lit_config.params.get("shmem-transport-wrappers", "0") == "1":
    config.available_features.add("shmem-transport-wrappers")
