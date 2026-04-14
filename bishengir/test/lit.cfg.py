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
config.name = 'bishengir'

# use_lit_shell may be None in some lit versions; use getattr for compatibility.
use_lit_shell = getattr(llvm_config, 'use_lit_shell', False)
config.test_format = lit.formats.ShTest(not use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.td', '.mlir', '.toy', '.ll',
                   '.tc', '.py', '.yaml', '.test', '.pdll', '.c']

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))
config.substitutions.append(("%host_cxx", config.host_cxx))
config.substitutions.append(("%host_cc", config.host_cc))
config.substitutions.append(("%bishengir_src_root", config.bishengir_src_root))

llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt',
                   'lit.cfg.py', 'lit.site.cfg.py']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.mlir_obj_root, 'test')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.mlir_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)
llvm_config.with_environment(
    'PATH', config.bishengir_tools_dir, append_path=True)

tool_dirs = [config.mlir_tools_dir,
             config.llvm_tools_dir, config.bishengir_tools_dir]
tools = [
    'bishengir-opt',
    "bishengir-options-tblgen",
    'bishengir-target-spec-tblgen',
    "bishengir-capi-ir-test",
    "bishengir-capi-pass-test",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)


# FileCheck -enable-var-scope is enabled by default in MLIR test
# This option avoids to accidentally reuse variable across -LABEL match,
# it can be explicitly opted-in by prefixing the variable name with $
config.environment['FILECHECK_OPTS'] = "-enable-var-scope --allow-unused-prefixes=false"

if config.enable_assertions:
    config.available_features.add('asserts')
else:
    config.available_features.add('noasserts')

if config.enable_execution_engine:
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

if ('bisheng' in config.bisheng_compiler_executable and os.path.isfile(config.bisheng_compiler_executable)):
    config.available_features.add('enable-lir-compile')
    llvm_config.with_environment('BISHENG_INSTALL_PATH', os.path.dirname(
        config.bisheng_compiler_executable), append_path=True)

if shutil.which('hivmc'):
    config.available_features.add('hivmc')