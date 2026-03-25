# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compiler module for AscendNPU IR.

This module provides the compile function that wraps the bishengir-compile binary.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, List, Union, NamedTuple


class CompileResult(NamedTuple):
    """Result of the compilation process.
    
    Attributes:
        returncode: Exit code of the compilation process.
        stdout: Standard output from the compilation process.
        stderr: Standard error from the compilation process.
        output_path: Path to the output file.
    """
    returncode: int
    stdout: str
    stderr: str
    output_path: str


def _get_compiler_path() -> Path:
    """Get the path to the bishengir-compile binary.
    
    Returns:
        Path to the bishengir-compile executable.
    
    Raises:
        FileNotFoundError: If the compiler binary cannot be found.
    """
    # Get the directory where this package is installed
    package_dir = Path(__file__).parent
    
    # Binary name - only need bishengir-compile
    binary_name = "bishengir-compile"
    
    # Look for the binary in the package's bin directory (new structure)
    binary_path = package_dir / "bin" / binary_name
    if binary_path.exists():
        return binary_path
    
    # Fallback to looking directly in the package directory (old structure)
    binary_path = package_dir / binary_name
    if binary_path.exists():
        return binary_path
    
    # If not found in package, try to find in PATH
    try:
        result = subprocess.run(
            ["which", "bishengir-compile"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            return Path(result.stdout.strip().split('\n')[0])
    except Exception:
        pass
    
    raise FileNotFoundError(
        f"Could not find bishengir-compile binary. "
        f"Expected location: {binary_path}"
    )


def _check_hivmc_available() -> bool:
    """Check if hivmc binary is available in PATH.
    
    bishengir-compile requires hivmc to generate binary kernel files.
    
    Returns:
        True if hivmc is available, False otherwise.
    """
    try:
        binary_name = "hivmc"
        result = subprocess.run(
            ["which", binary_name],
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0
    except Exception:
        return False


def compile(
    input: str,
    output_path: Optional[Union[str, Path]] = None,
    option: Optional[List[str]] = None
) -> CompileResult:
    """Compile MLIR string using the bishengir-compile compiler.
    
    This function wraps the bishengir-compile binary, which compiles MLIR code
    to binary kernel files for Ascend NPU. The compiler requires both
    bishengir-compile and hivmc binaries to be available in PATH.
    
    Args:
        input: MLIR string text to compile.
        output_path: Path to the output file. If not specified, the compiler
                     will use its default output location.
        option: List of compilation options to pass to the compiler.
                Example: ["-enable-hfusion-compile=true", "-enable-lir-compile=false"]
    
    Returns:
        A CompileResult object containing:
        - returncode: Exit code of the compilation process
        - stdout: Standard output from the compilation process
        - stderr: Standard error from the compilation process
        - output_path: Path to the output file
    
    Raises:
        FileNotFoundError: If the compiler binary or hivmc is not found.
        ValueError: If input is empty or None.
    
    Note:
        The bishengir-compile tool requires hivmc binary to be available in PATH
        for generating binary kernel files. Make sure both tools are installed
        and accessible before using this function.
    
    Example:
        >>> import ascendnpuir
        >>> mlir_text = '''
        ... module {
        ...   func.func @main() {
        ...     return
        ...   }
        ... }
        ... '''
        >>> result = ascendnpuir.compile(
        ...     mlir_text,
        ...     output_path="model.o",
        ...     option=["-enable-hfusion-compile=true", "-enable-lir-compile=false"]
        ... )
        >>> print(f"Compiled to: {result.output_path}")
        >>> print(f"Return code: {result.returncode}")
        >>> print(f"Stdout: {result.stdout}")
        >>> print(f"Stderr: {result.stderr}")
    """
    # Validate input
    if not input or not input.strip():
        raise ValueError("Input MLIR string cannot be empty")
    
    # Get the compiler path
    compiler_path = _get_compiler_path()
    
    # Check if hivmc is available (required for binary kernel generation)
    if not _check_hivmc_available():
        raise FileNotFoundError(
            "hivmc binary not found in PATH. "
            "bishengir-compile requires hivmc to generate binary kernel files. "
            "Please ensure hivmc is installed and accessible in your PATH."
        )
    
    # Build the command
    cmd = [str(compiler_path)]
    
    # Use stdin for input (pass "-" to indicate reading from stdin)
    cmd.append("-")
    
    # Add output file
    if output_path:
        cmd.extend(["-o", str(output_path)])
    
    # Add compilation options
    if option:
        cmd.extend(option)
    
    # Run the compiler with MLIR text as stdin
    result = subprocess.run(
        cmd,
        input=input,
        capture_output=True,
        text=True,
        check=False
    )
    
    # Determine the output file path
    if output_path:
        final_output_path = str(output_path)
    else:
        # If no output file was specified, return a default path
        # The compiler typically generates output in the current directory
        final_output_path = "a.out"
    
    return CompileResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        output_path=final_output_path
    )


__all__ = ["compile", "CompileResult"]