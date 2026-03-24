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

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist
import os
import shutil
import subprocess
import sys
from pathlib import Path


class CustomBuildPy(build_py):
    """Custom build command to copy the bishengir-compile binary."""
    
    def run(self):
        # Run the standard build_py first
        build_py.run(self)
        
        # Copy the bishengir-compile binary to the package
        self._copy_compiler_binary()
    
    def _copy_compiler_binary(self):
        """Copy all binaries and subdirectories from the build directory to the package bin directory."""
        # Determine the build directory
        build_dir = os.environ.get('BISHENGIR_BUILD_DIR')
        if not build_dir:
            # Try to find the build directory relative to the wheel directory
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            build_dir = project_root / "build"
        
        build_dir = Path(build_dir)
        
        # Look for the bin directory in common locations
        possible_bin_dirs = [
            build_dir,
            build_dir / "bin",
            build_dir / "install" / "bin",
        ]
        
        source_bin_dir = None
        for bin_dir in possible_bin_dirs:
            if bin_dir.exists() and bin_dir.is_dir():
                source_bin_dir = bin_dir
                break
        
        if not source_bin_dir:
            print(f"Warning: Could not find bin directory in expected locations:")
            for bin_dir in possible_bin_dirs:
                print(f"  - {bin_dir}")
            print("The package will be built without compiler binaries.")
            print("Set BISHENGIR_BUILD_DIR environment variable to specify the build directory.")
            return
        
        # Create the bin directory in the package
        package_bin_dir = Path(self.build_lib) / "ascendnpuir" / "bin"
        package_bin_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all files and subdirectories from source_bin_dir to package_bin_dir
        print(f"Copying all files and subdirectories from {source_bin_dir} to {package_bin_dir}")
        
        # First, remove any existing files in the package bin directory
        for item in package_bin_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        
        # Copy the entire bin directory structure
        for item in source_bin_dir.iterdir():
            dest_path = package_bin_dir / item.name
            if item.is_file():
                print(f"  Copying file: {item.name}")
                shutil.copy2(item, dest_path)
                # Make the binary executable on Unix-like systems
                if sys.platform != "win32":
                    os.chmod(dest_path, 0o755)
            elif item.is_dir():
                print(f"  Copying directory: {item.name}")
                shutil.copytree(item, dest_path, dirs_exist_ok=True)
                # Make all files in subdirectories executable on Unix-like systems
                if sys.platform != "win32":
                    for sub_item in dest_path.rglob('*'):
                        if sub_item.is_file():
                            os.chmod(sub_item, 0o755)


class CustomSdist(sdist):
    """Custom sdist command to ensure README is included."""
    
    def run(self):
        # Create README.md if it doesn't exist
        readme_path = Path(__file__).parent / "README.md"
        if not readme_path.exists():
            print("Creating README.md")
            readme_path.write_text("""# AscendNPU IR Compiler

Python bindings for the bishengir-compile compiler.

## Installation

bash
pip install ascendnpuir

## Usage

python
import ascendnpuir

# Compile a model
output = ascendnpuir.compile(
    "model.mlir",
    output_file="model.o",
    target="aarch64-unknown-linux-gnu",
    opt_level=2
)
print(f"Compiled to: {output}")

## License

Apache License 2.0
""")
        sdist.run(self)


setup(
    name="ascendnpuir",
    version="1.1",
    description="AscendNPU IR Compiler - Python bindings for bishengir-compile",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Huawei Technologies Co., Ltd.",
    author_email="support@huawei.com",
    url="https://github.com/AscendNPU/AscendNPU-IR",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.8",
    cmdclass={
        'build_py': CustomBuildPy,
        'sdist': CustomSdist,
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Compilers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="compiler npu ascend mlir ir",
    project_urls={
        "Homepage": "https://github.com/AscendNPU/AscendNPU-IR",
        "Documentation": "https://ascendnpu-ir.readthedocs.io",
        "Repository": "https://github.com/AscendNPU/AscendNPU-IR",
        "Bug-Tracker": "https://github.com/AscendNPU/AscendNPU-IR/issues",
    },
)