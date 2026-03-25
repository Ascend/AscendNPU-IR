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
from wheel.bdist_wheel import bdist_wheel
import os
import shutil
import subprocess
import sys
import platform
from pathlib import Path


class CustomBuildPy(build_py):
    """Custom build command to copy the bishengir-compile binary."""
    
    def run(self):
        # Run the standard build_py first
        build_py.run(self)
        
        # Copy the bishengir-compile binary to the package
        self._copy_compiler_binary()
    
    def _copy_compiler_binary(self):
        """Copy all files and directories from the bishengir-output directory to the package."""
        # Determine the build directory
        build_dir = os.environ.get('BISHENGIR_BUILD_DIR')
        if not build_dir:
            # Try to find the build directory relative to the wheel directory
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            build_dir = project_root / "bishengir-output"  # Default to bishengir-output
        
        build_dir = Path(build_dir)
        
        # Check if build directory exists
        if not build_dir.exists() or not build_dir.is_dir():
            print(f"Warning: bishengir-output directory not found at: {build_dir}")
            print("The package will be built without compiler binaries.")
            print("Set BISHENGIR_BUILD_DIR environment variable to specify the build directory.")
            return
        
        # Create the destination directory in the package
        package_dir = Path(self.build_lib) / "ascendnpuir"
        
        # Copy all files and subdirectories from build_dir to package_dir
        print(f"Copying all files and directories from {build_dir} to {package_dir}")
        
        # Copy each top-level item in bishengir-output
        for item in build_dir.iterdir():
            dest_path = package_dir / item.name
            
            # Remove existing item if it exists
            if dest_path.exists():
                if dest_path.is_file():
                    dest_path.unlink()
                elif dest_path.is_dir():
                    shutil.rmtree(dest_path)
            
            # Copy the item
            if item.is_file():
                print(f"  Copying file: {item.name}")
                shutil.copy2(item, dest_path)
                # Make the binary executable on Unix-like systems
                if sys.platform != "win32" and item.name.startswith("bishengir-"):
                    os.chmod(dest_path, 0o755)
            elif item.is_dir():
                print(f"  Copying directory: {item.name}")
                shutil.copytree(item, dest_path, dirs_exist_ok=True)
                # Make all executable files in subdirectories executable on Unix-like systems
                if sys.platform != "win32":
                    for sub_item in dest_path.rglob('*'):
                        if sub_item.is_file() and sub_item.name.startswith("bishengir-"):
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


class CustomBdistWheel(bdist_wheel):
    """Custom bdist_wheel command to set platform-specific tags."""
    
    def get_platform(self):
        """Get the platform tag for the wheel."""
        # Get the platform information
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        # Map machine names to wheel platform tags
        machine_map = {
            'x86_64': 'x86_64',
            'amd64': 'x86_64',
            'aarch64': 'aarch64',
            'arm64': 'aarch64',
            'armv7l': 'armv7l',
        }
        
        # Get the normalized machine name
        normalized_machine = machine_map.get(machine, machine)
        
        # Build the platform tag based on the OS
        if system == 'linux':
            # For Linux, use manylinux tag for better compatibility
            # We'll use manylinux2014 for glibc 2.17+
            # Check if we're in a manylinux environment
            if os.path.exists('/etc/centos-release') or os.path.exists('/etc/redhat-release'):
                # Assume manylinux2014 compatible
                platform_tag = f'manylinux2014_{normalized_machine}'
            else:
                # Use linux tag with architecture
                platform_tag = f'linux_{normalized_machine}'
        elif system == 'darwin':
            # macOS
            platform_tag = f'macosx_{platform.mac_ver()[0].replace(".", "_")}_{normalized_machine}'
        elif system == 'windows':
            # Windows
            platform_tag = f'win_{normalized_machine}'
        else:
            # Fallback to generic platform
            platform_tag = f'{system}_{normalized_machine}'
        
        return platform_tag
    
    def finalize_options(self):
        bdist_wheel.finalize_options(self)
        # Set the platform tag
        self.plat_name = self.get_platform()
        # Mark this as a platform-specific wheel (not pure Python)
        self.root_is_pure = False


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
        'bdist_wheel': CustomBdistWheel,
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