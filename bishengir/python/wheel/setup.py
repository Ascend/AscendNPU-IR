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

from setuptools import setup, Extension
from setuptools.command.build_py import build_py
import os
import shutil
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


# Standard setuptools way - no CustomBdistWheel
# This uses a minimal C extension to automatically generate platform-specific tags
setup(
    cmdclass={
        'build_py': CustomBuildPy,
    },
    ext_modules=[
        Extension(
            'ascendnpuir._placeholder',
            sources=['ascendnpuir/_placeholder.c'],
        )
    ],
)
