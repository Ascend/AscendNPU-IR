#!/bin/bash
# This script is used to build the ascendnpuir Python wheel package.
#
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

SCRIPT_ROOT="$(dirname "$(realpath "$0")")"
GIT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
WHEEL_DIR="${GIT_ROOT}/bishengir/python/wheel"

# Default build directory
BUILD_DIR="${GIT_ROOT}/bishengir-output"

# Parse command options
usage() {
  echo -e "build_wheel.sh - Build the ascendnpuir Python wheel package.

    SYNOPSIS:
      build_wheel.sh [OPTIONS]

    Options:
      -b, --build-dir DIR    Path to the build directory containing bishengir-compile binary
                             (Default: ${BUILD_DIR})
      -h, --help             Print this help message
      -o, --output-dir DIR   Output directory for the wheel package
                             (Default: ${WHEEL_DIR}/dist)
      --clean                Clean the build artifacts before building
      --upload               Upload to PyPI after building (requires twine)

    Examples:
      # Build with default settings
      ./build_wheel.sh

      # Build with custom build directory
      ./build_wheel.sh --build-dir /path/to/build

      # Build and upload to PyPI
      ./build_wheel.sh --upload
      "
}

CLEAN_BUILD=false
UPLOAD_TO_PYPI=false
OUTPUT_DIR="${WHEEL_DIR}/dist"

while [[ $# -gt 0 ]]; do
  case $1 in
    -b|--build-dir)
      BUILD_DIR="$(realpath "$2")"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -o|--output-dir)
      OUTPUT_DIR="$(realpath "$2")"
      shift 2
      ;;
    --clean)
      CLEAN_BUILD=true
      shift
      ;;
    --upload)
      UPLOAD_TO_PYPI=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

# Check if build directory exists
if [ ! -d "${BUILD_DIR}" ]; then
  echo "Error: Build directory does not exist: ${BUILD_DIR}"
  echo "Please run the build.sh script first to build the bishengir-compile binary."
  exit 1
fi

# Check for bishengir-compile binary in bishengir-output directory
BINARY_NAME="bishengir-compile"
POSSIBLE_BINARY_PATHS=(
  "${BUILD_DIR}/bin/${BINARY_NAME}"
  "${BUILD_DIR}/${BINARY_NAME}"  # Fallback to old structure
)

# Check if the binary exists
BINARY_PATH=""
for path in "${POSSIBLE_BINARY_PATHS[@]}"; do
  if [ -f "${path}" ]; then
    BINARY_PATH="${path}"
    break
  fi
done

if [ -z "${BINARY_PATH}" ]; then
  echo "Error: Could not find ${BINARY_NAME} in expected locations:"
  for path in "${POSSIBLE_BINARY_PATHS[@]}"; do
    echo "  - ${path}"
  done
  echo "Please ensure the bishengir-compile binary has been built."
  exit 1
fi

# Print found binary
echo "Found bishengir-compile binary at: ${BINARY_PATH}"

# Navigate to wheel directory
cd "${WHEEL_DIR}"

# Clean build artifacts if requested
if [ "${CLEAN_BUILD}" = true ]; then
  echo "Cleaning build artifacts..."
  rm -rf build/ dist/ *.egg-info ascendnpuir.egg-info
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Set environment variable for setup.py
export BISHENGIR_BUILD_DIR="${BUILD_DIR}"

# Check if build module is available
if ! python3 -c "import build" 2>/dev/null; then
  echo "Installing build module..."
  python3 -m pip install --upgrade build
fi

# Build the wheel
echo "Building wheel package..."
python3 -m build --outdir "${OUTPUT_DIR}"

# Check if build was successful
if [ $? -eq 0 ]; then
  echo "Successfully built wheel package!"
  echo "Wheel package location: ${OUTPUT_DIR}"
  
  # List the built packages
  echo ""
  echo "Built packages:"
  ls -lh "${OUTPUT_DIR}"
  
  # Use auditwheel to automatically set the correct manylinux platform tag
  # This is the standard way to ensure binary compatibility
  echo ""
  echo "Setting platform tag with auditwheel (auto-detecting manylinux version)..."
  
  # Check if auditwheel is available
  if ! python3 -c "import auditwheel" 2>/dev/null; then
    echo "Installing auditwheel..."
    python3 -m pip install --upgrade auditwheel
  fi
  
  # Auto-detect the best manylinux platform for current environment
  MANYLINUX_PLATFORM=$(python3 -c "
import platform
import sys
import os

machine = platform.machine().lower()

if platform.system() != 'Linux':
    print('')
    sys.exit(0)

# Let's use auditwheel's own policies to find the best one
try:
    from auditwheel.policy import POLICIES
    if POLICIES:
        # Find the highest priority policy that starts with 'manylinux_'
        highest_priority = -1
        best_policy = None
        for policy in POLICIES:
            if policy['name'].startswith('manylinux_') and policy['priority'] > highest_priority:
                highest_priority = policy['priority']
                best_policy = policy
        if best_policy:
            print(best_policy['name'])
            sys.exit(0)
except Exception:
    pass

# Fallback: if we can't detect, use a reasonable default
# Check common patterns in manylinux containers
version = '2_28'  # Default for modern containers

# Final fallback
print(f'manylinux_{version}_{machine}')
")
  
  # Create directory for processed wheels
  PROCESSED_DIR="${OUTPUT_DIR}/processed"
  mkdir -p "${PROCESSED_DIR}"
  
  # Process the wheel with auditwheel
  for wheel_file in "${OUTPUT_DIR}"/*.whl; do
    if [ -f "${wheel_file}" ]; then
      echo "Processing: $(basename "${wheel_file}")"
      
      # Use auto-detected platform to force a single, clean tag
      if [ -n "${MANYLINUX_PLATFORM}" ]; then
        if python3 -m auditwheel repair "${wheel_file}" --plat "${MANYLINUX_PLATFORM}" -w "${PROCESSED_DIR}"; then
          echo "Successfully set platform tag to ${MANYLINUX_PLATFORM}!"
        else
          echo "Warning: auditwheel failed, keeping original wheel"
          cp "${wheel_file}" "${PROCESSED_DIR}/"
        fi
      else
        if python3 -m auditwheel repair "${wheel_file}" -w "${PROCESSED_DIR}"; then
          echo "Successfully set platform tag!"
        else
          echo "Warning: auditwheel failed, keeping original wheel"
          cp "${wheel_file}" "${PROCESSED_DIR}/"
        fi
      fi
    fi
  done
  
  # Replace original wheels with processed ones and ensure a single clean tag
  if [ -n "$(ls -A "${PROCESSED_DIR}")" ]; then
    echo ""
    echo "Using processed wheels:"
    ls -lh "${PROCESSED_DIR}"
    
    # Move and rename to ensure only one platform tag
    for wheel_file in "${PROCESSED_DIR}"/*.whl; do
      if [ -f "${wheel_file}" ]; then
        wheel_name=$(basename "${wheel_file}")
        
        # If we detected a specific platform tag, rename to use only that
        if [ -n "${MANYLINUX_PLATFORM}" ]; then
          # Replace any multiple tags with just our desired platform tag
          new_wheel_name=$(echo "${wheel_name}" | sed -E "s/-[^-]+\\.whl/-${MANYLINUX_PLATFORM}.whl/")
          mv "${wheel_file}" "${OUTPUT_DIR}/${new_wheel_name}"
        else
          mv "${wheel_file}" "${OUTPUT_DIR}/"
        fi
      fi
    done
  fi
  
  # Clean up processed directory
  rmdir "${PROCESSED_DIR}" 2>/dev/null || true
  
  echo ""
  echo "Final wheels:"
  ls -lh "${OUTPUT_DIR}"
  
  # Upload to PyPI if requested
  if [ "${UPLOAD_TO_PYPI}" = true ]; then
    echo ""
    echo "Uploading to PyPI..."
    
    # Check if twine is installed
    if ! python3 -c "import twine" 2>/dev/null; then
      echo "Installing twine..."
      python3 -m pip install --upgrade twine
    fi
    
    # Upload to PyPI
    python3 -m twine upload "${OUTPUT_DIR}"/*
    
    if [ $? -eq 0 ]; then
      echo "Successfully uploaded to PyPI!"
    else
      echo "Error: Failed to upload to PyPI"
      exit 1
    fi
  fi
else
  echo "Error: Failed to build wheel package"
  exit 1
fi

echo ""
echo "Build complete!"
