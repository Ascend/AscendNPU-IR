#!/bin/bash
# This script is used to build the bishengir project.
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

set -e  # Exit immediately on error

# On macOS, re-exec with Homebrew Bash if system Bash < 4.2 (Linux/macOS compatible)
OS_TYPE_early=$(uname -s)
if [ "$OS_TYPE_early" = "Darwin" ]; then
  need_reexec=0
  if [ -z "${BASH_VERSINFO[0]:-}" ]; then need_reexec=1; fi
  if [ "${BASH_VERSINFO[0]:-0}" -lt 4 ]; then need_reexec=1; fi
  if [ "${BASH_VERSINFO[0]:-0}" -eq 4 ] && [ "${BASH_VERSINFO[1]:-0}" -lt 2 ]; then need_reexec=1; fi
  if [ "$need_reexec" = "1" ]; then
    for bash_candidate in /opt/homebrew/bin/bash /usr/local/bin/bash; do
      if [ -x "$bash_candidate" ]; then
        exec "$bash_candidate" "$0" "$@"
      fi
    done
  fi
fi

# Get script directory (macOS-compatible when invoked via symlink)
get_script_root() {
    local source="${BASH_SOURCE[0]}"
    local dir
    while [[ -L "$source" ]]; do
        dir="$(cd -P "$(dirname "$source")" && pwd)"
        source="$(readlink "$source")"
        [[ "$source" != /* ]] && source="$dir/$source"
    done
    dir="$(cd -P "$(dirname "$source")" && pwd)"
    echo "$dir"
}

if [ -z "${BASH_VERSINFO:-}" ]; then
    echo "Bash not found " >&2
    exit 1
fi

if [ "${BASH_VERSINFO[0]}" -lt 4 ] || \
   [ "${BASH_VERSINFO[0]}" -eq 4 -a "${BASH_VERSINFO[1]}" -lt 2 ]; then
    echo "Require Bash version 4.2 or higher version" >&2
    echo "Current Bash version is ${BASH_VERSION}" >&2
    exit 1
fi

OS_TYPE=$(uname -s)
GIT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
if [ -z "$GIT_ROOT" ]; then
    echo "Error: Not in a git repository" >&2
    exit 1
fi

THIRD_PARTY_FOLDER="$GIT_ROOT/third-party"
SCRIPT_ROOT="$(get_script_root)"
readonly SCRIPT_NAME="$(basename "$0")"
readonly ENABLE_PROJECTS="mlir"

# Initialize variables
BUILD_TYPE="Release"
# Default compiler by OS: system clang on macOS to avoid LLD/libc++ conflicts; clang on Linux
if [[ "$OS_TYPE" == "Darwin" ]]; then
  if [ -x "/usr/bin/clang" ] && [ -x "/usr/bin/clang++" ]; then
    C_COMPILER="/usr/bin/clang"
    CXX_COMPILER="/usr/bin/clang++"
  else
    C_COMPILER="clang"
    CXX_COMPILER="clang++"
  fi
else
  C_COMPILER="clang"
  CXX_COMPILER="clang++"
fi
BUILD_DIR="${GIT_ROOT}/build"
BUILD_SCRIPTS=(
  "apply_patches.sh"
  "patches"
  "build.sh"
)
BUILD_BISHENGIR_DOC="OFF"
LLVM_SOURCE_DIR="$THIRD_PARTY_FOLDER/llvm-project"
TORCH_MLIR_SOURCE_DIR="$THIRD_PARTY_FOLDER/torch-mlir"
BISHENGIR_SOURCE_DIR="$GIT_ROOT"
ENABLE_ASSERTION="OFF"
PYTHON_BINDING="OFF"
BUILD_TORCH_MLIR="OFF"
CCACHE_BUILD="ON"
SAFETY_OPTIONS=""
SAFETY_LD_OPTIONS=""
BISHENGIR_PUBLISH="ON"
LLVM_BUILD_TARGETS="host"
BISHENGIR_BUILD_TEMPLATE="OFF"
BISHENG_COMPILER=""
APPLY_PATCHES=""
BUILD_TEST=""
NO_INSTALL=""
REBUILD=""
SKIP_RPATH_OPTION="FALSE"
CMAKE_OPTIONS=""
INSTALL_PREFIX=""

# Thread count: 3/4 of CPU cores (fallback to 1 if detection fails)
if [[ "$OS_TYPE" == "Darwin" ]]; then
  NCPU=$(sysctl -n hw.ncpu 2>/dev/null) || NCPU=1
else
  NCPU=$(grep -c "processor" /proc/cpuinfo 2>/dev/null) || NCPU=1
fi
THREADS=$((NCPU * 3 / 4))
(( THREADS > 1 )) || THREADS=1

# Help message
usage() {
  echo -e "${SCRIPT_NAME} - Build the BiShengIR project.

    SYNOPSIS:
      ${SCRIPT_NAME}
                [--add-cmake-options CMAKE_OPTIONS]
                [--apply-patches]
                [--bisheng-compiler BISHENG_COMPILER]
                [--bishengir-publish VALUE]
                [-o | --build PATH]
                [-t | --build-bishengir-template]
                [--build-bishengir-doc]
                [--build-test]
                [--build-type BUILD_TYPE]
                [--build-torch-mlir]
                [--c-compiler C_COMPILER] [--cxx-compiler CXX_COMPILER]
                [--disable-ccache]
                [--fast-build]
                [-h | --help]
                [--install-prefix INSTALL_PREFIX]
                [-j | --jobs JOBS]
                [--llvm-source-dir DIR]
                [--python-binding]
                [-r | --rebuild]
                [--safety-options]
                [--safety-ld-options]
                [--skip-rpath]
                [--enable-cpu-runner]

    Options:
      --add-cmake-options CMAKE_OPTIONS    Add options to CMake; use quotes for multiple, e.g. --add-cmake-options '-DFOO=ON -DBAR=1'. (Default: null)
      --apply-patches                      Apply patches to third-party submodules. (Default: disabled)
      --bisheng-compiler BISHENG_COMPILER   Path to bisheng compiler. (Default: null)
      --bishengir-publish VALUE            Whether to disable features is currently unpublished. (Default: ON)
      -o, --build PATH                      Path to directory which CMake will use as the root of build directory
                                           (Default: build)
      --build-bishengir-doc                Whether to build BiShengIR documentation. (Default: disabled)
      -t, --build-bishengir-template       Whether to build BiShengIR template. (Default: disabled)
      -build-test                          Whether to build bishengir-test (Default: disabled)
      --build-type BUILD_TYPE              Specifies the build type. (Default: Release)
      --build-torch-mlir                   Whether to build torch-mlir. (Default: disabled)
      --c-compiler C_COMPILER              The full path to the compiler for C (Default: clang)
      --cxx-compiler CXX_COMPILER          The full path to the compiler for C++ (Default: clang++)
      --disable-ccache                     Disable ccache to build toolchain. (Default: disabled)
      --fast-build                         Skip the installation. (Default: disabled)
      -h, --help                           Print this help message.
      --install-prefix INSTALL_PREFIX      CMake install prefix. (Default: BUILD_DIR/install)
      -j, --jobs JOBS                      Set the threads when building
                                           (Default: use 3/4 of processing units)
      --llvm-source-dir DIR                LLVM project's root directory. (Default: 'third-party/llvm-project')
      --python-binding                     Whether to enable MLIR Python Binding (Default: disabled)
      -r, --rebuild                        Rebuild (Default: disabled)
      --safety-options                     Whether to build with safe compile options. (Default: disabled)
      --safety-ld-options                  Whether to build with safe options for linking. (Default: disabled)
      --skip-rpath                         Disable the Run-time Search Path option. (Default: disabled)
      --torch-mlir-source-dir DIR          Torch-MLIR project's root directory. (Default: 'third-party/torch-mlir')
      --enable-cpu-runner                  Enable the compilation of CPU runner targets
      "
}

# Parse arguments manually (no getopt dependency)
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --add-cmake-options)
                if [[ -z "${2+x}" ]]; then
                    echo "Error: --add-cmake-options requires an argument"
                    exit 1
                fi
                CMAKE_OPTIONS+=" $2"
                shift 2
                ;;
            --add-cmake-options=*)
                CMAKE_OPTIONS+=" ${1#--add-cmake-options=}"
                shift
                ;;
            --apply-patches)
                APPLY_PATCHES="1"
                shift
                ;;
            --bisheng-compiler)
                if [[ -z "$2" || "$2" == -* ]]; then
                    echo "Error: --bisheng-compiler requires an argument"
                    exit 1
                fi
                BISHENG_COMPILER="$2"
                shift 2
                ;;
            --bisheng-compiler=*)
                BISHENG_COMPILER="${1#--bisheng-compiler=}"
                shift
                ;;
            --bishengir-publish)
                if [[ -z "$2" || "$2" == -* ]]; then
                    echo "Error: --bishengir-publish requires an argument"
                    exit 1
                fi
                BISHENGIR_PUBLISH="$2"
                shift 2
                ;;
            --bishengir-publish=*)
                BISHENGIR_PUBLISH="${1#--bishengir-publish=}"
                shift
                ;;
            -o|--build)
                if [[ -z "$2" || "$2" == -* ]]; then
                    echo "Error: --build requires a path argument"
                    exit 1
                fi
                BUILD_DIR="$2"
                shift 2
                ;;
            --build=*)
                BUILD_DIR="${1#--build=}"
                shift
                ;;
            -t|--build-bishengir-template)
                BISHENGIR_BUILD_TEMPLATE="ON"
                shift
                ;;
            --build-bishengir-template=*)
                BISHENGIR_BUILD_TEMPLATE="${1#--build-bishengir-template=}"
                shift
                ;;
            --build-bishengir-doc)
                BUILD_BISHENGIR_DOC="ON"
                shift
                ;;
            --build-test)
                BUILD_TEST="1"
                shift
                ;;
            --build-type)
                if [[ -z "$2" || "$2" == -* ]]; then
                    echo "Error: --build-type requires an argument"
                    exit 1
                fi
                BUILD_TYPE="$2"
                shift 2
                ;;
            --build-type=*)
                BUILD_TYPE="${1#--build-type=}"
                shift
                ;;
            --build-torch-mlir)
                BUILD_TORCH_MLIR="ON"
                shift
                ;;
            --c-compiler)
                if [[ -z "$2" || "$2" == -* ]]; then
                    echo "Error: --c-compiler requires an argument"
                    exit 1
                fi
                C_COMPILER="$2"
                shift 2
                ;;
            --c-compiler=*)
                C_COMPILER="${1#--c-compiler=}"
                shift
                ;;
            --cxx-compiler)
                if [[ -z "$2" || "$2" == -* ]]; then
                    echo "Error: --cxx-compiler requires an argument"
                    exit 1
                fi
                CXX_COMPILER="$2"
                shift 2
                ;;
            --cxx-compiler=*)
                CXX_COMPILER="${1#--cxx-compiler=}"
                shift
                ;;
            --disable-ccache)
                CCACHE_BUILD="OFF"
                shift
                ;;
            --enable-assertion)
                ENABLE_ASSERTION="ON"
                shift
                ;;
            --fast-build)
                NO_INSTALL="1"
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            --install-prefix)
                if [[ -z "$2" || "$2" == -* ]]; then
                    echo "Error: --install-prefix requires an argument"
                    exit 1
                fi
                INSTALL_PREFIX="$2"
                shift 2
                ;;
            --install-prefix=*)
                INSTALL_PREFIX="${1#--install-prefix=}"
                shift
                ;;
            -j|--jobs)
                if [[ -z "$2" || "$2" == -* ]]; then
                    echo "Error: --jobs requires an argument"
                    exit 1
                fi
                THREADS="$2"
                shift 2
                ;;
            --jobs=*)
                THREADS="${1#--jobs=}"
                shift
                ;;
            --llvm-source-dir)
                if [[ -z "$2" || "$2" == -* ]]; then
                    echo "Error: --llvm-source-dir requires an argument"
                    exit 1
                fi
                LLVM_SOURCE_DIR="$2"
                shift 2
                ;;
            --llvm-source-dir=*)
                LLVM_SOURCE_DIR="${1#--llvm-source-dir=}"
                shift
                ;;
            --python-binding)
                PYTHON_BINDING="ON"
                shift
                ;;
            -r|--rebuild)
                REBUILD="1"
                shift
                ;;
            --safety-options)
                SAFETY_OPTIONS="-fPIC -fstack-protector-strong"
                shift
                ;;
            --safety-ld-options)
                SAFETY_LD_OPTIONS="-Wl,-z,relro,-z,now"
                shift
                ;;
            --skip-rpath)
                SKIP_RPATH_OPTION="TRUE"
                shift
                ;;
            --torch-mlir-source-dir)
                if [[ -z "$2" || "$2" == -* ]]; then
                    echo "Error: --torch-mlir-source-dir requires an argument"
                    exit 1
                fi
                TORCH_MLIR_SOURCE_DIR="$2"
                shift 2
                ;;
            --torch-mlir-source-dir=*)
                TORCH_MLIR_SOURCE_DIR="${1#--torch-mlir-source-dir=}"
                shift
                ;;
            --enable-cpu-runner)
                LLVM_BUILD_TARGETS+=";Native"
                shift
                ;;
            --)
                shift
                break
                ;;
            -*)
                echo "Error: Unknown option: $1"
                usage
                exit 1
                ;;
            *)
                echo "Error: Unexpected argument: $1"
                usage
                exit 1
                ;;
        esac
    done
}

parse_arguments "$@"

# Check required tools and environment
check_dependencies() {
    echo "Checking dependencies..."
    
    if ! command -v cmake >/dev/null 2>&1; then
        echo "CMake not found. Please install CMake >= 3.28"
        exit 1
    fi
    
    if ! command -v ninja >/dev/null 2>&1; then
        echo "Ninja not found. Please install Ninja"
        exit 1
    fi
    
    if [ "$OS_TYPE" = "Darwin" ]; then
        if ! xcode-select -p >/dev/null 2>&1; then
            echo "Xcode Command Line Tools not found. Please install with: xcode-select --install"
            exit 1
        fi
        
        # GNU patch (gpatch) needed for --merge on macOS
        if [[ -n "$APPLY_PATCHES" ]] && ! command -v gpatch >/dev/null 2>&1; then
            echo "Warning: gpatch not found. If applying patches fails, install with: brew install gpatch"
        fi
    fi
    
    if [ "$OS_TYPE" = "Linux" ]; then
        if ! command -v g++ >/dev/null 2>&1; then
            echo "build-essential not found. Please install with: sudo apt install build-essential"
            exit 1
        fi
    fi

    echo "All dependencies satisfied."
}

clean_build_dir() {
  if [[ "${BUILD_DIR}" = "${SCRIPT_ROOT}" ]]; then
    find "${BUILD_DIR}" -mindepth 1 -maxdepth 1 \
      $(printf -- "-not -name %s " "${BUILD_SCRIPTS[@]}") \
      -exec rm -rf {} +
  else
    [[ -n "${BUILD_DIR}" ]] && rm -rf "${BUILD_DIR}"/CMake*
    mkdir -p "${BUILD_DIR}"
  fi
}

if [[ -z "${INSTALL_PREFIX}" ]]; then
  readonly INSTALL_PREFIX="${BUILD_DIR}/install"
fi

cmake_generate() {
  local torch_mlir_option=""
  local enable_external_projects="bishengir"

  if [ "${BISHENGIR_BUILD_TEMPLATE}" = "ON" ]; then
    if [ ! -d "$BISHENG_COMPILER" ]; then
        echo "Path to bisheng compiler "$BISHENG_COMPILER" does not exist"
      else
        export BISHENG_INSTALL_PATH="${BISHENG_COMPILER}"
    fi
  fi

  if [ "${BUILD_TORCH_MLIR}" = "ON" ]; then
    enable_external_projects="${enable_external_projects};torch-mlir"
    torch_mlir_option="-DPython3_FIND_VIRTUALENV=ONLY\
                      -DTORCH_MLIR_ENABLE_JIT_IR_IMPORTER=OFF\
                      -DTORCH_MLIR_ENABLE_STABLEHLO=OFF\
                      -DTORCH_MLIR_ENABLE_TOSA=OFF\
                      -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=ON\
                      -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR=${TORCH_MLIR_SOURCE_DIR}"
  fi

  # set the default for CCACHE_BUILD to off if ccache is not installed
  local build_ccache_build=""
  if ! command -v ccache >/dev/null 2>&1; then
    echo "ccache could not be found" >&2
    build_ccache_build="OFF"
  else
    build_ccache_build=$CCACHE_BUILD
  fi

  local build_skip_rpath_option=""
  if [ "${SKIP_RPATH_OPTION}" = "TRUE" ] && [ "${PYTHON_BINDING}" = "ON" ]; then
    echo "Currently python binding requires rpath. Overriding --skip_rpath to FALSE."
    build_skip_rpath_option="FALSE"
  elif [ "${SKIP_RPATH_OPTION}" = "TRUE" ]; then
    build_skip_rpath_option="TRUE"
  else
    build_skip_rpath_option="FALSE"
  fi

  # Base compile flags
  COMMON_FLAGS="-fno-common -fvisibility=hidden -fno-strict-aliasing -pipe"
  COMMON_FLAGS+=" -Wformat=2 -Wfloat-equal -Wswitch-default -Wcast-align -Wvla -Wunused -Wundef"

  # OS-specific flags
  if [[ "$OS_TYPE" == "Darwin" ]]; then
    COMMON_FLAGS+=" -Wno-deprecated-declarations"
  else
    COMMON_FLAGS+=" -Wdate-time -Wframe-larger-than=8192"
  fi

  C_FLAGS="${SAFETY_OPTIONS} ${COMMON_FLAGS} -Wstrict-prototypes"
  CXX_FLAGS="${SAFETY_OPTIONS} ${COMMON_FLAGS} -Wnon-virtual-dtor -Wno-unknown-warning-option"

  # Linker flags by OS
  if [[ "$OS_TYPE" == "Darwin" ]]; then
    LD_FLAGS="${SAFETY_LD_OPTIONS}"
    # Add Homebrew libc++ only when using Homebrew LLVM explicitly (avoid undefined symbols with system clang)
    if [[ -d "/opt/homebrew/opt/llvm/lib" ]] && [[ "${CXX_COMPILER}" == *"opt/llvm"* || "${CXX_COMPILER}" == *"homebrew"* ]]; then
      HOMEBREW_PREFIX="/opt/homebrew"
      LLVM_PATH="${HOMEBREW_PREFIX}/opt/llvm"
      LLVM_LIBCXX_PATH="${LLVM_PATH}/lib/c++"
      if [[ -d "$LLVM_LIBCXX_PATH" ]]; then
        LD_FLAGS="${LD_FLAGS} -L${LLVM_LIBCXX_PATH} -L${LLVM_PATH}/lib -Wl,-rpath,${LLVM_LIBCXX_PATH} -Wl,-rpath,${LLVM_PATH}/lib -lc++"
      else
        LD_FLAGS="${LD_FLAGS} -L${LLVM_PATH}/lib -Wl,-rpath,${LLVM_PATH}/lib -lc++"
      fi
    fi
    LD_FLAGS="${LD_FLAGS} -Wl,-dead_strip"
    LD_FLAGS=${LD_FLAGS//-Wl,-z,relro,-z,now/}
    LD_FLAGS=${LD_FLAGS//-Wl,-Bsymbolic-functions/}
    LD_FLAGS=${LD_FLAGS//-rdynamic/}
  else
    LD_FLAGS="${SAFETY_LD_OPTIONS} -Wl,-Bsymbolic-functions -rdynamic"
  fi

  # Python binding options (e.g. framework search on macOS)
  PYTHON_OPTIONS=""
  if [[ "${PYTHON_BINDING}" == "ON" ]] && [[ "$OS_TYPE" == "Darwin" ]]; then
    PYTHON_OPTIONS="-DPython3_FIND_FRAMEWORK=LAST"
  fi

  # Use system ld on macOS with system clang to avoid LLD/libc++ symbol issues
  CMAKE_LINKER_OPT=""
  if [[ "$OS_TYPE" == "Darwin" ]] && [[ "${CXX_COMPILER}" != *"opt/llvm"* && "${CXX_COMPILER}" != *"homebrew"* ]]; then
    if [[ -x "/usr/bin/ld" ]]; then
      CMAKE_LINKER_OPT="-DCMAKE_LINKER=/usr/bin/ld"
    fi
  fi

  echo "Running CMake configuration..."
  echo "Build directory: ${BUILD_DIR}"
  echo "LLVM source: ${LLVM_SOURCE_DIR}/llvm"

  cmake "${LLVM_SOURCE_DIR}/llvm" -G Ninja \
    ${CMAKE_LINKER_OPT} \
    "-B${BUILD_DIR}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_C_COMPILER="${C_COMPILER}" \
    -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DLLVM_ENABLE_PROJECTS="${ENABLE_PROJECTS}" \
    -DLLVM_EXTERNAL_PROJECTS="${enable_external_projects}" \
    -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR="${BISHENGIR_SOURCE_DIR}" \
    -DLLVM_TARGETS_TO_BUILD="${LLVM_BUILD_TARGETS}" \
    ${torch_mlir_option} \
    ${PYTHON_OPTIONS} \
    -DLLVM_ENABLE_ASSERTIONS="${ENABLE_ASSERTION}" \
    -DMLIR_ENABLE_BINDINGS_PYTHON="${PYTHON_BINDING}" \
    -DLLVM_CCACHE_BUILD="${build_ccache_build}" \
    -DCMAKE_C_FLAGS="${C_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
    -DCMAKE_EXE_LINKER_FLAGS="${LD_FLAGS}" \
    -DCMAKE_MODULE_LINKER_FLAGS="${LD_FLAGS}" \
    -DCMAKE_SHARED_LINKER_FLAGS="${LD_FLAGS}" \
    -DCMAKE_SKIP_RPATH="${build_skip_rpath_option}" \
    -DLLVM_INSTALL_UTILS=ON \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DBSPUB_DAVINCI_BISHENGIR=ON \
    -DBISHENGIR_PUBLISH="${BISHENGIR_PUBLISH}" \
    -DBISHENG_COMPILER_PATH="${BISHENG_COMPILER}" \
    -DBISHENGIR_BUILD_TEMPLATE="${BISHENGIR_BUILD_TEMPLATE}" \
    ${CMAKE_OPTIONS}
}

cmake_build() {
  local targets="check-mlir;check-bishengir"
  echo "Building with ${THREADS} threads..."
  if [[ -n "$BUILD_TEST" ]]; then
    ( cd "${BUILD_DIR}" && LIT_OPTS="--timeout=30" cmake --build . -j "${THREADS}" --target "${targets}" ) || exit 1
  else
    ( cd "${BUILD_DIR}" && ninja -j "${THREADS}" ) || exit 1
  fi
  if [[ "${BUILD_BISHENGIR_DOC}" == "ON" ]]; then
    ( cd "${BUILD_DIR}" && cmake --build . -j "${THREADS}" --target "bishengir-doc" ) || exit 1
  fi
}

cmake_install() {
  ( cd "${BUILD_DIR}" && \
    if [[ "${BISHENGIR_PUBLISH}" == "ON" ]]; then
      cmake --build . --target install-bishengir-publish-products
    else
      cmake --install .
    fi
  )
}

main() {
  echo "Starting BiShengIR build process..."
  echo "OS: $OS_TYPE"
  echo "Build directory: $BUILD_DIR"
  echo "C compiler: $C_COMPILER | CXX compiler: $CXX_COMPILER"

  check_dependencies

  if [[ -n "$APPLY_PATCHES" ]]; then
    echo "Applying patches..."
    if [[ -f "${SCRIPT_ROOT}/apply_patches.sh" ]]; then
      if [[ "$OS_TYPE" == "Darwin" ]] && command -v gpatch >/dev/null 2>&1; then
        export PATCH_CMD="gpatch"
        echo "Using gpatch for patching"
      fi
      # Invoke in subshell so apply_patches.sh does not receive build.sh arguments
      bash "${SCRIPT_ROOT}/apply_patches.sh"
    else
      echo "Warning: apply_patches.sh not found at ${SCRIPT_ROOT}/apply_patches.sh"
    fi
  else
    echo "APPLY_PATCHES is not set, skipping patches"
  fi

  if [[ -n "$REBUILD" ]]; then
    echo "Cleaning build directory for rebuild..."
    clean_build_dir
    cmake_generate
  elif [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    echo "First build, generating CMake configuration..."
    mkdir -p "${BUILD_DIR}"
    cmake_generate
  fi

  cmake_build

  if [[ -z "$BUILD_TEST" ]] && [[ -z "$NO_INSTALL" ]]; then
    cmake_install
  fi

  echo "Build Done!!!"
}

main "$@"