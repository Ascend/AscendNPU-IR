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

set -e

GIT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
THIRD_PARTY_FOLDER=$GIT_ROOT/third-party
SCRIPT_ROOT="$(dirname "$(realpath "$0")")"
readonly SCRIPT_NAME="$(basename $0)"
readonly ENABLE_PROJECTS="mlir"

# Parse command options.
readonly LONG_OPTS=(
  "add-cmake-options:"
  "apply-patches"
  "bishengir-publish:"
  "build:"
  "build-bishengir-doc"
  "build-test"
  "build-type:"
  "build-torch-mlir"
  "c-compiler:"
  "cxx-compiler:"
  "disable-ccache"
  "enable-assertion"
  "fast-build"
  "help"
  "install-prefix:"
  "jobs:"
  "llvm-source-dir:"
  "python-binding"
  "rebuild"
  "safety-options"
  "safety-ld-options"
  "skip-rpath"
  "enable-cpu-runner"
)

readonly GETOPT_LONGOPTIONS=$(printf "%s," "${LONG_OPTS[@]}")
# parse the input parameter
readonly TEMP=$(getopt -o hrj:o:st -l "${GETOPT_LONGOPTIONS}" -n "${SCRIPT_NAME}" -- "$@")

BUILD_TYPE="Release"
C_COMPILER="clang"
CXX_COMPILER="clang++"
THREADS=$(($(grep -c "processor" /proc/cpuinfo) * 3 / 4))
THREADS=$((${THREADS} > 1 ? ${THREADS} : 1))
BUILD_DIR="${GIT_ROOT}/build"
BUILD_SCRIPTS=(
  "apply_patches.sh"
  "patches"
  "build.sh"
)
BUILD_BISHENGIR_DOC="OFF"
# We assume that the build script is executed in the build directory.
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

# help infomation
usage() {
  echo -e "${SCRIPT_NAME} - Build the BiShengIR project.

    SYNOPSIS:
      ${SCRIPT_NAME}
                [--add-cmake-options CMAKE_OPTIONS]
                [--apply-patches]
                [--bishengir-publish]
                [-o | --build PATH]
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
                [-t | --build-bishengir-template]
                [--safety-options]
                [--safety-ld-options]
                [--skip_rpath]
                [--enable-cpu-runner]

    Options:
      --add-cmake-options CMAKE_OPTIONS    Add options to CMake. (Default: null)
      --apply-patches                      Apply patches to third-party submodules. (Default: disabled)
      --bishengir-publish                  Whether to disable features is currently unpublished. (Default: ON)
      -o, --build BUILD_PATH               Path to directory which CMake will use as the root of build directory
                                           (Default: build)
      --build-bishengir-doc                Whether to build BiShengIR documentation. (Default: disabled)
      --build-test                         Whether to build bishengir-test (Default: disabled)
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

if [ $? != 0 ]; then
  echo "Terminating..." >&2
  exit 1
fi
eval set -- "${TEMP}"

while true; do
  case "$1" in
  --add-cmake-options)
    CMAKE_OPTIONS+=" $2"
    shift 2
    ;;
  --apply-patches)
    readonly APPLY_PATCHES=""
    shift
    ;;
  --bishengir-publish)
    BISHENGIR_PUBLISH="$2"
    shift 2
    ;;
  -o | --build)
    BUILD_DIR="$(realpath "$2")"
    shift 2
    ;;
  --build-bishengir-doc)
    BUILD_BISHENGIR_DOC="ON"
    shift
    ;;
  --build-test)
    readonly BUILD_TEST=""
    shift
    ;;
  --build-type)
    BUILD_TYPE="$2"
    shift 2
    ;;
  --build-torch-mlir)
    BUILD_TORCH_MLIR="ON"
    shift
    ;;
  --c-compiler)
    C_COMPILER="$2"
    shift 2
    ;;
  --cxx-compiler)
    CXX_COMPILER="$2"
    shift 2
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
    readonly NO_INSTALL=""
    shift
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  --install-prefix)
    readonly INSTALL_PREFIX="$(realpath "$2")"
    shift 2
    ;;
  -j | --jobs)
    THREADS="$2"
    shift 2
    ;;
  --llvm-source-dir)
    LLVM_SOURCE_DIR="$(realpath "$2")"
    shift 2
    ;;
  --python-binding)
    PYTHON_BINDING="ON"
    shift
    ;;
  -r | --rebuild)
    readonly REBUILD=""
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
    TORCH_MLIR_SOURCE_DIR="$(realpath "$2")"
    shift 2
    ;;
  --enable-cpu-runner)
    LLVM_BUILD_TARGETS+=";Native"
    shift
    ;;
  --)
    shift
    break
    ;;
  *)
    break
    ;;
  esac
done

clean_build_dir() {
  if [[ "${BUILD_DIR}" = "${SCRIPT_ROOT}" ]]; then
    # If the build directory is "build-tools", then the build script should be preserved.
    find "${BUILD_DIR}" -mindepth 1 -maxdepth 1 \
      $(printf -- "-not -name %s " ${BUILD_SCRIPTS[@]}) \
      -exec rm -rf {} +
  else
    [[ -n "${BUILD_DIR}" ]] && rm -rf ${BUILD_DIR}/CMake*
    mkdir -p "${BUILD_DIR}"
  fi
}

if [[ -z "${INSTALL_PREFIX+x}" ]]; then
  readonly INSTALL_PREFIX="${BUILD_DIR}/install"
fi

cmake_generate() {
  local torch_mlir_option=""
  local enable_external_projects="bishengir"
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

  COMMON_FLAGS="\
  -fno-common \
  -fvisibility=hidden \
  -fno-strict-aliasing \
  -pipe \
  -Wformat=2 \
  -Wdate-time \
  -Wfloat-equal \
  -Wswitch-default \
  -Wcast-align \
  -Wvla \
  -Wunused \
  -Wundef \
  -Wframe-larger-than=8192"

  C_FLAGS="${SAFETY_OPTIONS} ${COMMON_FLAGS} -Wstrict-prototypes"
  CXX_FLAGS="${SAFETY_OPTIONS} ${COMMON_FLAGS} -Wnon-virtual-dtor -Wno-unknown-warning-option"
  LD_FLAGS="${SAFETY_LD_OPTIONS} -Wl,-Bsymbolic-functions -rdynamic"

  cmake $LLVM_SOURCE_DIR/llvm -G Ninja \
    "-B${BUILD_DIR}" \
    -DCMAKE_C_COMPILER="${C_COMPILER}" \
    -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DLLVM_ENABLE_PROJECTS="${ENABLE_PROJECTS}" \
    -DLLVM_EXTERNAL_PROJECTS="${enable_external_projects}" \
    -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR="${BISHENGIR_SOURCE_DIR}" \
    -DLLVM_TARGETS_TO_BUILD="${LLVM_BUILD_TARGETS}" \
    ${torch_mlir_option} \
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
    ${CMAKE_OPTIONS}
}

cmake_build() {
  cd ${BUILD_DIR}
  local targets="check-mlir;check-bishengir"

  if [[ -v BUILD_TEST ]]; then
    LIT_OPTS="--timeout=30" cmake --build . -j "${THREADS}" --target "${targets}" || exit 1
  else
    ninja -j "${THREADS}" || exit 1
  fi

  if [ "${BUILD_BISHENGIR_DOC}" = "ON" ]; then
    cmake --build . -j "${THREADS}" --target "bishengir-doc" || exit 1
  fi
  cd -
}

cmake_install() {
  cd ${BUILD_DIR}
  if [ "${BISHENGIR_PUBLISH}" = "ON" ]; then
    cmake --build . --target install-bishengir-publish-products
  else
    cmake --install "${BUILD_DIR}"
  fi
  cd -
}

main() {
  if [[ -v APPLY_PATCHES ]]; then
    source ${SCRIPT_ROOT}/apply_patches.sh
  fi

  # Rebuild.
  if [[ -v REBUILD ]]; then
    clean_build_dir
    cmake_generate
  elif [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    mkdir -p ${BUILD_DIR}
    # First build.
    cmake_generate
  fi

  # Build.
  cmake_build

  # Install.
  if [ ! -v BUILD_TEST ] && [ -z "${NO_INSTALL+x}" ]; then
    cmake_install
  fi

  echo "Build Done!!!"
}

main "$@"