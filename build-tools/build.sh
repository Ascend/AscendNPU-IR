#!/bin/bash
# This script is used to build the bishengir project.
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

set -ex

GIT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
THIRD_PARTY_FOLDER=$GIT_ROOT/third-party
readonly SCRIPT_NAME="$(basename $0)"
readonly SCRIPT_ROOT="$(dirname "$(realpath "$0")")"
readonly ENABLE_PROJECTS="mlir;llvm"
# Parse command options.
readonly LONG_OPTS=(
  "add-cmake-options:"
  "apply-patches"
  "bishengir-publish"
  "build:"
  "build-bishengir-so"
  "disable-cann"
  "build-bishengir-doc"
  "build-test"
  "build-torch-mlir"
  "build-type:"
  "c-compiler:"
  "cxx-compiler:"
  "enable-assertion"
  "python-binding"
  "disable-werror"
  "disable-mlir-werror"
  "disable-bishengir-werror"
  "enable-lld"
  "build-triton"
  "disable-ccache"
  "disable-werror"
  "fast-build"
  "help"
  "install-prefix:"
  "jobs:"
  "python-binding"
  "rebuild"
  "shared-libs"
  "safety_options"
  "safety_ld_options"
  "skip_rpath"
  "bishengir-publish"
)
readonly GETOPT_LONGOPTIONS=$(
  IFS=','
  echo "${LONG_OPTS[*]}"
)
# parse the input parameter
readonly TEMP=$(getopt -o hrj:o:st -l "${GETOPT_LONGOPTIONS}" -n "${SCRIPT_NAME}" -- "$@")

# The list of static libraries that will be packed into libBiShengIR.so
readonly WHITELISTED_A=(
  "libMLIRAnnotationDialect.a"
  "libMLIRAnnotationTransforms.a"
  "libMLIRArithToHIVMLLVM.a"
  "libMLIRBiShengIRCompileLib.a"
  "libBiShengIRDialectUtils.a"
  "libMLIRHIVMDialect.a"
  "libMLIRHIVMPipelines.a"
  "libMLIRHIVMTransformOps.a"
  "libMLIRHIVMTransforms.a"
  "libMLIRHIVMUtils.a"
  "libBiShengIRTensorTransforms.a"
  "libMLIRHACCDialect.a"
  "libMLIRHACCUtils.a"
  "libMLIRHMAPDialect.a"
  "libBishengIRMemRefTransforms.a"
  "libBishengIRArithTransforms.a"
  "libBiShengIRSCFTransformOps.a"
  "libBishengIRSCFTransforms.a"
  "libBiShengIRSCFUtils.a"
  "libBiShengIRArithToAffine.a"
  "libBiShengIRTorchPipelines.a"
  "libBiShengIRTensorToHFusion.a"
  "libBiShengIRTensorToHIVM.a"
  "libBishengIRTransform.a"
  "libMLIRMathExtDialect.a"
  "libBishengIRTensorUtils.a"
  "libBiShengIRLLVMCommonConversion.a"
  "libBiShengIRFuncToLLVM.a"
  "libBiShengIRLowerMesh.a"
  "libBiShengIRMeshTransforms.a"
  "libBiShengIRMeshDialect.a"
  "libMLIRMemRefExtDialect.a"
  "libMLIRHACCTransforms.a"
  "libBiShengIRMemRefExtLowering.a"
  "libBiShengIRMemRefDialect.a"
  "libBiShengIRTensorDialect.a"
  "libBiShengIRLinalgDialectExt.a"
)

readonly LLVM_SOURCE_DIR="$THIRD_PARTY_FOLDER/llvm-project"
readonly LLVM_BUILD_DIR=$(pwd)
readonly LLVM_ROOT_DIR=$(pwd)/..

BUILD_TYPE="Release"
C_COMPILER="clang"
CXX_COMPILER="clang++"
THREADS=$(($(grep -c "processor" /proc/cpuinfo) * 3 / 4))
THREADS=$((${THREADS} > 1 ? ${THREADS} : 1))
BUILD_DIR="${SCRIPT_ROOT}"
ENABLE_EXTERNAL_PROJECTS="bishengir"
BISHENGIR_DISABLE_CANN="OFF"
ENABLE_ASSERTION="OFF"
ENABLE_WERROR="ON"
MLIR_WERROR="ON"
BISHENGIR_WERROR="ON"
ENABLE_LLD="OFF"
PYTHON_BINDING="OFF"
SHARED_LIBS="OFF"
BUILD_TORCH_MLIR="OFF"
BUILD_TRITON="OFF"
BUILD_SCRIPTS=(
  "apply_patches.sh"
  "build.sh"
)
BUILD_BISHENGIR_DOC="OFF"
CCACHE_BUILD="ON"
SAFETY_OPTIONS=""
SAFETY_LD_OPTIONS=""
SKIP_RPATH_OPTION="FALSE"
BISHENGIR_PUBLISH="OFF"
BUILD_TARGETS="host"
BUILD_DIR="${GIT_ROOT}/build"

# help infomation
usage() {
  echo -e "${SCRIPT_NAME} - Build the BiShengIR project.

    SYNOPSIS:
      ${SCRIPT_NAME}  [-h | --help] [-r | --rebuild] [-j | --jobs JOBS] [-o | --build PATH]
                [-s | --build-bishengir-so] [--disable-cann] [--apply-patches]
                [--c-compiler C_COMPILER] [--cxx-compiler CXX_COMPILER]
                [--build-type BUILD_TYPE] [--build-test] [--python-binding]
                [--disable-werror] [--disable-mlir-werror] [--disable-bishengir-werror]
                [--shared-libs] [--add-cmake-options CMAKE_OPTIONS] [--disable-ccache] [--safety_options]
                [--safety_ld_options] [--skip_rpath] [--install-prefix INSTALL_PREFIX] [--fast-build]
                [--build-torch-mlir] [--build-triton] [--enalbe-pydsl] [--bishengir-publish] [--build-bishengir-doc]

    Options:
      -h, --help                           Print this help message
      --apply-patches                      Apply patches to third-party submodules. (Default: disabled)
      -r, --rebuild                        Rebuild (Default: incremental compiler)
      -j, --jobs JOBS                      Set the threads when building
                                           (Default: use 3/4 of processing units)
      -o, --build BUILD_PATH               Path to directory which CMake will use as the root of build directory
                                           (Default: build_BiShengIR)
      -s, --build-bishengir-so             Build shared libBiShengIR.so (Default: OFF)
      --disable-cann                       Disable the CANN dependency (Default: OFF)
      --c-compiler C_COMPILER              The full path to the compiler for C (Default: clang)
      --cxx-compiler CXX_COMPILER          The full path to the compiler for C++ (Default: clang++)
      --build-type BUILD_TYPE              Specifies the build type. (Default: Release)
      --build-test                         Whether to build bishengir-test (Default: OFF)
      --enable-assertion                   Whether to build with assertion (Default: OFF)
      --python-binding                     Whether to enable MLIR Python Binding (Default: OFF)
      --disable-werror                     Disable the -Werror compile option.
      --disable-mlir-werror                Disable the -Werror compile flag for MLIR cod
      --disable-bishengir-werror           Disable the -Werror compile flag for BiShengIR code
      --shared-libs                        Whether to build shared library (Default: OFF)
      --build-torch-mlir                   Whether to build torch-mlir
      --build-triton                       Whether to build triton
      --add-cmake-options CMAKE_OPTIONS    Add options to CMake. (Default: null)
      --disable-ccache                     Disable ccache to build toolchain.
      --safety_options                     Whether to build with safe compile options. (Default: null)
      --safety_ld_options                  Whether to build with safe options for linking. (Default: null)
      --skip_rpath                         Disable the Run-time Search Path option.
      --install-prefix INSTALL_PREFIX      CMake install prefix. (Default: BUILD_DIR/install)
      --fast-build                         Skip the installation.
      --bishengir-publish                  Whether to disable features that we don't want to expose to users. (Default: OFF)
      --enable-pydsl                       Enable the PyDSL(BiShengTile) project.
      --build-bishengir-doc                Whether to build BiShengIR documentation. (Default: OFF)
      "
}

if [ $? != 0 ]; then
  echo "Terminating..." >&2
  exit 1
fi
eval set -- "${TEMP}"

while true; do
  case "$1" in
  -h | --help)
    usage
    exit 0
    ;;
  --apply-patches)
    readonly APPLY_PATCHES=""
    shift
    ;;
  -r | --rebuild)
    readonly REBUILD=""
    shift
    ;;
  -j | --jobs)
    THREADS="$2"
    shift 2
    ;;
  -o | --build)
    BUILD_DIR="$(realpath "$2")"
    shift 2
    ;;
  -s | --build-bishengir-so)
    readonly BUILD_LIB_BISHENGIR_SO=""
    shift
    ;;
  --disable-cann)
    BISHENGIR_DISABLE_CANN="ON"
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
  --build-bishengir-doc)
    BUILD_BISHENGIR_DOC="ON"
    shift
    ;;
  --build-type)
    BUILD_TYPE="$2"
    shift 2
    ;;
  --build-test)
    readonly BUILD_TEST=""
    shift
    ;;
  --enable-assertion)
    ENABLE_ASSERTION="ON"
    shift
    ;;
  --python-binding)
    PYTHON_BINDING="ON"
    shift
    ;;
  --disable-werror)
    ENABLE_WERROR="OFF"
    shift
    ;;
  --disable-mlir-werror)
    MLIR_WERROR="OFF"
    shift
    ;;
  --disable-bishengir-werror)
    BISHENGIR_WERROR="OFF"
    shift
    ;;
  --enable-lld)
    ENABLE_LLD="ON"
    shift
    ;;
  --shared-libs)
    SHARED_LIBS="ON"
    shift
    ;;
  --build-torch-mlir)
    BUILD_TORCH_MLIR="ON"
    shift
    ;;
  --build-triton)
    BUILD_TRITON="ON"
    shift
    ;;
  --disable-ccache)
    CCACHE_BUILD="OFF"
    shift
    ;;
  --safety_options)
    SAFETY_OPTIONS="-fPIC -fstack-protector-strong"
    shift
    ;;
  --safety_ld_options)
    SAFETY_LD_OPTIONS="-s -Wl,-z,relro,-z,now"
    shift
    ;;
  --skip_rpath)
    SKIP_RPATH_OPTION="TRUE"
    shift
    ;;
  --add-cmake-options)
    CMAKE_OPTIONS+=" $2"
    shift 2
    ;;
  --install-prefix)
    readonly INSTALL_PREFIX="$(realpath "$2")"
    shift 2
    ;;
  --fast-build)
    readonly NO_INSTALL=""
    shift
    ;;
  --bishengir-publish)
    BISHENGIR_PUBLISH="ON"
    shift
    ;;
  --enable-pydsl)
    readonly ENABLE_EXTERNAL_PROJECTS="${ENABLE_EXTERNAL_PROJECTS};mlir-bisheng"
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
    # If the build directory is "build_BiShengIR", then the build script should be preserved.
    find "${BUILD_DIR}" -mindepth 1 -maxdepth 1 \
      $(printf -- "-not -name %s " ${BUILD_SCRIPTS[@]}) \
      -exec rm -rf {} +
  else
    [[ -n "${BUILD_DIR}" ]] && rm -rf "${BUILD_DIR}"
    mkdir "${BUILD_DIR}"
  fi
}

if [[ -z "${INSTALL_PREFIX+x}" ]]; then
  readonly INSTALL_PREFIX="${BUILD_DIR}/install"
fi

cmake_generate() {
  cd ${BUILD_DIR}
  local torch_mlir_option=""
  local enable_external_projects="bishengir"
  if [ "${BUILD_TORCH_MLIR}" = "ON" ]; then
    enable_external_projects="${enable_external_projects};torch-mlir"
    torch_mlir_option="-DPython3_FIND_VIRTUALENV=ONLY -DTORCH_MLIR_ENABLE_JIT_IR_IMPORTER=OFF -DTORCH_MLIR_ENABLE_STABLEHLO=OFF -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=ON -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR=../torch-mlir"
  fi

  local triton_options=""
  if [ "${BUILD_TRITON}" = "ON" ]; then
    enable_external_projects="${enable_external_projects};triton"
    triton_options="-DLLVM_EXTERNAL_TRITON_SOURCE_DIR=../triton"
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
  echo "cmake $LLVM_SOURCE_DIR/llvm -G Ninja \
            -DCMAKE_C_COMPILER="${C_COMPILER}" \
            -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
            -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
            -DLLVM_ENABLE_WERROR="${ENABLE_WERROR}" \
            -DLLVM_ENABLE_LLD="${ENABLE_LLD}" \
            -DLLVM_ENABLE_PROJECTS="${ENABLE_PROJECTS}" \
            -DLLVM_EXTERNAL_PROJECTS="${enable_external_projects}" \
            -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR=../bishengir \
            -DLLVM_EXTERNAL_MLIR_BISHENG_SOURCE_DIR=../mlir-bisheng \
            -DMLIR_ENABLE_WERROR="${MLIR_WERROR}" \
            -DBISHENGIR_ENABLE_WERROR="${BISHENGIR_WERROR}" \
            ${torch_mlir_option} \
            ${triton_options} \
            -DLLVM_TARGETS_TO_BUILD="${BUILD_TARGETS}" \
            -DLLVM_ENABLE_HIIPU=ON \
            -DBISHENGIR_DISABLE_CANN="${BISHENGIR_DISABLE_CANN}" \
            -DLLVM_ENABLE_ASSERTIONS="${ENABLE_ASSERTION}" \
            -DMLIR_ENABLE_BINDINGS_PYTHON="${PYTHON_BINDING}" \
            -DBUILD_SHARED_LIBS="${SHARED_LIBS}" \
            -DBSPRIV_DAVINCI=ON \
            -DLLVM_BSPRIV_DAVINCI_BISHENGIR=ON \
            -DLLVM_CCACHE_BUILD="${build_ccache_build}" \
            -DCMAKE_C_FLAGS="${SAFETY_OPTIONS}" \
            -DCMAKE_CXX_FLAGS="${SAFETY_OPTIONS}" \
            -DCMAKE_EXE_LINKER_FLAGS="${SAFETY_LD_OPTIONS}" \
            -DCMAKE_MODULE_LINKER_FLAGS="${SAFETY_LD_OPTIONS}" \
            -DCMAKE_SHARED_LINKER_FLAGS="${SAFETY_LD_OPTIONS}" \
            -DCMAKE_SKIP_RPATH="${build_skip_rpath_option}" \
            -DLLVM_INSTALL_UTILS=ON \
            -DBISHENGIR_PUBLISH="${BISHENGIR_PUBLISH}" \
            -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
            -DLLVM_LIT_ARGS="-sv -j${THREADS}" \
            -DMLIR_ENABLE_BISHENGIR_EXTENTION=ON \
            -DBISHENGIR_ENABLE_PM_CL_OPTIONS=ON \
            ${CMAKE_OPTIONS}"

  cmake $LLVM_SOURCE_DIR/llvm -G Ninja \
    -DCMAKE_C_COMPILER="${C_COMPILER}" \
    -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DLLVM_ENABLE_WERROR="${ENABLE_WERROR}" \
    -DLLVM_ENABLE_LLD="${ENABLE_LLD}" \
    -DLLVM_ENABLE_PROJECTS="${ENABLE_PROJECTS}" \
    -DLLVM_EXTERNAL_PROJECTS="${enable_external_projects}" \
    -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR=../bishengir \
    -DLLVM_EXTERNAL_MLIR_BISHENG_SOURCE_DIR=../mlir-bisheng \
    -DMLIR_ENABLE_WERROR="${MLIR_WERROR}" \
    -DBISHENGIR_ENABLE_WERROR="${BISHENGIR_WERROR}" \
    ${torch_mlir_option} \
    ${triton_options} \
    -DLLVM_TARGETS_TO_BUILD="${BUILD_TARGETS}" \
    -DLLVM_ENABLE_HIIPU=ON \
    -DBISHENGIR_DISABLE_CANN="${BISHENGIR_DISABLE_CANN}" \
    -DLLVM_ENABLE_ASSERTIONS="${ENABLE_ASSERTION}" \
    -DMLIR_ENABLE_BINDINGS_PYTHON="${PYTHON_BINDING}" \
    -DBUILD_SHARED_LIBS="${SHARED_LIBS}" \
    -DBSPRIV_DAVINCI=ON \
    -DLLVM_BSPRIV_DAVINCI_BISHENGIR=ON \
    -DLLVM_CCACHE_BUILD="${build_ccache_build}" \
    -DCMAKE_C_FLAGS="${SAFETY_OPTIONS}" \
    -DCMAKE_CXX_FLAGS="${SAFETY_OPTIONS}" \
    -DCMAKE_EXE_LINKER_FLAGS="${SAFETY_LD_OPTIONS}" \
    -DCMAKE_MODULE_LINKER_FLAGS="${SAFETY_LD_OPTIONS}" \
    -DCMAKE_SHARED_LINKER_FLAGS="${SAFETY_LD_OPTIONS}" \
    -DCMAKE_SKIP_RPATH="${build_skip_rpath_option}" \
    -DLLVM_INSTALL_UTILS=ON \
    -DBISHENGIR_PUBLISH="${BISHENGIR_PUBLISH}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DLLVM_LIT_ARGS="-sv -j${THREADS}" \
    -DMLIR_ENABLE_BISHENGIR_EXTENTION=ON \
    -DBISHENGIR_ENABLE_PM_CL_OPTIONS=ON \
    ${CMAKE_OPTIONS}
}

cmake_build() {
#  local targets="check-mlir;check-bishengir"
  local targets="check-bishengir"
  if [[ -v BUILD_TEST ]]; then
    cmake --build . -j "${THREADS}" --target "${targets}" || exit 1
  else
    ninja -j "${THREADS}" || exit 1
  fi

  if [ "${BUILD_BISHENGIR_DOC}" = "ON" ]; then
    cmake --build . -j "${THREADS}" --target "bishengir-doc" || exit 1
  fi
}

cmake_install() {
  cmake --install "${BUILD_DIR}"
}

build_bishengir_so() {
  cd lib

  GET_WHITELISTED_A=$(printf "%s " "${WHITELISTED_A[@]}")

  if [ "${BUILD_TORCH_MLIR}" = "OFF" ]; then
    GET_WHITELISTED_A=(${GET_WHITELISTED_A[*]/"libBiShengIRTorchPipelines.a"})
  fi

  g++ -shared -o libBiShengIR.so -Wl,--whole-archive ${GET_WHITELISTED_A[*]} -Wl,--no-whole-archive -Wl,-z,relro,-z,now -D_FORTIFY_SOURCE=2 -O2 -s -ftrapv
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
    # First build.
    cmake_generate
  fi

  # Build.
  cmake_build

  if [[ -v BUILD_LIB_BISHENGIR_SO ]]; then
    build_bishengir_so
  fi

  # Install.
  if [ ! -v BUILD_TEST ] && [ -z "${NO_INSTALL+x}" ]; then
    cmake_install
  fi

  echo "Build Done!!!"
}

main "$@"
