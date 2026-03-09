# Build and installation

This document describes dependency installation, build methods (source and binary), and running tests for AscendNPU IR.

## 1 Install dependencies

### 1.1 Build dependencies

#### 1.1.1 Compiler and toolchain requirements

Minimum requirements:

- CMake >= 3.28
- Ninja >= 1.12.0

Recommended:

- Clang >= 10
- LLD >= 10 (LLVM LLD significantly speeds up builds)

#### 1.1.2 Source preparation

1. Clone the repository (then enter the repo directory, typically `ascendnpu-ir`):

```bash
git clone https://gitcode.com/Ascend/ascendnpu-ir.git
cd ascendnpu-ir
```

2. Initialize and update submodules

The project depends on LLVM, Torch-MLIR, and other third-party code; submodules must be updated to the required commits.

```bash
# Recursively init and update all submodules
git submodule update --init --recursive
```

### 1.2 Runtime dependencies

#### 1.2.1 CANN package installation

End-to-end runs depend on the CANN environment.

1. Download CANN: obtain the toolkit package and the ops package for your hardware from the [Ascend CANN download page](https://www.hiascend.com/developer/download/community/result?module=cann).

2. Install CANN:

```bash
# Example: x86 A3, CANN 8.5.0
chmod +x Ascend-cann-toolkit_8.5.0_linux-x86_64.run
chmod +x Ascend-cann-A3-ops_8.5.0_linux-x86_64.run
./Ascend-cann-toolkit_8.5.0_linux-x86_64.run --full [--install-path=${PATH-TO-CANN}]
./Ascend-cann-A3-ops_8.5.0_linux-x86_64.run --install [--install-path=${PATH-TO-CANN}]
```

3. Set environment variables:

```bash
source ${PATH-TO-CANN}/ascend-toolkit/set_env.sh
```

## 2 Build commands

### 2.1 Source build

#### 2.1.1 Using the build script (recommended)

A convenience script `build.sh` automates configuration and build.

```bash
# First run from repo root
./build-tools/build.sh -o ./build --build-type Debug --apply-patches [optional args]
# Later runs
./build-tools/build.sh -o ./build --build-type Debug [optional args]
```

Common options:

- `--apply-patches`: Enable AscendNPU IR extensions for third-party repos; required on first build.
- `-o`: Build output directory.
- `--build-type`: Build type, e.g. "Release", "Debug".

#### 2.1.2 Manual build (advanced)

From the repo root:

```bash
mkdir -p build
cd build

# Configure (LLVM_EXTERNAL_BISHENGIR_SOURCE_DIR = repo root, parent of build)
export LLVM_SOURCE_DIR=$(realpath ../third-party/llvm-project/llvm)
cmake ${LLVM_SOURCE_DIR} -G Ninja \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_EXTERNAL_PROJECTS="bishengir" \
    -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR="$(realpath ..)" \
    -DBSPUB_DAVINCI_BISHENGIR=ON \
    [other CMake options as needed]

ninja -j32
```

For LLVM >= 21, add `-DLLVM_MAJOR_VERSION_21_COMPATIBLE=ON`.

### 2.2 Binary installation

#### 2.2.1 With CANN package

AscendNPU IR binaries can be installed as part of the CANN toolkit; see "1.2.1 CANN package installation" above.

#### 2.2.2 Standalone AscendNPU IR package

A standalone installer is also available.

1. Download the AscendNPU IR installer for your version.

2. Install:

```bash
# Example: x86, AscendNPU IR 1.0.0
chmod +x ascendnpu-ir_1.0.0_linux-x86.run
./ascendnpu-ir_1.0.0_linux-x86.run --install [--install-path=${PATH-TO-ASCENDNPU-IR}]
```

### 2.3 Environment variables

Add the directory containing `bishengir-compile` to your PATH:

```bash
export PATH=${PATH-TO-BISHENGIR-COMPILE}:$PATH
```

## 3 Running tests

### 3.1 Build test target

```bash
# From the build directory
cmake --build . --target "check-bishengir"
```

### 3.2 Run tests with LLVM LIT

```bash
# From the build directory
./bin/llvm-lit ../bishengir/test
```

## 4 FAQ

**Q:** When building with `build-tools/build.sh`, I get "ninja: error: loading 'build.ninja': No such file or directory". What should I do?  
**A:** Add the `-r` option to rerun CMake and regenerate `build.ninja`.

**Q:** Build fails with "Too many open files". What should I do?  
**A:** Raise the per-process open-file limit, e.g. `ulimit -n 65535`.

**Q:** Build fails with:
```bash
The CMAKE_CXX_COMPILER:
  clang++
is not a full path and was not found in the PATH.
```
What should I do?  
**A:** Specify the C++ compiler with `--cxx-compiler=${CXX-COMPILER-PATH}` or reinstall/use another version (e.g. clang++-15).
