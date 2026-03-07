# Build and Installation

This guide describes dependency setup, build methods (source/binary), and test execution for AscendNPU IR.

## 1 Install Dependencies

### 1.1 Build dependencies

#### 1.1.1 Compiler and toolchain requirements

Basic compiler and toolchain requirements:

- CMake >= 3.28
- Ninja >= 1.12.0

Recommended:

- Clang >= 10
- LLD >= 10 (LLVM LLD can significantly improve build speed)

#### 1.1.2 Source preparation

1. Clone the main repository (after cloning, enter the repository directory, typically named `ascendnpu-ir`):

```bash
git clone https://gitcode.com/Ascend/ascendnpu-ir.git
cd ascendnpu-ir
```

1. Initialize and update submodules.

This project depends on third-party libraries such as LLVM and Torch-MLIR. Pull submodules and update them to the specified commit IDs:

```bash
# Recursively pull all submodules
git submodule update --init --recursive
```

### 1.2 Runtime dependencies

#### 1.2.1 Install CANN packages

End-to-end execution of AscendNPU IR depends on the CANN environment.

1. Download CANN packages: both the toolkit package and hardware-matched ops package are required. They can be obtained from [Ascend Community CANN downloads](https://www.hiascend.com/developer/download/community/result?module=cann).

2. Install CANN packages:

```bash
# Example: x86 A3 environment with CANN 8.5.0
chmod +x Ascend-cann-toolkit_8.5.0_linux-x86_64.run
chmod +x Ascend-cann-A3-ops_8.5.0_linux-x86_64.run
./Ascend-cann-toolkit_8.5.0_linux-x86_64.run --full [--install-path=${PATH-TO-CANN}]
./Ascend-cann-A3-ops_8.5.0_linux-x86_64.run --install [--install-path=${PATH-TO-CANN}]
```

1. Set environment variables:

```bash
source ${PATH-TO-CANN}/ascend-toolkit/set_env.sh
```

## 2 Build Commands

### 2.1 Source installation

#### 2.1.1 Use the provided build script (recommended)

We provide a convenient build script, `build.sh`, to automate configuration and building.

```bash
# First run in the repository root
./build-tools/build.sh -o ./build --build-type Debug --apply-patches [optional-args]
# Subsequent runs in the repository root
./build-tools/build.sh -o ./build --build-type Debug [optional-args]
```

Common script options:

- `--apply-patches`: enables AscendNPU IR extensions for third-party repositories. Required on the first build.
- `-o`: build output path.
- `--build-type`: build type, such as "Release" or "Debug".

#### 2.1.2 Manual build (for advanced users)

If you prefer to control the process manually, you can refer to the commands used in `build.sh`:

```bash
# In the repository root
mkdir -p build
cd build

# Configure with CMake (LLVM_EXTERNAL_BISHENGIR_SOURCE_DIR points to project root, i.e., parent of build)
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

Note: when using LLVM version 21 or later, add `-DLLVM_MAJOR_VERSION_21_COMPATIBLE=ON`.

### 2.2 Binary installation

#### 2.2.1 Install with CANN package

AscendNPU IR binaries are installed together with the CANN toolkit package. See "1.2.1 Install CANN packages" above.

#### 2.2.2 Standalone AscendNPU IR package installation

A standalone AscendNPU IR package is also available.

1. Download the AscendNPU IR installer package:

```bash
# Download the corresponding AscendNPU IR installer package version
```

1. Install AscendNPU IR:

```bash
# Example: x86 environment with AscendNPU IR 1.0.0
chmod +x ascendnpu-ir_1.0.0_linux-x86.run
./ascendnpu-ir_1.0.0_linux-x86.run --install [--install-path=${PATH-TO-ASCENDNPU-IR}]
```

### 2.3 Environment variable setup

To use AscendNPU IR, add the path that contains the `bishengir-compile` executable to `PATH`:

```bash
# Add ${PATH-TO-BISHENGIR-COMPILE}, where bishengir-compile is located, into PATH
export PATH=${PATH-TO-BISHENGIR-COMPILE}:$PATH
```

## 3 Run Tests

### 3.1 Build test targets

```bash
# In the `build` directory
cmake --build . --target "check-bishengir"
```

### 3.2 Run test suites with LLVM-LIT

```bash
# In the `build` directory
./bin/llvm-lit ../bishengir/test
```

## 4 FAQ

Q: When building with `build-tools/build.sh`, how to fix this error: "ninja: error: loading 'build.ninja': No such file or directory"? 
A: Add the `-r` option when invoking `build-tools/build.sh` to re-run CMake and regenerate `build.ninja`.

Q: How to fix the "Too many open files" error during build? 
A: The number of simultaneously opened files exceeds the system limit. Increase it with `ulimit -n xxx`, for example `ulimit -n 65535`.

Q: How to fix this build error?

```bash
The CMAKE_CXX_COMPILER:

 clang++

is not a full path and was not found in the PATH.
```

A: The C++ compiler is not specified or the compiler binary is invalid. First try `--cxx-compiler=${CXX-COMPILER-PATH}` to specify the compiler. If the issue remains, reinstall or switch to another compiler version, such as the recommended `clang++-15`.
