
# 🛠️ Build Guide

## 🧭 Overview
AscendNPU IR (AscendNPU Intermediate Representation) is an intermediate representation built on MLIR (Multi-Level Intermediate Representation) designed for compiling Ascend-compatible operators. This guide will assist you in setting up the development environment on your local machine, obtaining the source code, and successfully building this project.

## Environment Setup

### Compiler and Toolchain

The following are the fundamental compiler and toolchain requirements:

- CMake >= 3.28
- Ninja >= 1.12.0

Recommendation：
- Clang >= 10
- LLD >= 10 (Using LLVM LLD will significantly enhance build speed)

## 📥Source Code Preparation

1. Clone the main repository：

```bash
git clone https://gitcode.com/Ascend/ascendnpu-ir.git
cd AscendNPU-IR
```

2. Initialize and update submodules

This project relies on third-party libraries such as LLVM and Torch-MLIR, which need to be pulled and updated to the specified commit ID.

```bash
# Recursively pull all submodules
git submodule update --init --recursive
```

## 🏗️ Build AscendNPU IR as an External LLVM project

### Using the Provided Build Script (Recommended)

We provide a convenient build script `build.sh` to automate configuration and build process.

```bash
# Running for the first time in the project root directory
./build-tools/build.sh -o ./build --build-type Debug --apply-patches [Optional]
# Not first time running in the project root directory
./build-tools/build.sh -o ./build --build-type Debug [Optional]
```

Common script parameters:

- `--apply-patches`：Enables the extended functionality of AscendNPU IR for third-party repositories, Recommended for initial compilation.
- `-o`: Output path for compiled artefacts
- `--build-type`: Build type, such as "Release" and "Debug".

### Manual Build (For Advanced Users)

If you wish to manually control the process, refer to the commands within the `build.sh` script:

```bash
# In the root directory of the project
mkdir -p build
cd build

# Run CMake for configuration
cmake -G Ninja .. \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_EXTERNAL_PROJECTS="bishengir" \
    -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR="AscendNPU-IR" \ # Project root directory
    -DBSPUB_DAVINCI_BISHENGIR=ON # Mandatory! Enables AscendNPU IR extensions for third-party repositories
    [Other CMake Options as required]

ninja -j32
```

Note: When the LLVM version is 21 or higher, add the option -DLLVM_MAJOR_VERSION_21_COMPATIBLE=ON

## 🧪 Run tests

### Compile Test Targets

```bash
# In the `build` directory
cmake --build . --target "check-bishengir"
```

### Execute Test Suits with LLVM-LIT

```bash
# In the `build` directory
./bin/llvm-lit ../bishengir/test
```
