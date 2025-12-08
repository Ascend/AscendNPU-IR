
# 🛠️ 构建指南

## 🧭 概述
AscendNPU IR（AscendNPU Intermediate Representation）是基于MLIR（Multi-Level Intermediate Representation）构建的面向昇腾亲和算子编译时使用的中间表示。本指南将帮助您在本地机器上配置开发环境、获取源代码并成功构建本项目。

## 环境准备

### 编译器与工具链

以下为基础的编译器与工具链要求：

- CMake >= 3.28
- Ninja >= 1.12.0

推荐使用：
- Clang >= 10
- LLD >= 10 （使用LLVM LLD将显著提升构建速度）

## 📥源码准备

1. 克隆主仓库：

```bash
git clone https://gitcode.com/Ascend/ascendnpu-ir.git
cd AscendNPU-IR
```

2. 初始化并更新子模块（Submodules）

本项目依赖LLVM、Torch-MLIR等三方库，需要拉取并更新到指定的commit id。

```bash
# 递归地拉取所有子模块
git submodule update --init --recursive
```

## 🏗️ 将AscendNPU IR构建为外部LLVM项目

### 使用提供的构建脚本（推荐）

我们提供了一个便捷的构建脚本 `build.sh` 来自动化配置和构建过程。

```bash
# 首次在项目根目录下运行
./build-tools/build.sh -o ./build --build-type Debug --apply-patches [可选参数]
# 非首次在项目根目录下运行
./build-tools/build.sh -o ./build --build-type Debug [可选参数]
```

脚本常见参数：

- `--apply-patches`：使能AscendNPU IR对三方仓库的扩展功能，首次编译时需要启用。
- `-o`：编译产物输出路径
- `--build-type`：构建类型，如"Release"、"Debug"。

### 手动构建（供高级用户参考）

如果您希望手动控制过程，可以参考`build.sh`脚本内部的命令：

```bash
# 在项目根目录下
mkdir -p build
cd build

# 运行 CMake 进行配置
export LLVM_SOURCE_DIR=$(realpath ../third-party/llvm-project/llvm)
cmake ${LLVM_SOURCE_DIR} -G Ninja \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_EXTERNAL_PROJECTS="bishengir" \
    -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR="AscendNPU-IR" \ # 项目根目录
    -DBSPUB_DAVINCI_BISHENGIR=ON # 必须项！用于使能AscendNPU IR对于三方仓库的扩展
    [其他您需要的 CMake 选项]

ninja -j32
```

## 🧪 运行测试

### 编译测试Target

```bash
# 在 `build` 目录下
cmake --build . --target "check-bishengir"
```

### 使用LLVM-LIT执行测试套

```bash
# 在 `build` 目录下
./bin/llvm-lit ../bishengir/test
```
