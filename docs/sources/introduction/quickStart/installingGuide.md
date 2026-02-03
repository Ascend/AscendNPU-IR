# 快速开始

## 1 安装依赖

### 1.1 构建依赖

#### 1.1.1 编译器与工具链要求

以下为基础的编译器与工具链要求：

- CMake >= 3.28
- Ninja >= 1.12.0

推荐使用：

- Clang >= 10
- LLD >= 10 （使用LLVM LLD将显著提升构建速度）

#### 1.1.2 源码准备

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

### 1.2 运行依赖

#### 1.2.1 CANN包安装

AscendNPU-IR端到端运行依赖CANN环境。

1. 下载CANN包：

```bash
# 需要下载toolkit包和对应硬件的ops包
https://www.hiascend.com/developer/download/community/result?module=cann
```

2. 安装CANN包：

```bash
# 以x86系统A3环境，CANN 8.5.0版本为例
chmod +x Ascend-cann-toolkit_8.5.0_linux-x86_64.run
chmod +x Ascend-cann-A3-ops_8.5.0_linux-x86_64.run
./Ascend-cann-toolkit_8.5.0_linux-x86_64.run --full [--install-path=${PATH-TO-CANN}]
./Ascend-cann-A3-ops_8.5.0_linux-x86_64.run --install [--install-path=${PATH-TO-CANN}]
```

3. 设置环境变量：

```bash
source ${PATH-TO-CANN}/ascend-toolkit/set_env.sh
```



## 2 构建指令

### 2.1 源码安装

#### 2.1.1 使用提供的构建脚本（推荐）

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

注：LLVM 版本大于等于 21 时添加 -DLLVM_MAJOR_VERSION_21_COMPATIBLE=ON 选项

### 1.2 二进制安装

#### 1.2.1 随CANN包安装

AscendNPU-IR二进制会随CANN toolkit包一起安装，参考1.2.1。

#### 1.2.2 AscendNPU-IR包单独安装

AscendNPU-IR有独立安装包以供使用。

1. 下载AscendNPU-IR安装包

```bash
# 下载对应版本的AscendNPU-IR安装包
```

2. 安装AscendNPU-IR

```bash
# 以x86系统环境，AscendNPU-IR 1.0.0版本为例
chmod +x ascendnpu-ir_1.0.0_linux-x86.run
./ascendnpu-ir_1.0.0_linux-x86.run --install [--install-path=${PATH-TO-ASCENDNPU-IR}]
```

### 1.3 环境变量设置

要使用AscendNPU-IR，需要将bishengir-compile可执行文件所在的路径加入到PATH环境变量中。

```bash
# 将bishengir-compile所在目录${PATH-TO-BISHENGIR-COMPILE}加入到PATH环境变量中
export PATH=${PATH-TO-BISHENGIR-COMPILE}:$PATH
```



## 3 运行测试

### 3.1 编译测试Target

```bash
# 在 `build` 目录下
cmake --build . --target "check-bishengir"
```

### 3.2 使用LLVM-LIT执行测试套

```bash
# 在 `build` 目录下
./bin/llvm-lit ../bishengir/test
```



## 4 FAQ
