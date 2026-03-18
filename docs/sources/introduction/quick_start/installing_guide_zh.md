# 构建安装

本文说明 AscendNPU IR 的依赖安装、构建方式（源码/二进制）及运行测试步骤。

## 安装依赖

### 构建依赖

#### 编译器与工具链要求

以下为基础的编译器与工具链要求：

- CMake >= 3.28
- Ninja >= 1.12.0

推荐使用：

- Clang >= 10
- LLD >= 10 （使用LLVM LLD将显著提升构建速度）

#### 源码准备

1. 克隆主仓库（克隆后进入仓库目录，目录名通常为 `ascendnpu-ir`）：

```bash
git clone https://gitcode.com/Ascend/ascendnpu-ir.git
cd ascendnpu-ir
```

1. 初始化并更新子模块（Submodules）

本项目依赖LLVM、Torch-MLIR等三方库，需要拉取并更新到指定的commit id。

```bash
# 递归地拉取所有子模块
git submodule update --init --recursive
```

### 运行依赖

#### CANN包安装

AscendNPU-IR端到端运行依赖CANN环境。

1. 下载 CANN 包：需下载 toolkit 包及与硬件对应的 ops 包，可从 [昇腾社区 CANN 下载页](https://www.hiascend.com/cann/download) 获取。

2. 安装CANN包：

```bash
# 以x86系统A3环境，{version}为CANN版本，如9.0.0
chmod +x Ascend-cann_{version}_linux-x86_64.run
chmod +x Ascend-cann-A3-ops_{version}_linux-x86_64.run
./Ascend-cann_{version}_linux-x86_64.run --full [--install-path=${PATH-TO-CANN}]
./Ascend-cann-A3-ops_{version}_linux-x86_64.run --install [--install-path=${PATH-TO-CANN}]
```

3. 设置环境变量：

```bash
# 若是8.5.0及更早期的版本，路径为 ${PATH-TO-CANN}/ascend-toolkit/set_env.sh
source ${PATH-TO-CANN}/cann/set_env.sh
```

## 构建指令

### 源码安装

#### 使用提供的构建脚本（推荐）

在项目根目录下执行 `./build-tools/build.sh` 即可完成配置、构建和安装。脚本会自动处理 CMake 配置、Ninja 编译及安装步骤。

**首次构建**（需先初始化子模块并应用补丁）：

```bash
# 在项目根目录下
./build-tools/build.sh -o ./build --build-type Release --apply-patches
```

**后续构建**（已存在构建目录时）：

```bash
./build-tools/build.sh -o ./build --build-type Release
```

**重新构建**（清空构建目录并重新配置）：

```bash
./build-tools/build.sh -o ./build --build-type Release -r
```

#### 脚本参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-o`, `--build PATH` | 构建产物输出目录 | `./build` |
| `--build-type TYPE` | 构建类型 | `Release` |
| `--apply-patches` | 对三方子模块应用补丁，**首次构建必须启用** | 关闭 |
| `-r`, `--rebuild` | 清空构建目录并重新配置 | 关闭 |
| `-j`, `--jobs N` | 并行编译线程数 | CPU 核心数的 3/4 |
| `--install-prefix PATH` | 安装路径 | `BUILD_DIR/install` |
| `--c-compiler PATH` | C 编译器路径 | `clang` |
| `--cxx-compiler PATH` | C++ 编译器路径 | `clang++` |
| `--llvm-source-dir DIR` | LLVM 源码目录 | `third-party/llvm-project` |
| `--build-test` | 构建并运行测试 | 关闭 |
| `--build-bishengir-doc` | 构建 BiShengIR 文档 | 关闭 |
| `-t`, `--build-bishengir-template` | 构建 BiShengIR 模板库， **启用该选项需要安装CANN 9.0.0，运行端到端用例必须启用** | 关闭 |
| `--bisheng-compiler PATH` | bisheng编译器所在目录，**构建模板库时需指定** | 无 |
| `--build-torch-mlir` | 同时构建 torch-mlir | 关闭 |
| `--python-binding` | 启用 MLIR python-binding | 关闭 |
| `--enable-cpu-runner` | 启用 CPU runner | 关闭 |
| `--disable-ccache` | 禁用 ccache | 启用（若已安装） |
| `--enable-assertion` | 启用断言 | 关闭 |
| `--fast-build` | 跳过安装步骤 | 关闭 |
| `--add-cmake-options OPTIONS` | 追加 CMake 选项 | 无 |

#### 常用示例

```bash
# Debug 构建并运行测试
./build-tools/build.sh -o ./build --build-type Debug --apply-patches --build-test

# 指定编译器与线程数
./build-tools/build.sh -o ./build --c-compiler /usr/bin/clang-15 --cxx-compiler /usr/bin/clang++-15 -j 256

# 快速构建（不执行安装）
./build-tools/build.sh -o ./build --fast-build

# 重新构建并构建模板库（端到端用例执行依赖模板库）
./build-tools/build.sh -r -o ./build --fast-build -t --bisheng-compiler=/usr/Ascend/cann/bin
```

#### 手动构建（供高级用户参考）

若需完全手动控制构建流程，可参考以下步骤。

**前置条件**：已完成子模块初始化（`git submodule update --init --recursive`）及补丁应用（可执行 `source build-tools/apply_patches.sh`）。

```bash
# 在项目根目录下
mkdir -p build
cd build

# LLVM 源码路径：third-party/llvm-project/llvm
export LLVM_SOURCE_DIR="$(realpath ../third-party/llvm-project)"
cmake ${LLVM_SOURCE_DIR}/llvm -G Ninja \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_EXTERNAL_PROJECTS="bishengir" \
    -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR="$(realpath ..)" \
    -DBSPUB_DAVINCI_BISHENGIR=ON \
    # [-DCMAKE_INSTALL_PREFIX="${PWD}/install"] \
    # [-DLLVM_MAJOR_VERSION_21_COMPATIBLE=ON] \
    # [-DLLVM_ENABLE_ASSERTIONS=ON] \
    # [-DMLIR_ENABLE_BINDINGS_PYTHON=ON] \
    # [-DLLVM_TARGETS_TO_BUILD="host;Native"] \
    # [-DBISHENGIR_PUBLISH=OFF] \
    # [-DBISHENGIR_BUILD_TEMPLATE=ON -DBISHENG_COMPILER_PATH=/path/to/bisheng-compiler] \
    # [-D其他选项=值]

ninja -j32
```

**说明**：`[]` 表示可选项。使用某选项时，去掉行首 `# ` 和 `[]`，将参数加入命令。

| 可选参数 | 说明 |
|----------|------|
| `-DCMAKE_INSTALL_PREFIX="${PWD}/install"` | 安装路径 |
| `-DLLVM_MAJOR_VERSION_21_COMPATIBLE=ON` | LLVM 版本 ≥ 21 时需添加 |
| `-DLLVM_ENABLE_ASSERTIONS=ON` | 启用断言（Debug 时常用） |
| `-DMLIR_ENABLE_BINDINGS_PYTHON=ON` | 启用 MLIR python-binding |
| `-DLLVM_TARGETS_TO_BUILD="host;Native"` | 启用 CPU runner |
| `-DBISHENGIR_PUBLISH=OFF` | 关闭未发布功能 |
| `-DBISHENGIR_BUILD_TEMPLATE=ON -DBISHENG_COMPILER_PATH=...` | 构建 BiShengIR 模板库 |

### 二进制安装

#### 随 CANN 包安装

AscendNPU-IR二进制会随CANN toolkit包一起安装，参见上文「1.2.1 CANN 包安装」。

#### AscendNPU-IR 包单独安装

AscendNPU-IR有独立安装包以供使用。

1. 下载AscendNPU-IR安装包

```bash
# 下载对应版本的AscendNPU-IR安装包
```

1. 安装AscendNPU-IR

```bash
# 以x86系统环境，AscendNPU-IR 1.0.0版本为例
chmod +x ascendnpu-ir_1.0.0_linux-x86.run
./ascendnpu-ir_1.0.0_linux-x86.run --install [--install-path=${PATH-TO-ASCENDNPU-IR}]
```

### 环境变量设置

要使用AscendNPU-IR，需要将bishengir-compile可执行文件所在的路径加入到PATH环境变量中。

```bash
# 将bishengir-compile所在目录${PATH-TO-BISHENGIR-COMPILE}加入到PATH环境变量中
export PATH=${PATH-TO-ASCENDNPU-IR}/tools/bishengir/bin:$PATH
```

## 运行测试

### 编译测试Target

```bash
# 在 `build` 目录下
cmake --build . --target "check-bishengir"
```

该命令会执行 `check-mlir` 与 `check-bishengir`，通过 llvm-lit 运行 BiShengIR 测试套。

#### 典型输出示例

**测试通过时：**

```
-- Testing: 150 tests, 8 workers --
...

Testing Time: 45.23s
  Passed: 142
  Unsupported: 6
  Skipped: 2
```

**测试失败时：**

```
-- Testing: 150 tests, 8 workers --
PASS: bishengir :: bishengir-compile/commandline.mlir (1 of 150)
...
FAIL: bishengir :: test/failing-case.mlir (42 of 150)
********************
FAIL: bishengir :: test/failing-case.mlir
********************
...（失败详情）...

********************
Failed Tests (1):
  bishengir :: test/failing-case.mlir
********************
Testing Time: 38.12s
  Passed: 141
  Failed: 1
  ...
```

#### 测试通过判定

**测试通过**：命令退出码为 0，且 `Failed` 为 0。以下结果均计入通过：

- **PASS**：测试正常通过
- **UNSUPPORTED**：当前环境不支持（如 `UNSUPPORTED: bishengir_published`）
- **XFAIL**：预期失败且实际失败

**测试失败**：命令退出码非 0，或 `Failed` > 0。以下结果均计入失败：

- **FAIL**：测试执行失败
- **XPASS**：预期失败但实际通过
- **UNRESOLVED**：无法判定结果
- **TIMEOUT**：超时

### 使用LLVM-LIT执行测试套

```bash
# 在 `build` 目录下
./bin/llvm-lit ../bishengir/test
```

可直接指定测试路径，例如：`./bin/llvm-lit ../bishengir/test/bishengir-compile/commandline.mlir`。

## FAQ

Q：调用build-tools/build.sh脚本构建时，遇到报错"ninja: error: loading 'build.ninja': No such file or directory"应该如何处理？  
A：在调用build-tools/build.sh脚本时添加"-r"选项，重新执行CMake并生成新的build.ninja文件。

Q：构建时遇到报错"Too many open files"应该如何处理？  
A：文件同时打开数量超过了系统中配置的上限，可以通过"ulimit -n xxx"来修改文件同时打开数量上限，如"ulimit -n 65535"。

Q：构建时遇到报错

```bash
 The CMAKE_CXX_COMPILER:

  clang++

 is not a full path and was not found in the PATH.
```

应该如何处理？  
A：未指定C++编译器或C++编译器二进制存在问题，首先尝试通过"--cxx-compiler=${CXX-COMPILER-PATH}"指定要使用的C++编译器，如果已经指定了C++编译器仍然报错，则尝试重新安装或使用其他版本的C++编译器，如使用推荐的clang++-15。
