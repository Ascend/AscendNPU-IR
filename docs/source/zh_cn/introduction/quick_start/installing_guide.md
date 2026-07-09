# 构建安装

本文说明`AscendNPU IR`的依赖安装、构建方式（源码/二进制）及运行测试步骤。

## 安装依赖

### 构建依赖

#### 编译器与工具链要求

以下为基础的编译器与工具链要求：

- `CMake` >= `3.28`
- `Ninja` >= `1.12.0`

推荐使用：

- `Clang` >= `10`
- `LLD` >= `10`（使用`LLVM LLD`将显著提升构建速度）

#### 源码准备

1. 克隆主仓库（克隆后进入仓库目录，目录名通常为`ascendnpu-ir`）。

   ```bash
   git clone https://gitcode.com/Ascend/ascendnpu-ir.git
   cd ascendnpu-ir
   ```

2. 初始化并更新子模块（`Submodules`）。

   本项目依赖`LLVM`、`Torch-MLIR`等三方库，需要拉取并更新到指定的`commit id`。

   ```bash
   # 递归地拉取所有子模块
   git submodule update --init --recursive
   ```

### 运行依赖

#### CANN包安装

`AscendNPU IR`端到端运行依赖`CANN`环境。

1. 下载`CANN`包：需下载`toolkit`包及与硬件对应的`ops`包，可从[昇腾社区CANN下载页](https://www.hiascend.com/cann/download)获取。

2. 安装`CANN`包：

   ```bash
   # 以 x86 系统 A3 环境，{version} 为 CANN 版本，如 9.0.0
   chmod +x Ascend-cann_{version}_linux-x86_64.run
   chmod +x Ascend-cann-A3-ops_{version}_linux-x86_64.run
   ./Ascend-cann_{version}_linux-x86_64.run --full [--install-path=${PATH-TO-CANN}]
   ./Ascend-cann-A3-ops_{version}_linux-x86_64.run --install [--install-path=${PATH-TO-CANN}]
   ```

3. 设置环境变量：

   ```bash
   # 若是 8.5.0 及更早期的版本，路径为 ${PATH-TO-CANN}/ascend-toolkit/set_env.sh
   source ${PATH-TO-CANN}/cann/set_env.sh
   ```

## 构建指令

### 源码安装

#### 使用提供的构建脚本（推荐）

在项目根目录下执行`./build-tools/build.sh`即可完成配置、构建和安装。脚本会自动处理`CMake`配置、`Ninja`编译及安装步骤。

**首次构建**（需先初始化子模块）：

```bash
# 在项目根目录下
./build-tools/build.sh -o ./build --build-type Release
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
| `-r`, `--rebuild` | 清空构建目录并重新配置 | 关闭 |
| `-j`, `--jobs N` | 并行编译线程数 | `CPU`核心数的`3/4` |
| `--install-prefix PATH` | 安装路径 | `BUILD_DIR/install` |
| `--c-compiler PATH` | `C`编译器路径 | `clang` |
| `--cxx-compiler PATH` | `C++`编译器路径 | `clang++` |
| `--llvm-source-dir DIR` | `LLVM`源码目录 | `third-party/llvm-project` |
| `--build-test` | 构建并运行测试 | 关闭 |
| `--build-bishengir-doc` | 构建`BiShengIR`文档 | 关闭 |
| `-t`, `--build-bishengir-template` | 构建`BiShengIR`模板库，**启用该选项需要安装CANN 9.0.0，运行端到端用例必须启用** | 关闭 |
| `--bisheng-compiler PATH` | `bisheng`编译器所在目录，**构建模板库时需指定** | 无 |
| `--build-torch-mlir` | 同时构建`torch-mlir` | 关闭 |
| `--python-binding` | 启用`MLIR python-binding` | 关闭 |
| `--enable-cpu-runner` | 启用`CPU runner` | 关闭 |
| `--disable-ccache` | 禁用`ccache` | 启用（若已安装） |
| `--enable-assertion` | 启用断言 | 关闭 |
| `--fast-build` | 跳过安装步骤 | 关闭 |
| `--add-cmake-options OPTIONS` | 追加`CMake`选项 | 无 |

#### 常用示例

```bash
# Debug 构建并运行测试
./build-tools/build.sh -o ./build --build-type Debug --build-test

# 指定编译器与线程数
./build-tools/build.sh -o ./build --c-compiler /usr/bin/clang-15 --cxx-compiler /usr/bin/clang++-15 -j 256

# 快速构建（不执行安装）
./build-tools/build.sh -o ./build --fast-build

# 重新构建并构建模板库（端到端用例执行依赖模板库）
./build-tools/build.sh -r -o ./build --fast-build -t --bisheng-compiler=/usr/Ascend/cann/bin
```

#### 手动构建（供高级用户参考）

若需完全手动控制构建流程，可参考以下步骤。

**前置条件**：已完成子模块初始化（`git submodule update --init --recursive`）。

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

**说明**：`[]`表示可选项。使用某选项时，去掉行首`#`和`[]`，将参数加入命令。

| 可选参数 | 说明 |
|----------|------|
| `-DCMAKE_INSTALL_PREFIX="${PWD}/install"` | 安装路径 |
| `-DLLVM_MAJOR_VERSION_21_COMPATIBLE=ON` | `LLVM`版本 ≥ `21`时需添加 |
| `-DLLVM_ENABLE_ASSERTIONS=ON` | 启用断言（`Debug`时常用） |
| `-DMLIR_ENABLE_BINDINGS_PYTHON=ON` | 启用`MLIR python-binding` |
| `-DLLVM_TARGETS_TO_BUILD="host;Native"` | 启用`CPU runner` |
| `-DBISHENGIR_PUBLISH=OFF` | 关闭未发布功能 |
| `-DBISHENGIR_BUILD_TEMPLATE=ON -DBISHENG_COMPILER_PATH=...` | 构建`BiShengIR`模板库 |

### 二进制安装

`AscendNPU IR`二进制会随`CANN toolkit`包一起安装，参见上文[CANN包安装](#cann包安装)。

## 直接使用Docker镜像

使用`Docker`镜像进行开发与验证，无需配置环境。

### 复用CANN镜像

`CANN Toolkit`包中会包含完整的`AscendNPU IR`二进制，`Docker`镜像可以复用`CANN`的镜像。

可以在[https://quay.io/repository/ascend/cann?tab=tags](https://quay.io/repository/ascend/cann?tab=tags)中根据硬件平台和`CANN`版本查找相关的标签使用。例如`ascend/cann:9.0.0-a3-openeuler24.03-py3.12`

### 构建本地镜像

开发者也可以独立构建本地镜像，可参考`docker`目录中内置的不同架构的`Dockerfile`。
对于不同架构，均包含`Ascend CANN Toolkit`包与最新编译的`AscendNPU IR`。可自行安装`ops`包与`Torch`相关组件体验更多内容。
不同架构下的基础镜像信息为：

- `x86_64`: 基于`ubuntu:22.04`
- `aarch64`: 基于`openeuler/openeuler:24.03`

构建方式：

```bash
# 构建 x86_64 镜像
docker build -t ascendnpu-ir:latest -f docker/Dockerfile.x86_64 .
```

### 镜像使用

```bash
IMAGE_NAME="ascendnpu-ir:latest"       # CANN 镜像可以为 `ascend/cann:9.0.0-a3-openeuler24.03-py3.12`
# 进入后的环境可直接使用 AscendNPU IR 相关工具，如 bisheng-compile 等
docker run -it \
  --net=host --privileged \            # 主机模式开发
  --security-opt seccomp=unconfined \  # 关闭安全限制
  --device=/dev/davinci0 \             # 挂载 NPU 设备
  --device=/dev/davinci1 \
  --device=/dev/davinci2 \
  --device=/dev/davinci3 \
  --device=/dev/davinci4 \
  --device=/dev/davinci5 \
  --device=/dev/davinci6 \
  --device=/dev/davinci7 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \ # 挂载 dcmi 等设备
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  --name ascendnpu-ir   \              # 容器名称
  -v $(pwd):/workspace  \              # 挂载当前目录到容器
  -w /workspace         \
  $IMAGE_NAME  /bin/bash

bishengir-compile --version # 可以看到 AscendNPU IR 的版本信息
```

## 运行测试

### 编译测试Target

```bash
# 在 build 目录下
cmake --build . --target "check-mlir;check-bishengir"
```

该命令会执行`check-mlir`与`check-bishengir`，通过`llvm-lit`运行`BiShengIR`测试套。

#### 典型输出示例

**测试通过时**：

```text
-- Testing: 388 tests, 8 workers --
...

Testing Time: 45.23s

Total Discovered Tests: 388
  Unsupported: 89  (22.94%)
  Passed     : 299 (77.06)
```

**测试失败时**：

```text
-- Testing: 388 tests, 8 workers --
PASS: bishengir :: bishengir-compile/commandline.mlir (1 of 388)
...
FAIL: bishengir :: test/failing-case.mlir (42 of 388)
******************** TEST 'FAIL: bishengir :: test/failing-case1.mlir' FAILED ********************
...（失败详情）...

********************
FAIL: bishengir :: test/failing-case.mlir (256 of 388)
******************** TEST 'FAIL: bishengir :: test/failing-case2.mlir' FAILED ********************
...（失败详情）...

********************
********************
Failed Tests (2):
  bishengir :: test/failing-case1.mlir
  bishengir :: test/failing-case2.mlir

Testing Time: 38.12s

Total Discovered Tests: 388
  Unsupported:  86 (22.16%)
  Passed     : 300 (77.32%)
  Failed     :   2 (0.52%)
  ...
```

#### 测试通过判定

**测试通过**：命令退出码为`0`，且`Failed`为`0`。以下结果均计入通过：

- **PASS**：测试正常通过
- **UNSUPPORTED**：当前环境不支持（如`UNSUPPORTED: bishengir_published`）
- **XFAIL**：预期失败且实际失败

**测试失败**：命令退出码非`0`，或`Failed` > `0`。以下结果均计入失败：

- **FAIL**：测试执行失败
- **XPASS**：预期失败但实际通过
- **UNRESOLVED**：无法判定结果
- **TIMEOUT**：超时

### 使用LLVM-LIT执行测试套

```bash
# 在 build 目录下
./bin/llvm-lit ../bishengir/test
```

可直接指定测试路径，例如：`./bin/llvm-lit ../bishengir/test/bishengir-compile/commandline.mlir`。

## FAQ

Q：调用`build-tools/build.sh`脚本构建时，遇到报错`ninja: error: loading 'build.ninja': No such file or directory`应该如何处理？

A：在调用`build-tools/build.sh`脚本时添加`-r`选项，重新执行`CMake`并生成新的`build.ninja`文件。

Q：构建时遇到报错`Too many open files`应该如何处理？

A：文件同时打开数量超过了系统中配置的上限，可以通过`ulimit -n xxx`来修改文件同时打开数量上限，如`ulimit -n 65535`。

Q：构建时遇到报错

```bash
 The CMAKE_CXX_COMPILER:

 clang++

 is not a full path and was not found in the PATH.
```

应该如何处理？

A：未指定`C++`编译器或`C++`编译器二进制存在问题，首先尝试通过`--cxx-compiler=${CXX-COMPILER-PATH}`指定要使用的`C++`编译器，如果已经指定了`C++`编译器仍然报错，则尝试重新安装或使用其他版本的`C++`编译器，如使用推荐的`clang++-15`。
