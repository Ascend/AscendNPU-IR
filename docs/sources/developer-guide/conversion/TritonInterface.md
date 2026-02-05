# Triton接入

[Triton Ascend](https://gitcode.com/Ascend/triton-ascend/) 是一个协助 Triton 接入 Ascend 平台的重要组件。完成 Triton Ascend 的构建与安装后，使用者在执行 Triton 算子时，即可选用 Ascend 作为后端。

## 安装指南

### 环境准备

#### Python版本要求

当前Triton-Ascend要求的Python版本为:**py3.9-py3.11**。

#### 安装Ascend CANN

异构计算架构CANN（Compute Architecture for Neural Networks）是昇腾针对AI场景推出的异构计算架构，
向上支持多种AI框架，包括MindSpore、PyTorch、TensorFlow等，向下服务AI处理器与编程，发挥承上启下的关键作用，是提升昇腾AI处理器计算效率的关键平台。

您可以访问昇腾社区官网，根据其提供的软件安装指引完成 CANN 的安装配置。

在安装过程中，CANN 版本“**{version}**”请选择如下版本之一：

**CANN版本：**

- 商用版

| Triton-Ascend版本 | CANN商用版本 | CANN发布日期 |
|-------------------|----------------------|--------------------|
| 3.2.0             | CANN 8.5.0           | 2026/01/16         |
| 3.2.0rc4          | CANN 8.3.RC2         | 2025/11/20         |
|                   | CANN 8.3.RC1         | 2025/10/30         |

- 社区版

| Triton-Ascend版本 | CANN社区版本 | CANN发布日期 |
|-------------------|----------------------|--------------------|
| 3.2.0             | CANN 8.5.0           | 2026/01/16         |
| 3.2.0rc4          | CANN 8.3.RC2         | 2025/11/20         |
|                   | CANN 8.5.0.alpha001  | 2025/11/12         |
|                   | CANN 8.3.RC1         | 2025/10/30         |

并根据实际环境指定CPU架构 “**{arch}**”(aarch64/x86_64)、软件版本“**{version}**”对应的软件包。

建议下载安装 8.5.0 版本:

| 软件类型    | 软件包说明       | 软件包名称                       |
|---------|------------------|----------------------------------|
| Toolkit | CANN开发套件包   | Ascend-cann-toolkit_**{version}**_linux-**{arch}**.run  |
| Ops     | CANN二进制算子包 | Ascend-cann-A3-ops_**{version}**_linux-**{arch}**.run |

注意1：A2系列的Ops包命名与A3略有区别，参考格式（ Ascend-cann-910b-ops_**{version}**_linux-**{arch}**.run ）

注意2：8.5.0之前的版本对应的Ops包的包名略有区别，参考格式（ Atlas-A3-cann-kernels_**{version}**_linux-**{arch}**.run ）

[社区下载链接](https://www.hiascend.com/developer/download/community/result?module=cann) 可以找到对应的软件包。

[社区安装指引链接](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit) 提供了完整的安装流程说明与依赖项配置建议，适用于需要全面部署 CANN 环境的用户。

**CANN安装脚本**

以8.5.0的A3 CANN版本为例，我们提供了脚本式安装供您参考：
```bash

# 更改run包的执行权限
chmod +x Ascend-cann-toolkit_8.5.0_linux-aarch64.run
chmod +x Ascend-cann-A3-ops_8.5.0_linux-aarch64.run

# 普通安装（默认安装路径：/usr/local/Ascend）
sudo ./Ascend-cann-toolkit_8.5.0_linux-aarch64.run --install
# 默认安装路径（与 Toolkit 包一致：/usr/local/Ascend）
sudo ./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install
# 生效默认路径环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 安装CANN的python依赖
pip install attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pyyaml
```

- 注：如果用户未指定安装路径，则软件会安装到默认路径下，默认安装路径如下。root用户：`/usr/local/Ascend`，非root用户：`${HOME}/Ascend`，${HOME}为当前用户目录。
上述环境变量配置只在当前窗口生效，用户可以按需将```source ${HOME}/Ascend/ascend-toolkit/set_env.sh```命令写入环境变量配置文件（如.bashrc文件）。


#### 安装torch_npu

当前配套的torch_npu版本为2.7.1版本。

```bash
pip install torch_npu==2.7.1
```

注：如果出现报错`ERROR: No matching distribution found for torch==2.7.1+cpu`，可以尝试手动安装torch后再安装torch_npu。
```bash
pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

### 通过pip安装Triton-Ascend

#### 最新稳定版本
您可以通过pip安装Triton-Ascend的最新稳定版本。

```shell
pip install triton-ascend
```

- 注：如果已经安装有社区Triton，请先卸载社区Triton。再安装Triton-Ascend，避免发生冲突。
```shell
pip uninstall triton
pip install triton-ascend
```

#### nightly build版本
我们为用户提供了每日更新的nightly包，用户可通过以下命令进行安装。

```shell
pip install -i https://test.pypi.org/simple/ "triton-ascend<3.2.0rc" --pre --no-cache-dir
```
同时用户也能在 [历史列表](https://test.pypi.org/project/triton-ascend/#history) 中找到所有的nightly build包。

注意，如果您在执行`pip install`时遇到ssl相关报错，可追加`--trusted-host test.pypi.org --trusted-host test-files.pythonhosted.org`选项解决。

### 通过源码安装Triton-Ascend

如果您需要对 Triton-Ascend 进行开发或自定义修改，则应采用源代码编译安装的方法。这种方式允许您根据项目需求调整源代码，并编译安装定制化的 Triton-Ascend 版本。

#### 系统要求

- GCC >= 9.4.0
- GLIBC >= 2.27

#### 依赖

**安装系统库依赖**

安装zlib1g-dev/lld/clang，可选择安装ccache包用于加速构建。

- 推荐版本 clang >= 15
- 推荐版本 lld >= 15

```bash
以ubuntu系统为例：
sudo apt update
sudo apt install zlib1g-dev clang-15 lld-15
sudo apt install ccache # optional
```

Triton-Ascend的构建强依赖zlib1g-dev，如果您使用yum源，请参考如下命令安装：

```bash
sudo yum install -y zlib-devel
```

**安装python依赖**

```bash
pip install ninja cmake wheel pybind11 # build-time dependencies
```

#### 基于LLVM构建

Triton 使用 LLVM20 为 GPU 和 CPU 生成代码。同样，昇腾的毕昇编译器也依赖 LLVM 生成 NPU 代码，因此需要编译 LLVM 源码才能使用。请关注依赖的 LLVM 特定版本。LLVM的构建支持两种构建方式，**以下两种方式二选一即可**，无需重复执行。

**代码准备: `git checkout` 检出指定版本的LLVM.**

   ```bash
   git clone --no-checkout https://github.com/llvm/llvm-project.git
   cd llvm-project
   git checkout b5cc222d7429fe6f18c787f633d5262fac2e676f
   ```

**方式一: clang构建安装LLVM**

- 步骤1：推荐使用clang安装LLVM，环境上请安装clang、lld，并指定版本（推荐版本clang>=15，lld>=15），
  如未安装，请按下面指令安装clang、lld、ccache：

  ```bash
  apt-get install -y clang-15 lld-15 ccache
  ```

- 步骤2：设置环境变量 LLVM_INSTALL_PREFIX 为您的目标安装路径：

   ```bash
   export LLVM_INSTALL_PREFIX=/path/to/llvm-install
   ```

- 步骤3：执行以下命令进行构建和安装LLVM：

  ```bash
  cd $HOME/llvm-project  # 用户git clone 拉取的 LLVM 代码路径
  mkdir build
  cd build
  cmake ../llvm \
    -G Ninja \
    -DCMAKE_C_COMPILER=/usr/bin/clang-15 \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++-15 \
    -DCMAKE_LINKER=/usr/bin/lld-15 \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
    -DLLVM_ENABLE_LLD=ON \
    -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX}
  ninja install
  ```

**方式二: GCC构建安装LLVM**

- 步骤1：推荐使用clang，如果只能使用GCC安装，请注意[注1](#note1) [注2](#note2)。设置环境变量 LLVM_INSTALL_PREFIX 为您的目标安装路径：

   ```bash
   export LLVM_INSTALL_PREFIX=/path/to/llvm-install
   ```

- 步骤2：执行以下命令进行构建和安装：

   ```bash
   cd $HOME/llvm-project  # your clone of LLVM.
   mkdir build
   cd build
   cmake -G Ninja  ../llvm  \
      -DLLVM_CCACHE_BUILD=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_ENABLE_PROJECTS="mlir;llvm"  \
      -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
      -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX}
   ninja install
   ```

<a id="note1"></a>注1：若在编译时出现错误`ld.lld: error: undefined symbol`，可在步骤2中加入设置`-DLLVM_ENABLE_LLD=ON`。

<a id="note2"></a>注2：若环境上ccache已安装且正常运行，可设置`-DLLVM_CCACHE_BUILD=ON`加速构建, 否则请勿开启。

**克隆 Triton-Ascend**

```bash
git clone https://gitcode.com/Ascend/triton-ascend.git && cd triton-ascend/python
```

**构建 Triton-Ascend**

1. 源码安装

- 步骤1：请确认已设置 [基于LLVM构建] 章节中，LLVM安装的目标路径 ${LLVM_INSTALL_PREFIX}
- 步骤2：请确认已安装clang>=15，lld>=15，ccache

   ```bash
   LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
   TRITON_BUILD_WITH_CCACHE=true \
   TRITON_BUILD_WITH_CLANG_LLD=true \
   TRITON_BUILD_PROTON=OFF \
   TRITON_WHEEL_NAME="triton-ascend" \
   TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
   python3 setup.py install
   ```

- 注3：推荐GCC >= 9.4.0，如果GCC < 9.4.0，可能报错 “ld.lld: error: unable to find library -lstdc++fs”，说明链接器无法找到 stdc++fs 库。
该库用于支持 GCC 9 之前版本的文件系统特性。此时需要手动把 CMake 文件中相关代码片段的注释打开：

- triton-ascend/CMakeLists.txt

   ```bash
   if (NOT WIN32 AND NOT APPLE)
   link_libraries(stdc++fs)
   endif()
   ```

  取消注释后重新构建项目即可解决该问题。

2. 运行Triton示例

   安装运行时依赖，参考如下：
   ```bash
   cd triton-ascend && pip install -r requirements_dev.txt
   ```
   运行实例: [01-vector-add.py](../../third_party/ascend/tutorials/01-vector-add.py)
   ```bash
   # 设置CANN环境变量（以root用户默认安装路径`/usr/local/Ascend`为例）
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   # 运行tutorials示例：
   python3 ./triton-ascend/third_party/ascend/tutorials/01-vector-add.py
   ```
    观察到类似的输出即说明环境配置正确。
    ```
    tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
    tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
    The maximum difference between torch and triton is 0.0
    ```

## Triton Op到Ascend NPU IR Op的转换

Triton Ascend将Triton方言的高级GPU抽象操作逐级下降为Linalg、HFusion和HIVM等目标方言，最终生成可在Ascend NPU上高效执行的优化中间表示。

### 访存类Op

**triton::StoreOp**将被下降为memref::copy

**triton::LoadOp**将被下降为memref::copy及bufferization::ToTensorOp

**triton::AtomicRMWOp**将被下降为hivm::StoreOp或hfusion::AtomicXchgOp

**triton::AtomicCASOp**将会首先被下降为linalg::GenericOp

**triton::GatherOp**将被下降为func::CallOp，调用函数 `triton_gather`

### 指针运算类Op

**triton::AddPtrOp**将被下降为memref::ReinterpretCast

**triton::PtrToIntOp**将被下降为arith::IndexCastOp

**triton::IntToPtrOp**将被下降为hivm::PointerCastOp

**triton::AdvanceOp**将被下降为memref::ReinterpretCastOp

### 程序信息类Op

**triton::GetProgramIdOp**将被下降为a param in functionOp

**triton::GetNumProgramsOp**将被下降为a param in functionOp

**triton::AssertOp**将被下降为func::CallOp，调用函数 `triton_assert`

**triton::PrintOp**将被下降为func::CallOp，调用函数 `triton_print`

### 张量操作类Op

**triton::ReshapeOp**将被下降为tensor::ReshapeOp

**triton::ExpandDimsOp**将被下降为tensor::ExpandShapeOp

**triton::BroadcastOp**将被下降为linalg::BroadcastOp

**triton::TransOp**将被下降为linalg::TransposeOp

**triton::SplitOp**将被下降为tensor::ExtractSliceOp

**triton::JoinOp**将被下降为tensor::InsertSliceOp

**triton::CatOp**将被下降为tensor::InsertSliceOp

**triton::MakeRangeOp**将会首先被下降为linalg::GenericOp

**triton::SplatOp**将被下降为linalg::FillOp

**triton::SortOp**将被下降为func::CallOp，调用函数 `triton_sort`

### 数值计算类Op

**triton::MulhiUIOp**将被下降为arith::MulSIExtendedOp

**triton::PreciseDivFOp**将被下降为arith::DivFOp

**triton::PreciseSqrtOp**将被下降为math::SqrtOp

**triton::BitcastOp**将被下降为arith::BitcastOp

**triton::ClampFOp**将被下降为tensor::EmptyOp及linalg::FillOp

**triton::DotOp**将被下降为linalg::MatmulOp

**triton::DotScaledOp**将被下降为linalg::MatmulOp

### 归约类Op

**triton::ArgMinOp**将被下降为linalg::ReduceOp

**triton::ArgMaxOp**将被下降为linalg::ReduceOp

**triton::ReduceOp**将被下降为linalg::ReduceOp

**triton::ScanOp**将被下降为func::CallOp，调用函数 `triton_cumsum`或 `triton_cumprod`

## Triton独有扩展操作

我们为triton提供了多个适用于Ascend的语言特性，若要使能相关能力，你需要import以下的模块

```py
import triton.language.extra.cann.extension as al
```

其后即可使用相关的Ascend Language (al) 独有接口

### 内存访问操作

#### copy_from_ub_to_l1
昇腾支持从统一缓冲区（UB）复制数据到L1缓冲区。

**参数说明**

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `src` | tl.tensor 或 bl.buffer | 位于统一缓冲区（UB）的源数据 |
| `dst` | tl.tensor 或 bl.buffer | 位于L1内存的目标缓冲区 |

**使用示例**

```py
copy_from_ub_to_l1(ub_buffer, l1_tensor)
```

#### fixpipe
昇腾支持通过fixpipe直接将L0C上的张量存储到本地缓冲区，实现L0C到其他内存层级的数移動。

**参数说明**

| 参数名 | 类型 | 描述 | 默认值 |
|--------|------|------|--------|
| `src` | tl.tensor | 源张量（必须位于L0C内存区域） | - |
| `dst` | bl.buffer | 目标缓冲区（必须位于UB内存区域） | - |
| `dma_mode` | FixpipeDMAMode | DMA传输模式，控制布局转换 | `FixpipeDMAMode.NZ2ND` |
| `dual_dst_mode` | FixpipeDualDstMode | 双目标模式 | `FixpipeDualDstMode.NO_DUAL` |

**使用示例**

```py
fixpipe(l0c_tensor, ub_buffer, 
        dma_mode=FixpipeDMAMode.NZ2DN, 
        dual_dst_mode=FixpipeDualDstMode.ROW_SPLIT)
```

### 同步与调试操作

#### debug_barrier

昇腾提供多个同步模式，支持向量流水线内部同步模式，用于调试和性能优化时的细粒度同步控制。

详细的模式说明见SYNC_IN_VF章节

##### 参数说明

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `sync_mode` | SYNC_IN_VF | 向量流水线同步模式 |

**使用示例**

```py
@triton.jit
def kernel_debug_barrier(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(core_mode="vector"):
        al.debug_barrier(al.SYNC_IN_VF.VV_ALL)
```


#### sync_block_set

昇腾支持在计算单元和向量单元之间的设置同步事件。

**参数说明**

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `sender` | str | 发送单元类型 | 
| `receiver` | str | 接收单元类型 |
| `event_id` | TensorHandle | 事件标识符 |
| `sender_pipe_value` | - | 发送管道值 |
| `receiver_pipe_value` | - | 接收管道值 |

**使用示例**

```py
al.sync_block_set("cube", "vector", 5, pipe.PIPE_MTE1, pipe.PIPE_MTE3)
```

#### sync_block_wait

昇腾支持等待计算单元和向量单元之间的同步事件。

**参数说明**

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `sender` | str | 发送单元类型 | 
| `receiver` | str | 接收单元类型 |
| `event_id` | TensorHandle | 事件标识符 |
| `sender_pipe_value` | - | 发送管道值 |
| `receiver_pipe_value` | - | 接收管道值 |

**使用示例**

```py
al.sync_block_wait("cube", "vector", 5, pipe.PIPE_MTE1, pipe.PIPE_MTE3)
```

#### sync_block_all
昇腾支持对整个计算块进行全局同步，确保所有指定类型的计算核心完成当前操作。

**参数说明**

| 参数名 | 类型 | 描述 | 有效值 |
|--------|------|------|--------|
| `mode` | str | 同步模式，指定要同步的核心类型 | `"all_cube"`, `"all_vector"`, `"all"`, `"all_sub_vector"` |
| `event_id` | int | 同步事件标识符 | `0` ~ `15` |

**同步模式详解**

| 模式 | 描述 | 同步范围 |
|------|------|----------|
| `"all_cube"` | 同步所有立方核心 | 当前AI Core上的所有立方计算核心 |
| `"all_vector"` | 同步所有向量核心 | 当前AI Core上的所有向量计算核心 |
| `"all"` | 同步所有核心 | 当前AI Core上的所有计算核心（立方+向量） |
| `"all_sub_vector"` | 同步所有子向量核心 | 当前AI Core上的所有子向量核心 |

**使用示例**

```py
al.sync_block_all(mode="all_cube", event_id=0)
```

### 硬件查询与控制操作

#### sub_vec_id
昇腾支持获取当前AI Core上的向量核心索引。

**使用示例**

```py
vec_id = sub_vec_id()
```

#### sub_vec_num
昇腾支持获取单个AI Core上的向量核心数量。

**使用示例**

```py
vector_core_count = sub_vec_num()
```

#### scope
昇腾支持作用域管理器，用于创建具有共享特性的操作区块，可显式指定计算核心类型。

**参数说明**

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `core_mode` | str | 核心类型，指定区块内操作使用的计算核心， 只接受`"cube"`或`"vector"`两种模式 |

**核心模式选项**
| 模式 | 描述 |
|------|------|
| `"cube"` | 使用矩阵立方核心进行计算 |
| `"vector"` | 使用向量核心进行计算 |

**使用示例**

```py
@triton.jit
def kernel_scope_escape(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(core_mode="vector"):
        x = tl.load(x_ptr + i, mask=i < n)
```

#### parallel

昇腾扩展了 Python 标准的 `range` 功能，增加具有**平行执行语义**和**编译器提示**的`parallel`迭代器。

**参数说明**

| 参数 | 类型 | 说明 | 范例 |
|------|------|------|------|
| `arg1` | int | 起始值或结束值 | `parallel(10)` |
| `arg2` | int | 结束值 (可选) | `parallel(0, 10)` |
| `step` | int | 步长 (可选) | `parallel(0, 10, 2)` |
| `num_stages` | int | 流水线阶段数 (可选) | `parallel(0, 10, num_stages=3)` |
| `loop_unroll_factor` | int | 循环展开因子 (可选) | `parallel(0, 10, loop_unroll_factor=4)` |
| `bind_sub_block` | bool | `False` | 控制是否让多个向量核心参与循环执行 |

**限制**

目前 910B 最多支持2个向量核心

**使用示例**

```py
for i in triton.language.parallel(128, bind_sub_block=True):
    computation(i)
```

### 编译优化提示

#### compile_hint

昇腾支持向编译器传递优化提示信息，指导代码生成和性能优化。

**参数说明**

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `ptr` | tensor | 目标张量指针 |
| `hint_name` | str | 提示名称 |
| `hint_val` | 多种类型 | 提示值（可选） |

**使用示例**

```py
al.compile_hint(x0, "multi_buffer", 2)
al.compile_hint(x1, "bitwise_mask")
```

#### multibuffer

`multibuffer` 是用于为现有张量设置多重缓冲（Double Buffering）的函数，通过编译器提示优化数据流和计算重叠。

**参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| `src` | tensor | 要进行多重缓冲化的张量 |
| `size` | int | 缓冲副本的数量 |

**使用示例**

```py
multibuffer(x, 2)
```

### 自定义扩展

#### custom
昇腾支持调用自定义操作，允许开发者注册和使用硬件特定的扩展功能。

**参数说明**

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `name` | str | 自定义操作的名称（必须已注册） |
| `*args` | 可变参数 | 传递给自定义操作的参数 |
| `**kwargs` | 关键字参数 | 传递给自定义操作的关键字参数 |

**使用示例**

```py
@al.register_custom_op
class my_custom_op:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT

@triton.jit
def my_kernel(
    # params
):
    result = al.custom("my_custom_op", 
                       x,
                       x_ptr,
                       y_ptr + i,
                       (1, 2, 3),
                       [4.1, 5.2],
                       out=y)
```

### 张量切片操作

#### insert_slice
昇腾支持根据操作的偏移量、大小和步长参数，将一个张量插入到另一个张量中。

**参数说明**

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `ful` | Tensor | 接收插入的目标张量 |
| `sub` | Tensor | 要被插入的源张量 |
| `offsets` | 整数元组 | 插入操作的起始偏移量 |
| `sizes` | 整数元组 | 插入操作的大小范围 |
| `strides` | 整数元组 | 插入操作的步长参数 |

**使用示例**

```py
    result = insert_slice(
        ful=ful,
        sub=sub,
        offsets=offsets,
        sizes=sizes,
        strides=strides
    )
```

#### extract_slice
昇腾支持根据操作的偏移量、大小和步长参数，从另一个张量中提取指定的切片。

**参数说明**

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `ful` | Tensor | 要被提取切片的源张量 |
| `offsets` | 整数元组 | 提取操作的起始偏移量 |
| `sizes` | 整数元组 | 提取操作的大小范围 |
| `strides` | 整数元组 | 提取操作的步长参数 |

**使用示例**

```py
    result = extract_slice(
        ful=ful,
        offsets=offsets,
        sizes=sizes,
        strides=strides
    )
```

#### get_element
昇腾支持从张量中读取指定索引位置的单个元素值。

**参数说明**

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `src` | tensor | 要访问的源张量 |
| `indice` | int元组 | 指定要获取元素的索引位置 |

**使用示例**

```py
element = get_element(src, indice=(1, 2))
```

### 张量计算操作

#### sort
昇腾支持对输入张量沿指定维度进行排序操作。

**参数说明**

| 参数名 | 类型 | 描述 | 默认值 |
|--------|------|------|--------|
| `ptr` | tensor | 输入张量 | - |
| `dim` | int 或 tl.constexpr[int] | 要排序的维度 | `-1` |
| `descending` | bool 或 tl.constexpr[bool] | 排序方向，`True`表示降序，`False`表示升序 | `False` |

**使用示例**

```py
sorted_feature = sort(input_tensor, dim=2, descending=True)
```

#### flip
昇腾支持对输入张量沿指定维度进行翻转操作。

**参数说明**

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `ptr` | tensor | 输入张量 |
| `dim` | int 或 tl.constexpr[int] | 要翻转的维度 |

**使用示例**

```py
flipped_feature = flip(input_tensor, dim=2)
```

#### cast
昇腾支持将张量转换为指定的数据类型，支持数值转换、位转换和溢出处理。

**参数说明**

| 参数名 | 类型 | 描述 | 默认值 |
|--------|------|------|--------|
| `input` | tensor | 输入张量 | - |
| `dtype` | dtype | 目标数据类型 | - |
| `fp_downcast_rounding` | str, 可选 | 浮点数向下转换时的舍入模式 | `None` |
| `bitcast` | bool, 可选 | 是否进行位转换（而非数值转换） | `False` |
| `overflow_mode` | str, 可选 | 溢出处理模式 | `None` |

**使用示例**

```py
int8_tensor = cast(input_int32, dtype=tl.int8, overflow_mode="saturate")
```

### 索引与收集操作

#### _index_select
昇腾支持从源GM张量中根据索引UB张量在指定维度上进行数据收集操作，使用SIMT模板将值收集到输出UB张量中。此操作支持2D–5D张量。

**参数说明**

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `src` | pointer type | 源张量指针（位于GM中） |
| `index` | tensor | 用于收集的索引张量（位于UB中） |
| `dim` | int | 沿其进行收集的维度 |
| `bound` | int | 索引值的上边界 |
| `end_offset` | int元组 | 索引张量每个维度的结束偏移量 |
| `start_offset` | int元组 | 源张量每个维度的起始偏移量 |
| `src_stride` | int元组 | 源张量每个维度的步长 |
| `other` (可选) | scalar value | 当索引越界时的默认值（位于UB中） |
| `out` | tensor | 输出张量（位于UB中） |

**使用示例**

```py
_index_select(
    src=src_3d_ptr,
    index=index_2d_tile,
    dim=1,
    bound=50,
    end_offset=(2, 4, 64),
    start_offset=(0, 8, 0),
    src_stride=(256, 64, 1),
    other=0.0
)
```

#### index_put
昇腾支持将根据索引张量将值张量放置到目标张量中。

**参数说明**

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `ptr` | tensor (指针类型) | 目标张量指针（位于GM中） |
| `index` | tensor | 用于放置的索引（位于UB中） |
| `value` | tensor | 要储存的值（位于UB中） |
| `dim` | int32 | 沿其进行索引放置的维度 |
| `index_boundary` | int64 | 索引值的上边界 |
| `end_offset` | int元组 | 每个维度放置区域的结束偏移量 |
| `start_offset` | int元组 | 每个维度放置区域的起始偏移量 |
| `dst_stride` | int元组 | 目标张量每个维度的步长 |

**索引放置规则**

- 二维索引放置
    - dim = 0: `out[index[i]][start_offset[1]:end_offset[1]] = value[i][0:end_offset[1]-start_offset[1]]`

- 三维索引放置
    - dim = 0: `out[index[i]][start_offset[1]:end_offset[1]][start_offset[2]:end_offset[2]]  = value[i][0:end_offset[1]-start_offset[1]][0:end_offset[2]-start_offset[2]]`
    - dim = 1: `out[start_offset[0]:end_offset[0]][index[j]][start_offset[2]:end_offset[2]] = value[0:end_offset[0]-start_offset[0]][j][0:end_offset[2]-start_offset[2]]`

**约束条件**

- `ptr` 和 `value` 必须具有相同的秩
- `ptr.dtype` 目前仅支持 `float16`、`bfloat16`、`float32`
- `index` 必须是整数张量。如果 `index.rank` != 1，将被重塑为1D
- `index.numel` 必须等于 `value.shape[dim]`
- `value` 支持 2~5 维张量
- `dim` 必须有效（0 ≤ dim < rank(value) - 1）

**使用示例**

```py
index_put(
    ptr=dst_ptr,
    index=index_tile,
    value=value_tile,
    dim=0,
    index_boundary=4,
    end_offset=(2, 2),
    start_offset=(0, 0),
    dst_stride=(2, 1)
)
```

#### gather_out_to_ub
昇腾支持沿指定维度从GM中散点收集数据到UB中，此操作支持索引边界检查，确保高效且安全的数据搬运。

**参数说明**

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `src` | tensor (指针类型) | 源张量指针（位于GM中） |
| `index` | tensor | 用于收集的索引张量（位于UB中） |
| `index_boundary` | int64 | 索引值的上边界 |
| `dim` | int32 | 沿其进行收集的维度 |
| `src_stride` | int64元组 | 源张量每个维度的步长 |
| `end_offset` | int32元组 | 索引张量每个维度的结束偏移量 |
| `start_offset` | int32元组 | 索引张量每个维度的起始偏移量 |
| `other` | 标量值 (可选) | 当索引越界时使用的默认值（位于UB中） |

**返回值**

- **类型**: tensor
- **描述**: 位于UB中的结果张量，形状与 `index.shape` 相同


**散点收集规则**

-  一维索引收集
    - dim = 0: `out[i] = src[start_offset[0] + index[i]]`

- 二维索引收集
    - dim = 0: `out[i][j] = src[start_offset[0] + index[i][j]][start_offset[1] + j]`
    - dim = 1: `out[i][j] = src[start_offset[0] + i][start_offset[1] + index[i][j]]`

- 三维索引收集
    - dim = 0: `out[i][j][k] = src[start_offset[0] + index[i][j][k]][start_offset[1] + j][start_offset[2] + k]`
    - dim = 1: `out[i][j][k] = src[start_offset[0] + i][start_offset[1] + index[i][j][k]][start_offset[2] + k]`
    - dim = 2: `out[i][j][k] = src[start_offset[0] + i][start_offset[1] + j][start_offset[2] + index[i][j][k]]`

**约束条件**

- `src` 和 `index` 必须具有相同的秩
- `src.dtype` 目前仅支持 `float16`、`bfloat16`、`float32`
- `index` 必须是整数张量，秩在 1 到 5 之间
- `dim` 必须有效（0 ≤ dim < rank(index)）
- `other` 必须是标量值
- 对于每个不等于 `dim` 的维度 `i`，`index.size[i]` ≤ `src.size[i]`
- 输出形状与 `index.shape` 相同。如果 `index` 为 None，输出张量将是与 `index` 形状相同的空张量

**使用示例**

```py
gathered = gather_out_to_ub(
    src=src_ptr,
    index=index,
    index_boundary=4,
    dim=0,
    src_stride=(2, 1),
    end_offset=(2, 2),
    start_offset=(0, 0)
)
```

#### scatter_ub_to_out
昇腾支持沿指定维度从UB中散点储存数据到GM中，此操作支持索引边界检查，确保高效且安全的数据搬运。

**参数说明**

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `ptr` | tensor (指针类型) | 目标张量指针（位于GM中） |
| `value` | tensor | 要储存的图块值（位于UB中） |
| `index` | tensor | 散点储存使用的索引（位于UB中） |
| `index_boundary` | int64 | 索引值的上边界 |
| `dim` | int32 | 沿其进行散点储存的维度 |
| `dst_stride` | int64元组 | 目标张量每个维度的步长 |
| `end_offset` | int32元组 | 索引张量每个维度的结束偏移量 |
| `start_offset` | int32元组 | 索引张量每个维度的起始偏移量 |

**散点储存规则**

-  一维索引散点
    - dim = 0: `out[start_offset[0] + index[i]] = value[i]`

- 二维索引散点
    - dim = 0: `out[start_offset[0] + index[i][j]][start_offset[1] + j] = value[i][j]`
    - dim = 1: `out[start_offset[0] + i][start_offset[1] + index[i][j]] = value[i][j]`

- 三维索引散点
    - dim = 0: `out[start_offset[0] + index[i][j][k]][start_offset[1] + j][start_offset[2] + k] = value[i][j][k]`
    - dim = 1: `out[start_offset[0] + i][start_offset[1] + index[i][j][k]][start_offset[2] + k] = value[i][j][k]`
    - dim = 2: `out[start_offset[0] + i][start_offset[1] + j][start_offset[2] + index[i][j][k]] = value[i][j][k]`

**约束条件**

- `ptr`、`index` 和 `value` 必须具有相同的秩
- `ptr.dtype` 目前仅支持 `float16`、`bfloat16`、`float32`
- `index` 必须是整数张量，秩在 1 到 5 之间
- `dim` 必须有效（0 ≤ dim < rank(index)）
- 对于每个不等于 `dim` 的维度 `i`，`index.size[i]` ≤ `ptr.size[i]`
- 输出形状与 `index.shape` 相同。如果 `index` 为 None，输出张量将是与 `index` 形状相同的空张量

**使用示例**

```py
scatter_ub_to_out(
    ptr=dst_ptr,
    value=value,
    index=index,
    index_boundary=4,
    dim=0,
    dst_stride=(2, 1),
    end_offset=(2, 2),
    start_offset=(0, 0)
)
```

#### index_select_simd
昇腾支持平行索引选择操作，从GM多点选取数据直接载入到UB，实现零拷贝高效读取。

**参数说明**

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `src` | tensor (pointer type) | 源张量指针（位于GM中） |
| `dim` | int 或 constexpr | 沿其选择索引的维度 |
| `index` | tensor | 要选择的索引的一维张量（位于UB中） |
| `src_shape` | List[Union[int, tensor]] | 源张量的完整形状（可以是整数或张量） |
| `src_offset` | List[Union[int, tensor]] | 读取的起始偏移量（可以是整数或张量） |
| `read_shape` | List[Union[int, tensor]] | 要读取的大小（图块形状，可以是整数或张量） |

**约束条件**
- `read_shape[dim]` 必须为 `-1`
- `src_offset[dim]` 可以为 `-1`（将被忽略）
- 边界处理：当 `src_offset + read_shape > src_shape` 时，会自动截断到 `src_shape` 边界
- **不检查** `index` 是否包含越界值

**返回值**
- **返回类型**: tensor
- **描述**: 位于UB中的结果张量，其形状中的 `dim` 维度被替换为 `index` 的长度

**使用示例**
```py
result = al.index_select_simd(
    src_ptr,
    dim=1,
    index=indices,
    src_shape=[8, 100, 256],
    src_offset=[4, -1, 128],
    read_shape=[4, -1, 128]
)
```

## Triton独有扩展枚举

#### SYNC_IN_VF

| 枚举值 | 描述 |
|--------|----------|
| `VV_ALL` | 阻塞向量加载/存储指令的执行，直到所有向量加载/存储指令完成 |
| `VST_VLD` | 阻塞向量加载指令的执行，直到所有向量存储指令完成 |
| `VLD_VST` | 阻塞向量存储指令的执行，直到所有向量加载指令完成 |
| `VST_VST` | 阻塞向量存储指令的执行，直到所有向量存储指令完成 |
| `VS_ALL` | 阻塞标量加载/存储指令的执行，直到所有向量加载/存储指令完成 |
| `VST_LD` | 阻塞标量加载指令的执行，直到所有向量存储指令完成 |
| `VLD_ST` | 阻塞标量存储指令的执行，直到所有向量加载指令完成 |
| `VST_ST` | 阻塞标量存储指令的执行，直到所有向量存储指令完成 |
| `SV_ALL` | 阻塞向量加载/存储指令的执行，直到所有标量加载/存储指令完成 |
| `ST_VLD` | 阻塞向量加载指令的执行，直到所有标量存储指令完成 |
| `LD_VST` | 阻塞向量存储指令的执行，直到所有标量加载指令完成 |
| `ST_VST` | 阻塞向量存储指令的执行，直到所有标量存储指令完成 |

#### FixpipeDMAMode

| 枚举值 | 描述 |
|--------|------|
| `NZ2DN` | 非零存储格式到行列主序格式的数据转换 |
| `NZ2ND` | 非零存储格式到列行主序格式的数据转换 |
| `NZ2NZ` | 非零存储格式之间的数据转换（保持原格式） |

#### FixpipeDualDstMode

| 枚举值 | 描述 |
|--------|------|
| `NO_DUAL` | 不使用双目标模式，数据写入单一目标 |
| `COLUMN_SPLIT` | 列分割双目标模式，按列将数据分割到两个目标 |
| `ROW_SPLIT` | 行分割双目标模式，按行将数据分割到两个目标 |

#### FixpipePreQuantMode

| 枚举值 | 描述 |
|--------|------|
| `NO_QUANT` | 不进行预量化处理，保持原始数据格式 |
| `F322BF16` | 浮点32位到bfloat16格式的量化转换 |
| `F322F16` | 浮点32位到浮点16位格式的量化转换 |
| `S322I8` | 有符号32位整数到8位整数格式的量化转换 |

#### FixpipePreReluMode

| 枚举值 | 描述 |
|--------|------|
| `LEAKY_RELU` | Leaky ReLU激活函数处理 |
| `NO_RELU` | 不进行ReLU激活处理 |
| `NORMAL_RELU` | 标准ReLU激活函数处理 |
| `P_RELU` | Parametric ReLU激活函数处理 |

#### CORE

| 枚举值 | 描述 |
|--------|------|
| `VECTOR` | 向量计算核心 |
| `CUBE` | 矩阵立方计算核心 |
| `CUBE_OR_VECTOR` | 立方或向量计算核心（二选一） |
| `CUBE_AND_VECTOR` | 立方和向量计算核心（混合使用） |

#### MODE

| 枚举值 | 描述 |
|--------|------|
| `SIMD` | 单指令多数据执行模式 |
| `SIMT` | 单指令多线程执行模式 |
| `MIX` | 混合执行模式 |

#### PIPE

| 枚举值 | 描述 |
|--------|------|
| `PIPE_S` | 标量计算流水线 |
| `PIPE_V` | 向量计算流水线 |
| `PIPE_M` | 内存操作流水线 |
| `PIPE_MTE1` | 内存传输引擎1流水线 |
| `PIPE_MTE2` | 内存传输引擎2流水线 |
| `PIPE_MTE3` | 内存传输引擎3流水线 |
| `PIPE_ALL` | 所有流水线 |
| `PIPE_FIX` | 固定功能流水线 |

## Triton独有扩展装饰器

#### register_custom_op

昇腾内置函数装饰器，用于注册自定义操作，以便能在 al.custom() 中调用。
