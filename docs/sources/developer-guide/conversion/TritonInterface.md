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

### 调用Triton Kernel

在成功安装Triton-Ascend后，你可以尝试调用相关的Triton Kernel，参考以下源码，你可以通过执行`pytest -sv <file>.py`验证安装后的功能正确性。

```python
import pytest
import triton
import triton.language as tl
import torch
import torch_npu
import test_common


def torch_lt(x0, x1):
    return x0 < x1


@triton.jit
def triton_lt(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop1 in range(loops1):
        x_index = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + x_index, None)
        tmp1 = tl.load(in_ptr1 + x_index, None)
        tmp2 = tmp0 < tmp1
        tl.store(out_ptr0 + x_index, tmp2, None)


@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (32,), 1, 32, 32],
                         ])
def test_lt(param_list):
    # 生成数据
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    x1 = test_common.generate_tensor(shape, dtype).npu()
    # torch结果
    torch_res = torch_lt(x0, x1).to(eval('torch.' + dtype))
    # triton结果
    triton_res = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_lt[ncore, 1, 1](x0, x1, triton_res, xblock, xblock_sub)
    # 比较结果
    test_common.validate_cmp(dtype, triton_res, torch_res)

```

**动态tiling支持：** 通过[]内的grid参数配置并行粒度，通过XBLOCK和XBLOCK_SUB参数控制tiling大小，用户可按需调整。

**动态shape支持：** kernel自动适配任意长度的1D tensor，用户只需传入实际shape的数据即可

## Triton Op到Ascend NPU IR Op的转换

Triton Ascend将Triton方言的高级GPU抽象操作逐级下降为Linalg、HFusion和HIVM等目标方言，最终生成可在Ascend NPU上高效执行的优化中间表示。下表详细列出了各类Triton操作与其在下降过程中所对应的Ascend NPU IR操作。

| Triton Op | 目标 Ascend NPU IR Op | 描述 |
| :--- | :--- | :--- |
| **访存类 Op** | | |
| `triton::StoreOp` | `memref::copy` | 将数据存储到内存。 |
| `triton::LoadOp` | `memref::copy` + `bufferization::ToTensorOp` | 从内存加载数据。 |
| `triton::AtomicRMWOp` | `hivm::StoreOp` 或 `hfusion::AtomicXchgOp` | 执行原子的读-修改-写操作。 |
| `triton::AtomicCASOp` | `linalg::GenericOp` | 执行原子的比较并交换操作。 |
| `triton::GatherOp` | 先转换为`func::CallOp` (调用 `triton_gather`)<br>再转换为`hfusion::GatherOp` | 根据索引收集数据。 |
| **指针运算类 Op** | | |
| `triton::AddPtrOp` | `memref::ReinterpretCast` | 对指针进行偏移运算。 |
| `triton::PtrToIntOp` | `arith::IndexCastOp` | 将指针转换为整数。 |
| `triton::IntToPtrOp` | `hivm::PointerCastOp` | 将整数转换为指针。 |
| `triton::AdvanceOp` | `memref::ReinterpretCastOp` | 推进指针位置。 |
| **程序信息类 Op** | | |
| `triton::GetProgramIdOp` | `functionOp` 的参数 | 获取当前程序的ID。 |
| `triton::GetNumProgramsOp` | `functionOp` 的参数 | 获取总程序数量。 |
| `triton::AssertOp` | 先转换为`func::CallOp` (调用 `triton_assert`)<br>再转换为`hfusion::AssertOp` | 断言操作。 |
| `triton::PrintOp` | 先转换为`func::CallOp` (调用 `triton_print`)<br>再转换为`hfusion::PrintOp` | 打印操作。 |
| **张量操作类 Op** | | |
| `triton::ReshapeOp` | `tensor::ReshapeOp` | 改变张量形状。 |
| `triton::ExpandDimsOp` | `tensor::ExpandShapeOp` | 扩展张量维度。 |
| `triton::BroadcastOp` | `linalg::BroadcastOp` | 广播张量。 |
| `triton::TransOp` | `linalg::TransposeOp` | 转置张量。 |
| `triton::SplitOp` | `tensor::ExtractSliceOp` | 分割张量。 |
| `triton::JoinOp` | `tensor::InsertSliceOp` | 连接张量。 |
| `triton::CatOp` | `tensor::InsertSliceOp` | 拼接张量。 |
| `triton::MakeRangeOp` | `linalg::GenericOp` | 创建一个包含连续整数的张量。 |
| `triton::SplatOp` | `linalg::FillOp` | 用标量值填充张量。 |
| `triton::SortOp` | 先转换为`func::CallOp` (调用 `triton_sort`)<br>再转换为`hfusion::SortOp` | 对张量进行排序。 |
| **数值计算类 Op** | | |
| `triton::MulhiUIOp` | `arith::MulSIExtendedOp` | 无符号整数乘法，返回高位结果。 |
| `triton::PreciseDivFOp` | `arith::DivFOp` | 执行高精度浮点除法。 |
| `triton::PreciseSqrtOp` | `math::SqrtOp` | 执行高精度浮点平方根。 |
| `triton::BitcastOp` | `arith::BitcastOp` | 在不同类型之间进行位重新解释。 |
| `triton::ClampFOp` | `tensor::EmptyOp` + `linalg::FillOp` | 将浮点数限制在指定范围内。 |
| `triton::DotOp` | `linalg::MatmulOp` | 执行通用矩阵乘法。 |
| `triton::DotScaledOp` | `linalg::MatmulOp` | 执行带缩放因子的矩阵乘法。 |
| `triton::ascend::FlipOp` | 先转换为`func::CallOp` (调用 `triton_flip`)<br>再转换为`hfusion::FlipOp` | 执行带缩放因子的矩阵乘法。 |
| **归约类 Op** | | |
| `triton::ArgMinOp` | `linalg::ReduceOp` | 返回张量中最小值的索引。 |
| `triton::ArgMaxOp` | `linalg::ReduceOp` | 返回张量中最大值的索引。 |
| `triton::ReduceOp` | `linalg::ReduceOp` | 通用归约操作。 |
| `triton::ScanOp` | 先转换为`func::CallOp`（调用 `triton_cumsum` 或 `triton_cumprod`）<br>再转换为`hfusion::CumsumOp`及`hfusion::CumprodOp` | 执行扫描操作（如累计和、累计积）。 |

## Triton独有扩展操作

我们为triton提供了多个适用于Ascend的语言特性，若要使能相关能力，你需要import以下的模块

```py
import triton.language.extra.cann.extension as al
```

其后即可使用相关的Ascend Language (al) 独有接口

另外，由于Ascend Language提供的是底层接口，接口不具备兼容性

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
        x = tl.load(x_ptr + i, mask=i < n)
        y = tl.load(y_ptr + i, mask=i < n)
        result = x + y
        tl.store(out_ptr + i, result, mask=i < n)
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
@triton.jit
def triton_matmul_exp(A_ptr, B_ptr, C_ptr, TBuff_ptr,
                      M, N, K: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)

    offs_i = tl.arange(0, 1)[:, None]
    offs_j = tl.arange(0, 1)[None, :]
    offs_k = tl.arange(0, K)
    a_ptrs = A_ptr + (row + offs_i) * K + offs_k[None, :]
    a_vals = tl.load(a_ptrs)
    b_ptrs = B_ptr + offs_k[:, None] * N + (col + offs_j)
    b_vals = tl.load(b_ptrs)

    tbuff_ptrs = TBuff_ptr + (row + offs_i) * N + (col + offs_j)
    acc_11 = tl.dot(a_vals, b_vals)
    tl.store(tbuff_ptrs, acc_11)
    
    extension.sync_block_set("cube", "vector", 5, pipe.PIPE_MTE1, pipe.PIPE_MTE3)
    extension.sync_block_wait("cube", "vector", 5, pipe.PIPE_MTE1, pipe.PIPE_MTE3)
    
    acc_11_reload = tl.load(tbuff_ptrs)
    c_ptrs = C_ptr + (row + offs_i) * N + (col + offs_j)
    tl.store(c_ptrs, tl.exp(acc_11_reload))
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
@triton.jit
def triton_matmul_exp(A_ptr, B_ptr, C_ptr, TBuff_ptr,
                      M, N, K: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)

    offs_i = tl.arange(0, 1)[:, None]
    offs_j = tl.arange(0, 1)[None, :]
    offs_k = tl.arange(0, K)
    a_ptrs = A_ptr + (row + offs_i) * K + offs_k[None, :]
    a_vals = tl.load(a_ptrs)
    b_ptrs = B_ptr + offs_k[:, None] * N + (col + offs_j)
    b_vals = tl.load(b_ptrs)

    tbuff_ptrs = TBuff_ptr + (row + offs_i) * N + (col + offs_j)
    acc_11 = tl.dot(a_vals, b_vals)
    tl.store(tbuff_ptrs, acc_11)
    
    extension.sync_block_set("cube", "vector", 5, pipe.PIPE_MTE1, pipe.PIPE_MTE3)
    extension.sync_block_wait("cube", "vector", 5, pipe.PIPE_MTE1, pipe.PIPE_MTE3)
    
    acc_11_reload = tl.load(tbuff_ptrs)
    c_ptrs = C_ptr + (row + offs_i) * N + (col + offs_j)
    tl.store(c_ptrs, tl.exp(acc_11_reload))
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
| `"all_cube"` | 同步所有Cube核 | 当前AI Core上的所有Cube核 |
| `"all_vector"` | 同步所有Vector核 | 当前AI Core上的所有Vector核 |
| `"all"` | 同步所有核心 | 当前AI Core上的所有计算核心（Cube+Vector） |
| `"all_sub_vector"` | 同步所有子Vector核 | 当前AI Core上的所有子Vector核 |

**使用示例**

```py
@triton.jit
def test_sync_block_all():
    al.sync_block_all("all_cube", 8)
    al.sync_block_all("all_vector", 9)
    al.sync_block_all("all", 10)
    al.sync_block_all("all_sub_vector", 11)
```

### 硬件查询与控制操作

#### sub_vec_id
昇腾支持获取当前AI Core上的Vector核索引。

**使用示例**

```py
@triton.jit
def triton_matmul_exp(
    A_ptr,
    B_ptr,
    C_ptr,
    TBuff_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    sub_M: tl.constexpr,
):
    row_matmul = tl.program_id(0)
    col = tl.program_id(1)

    offs_i = tl.arange(0, tl.constexpr(M))[:, None]
    offs_j = tl.arange(0, N)[None, :]
    offs_k = tl.arange(0, K)

    a_ptrs = A_ptr + (row_matmul + offs_i) * K + offs_k[None, :]
    a_vals = tl.load(a_ptrs)

    b_ptrs = B_ptr + offs_k[:, None] * N + (col + offs_j)
    b_vals = tl.load(b_ptrs)

    tbuff_ptrs = TBuff_ptr + (row_matmul + offs_i) * N + (col + offs_j)

    acc_11 = tl.dot(a_vals, b_vals)
    tl.store(tbuff_ptrs, acc_11)
    sub_vec_id = al.sub_vec_id()
    row_exp = row_matmul + sub_M * sub_vec_id
    offs_exp_i = tl.arange(0, tl.constexpr(sub_M))[:, None]
    tbuff_exp_ptrs = TBuff_ptr + (row_exp + offs_exp_i) * N + (col + offs_j)
    acc_11_reload = tl.load(tbuff_exp_ptrs)
    c_ptrs = C_ptr + (row_exp + offs_exp_i) * N + (col + offs_j)
    tl.store(c_ptrs, tl.exp(acc_11_reload))
```

#### sub_vec_num
昇腾支持获取单个AI Core上的Vector核数量。

**使用示例**

```py
@triton.jit
def triton_matmul_exp(
    A_ptr,
    B_ptr,
    C_ptr,
    TBuff_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    row_matmul = tl.program_id(0)
    col = tl.program_id(1)

    offs_i = tl.arange(0, tl.constexpr(M))[:, None]
    offs_j = tl.arange(0, N)[None, :]
    offs_k = tl.arange(0, K)

    a_ptrs = A_ptr + (row_matmul + offs_i) * K + offs_k[None, :]
    a_vals = tl.load(a_ptrs)

    b_ptrs = B_ptr + offs_k[:, None] * N + (col + offs_j)
    b_vals = tl.load(b_ptrs)

    tbuff_ptrs = TBuff_ptr + (row_matmul + offs_i) * N + (col + offs_j)

    acc_11 = tl.dot(a_vals, b_vals)
    tl.store(tbuff_ptrs, acc_11)

    sub_vec_id = al.sub_vec_id()
    row_exp = row_matmul + (M // al.sub_vec_num()) * sub_vec_id
    offs_exp_i = tl.arange(0, M // al.sub_vec_num())[:, None]
    tbuff_exp_ptrs = TBuff_ptr + (row_exp + offs_exp_i) * N + (col + offs_j)
    acc_11_reload = tl.load(tbuff_exp_ptrs)
    c_ptrs = C_ptr + (row_exp + offs_exp_i) * N + (col + offs_j)
    tl.store(c_ptrs, tl.exp(acc_11_reload))
```

#### parallel

昇腾扩展了 Python 标准的 `range` 功能，增加具有**平行执行语义**的`parallel`迭代器。

**参数说明**

| 参数 | 类型 | 说明 | 范例 |
|------|------|------|------|
| `arg1` | int | 起始值或结束值 | `parallel(10)` |
| `arg2` | int | 结束值 (可选) | `parallel(0, 10)` |
| `step` | int | 步长 (可选) | `parallel(0, 10, 2)` |
| `num_stages` | int | 流水线阶段数 (可选) | `parallel(0, 10, num_stages=3)` |
| `loop_unroll_factor` | int | 循环展开因子 (可选) | `parallel(0, 10, loop_unroll_factor=4)` |

**限制**

目前 910B 最多支持2个Vector核

**使用示例**

```py
@triton.jit
def triton_add(in_ptr0, in_ptr1, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    idx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    x0 = tl.load(in_ptr0 + idx)
    x1 = tl.load(in_ptr1 + idx)
    ret = x0 + x1
    
    for _ in extension.parallel(2, 5, 2):
        ret = ret + x1
    
    for _ in extension.parallel(2, 10, 3):
        ret = ret + x0
    
    odx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    tl.store(out_ptr0 + odx, ret)
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
@triton.jit
def triton_where_lt_case1(in_ptr0, in_ptr1, cond_ptr, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        in0 = tl.load(in_ptr0 + xindex, xmask)
        in1 = tl.load(in_ptr1 + xindex, xmask)
        cond = tl.load(cond_ptr + xindex, xmask)
        mask = tl.where(cond, in1, in0)
        al.compile_hint(mask, "bitwise_mask")
        tl.store(out_ptr0 + (xindex), mask, xmask)
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
@triton.jit
def triton_compile_hint(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        al.multibuffer(tmp0, 2)
        tl.store(out_ptr0 + (xindex), tmp0, xmask)
```

#### scope
昇腾支持作用域管理器，为了一段区域代码添加hint信息，其中一种用法是通过`core_mode`指定cube或vector类型。

**参数说明**

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `core_mode` | str | 核心类型，指定区块内操作使用的计算核心， 只接受`"cube"`或`"vector"`两种模式 |

**核心模式选项**
| 模式 | 描述 |
|------|------|
| `"cube"` | 使用Cube核进行计算 |
| `"vector"` | 使用Vector核进行计算 |

**使用示例**

```py
@triton.jit
def kernel_debug_barrier(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(core_mode="vector"):
        al.debug_barrier(al.SYNC_IN_VF.VV_ALL)
        x = tl.load(x_ptr + i, mask=i < n)
        y = tl.load(y_ptr + i, mask=i < n)
        result = x + y
        tl.store(out_ptr + i, result, mask=i < n)
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
@triton.jit
def triton_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr, SLICE_OFFSET: tl.constexpr, SLICE_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    x_sub = al.extract_slice(x, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    y_sub = al.extract_slice(y, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    output_sub = x_sub + y_sub
    output = tl.load(output_ptr + offsets, mask=mask)
    output = al.insert_slice(output, output_sub, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    tl.store(output_ptr + offsets, output, mask=mask)
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
@triton.jit
def triton_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr, SLICE_OFFSET: tl.constexpr, SLICE_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    x_sub = al.extract_slice(x, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    y_sub = al.extract_slice(y, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    output_sub = x_sub + y_sub
    output = tl.load(output_ptr + offsets, mask=mask)
    output = al.insert_slice(output, output_sub, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    tl.store(output_ptr + offsets, output, mask=mask)
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
@triton.jit
def index_select_manual_kernel(in_ptr, indices_ptr, out_ptr, dim,
                                g_stride: tl.constexpr, indice_length: tl.constexpr,
                                g_block: tl.constexpr, g_block_sub: tl.constexpr, 
                                other_block: tl.constexpr):
    g_begin = tl.program_id(0) * g_block
    for goffs in range(0, g_block, g_block_sub):
        g_idx = tl.arange(0, g_block_sub) + g_begin + goffs
        g_mask = g_idx < indice_length
        indices = tl.load(indices_ptr + g_idx, g_mask, other=0)

        for other_offset in range(0, g_stride, other_block):
            tmp_buf = tl.zeros((g_block_sub, other_block), in_ptr.dtype.element_ty)
            other_idx = tl.arange(0, other_block) + other_offset
            other_mask = other_idx < g_stride
            
            for i in range(0, g_block_sub):
                gather_offset = al.get_element(indices, (i,)) * g_stride
                val = tl.load(in_ptr + gather_offset + other_idx, other_mask)
                tmp_buf = al.insert_slice(
                    tmp_buf, val[None, :],
                    offsets=(i, 0), sizes=(1, other_block), strides=(1, 1)
                )
            
            tl.store(out_ptr + g_idx[:, None] * g_stride + other_idx[None, :], 
                     tmp_buf, g_mask[:, None] & other_mask[None, :])
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
@triton.jit
def sort_kernel_2d(X, Z, N: tl.constexpr, M: tl.constexpr, descending: tl.constexpr):
    offx = tl.arange(0, M)
    offy = tl.arange(0, N) * M
    off2d = offx[None, :] + offy[:, None]
    x = tl.load(X + off2d)
    x = al.sort(x, descending=descending, dim=1)
    tl.store(Z + off2d, x)
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
@triton.jit
def cast_to_bool(output_ptr, x_ptr, dims: Dimensions, overflow_mode: tl.constexpr):
    xidx = tl.arange(0, dims.XB)
    yidx = tl.arange(0, dims.YB)
    zidx = tl.arange(0, dims.ZB)
    idx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]

    X = tl.load(x_ptr + idx)
    overflow_mode = "trunc" if overflow_mode == 0 else "saturate"
    ret = tl.cast(X, dtype=tl.int1, overflow_mode=overflow_mode)

    tl.store(output_ptr + idx, ret)
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
@triton.jit
def index_select_extension_kernel(in_ptr, indices_ptr, out_ptr, dim: tl.constexpr,
                                   other_numel: tl.constexpr,
                                   g_stride: tl.constexpr, indice_length: tl.constexpr,
                                   g_block: tl.constexpr, g_block_sub: tl.constexpr, 
                                   other_block: tl.constexpr):
    g_begin = tl.program_id(0) * g_block
    for goffs in range(0, g_block, g_block_sub):
        g_idx = tl.arange(0, g_block_sub) + g_begin + goffs
        g_mask = g_idx < indice_length
        indices = tl.load(indices_ptr + g_idx, g_mask, other=0)
        
        for other_offset in range(0, g_stride, other_block):
            other_idx = tl.arange(0, other_block) + other_offset
            other_mask = other_idx < g_stride
            
            tmp_buf = al.index_select_simd(
                src=in_ptr,
                dim=dim,
                index=indices,
                src_shape=(other_numel, g_stride),
                src_offset=(-1, 0),
                read_shape=(-1, other_block)
            )
            
            tl.store(out_ptr + g_idx[:, None] * g_stride + other_idx[None, :],
                     tmp_buf, g_mask[:, None] & other_mask[None, :])
```

## Triton独有定制化操作

Triton-Ascend的Custom Op支持用户自行定制操作并使用它。定制操作在运行时转换为对设备侧实现函数的调用，可以调用已有的库函数，也可以调用由用户提供的源码或字节码编译生成的实现函数。

### 基本用法

#### 注册定制操作

定制操作相关功能由triton昇腾扩展包提供。用户自定义的定制操作需要先注册才能被使用，可以通过扩展包提供的`register_custom_op`装饰一个类来定义并注册定制操作：

```python
import triton.language.extra.cann.extension as al

@al.register_custom_op
class my_custom_op:
    name = 'my_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT

```

注册一个最简单的定制操作要求至少提供name、core、pipe、mode这些基本属性，其中：

- name 操作名称，是这个定制操作的唯一标识符，如果省略则默认使用类名；
- core 表示运行在哪种昇腾核上；
- pipe 表示对应的流水线；
- mode 表示使用的编程模式。

#### 使用定制操作

已注册的定制操作可以通过昇腾扩展包提供的`custom()`函数进行调用，调用时需提供定制操作的名字以及参数：

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@triton.jit
def my_kernel(...):
    ...
    res = al.custom('my_custom_op', src, index, dim=0, out=dst)
    ...

```

`custom()`的参数包括操作名称、输入参数和可选的输出参数三部分：

- 操作名称，需要与注册的操作名称一致；
- 输入参数，不同的操作有不同的输入参数；
- 输出参数（可选），输出参数由`out`指定, 表示操作的输出；

如果通过`out`参数指明了输出变量，则该定制操作的返回值与输出变量保持一致；否则该操作的返回值不可用。

### 内置定制操作

内置定制操作的名称均以`"__builtin_"`开头，是triton-ascend内置的已定制好的操作，不需要注册即可直接使用。例如：

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@triton.jit
def my_kernel(...):
    ...
    dst = tl.full(dst_shape, 0, tl.float32)
    x = al.custom('__builtin_indirect_load', src, index, mask, other, out=dst)
    ...

```

具体内置定制操作会随着版本的不同而有所不同，请查阅相关版本的文档。

### 参数有效性检查

在没有约束的情况下，用户可以给`al.custom()`函数填入任何参数，如果填入的参数个数、参数类型等与期望的不符，则会导致运行时出错。

为了避免这种情况，提升定制操作用户的使用体验，我们可以给注册定制类提供构造函数来描述参数列表并执行参数有效性检查，例如：

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@al.register_custom_op
class my_custom_op:
    name = 'my_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT

    def __init__(self, src, index, dim, out=None):
        assert index.dtype.is_int(), "index must be an integer tensor"
        assert isinstance(dim, int), "dim must be an integer"
        assert out, "out is required"
        assert out.shape == index.shape, "out should have same shape as index"
        ...

```

注册类的构造函数参数列表就是调用定制操作要求的参数列表，在调用时需要提供符合要求的参数，例如：

```python
    res = al.custom('my_custom_op', src_ptr, index, dim=1, out=dst)
```

如果提供的参数不对，则编译时会报错。比如这里dim参数要求是一个整数常量，如果给定是一个浮点数，则会有如下报错：

```bash
    ...
    res = al.custom('my_custom_op', src_ptr, index, dim=1.0, out=dst)
          ^
AssertionError('dim must be an integer')
```

### 输出参数和返回值

`al.custom`会将out参数指明的输出参数返回，比如：

```python
x = al.custom('my_custom_op', src, index, out=dst)
```

会将dst返回给x。

out参数可以指明多个输出参数，`al.custom`返回一个元组包含这些输出参数：

```python
x, y = al.custom('my_custom_op', src, index, out=(dst1, dst2)）
```

会将dst1返回给x，将dst2返回给y。

没有out参数时，`al.custom`没有返回值（返回None）。

### 调用函数的符号名

定制操作最终会被转换为对设备侧实现函数的调用。我们可以通过注册定制操作类中的`symbol`属性来配置这个函数的符号名；如果没有设置`symbol`属性，则默认使用定制操作的名称作为函数名。

#### 静态符号名

如果某个定制操作总是固定调用某个设备侧函数，则可以静态设置符号名：

```python
@al.register_custom_op
class my_custom_op:
    name = 'my_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    symbol = '_my_custom_op_symbol_name_'

```

这样，`al.custom('my_custom_op', ...)` 就会固定对应设备侧的 `_my_custom_op_symbol_name_(...)` 函数。

#### 动态符号名

很多情况下，同一个定制操作需要根据输入参数的维度、类型等调用不同的设备侧函数，这时就需要动态设置符号名。与参数有效性检查类似，可以在注册定制操作类的构造函数里面动态设置符号名，例如：

```python
@al.register_custom_op
class my_custom_op:
    name = 'my_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT

    def __init__(self, src, index, dim, out=None):
        ...
        self.symbol = f"my_func_{len(index.shape)}d_{src.dtype.element_ty.cname}_{index.dtype.cname}"
        ...

```

当输入的src为指向float32类型的指针，index为int32类型的3维tensor时，上述定制操作对应的设备侧函数符号名为：`"my_func_3d_float_int32_t"`；不同的输入参数会对应不同的符号名。

注意这里类型名使用的`cname`，表示对应类型在AscendC语言中的名称，如int32对应的cname就是`int32_t`。因为我们通常会采用宏的方式来来声明这些函数，并将相关类型名嵌入到函数名中，所以`cname`会较为常用。

### 源码与编译

如果实现定制操作的函数需要从源代码或字节码编译产生，则需要在注册定制操作类的时候分别配置`source`和`compile`属性：

- source 实现定制操作函数的源代码或字节码文件路径；
- compile 实现定制操作函数的编译命令, 其中可以用`%<`和`%@`分别表示源文件和目标文件（与Makefile类似）；

与符号名类似，这两个属性也可以静态配置或在注册类构造函数中动态配置，例如：

```python
@al.register_custom_op
class my_custom_op:
    name = 'my_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    ...
    source = "workspace/my_custom_op.cce"
    compile = "bisheng -std=c++17 -O2 -o $@ -c $<"

```

### 参数转换规则

#### 参数顺序

定制操作转换为对应的函数调用，其参数顺序与python侧保持一致，输出参数(out，如果有的话)总是放在最后，例如下面python代码：

```python
al.custom('my_custom_op', src, index, dim, out=dst)
```

转换为函数调用相当于：

```C++
my_custom_op(src, index, dim, dst);
```

#### 列表和元组参数

python侧的tuple或list参数会被展平，比如：

```python
al.custom('my_custom_op', src, index, offsets=(1, 2, 3), out=dst)
```

转换为函数调用时，offsets参数会被展平：

```C++
my_custom_op(src, index, 1, 2, 3, dst);
```

### 常量参数类型

定制操作支持整数和浮点数的常量参数类型，但python的整数和浮点数类型并不区分位宽，因此我们只能默认将整数映射为int32_t类型，将浮点数映射为float类型；当实现函数的常量参数为其他位宽的类型时（比如：int64_t），就会因函数签名不匹配导致错误。

例如，有如下定制操作的实现函数签名：

```C++
custom_op_impl_func(memref_t<...> *src, memref_t<...> *idx, int64_t bound);
```

其bound参数要求为int64_t类型的整数。

在python侧调用定制操作是给出了bound常量参数的值：

```python
al.custom('my_custom_op', src, idx, bound=1024)
```

由于python的整数常量不区分位宽，我们只能默认将bound映射为int32_t，导致与实现函数签名不匹配而产生错误。

为了避免此类问题，我们建议实现函数的参数，如果是整数全部使用int32_t，浮点数全部使用float类型；在某些特定场景，我们还提供以下几种方法来指定具体类型：

#### 通过al.int64指定整数位宽

默认会将整数常量映射为int32_t类型，如果实现函数要求一个int64_t的类型，可以通过`al.int64`将整数包装起来，例如：

```python
al.custom('my_custom_op', src, idx, bound=al.int64(1024))
```

#### 通过type hint指明类型

在注册类的构造函数里面，可以给对应参数加上类型注解，比如：

```python
@al.register_custom_op
class my_custom_op:
    name = 'my_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT

    def __init__(self, src, idx, bound: tl.int64):
        ...

```

这样bound参数总是被映射为int64_t类型。

#### 动态指定参数类型

还有一种较为极端的情况，就是参数类型会根据其他参数而变化。比如bound的类型需要与idx的数据类型保持一致。这时可以通过arg_type在构造函数中动态指定类型，比如：

```python
@al.register_custom_op
class my_custom_op:
    name = 'my_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT

    def __init__(self, src, idx, bound):
        ...
        self.arg_type['bound'] = idx.dtype

```

### 封装定制操作

直接使用`al.custom`调用定制操作有时会略显麻烦，特别是在有输出参数时，在调用前需要准备好输出参数，例如：

```python
dst = tl.full(index.shape, 0, tl.float32)
x = al.custom('my_custom_op', src, index, out=dst)
```

我们可以将定制操作封装为一个操作函数，以方便使用，例如：

```python
@al.builtin
def my_custom_op(src, index, _builder=None):
    dst = tl.semantic.full(index.shape, 0, src.dtype.element_ty, _builder)
    return al.custom_semantic(_my_custom_op.name, src, index, out=dst, _builder=_builder)
```

封装的操作函数需要用`al.builtin`装饰，并通过`al.custom_semantic`调用定制操作；同时也可以利用`tl.semantic`提供的功能准备输出参数；注意：封装操作函数时需要给定一个额外的`_builder`参数，并传递给所有senamtic函数。

封装好的操作函数可以类似原生操作一样直接调用：

```
@triton.jit
def my_kernel(...):
    ...
    x = my_custom_op(src, index)
    ...
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
| `VECTOR` | Vector核 |
| `CUBE` | Cube核 |
| `CUBE_OR_VECTOR` | Cube或Vector核（二选一） |
| `CUBE_AND_VECTOR` | Cube和Vector核（混合使用） |

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

