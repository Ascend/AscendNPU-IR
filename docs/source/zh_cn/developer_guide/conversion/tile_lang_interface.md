# TileLang 接入

Tile Language Ascend（**tilelang-ascend**）是 tile-lang 领域特定语言针对华为昇腾 NPU（神经网络处理器）架构的专用变体，经过专门优化。它基于 tile-lang 的 Python 式语法和 [TVM](https://tvm.apache.org/) 编译器基础架构，使开发者能够高效地为昇腾处理器（包括 GEMM、向量运算和注意力机制等操作）创建高性能 AI 计算内核。tilelang-ascend 让开发者能够专注于生产效率，同时不牺牲在 NPU 上实现前沿性能所需的底层优化。

在 TileLang 生态系统中，我们专为昇腾开发了 NPU 中间表示（AscendNPU IR）基础设施，实现了与基于 MLIR 的开源 AI 编译器生态系统的无缝集成。这不仅提升了编译器栈的开放性和可扩展性，还为开发者提供了更灵活高效的自定义算子开发途径。编译器后端支持两种技术路线：[AscendNPU IR](https://github.com/tile-ai/tilelang-ascend/tree/npuir) 和 [Ascend C & PTO](https://github.com/tile-ai/tilelang-ascend/tree/ascendc_pto)。

![image](../../../images/developer_guide/npuir_architecture.png)

## 安装

### 环境准备

安装 Ascend Toolkit。

[下载安装包](https://www.hiascend.com/developer/download/community/result?cann=8.3.RC1.alpha002)，安装 `Ascend-cann-toolkit`。完整安装说明请参考[相关文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/softwareinst/instg/instg_0008.html?Mode=PmIns&OS=Debian&Software=cannToolKit)。

```bash
chmod +x Ascend-cann-toolkit_{ascend-cann-toolkit version}_linux-aarch64.run
./Ascend-cann-toolkit_{ascend-cann-toolkit version}_linux-aarch64.run --install
```

配置环境变量：

```bash
source /path/to/install/Ascend/ascend-toolkit/set_env.sh
```

准备 Python 环境（Python 版本在 3.7.*x* 到 3.11.4 之间，含边界），并确保 `pip3` 可用。

   Ascend Toolkit 安装依赖：

   ```bash
   pip3 install attrs cython 'numpy>=1.19.2,<=1.24.0' decorator sympy cffi pyyaml pathlib2 psutil protobuf==3.20.0 scipy requests absl-py
   ```

设置环境变量：

```bash
export ACL_OP_INIT_MODE=1
```

#### 构建

拉取代码：

```bash
git clone https://github.com/tile-ai/tilelang-ascend.git --recursive -b npuir
```

运行安装脚本：

> 注意: 如果环境中有gtest头文件但没有gtest的库文件，编译过程可能会引发异常。
> 可以通过临时移除环境中的gtest头文件或者添加库文件，或者tvm联合gtest一同编译来进行解决。

```bash
cd tilelang-ascend
# 在 3rdparty 中构建 AscendNPU-IR
bash install_npuir.sh
# 或者，使用本地 AscendNPU-IR 的替代构建方式
bash install_npuir.sh --bishengir-path=/path/to/AscendNPU-IR/build/install
# 假定当前目录是tilelang-ascend，选项可以是：--bishengir-path=./3rdparty/AscendNPU-IR/build/install
```

然后需要做下面任何一步来使能tilelang的环境设置

```bash
source ~/.bashrc

or

export PYTHONPATH=/path/to/tilelang-ascend/:$PYTHONPATH

or

open a new terminal
```

安装 torch_npu：

```bash
pip install pybind11 torch_npu
```

## 快速开始

以下代码使用 TileLang（NPU 编程的领域特定语言）实现了向量加法内核。该内核定义了一个并行内核，通过将数据加载到片上统一缓冲区（UB）、使用低级 NPU 指令（`npuir_add`）执行逐元素加法，并将结果写回全局内存，在 NPU 上对两个长度为 4096 的 float32 向量进行加法运算。测试函数将内核输出与 PyTorch 原生向量加法进行比较以验证正确性。示例在 NPU 设备上运行，演示了 TileLang 的基本工作流程：内核定义、编译为 AscendNPU IR，以及使用 PyTorch 张量执行。

### TileLang 内核（向量加法）

```python
# test_tilelang.py

import os

import tilelang
import tilelang.language as T  # Import TileLang DSL for kernel definition

import torch
import torch_npu  # Import NPU (Neural Processing Unit) backend support for PyTorch

# Clear any previously cached compiled kernels to ensure a clean run
tilelang.cache.clear_cache()

# Define data type and sequence length for the vector addition
dtype = "float32"
seq_len = 4096  # Length of the vectors to be added

def vec_add(N, block_N, dtype="float32"):
    """
    Define a vector addition kernel using TileLang.
    
    Parameters:
    - N: Total length of the vectors.
    - block_N: Number of elements processed per kernel thread/block.
    - dtype: Data type of the tensors (default: "float32").
    
    Returns:
    - A TileLang prim_func representing the vector addition kernel.
    """
    n_num = N // block_N  # Number of blocks (each block processes `block_N` elements)

    @T.prim_func
    def main(
        A: T.Tensor((N), dtype),  # Input tensor A
        B: T.Tensor((N), dtype),  # Input tensor B
        C: T.Tensor((N), dtype),  # Output tensor C = A + B
        shape: T.int32,           # Actual size (used for handling tail cases if N is not divisible by block_N)
    ):
        # Launch kernel with `n_num` parallel threads on the NPU
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            # Allocate on-chip Unified Buffer (UB) for local computation
            A_VEC = T.alloc_ub((block_N), dtype)
            B_VEC = T.alloc_ub((block_N), dtype)
            C_VEC = T.alloc_ub((block_N), dtype)

            # Calculate the starting index for this thread
            start_idx = cid * block_N
            # Compute remaining elements from this start index to the end of the tensor
            remaining = shape - start_idx
            # Determine how many elements this thread should actually process (handles tail)
            tail_size = T.min(block_N, remaining)

            # Copy data from global memory (A, B) into on-chip buffers (A_VEC, B_VEC)
            T.copy(A[start_idx], A_VEC[:tail_size])
            T.copy(B[start_idx], B_VEC[:tail_size])

            # Perform vector addition on the NPU using low-level NPU IR instruction
            T.npuir_add(A_VEC, B_VEC, C_VEC)

            # Write the result back from on-chip buffer (C_VEC) to global memory (C)
            T.copy(C_VEC[:tail_size], C[start_idx])

    return main

def test_vec_add():
    """
    Test function to validate the vector addition kernel.
    Compares the result of the custom TileLang kernel against PyTorch's native addition.
    """
    # Set the target NPU device (device ID 6 in this case)
    torch.npu.set_device(0)

    # Instantiate the vector addition kernel for the full sequence length (single block)
    func = vec_add(seq_len, seq_len)

    # Compile the TileLang function to NPU IR for execution on the NPU
    compiled_kernel = tilelang.compile(func, target="npuir")

    # Create random input tensors on the NPU
    v1 = torch.randn(size=[seq_len], dtype=eval("torch." + dtype)).npu()
    v2 = torch.randn(size=[seq_len], dtype=eval("torch." + dtype)).npu()
    v3 = torch.zeros(size=[seq_len], dtype=eval("torch." + dtype)).npu()  # Output buffer

    # Compute reference result using PyTorch's native addition (on NPU)
    y_ref = v1.cpu() + v2.cpu()

    # Launch the compiled TileLang kernel
    compiled_kernel(v1, v2, v3, seq_len)

    # Print both results for visual comparison (should be nearly identical)
    print("Reference result (PyTorch):")
    print(y_ref)
    print("TileLang kernel result:")
    print(v3.cpu())

if __name__ == "__main__":
    test_vec_add()
```

运行 `python3 test_tilelang.py`, 我们可以看到执行成功的结果

```bash
Reference result (PyTorch):
tensor([-0.9222,  1.9638,  0.6157,  ...,  0.4924,  0.3776, -0.2921])
TileLang kernel result:
tensor([-0.9222,  1.9638,  0.6157,  ...,  0.4924,  0.3776, -0.2921])
```

### AscendNPU-IR（向量加法）

当开启打印选项`export TILELANG_DUMP_IR=1`后，会将TVM IR与AscendNPU IR都打印出来，其中AscendNPU IR部分如下：

```mlir
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @main(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf32, #hivm.address_space<gm>>, %arg4: memref<?xf32, #hivm.address_space<gm>>, %arg5: memref<?xf32, #hivm.address_space<gm>>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [4096], strides: [%0] : memref<?xf32, #hivm.address_space<gm>> to memref<4096xf32, strided<[1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [4096], strides: [%0] : memref<?xf32, #hivm.address_space<gm>> to memref<4096xf32, strided<[1]>, #hivm.address_space<gm>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [4096], strides: [%0] : memref<?xf32, #hivm.address_space<gm>> to memref<4096xf32, strided<[1]>, #hivm.address_space<gm>>
    %1 = hivm.hir.get_block_idx -> i64
    %2 = arith.trunci %1 : i64 to i32
    %alloc = memref.alloc() : memref<4096xf32, strided<[1]>, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() : memref<4096xf32, strided<[1]>, #hivm.address_space<ub>>
    %alloc_3 = memref.alloc() : memref<4096xf32, strided<[1]>, #hivm.address_space<ub>>
    %c4096_i32 = arith.constant 4096 : i32
    %3 = arith.minsi %c4096_i32, %arg6 : i32
    %4 = arith.index_cast %3 : i32 to index
    %subview = memref.subview %reinterpret_cast[0] [%4] [1] : memref<4096xf32, strided<[1]>, #hivm.address_space<gm>> to memref<?xf32, strided<[1]>, #hivm.address_space<gm>>
    %subview_4 = memref.subview %alloc[0] [%4] [1] : memref<4096xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    memref.copy %subview, %subview_4 : memref<?xf32, strided<[1]>, #hivm.address_space<gm>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_5 = memref.subview %reinterpret_cast_1[0] [%4] [1] : memref<4096xf32, strided<[1]>, #hivm.address_space<gm>> to memref<?xf32, strided<[1]>, #hivm.address_space<gm>>
    %subview_6 = memref.subview %alloc_2[0] [%4] [1] : memref<4096xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    memref.copy %subview_5, %subview_6 : memref<?xf32, strided<[1]>, #hivm.address_space<gm>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%alloc, %alloc_2 : memref<4096xf32, strided<[1]>, #hivm.address_space<ub>>, memref<4096xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%alloc_3 : memref<4096xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_7 = memref.subview %alloc_3[0] [%4] [1] : memref<4096xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_8 = memref.subview %reinterpret_cast_0[0] [%4] [1] : memref<4096xf32, strided<[1]>, #hivm.address_space<gm>> to memref<?xf32, strided<[1]>, #hivm.address_space<gm>>
    memref.copy %subview_7, %subview_8 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<gm>>
    return
  }
}
```
