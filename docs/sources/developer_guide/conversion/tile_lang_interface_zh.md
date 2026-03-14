# TileLang 接入

Tile Language Ascend（**tilelang-ascend**）是 tile-lang 领域特定语言针对华为昇腾 NPU（神经网络处理器）架构的专用变体，经过专门优化。它基于 tile-lang 的 Python 式语法和 [TVM](https://tvm.apache.org/) 编译器基础架构，使开发者能够高效地为昇腾处理器（包括 GEMM、向量运算和注意力机制等操作）创建高性能 AI 计算内核。tilelang-ascend 让开发者能够专注于生产效率，同时不牺牲在 NPU 上实现前沿性能所需的底层优化。

在 TileLang 生态系统中，我们专为昇腾开发了 NPU 中间表示（AscendNPU IR）基础设施，实现了与基于 MLIR 的开源 AI 编译器生态系统的无缝集成。这不仅提升了编译器栈的开放性和可扩展性，还为开发者提供了更灵活高效的自定义算子开发途径。编译器后端支持两种技术路线：[AscendNPU IR](https://github.com/tile-ai/tilelang-ascend/tree/npuir) 和 [Ascend C & PTO](https://github.com/tile-ai/tilelang-ascend/tree/ascendc_pto)。

![](figs/npuir_architecture.png)

## 安装

### 环境准备

安装 Ascend Toolkit。

[下载安装包](https://www.hiascend.com/developer/download/community/result?cann=8.3.RC1.alpha002)，安装 `Ascend-cann-toolkit`。完整安装说明请参考[相关文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/softwareinst/instg/instg_0008.html?Mode=PmIns&OS=Debian&Software=cannToolKit)。

```shell
chmod +x Ascend-cann-toolkit_{ascend-cann-toolkit version}_linux-aarch64.run
./Ascend-cann-toolkit_{ascend-cann-toolkit version}_linux-aarch64.run --install
```

配置环境变量：

```
source /path/to/install/Ascend/ascend-toolkit/set_env.sh
```

准备 Python 环境（Python 版本在 3.7.*x* 到 3.11.4 之间，含边界），并确保 `pip3` 可用。

   Ascend Toolkit 安装依赖：

   ```shell
   pip3 install attrs cython 'numpy>=1.19.2,<=1.24.0' decorator sympy cffi pyyaml pathlib2 psutil protobuf==3.20.0 scipy requests absl-py
   ```

设置环境变量：

```shell
export ACL_OP_INIT_MODE=1
```

#### 构建

拉取代码：

```shell
git clone https://github.com/tile-ai/tilelang-ascend.git --recursive -b npuir
```

运行安装脚本：

```shell
cd tilelang-ascend
# 在 3rdparty 中构建 AscendNPU-IR
bash install_npuir.sh
# 使用本地 AscendNPU-IR 的替代构建方式
bash install_npuir.sh --bishengir-path=/path/to/bishengir-compile
```

安装 torch_npu：

```shell
pip install pybind11 torch_npu
```

## 快速开始

以下代码使用 TileLang（NPU 编程的领域特定语言）实现了向量加法内核。该内核定义了一个并行内核，通过将数据加载到片上统一缓冲区（UB）、使用低级 NPU 指令（`npuir_add`）执行逐元素加法，并将结果写回全局内存，在 NPU 上对两个长度为 4096 的 float32 向量进行加法运算。测试函数将内核输出与 PyTorch 原生向量加法进行比较以验证正确性。示例在 NPU 设备上运行，演示了 TileLang 的基本工作流程：内核定义、编译为 AscendNPU IR，以及使用 PyTorch 张量执行。

### TileLang 内核（向量加法）

```python
# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import os

import tilelang
import tilelang.language as T  # 导入 TileLang DSL 用于内核定义

import torch
import torch_npu  # 导入 NPU（神经网络处理器）PyTorch 后端支持

# 清除之前缓存的编译内核，确保干净运行
tilelang.cache.clear_cache()

# 定义向量加法的数据类型和序列长度
dtype = "float32"
seq_len = 4096  # 待相加向量的长度

def vec_add(N, block_N, dtype="float32"):
    """
    使用 TileLang 定义向量加法内核。

    参数：
    - N: 向量总长度。
    - block_N: 每个内核线程/块处理的元素数量。
    - dtype: 张量的数据类型（默认："float32"）。

    返回：
    - 表示向量加法内核的 TileLang prim_func。
    """
    n_num = N // block_N  # 块数量（每块处理 `block_N` 个元素）

    @T.prim_func
    def main(
        A: T.Tensor((N), dtype),  # 输入张量 A
        B: T.Tensor((N), dtype),  # 输入张量 B
        C: T.Tensor((N), dtype),  # 输出张量 C = A + B
        shape: T.int32,           # 实际大小（用于处理 N 不能被 block_N 整除的尾部情况）
    ):
        # 在 NPU 上以 `n_num` 个并行线程启动内核
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            # 分配片上统一缓冲区（UB）用于本地计算
            A_VEC = T.alloc_ub((block_N), dtype)
            B_VEC = T.alloc_ub((block_N), dtype)
            C_VEC = T.alloc_ub((block_N), dtype)

            # 计算当前线程的起始索引
            start_idx = cid * block_N
            # 计算从该起始索引到张量末尾的剩余元素数
            remaining = shape - start_idx
            # 确定当前线程实际应处理的元素数（处理尾部情况）
            tail_size = T.min(block_N, remaining)

            # 将数据从全局内存（A、B）复制到片上缓冲区（A_VEC、B_VEC）
            T.copy(A[start_idx], A_VEC, [tail_size])
            T.copy(B[start_idx], B_VEC, [tail_size])

            # 使用低级 NPU IR 指令在 NPU 上执行向量加法
            T.npuir_add(A_VEC, B_VEC, C_VEC)

            # 将结果从片上缓冲区（C_VEC）写回全局内存（C）
            T.copy(C_VEC, C[start_idx], [tail_size])

    return main

def test_vec_add():
    """
    验证向量加法内核的测试函数。
    将自定义 TileLang 内核的结果与 PyTorch 原生加法进行比较。
    """
    # 设置目标 NPU 设备
    torch.npu.set_device(0)

    # 为完整序列长度实例化向量加法内核（单块）
    func = vec_add(seq_len, seq_len)

    # 将 TileLang 函数编译为 NPU IR 以在 NPU 上执行
    compiled_kernel = tilelang.compile(func, target="npuir")

    # 在 NPU 上创建随机输入张量
    v1 = torch.randn(size=[seq_len], dtype=eval("torch." + dtype)).npu()
    v2 = torch.randn(size=[seq_len], dtype=eval("torch." + dtype)).npu()
    v3 = torch.zeros(size=[seq_len], dtype=eval("torch." + dtype)).npu()  # 输出缓冲区

    # 使用 PyTorch 原生加法计算参考结果（在 NPU 上）
    y_ref = v1 + v2

    # 启动已编译的 TileLang 内核
    compiled_kernel(v1, v2, v3, seq_len)

    # 打印两个结果进行对比（应基本相同）
    print("参考结果（PyTorch）：")
    print(y_ref)
    print("TileLang 内核结果：")
    print(v3)

if __name__ == "__main__":
    test_vec_add()
  ```

### AscendNPU-IR（向量加法）

以上内核生成如下 AscendNPU-IR：

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
    %3 = arith.muli %2, %c4096_i32 : i32
    %4 = arith.subi %arg6, %3 : i32
    %5 = arith.minsi %c4096_i32, %4 : i32
    %6 = arith.index_cast %3 : i32 to index
    %7 = arith.index_cast %5 : i32 to index
    %subview = memref.subview %reinterpret_cast[%6] [%7] [1] : memref<4096xf32, strided<[1]>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_4 = memref.subview %alloc[0] [%7] [1] : memref<4096xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    memref.copy %subview, %subview_4 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_5 = memref.subview %reinterpret_cast_1[%6] [%7] [1] : memref<4096xf32, strided<[1]>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_6 = memref.subview %alloc_2[0] [%7] [1] : memref<4096xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    memref.copy %subview_5, %subview_6 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%alloc, %alloc_2 : memref<4096xf32, strided<[1]>, #hivm.address_space<ub>>, memref<4096xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%alloc_3 : memref<4096xf32, strided<[1]>, #hivm.address_space<ub>>)
    %subview_7 = memref.subview %alloc_3[0] [%7] [1] : memref<4096xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_8 = memref.subview %reinterpret_cast_0[%6] [%7] [1] : memref<4096xf32, strided<[1]>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview_7, %subview_8 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    return
  }
}
  ```
