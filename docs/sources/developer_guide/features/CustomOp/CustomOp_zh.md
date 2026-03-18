# 自定义算子（CustomOp）

## 概述

AscendNPU-IR 已为上游模型支持丰富的算子集合。然而，在某些场景下，用户需要定义自己的算子来执行自定义计算：

- 现有算子的组合无法满足所需的计算需求。
- 厂商希望自定义算子保持私有。
- 多个算子的组合无法达到最优性能。

自定义算子允许用户自由使用 AscendNPU-IR 提供的接口，提供能与其他算子一起编译的自有算子。

---

### 硬件背景

不涉及

---

### 算法原理

不涉及

---

### 接口说明

自定义算子的通用接口如下：

- **name**：唯一算子名称。

         注意：某些名称为内置算子保留，通常以 "__builtin" 开头。
                编译器会将这些内置算子链接到随 bishengir-compile 一起提供的
                自包含模板库。

                对于普通名称/场景，用户需指定实现位置/编译命令，
                以及所有必要的信息。

         可用的内置名称：
            "__builtin_gather_load"

- **inputs**：输入参数。
- **outputs**：输出结果，指定为 "init" 操作数，作为操作结果的初始值，
              或操作结果将写入的初始位置。

为了快速适应未来的功能扩展，自定义算子依赖属性来获取必要信息：

- **CoreType**：在哪种核类型上执行，参见 TCoreTypeAttr。
- **Pipe**：在哪个 pipe 上执行，参见 PipeAttr。
- **VFMode**：在向量单元上的运行模式，参见 VFModeAttr。
             当核类型为 cube 时，此属性被忽略。

             注意：对于内置算子，用户可以指定或不指定这些信息，
                   编译器会帮助检查正确性并进行规范化。
- **Symbol**: 实现函数名称

TODO：
- **Implementation linkage**：用户提供的实现和链接流程。
- **Multi-Pipes (Macro CustomOp)**：自定义算子希望使用多个 pipe（在 HIVM 语境中为 MacroOp）。

---

#### 降级流程

```
┌─────────────────────────────────────────────────────────────────┐
│                          CustomOp                               │
│    hivm.hir.custom "name" { attrs... } ins(..) outs(...)        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  HIVMToStandard                                                 │
│  ───────────────────────────────────────────────────────────────│
│  • 内置算子                                                     │
│    -> 调用内置库                                                │
│  • 用户提供的实现 ->                                            │
|    -> 调用用户提供的函数名                                      |
|      -> bishengir-compile 使用用户提供的链接命令进行链接        |
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
            毕昇编译器将其编译为 Object Files
```

---

### 约束能力

### ✅ 能力

| 特性 | 说明 |
| ------------------------------- | ------------------------------------------------------------ |
| **CoreType** | 自定义算子执行核。 |
| **Pipe** | 自定义算子执行 pipe。 |
| **VFMode** | 自定义算子在向量核上的运行模式：SIMT/SIMD/MIX。 |
| **Symbol** | 使用者提供之函数名称 |
| **内置算子** | 一组内置算子（名称预留）。 |

### ⚠️ 限制

| 限制 | 说明 | 状态 |
| ---------------------------- | --------------------------------------------------------- | ------------------------------------------------------- |
| **用户实现** | 自定义算子降级到用户提供的实现： | 进行中。 |
| | - HIVM IR 链接到用户提供的源码/Objects Files | |
| | - 向 bisheng-compile 注册特定链接命令 | |
| **Pass 交互** | 适配自定义算子的变换 Pass： | 不适用，进行中。 |
| | - Flatten 优化 | |
| | - 对齐调整 | |
| | - 内存规划 | |
| | - 布局变换 | |
| | - ... 更多待补充 | |

---

## MLIR 示例

### 内置算子

```mlir
%0 = hivm.hir.custom
       "__builtin_gather_load"
       ins(%arg0, %arg1, %c4_i64, %c0_i32, %c2_i64, %c1_i64, %c2_i32, %c2_i32, %c0_i32, %c0_i32
           : memref<?xf32>, tensor<3x3xi64>, i64, i32, i64, i64, i32, i32, i32, i32)
       outs(%empty : tensor<3x3xf32>) -> tensor<3x3xf32>

```

### 自定义算子

```mlir
%0 = hivm.hir.custom
      { hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.vf_mode = #hivm.vf_mode<SIMD>,
        symbol = "my_custom" }
      "my_custom_op"
      ins(%arg0, %arg1, %c4_i64, %c0_i32, %c2_i64, %c1_i64, %c2_i32, %c2_i32, %c0_i32, %c0_i32
          : memref<?xf32>, tensor<3x3xi64>, i64, i32, i64, i64, i32, i32, i32, i32)
      outs(%empty : tensor<3x3xf32>) -> tensor<3x3xf32>
```

---

## TRITON 自定义算子降级示例

```python
# 更多关于 Triton 自定义算子设计的详情，请参考
# https://gitcode.com/Ascend/triton-ascend/pull/988

import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def builtin_index_select_kernel(src_ptr, index_ptr, out_ptr):
    # 为输出张量定义 2x2 tile 索引
    r = tl.arange(0, 2)[:, None]  # 行索引：形状 [2, 1]
    c = tl.arange(0, 2)[None, :]  # 列索引：形状 [1, 2]

    # 将索引张量（形状 [2]）从 GM 加载到 UB
    idx = tl.load(index_ptr + tl.arange(0, 2))
    # 在 UB 中初始化空的 2x2 输出 tile（默认值：0）
    dst = tl.full((2, 2), 0, dtype=tl.float32)

    # 调用 __builtin_index_select 自定义算子进行 gather
    out_tile = al.custom(
        "__builtin_index_select",
        src_ptr,          # 指向 GM 中源张量的指针
        idx,              # 用于 gather 的索引张量（在 UB 中）
        dim=0,            # gather 的维度
        bound=4,          # 有效索引值的上界（越界检查）
        end_offset=(2, 2),# 索引张量每个维度的结束偏移
        start_offset=(0, 0), # 源张量每个维度的起始偏移
        src_stride=(4, 1),# GM 中源张量每个维度的步长
        out=dst           # 用于存储 gather 结果的输出张量（在 UB 中）
    )

    # 将 gather 后的 tile 从 UB 存回 GM 中的输出张量
    tl.store(out_ptr + r * 2 + c, out_tile)


if __name__ == "__main__":
    src = torch.tensor(
        [[10., 11., 12., 13.],
         [20., 21., 22., 23.],
         [30., 31., 32., 33.],
         [40., 41., 42., 43.]],
        device="npu",
        dtype=torch.float32,
    )
    index = torch.tensor([2, 0], device="npu", dtype=torch.int32)
    out = torch.empty((2, 2), device="npu", dtype=torch.float32)
    ref = torch.index_select(src, 0, index.to(torch.int64))[:, :2]
    builtin_index_select_kernel[(1,)](src, index, out)
    torch.testing.assert_close(out, ref) # ref: [[30., 31.], [10., 11.]]
```

降级为 MLIR：

```mlir
module {
  func.func @builtin_index_select_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32}, %arg3: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "mix_simd_simt"} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2x2xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x2xf32>) -> tensor<2x2xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [2], strides: [1] : memref<?xi32> to memref<2xi32, strided<[1]>>
    %alloc = memref.alloc() : memref<2xi32>
    memref.copy %reinterpret_cast, %alloc : memref<2xi32, strided<[1]>> to memref<2xi32>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<2xi32>
    %3 = hivm.hir.custom {extra_attr = "src_stride_len=2", hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMT>} "__builtin_index_select" ins(%arg2, %2, %c0_i32, %c4_i32, %c2_i32, %c2_i32, %c0_i32, %c0_i32, %c4_i32, %c1_i32 : memref<?xf32>, tensor<2xi32>, i32, i32, i32, i32, i32, i32, i32, i32) outs(%1 : tensor<2x2xf32>) -> tensor<2x2xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [2, 2], strides: [2, 1] : memref<?xf32> to memref<2x2xf32, strided<[2, 1]>>
    bufferization.materialize_in_destination %3 in writable %reinterpret_cast_0 : (tensor<2x2xf32>, memref<2x2xf32, strided<[2, 1]>>) -> ()
    return
  }
}
```
