# 自定义算子（CustomOp）

## 概述

AscendNPU-IR 已为上游模型支持丰富的算子集合。然而，在某些场景下，用户需要定义自己的算子来执行自定义计算：

- 现有算子的组合无法满足所需的计算需求。
- 厂商希望自定义算子保持私有。
- 多个算子的组合无法达到最优性能。

自定义算子允许用户自由使用 AscendNPU-IR 提供的接口，提供能与其他算子一起编译的自有算子。

### 硬件背景

不涉及

### 算法原理

不涉及

### 接口说明

自定义算子的通用接口如下：

- **name**：唯一算子名称。

         注意：某些名称为内置算子保留，通常以 "__builtin" 开头。
                编译器会将这些内置算子链接到随 bishengir-compile 一起提供的
                自包含模板库。

                对于普通名称/场景，用户需指定实现位置/编译命令，
                以及所有必要的信息。

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

#### 降级流程

```text
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

### 约束能力

### ✅ 能力

| 特性 | 说明 |
| ------------------------------- | ------------------------------------------------------------ |
| **CoreType** | 自定义算子执行核。 |
| **Pipe** | 自定义算子执行 pipe。 |
| **VFMode** | 自定义算子在向量核上的运行模式：SIMT/SIMD/MIX。 |
| **Symbol** | 使用者提供的函数名称 |
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

#### TRITON 自定义算子降级示例

Python脚本：`test_custom_op.py`

```python
# 更多关于 Triton 自定义算子设计的详情，请参考
# https://gitcode.com/Ascend/triton-ascend/pull/988

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

import torch
import torch_npu

import pytest

def torch_add(a, b):
    return a + b

@al.register_custom_op
class add:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD
    
    def __init__(self, a, b, out=None):
      assert out, "out is required"
      self.symbol = "custom_add_" + str(a.dtype)
      self.bitcode = 'add.bc'

@triton.jit
def triton_custom_add(
    output_ptr,
    a_ptr,
    b_ptr,
    L: tl.constexpr
):
    idx = tl.arange(0, L)

    a = tl.load(a_ptr + idx)
    b = tl.load(b_ptr + idx)

    buf = tl.full([L], 0, a.dtype)
    res = al.custom("add", a, b, out=buf)

    tl.store(output_ptr + idx, res)


testlist = [
  (32)
]

typelist = [torch.int32]

@pytest.mark.parametrize("DT", typelist)
@pytest.mark.parametrize("L", testlist)
def test_custom(DT, L):
    a = torch.ones(L, dtype=DT).npu()
    b = torch.ones(L, dtype=DT).npu()
     
    ref = torch_add(a, b)

    out = torch.zeros(L, dtype=DT).npu()
    triton_custom_add[1, 1, 1](out, a, b, L)
  
    torch.testing.assert_close(out, ref)
```

CPP API 定义：`add.cpp`

```C++
#define __aiv__ [aicore]
#define INTRINSIC_NO_ARGS(NAME) NAME()
#define INTRINSIC(NAME, ...) NAME(__VA_ARGS__)

template <typename T, size_t Dim>
struct memref_t {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[Dim];
  int64_t strides[Dim];
};

template <size_t OPERANUM, typename SRC_T, typename DST_T = SRC_T>
struct intrin_args {
  __ubuf__ DST_T *dst;
  __ubuf__ SRC_T *src[OPERANUM];
  SRC_T scalar;
  uint64_t repeat;
  uint16_t dst_block_stride;
  uint16_t src_block_stride[OPERANUM];
  uint16_t dst_repeat_stride;
  uint16_t src_repeat_stride[OPERANUM];
};

template <typename SRC_TYPE, typename DST_TYPE = SRC_TYPE>
__aiv__ __attribute__((always_inline)) void
vector_eltwise_vadd_intrin(intrin_args<2, SRC_TYPE, DST_TYPE> args) {
#define ELTWISE_VV_ARGS                                                        \
  args.dst, args.src[0], args.src[1], args.repeat, args.dst_block_stride,      \
      args.src_block_stride[0], args.src_block_stride[1],                      \
      args.dst_repeat_stride, args.src_repeat_stride[0],                       \
      args.src_repeat_stride[1]

  INTRINSIC(vadd, ELTWISE_VV_ARGS);
}

extern "C" {
__aiv__ __attribute__((always_inline)) void _mlir_ciface_custom_add_int32(
    memref_t<__ubuf__ int32_t, 1> *src0, memref_t<__ubuf__ int32_t, 1> *src1,
    memref_t<__ubuf__ int32_t, 1> *dst) {
  uint16_t src0_block_stride = 1;
  uint16_t src1_block_stride = 1;
  uint16_t src0_repeat_stride = 8;
  uint16_t src1_repeat_stride = 8;
  auto new_src0_ptr = src0->aligned + src0->offset;
  auto new_src1_ptr = src1->aligned + src1->offset;
  auto dst_ptr = dst->aligned + dst->offset;
  INTRINSIC_NO_ARGS(set_mask_count);
  const int64_t n = dst->sizes[0];
  INTRINSIC(set_vector_mask, 0, n);
  vector_eltwise_vadd_intrin<int32_t>(
      intrin_args<2, int32_t>{dst_ptr,
                        {new_src0_ptr, new_src1_ptr},
                        0,
                        1,
                        1,
                        {src0_block_stride, src1_block_stride},
                        8,
                        {src0_repeat_stride, src1_repeat_stride}});
  INTRINSIC_NO_ARGS(set_mask_norm);
}
}
```

编译`.bc`文件指令：

```bash
ccec -x cce --cce-aicore-arch=dav-c220-vec --cce-aicore-only -c -emit-llvm ./add.cpp -o ./add.bc
```

Python脚本执行命令：

```bash
python -m pytest -sv test_custom_op.py
```

降级为 MLIR：

```mlir
module attributes {hacc.target = #hacc.target<"Ascend910B3">} {
  func.func @triton_custom_add(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<32xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<32xi32>) -> tensor<32xi32>
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [32], strides: [1] : memref<?xi32> to memref<32xi32, strided<[1]>>
    %alloc = memref.alloc() : memref<32xi32>
    memref.copy %reinterpret_cast, %alloc : memref<32xi32, strided<[1]>> to memref<32xi32>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<32xi32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [32], strides: [1] : memref<?xi32> to memref<32xi32, strided<[1]>>
    %alloc_1 = memref.alloc() : memref<32xi32>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<32xi32, strided<[1]>> to memref<32xi32>
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<32xi32>
    %4 = hivm.hir.custom {bitcode = "/home/test/add.bc", hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "custom_add_int32"} "add" ins(%2, %3 : tensor<32xi32>, tensor<32xi32>) outs(%1 : tensor<32xi32>) -> tensor<32xi32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [32], strides: [1] : memref<?xi32> to memref<32xi32, strided<[1]>>
    bufferization.materialize_in_destination %4 in writable %reinterpret_cast_2 : (tensor<32xi32>, memref<32xi32, strided<[1]>>) -> ()
    return
  }
}
```
