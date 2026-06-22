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
- **Pipe**：在哪个 pipe 上执行，参见 PipeAttr（用于 `hivm.hir.custom`）。
- **InPipe**：Macro 自定义算子的输入 pipe，参见 PipeAttr（`hivm.pipe_in`）。
- **OutPipe**：Macro 自定义算子的输出 pipe，参见 PipeAttr（`hivm.pipe_out`）。
- **VFMode**：在向量单元上的运行模式，参见 VFModeAttr。
             当核类型为 cube 时，此属性被忽略。

             注意：对于内置算子，用户可以指定或不指定这些信息，
                   编译器会帮助检查正确性并进行规范化。
- **Symbol**: 实现函数名称。
- **sync_event_slots**：Macro 自定义算子的同步槽位元信息。GraphSyncSolver 会填充
  对应的 `sync_related_args`，并在需要时在 macro 前后注入 set/wait flag。
  列表长度必须与 macro 实现体内、同一 pipe 对上的 `set_flag`/`wait_flag` 对数一致。
  详见 [Macro sync_event_slots](#macro-sync-event-slots)。
- **iterator_types**：逐操作数的迭代器语义，供结构化降级与 flatten 类 Pass 使用
  （`HIVM_IteratorTypeAttr`）。可选；设置时长度应覆盖参与 tiling 的输入与输出。
- **indexing_map**：逐操作数的仿射索引映射（与 Linalg `indexing_maps` 作用相同）。
  可选 `ArrayAttr<AffineMapAttr>`。
- **max_rank**：flatten/布局类 Pass 支持的最大张量秩。可选 `i64` 属性，默认 **5**。
- **align_dim**：逐操作数的维度对齐提示，通过 **arg_attrs** 挂在对应操作数上（见下）。
- **arg_attrs**：逐操作数字典属性的 `ArrayAttr`（例如操作数 2 上的 `{align_dim = 1 : i64}`）。
  Triton 前端根据注册类上的 `align_dim` 自动生成。
- **extra_buffers_types** / **extra_buffers_sizes**：临时 buffer 的元素类型与一维大小
  （元素个数）。`hivm-alloc-extra-buffer` Pass 会分配 `memref` 并追加到 **temp_buffers**
  （汇编中为 `tmps`）。
- **temp_buffers**：传给设备实现的临时 memref（`tmps` 操作数段）。通常由
  `extra_buffers_*` 属性自动填充。
- **no_side_effect**：单位属性，表示算子无副作用。
- **bitcode** / **source** / **compile**：实现产物路径及可选编译命令（通常由 Triton 前端设置）。

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
| **Pipe** | 自定义算子执行 pipe（`hivm.hir.custom`）。 |
| **InPipe / OutPipe** | Macro 自定义算子的输入/输出 pipe（`hivm.hir.custom_macro`）。 |
| **VFMode** | 自定义算子在向量核上的运行模式：SIMT/SIMD/MIX。 |
| **Symbol** | 使用者提供的函数名称 |
| **sync_event_slots** | Macro 同步槽位声明，用于 GraphSyncSolver 集成。 |
| **iterator_types** | Tiling / flatten 的逐操作数迭代器语义。 |
| **indexing_map** | 结构化降级的逐操作数仿射映射。 |
| **max_rank** | 布局 Pass 支持的最大张量秩（默认 5）。 |
| **align_dim** / **arg_attrs** | 对齐调整 Pass 的逐操作数对齐提示。 |
| **extra_buffers_*** / **tmps** | 临时 buffer 声明与分配。 |
| **no_side_effect** | 纯算子标记，便于优化。 |
| **内置算子** | 一组内置算子（名称预留）。 |

### ⚠️ 限制

| 限制 | 说明 | 状态 |
| ---------------------------- | --------------------------------------------------------- | ------------------------------------------------------- |
| **用户实现** | 自定义算子降级到用户提供的实现：<br>- HIVM IR 链接到用户提供的源码/Objects Files<br>- 向 bishengir-compile 注册特定链接命令 | 进行中。 |
| **Pass 交互** | 适配自定义算子的变换 Pass：<br>- Flatten 优化<br>- 对齐调整<br>- 内存规划<br>- 布局变换<br>- ... 更多待补充 | 不适用，进行中。 |

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

### 自定义 Macro 算子

```mlir
%0 = hivm.hir.custom_macro
      { hivm.tcore_type = #hivm.tcore_type<VECTOR>,
        hivm.pipe_in = #hivm.pipe<PIPE_MTE2>,
        hivm.pipe_out = #hivm.pipe<PIPE_V>,
        hivm.vf_mode = #hivm.vf_mode<SIMD>,
        symbol = "custom_macro_add_int32",
        sync_event_slots = [
          #hivm.sync_event_slot<#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_V>>
        ] }
      "macro_add"
      ins(%arg0, %arg1 : memref<?xi32>, memref<?xi32>)
      outs(%dst : memref<32xi32, #hivm.address_space<ub>>)
```

<a id="macro-sync-event-slots"></a>

#### Macro `sync_event_slots`：单槽位与双槽位

在 **InPipe**（如 `PIPE_MTE2`）搬运数据、在 **OutPipe**（如 `PIPE_V`）上计算的 Macro
自定义算子，通常需要在设备实现中插入 `set_flag` / `wait_flag`，保证 MTE2 写完 UB 后
向量 pipe 再读取。

在 Triton 侧通过 `sync_event_slots` 与 `SYNC_HINT.INTERNAL` 声明这些**宏内部**同步。
**GraphSyncSolver** 为每个槽位分配独立 event id，并在需要时在 macro 前后补充
set/wait，使其与周围 HIVM 算子正确衔接。

**规则：** macro 体内每一对相同 pipe 方向的 `set_flag`/`wait_flag`，在 Python/MLIR 中
对应声明一个 `sync_event_slots` 条目。

| 模式 | Python / MLIR 槽位数 | 典型设备代码 | 适用场景 |
| ---- | -------------------- | ------------ | -------- |
| **单槽位** | 1 × `(PIPE_MTE2, PIPE_V, INTERNAL)` | 两次 GM→UB 搬运完成后，一次 `set_flag` + `wait_flag`，再执行向量计算 | 默认：两次 load 连续执行，向量开始前只需一次 MTE2→V 同步 |
| **双槽位** | 2 × `(PIPE_MTE2, PIPE_V, INTERNAL)` | `load` → `set_flag(0)` → `load` → `set_flag(1)` → `wait_flag(0)` → `wait_flag(1)` → 向量计算 | 每次 GM→UB 需要独立 event（流水重叠、或需按次确认搬运完成） |

**单槽位 — 原因**

两个操作数都在 MTE2 上搬入 UB 后，向量 pipe 才开始计算。一个 event 即可表示「本 macro
所需的 MTE2 输入搬运已全部完成」，PIPE_V 可安全读取两块 UB scratch：

```text
load_gm_to_ubuf(src0)
load_gm_to_ubuf(src1)
set_flag(MTE2 → V, event 0)
wait_flag(MTE2 → V, event 0)
vector_vadd(...)
```

**双槽位 — 原因**

第一次 load 后立即 `set_flag`，可在追踪第一次搬运完成的同时启动第二次 GM→UB。
每个槽位对应 GraphSyncSolver 分配的一个 event id：

```text
load_gm_to_ubuf(src0)
set_flag(MTE2 → V, event 0)
load_gm_to_ubuf(src1)
set_flag(MTE2 → V, event 1)
wait_flag(MTE2 → V, event 0)
wait_flag(MTE2 → V, event 1)
vector_vadd(...)
```

双槽位对应的 MLIR：

```mlir
sync_event_slots = [
  #hivm.sync_event_slot<#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_V>>,
  #hivm.sync_event_slot<#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_V>>
]
```

若 Python/MLIR 中的槽位数量与 `.cpp` 实现中的 `set_flag`/`wait_flag` 对数不一致，
GraphSyncSolver 分配的 event id 将与 kernel 不匹配，是常见的集成错误。

#### Tiling 属性（`iterator_types`、`indexing_map`、`max_rank`）

这些属性使 CustomOp 实现 HIVM 结构化算子接口（`getIteratorTypesArray`、
`getIndexingMaps`），flatten、broadcast 与布局 Pass 可像处理原生结构化算子一样分析操作数。

支持的 `iterator_types` 取值：`parallel`、`broadcast`、`transpose`、`reduction`、
`interleave`、`deinterleave`、`inverse`、`pad`、`concat`、`gather`、`cumulative`、`opaque`。

MLIR 示例（摘自 `custom-op-attrs.mlir`）：

```mlir
#map2d = affine_map<(d0, d1) -> (d0, d1)>
%0 = hivm.hir.custom
    {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
     hivm.pipe = #hivm.pipe<PIPE_V>,
     hivm.vf_mode = #hivm.vf_mode<SIMD>,
     symbol = "k_named_maps",
     max_rank = 5 : i64,
     iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>],
     indexing_map = [#map2d, #map2d, #map2d]}
    "user.named_maps"
    ins(%arg0, %arg1 : memref<2x2xf32>, tensor<2x2xf32>)
    outs(%empty : tensor<2x2xf32>) -> tensor<2x2xf32>
```

Triton 注册示例：

```python
@al.register_custom_op
class tiled_custom_op:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    symbol = "my_tiled_func"
    bitcode = "/path/to/kernel.bc"
    iterator_types = [
        al.IteratorType.Parallel,
        al.IteratorType.Broadcast,
    ]

    def __init__(self, x, y, out=None):
        # 每个结构化操作数（输入 + 输出）对应一个仿射映射。
        self.indexing_map = [
            al.affine_map.get_identity(2),
            al.affine_map.get_identity(2),
            al.affine_map.get_identity(2),
        ]
```

#### 操作数对齐（`align_dim`、`arg_attrs`）

`align_dim` 标记某操作数应在哪个维度上对齐。在 MLIR 中体现为 `arg_attrs` 里该操作数
索引上的字典项：

```mlir
%0 = hivm.hir.custom { ... }
    "my_custom_op"
    ins(%arg0 {align_dim = 1 : i64}, %arg1 {align_dim = 0 : i64} : memref<?xf32>, memref<?xf32>)
    outs(%dst : tensor<?xf32>) -> tensor<?xf32>
```

Triton 示例（在 `__init__` 中设置；键为参数名或位置索引）：

```python
def __init__(self, x, ptr1, ptr2, out=None):
    self.align_dim = {"ptr2": 1, 1: 0}  # ptr2 第 1 维；第 2 个参数第 0 维
```

#### 额外临时 buffer（`extra_buffers_*`、`temp_buffers`）

在算子上声明临时 buffer 类型与大小；`hivm-alloc-extra-buffer` 会分配一维 memref 并
接入 `tmps` 段：

```mlir
// 经 hivm-alloc-extra-buffer 之后：
%alloc0 = memref.alloc() : memref<512xf32>
%alloc1 = memref.alloc() : memref<128xf16>
%0 = hivm.hir.custom
     { extra_buffers_types = [f32, f16],
       extra_buffers_sizes = [512 : i64, 128 : i64], ... }
     "my_custom_op"
     ins(%arg0 : memref<2x2xf32>)
     tmps(%alloc0, %alloc1 : memref<512xf32>, memref<128xf16>)
     outs(%empty : tensor<2x2xf32>) -> tensor<2x2xf32>
```

Triton 示例（单个 buffer 用元组，多个用 `(dtype, size)` 列表）：

```python
@al.register_custom_op
class my_custom_op_extra_buf:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    symbol = "my_extra_buf_func"
    bitcode = "/path/to/kernel.bc"

    def __init__(self, x, out=None):
        self.extra_buffers = [
            (tl.bfloat16, 256),
            (tl.float32, 512),
        ]
        # 或单个 buffer：self.extra_buffers = (tl.float16, 128)
```

设备 kernel 按 `extra_buffers` 声明顺序，通过 `tmps` 操作数接收分配的 scratch memref。

#### `no_side_effect`

标记仅读取输入、写入声明输出的算子：

```mlir
%0 = hivm.hir.custom {no_side_effect, symbol = "pure_kernel", ...}
    "pure_op" ins(...) outs(...) -> ...
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

#### TRITON 自定义 Macro 算子降级示例

Python脚本：`test_custom_macro_op.py`

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

import torch
import torch_npu
import pytest

@al.register_custom_op
class macro_add:
    core = al.CORE.VECTOR
    pipe = (al.PIPE.PIPE_MTE2, al.PIPE.PIPE_V)
    mode = al.MODE.SIMD
    sync_event_slots = [
        (al.PIPE.PIPE_MTE2, al.PIPE.PIPE_V, al.SYNC_HINT.INTERNAL),
    ]

    def __init__(self, a, b, out=None):
        assert out is not None, "out is required"
        self.symbol = "custom_macro_add_int32"
        self.bitcode = "macro_add.bc"

@triton.jit
def triton_custom_macro_add(output_ptr, a_ptr, b_ptr, L: tl.constexpr):
    idx = tl.arange(0, L)
    a = tl.load(a_ptr + idx)
    b = tl.load(b_ptr + idx)
    buf = tl.full([L], 0, tl.int32)
    res = al.custom("macro_add", a, b, out=buf)
    tl.store(output_ptr + idx, res)

@pytest.mark.parametrize("L", [32])
def test_custom_macro(L):
    a = torch.ones(L, dtype=torch.int32).npu()
    b = torch.ones(L, dtype=torch.int32).npu()
    out = torch.zeros(L, dtype=torch.int32).npu()
    triton_custom_macro_add[(1,)](out, a, b, L)
    torch.testing.assert_close(out.cpu(), (a + b).cpu())
```

CPP API 定义：`macro_add.cpp`

```c++
#define __aiv__ [aicore]
#define INTRINSIC_NO_ARGS(NAME) NAME()
#define INTRINSIC(NAME, ...) NAME(__VA_ARGS__)

constexpr int64_t UB_SCRATCH_SRC0_BYTES = 256;
constexpr int64_t UB_SCRATCH_SRC1_BYTES = 384;
constexpr int64_t kGmToUbChunkElems = 16;

template <typename T, size_t Dim>
struct memref_t {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[Dim];
  int64_t strides[Dim];
};

extern "C" {
__aiv__ __attribute__((always_inline)) void _mlir_ciface_custom_macro_add_int32(
    memref_t<__gm__ int32_t, 1> *src0, memref_t<__gm__ int32_t, 1> *src1,
    memref_t<__ubuf__ int32_t, 1> *dst) {
  const int64_t n = dst->sizes[0];
  auto ub_src0 = ub_scratch_at_bytes(UB_SCRATCH_SRC0_BYTES, n);
  auto ub_src1 = ub_scratch_at_bytes(UB_SCRATCH_SRC1_BYTES, n);
  memref_t<__gm__ int32_t, 1> gm_src0 = {src0->allocated, src0->aligned,
                                         src0->offset, {n}, {1}};
  memref_t<__gm__ int32_t, 1> gm_src1 = {src1->allocated, src1->aligned,
                                         src1->offset, {n}, {1}};

  // 单内部 event：两次 GM→UB 后一次 MTE2→V 握手。
  load_gm_to_ubuf_1d(&gm_src0, &ub_src0);
  load_gm_to_ubuf_1d(&gm_src1, &ub_src1);
  INTRINSIC(set_flag, PIPE_MTE2, PIPE_V, 0);
  INTRINSIC(wait_flag, PIPE_MTE2, PIPE_V, 0);
  vector_vadd_ub(&ub_src0, &ub_src1, dst);
}
}
```

编译`.bc`文件指令：

```bash
ccec -x cce --cce-aicore-arch=dav-c220-vec --cce-aicore-only -c -emit-llvm ./macro_add.cpp -o ./macro_add.bc
```

Python脚本执行命令：

```bash
python -m pytest -sv test_custom_macro_op.py
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
