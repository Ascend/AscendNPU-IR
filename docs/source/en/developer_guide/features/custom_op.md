# Custom Operators

## Overview

AscendNPU IR already provides a rich set of operators for upstream models. However, in some business scenarios, users still need custom operators to implement dedicated computation logic. Typical scenarios include:

- The combination of existing operators cannot meet the required computation needs.
- Vendors want to keep custom operators private.
- The combination of multiple operators cannot achieve optimal performance.

Custom operators allow users to freely use the interfaces provided by AscendNPU IR and provide their own operators that can be compiled together with other operators.

## Interface Description

**Parameters**:

| Parameter | Description |
| --------- | ----------- |
| `name`    | Unique operator name.<br>Note: Some names are reserved for built-in operators, mostly prefixed with `__builtin`. The compiler automatically links such built-in operators to the built-in template library bundled with `bishengir-compile`, requiring no additional user configuration.<br>If a custom operator name is used, the user must specify the implementation location, the compilation command, and all necessary information. |
| `inputs`  | Input parameters. |
| `outputs` | Output results, which can be specified as the `init` operand, used as the initial value of the operation result or the initial location where the operation result is written. |

**Attributes**:

| Attribute | Description | Remarks/Example |
|--------|------|----------|
| `CoreType` | The core type on which the operator executes. | See `TCoreTypeAttr`. |
| `Pipe` | The pipe on which the operator executes (used for `hivm.hir.custom`). | See `PipeAttr`. |
| `InPipe` | The input pipe of a Macro custom operator. | See `PipeAttr` (`hivm.pipe_in`). |
| `OutPipe` | The output pipe of a Macro custom operator. | See `PipeAttr` (`hivm.pipe_out`). |
| `VFMode` | The running mode on the vector unit. | See `VFModeAttr`. This attribute is ignored when the core type is Cube. Note: Built-in operators may or may not specify it; the compiler checks correctness and normalizes it. |
| `Symbol` | The name of the implementation function. | - |
| `sync_event_slots` | Synchronization slot metadata of a Macro custom operator. `GraphSyncSolver` fills `sync_related_args` accordingly and injects `set/wait flag` before and after the macro. | The list length must match the number of `set_flag`/`wait_flag` pairs on the same pipe pair within the macro implementation body. See [Macro sync_event_slots](#macro-operator-sync-event-slots-sync_event_slots) for details. |
| `iterator_types` | Per-operand iterator semantics, used by structured lowering and flatten-type passes (`HIVM_IteratorTypeAttr`). | Optional; when set, the length should cover the inputs and outputs participating in tiling. |
| `indexing_map` | Per-operand affine index mapping (same role as Linalg's `indexing_maps`). | Optional, of type `ArrayAttr<AffineMapAttr>`. |
| `max_rank` | The maximum tensor rank supported by flatten/layout-type passes. | Optional `i64` attribute, default value 5. |
| `align_dim` | Per-operand dimension alignment hint. | Attached to the corresponding operand via `arg_attrs`. |
| `arg_attrs` | Per-operand dictionary attribute array. | Of type `ArrayAttr` (for example, `{align_dim = 1 : i64}` on operand 2). The Triton frontend generates it automatically based on `align_dim` on the registered class. |
| `extra_buffers_types` | The element type of temporary buffers. | The `hivm-alloc-extra-buffer` pass allocates `memref` and appends them to `temp_buffers` (`tmps` in assembly). |
| `extra_buffers_sizes` | The one-dimensional size (number of elements) of temporary buffers. | Same as above. |
| `temp_buffers` | Temporary `memref` passed to the device implementation (the `tmps` operand segment). | Usually filled automatically by the `extra_buffers_*` attributes; no manual setting is required. |
| `no_side_effect` | Indicates that the operator has no side effects. | - |
| `bitcode` / `source` / `compile` | The path of the implementation artifact and an optional compilation command. | Usually set by the Triton frontend. |

## Lowering Flow

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
│  • built-in operator                                             │
│    -> call the built-in library                                  │
│  • user-provided implementation ->                               │
|    -> call the user-provided function name                      |
|      -> bishengir-compile links using the user-provided link command |
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
            BiSheng Compiler compiles them into Object Files
```

## Supported Capabilities

| Feature | Description |
| ------------------------------- | ------------------------------------------------------------ |
| CoreType | Execution core of the custom operator. |
| Pipe | Pipe on which the custom operator executes (`hivm.hir.custom`). |
| InPipe / OutPipe | Input/output pipe of the Macro custom operator (`hivm.hir.custom_macro`). |
| VFMode | Running mode of the custom operator on the vector core: SIMT/SIMD/MIX. |
| Symbol | Function name provided by the user. |
| sync_event_slots | Macro sync slot declaration, used for GraphSyncSolver integration. |
| iterator_types | Per-operand iterator semantics for Tiling / flatten. |
| indexing_map | Per-operand affine mapping for structured lowering. |
| max_rank | Maximum tensor rank supported by the layout pass (default 5). |
| align_dim / arg_attrs | Per-operand alignment hints for the alignment adjustment pass. |
| extra_buffers_* / tmps | Temporary buffer declaration and allocation. |
| no_side_effect | Pure operator marker for optimization. |
| built-in operator | A set of built-in operators (names reserved). |

## Constraints

The current custom operator system has two capabilities that are still being improved, as follows:

- **Incomplete adaptation of user-side implementation**

  It supports lowering a custom operator to a user-provided implementation. The related process includes linking HIVM IR to user-provided source code or object files and registering specific link commands with bishengir-compile. This feature is still under development.

- **Incomplete adaptation of pass interaction**

  The transformation passes for various custom operators, including Flatten optimization, alignment adjustment, memory planning, and layout transformation, still require additional adaptation logic, and the related work is in progress.

## MLIR Example

### Operator Declaration Example

**Built-in operator**:

```mlir
%0 = hivm.hir.custom
       "__builtin_gather_load"
       ins(%arg0, %arg1, %c4_i64, %c0_i32, %c2_i64, %c1_i64, %c2_i32, %c2_i32, %c0_i32, %c0_i32
           : memref<?xf32>, tensor<3x3xi64>, i64, i32, i64, i64, i32, i32, i32, i32)
       outs(%empty : tensor<3x3xf32>) -> tensor<3x3xf32>

```

**Custom operator**:

```mlir
%0 = hivm.hir.custom
      { hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.vf_mode = #hivm.vf_mode<SIMD>,
        symbol = "my_custom" }
      "my_custom_op"
      ins(%arg0, %arg1, %c4_i64, %c0_i32, %c2_i64, %c1_i64, %c2_i32, %c2_i32, %c0_i32, %c0_i32
          : memref<?xf32>, tensor<3x3xi64>, i64, i32, i64, i64, i32, i32, i32, i32)
      outs(%empty : tensor<3x3xf32>) -> tensor<3x3xf32>
```

**Custom Macro operator**:

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

### Core Attribute Usage Example

#### Macro Operator Sync Event Slots (sync_event_slots)

For a Macro custom operator that moves data in the InPipe (such as `PIPE_MTE2`) and performs computation in the OutPipe (such as `PIPE_V`), it is usually necessary to insert `set_flag` and `wait_flag` operations in the device implementation to ensure that the vector Pipe starts reading only after MTE2 has finished writing data into the UB.

On the Triton side, declare these intra-macro synchronization relationships through `sync_event_slots` and `SYNC_HINT.INTERNAL`.

GraphSyncSolver assigns an independent event ID to each slot and, when necessary, inserts set/wait operations before and after the macro operator to ensure that it correctly connects with the upstream and downstream HIVM operators.

**Rule**: In the macro implementation body, each pair of `set_flag`/`wait_flag` for Pipes in the same direction requires a corresponding `sync_event_slots` entry to be declared in Python/MLIR.

| Mode | Python/MLIR Slot Count | Typical Device Code | Applicable Scenario |
| ------ | ----------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Single slot | 1 × `(PIPE_MTE2, PIPE_V, INTERNAL)` | After two GM→UB transfers complete, one `set_flag` + `wait_flag`, then vector computation | Default scenario: two loads execute consecutively, and only one MTE2-to-V synchronization is required before vector computation starts |
| Dual slot | 2 × `(PIPE_MTE2, PIPE_V, INTERNAL)` | `load` → `set_flag(0)` → `load` → `set_flag(1)` → `wait_flag(0)` → `wait_flag(1)` → vector computation | Scenarios where each GM→UB transfer requires an independent event, such as pipeline overlap or per-transfer completion confirmation |

**Single slot description**:

Only after both operands have been moved into the UB through MTE2 does the vector Pipe start computation. A single event is sufficient to indicate that all MTE2 input transfers required by this macro have completed, and PIPE_V can safely read the two UB temporary buffers. The corresponding pseudocode is as follows:

```text
load_gm_to_ubuf(src0)
load_gm_to_ubuf(src1)
set_flag(MTE2 → V, event 0)
wait_flag(MTE2 → V, event 0)
vector_vadd(...)
```

**Dual slot description**:

Executing `set_flag` immediately after the first load allows the second GM-to-UB transfer to start while waiting for the first transfer to complete. Each slot corresponds to an independent event ID assigned by GraphSyncSolver. The corresponding pseudocode is as follows:

```text
load_gm_to_ubuf(src0)
set_flag(MTE2 → V, event 0)
load_gm_to_ubuf(src1)
set_flag(MTE2 → V, event 1)
wait_flag(MTE2 → V, event 0)
wait_flag(MTE2 → V, event 1)
vector_vadd(...)
```

The MLIR corresponding to the dual slots:

```mlir
sync_event_slots = [
  #hivm.sync_event_slot<#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_V>>,
  #hivm.sync_event_slot<#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_V>>
]
```

If the number of slots declared in Python/MLIR does not match the number of `set_flag`/`wait_flag` pairs in the C++ implementation, the event IDs allocated by GraphSyncSolver will not match the kernel. This is a common error when integrating custom Macro operators.

#### Tiling attributes (iterator_types, indexing_map, max_rank)

By configuring these attributes, CustomOp can implement the HIVM structured operator interfaces (`getIteratorTypesArray`, `getIndexingMaps`), enabling the flatten, broadcast, and layout passes to analyze and optimize the operands of custom operators in the same way they handle native structured operators.

The supported values of `iterator_types` include: `parallel`, `broadcast`, `transpose`, `reduction`, `interleave`, `deinterleave`, `inverse`, `pad`, `concat`, `gather`, `cumulative`, and `opaque`.

MLIR example (excerpted from `custom-op-attrs.mlir`):

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

Triton registration example:

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
        # Each structured operand (input + output) corresponds to one affine map.
        self.indexing_map = [
            al.affine_map.get_identity(2),
            al.affine_map.get_identity(2),
            al.affine_map.get_identity(2),
        ]
```

#### Operand Alignment Attribute (align_dim, arg_attrs)

`align_dim` is used to mark the dimension of a specified operand that needs to be aligned. In MLIR, this attribute is attached to the operand at the corresponding index through `arg_attrs`. An example is as follows:

```mlir
%0 = hivm.hir.custom { ... }
    "my_custom_op"
    ins(%arg0 {align_dim = 1 : i64}, %arg1 {align_dim = 0 : i64} : memref<?xf32>, memref<?xf32>)
    outs(%dst : tensor<?xf32>) -> tensor<?xf32>
```

Triton example (set in `__init__`, with the key being the parameter name or positional index):

```python
def __init__(self, x, ptr1, ptr2, out=None):
    self.align_dim = {"ptr2": 1, 1: 0}  # Dimension 1 of ptr2; dimension 0 of the second parameter.
```

#### Temporary Buffer Attributes (extra_buffers_*, temp_buffers)

The type and size of temporary buffers can be declared on the operator through the related attributes. `hivm-alloc-extra-buffer` automatically allocates a one-dimensional memref and connects it to the `tmps` operand segment of the operator. An example is as follows:

```mlir
// After hivm-alloc-extra-buffer:
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

Triton example (a tuple for a single buffer, or a list of `(dtype, size)` for multiple buffers):

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
        # Or for a single buffer: self.extra_buffers = (tl.float16, 128)
```

The device kernel receives the allocated scratch memref through the `tmps` operands in the declaration order of `extra_buffers`.

#### no_side_effect

This attribute marks a pure operator that only reads inputs and only writes declared outputs. An example is as follows:

```mlir
%0 = hivm.hir.custom {no_side_effect, symbol = "pure_kernel", ...}
    "pure_op" ins(...) outs(...) -> ...
```

### Triton custom operator example

#### Standard custom operator example

Python script: `test_custom_op.py`

```python
# For more details about Triton custom operator design, see
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

C++ API definition: `add.cpp`

```cpp
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

Command to compile the `.bc` file:

```bash
ccec -x cce --cce-aicore-arch=dav-c220-vec --cce-aicore-only -c -emit-llvm ./add.cpp -o ./add.bc
```

Command to run the Python script:

```bash
python -m pytest -sv test_custom_op.py
```

#### Macro custom operator example

Python script: `test_custom_macro_op.py`

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

C++ API definition: `macro_add.cpp`

```cpp
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

  // Single internal event: two GM→UB transfers followed by one MTE2→V handshake.
  load_gm_to_ubuf_1d(&gm_src0, &ub_src0);
  load_gm_to_ubuf_1d(&gm_src1, &ub_src1);
  INTRINSIC(set_flag, PIPE_MTE2, PIPE_V, 0);
  INTRINSIC(wait_flag, PIPE_MTE2, PIPE_V, 0);
  vector_vadd_ub(&ub_src0, &ub_src1, dst);
}
}
```

Command to compile the `.bc` file:

```bash
ccec -x cce --cce-aicore-arch=dav-c220-vec --cce-aicore-only -c -emit-llvm ./macro_add.cpp -o ./macro_add.bc
```

Python script execution command:

```bash
python -m pytest -sv test_custom_macro_op.py
```

Lowered to MLIR:

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
