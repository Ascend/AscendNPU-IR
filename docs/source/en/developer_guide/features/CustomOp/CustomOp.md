# CustomOp

## Overview

AscendNPU-IR already supports a rich operator set for upstream models. However, in certain scenarios, there are needs to define their own operators to perform custom computations:

- Supported operators' combination could not fulfill desired computations.
- The vendor wants custom operators to be private.
- Combining multiple operators could not reach optimal performance.

Custom operators allow users to freely use the APIs provided by AscendNPU-IR to provide their own operators that compiles with other operators.

### Hardware Background

N/A

### Algorithm Principle

N/A

### API Description

Generic API for custom op:

- **name**: unique op name.

         Note : there are names reserved for builtins, usually starting with "__builtin".
                Compiler will link these builtins to self-contained template library, 
                which comes together within bishengir-compile. 

                For normal names/cases, user needs to specify implementation location/compilation commands,
                and all the necessary information.

- **inputs**: input parameters.
- **outputs**: output results, designated "init" operands, which act as initial values for the results of the operation 
              or the init locations to which the results of the op will be written.

In order to adapt to future enhancements quickly and dynamically, custom op relies on attributes to retrieve necessary information:

- **CoreType**: which core type to execute on. Refer to TCoreTypeAttr.
- **Pipe**: which pipe to execute on. Refer to PipeAttr.
- InPipe   : input pipe for macro custom op, refer to PipeAttr (`hivm.pipe_in`).
- OutPipe  : output pipe for macro custom op, refer to PipeAttr (`hivm.pipe_out`).
- VFMode   : which mode to run on vector units, refer to VFModeAttr.
             this attribute is ignored when core type is cube.

             Note : for builtins, user could specify these information or not,
                    compiler will help to check the correctness and canonicalize.
- Symbol   : implementation function name.
- sync_event_slots : optional sync slot metadata for macro custom op. GraphSyncSolver
                     fills the corresponding `sync_related_args` and can inject
                     required set/wait flags around the macro. The list length must
                     match the number of internal `set_flag`/`wait_flag` pairs
                     declared in the macro implementation for the same pipe pair.
                     See [Macro sync_event_slots](#macro-sync-event-slots).
- iterator_types : per-operand iterator semantics for structured lowering and
                   flatten passes (`HIVM_IteratorTypeAttr`). Optional; when set,
                   length should cover inputs and outputs that participate in tiling.
- indexing_map : per-operand affine index maps (same role as Linalg `indexing_maps`).
                 Optional `ArrayAttr<AffineMapAttr>`.
- max_rank : maximum tensor rank supported by flatten/layout passes. Optional `i64`
             attribute; default is **5**.
- align_dim : per-operand dimension alignment hint. Stored on the matching operand
              via **arg_attrs** (see below).
- arg_attrs : `ArrayAttr` of per-operand dictionary attributes (for example
              `{align_dim = 1 : i64}` on operand 2). The Triton frontend builds this
              from `align_dim` on the registration class.
- extra_buffers_types / extra_buffers_sizes : scratch buffer element types and 1-D
              sizes (element counts). The `hivm-alloc-extra-buffer` pass allocates
              `memref` values and appends them to **temp_buffers** (`tmps` in assembly).
- temp_buffers : scratch memrefs passed to the device implementation (`tmps` operand
                 segment). Often populated automatically from `extra_buffers_*` attributes.
- no_side_effect : unit attribute; marks the op as side-effect free.
- bitcode / source / compile : path to the implementation artifact and optional compile
                               recipe (typically set by the Triton frontend).

## Lowering Process

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
│  • Builtins                                                     │
│    -> call to builtins libraries                                │
│  • User provided implementations ->                             │
|    -> call to user provided function name                       |
|      -> bishengir-compile link with user provided link commands |
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
            BiSheng Compiler compiles to objects
```

### Constraints and Capabilities

#### ✅ Capabilities

| Feature                         | Description                                                  |
| ------------------------------- | ------------------------------------------------------------ |
| **CoreType**                    | Custom op execution core.                                    |
| **Pipe**                        | Custom op execution pipe (`hivm.hir.custom`).                |
| **InPipe / OutPipe**            | Macro custom op input/output pipes (`hivm.hir.custom_macro`). |
| **VFMode**                      | Custom op running mode on vector core, SIMT/SIMD/MIX.        |
| **Symbol**                      | User provided implementation function name                   |
| **sync_event_slots**            | Macro sync-slot declaration for GraphSyncSolver integration. |
| **iterator_types**              | Tiling / flatten iterator semantics per operand.               |
| **indexing_map**                | Per-operand affine maps for structured lowering.               |
| **max_rank**                    | Upper bound on tensor rank for layout passes (default 5).      |
| **align_dim** / **arg_attrs**   | Per-operand alignment hints for adjustment passes.             |
| **extra_buffers_*** / **tmps**  | Scratch buffer declaration and allocation.                     |
| **no_side_effect**              | Pure-op marking for optimization.                              |
| **Builtins**                    | Set of builtins (name reserved).                             |

#### ⚠️ Limitations

| Limitation                   | Description                                               | Status                                                                 |
| ---------------------------- | --------------------------------------------------------- | ------------------------------------------------------- |
| **User implementations**     | Custom op lowered to user provided implementations:<br>- HIVM IR link to user provided sources/objects<br>- Specific commands registration to bishengir-compile | Work in progress. |
| **Passes interactions**      | Transformation passes that adapt to custom op:<br>- Flatten optimization<br>- Alignment adjustment<br>- Memory planning<br>- Layout transformation<br>- ... more to go | NA, work in progress. |

### MLIR Example

#### Builtin

```mlir
%0 = hivm.hir.custom
       "__builtin_gather_load"
       ins(%arg0, %arg1, %c4_i64, %c0_i32, %c2_i64, %c1_i64, %c2_i32, %c2_i32, %c0_i32, %c0_i32
           : memref<?xf32>, tensor<3x3xi64>, i64, i32, i64, i64, i32, i32, i32, i32)
       outs(%empty : tensor<3x3xf32>) -> tensor<3x3xf32>

```

#### Custom

```mlir
%0 = hivm.hir.custom
      { hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.vf_mode = #hivm.vf_mode<SIMD>,
        symbol = "my_custom" }
      "my_custom_op"
      ins(%arg0, %arg1, %c4_i64, %c0_i32, %c2_i64, %c1_i64, %c2_i32, %c2_i32, %c0_i32, %c0_i32
          : memref<?xf32>, tensor<3x3xi64>, i64, i32, i64, i64, i32, i32, i32, i32)
      outs(%empty : tensor<3x3xf32>) -> tensor<3x3xf32>
```

#### Custom Macro

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

#### Macro `sync_event_slots`: one slot vs two slots

Macro custom ops that move data on **InPipe** (for example `PIPE_MTE2`) and compute on
**OutPipe** (for example `PIPE_V`) often need explicit `set_flag` / `wait_flag` inside
the device implementation to synchronize MTE2 completion before the vector pipe reads UB.

Register those internal handshakes in Triton via `sync_event_slots` with
`SYNC_HINT.INTERNAL`. **GraphSyncSolver** maps each slot to a distinct event id and may
inject additional set/wait flags around the macro so it composes safely with surrounding
HIVM ops.

**Rule:** declare one `sync_event_slots` entry per internal `set_flag`/`wait_flag` pair
for the same producer/consumer pipe pair inside the macro body.

| Pattern | Slots in Python / MLIR | Typical device code | When to use |
| ------- | ---------------------- | ------------------- | ----------- |
| **Single slot** | 1 × `(PIPE_MTE2, PIPE_V, INTERNAL)` | Both GM→UB loads, then one `set_flag` + `wait_flag`, then vector op | Default when copies run back-to-back and a single MTE2→V fence before vector is enough |
| **Two slots** | 2 × `(PIPE_MTE2, PIPE_V, INTERNAL)` | `load` → `set_flag(0)` → `load` → `set_flag(1)` → `wait_flag(0)` → `wait_flag(1)` → vector op | When each GM→UB transfer needs its own event (pipeline overlap, or per-transfer visibility) |

**Single slot — why it works**

Both operands are copied to UB on MTE2 before any vector work. One event signals that
*all* MTE2 traffic required for the macro inputs has finished, so PIPE_V can safely
read both UB scratch buffers:

```text
load_gm_to_ubuf(src0)
load_gm_to_ubuf(src1)
set_flag(MTE2 → V, event 0)
wait_flag(MTE2 → V, event 0)
vector_vadd(...)
```

**Two slots — why you might need them**

Issuing `set_flag` after the first load lets MTE2 start the second transfer while the
compiler/runtime tracks completion per buffer. Each slot reserves a distinct event id for
GraphSyncSolver:

```text
load_gm_to_ubuf(src0)
set_flag(MTE2 → V, event 0)
load_gm_to_ubuf(src1)
set_flag(MTE2 → V, event 1)
wait_flag(MTE2 → V, event 0)
wait_flag(MTE2 → V, event 1)
vector_vadd(...)
```

MLIR for two internal slots:

```mlir
sync_event_slots = [
  #hivm.sync_event_slot<#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_V>>,
  #hivm.sync_event_slot<#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_V>>
]
```

Mismatch between the slot count in Python/MLIR and the `set_flag`/`wait_flag` count in
the `.cpp` implementation is a common integration bug: GraphSyncSolver will assign event
ids that no longer line up with the kernel.

#### Tiling attributes (`iterator_types`, `indexing_map`, `max_rank`)

These attributes connect CustomOp to HIVM structured-op interfaces (`getIteratorTypesArray`,
`getIndexingMaps`) so flatten, broadcast, and layout passes can reason about operands
the same way as native structured ops.

Supported `iterator_types` values: `parallel`, `broadcast`, `transpose`, `reduction`,
`interleave`, `deinterleave`, `inverse`, `pad`, `concat`, `gather`, `cumulative`, `opaque`.

MLIR example (from `custom-op-attrs.mlir`):

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
        # One affine map per structured operand (inputs + outputs).
        self.indexing_map = [
            al.affine_map.get_identity(2),
            al.affine_map.get_identity(2),
            al.affine_map.get_identity(2),
        ]
```

#### Operand alignment (`align_dim`, `arg_attrs`)

`align_dim` tags which dimension of a given operand should be aligned. In MLIR it
appears as a dictionary entry inside `arg_attrs` on that operand index:

```mlir
%0 = hivm.hir.custom { ... }
    "my_custom_op"
    ins(%arg0 {align_dim = 1 : i64}, %arg1 {align_dim = 0 : i64} : memref<?xf32>, memref<?xf32>)
    outs(%dst : tensor<?xf32>) -> tensor<?xf32>
```

Triton example (set in `__init__`; keys are argument names or positional indices):

```python
def __init__(self, x, ptr1, ptr2, out=None):
    self.align_dim = {"ptr2": 1, 1: 0}  # ptr2 dim 1; 2nd arg dim 0
```

#### Extra scratch buffers (`extra_buffers_*`, `temp_buffers`)

Declare scratch types and sizes on the op; `hivm-alloc-extra-buffer` allocates 1-D
memrefs and wires them into the `tmps` segment:

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

Triton example (single buffer as tuple, or list of `(dtype, size)` pairs):

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
        # Or a single buffer: self.extra_buffers = (tl.float16, 128)
```

The device kernel receives the allocated scratch memrefs through the `tmps` operands
in the same order as `extra_buffers`.

#### `no_side_effect`

Mark ops that only read inputs and write declared outputs:

```mlir
%0 = hivm.hir.custom {no_side_effect, symbol = "pure_kernel", ...}
    "pure_op" ins(...) outs(...) -> ...
```

#### TRITON CustomOp Lowering Example

Python script: `test_custom_op.py`

```python
# For more details of Triton custom op design, refer to
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

CPP API definition: `add.cpp`

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

Command for compiling the `.bc` file:

```bash
ccec -x cce --cce-aicore-arch=dav-c220-vec --cce-aicore-only -c -emit-llvm ./add.cpp -o ./add.bc
```

Command for Python script execution:

```bash
python -m pytest -sv test_custom_op.py
```

#### TRITON CustomMacroOp Lowering Example

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

CPP API definition: `macro_add.cpp`

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

  // Single internal event: both GM→UB loads, then one MTE2→V handshake.
  load_gm_to_ubuf_1d(&gm_src0, &ub_src0);
  load_gm_to_ubuf_1d(&gm_src1, &ub_src1);
  INTRINSIC(set_flag, PIPE_MTE2, PIPE_V, 0);
  INTRINSIC(wait_flag, PIPE_MTE2, PIPE_V, 0);
  vector_vadd_ub(&ub_src0, &ub_src1, dst);
}
}
```

Command for compiling the `.bc` file:

```bash
ccec -x cce --cce-aicore-arch=dav-c220-vec --cce-aicore-only -c -emit-llvm ./macro_add.cpp -o ./macro_add.bc
```

Command for Python script execution:

```bash
python -m pytest -sv test_custom_macro_op.py
```

Lowering to MLIR:

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
