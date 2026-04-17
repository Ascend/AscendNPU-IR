# CustomOp

## Overview

AscendNPU-IR already supports a rich operator set for upstream models. However, in certain scenarios, there are needs to define their own operators to perform custom computations:

- Supported operators' combination couldn't fulfill desired computations.
- Vendor wants the custom operator to be private.
- Combining multiple operators could not reach optimal performance.

Custom operator allows users to freely use the interfaces provided by AscendNPU-IR to provide their own operators that compiles with other operators.

### Hardware Background

N/A

### Algorithm Principle

N/A

### Interface Description

Generic interface for custom op as following:

- name : unique op name.

         Note : there are names reserved for builtins, usually starts with "__builtin".
                Compiler will link these builtins to self-contained template library,
                which comes together within bishengir-compile.

                For normal names/cases, user needs to specify implementation location/compilation commands,
                and all the necessary information.

- inputs : input parameters.
- outputs : output results, designated "init" operands, which act as initial values for the results
            of the operation or the init locations to which the results of the op will be written.

In order to adapt to future enhancements quickly and dynamically, custom op relies on attributes
to retrieve necessary information:

- CoreType : which core type to execute on, refer to TCoreTypeAttr.
- Pipe     : which pipe to execute on, refer to PipeAttr.
- VFMode   : which mode to run on vector units, refer to VFModeAttr.
             this attribute is ignored when core type is cube.

             Note : for builtins, user could specify these information or not,
                    compiler will help to check the correctness and canonicalize.
- Symbol   : Implementation function name

TODO:

- Implementation linkage       : user provided implementation and linking process.
- Multi-Pipes (Macro CustomOp) : custom op that uses multiple pipes, which is a MacroOp in HIVM's context.

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
| **Pipe**                        | Custom op execution pipe.                                    |
| **VFMode**                      | Custom op running mode on vector core, SIMT/SIMD/MIX.        |
| **Symbol**                      | User provided implementation function name                   |
| **Builtins**                    | Set of builtins (name reserved).                             |

#### ⚠️ Limitations

| Limitation                   | Description                                               | Status                                                                 |
| ---------------------------- | --------------------------------------------------------- | ------------------------------------------------------- |
| **User implementations**     | Custom op lowered to user provided implementations:       | Work in progress.
|                              | - HIVM IR link to user provided sources/objects           |
|                              | - Specific commands registration to bishengir-compile     |
| **Passes interactions**      | Transformation passes that adapt to custom op:            | NA, work in progress.
|                              | - Flatten optimization                                    |
|                              | - Alignment adjustment                                    |
|                              | - Memory planning                                         |
|                              | - Layout transformation                                   |
|                              | - ... more to go                                          |

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

#### TRITON CustomOp Lowering Example

Python script: `test_custom_op.py`

```python
# For more detail of Triton custom op design, please refer to
# https://gitcode.com/Ascend/triton-ascend/pull/988 for more details

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
