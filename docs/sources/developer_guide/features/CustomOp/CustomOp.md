# CustomOp

## Overview

AscendNPU-IR already supports a rich operator set for upstream models. However, in certain scenarios, there are needs to define their own operators to perform custom computations:

- Supported operaters' combination couldn't fulfill desired computations.
- Vendor wants the custom operator to be private.
- Combining multiple operators could not reach optimal performance.

Custom operator allows users to freely use the interfaces provided by AscendNPU-IR to provide their own operators that compiles with other operators.

---

## Interface

Generic interface for custom op as following:
- name : unique op name.

         Note : there are names reserved for builtins, usually starts with "__builtin".
                Compiler will link these builtins to self-contained template library,
                which comes together within bishengir-compile.

                For normal names/cases, user needs to specify implementation location/compilation commands (TODO),
                and all ther necessary informations.

         Available builtin names :
            "__builtin_gather_load"

- inputs : input parameters.
- outputs : output results, designated "init" operands, which act as initial values for the results
            of the operation or the init locations to which the results of the op will be written.

In order to adapt to future enhancements quickly and dynamically, custom op relies on attributes
to retreive necessary information, required informations are:
- CoreType : which core type to execute on, refer to TCoreTypeAttr.
- Pipe     : which pipe to execute on, refer to PipeAttr.
- VFMode   : which mode to run on vector units, refer to VFModeAttr.
             this attribute is ignored when core type is cube.

             Note : for builtins, user could specify these informations or not,
                    compiler will help to check the correctness and canonicalize.

TODO:
- Impl : user provided implementation and linking process.
- Multi Pipe : custom op wants to use multiple pipes, which is a MacroOp in HIVM's context.

---

## Lowering Process

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
│  • Builtins                                                     │
│    -> call to builtins libraries                                │
│  • User provided implementations (WIP) ->                       │
|    -> call to user provided function name                       |
|      -> bishengir-compile link with user provided link commands |
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
            BiSheng Compiler compiles to objects
```
---

## Capability & Limitation

### ✅ Capabilities

| Feature                         | Description                                                  |
| ------------------------------- | ------------------------------------------------------------ |
| **CoreType**                    | Custom op execution core.                                    |
| **Pipe**                        | Custom op execution pipe.                                    |
| **VFMode**                      | Custom op running mode on vector core, SIMT/SIMD/MIX.        |
| **Buitlins**                    | Set of builtins (name reserved).                             |

### ⚠️ Limitations

| Limitation                   | Description                                               | Status                                                                 |
| ---------------------------- | --------------------------------------------------------- | ------------------------------------------------------- |
| **User implementations**     | Custom op lowered to user provided implementations:       | Work in progress.
|                              | - HIVM IR link to user provided sources/bitcodes/objects  |
|                              | - User implementations registeration                      |
|                              | - Spcific link commands registeration to bisheng-compile  |
| **Passes interactions**      | Transformation passes that adapt to custom op:            | NA, work in progress.
|                              | - Flatten optimization                                    |
|                              | - Alignment adjustment                                    |
|                              | - Memory planning                                         |
|                              | - Layout transformation                                   |
|                              | - ... more to go                                          |

---

## MLIR Example

### Builtin

```mlir
%0 = hivm.hir.custom
       "__builtin_gather_load"
       ins(%arg0, %arg1, %c4_i64, %c0_i32, %c2_i64, %c1_i64, %c2_i32, %c2_i32, %c0_i32, %c0_i32
           : memref<?xf32>, tensor<3x3xi64>, i64, i32, i64, i64, i32, i32, i32, i32)
       outs(%empty : tensor<3x3xf32>) -> tensor<3x3xf32>

```

### Custom

```mlir
%0 = hivm.hir.custom
      { hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.vf_mode = #hivm.vf_mode<SIMD> }
      "my_custom_op"
      ins(%arg0, %arg1, %c4_i64, %c0_i32, %c2_i64, %c1_i64, %c2_i32, %c2_i32, %c0_i32, %c0_i32
          : memref<?xf32>, tensor<3x3xi64>, i64, i32, i64, i64, i32, i32, i32, i32)
      outs(%empty : tensor<3x3xf32>) -> tensor<3x3xf32>
```

---

## TRITON CustomOp Lowering Example
```python
# For more detail of Triton custom op design, please refer to
# https://gitcode.com/Ascend/triton-ascend/pull/988 for more details

import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def builtin_index_select_kernel(src_ptr, index_ptr, out_ptr):
    # Define 2x2 tile indices for output tensor
    r = tl.arange(0, 2)[:, None]  # Row indices: shape [2, 1]
    c = tl.arange(0, 2)[None, :]  # Column indices: shape [1, 2]

    # Load index tensor (shape [2]) from GM to UB
    idx = tl.load(index_ptr + tl.arange(0, 2))
    # Initialize empty 2x2 output tile in UB (default value: 0)
    dst = tl.full((2, 2), 0, dtype=tl.float32)

    # Invoke __builtin_index_select custom op to gather elements
    out_tile = al.custom(
        "__builtin_index_select",
        src_ptr,          # Pointer to source tensor in GM
        idx,              # Index tensor (in UB) for gathering
        dim=0,            # Dimension to gather along
        bound=4,          # Upper bound for valid index values (out-of-bound check)
        end_offset=(2, 2),# End offsets of each dimension for the index tensor
        start_offset=(0, 0), # Start offsets of each dimension for the source tensor
        src_stride=(4, 1),# Stride of each dimension for the source tensor in GM
        out=dst           # Output tensor (in UB) to store gathered elements
    )

    # Store the gathered tile from UB to output tensor in GM
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
Lowering to MLIR

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
