# Framework Integration

AscendNPU IR supports integration of frameworks (PyTorch/TensorFlow/MindSpore) in two ways:

- **DSL integration**: Integrate through domain-specific languages such as Triton and TileLang, compiling operators into AscendNPU IR.
- **IR integration**: Integrate through IR representation, supporting multi-level access via Torch IR, Linalg/HFusion IR, and HIVM IR, with automatic operator fusion and splitting to generate high-performance Ascend-affinity operators.

## DSL Integration

AscendNPU IR supports integration with languages and frameworks such as Triton and TileLang, enabling third-party DSLs to support Ascend hardware and run custom operators on the NPU.

| Integration Method | Description |
|----------|------|
| [Triton Integration](triton_interface.md) | Uses Triton to write high-performance kernels and runs them on Ascend NPUs through Triton Ascend. Includes installation, environment, operator mapping, and Ascend extension instructions. |
| [TileLang Integration](tile_lang_interface.md) | Uses TileLang Ascend (a DSL based on tile-lang/TVM) to develop kernels for Ascend NPUs (such as GEMM, vector operations, and attention). Includes environment, build, and quick start. |

## IR Integration

AscendNPU IR supports multi-level IR integration, with different levels varying in abstraction degree and control granularity (see [IR Integration Overview - Multi-level IR Abstraction Architecture](interface_api.md#multi-level-ir-abstraction-architecture)):

- Torch IR: Framework-level ATen operators, converted to Linalg/HFusion through passes.
- Linalg/HFusion IR: The general tensor algebra layer and the hardware-aware fusion layer, where standard MLIR dialects express operator semantics, and HFusion automatically performs fusion, tiling, and scheduling.
- HIVM IR: The NPU instruction layer, which directly maps hardware instructions, explicitly controls the memory hierarchy (GM/UB/L1/L0) and the compute pipeline (Vector/Cube/MTE), and supports fine-grained tuning.

### Torch IR Integration

Directly use the ATen operators of the Torch dialect, automatically convert them to Linalg/HFusion Named Ops through passes such as `convert-torch-to-hfusion`, and then enter the automatic fusion and scheduling flow.

#### Torch-to-AscendNPU IR Conversion Flow

Torch IR is integrated into AscendNPU IR through the `torch-backend-to-named-op-backend-pipeline` conversion pipeline. The BiShengIR custom `convert-torch-to-hfusion` pass preferentially converts Torch ATen operators into Linalg/HFusion Named Ops, while uncovered operators fall back to the standard lowering path of the upstream torch-mlir. The main conversion stages are as follows:

- `convert-torch-to-hfusion`: A BiShengIR custom conversion that covers 55+ ATen operators to Linalg/HFusion Named Ops.
- `convert-torch-to-linalg`: An upstream torch-mlir conversion that handles the remaining operators.
- `convert-torch-to-scf` / `arith` / `tensor`: The upstream torch-mlir completes conversions such as control flow, arithmetic, and `tensor`.
- `func-backend-type-conversion`: Converts Torch types (`!torch.vtensor`) into standard `builtin` types (`tensor`).

**Example torch.mlir**:

```mlir
func.func @torch_mul(%arg0: !torch.vtensor<[4096],f16>, %arg1: !torch.vtensor<[1,56,4096],f16>) -> !torch.vtensor<[1,56,4096],f16>
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[4096],f16>, !torch.vtensor<[1,56,4096],f16> -> !torch.vtensor<[1,56,4096],f16>
  return %0 : !torch.vtensor<[1,56,4096],f16>
}
```

**Invocation method**: There are two methods, both of which share the same compilation pipeline.

- **Step-by-step conversion**: First convert Torch IR into Linalg/HFusion IR. This is suitable for scenarios where intermediate IR needs to be cached or inspected. After the conversion is complete, `torch_to_hfusion.mlir` can be used as input to continue compilation and generate the binary by following the [Linalg/HFusion IR Integration](#linalghfusion-ir-integration) flow.
    - Command: `bishengir-opt -torch-backend-to-named-op-backend-pipeline torch.mlir -o torch_to_hfusion.mlir`
    - Expected output: an MLIR text file (in `.mlir` format) containing the converted Linalg/HFusion IR. For example:

```mlir
func.func @torch.aten.mul_tensor(%arg0: tensor<4096xf16>, %arg1: tensor<1x56x4096xf16>) -> tensor<1x56x4096xf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %0 = tensor.empty() : tensor<1x56x4096xf16>
  %broadcasted = linalg.broadcast ins(%arg0 : tensor<4096xf16>) outs(%0 : tensor<1x56x4096xf16>) dimensions = [0, 1] 
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %arg1 : tensor<1x56x4096xf16>, tensor<1x56x4096xf16>) outs(%0 : tensor<1x56x4096xf16>) -> tensor<1x56x4096xf16>
  return %1 : tensor<1x56x4096xf16>
}
```

- **End-to-end compilation**: Use `bishengir-compile` to compile Torch IR directly into an executable binary, going through the complete Torch → HFusion → HIVM IR compilation pipeline.
    - Command: `bishengir-compile -enable-torch-compile=true -enable-hfusion-compile=true -enable-hivm-compile=true -target=Ascend910B1 torch.mlir -o torch_kernel.o`
    - Expected output: an Ascend NPU operator binary file (in `.o` format), which can run on the device in conjunction with the CANN runtime.

#### Supported Torch Operators

##### Elementwise Binary

| Torch Op | Conversion Target |
|----------|----------|
| `aten.add.Tensor` / `aten.add.Scalar` | `linalg.binary_fn<add>` |
| `aten.sub.Tensor` / `aten.sub.Scalar` | `linalg.binary_fn<sub>` |
| `aten.mul.Tensor` / `aten.mul.Scalar` | `linalg.binary_fn<mul>` |
| `aten.div.Tensor` / `aten.div.Scalar` | `linalg.binary_fn<div>` |
| `aten.maximum` | `linalg.binary_fn<max_signed>` |
| `aten.minimum` | `linalg.binary_fn<min_signed>` |
| `aten.clamp_min` / `aten.clamp_min.Tensor` | `linalg.binary_fn<max_signed>` |
| `aten.clamp_max` / `aten.clamp_max.Tensor` | `linalg.binary_fn<min_signed>` |
| `aten.clamp` | `max_signed` + `min_signed` combination |
| `aten.pow.Tensor_Tensor` / `aten.pow.Tensor_Scalar` / `aten.pow.Scalar` | `hfusion.binary_fn<powf>` |
| `aten.logical_and` | `hfusion.binary_fn<vand>` |
| `aten.logical_or` | `hfusion.binary_fn<vor>` |

##### Elementwise Unary

| Torch Op | Conversion Target |
|----------|----------|
| `aten.abs` | `linalg.unary_fn<abs>` |
| `aten.ceil` | `linalg.unary_fn<ceil>` |
| `aten.floor` | `linalg.unary_fn<floor>` |
| `aten.neg` | `linalg.unary_fn<negf>` |
| `aten.log` | `linalg.unary_fn<log>` |
| `aten.exp` | `linalg.unary_fn<exp>` |
| `aten.reciprocal` | `hfusion.unary_fn<rec>` |
| `aten.relu` | `hfusion.unary_fn<relu>` |
| `aten.rsqrt` | `hfusion.unary_fn<rsqrt>` |
| `aten.sqrt` | `hfusion.unary_fn<sqrt>` |
| `aten.erf` | `hfusion.unary_fn<erf>` |
| `aten.tanh` | `hfusion.unary_fn<tanh>` |
| `aten.sin` | `hfusion.unary_fn<sin>` |
| `aten.cos` | `hfusion.unary_fn<cos>` |
| `aten.bitwise_not` | `hfusion.unary_fn<vnot>` |
| `aten.sigmoid` | Decomposed into `negf` -> `exp` -> `add` -> `div` |
| `aten.gelu` | Decomposed into a `tanh` approximation |

##### Compare

| Torch Op | Conversion Target |
|----------|----------|
| `aten.gt.Scalar` / `aten.gt.Tensor` | `hfusion.compare_fn<vgt>` |
| `aten.lt.Scalar` / `aten.lt.Tensor` | `hfusion.compare_fn<vlt>` |
| `aten.ge.Scalar` / `aten.ge.Tensor` | `hfusion.compare_fn<vge>` |
| `aten.le.Scalar` / `aten.le.Tensor` | `hfusion.compare_fn<vle>` |
| `aten.eq.Scalar` / `aten.eq.Tensor` | `hfusion.compare_fn<veq>` |
| `aten.ne.Scalar` / `aten.ne.Tensor` | `hfusion.compare_fn<vne>` |

##### Reduction

| Torch Op | Conversion Target |
|----------|----------|
| `aten.sum` / `aten.sum.dim_IntList` | `linalg.reduce` + `arith.addf`/`addi` |
| `aten.prod` / `aten.prod.dim_int` | `linalg.reduce` + `arith.mulf`/`muli` |
| `aten.max` | `linalg.reduce` + `arith.maximumf`/`maxsi` |
| `aten.min` | `linalg.reduce` + `arith.minimumf`/`minsi` |
| `aten.max.dim` | `hfusion.reduce_with_index` (MAX) |
| `aten.min.dim` | `hfusion.reduce_with_index` (MIN) |
| `aten.any` / `aten.any.dim` / `aten.any.dims` | `linalg.reduce` + `arith.ori` |
| `aten.all` / `aten.all.dim` | `linalg.reduce` + `arith.andi` |

##### Data Movement

| Torch Op | Conversion Target |
|----------|----------|
| `aten.permute` | `linalg.transpose` |
| `aten.broadcast_to` | `linalg.broadcast` |

##### Others

| Torch Op | Conversion Target |
|----------|----------|
| `aten.to.dtype` | `hfusion.cast` |
| `aten.where.self` | `hfusion.select` |
| `aten.arange.start_step` | `hfusion.arange` |

### Linalg/HFusion IR Integration

Standard MLIR dialects such as Linalg/Tensor and HFusion are used to express operator semantics, which then directly enter the automatic fusion and scheduling flow at the Linalg/HFusion IR level.

**Example hfusion.mlir**:

```mlir
func.func @hfusion_reduce_mul(%arg0: tensor<40960xf32>, %arg1: tensor<40960x1024xf32>, %arg2: tensor<40960x1024xf32>, %arg3: tensor<40960x1024xf32>) -> tensor<40960xf32>
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %1 = tensor.empty() : tensor<40960x1024xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg1, %arg2 : tensor<40960x1024xf32>, tensor<40960x1024xf32>) outs(%arg3: tensor<40960x1024xf32>) -> tensor<40960x1024xf32>
  %4 = tensor.empty() : tensor<40960xf32>
  %sum = linalg.reduce {arith.addf} ins(%3 : tensor<40960x1024xf32>) 
                                    outs(%4 : tensor<40960xf32>) dimensions = [1]
  %5 = tensor.empty() : tensor<40960xf32>
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %sum : tensor<40960xf32>, tensor<40960xf32>) 
                                                                  outs(%5: tensor<40960xf32>) -> tensor<40960xf32>
  return %6 : tensor<40960xf32>
}
```

**Invocation method**:

- Command: `bishengir-compile -enable-hfusion-compile=true -enable-hivm-compile=true -target=Ascend910B1 hfusion.mlir -o hfusion_kernel.o`
- Expected output: an Ascend NPU operator binary file (in `.o` format), which can run on the device in conjunction with the CANN runtime.

**Automatic fusion**:

After Linalg/HFusion IR integration, the HFusion compilation flow performs automatic fusion and scheduling on operators that meet the fusion conditions: multiple operators are merged into the same kernel for execution, so that intermediate results are reused in on-chip memory and global memory reads and writes are reduced. Based on the fusion pattern and operator characteristics, the Tiling scheme and scheduling strategy are automatically selected to generate an efficient execution schedule for the Ascend NPU. The fused IR then goes through steps such as Tiling, loop generation, and Transform Dialect application, and is finally pushed down to HIVM to generate an executable binary.

The supported Op types include:

- `Elemwise`
- `Broadcast`
- `Reduce`
- `Transpose`
- `Concat`

For detailed information about the algorithm principles, constraint capabilities, and architecture design of automatic fusion, see [HFusion AutoSchedule Automatic Fusion and Scheduling](../features/hfusion_auto_schedule.md).

### HIVM IR Integration

For scenarios that require fine-grained control over hardware behavior, you can directly write kernels using the HIVM dialect to explicitly manage the memory hierarchy and computation pipeline.

**Example hivm.mlir**:

```mlir
func.func @hivm_vadd(%valueA: memref<16xf16, #hivm.address_space<gm>>,
                       %valueB: memref<16xf16, #hivm.address_space<gm>>,
                       %valueC: memref<16xf16, #hivm.address_space<gm>>)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %ubA = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%valueA : memref<16xf16, #hivm.address_space<gm>>)
                outs(%ubA : memref<16xf16, #hivm.address_space<ub>>)
  %ubB = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%valueB : memref<16xf16, #hivm.address_space<gm>>)
                outs(%ubB : memref<16xf16, #hivm.address_space<ub>>)
  %ubC = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
  hivm.hir.vadd ins(%ubA, %ubB : memref<16xf16, #hivm.address_space<ub>>,
                                 memref<16xf16, #hivm.address_space<ub>>)
                outs(%ubC : memref<16xf16, #hivm.address_space<ub>>)
  hivm.hir.store ins(%ubC : memref<16xf16, #hivm.address_space<ub>>)
                 outs(%valueC : memref<16xf16, #hivm.address_space<gm>>)
  return
}
```

At the HIVM layer, `#hivm.address_space` annotates the memory hierarchy: `gm` (Global Memory), `ub` (Unified Buffer), `l1` (L1 Buffer), and `l0a`/`l0b`/`l0c` (L0 Buffer). Explicit DMA transfers are performed through `hivm.hir.load`/`hivm.hir.store`, and on-chip computation is completed through instructions such as `hivm.hir.vadd`.

**Invocation method**: The HIVM layer does not require enabling the HFusion compilation flow. The default HIVM compilation flow performs optimizations such as synchronization insertion and memory planning.

- Command: `bishengir-compile -enable-hfusion-compile=false -enable-hivm-compile=true -target=Ascend910B1 hivm.mlir -o hivm_kernel.o`
- Expected output: An Ascend NPU operator binary file (`.o` format), which can run on the device in conjunction with the CANN runtime.

For the concepts of the IR layer, common compilation options, and other integration paths (such as Triton and TileLang), see [IR Integration Overview](interface_api.md).
