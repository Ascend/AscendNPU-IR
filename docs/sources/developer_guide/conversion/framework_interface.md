# Framework Integration

AscendNPU IR supports framework integration (PyTorch/TensorFlow/MindSpore) in two ways:

- **DSL integration**: Integrate via domain-specific languages such as Triton and TileLang, which compile to AscendNPU IR.
- **IR integration**: Integrate via IR representation, supporting multi-level IR (Torch IR, Linalg/HFusion IR, HIVM IR), with automatic fusion and tiling for Ascend-friendly kernels.

## DSL integration

AscendNPU IR supports upstream integration with languages and frameworks such as Triton and TileLang, so that third-party DSLs can target Ascend hardware and run custom operators on the NPU.

| Integration | Description | Document |
|-------------|-------------|----------|
| **Triton** | Use Triton to write high-performance kernels and run them on Ascend NPU via Triton Ascend. Covers installation, environment, op mapping, and Ascend extensions. | [Triton interface](triton_interface.md) |
| **TileLang** | Use TileLang Ascend (tile-lang/TVM-based DSL) to develop kernels for Ascend NPU (e.g., GEMM, vector ops, attention). Covers environment, build, and quick start. | [TileLang interface](tile_lang_interface.md) |

## IR integration

AscendNPU IR supports multi-level IR integration; each level differs in abstraction and control granularity (see [Interface API - Multi-level IR Abstraction](interface_api.md#multi-level-ir-abstraction)):

- **Torch IR**: Framework-level ATen ops, lowered to Linalg/HFusion via Passes.
- **Linalg/HFusion IR**: General tensor algebra and hardware-aware fusion layer; standard MLIR dialects for operator semantics, HFusion performs fusion, tiling, and scheduling automatically.
- **HIVM IR**: NPU instruction layer; direct mapping to hardware instructions, explicit control of memory hierarchy (GM/UB/L1/L0) and compute pipelines (Vector/Cube/MTE) for fine-grained tuning.

### Torch IR integration

Use Torch dialect ATen ops; Passes such as `convert-torch-to-hfusion` lower to Linalg/HFusion named ops, then enter the fusion and scheduling flow.

#### Torch → AscendNPU IR pipeline

Torch IR is integrated via the `torch-backend-to-named-op-backend-pipeline` conversion pipeline. The custom `convert-torch-to-hfusion` Pass lowers Torch ATen ops to Linalg/HFusion named ops first; uncovered ops fall back to the standard lowering path of upstream torch-mlir. Main conversion stages:

- `convert-torch-to-hfusion`: BishengIR custom lowering for 55+ ATen ops to Linalg/HFusion named ops.
- `convert-torch-to-linalg`: Upstream torch-mlir for remaining ops.
- `convert-torch-to-scf / arith / tensor`: Upstream torch-mlir for control flow, arithmetic, and tensor conversion.
- `func-backend-type-conversion`: Converts Torch types (`!torch.vtensor`) to builtin types (`tensor`).

#### Example

```
func.func @torch.aten.mul_tensor(%arg0: !torch.vtensor<[4096],f16>, %arg1: !torch.vtensor<[1,56,4096],f16>) -> !torch.vtensor<[1,56,4096],f16>
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[4096],f16>, !torch.vtensor<[1,56,4096],f16> -> !torch.vtensor<[1,56,4096],f16>
  return %0 : !torch.vtensor<[1,56,4096],f16>
}
```

#### Invocation

There are two invocation methods; both share the same compile pipeline:

- **Stepwise conversion**: Use `bishengir-opt -torch-backend-to-named-op-backend-pipeline` to convert Torch IR to Linalg/HFusion IR first, then compile via the [Linalg/HFusion IR integration](#22-linalghfusion-ir-integration) flow; suitable for caching intermediate IR.
- **End-to-end compilation**: Use `bishengir-compile` to compile Torch IR to binary directly.

```
# Stepwise conversion; expected output is Linalg/HFusion IR
bishengir-opt -torch-backend-to-named-op-backend-pipeline test.mlir -o hfusion.mlir
# Expected result
func.func @torch.aten.mul_tensor(%arg0: tensor<4096xf16>, %arg1: tensor<1x56x4096xf16>) -> tensor<1x56x4096xf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %0 = tensor.empty() : tensor<1x56x4096xf16>
  %broadcasted = linalg.broadcast ins(%arg0 : tensor<4096xf16>) outs(%0 : tensor<1x56x4096xf16>) dimensions = [0, 1] 
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %arg1 : tensor<1x56x4096xf16>, tensor<1x56x4096xf16>) outs(%0 : tensor<1x56x4096xf16>) -> tensor<1x56x4096xf16>
  return %1 : tensor<1x56x4096xf16>
}

# End-to-end compilation; runs through Torch/HFusion/HIVM IR pipeline and produces binary directly
bishengir-compile -enable-torch-compile=true -enable-hfusion-compile=true -enable-hivm-compile=true -target=Ascend910B1 test.mlir
```

#### Supported Torch ops

##### Elementwise binary

| Torch Op | Lowering target |
|----------|-----------------|
| `aten.add.Tensor` / `aten.add.Scalar` | `linalg.binary_fn<add>` |
| `aten.sub.Tensor` / `aten.sub.Scalar` | `linalg.binary_fn<sub>` |
| `aten.mul.Tensor` / `aten.mul.Scalar` | `linalg.binary_fn<mul>` |
| `aten.div.Tensor` / `aten.div.Scalar` | `linalg.binary_fn<div>` |
| `aten.maximum` | `linalg.binary_fn<max_signed>` |
| `aten.minimum` | `linalg.binary_fn<min_signed>` |
| `aten.clamp_min` / `aten.clamp_min.Tensor` | `linalg.binary_fn<max_signed>` |
| `aten.clamp_max` / `aten.clamp_max.Tensor` | `linalg.binary_fn<min_signed>` |
| `aten.clamp` | Combination of `max_signed` + `min_signed` |
| `aten.pow.Tensor_Tensor` / `aten.pow.Tensor_Scalar` / `aten.pow.Scalar` | `hfusion.binary_fn<powf>` |
| `aten.logical_and` | `hfusion.binary_fn<vand>` |
| `aten.logical_or` | `hfusion.binary_fn<vor>` |

##### Elementwise unary

| Torch Op | Lowering target |
|----------|-----------------|
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
| `aten.sigmoid` | Decomposed to negf -> exp -> add -> div |
| `aten.gelu` | Decomposed to tanh-based approximation |

##### Compare

| Torch Op | Lowering target |
|----------|-----------------|
| `aten.gt.Scalar` / `aten.gt.Tensor` | `hfusion.compare_fn<vgt>` |
| `aten.lt.Scalar` / `aten.lt.Tensor` | `hfusion.compare_fn<vlt>` |
| `aten.ge.Scalar` / `aten.ge.Tensor` | `hfusion.compare_fn<vge>` |
| `aten.le.Scalar` / `aten.le.Tensor` | `hfusion.compare_fn<vle>` |
| `aten.eq.Scalar` / `aten.eq.Tensor` | `hfusion.compare_fn<veq>` |
| `aten.ne.Scalar` / `aten.ne.Tensor` | `hfusion.compare_fn<vne>` |

##### Reduction

| Torch Op | Lowering target |
|----------|-----------------|
| `aten.sum` / `aten.sum.dim_IntList` | `linalg.reduce` + `arith.addf/addi` |
| `aten.prod` / `aten.prod.dim_int` | `linalg.reduce` + `arith.mulf/muli` |
| `aten.max` | `linalg.reduce` + `arith.maximumf/maxsi` |
| `aten.min` | `linalg.reduce` + `arith.minimumf/minsi` |
| `aten.max.dim` | `hfusion.reduce_with_index` (MAX) |
| `aten.min.dim` | `hfusion.reduce_with_index` (MIN) |
| `aten.any` / `aten.any.dim` / `aten.any.dims` | `linalg.reduce` + `arith.ori` |
| `aten.all` / `aten.all.dim` | `linalg.reduce` + `arith.andi` |

##### Data movement

| Torch Op | Lowering target |
|----------|-----------------|
| `aten.permute` | `linalg.transpose` |
| `aten.broadcast_to` | `linalg.broadcast` |

##### Other

| Torch Op | Lowering target |
|----------|-----------------|
| `aten.to.dtype` | `hfusion.cast` |
| `aten.where.self` | `hfusion.select` |
| `aten.arange.start_step` | `hfusion.arange` |

### Linalg/HFusion IR integration

Use Linalg/Tensor, HFusion, and other standard MLIR dialects for operator semantics; input goes directly into the Linalg/HFusion IR layer's fusion and scheduling flow.

#### Example

```
func.func @test_reduce(%arg0: tensor<40960xf32>, %arg1: tensor<40960x1024xf32>, %arg2: tensor<40960x1024xf32>, %arg3: tensor<40960x1024xf32>) -> tensor<40960xf32>
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

#### Invocation

```
# End-to-end compilation; runs through HFusion/HIVM IR pipeline and produces binary directly
bishengir-compile -enable-hfusion-compile=true -enable-hivm-compile=true -target=Ascend910B1 test.mlir
```

#### Automatic fusion

Once Linalg/HFusion IR is ingested, the HFusion compile flow performs **automatic fusion and scheduling** on eligible ops: multiple ops are merged into the same kernel so intermediate results are reused in on-chip memory and global memory traffic is reduced; scheduling and Tiling strategies are selected automatically based on fusion patterns and operator traits, producing efficient schedules for Ascend NPU. After fusion, the IR passes through Tiling, loop generation, Transform Dialect application, and similar steps before being lowered to HIVM and emitting an executable binary.

Supported op types:

- Elemwise
- Broadcast
- Reduce
- Transpose
- Concat

For algorithm details, constraints, architecture, and related topics, see [HFusion AutoSchedule: Automatic Fusion and Scheduling](../features/AutoSchedule/HFusion_AutoSchedule.md).

### HIVM IR integration

For fine-grained hardware control, you can write kernels directly in the HIVM dialect, managing memory hierarchy and compute pipelines explicitly.

#### Example

```
func.func @vadd_kernel(%valueA: memref<16xf16, #hivm.address_space<gm>>,
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

HIVM uses `#hivm.address_space` to annotate memory hierarchy: `gm` (global memory), `ub` (Unified Buffer), `l1` (L1 Buffer), `l0a`/`l0b`/`l0c` (L0 Buffer). Use `hivm.hir.load`/`hivm.hir.store` for explicit DMA transfers and `hivm.hir.vadd` and similar ops for on-chip compute.

#### Invocation

HIVM does not require the HFusion compile pipeline. The default HIVM compile pipeline performs sync insertion, memory planning, and other optimizations:

```
# End-to-end compilation; runs through HIVM IR pipeline and produces binary directly
bishengir-compile -enable-hfusion-compile=false -enable-hivm-compile=true -target=Ascend910B1 test.mlir
```

For IR-level concepts, common compile options, and other integration paths (e.g., Triton, TileLang), see [Interface API](interface_api.md).
