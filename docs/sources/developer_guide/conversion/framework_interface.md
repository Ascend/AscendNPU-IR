# Framework Integration

AscendNPU IR supports framework integration (PyTorch/TensorFlow/MindSpore) in two ways:

1. **DSL integration**: Integrate via domain-specific languages such as Triton and TileLang, which compile to AscendNPU IR.
2. **IR integration**: Integrate via IR representation, including Torch IR and Linalg IR, with automatic fusion and tiling for Ascend-friendly kernels.

## 1. DSL integration

AscendNPU IR supports upstream integration with languages and frameworks such as Triton and TileLang, so that third-party DSLs can target Ascend hardware and run custom operators on the NPU.

| Integration | Description | Document |
|-------------|-------------|----------|
| **Triton** | Use Triton to write high-performance kernels and run them on Ascend NPU via Triton Ascend. Covers installation, environment, op mapping, and Ascend extensions. | [Triton interface](triton_interface.md) |
| **TileLang** | Use TileLang Ascend (tile-lang/TVM-based DSL) to develop kernels for Ascend NPU (e.g., GEMM, vector ops, attention). Covers environment, build, and quick start. | [TileLang interface](tile_lang_interface.md) |

## 2. IR integration

### 2.1 Torch IR integration

Use Torch dialect ATen ops; Passes such as `convert-torch-to-hfusion` lower to Linalg/HFusion named ops, then fusion and scheduling.

#### Torch → AscendNPU IR pipeline

Torch IR is integrated via the `torch-backend-to-named-op-backend-pipeline`. The custom `convert-torch-to-hfusion` Pass lowers Torch ATen ops to Linalg/HFusion named ops first; uncovered ops fall back to upstream torch-mlir. Stages:

1. `convert-torch-to-hfusion`: BishengIR custom lowering for 55+ ATen ops to Linalg/HFusion named ops.
2. `convert-torch-to-linalg`: Upstream torch-mlir for remaining ops.
3. `convert-torch-to-scf / arith / tensor`: Upstream control flow, arithmetic, tensor.
4. `func-backend-type-conversion`: Torch types (`!torch.vtensor`) to builtin types (`tensor`).

#### Example

```
func.func @torch.aten.mul_tensor(
    %arg0: !torch.vtensor<[4096],f16>,
    %arg1: !torch.vtensor<[1,56,4096],f16>
) -> !torch.vtensor<[1,56,4096],f16>
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %0 = torch.aten.mul.Tensor %arg0, %arg1
      : !torch.vtensor<[4096],f16>, !torch.vtensor<[1,56,4096],f16>
      -> !torch.vtensor<[1,56,4096],f16>
  return %0 : !torch.vtensor<[1,56,4096],f16>
}
```

#### Invocation

```
bishengir-compile -enable-hfusion-compile=true -enable-torch-compile=true -block-dim=20 -target=Ascend910B1 test.mlir
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

### 2.2 Linalg IR integration

Use Linalg/Tensor and other standard MLIR dialects; input goes directly into fusion and scheduling.

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

#### Ops supported for automatic fusion

- Elemwise
- Broadcast
- Reduce
- Transpose
- Concat

#### Invocation

```
bishengir-compile -enable-hfusion-compile=true -block-dim=20 -target=Ascend910B1 test.mlir
```

For IR-level concepts, common compile options, and other paths (e.g., Triton, TileLang, HIVM), see [Interface API](interface_api.md).
