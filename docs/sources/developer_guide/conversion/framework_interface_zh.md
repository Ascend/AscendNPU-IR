# 框架接入

AscendNPU IR 支持框架（PyTorch/TensorFlow/MindSpore）接入，有两种方式：

- **DSL 接入方式**：通过 Triton、TileLang 等领域特定语言接入，将算子编译为 AscendNPU IR。
- **IR 接入方式**：通过 IR 表示接入，支持 Torch IR、Linalg/HFusion IR、HIVM IR 多层级接入，支持自动算子融合和切分，生成昇腾亲和的高性能算子。

## 1. DSL 接入方式

AscendNPU IR 向上支持与 Triton、TileLang 等语言或框架的对接，使能三方 DSL 支持昇腾硬件，在 NPU 上运行自定义算子。

| 接入方式 | 说明 | 文档 |
|----------|------|------|
| **Triton** | 使用 Triton 编写高性能内核，通过 Triton Ascend 在昇腾 NPU 上运行。含安装、环境、算子映射及昇腾扩展说明。 | [Triton 接入](triton_interface_zh.md) |
| **TileLang** | 使用 TileLang Ascend（基于 tile-lang/TVM 的 DSL）开发面向昇腾 NPU 的内核（如 GEMM、向量运算、attention）。含环境、构建与快速开始。 | [TileLang 接入](tile_lang_interface_zh.md) |

## 2. IR 接入方式

AscendNPU IR 支持多层级 IR 接入，不同层级在抽象程度和控制粒度上有所差异（详见 [IR 接入简介 - 多级 IR 抽象架构](interface_api_zh.md#多级ir抽象架构)）：

- **Torch IR**：框架层 ATen 算子，经 Pass 转换为 Linalg/HFusion。
- **Linalg/HFusion IR**：通用张量代数层与硬件感知融合层，标准 MLIR dialect 表达算子语义，HFusion 自动完成融合、切分和调度。
- **HIVM IR**：NPU 指令层，直接映射硬件指令，显式控制存储层级（GM/UB/L1/L0）和计算流水线（Vector/Cube/MTE），支持精细粒度调优。

### 2.1 Torch IR 接入

直接使用 Torch dialect 的 ATen 算子，通过 `convert-torch-to-hfusion` 等 Pass 自动转换为 Linalg/HFusion Named Op，再进入自动融合和调度流程。

#### Torch → AscendNPU IR 转换流程

Torch IR 通过 `torch-backend-to-named-op-backend-pipeline` 转换流水线接入 AscendNPU IR。BishengIR 自定义的 `convert-torch-to-hfusion` Pass 优先将 Torch ATen 算子转换为 Linalg/HFusion Named Op，未覆盖的算子回退到上游 torch-mlir 的标准 lowering 通路。主要转换阶段如下：

- `convert-torch-to-hfusion`：BishengIR 自定义转换，覆盖 55+ 个 ATen 算子到 Linalg/HFusion Named Op。
- `convert-torch-to-linalg`：上游 torch-mlir 转换，处理剩余算子。
- `convert-torch-to-scf / arith / tensor`：上游 torch-mlir 完成控制流、算术、tensor 等转换。
- `func-backend-type-conversion`：将 Torch 类型（`!torch.vtensor`）转换为标准 builtin 类型（`tensor`）。

#### 用例

```
func.func @torch.aten.mul_tensor(%arg0: !torch.vtensor<[4096],f16>, %arg1: !torch.vtensor<[1,56,4096],f16>) -> !torch.vtensor<[1,56,4096],f16>
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[4096],f16>, !torch.vtensor<[1,56,4096],f16> -> !torch.vtensor<[1,56,4096],f16>
  return %0 : !torch.vtensor<[1,56,4096],f16>
}
```

#### 调用方式

有两种调用方式，二者共享同一套编译 pipeline：

- **分步转换**：先用 `bishengir-opt -torch-backend-to-named-op-backend-pipeline` 将 Torch IR 转换为 Linalg/HFusion IR，再按 [Linalg/HFusion IR 接入](#22-linalghfusion-ir-接入) 流程编译；适用于需要缓存中间 IR 的场景。
- **端到端编译**：使用 `bishengir-compile` 直接编译 Torch IR 到二进制。

```
# 分步转换，预期生成 Linalg/HFusion IR
bishengir-opt -torch-backend-to-named-op-backend-pipeline test.mlir -o hfusion.mlir
# 预期结果
func.func @torch.aten.mul_tensor(%arg0: tensor<4096xf16>, %arg1: tensor<1x56x4096xf16>) -> tensor<1x56x4096xf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %0 = tensor.empty() : tensor<1x56x4096xf16>
  %broadcasted = linalg.broadcast ins(%arg0 : tensor<4096xf16>) outs(%0 : tensor<1x56x4096xf16>) dimensions = [0, 1] 
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %arg1 : tensor<1x56x4096xf16>, tensor<1x56x4096xf16>) outs(%0 : tensor<1x56x4096xf16>) -> tensor<1x56x4096xf16>
  return %1 : tensor<1x56x4096xf16>
}

# 端到端编译，经过 Torch/HFusion/HIVM IR 编译 pipeline，直接生成二进制
bishengir-compile -enable-torch-compile=true -enable-hfusion-compile=true -enable-hivm-compile=true -target=Ascend910B1 test.mlir
```

#### 支持的 Torch 算子

##### Elementwise Binary

| Torch Op | 转换目标 |
|----------|----------|
| `aten.add.Tensor` / `aten.add.Scalar` | `linalg.binary_fn<add>` |
| `aten.sub.Tensor` / `aten.sub.Scalar` | `linalg.binary_fn<sub>` |
| `aten.mul.Tensor` / `aten.mul.Scalar` | `linalg.binary_fn<mul>` |
| `aten.div.Tensor` / `aten.div.Scalar` | `linalg.binary_fn<div>` |
| `aten.maximum` | `linalg.binary_fn<max_signed>` |
| `aten.minimum` | `linalg.binary_fn<min_signed>` |
| `aten.clamp_min` / `aten.clamp_min.Tensor` | `linalg.binary_fn<max_signed>` |
| `aten.clamp_max` / `aten.clamp_max.Tensor` | `linalg.binary_fn<min_signed>` |
| `aten.clamp` | `max_signed` + `min_signed` 组合 |
| `aten.pow.Tensor_Tensor` / `aten.pow.Tensor_Scalar` / `aten.pow.Scalar` | `hfusion.binary_fn<powf>` |
| `aten.logical_and` | `hfusion.binary_fn<vand>` |
| `aten.logical_or` | `hfusion.binary_fn<vor>` |

##### Elementwise Unary

| Torch Op | 转换目标 |
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
| `aten.sigmoid` | 分解为 negf -> exp -> add -> div |
| `aten.gelu` | 分解为 tanh 近似实现 |

##### Compare

| Torch Op | 转换目标 |
|----------|----------|
| `aten.gt.Scalar` / `aten.gt.Tensor` | `hfusion.compare_fn<vgt>` |
| `aten.lt.Scalar` / `aten.lt.Tensor` | `hfusion.compare_fn<vlt>` |
| `aten.ge.Scalar` / `aten.ge.Tensor` | `hfusion.compare_fn<vge>` |
| `aten.le.Scalar` / `aten.le.Tensor` | `hfusion.compare_fn<vle>` |
| `aten.eq.Scalar` / `aten.eq.Tensor` | `hfusion.compare_fn<veq>` |
| `aten.ne.Scalar` / `aten.ne.Tensor` | `hfusion.compare_fn<vne>` |

##### Reduction

| Torch Op | 转换目标 |
|----------|----------|
| `aten.sum` / `aten.sum.dim_IntList` | `linalg.reduce` + `arith.addf/addi` |
| `aten.prod` / `aten.prod.dim_int` | `linalg.reduce` + `arith.mulf/muli` |
| `aten.max` | `linalg.reduce` + `arith.maximumf/maxsi` |
| `aten.min` | `linalg.reduce` + `arith.minimumf/minsi` |
| `aten.max.dim` | `hfusion.reduce_with_index` (MAX) |
| `aten.min.dim` | `hfusion.reduce_with_index` (MIN) |
| `aten.any` / `aten.any.dim` / `aten.any.dims` | `linalg.reduce` + `arith.ori` |
| `aten.all` / `aten.all.dim` | `linalg.reduce` + `arith.andi` |

##### Data Movement

| Torch Op | 转换目标 |
|----------|----------|
| `aten.permute` | `linalg.transpose` |
| `aten.broadcast_to` | `linalg.broadcast` |

##### 其他

| Torch Op | 转换目标 |
|----------|----------|
| `aten.to.dtype` | `hfusion.cast` |
| `aten.where.self` | `hfusion.select` |
| `aten.arange.start_step` | `hfusion.arange` |

### 2.2 Linalg/HFusion IR 接入

使用 Linalg/Tensor、HFusion 等标准 MLIR dialect 表达算子语义，直接进入 Linalg/HFusion IR 层级的自动融合和调度流程。

#### 用例

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

#### 自动融合支持的 Op 范围

- Elemwise
- Broadcast
- Reduce
- Transpose
- Concat

#### 调用方式

```
# 端到端编译，经过 HFusion/HIVM IR 编译 pipeline，直接生成二进制
bishengir-compile -enable-hfusion-compile=true -enable-hivm-compile=true -target=Ascend910B1 test.mlir
```

### 2.3 HIVM IR 接入

对于需要精细控制硬件行为的场景，可以直接使用 HIVM dialect 编写 kernel，显式管理存储层级和计算流水线。

#### 用例

```
module {
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
}
```

HIVM 层使用 `#hivm.address_space` 标注存储层级：`gm`（全局内存）、`ub`（Unified Buffer）、`l1`（L1 Buffer）、`l0a`/`l0b`/`l0c`（L0 Buffer）。通过 `hivm.hir.load`/`hivm.hir.store` 进行显式 DMA 搬运，通过 `hivm.hir.vadd` 等指令在片上完成计算。

#### 调用方式

HIVM 层无需使能 HFusion 编译流程，默认的 HIVM 编译流程会完成同步插入、内存规划等优化：

```
# 端到端编译，经过 HIVM IR 编译 pipeline，直接生成二进制
bishengir-compile -enable-hfusion-compile=false -enable-hivm-compile=true -target=Ascend910B1 test.mlir
```

关于 IR 层概念、公共编译选项及其他接入路径（如 Triton、TileLang），请参阅 [IR 接入简介](interface_api_zh.md)。
