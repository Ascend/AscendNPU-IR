# IR接入简介

## 多级IR抽象架构

- 提供一系列高层抽象接口，屏蔽底层细节，将硬件无关表达映射到底层指令，提升算子开发易用性
- 提供细粒度性能控制接口，能够精准控制片上内存地址、流水同步插入位置以及是否使能乒乓流水优化等
- 基于多级接口支持自定义DSL/生态框架灵活对接，实现自定义算子在昇腾 NPU 上的高性能运行。

```
  Torch-MLIR / Triton       (框架/DSL层)
         |
         v
  Linalg / Tensor            (通用张量代数层)
         |
         v
  HFusion                    (硬件感知融合调度层)
         |
         v
  HIVM                       (NPU指令层)
         |
         v
  LIR -> Binary              (机器码生成)
```

- **Linalg / Tensor层**：使用标准MLIR dialect表达算子语义，支持Elemwise、Broadcast、Reduce、Transpose、Concat等运算，HFusion自动完成算子融合、切分和调度
- **HFusion层**：提供昇腾NPU感知的Named Op（如`hfusion.elemwise_unary`、`hfusion.cast`、`hfusion.select`、`hfusion.reduce_with_index`等），支持tensor语义，自动完成bufferization、tiling和调度
- **HIVM层**：直接映射NPU硬件指令，显式控制存储层级（GM/UB/L1/L0）、计算流水线（Vector/Cube/MTE）和同步原语，支持精细粒度的性能调优

上述多级接口支持自定义DSL和生态框架灵活对接。Triton和PyTorch等框架通过IR转换接入上述流程，实现自定义算子在昇腾NPU上的高性能运行。

## 公共编译选项与函数属性

### 函数属性

以下属性用于标注kernel入口函数，各接入路径通用：

| 属性 | 说明 |
|------|------|
| `hacc.entry` | 标记当前函数为kernel入口 |
| `hacc.function_kind = #hacc.function_kind<DEVICE>` | 表示函数运行在DEVICE设备侧 |
| `hacc.function_kind = #hacc.function_kind<HOST>` | 表示函数运行在HOST侧，HFusion会自动outline出设备kernel |

示例：

```
func.func @kernel(...) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  ...
}
```

### 编译选项

`bishengir-compile`提供以下公共编译选项：

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `-target` | `Ascend910B1` | 目标设备，用于获取核数、片上内存大小等硬件规格 |
| `-block-dim` | `1` | 指定使用的block数量，编译后kernel携带`hacc.block_dim`属性 |
| `-enable-hfusion-compile` | `false` | 使能HFusion编译流程（融合、调度、切分） |
| `-enable-hivm-compile` | `true` | 使能HIVM编译流程（转换到HIVM指令并优化） |
| `-enable-torch-compile` | `false` | 使能Torch-MLIR编译流程 |
| `-enable-triton-kernel-compile` | `false` | 使能Triton kernel编译流程 |

支持的目标设备：

- Ascend910B系列：`Ascend910B1`、`Ascend910B2`、`Ascend910B3`、`Ascend910B4`等
- Ascend910_93系列：`Ascend910_9362`、`Ascend910_9372`、`Ascend910_9381`等
- Ascend910_95系列：`Ascend910_950z`、`Ascend910_9579`、`Ascend910_9589`等

## Triton 接入

Triton是目前最主流的高性能算子开发编程语言，可以通过[Triton Ascend](https://gitcode.com/Ascend/triton-ascend/)将Triton算子转换为MLIR，从而接入AscendNPU IR生态。详细的Triton接入指南请参考[Triton接入](triton_interface.md)。

### Triton算子示例

```python
@triton.jit
def add_kernel(x_ptr,  # 指向第一个输入向量的指针
               y_ptr,  # 指向第二个输入向量的指针
               output_ptr,  # 指向输出向量的指针
               n_elements,  # 向量的大小
               BLOCK_SIZE: tl.constexpr,  # 每个程序应处理的元素数量
               ):
    pid = tl.program_id(axis=0)  # 使用一维启动网格，因此 axis 为 0
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

### Triton Ascend生成的MLIR

```
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @add_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1024 = arith.constant 1024 : index
    %c1024_i32 = arith.constant 1024 : i32
    %0 = arith.muli %arg9, %c1024_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<1024xf32>
    %2 = arith.addi %1, %c1024 : index
    %3 = arith.index_cast %arg5 : i32 to index
    %4 = arith.maxsi %1, %3 : index
    %5 = arith.minsi %2, %4 : index
    %6 = arith.subi %5, %1 : index
    %7 = arith.cmpi slt, %6, %c1024 : index
    scf.if %7 {
      linalg.fill ins(%cst : f32) outs(%alloc : memref<1024xf32>)
    }
    %subview = memref.subview %reinterpret_cast[0] [%6] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_0 = memref.subview %alloc[0] [%6] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_0 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %8 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%1], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %alloc_2 = memref.alloc() : memref<1024xf32>
    scf.if %7 {
      linalg.fill ins(%cst : f32) outs(%alloc_2 : memref<1024xf32>)
    }
    %subview_3 = memref.subview %reinterpret_cast_1[0] [%6] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_4 = memref.subview %alloc_2[0] [%6] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview_3, %subview_4 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %9 = bufferization.to_tensor %alloc_2 restrict writable : memref<1024xf32>
    %10 = arith.addf %8, %9 : tensor<1024xf32>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%1], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %extracted_slice = tensor.extract_slice %10[0] [%6] [1] : tensor<1024xf32> to tensor<?xf32>
    %subview_6 = memref.subview %reinterpret_cast_5[0] [%6] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}
```

### 调用方式

```
bishengir-compile -enable-hfusion-compile=true -enable-triton-kernel-compile=true ttadapter.mlir
```

## TileLang 接入

TileLang（tilelang-ascend）是面向昇腾NPU的领域特定语言，基于 tile-lang 的 Python 语法和 [TVM](https://tvm.apache.org/) 构建，支持 GEMM、向量运算和注意力机制等算子，可将算子编译为 AscendNPU IR（HIVM）在昇腾 NPU 上运行。详细的 TileLang 接入说明请参考 [TileLang 接入](tile_lang_interface_zh.md)。

## 框架接入

AscendNPU IR支持框架（PyTorch/TensorFlow/MindSpore）直接通过IR接入，支持自动算子融合和切分，生成昇腾亲和的高性能算子。框架可以通过以下两种方式接入：

- **Torch IR接入**：直接使用Torch dialect的ATen算子，通过`convert-torch-to-hfusion`等Pass自动转换为Linalg/HFusion Named Op，再进入自动融合和调度流程
- **Linalg IR接入**：使用Linalg/Tensor等标准MLIR dialect表达算子语义，直接进入自动融合和调度流程

### Torch IR 接入

#### Torch2AscendNPU IR 转换流程

Torch IR通过`torch-backend-to-named-op-backend-pipeline`转换流水线接入AscendNPU IR。BishengIR自定义的`convert-torch-to-hfusion` Pass优先将Torch ATen算子转换为Linalg/HFusion Named Op，未覆盖的算子回退到上游torch-mlir的标准lowering通路。主要转换阶段如下：

1. `convert-torch-to-hfusion`：BishengIR自定义转换，覆盖55+个ATen算子到Linalg/HFusion Named Op
2. `convert-torch-to-linalg`：上游torch-mlir转换，处理剩余算子
3. `convert-torch-to-scf / arith / tensor`：上游torch-mlir完成控制流、算术、tensor等转换
4. `func-backend-type-conversion`：将Torch类型（`!torch.vtensor`）转换为标准builtin类型（`tensor`）

#### 用例

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

#### 调用方式

```
bishengir-compile -enable-hfusion-compile=true -enable-torch-compile=true -block-dim=20 -target=Ascend910B1 test.mlir
```

#### 支持的Torch算子

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

### Linalg IR 接入

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

#### 自动融合支持的Op范围

- Elemwise
- Broadcast
- Reduce
- Transpose
- Concat

#### 调用方式

```
bishengir-compile -enable-hfusion-compile=true -block-dim=20 -target=Ascend910B1 test.mlir
```

## HIVM 指令级接入

对于需要精细控制硬件行为的场景，可以直接使用HIVM dialect编写kernel，显式管理存储层级和计算流水线。

### 用例

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

HIVM层使用`#hivm.address_space`标注存储层级：`gm`（全局内存）、`ub`（Unified Buffer）、`l1`（L1 Buffer）、`l0a`/`l0b`/`l0c`（L0 Buffer）。通过`hivm.hir.load`/`hivm.hir.store`进行显式DMA搬运，通过`hivm.hir.vadd`等指令在片上完成计算。

### 调用方式

HIVM层无需使能HFusion编译流程，默认的HIVM编译流程会完成同步插入、内存规划等优化：

```
bishengir-compile -target=Ascend910B1 test.mlir
```

---
