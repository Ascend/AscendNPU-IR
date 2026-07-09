# IR 接入简介

## 多级 IR 抽象架构

- 提供一系列高层抽象接口，屏蔽底层细节，将硬件无关表达映射到底层指令，提升算子开发易用性。
- 提供细粒度性能控制接口，能够精准控制片上内存地址、流水同步插入位置以及是否使能乒乓流水优化等。
- 基于多级接口支持自定义`DSL`/生态框架灵活对接，实现自定义算子在昇腾`NPU`上的高性能运行。

```text
  Torch-MLIR / Triton      （框架/DSL 层）
          |
          v
  Linalg / Tensor           （通用张量代数层）
          |
          v
  HFusion                   （硬件感知融合调度层）
          |
          v
  HIVM                      （NPU 指令层）
          |
          v
  LIR -> Binary             （机器码生成）
```

- `Linalg / Tensor`层：使用标准`MLIR dialect`表达算子语义，支持`Elemwise`、`Broadcast`、`Reduce`、`Transpose`、`Concat`等运算，`HFusion`自动完成算子融合、切分和调度。
- `HFusion`层：提供昇腾`NPU`感知的`Named Op`（如`hfusion.elemwise_unary`、`hfusion.cast`、`hfusion.select`、`hfusion.reduce_with_index`等），支持`tensor`语义，自动完成`bufferization`、`tiling`和调度。
- `HIVM`层：直接映射`NPU`硬件指令，显式控制存储层级（`GM`/`UB`/`L1`/`L0`）、计算流水线（`Vector`/`Cube`/`MTE`）和同步原语，支持精细粒度的性能调优。

上述多级接口支持自定义`DSL`和生态框架灵活对接。`Triton`和`PyTorch`等框架通过`IR`转换接入上述流程，实现自定义算子在昇腾`NPU`上的高性能运行。

## 编译选项与函数属性

### 编译选项

`bishengir-compile`提供以下公共编译选项：

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `-target` | `Ascend<Name>` | 目标设备，用于获取核数、片上内存大小等硬件规格，可通过查询`npu-smi info` |
| `-block-dim` | `1` | 指定使用的`block`数量，编译后`kernel`携带`hacc.block_dim`属性 |
| `-enable-hfusion-compile` | `false` | 使能`HFusion`编译流程（融合、调度、切分） |
| `-enable-hivm-compile` | `true` | 使能`HIVM`编译流程（转换到`HIVM`指令并优化） |
| `-enable-torch-compile` | `false` | 使能`Torch-MLIR`编译流程 |
| `-enable-triton-kernel-compile` | `false` | 使能`Triton kernel`编译流程 |

支持的目标设备包括`Atlas A2`/`A3`、`Ascend 950PR`/`Ascend 950DT`系列。

### 函数属性

以下属性用于标注`kernel`入口函数，各接入路径通用：

| 属性 | 说明 |
|------|------|
| `hacc.entry` | 标记当前函数为`kernel`入口 |
| `hacc.function_kind = #hacc.function_kind<DEVICE>` | 表示函数运行在`DEVICE`设备侧 |
| `hacc.function_kind = #hacc.function_kind<HOST>` | 表示函数运行在`HOST`侧，`HFusion`会自动`outline`出设备`kernel` |

示例：

```mlir
func.func @kernel(...) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  ...
}
```

## Triton 接入

`Triton`是主流的高性能算子开发编程语言。[Triton Ascend](https://gitcode.com/Ascend/triton-ascend/)将`Triton`算子转换为`MLIR`，接入`AscendNPU IR`生态，支持在昇腾`NPU`上运行`Triton`内核。详细的`Triton`接入指南（含安装、环境、算子映射及昇腾扩展）请参考[Triton 接入](triton_interface.md)。

## TileLang 接入

`TileLang`（`tilelang-ascend`）是面向昇腾`NPU`的领域特定语言，基于`tile-lang`的`Python`语法和[TVM](https://tvm.apache.org/)构建，支持`GEMM`、向量运算和注意力机制等算子，可将算子编译为`AscendNPU IR`（`HIVM`）在昇腾`NPU`上运行。详细的`TileLang`接入说明请参考[TileLang 接入](tile_lang_interface.md)。

## 框架接入

`AscendNPU IR`支持框架（`PyTorch`/`TensorFlow`/`MindSpore`）接入，有两种方式：

- DSL接入方式：如`Triton`、`TileLang`
- IR接入方式：如`Torch IR`、`Linalg`/`HFusion IR`、`HIVM IR`

详细的框架接入说明请参考[框架接入](framework_interface.md)。
