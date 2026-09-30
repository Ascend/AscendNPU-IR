# hfusion方言

Hybrid Fusion（HFusion）方言。

## 操作定义

### hfusion.arange (hfusion::ArangeOp)

**功能**：生成序列，与标准arange存在差异，支持偏移量（默认值0）与多维场景并配套多维步长。偏移量、步长定义规则与内存描述符保持一致。

三维arange取值计算公式：

`arange[i, j, k] = offset + stride[0] * i + stride[1] * j + stride[2] * k`

**特性**：`AttrSizedOperandSegments`、`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`

**接口**：`DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `offset` | index类型偏移值 |
| `strides` | 变长index类型步长数组 |
| `init` | 任意类型带形状初始化张量 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `result_tensor` | 任意类型带形状输出张量 |

### hfusion.assert (hfusion::AssertOp)

**功能**：设备端调试断言。接收字符串提示信息与标量或张量判断条件。

**语法**：

```mlir
operation ::= `hfusion.assert` $msg attr-dict $cond `:` type($cond)
```

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `msg` | `::mlir::StringAttr` | 字符串提示文本 |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `cond` | 整数或任意类型定维张量 |

### hfusion.atomic_cas (hfusion::AtomicCasOp)

**功能**：原子比较交换（CAS）操作。包含内存地址、预期旧值、新值三个输入；仅当内存值等于预期旧值时，将内存更新为新值，无论是否更新均返回内存原始值。

**约束**：输入、输出秩与元素类型必须完全一致。

**参数说明**：

- `src0`：预期旧值
- `src1`：待写入新值
- `dst`：全局内存目标地址

**语法**：

```mlir
operation ::= `hfusion.atomic_cas` attr-dict `ins` `(` $input `:` type($input) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`->` type($output)^)?
```

**示例**：

```mlir
hfusion.atomic_cas ins(%src0, %src1 : memref<?xf32>, memref<?xf32>) outs(%dst : memref<?xf32>)
%result = hfusion.atomic_cas ins(%src0, %src1 : tensor<?xf32>, tensor<?xf32>) outs(%dst : tensor<?xf32>) -> tensor<?xf32>
```

**特性**：`SameOperandsAndResultRank`

**接口**：`MemoryEffectOpInterface`

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 变长张量或内存视图输入 |
| `dst` | 目标张量或内存视图 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 输出张量或内存视图 |

### hfusion.atomic_rmw (hfusion::AtomicRMWOp)

**功能**：原子读改写操作。流程：读取内存原值 → 根据原子类型执行计算 → 写入新值并返回原值，全程不可中断。

**约束**：输入、输出内存视图秩与元素类型必须完全一致。

**参数说明**：

- `src`：计算新值
- `dst`：全局内存目标地址

**语法**：

```mlir
operation ::= `hfusion.atomic_rmw` attr-dict `ins` `(` $input `:` type($input) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `atomic_kind` `=` $atomic_kind
              (`->` type($output)^)?
```

**示例**：

```mlir
hfusion.atomic_rmw ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) atomic_kind = <add>
%result = hfusion.atomic_rmw ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) atomic_kind = <or> -> tensor<?xf32>
```

**特性**：`SameOperandsAndResultRank`

**接口**：`MemoryEffectOpInterface`

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `atomic_kind` | `::mlir::hfusion::AtomicKindAttr` | 原子操作类型，可选值：none、add、max、min、and、or、xor、cas、xchg、umax、umin |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 输入张量或内存视图 |
| `dst` | 目标张量或内存视图 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 输出张量或内存视图 |

### hfusion.atomic_xchg (hfusion::AtomicXchgOp)

**功能**：原子交换操作。流程：读取内存原值 → 写入新值 → 返回原值，全程不可中断。

**约束**：输入、输出内存视图秩与元素类型必须完全一致。

**参数说明**：

- `src`：待写入新值
- `dst`：全局内存目标地址
- `mask`：可选掩码元素

**语法**：

```mlir
operation ::= `hfusion.atomic_xchg` attr-dict `ins` `(` $input `:` type($input) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`mask` `(` $mask^ `:` type($mask) `)`)?
              (`->` type($output)^)?
```

**示例**：

```mlir
hfusion.atomic_xchg ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) mask(%m : memref<?xi1>)
%result = hfusion.atomic_xchg ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) mask(%m : memref<?xi1>) -> tensor<?xf32>
```

**特性**：`SameOperandsAndResultRank`

**接口**：`MemoryEffectOpInterface`

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 输入张量或内存视图 |
| `dst` | 目标张量或内存视图 |
| `mask` | 可选掩码张量或内存视图 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 输出张量或内存视图 |

### hfusion.barrier (hfusion::BarrierOp)

**功能**：同步单个计算核内所有流水线。

**语法**：

```mlir
operation ::= `hfusion.barrier` attr-dict
```

### hfusion.bitcast (hfusion::BitcastOp)

**功能**：逐元素执行比特类型转换。

**特性**：`AttrSizedOperandSegments`、`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`

**接口**：`DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `inputs` | 变长任意类型输入 |
| `outputs` | 变长任意带形状输出张量 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `result_tensors` | 变长任意类型定维输出张量 |

### hfusion.cast (hfusion::CastOp)

**功能**：逐元素执行数值类型转换。

**特性**：`AttrSizedOperandSegments`、`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`

**接口**：`DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `round_mode` | `::mlir::hfusion::RoundModeAttr` | 舍入模式：RINT、ROUND、FLOOR、CEIL、TRUNC、ODD |
| `enable_overflow` | `::mlir::BoolAttr` | 是否开启溢出检测 |
| `cast` | `::mlir::hfusion::TypeFnAttr` | 转换类型：cast_signed、cast_unsigned、bitcast |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `inputs` | 变长任意类型输入 |
| `outputs` | 变长任意带形状输出张量 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `result_tensors` | 变长任意类型定维输出张量 |

### hfusion.compare (hfusion::CompareOp)

**功能**：逐元素执行比较运算，不会对输入做数值类型提升转换。

**特性**：`AttrSizedOperandSegments`、`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`

**接口**：`DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `compare_fn` | `::mlir::hfusion::CompareFnAttr` | 比较函数：veq、vne、vle、vlt、vge、vgt、vule、vult、vuge、vugt |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `inputs` | 变长任意类型输入 |
| `outputs` | 变长任意带形状输出张量 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `result_tensors` | 变长任意类型定维输出张量 |

### hfusion.conv1d (hfusion::Conv1DOp)

**功能**：一维卷积运算，语义对齐`triton.language.extra.cann.extension.conv1d(input, weight, bias, groups, padding_size)`，默认参数对齐`torch.nn.functional.conv1d(stride=1, dilation=1, padding_mode='zeros')`。

**维度约束**：

- input：[iC, iW]（输入通道数，输入宽度）或 [N, iC, iW]（批大小，输入通道数，输入宽度）
- weight：[oC, iC/groups, wW]（输出通道数，每组输入通道数，卷积核宽度）
- bias（可选）：[oC]（输出通道偏置）
- init或result：[oC, oW]（输出通道数，输出宽度）或 [N, oC, oW]（批大小，输出通道数，输出宽度）

**可配置参数（含默认值）**：

- stride：卷积步长。接受[1, 255]内的整数标量，默认1
- padding：零填充元素个数。接受整数标量或2元素整数数组[paddingL, paddingR]，默认0；标量形式对左右两侧同时生效
- dilation：卷积核空洞率，取值[1, 255]，默认1
- groups：分组卷积组数，默认1

**特性**：`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`

**接口**：`BiShengIRAggregatedOpInterface`、`DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `stride` | `::mlir::IntegerAttr` | 32位无符号整数属性 |
| `padding` | `::mlir::Attribute` | 任意属性 |
| `dilation` | `::mlir::IntegerAttr` | 32位无符号整数属性 |
| `groups` | `::mlir::IntegerAttr` | 32位无符号整数属性 |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 任意带形状输入张量 |
| `weight` | 任意带形状权重张量 |
| `bias` | 任意带形状偏置张量 |
| `init` | 任意带形状初始化张量 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `result` | 任意类型输出张量 |

### hfusion.conv2d (hfusion::Conv2DOp)

**功能**：二维卷积运算，语义对齐`triton.language.extra.cann.extension.conv2d(input, weight, bias, groups, padding_size)`，默认参数对齐`torch.nn.functional.conv2d(stride=1, dilation=1, padding_mode='zeros')`。

**维度约束**：

- input：[iC, iH, iW]（输入通道数，输入高度，输入宽度）或 [N, iC, iH, iW]（批大小，输入通道数，输入高度，输入宽度）
- weight：[oC, iC/groups, wH, wW]（输出通道数，每组输入通道数，卷积核高度，卷积核宽度）
- bias（可选）：[oC]（输出通道偏置）
- init或result：[oC, oH, oW]（输出通道数，输出高度，输出宽度）或 [N, oC, oH, oW]（批大小，输出通道数，输出高度，输出宽度）

**可配置参数（含默认值）**：

- stride：卷积步长。接受整数标量或2元素整数数组[strideH, strideW]，每项取值[1, 255]，默认1
- padding：零填充元素个数。接受整数标量、2元素整数数组[paddingH, paddingW]或4元素整数数组[paddingT, paddingB, paddingL, paddingR]，默认0；标量与2元素形式在各维度上应用对称填充
- dilation：卷积核空洞率。接受整数标量或2元素整数数组[dilationH, dilationW]，每项取值[1, 255]，默认1
- groups：分组卷积组数，默认1

**特性**：`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`

**接口**：`BiShengIRAggregatedOpInterface`、`DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `stride` | `::mlir::Attribute` | 任意属性 |
| `padding` | `::mlir::Attribute` | 任意属性 |
| `dilation` | `::mlir::Attribute` | 任意属性 |
| `groups` | `::mlir::IntegerAttr` | 32位无符号整数属性 |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 任意带形状输入张量 |
| `weight` | 任意带形状权重张量 |
| `bias` | 任意带形状偏置张量 |
| `init` | 任意带形状初始化张量 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `result` | 任意类型输出张量 |

### hfusion.conv3d (hfusion::Conv3DOp)

**功能**：三维卷积运算，语义对齐`triton.language.extra.cann.extension.conv3d(input, weight, bias, groups, padding_size)`，默认参数对齐`torch.nn.functional.conv3d(stride=1, dilation=1, padding_mode='zeros')`。

**维度约束**：

- input：[iC, iD, iH, iW]（输入通道数，输入深度，输入高度，输入宽度）或 [N, iC, iD, iH, iW]（批大小，输入通道数，输入深度，输入高度，输入宽度）
- weight：[oC, iC/groups, wD, wH, wW]（输出通道数，每组输入通道数，卷积核深度，卷积核高度，卷积核宽度）
- bias（可选）：[oC]（输出通道偏置）
- init或result：[oC, oD, oH, oW]（输出通道数，输出深度，输出高度，输出宽度）或 [N, oC, oD, oH, oW]（批大小，输出通道数，输出深度，输出高度，输出宽度）

**可配置参数（含默认值）**：

- stride：卷积步长。接受整数标量或3元素整数数组[strideD, strideH, strideW]，默认1；当前strideD必须为1，strideH与strideW取值[1, 255]
- padding：零填充元素个数。接受整数标量、3元素整数数组[paddingD, paddingH, paddingW]或6元素整数数组[paddingFront, paddingBack, paddingT, paddingB, paddingL, paddingR]，默认0；标量与3元素形式在各维度上应用对称填充
- dilation：卷积核空洞率。接受整数标量或3元素整数数组[dilationD, dilationH, dilationW]，默认1；当前dilationD必须为1，dilationH与dilationW取值[1, 255]
- groups：分组卷积组数，默认1

**特性**：`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`

**接口**：`DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `stride` | `::mlir::Attribute` | 任意属性 |
| `padding` | `::mlir::Attribute` | 任意属性 |
| `dilation` | `::mlir::Attribute` | 任意属性 |
| `groups` | `::mlir::IntegerAttr` | 32位无符号整数属性 |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 任意带形状输入张量 |
| `weight` | 任意带形状权重张量 |
| `bias` | 任意带形状偏置张量 |
| `init` | 任意带形状初始化张量 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `result` | 任意类型输出张量 |

### hfusion.cummax (hfusion::CummaxOp)

**功能**：计算输入张量指定维度上的累积最大值，当前仅支持单个累积维度。

**语法**：

```mlir
operation ::= `hfusion.cummax` $input attr-dict `:` type($input)
              `cum_dims` `=` $cum_dims
              `reverse` `=` $reverse
              `->` type($output)
```

**特性**：`AlwaysSpeculatableImplTrait`、`SameOperandsAndResultRank`

**接口**：`ConditionallySpeculatable`、`NoMemoryEffect`

**内存效应**：`MemoryEffects::Effect{}`

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `cum_dims` | `::mlir::DenseI64ArrayAttr` | i64稠密数组属性，需按递增顺序排列 |
| `reverse` | `::mlir::BoolAttr` | 布尔属性 |
| `propagate_nan` | `::mlir::BoolAttr` | 布尔属性 |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 定维张量，支持bfloat16、16/32位浮点、8/16/32/64位无符号整数 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 与输入同类型定维张量 |

### hfusion.cummin (hfusion::CumminOp)

**功能**：计算输入张量指定维度上的累积最小值，当前仅支持单个累积维度。

**语法**：

```mlir
operation ::= `hfusion.cummin` $input attr-dict `:` type($input)
              `cum_dims` `=` $cum_dims
              `reverse` `=` $reverse
              `->` type($output)
```

**特性**：`AlwaysSpeculatableImplTrait`、`SameOperandsAndResultRank`

**接口**：`ConditionallySpeculatable`、`NoMemoryEffect`

**内存效应**：`MemoryEffects::Effect{}`

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `cum_dims` | `::mlir::DenseI64ArrayAttr` | i64稠密数组属性，需按递增顺序排列 |
| `reverse` | `::mlir::BoolAttr` | 布尔属性 |
| `propagate_nan` | `::mlir::BoolAttr` | 布尔属性 |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 定维张量，支持bfloat16、16/32位浮点、8/16/32/64位无符号整数 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 与输入同类型定维张量 |

### hfusion.cumprod (hfusion::CumprodOp)

**功能**：在指定维度计算张量累积乘积，reverse控制累积方向，当前仅支持单维度累积。

**语法**：

```mlir
operation ::= `hfusion.cumprod` $input attr-dict `:` type($input) `cum_dims` `=` $cum_dims `reverse` `=` $reverse `->` type($output)
```

**特性**：`AlwaysSpeculatableImplTrait`、`SameOperandsAndResultRank`

**接口**：`ConditionallySpeculatable`、`NoMemoryEffect`

**内存效应**：`MemoryEffects::Effect{}`

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `cum_dims` | `::mlir::DenseI64ArrayAttr` | 累积维度数组，维度序号升序排列 |
| `reverse` | `::mlir::BoolAttr` | 是否反向累积 |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 定维张量，支持bfloat16、16/32位浮点、8/16/32/64位无符号整数 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 与输入同类型、同秩输出张量 |

### hfusion.cumsum (hfusion::CumsumOp)

**功能**：在指定维度计算张量累积和，reverse控制累积方向，当前仅支持单维度累积。

**语法**：

```mlir
operation ::= `hfusion.cumsum` $input attr-dict `:` type($input) `cum_dims` `=` $cum_dims `reverse` `=` $reverse `->` type($output)
```

**特性**：`AlwaysSpeculatableImplTrait`、`SameOperandsAndResultRank`

**接口**：`ConditionallySpeculatable`、`NoMemoryEffect`

**内存效应**：`MemoryEffects::Effect{}`

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `cum_dims` | `::mlir::DenseI64ArrayAttr` | 累积维度数组，维度序号升序排列 |
| `reverse` | `::mlir::BoolAttr` | 是否反向累积 |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 定维张量，支持bfloat16、16/32位浮点、8/16/32/64位无符号整数 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 与输入同类型、同秩输出张量 |

### hfusion.cyl_bessel_i0 (hfusion::CylBesselI0Op)

**功能**：逐元素计算零阶第一类修正贝塞尔函数，输入与输出必须为相同定维张量类型。

**语法**：

```mlir
operation ::= `hfusion.cyl_bessel_i0` $input attr-dict `:` type($input) `->` type($output)
```

**特性**：`AlwaysSpeculatableImplTrait`、`SameOperandsAndResultType`

**接口**：`ConditionallySpeculatable`、`InferTypeOpInterface`、`NoMemoryEffect`

**内存效应**：`MemoryEffects::Effect{}`

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 定维浮点张量，支持16/32位浮点 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 与输入同类型定维浮点张量 |

### hfusion.deinterleave (hfusion::DeinterleaveOp)

**功能**：对输入张量最后一维解交织，拆分两组元素：偶数索引、奇数索引，输入最后一维长度必须为2的倍数。

channelIndex控制输出：-1输出两组、0仅输出偶数通道、1仅输出奇数通道。

**语法**：

```mlir
operation ::= `hfusion.deinterleave` $input custom<HFusionDeinterleave>($channelIndex) attr-dict `:` type($input) `->` type($output)
```

**约束**：输入张量最后一维尺寸必须为2的倍数。

**特性**：`AlwaysSpeculatableImplTrait`、`Commutative`、`SameOperandsAndResultRank`

**接口**：`ConditionallySpeculatable`、`NoMemoryEffect`、`ReifyRankedShapedTypeOpInterface`

**内存效应**：`MemoryEffects::Effect{}`

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `channelIndex` | `::mlir::IntegerAttr` | 64位无符号整数，通道输出控制标识 |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 任意类型定维输入张量 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 变长任意类型定维输出张量 |

### hfusion.elemwise_binary (hfusion::ElemwiseBinaryOp)

**功能**：逐元素二元运算，自动将输入数值提升至输出（累加器）数据类型。

**特性**：`AttrSizedOperandSegments`、`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`

**接口**：`DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `fun` | `::mlir::hfusion::BinaryFnAttr` | 二元运算函数：vor、vand、vxor、minf、maxf、powf、mod、modui、shli、shrsi、shrui、ldexp、ceildivsi、ceildivui、floordivsi、powi、minnumf、maxnumf |
| `cast` | `::mlir::hfusion::TypeFnAttr` | 类型转换规则：cast_signed、cast_unsigned、bitcast |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `inputs` | 变长任意类型输入 |
| `outputs` | 变长任意带形状输出张量 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `result_tensors` | 变长任意类型定维输出张量 |

### hfusion.elemwise_ternary (hfusion::ElemwiseTernaryOp)

**功能**：逐元素应用三元函数`fun`。对输入操作数执行数值类型转换，提升为与累加器（输出）相同的数据类型。

**特性**：`AttrSizedOperandSegments`、`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`

**接口**：`DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `fun` | `::mlir::hfusion::TernaryFnAttr` | 三元函数，32位无符号整数取值0~1：select、fma |
| `cast` | `::mlir::hfusion::TypeFnAttr` | 类型转换方式，32位无符号整数取值0~2：cast_signed、cast_unsigned、bitcast |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `inputs` | 变长任意类型输入 |
| `outputs` | 变长任意带形状输出张量 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `result_tensors` | 变长任意类型定维输出张量 |

### hfusion.elemwise_unary (hfusion::ElemwiseUnaryOp)

**功能**：逐元素一元运算，自动将输入数值提升至输出（累加器）数据类型。

**特性**：`AttrSizedOperandSegments`、`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`

**接口**：`DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `fun` | `::mlir::hfusion::UnaryFnAttr` | 一元运算函数：relu、sqrt、rsqrt、rec、vnot、tanh、sin、cos、atan、tan、absi、erf、log2、log10、log1p、exp2、expm1、ilogb |
| `cast` | `::mlir::hfusion::TypeFnAttr` | 类型转换规则：cast_signed、cast_unsigned、bitcast |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `inputs` | 变长任意类型输入 |
| `outputs` | 变长任意带形状输出张量 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `result_tensors` | 变长任意类型定维输出张量 |

### hfusion.embedding_gather (hfusion::EmbeddingGatherOp)

**功能**：基于gather语义的嵌入查找操作（embedding table lookup）。输入为全局内存中的二维嵌入表与一维或二维索引张量，按索引收集对应嵌入向量，生成二维或三维输出张量。

语义等效（带边界检查）：

`result[b][i][d] = src[index[b][i]][d]`

其中：b为批维度，i为序列（位置）维度，d为嵌入特征维度。

**语法**：

```mlir
operation ::= `hfusion.embedding_gather` `ins` `(` $src `:` type($src) `,` $index `:` type($index) `,`
              $bound `:` type($bound) `,`
              `[` $offsets `:` type($offsets) `]` `,`
              `[` $numels `:` type($numels) `]` `)`
              `outs` `(` $dst `:` type($dst) `)`
              attr-dict
              (`->` type($result)^)?
```

**特性**：`SameVariadicOperandSize`

**接口**：`MemoryEffectOpInterface`

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `src` | 任意类型内存视图 |
| `index` | 任意类型定维索引张量 |
| `dst` | 任意类型定维输出张量 |
| `bound` | 32/64位无符号整数 |
| `offsets` | 变长32/64位无符号整数 |
| `numels` | 变长32/64位无符号整数 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `result` | 任意类型定维输出张量 |

### hfusion.erfinv (hfusion::ErfInvOp)

**功能**：逐元素计算逆误差函数，输入与输出必须为相同定维张量类型。

**语法**：

```mlir
operation ::= `hfusion.erfinv` $input attr-dict `:` type($input) `->` type($output)
```

**特性**：`AlwaysSpeculatableImplTrait`、`SameOperandsAndResultType`

**接口**：`ConditionallySpeculatable`、`InferTypeOpInterface`、`NoMemoryEffect`

**内存效应**：`MemoryEffects::Effect{}`

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 定维浮点张量，支持16/32位浮点 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 与输入同类型定维浮点张量 |

### hfusion.flip (hfusion::FlipOp)

**功能**：沿指定维度翻转张量，当前仅支持最后一维。

**语法**：

```mlir
operation ::= `hfusion.flip` $input attr-dict `:` type($input)
              `flip_axis` `=` $flip_axis
              `->` type($output)
```

**约束**：仅支持沿张量最后一维执行翻转。

**特性**：`AlwaysSpeculatableImplTrait`、`Commutative`

**接口**：`ConditionallySpeculatable`、`NoMemoryEffect`

**内存效应**：`MemoryEffects::Effect{}`

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `flip_axis` | `::mlir::IntegerAttr` | 64位无符号整数，待翻转维度序号 |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 任意类型定维输入张量 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 与输入同类型、同秩输出张量 |

### hfusion.gather (hfusion::GatherOp)

**功能**：沿指定轴从源张量收集元素，非收集维度形状与输入保持一致，对齐triton.language.gather语义。

**特性**：`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`

**接口**：`BiShengIRAggregatedOpInterface`、`DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `axis` | `::mlir::IntegerAttr` | 64位无符号整数，收集维度序号 |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `src` | 任意带形状源张量 |
| `index` | 任意带形状索引张量 |
| `init` | 任意带形状初始化输出张量 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `result` | 变长任意类型输出张量 |

### hfusion.gather_load (hfusion::GatherLoadOp)

**功能**：执行稀疏内存加载，支持可选掩码与回填值。给定源内存缓冲区与偏移张量，按指定偏移位置从源中收集元素生成输出张量。

**参数含义**如下：

- base：待加载数据基地址
- indices：待加载数据索引
- burst_len：表示索引值在最内维上的连续性
- mask：（可选）加载掩码，控制哪些位置实际加载、哪些位置不加载；除块指针场景外均有效
- other：（可选）掩码对应元素为false时的默认（回填）值；除块指针场景外均有效
- dst：目标缓冲区或张量，指定输出形状与类型
- cache：（可选）枚举提示，控制NPU L1缓存对全局内存加载数据的缓存策略
- evict：（可选）枚举提示，控制缓存空间占满时数据被移出NPU缓存的优先级
- isVolatile：（可选）标记为volatile的加载操作不允许被任何形式优化

**语法**：

```mlir
operation ::= `hfusion.gather_load` `ins` `(` $base `:` type($base) `,` $indices `:` type($indices) `,`
              $burst_len `:` type($burst_len)
              (`,` $mask `:` type($mask)^)? (`,` $other `:` type($other)^)? `)`
              `outs` `(` $dst `:` type($dst) `)`
              attr-dict
              (`->` type($result)^)?
```

**特性**：`InferTypeOpAdaptor`、`SameVariadicOperandSize`

**接口**：`DestinationStyleOpInterface`、`InferTypeOpInterface`、`MemoryEffectOpInterface`

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `cache` | `::mlir::hfusion::CacheModifierAttr` | 缓存修改策略 |
| `evict` | `::mlir::hfusion::EvictionPolicyAttr` | 缓存淘汰策略 |
| `isVolatile` | `::mlir::BoolAttr` | 布尔属性 |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `base` | 任意类型内存视图 |
| `indices` | 32/64位无符号整数定维张量 |
| `burst_len` | 32/64位无符号整数 |
| `mask` | 1位无符号整数定维张量 |
| `other` | 任意类型定维张量 |
| `dst` | 张量或内存视图 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `result` | 任意类型定维输出张量 |

### hfusion.gather_mask (hfusion::GatherMaskOp)

**功能**：按掩码张量中的非零值从`src`张量收集数据。`mask`对应元素非零（bool类型为True，int8类型为1）处的`src`元素被收集，写入预分配的`init`张量（等效HIVM的`dst`），并将`init`作为`result`返回以保持函数式兼容。`init`、`result`张量与`src`张量形状、元素类型相同；额外返回`dst_size`表示实际收集到的数据长度。

**用法**：

`result, dst_size = gather_mask(src, mask, init)`

**特性**：`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`

**接口**：`DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `src` | 任意带形状源张量 |
| `mask` | 任意带形状掩码张量 |
| `init` | 变长任意带形状初始化张量 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `result` | 变长任意类型定维输出张量 |

### hfusion.gatherT (hfusion::GatherTOp)

**功能**：gather操作，沿指定轴按索引张量从输入张量中选取元素。输入为源GM内存缓冲区与UB索引张量，按索引位置加偏移从源中收集元素，生成UB输出张量，支持1D~5D。

输出张量各位置取值规则：

```text
1D:
  dst[i] = src[index[i]]
2D:
  dim=0:
    dst[i][j] = src[index[i][j]][j]
  dim=1:
    dst[i][j] = src[i][index[i][j]]
3D:
  dim=0:
    dst[i][j][k] = src[index[i][j][k]][j][k]
  dim=1:
    dst[i][j][k] = src[i][index[i][j][k]][k]
  dim=2:
    dst[i][j][k] = src[i][j][index[i][j][k]]
```

**参数含义**：

- src：源GM内存缓冲区
- index：UB张量，指定在src中的位置
- dst：目标UB张量，指定输出形状与类型
- bound：收集维度大小
- dim：gather操作维度
- src_stride：源张量步长
- index_shape：索引张量形状
- offsets：dst缓冲区中的偏移

**语法**：

```mlir
operation ::= `hfusion.gatherT` `ins` `(` $src `:` type($src) `,`
              $index `:` type($index) `,`
              $bound `:` type($bound) `,`
              $dim `:` type($dim) `,`
              `[` $src_stride `:` type($src_stride) `]` `,`
              `[` $index_shape `:` type($index_shape) `]` `,`
              `[` $offsets `:` type($offsets) `]` `)`
              `outs` `(` $dst `:` type($dst) `)`
              attr-dict
              (`->` type($result)^)?
```

**特性**：`SameVariadicOperandSize`

**接口**：`MemoryEffectOpInterface`

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `src` | 任意类型内存视图 |
| `index` | 任意类型定维索引张量 |
| `dst` | 任意类型定维输出张量 |
| `bound` | 32/64位无符号整数 |
| `dim` | 32/64位无符号整数 |
| `src_stride` | 变长32/64位无符号整数 |
| `index_shape` | 变长32/64位无符号整数 |
| `offsets` | 变长32/64位无符号整数 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `result` | 任意类型定维输出张量 |

### hfusion.group_matmul (hfusion::GroupMatmulOp)

**功能**：分组矩阵乘法，用于MoE场景，为每个专家权重与对应Token执行矩阵乘。

**特性**：`AttrSizedOperandSegments`、`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`

**接口**：`DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `inputs` | 变长任意类型输入 |
| `outputs` | 变长任意带形状输出张量 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `result_tensors` | 变长任意类型定维输出张量 |

### hfusion.histogram (hfusion::HistogramOp)

**功能**：整数张量直方图统计，支持可选掩码，仅统计掩码为true的元素；输出为一维张量，长度等于分箱数量，分箱数为编译期常量。

**语法**：

```mlir
operation ::= `hfusion.histogram` $input `,` $num_bins (`,` $mask^)? attr-dict `:` type($input) (`,` type($mask)^)? `->` type($output)
```

**接口**：`BiShengIRAggregatedOpInterface`

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `num_bins` | `::mlir::IntegerAttr` | 64位无符号整数，直方图分箱总数 |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 定维整数张量，支持8/16/32/64位无符号整数 |
| `mask` | 可选1比特定维掩码张量 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 一维整数张量，支持32/64位无符号整数 |

### hfusion.hypot (hfusion::HypotOp)

**功能**：逐元素计算两个或三个张量的斜边长度（平方和开根）。

计算公式：

`hypot(x, y) = sqrt(x * x + y * y)`
`hypot(x, y, z) = sqrt(x * x + y * y + z * z)`

第三操作数可选，因此仅支持2输入与3输入两种形式。BF16仅支持2输入形式；3输入形式支持F16、F32。

**语法**：

```mlir
operation ::= `hfusion.hypot` $x `,` $y (`,` $z^)?
              attr-dict
              `:` type($x) `,` type($y) (`,` type($z)^)? `->` type($output)
```

**特性**：`AlwaysSpeculatableImplTrait`

**接口**：`ConditionallySpeculatable`、`NoMemoryEffect`

**内存效应**：`MemoryEffects::Effect{}`

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `x` | 定维张量，支持bfloat16、16/32位浮点 |
| `y` | 定维张量，支持bfloat16、16/32位浮点 |
| `z` | 定维张量，支持bfloat16、16/32位浮点 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 与输入同类型定维张量 |

### hfusion.index_put (hfusion::IndexPutOp)

**功能**：IndexPut操作，以SIMT模板将`value` UB张量中的值，按`index` UB张量指定的位置偏移，沿指定scatter维度写入`dst` GM缓冲区，支持2D~5D。

目标张量各位置赋值规则：

```text
2D:
  for dim = 0:
  dst[index[i]][offset[1] + j] = value[i][j]
3D:
  for dim = 0:
  dst[index[i]][offset[1] + j][offset[2] + k] = value[i][j][k]
  for dim = 1:
  dst[offset[0] + i][index[j]][offset[2] + k] = value[i][j][k]
4D:
  for dim = 0:
  dst[index[i]][offset[1] + j][offset[2] + k][offset[3] + l] = value[i][j][k][l]
  for dim = 1:
  dst[offset[0] + i][index[j]][offset[2] + k][offset[3] + l] = value[i][j][k][l]
  for dim = 2:
  dst[offset[0] + i][offset[1] + j][index[k]][offset[3] + l] = value[i][j][k][l]
```

**参数含义**：

- dst：GM上目标张量指针
- index：UB上的索引张量tile指针，指定目标张量中的位置；索引不得重复，且必须为指定scatter_dim维度上目标张量边界内的有效索引
- value：UB上的值tile指针
- scatter_dim：索引作用的收集维度
- bound：索引值上界
- end_offset：值tile的结束偏移；索引tile的结束偏移为元组中第dim个值
- dst_shape：值tile的起始偏移；索引tile的结束偏移为元组中第dim个值
- dst_offset：目标全局张量的步长

**语法**：

```mlir
operation ::= `hfusion.index_put` `ins` `(` $dst `:` type($dst) `,`
              $index `:` type($index) `,`
              $value `:` type($value) `,`
              $scatter_dim `:` type($scatter_dim) `,`
              $bound `:` type($bound) `,`
              `[` $end_offset `:` type($end_offset) `]` `,`
              `[` $start_offset `:` type($start_offset) `]` `,`
              `[` $dst_stride `:` type($dst_stride) `]` `)`
              attr-dict
```

**特性**：`SameVariadicOperandSize`

**接口**：`MemoryEffectOpInterface`

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `dst` | 任意类型内存视图 |
| `index` | 任意类型定维索引张量 |
| `value` | 任意类型定维值张量 |
| `scatter_dim` | 32/64位无符号整数 |
| `bound` | 32/64位无符号整数 |
| `end_offset` | 变长32/64位无符号整数 |
| `start_offset` | 变长32/64位无符号整数 |
| `dst_stride` | 变长32/64位无符号整数 |

### hfusion.indirect_load (hfusion::IndirectLoadOp)

**功能**：执行带掩码与回填值的间接内存加载。给定源内存缓冲区与偏移张量，按指定偏移位置从源中收集元素生成输出张量，支持1D~5D。

输出张量各位置取值规则：

```text
1D:
    dst[i] = mask[i] ? src[offsets[i]] : other[i]
2D:
    dst[i][j] = mask[i][j] ? src[offsets[i][j]] : other[i][j]
3D:
    dst[i][j][k] = mask[i][j][k] ? src[offsets[i][j][k]] : other[i][j][k]
4D:
    dst[i][j][k][l] = mask[i][j][k][l] ? src[offsets[i][j][k][l]] : other[i][j][k][l]
5D:
    dst[i][j][k][l][m] = mask[i][j][k][l][m] ? src[offsets[i][j][k][l][m]] : other[i][j][k][l][m]
```

**参数含义**：

- src：待加载的源内存缓冲区
- offsets：指定源缓冲区位置的索引
- mask：布尔掩码，控制加载哪些元素
- other：掩码为false时使用的回填值
- dst：目标张量，指定输出形状与类型
- isVolatile：默认为true。仅当分析证明该间接加载的GM地址不会被任何操作写入时才可置为false

该操作适用于稀疏数据访问模式与带条件加载语义的gather操作。

**语法**：

```mlir
operation ::= `hfusion.indirect_load` `ins` `(` $src `:` type($src) `,` $offsets `:` type($offsets)
              `,` $mask `:` type($mask)
              `,` $other `:` type($other) `)`
              `outs` `(` $dst `:` type($dst) `)`
              attr-dict
              (`->` type($result)^)?
```

**接口**：`MemoryEffectOpInterface`

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `isVolatile` | `::mlir::BoolAttr` | 布尔属性 |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `src` | 任意类型内存视图 |
| `offsets` | 32/64位无符号整数张量 |
| `dst` | 任意类型定维张量 |
| `mask` | 1位/8位无符号整数张量 |
| `other` | 整数或浮点张量 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `result` | 任意类型定维输出张量 |

### hfusion.indirect_store (hfusion::IndirectStoreOp)

**功能**：以SIMT模板执行间接内存存储。给定UB源张量与偏移张量，将源元素按指定偏移位置存入目标GM缓冲区，支持掩码以条件控制存储哪些元素，支持1D~5D。

源张量各位置存储规则：

```text
1D:
    if (mask[i]) dst[offsets[i]] = src[i]
2D:
    if (mask[i][j]) dst[offsets[i][j]] = src[i][j]
3D:
    if (mask[i][j][k]) dst[offsets[i][j][k]] = src[i][j][k]
4D:
    if (mask[i][j][k][l]) dst[offsets[i][j][k][l]] = src[i][j][k][l]
5D:
    if (mask[i][j][k][l][m]) dst[offsets[i][j][k][l][m]] = src[i][j][k][l][m]
```

**参数含义**：

- src：UB上待存储的源张量
- offsets：指定目标缓冲区位置的索引
- dst：待写入的目标GM内存缓冲区
- mask：可选布尔掩码，控制存储哪些元素

未提供掩码时，源张量所有元素均存入目标缓冲区对应偏移位置。

**语法**：

```mlir
operation ::= `hfusion.indirect_store` `ins` `(` $src `:` type($src) `,` $offsets `:` type($offsets)
              (`,` $mask^ `:` type($mask))? `)`
              `outs` `(` $dst `:` type($dst) `)`
              attr-dict
```

**接口**：`MemoryEffectOpInterface`

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `dst` | 任意类型内存视图 |
| `offsets` | 32/64位无符号整数张量 |
| `src` | 任意类型定维源张量 |
| `mask` | 1位/8位无符号整数张量 |

### hfusion.interleave (hfusion::InterleaveOp)

**功能**：沿最后一维交织多个张量元素，当前仅支持2个输入张量；所有输入形状、秩必须完全一致。

**语法**：

```mlir
operation ::= `hfusion.interleave` $input attr-dict `:` type($input) `->` type($output)
```

**约束**：仅支持2个输入张量，全部输入张量秩、形状必须完全相同。

**特性**：`AlwaysSpeculatableImplTrait`、`Commutative`、`SameOperandsAndResultRank`

**接口**：`ConditionallySpeculatable`、`NoMemoryEffect`、`ReifyRankedShapedTypeOpInterface`

**内存效应**：`MemoryEffects::Effect{}`

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 变长任意类型定维输入张量 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 任意类型定维交织输出张量 |

### hfusion.isfinite (hfusion::IsFiniteOp)

**功能**：判断浮点张量每个元素是否为有限值（非NaN、非正负无穷）。

**语法**：

```mlir
operation ::= `hfusion.isfinite` $input attr-dict `:` type($input) `->` type($output)
```

**特性**：`AlwaysSpeculatableImplTrait`、`SameOperandsAndResultRank`

**接口**：`BiShengIRAggregatedOpInterface`、`ConditionallySpeculatable`、`NoMemoryEffect`

**内存效应**：`MemoryEffects::Effect{}`

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 定维浮点张量，支持bfloat16、16/32位浮点 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 1比特定维布尔张量 |

### hfusion.isinf (hfusion::IsInfOp)

**功能**：判断浮点张量每个元素是否为正负无穷。

**语法**：

```mlir
operation ::= `hfusion.isinf` $input attr-dict `:` type($input) `->` type($output)
```

**特性**：`AlwaysSpeculatableImplTrait`、`SameOperandsAndResultRank`

**接口**：`ConditionallySpeculatable`、`NoMemoryEffect`

**内存效应**：`MemoryEffects::Effect{}`

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 定维浮点张量，支持bfloat16、16/32位浮点 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 1比特定维布尔张量 |

### hfusion.isnan (hfusion::IsNanOp)

**功能**：判断浮点张量每个元素是否为NaN。

**语法**：

```mlir
operation ::= `hfusion.isnan` $input attr-dict `:` type($input) `->` type($output)
```

**特性**：`AlwaysSpeculatableImplTrait`、`SameOperandsAndResultRank`

**接口**：`ConditionallySpeculatable`、`NoMemoryEffect`

**内存效应**：`MemoryEffects::Effect{}`

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 定维浮点张量，支持bfloat16、16/32位浮点 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 1比特定维布尔张量 |

### hfusion.load (hfusion::LoadOp)

**功能**：逐元素读取张量数据，不执行数值类型转换。

**特性**：`AttrSizedOperandSegments`、`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`

**接口**：`DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `inputs` | 变长任意类型输入 |
| `outputs` | 变长任意带形状输出张量 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `result_tensors` | 变长任意类型定维输出张量 |

### hfusion.matmul_mx (hfusion::MatMulMxOp)

**功能**：执行块缩放（block-scaled）矩阵乘法。输入采用微缩放（microscaling）格式并使用给定缩放因子隐式缩放，常用于FP8、FP4等量化数据类型；直接在硬件（Tensor Core）上计算`C = (A * scale_a) dot (B * scale_b)`。`lhs`与`rhs`使用微缩放格式，规范详见[OCP microscaling 规范](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)。

**语法**：

```mlir
operation ::= `hfusion.matmul_mx` attr-dict `ins` `(`
              $inputA
              `,` $inputB
              `,` $scaleA
              `,` $scaleB
              `:`
              type($inputA)
              `,` type($inputB)
              `,` type($scaleA)
              `,` type($scaleB)
              `)`
              `outs` `(` $acc `:` type($acc) `)`
              `->` type($result)
```

**特性**：`AlwaysSpeculatableImplTrait`

**接口**：`BiShengIRAggregatedOpInterface`、`ConditionallySpeculatable`、`DestinationStyleOpInterface`、`NoMemoryEffect`

**内存效应**：`MemoryEffects::Effect{}`

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `lhsFormat` | `::mlir::hfusion::DataformatAttr` | MatMulMx微缩放浮点数据格式：fp8_e5m2_t（FP8E5M2_T）、fp8_e4m3_t（FP8E4M3_T）、fp4_e2m1_t（FP4E2M1_T） |
| `rhsFormat` | `::mlir::hfusion::DataformatAttr` | MatMulMx微缩放浮点数据格式：fp8_e5m2_t（FP8E5M2_T）、fp8_e4m3_t（FP8E4M3_T）、fp4_e2m1_t（FP4E2M1_T） |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `inputA` | 定维张量，支持f8E4M3FN、f8E5M2、8位无符号或符号无关整数 |
| `inputB` | 定维张量，支持f8E4M3FN、f8E5M2、8位无符号或符号无关整数 |
| `scaleA` | 8位无符号或符号无关整数定维张量 |
| `scaleB` | 8位无符号或符号无关整数定维张量 |
| `acc` | 浮点定维累加张量 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `result` | 浮点定维输出张量 |

### hfusion.mulext (hfusion::MulExtOp)

**功能**：有符号整数扩展乘法，输入N位整数，输出两组N位结果：乘积低半段、乘积高半段；低半段等价普通乘法结果。

**语法**：

```mlir
operation ::= `hfusion.mulext` $lhs `,` $rhs attr-dict `:` type($lhs)
```

**特性**：`AlwaysSpeculatableImplTrait`、`Commutative`

**接口**：`ConditionallySpeculatable`、`InferTypeOpInterface`、`NoMemoryEffect`

**内存效应**：`MemoryEffects::Effect{}`

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `lhs` | signless-integer-like类型 |
| `rhs` | signless-integer-like类型 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `low` | 乘积低半段，signless-integer-like类型 |
| `high` | 乘积高半段，signless-integer-like类型 |

### hfusion.mulextui (hfusion::MulExtUiOp)

**功能**：无符号整数扩展乘法。对零扩展的无符号操作数执行（2*N）位乘法，返回两个N位结果：乘积低半段与乘积高半段；低半段与同操作数普通乘法结果相同。

**语法**：

```mlir
operation ::= `hfusion.mulextui` $lhs `,` $rhs attr-dict `:` type($lhs)
```

**特性**：`AlwaysSpeculatableImplTrait`、`Commutative`

**接口**：`ConditionallySpeculatable`、`InferTypeOpInterface`、`NoMemoryEffect`

**内存效应**：`MemoryEffects::Effect{}`

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `lhs` | signless-integer-like类型 |
| `rhs` | signless-integer-like类型 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `low` | signless-integer-like类型 |
| `high` | signless-integer-like类型 |

### hfusion.nextafter (hfusion::NextAfterOp)

**功能**：逐元素计算沿`y`方向上`x`的下一个可表示浮点值，操作数与结果必须为相同定维张量类型。

**语法**：

```mlir
operation ::= `hfusion.nextafter` $x `,` $y attr-dict `:` type($x) `,` type($y) `->` type($output)
```

**特性**：`AlwaysSpeculatableImplTrait`、`SameOperandsAndResultType`

**接口**：`ConditionallySpeculatable`、`InferTypeOpInterface`、`NoMemoryEffect`

**内存效应**：`MemoryEffects::Effect{}`

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `x` | 定维浮点张量，支持16/32位浮点 |
| `y` | 定维浮点张量，支持16/32位浮点 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 与输入同类型定维浮点张量 |

### hfusion.print (hfusion::PrintOp)

**功能**：设备端调试打印，接收前缀字符串与标量或张量，支持十六进制输出开关。

**语法**：

```mlir
operation ::= `hfusion.print` $prefix attr-dict $arg `:` type($arg)
```

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `prefix` | `::mlir::StringAttr` | 打印前缀文本 |
| `hex` | `::mlir::BoolAttr` | 是否以十六进制打印 |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `arg` | 整数、浮点或任意类型定维张量 |

### hfusion.reduce_with_index (hfusion::ReduceWithIndexOp)

**功能**：带索引的最大、最小值规约运算，仅支持单规约维度。两种使用模式：输入+索引张量输出结果与索引；仅输入张量自动生成索引。tie_break_left控制相等值取最左或最右索引。

**特性**：`AttrSizedOperandSegments`、`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`

**接口**：`DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `reduce_kind` | `::mlir::hfusion::ReduceWithIndexKindAttr` | 规约类型（最大、最小，含无符号分支） |
| `tie_break_left` | `::mlir::BoolAttr` | 等值时是否取左侧索引 |
| `dimensions` | `::mlir::DenseI64ArrayAttr` | 规约维度数组，升序排列 |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `inputs` | 变长任意带形状输入张量 |
| `inits` | 变长任意带形状初始化张量 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `result` | 变长任意类型输出张量 |

### hfusion.scatter_store (hfusion::ScatterStoreOp)

**功能**：执行稀疏内存存储，支持可选掩码。给定源张量与UB偏移张量，将源元素按指定偏移位置存入目标GM缓冲区，支持掩码以条件控制存储哪些元素。参数含义与gather_load相同。

**语法**：

```mlir
operation ::= `hfusion.scatter_store` `ins` `(` $indices `:` type($indices) `,`
              $data `:` type($data) `,` $burst_len `:` type($burst_len)
              (`,` $mask `:` type($mask)^)? `)`
              `outs` `(` $base `:` type($base) `)`
              attr-dict
              (`->` type($result)^)?
```

**特性**：`SameVariadicOperandSize`

**接口**：`DestinationStyleOpInterface`、`MemoryEffectOpInterface`

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `cache` | `::mlir::hfusion::CacheModifierAttr` | 缓存修改策略 |
| `evict` | `::mlir::hfusion::EvictionPolicyAttr` | 缓存淘汰策略 |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `indices` | 32/64位无符号整数定维张量 |
| `data` | 任意类型定维数据张量 |
| `burst_len` | 32/64位无符号整数 |
| `mask` | 1位无符号整数定维张量 |
| `base` | 张量或内存视图 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `result` | 任意类型定维输出张量 |

### hfusion.scatterT (hfusion::ScatterTOp)

**功能**：scatter操作，以SIMT模板将`value`张量中的所有值，按`index_tile`张量指定的索引写入`dst`。将UB张量中的值按`index_tile` UB张量指定的位置偏移写入`dst` GM缓冲区，支持2D~5D。

以3D为例的赋值规则：

```text
for dim = 0:
Dst[index_tile[i][j][k]][j][k] = value[i][j][k]
for dim = 1:
Dst[i][index_tile[i][j][k]][k] = value[i][j][k]
for dim = 2:
Dst[i][j][index_tile[i][j][k]] = value[i][j][k]
```

**参数含义**：

- dst：目标GM内存缓冲区基指针
- value：待存储的UB值张量
- index_tile：UB索引张量，指定目标张量中的位置，不应重复
- index_boundary：索引上界
- dim：执行赋值操作的维度
- dst_stride：目标步长元组
- index_shape：索引张量形状
- offsets：索引tile各维度的偏移元组

**语法**：

```mlir
operation ::= `hfusion.scatterT` `ins` `(` $dst `:` type($dst) `,`
              $value `:` type($value)`,`
              $index_tile `:` type($index_tile)`,`
              $index_boundary `:` type($index_boundary)`,`
              $dim `:` type($dim)`,`
              `[` $dst_stride `:` type($dst_stride) `]` `,`
              `[` $index_shape `:` type($index_shape) `]` `,`
              `[` $offsets `:` type($offsets) `]` `)`
              attr-dict
```

**特性**：`AttrSizedOperandSegments`

**接口**：`MemoryEffectOpInterface`

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `dst` | 任意类型内存视图 |
| `value` | 任意类型定维值张量 |
| `index_tile` | 任意类型定维索引张量 |
| `index_boundary` | 32/64位无符号整数 |
| `dim` | 32/64位无符号整数 |
| `dst_stride` | 变长32/64位无符号整数 |
| `index_shape` | 变长32/64位无符号整数 |
| `offsets` | 变长32/64位无符号整数 |

### hfusion.select (hfusion::SelectOp)

**功能**：根据首个二元条件操作数选择对应数值。

**特性**：`AttrSizedOperandSegments`、`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`

**接口**：`DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `inputs` | 变长任意类型输入 |
| `outputs` | 变长任意带形状输出张量 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `result_tensors` | 变长任意类型定维输出张量 |

### hfusion.signbit (hfusion::SignBitOp)

**功能**：提取浮点值的符号位。符号位置位时返回1（i8）或true（i1），否则返回0。

**特性**：`AlwaysSpeculatableImplTrait`、`Elementwise`、`SameOperandsAndResultShape`

**接口**：`ConditionallySpeculatable`、`NoMemoryEffect`

**内存效应**：`MemoryEffects::Effect{}`

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 定维浮点张量，支持16/32位浮点 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 1位/8位无符号整数定维张量 |

### hfusion.sort (hfusion::SortOp)

**功能**：沿指定轴排序张量，输出排序数值与对应索引。

**约束**：

1. 输入向量和输出向量必须具有相同的秩。
2. 当前仅支持尾部轴排序。

**参数说明**：

- `src`：待排序的张量或memref
- `dst_value`：用于存储排序后值的张量或memref
- `dst_index`：用于存储与dst_value对应的索引的张量或memref
- `descending`：决定按升序还是降序排序。默认为false，即升序
- `sort_axis`：待排序的轴

**语法**：

```mlir
operation ::= `hfusion.sort` attr-dict `ins` `(` $src `:` type($src) `)`
              `descending` `=` $descending
              `sort_axis` `=` $sort_axis
              (`->` type($result)^)?
```

**示例**：

```mlir
%result = hfusion.sort ins(%src : tensor<?xf32>) descending = true sort_axis = 0 -> tensor<?xf32>
```

**特性**：`SameOperandsAndResultRank`

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `descending` | `::mlir::BoolAttr` | 是否降序排列 |
| `sort_axis` | `::mlir::IntegerAttr` | 64位无符号整数，排序维度序号 |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `src` | 待排序张量或内存视图 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `result` | 变长任意类型定维输出张量 |

### hfusion.store (hfusion::StoreOp)

**功能**：逐元素存储张量数据，不执行数值类型转换，支持原子写入模式。

**特性**：`AttrSizedOperandSegments`、`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`

**接口**：`DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `atomic_kind` | `::mlir::hfusion::AtomicKindAttr` | 原子操作类型，可选值：none、add、max、min、and、or、xor、cas、xchg、umax、umin |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `inputs` | 变长任意类型输入 |
| `outputs` | 变长任意带形状输出张量 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `result_tensors` | 变长任意类型定维输出张量 |

### hfusion.stride_load (hfusion::StrideLoadOp)

**功能**：执行带标量填充的跨步内存加载。给定源GM缓冲区、一个线性基偏移、各维物理步长与逻辑有效元素个数，按下式加载生成目标张量或内存视图：

```text
dst[i, ...] = all(dim_index < numel[dim])
    ? src[offset + sum(dim_index * stride[dim])]
    : other
```

该操作面向SIMT Stride Load，当前实现支持1D~3D。

**语法**：

```mlir
operation ::= `hfusion.stride_load` `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `offset` `(` $offset `:` type($offset) `)`
              `other` `(` $other `:` type($other) `)`
              `strides` `(` `[` $stride `:` type($stride) `]` `)`
              `numels` `(` `[` $numel `:` type($numel) `]` `)`
              attr-dict
              (`->` type($result)^)?
```

**特性**：`SameVariadicOperandSize`

**接口**：`MemoryEffectOpInterface`

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `src` | 任意类型内存视图 |
| `dst` | 任意类型定维张量 |
| `offset` | 32/64位无符号整数 |
| `other` | 任意类型 |
| `stride` | 变长32/64位无符号整数 |
| `numel` | 变长32/64位无符号整数 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `result` | 任意类型定维输出张量 |

### hfusion.stride_store (hfusion::StrideStoreOp)

**功能**：执行跨步内存存储。给定目标GM缓冲区、源张量、一个线性基偏移、各维物理步长与逻辑有效元素个数，在满足`all(dim_index < numel[dim])`的位置按下式存储：

```text
dst[offset + sum(dim_index * stride[dim])] = src[i, ...]
```

该操作面向SIMT Stride Store，当前实现支持1D~3D。

**语法**：

```mlir
operation ::= `hfusion.stride_store` `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `offset` `(` $offset `:` type($offset) `)`
              `strides` `(` `[` $stride `:` type($stride) `]` `)`
              `numels` `(` `[` $numel `:` type($numel) `]` `)`
              attr-dict
```

**特性**：`SameVariadicOperandSize`

**接口**：`MemoryEffectOpInterface`

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `dst` | 任意类型内存视图 |
| `src` | 任意类型定维源张量 |
| `offset` | 32/64位无符号整数 |
| `stride` | 变长32/64位无符号整数 |
| `numel` | 变长32/64位无符号整数 |

### hfusion.symbolic_dim (hfusion::SymbolicDimOp)

**功能**：通过符号名称引用符号维度，返回index类型数值。

**语法**：

```mlir
operation ::= `hfusion.symbolic_dim` $symbolName attr-dict `:` type($result)
```

**特性**：`AlwaysSpeculatableImplTrait`

**接口**：`ConditionallySpeculatable`、`InferTypeOpInterface`、`NoMemoryEffect`

**内存效应**：`MemoryEffects::Effect{}`

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `symbolName` | `::mlir::SymbolRefAttr` | 符号维度引用 |

**结果**：

| 结果 | 说明 |
| :----: | ----------- |
| `result` | index类型维度值 |

## 属性

### AtomicKindAttr

**语法**：

```mlir
#hfusion.atomic_kind<
  ::mlir::hfusion::AtomicKind   # value
>
```

**功能**：原子操作类型属性，32位无符号整数取值范围0~10。

枚举取值：none、add、max、min、and、or、xor、cas、xchg、umax、umin

**参数**：

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hfusion::AtomicKind` | AtomicKind枚举值 |

### BinaryFnAttr

**语法**：

```mlir
#hfusion.binary_fn<
  ::mlir::hfusion::BinaryFn   # value
>
```

**功能**：二元逐元素运算函数属性，32位无符号整数取值范围0~17。

枚举取值：vor、vand、vxor、minf、maxf、powf、mod、modui、shli、shrsi、shrui、ldexp、ceildivsi、ceildivui、floordivsi、powi、minnumf、maxnumf

**参数**：

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hfusion::BinaryFn` | BinaryFn枚举值 |

### CompareFnAttr

**语法**：

```mlir
#hfusion.compare_fn<
  ::mlir::hfusion::CompareFn   # value
>
```

**功能**：逐元素比较函数属性，32位无符号整数取值范围0~9。

枚举取值：veq、vne、vle、vlt、vge、vgt、vule、vult、vuge、vugt

**参数**：

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hfusion::CompareFn` | CompareFn枚举值 |

### BindSubBlockAttr

**语法**：`#hfusion.bind_sub_block`

**功能**：标记绑定子块专用操作。

### CacheModifierAttr

**语法**：

```mlir
#hfusion.cache_modifier<
  ::mlir::hfusion::CacheModifier   # policy
>
```

**功能**：缓存修改策略属性。

枚举取值：none、ca、cg、wb、cs、wt、cv

**参数**：

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| policy | `::mlir::hfusion::CacheModifier` | CacheModifier枚举值 |

### EvictionPolicyAttr

**语法**：

```mlir
#hfusion.eviction_policy<
  ::mlir::hfusion::EvictionPolicy   # policy
>
```

**功能**：缓存淘汰策略属性。

枚举取值：EvictNormal、EvictFirst、EvictLast

**参数**：

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| policy | `::mlir::hfusion::EvictionPolicy` | EvictionPolicy枚举值 |

### FusionKindAttr

**语法**：

```mlir
#hfusion.fusion_kind<
  ::mlir::hfusion::FusionKind   # fusion_kind
>
```

**功能**：HFusion融合内核类型标识属性。

**参数**：

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| fusion_kind | `::mlir::hfusion::FusionKind` | FusionKind枚举值 |

### InsertSliceSourceIndexAttr

**语法**：`#hfusion.insert_slice_source_index`

**功能**：标记concat操作中作为insert_slice输入的操作数序号。

### MultiBufferAttr

**语法**：`#hfusion.multi_buffer`

**功能**：目标操作多缓冲配置属性。

### PaddingOptionAttr

**语法**：

```mlir
#hfusion.padding_option<
  ::mlir::hfusion::PaddingOption   # option
>
```

**功能**：填充选项属性。

枚举取值：zero、nan

**参数**：

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| option | `::mlir::hfusion::PaddingOption` | PaddingOption枚举值 |

### ReduceComposeAttr

**语法**：`#hfusion.reduce_composed`

**功能**：标记组合式规约运算。

### ReduceWithIndexKindAttr

**语法**：

```mlir
#hfusion.reduce_with_index_kind<
  ::mlir::hfusion::ReduceWithIndexKind   # reduce_with_index_kind
>
```

**功能**：带索引规约运算类型属性。

**参数**：

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| reduce_with_index_kind | `::mlir::hfusion::ReduceWithIndexKind` | ReduceWithIndexKind枚举值 |

### ReturnOperandNumAttr

**语法**：`#hfusion.return_operand_num`

**功能**：指定当前参数对应函数返回值下标。

### StrideAlignDimsAttr

**语法**：`#hfusion.stride_align_dims`

**功能**：标记需要步长对齐的维度。

### StrideAlignValueInByteAttr

**语法**：`#hfusion.stride_align_value_in_byte`

**功能**：步长对齐字节数值配置属性。

### RoundModeAttr

**语法**：

```mlir
#hfusion.round_mode<
  ::mlir::hfusion::RoundMode   # value
>
```

**功能**：数值转换舍入模式属性，32位无符号整数取值范围0~6。

枚举说明：

- RINT：四舍五入到最近偶数
- ROUND：四舍五入远离零
- FLOOR：向负无穷取整
- CEIL：向正无穷取整
- TRUNC：向零截断
- ODD：冯·诺依曼奇数舍入

**参数**：

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hfusion::RoundMode` | RoundMode枚举值 |

### TernaryFnAttr

**语法**：

```mlir
#hfusion.ternary_fn<
  ::mlir::hfusion::TernaryFn   # value
>
```

**功能**：三元运算函数属性，仅支持select。

**参数**：

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hfusion::TernaryFn` | TernaryFn枚举值 |

### TypeFnAttr

**语法**：

```mlir
#hfusion.type_fn<
  ::mlir::hfusion::TypeFn   # value
>
```

**功能**：类型转换规则属性，32位无符号整数取值0~2。

枚举取值：cast_signed、cast_unsigned、bitcast

**参数**：

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hfusion::TypeFn` | TypeFn枚举值 |

### UnaryFnAttr

**语法**：

```mlir
#hfusion.unary_fn<
  ::mlir::hfusion::UnaryFn   # value
>
```

**功能**：一元逐元素运算函数属性，32位无符号整数取值0~17。

枚举取值：relu、sqrt、rsqrt、rec、vnot、tanh、sin、cos、atan、tan、absi、erf、log2、log10、log1p、exp2、expm1、ilogb

**参数**：

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hfusion::UnaryFn` | UnaryFn枚举值 |

### UnsignedModeAttr

**语法**：

```mlir
#hfusion.unsigned_mode<
  ::mlir::hfusion::UnsignedMode   # value
>
```

**功能**：标识有符号与无符号整数间的转换（如 (u)int → (u)int）信息的属性，32位无符号整数取值范围0~3。

枚举取值：si2si、si2ui、ui2si、ui2ui

**参数**：

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hfusion::UnsignedMode` | UnsignedMode枚举值 |

## 枚举

### AtomicKind

**取值范围**：32位无符号整数0~10

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| NONE | 0 | none |
| ADD | 1 | add |
| MAX | 2 | max |
| MIN | 3 | min |
| AND | 4 | and |
| OR | 5 | or |
| XOR | 6 | xor |
| CAS | 7 | cas |
| XCHG | 8 | xchg |
| UMAX | 9 | umax |
| UMIN | 10 | umin |

### BinaryFn

**取值范围**：32位无符号整数0~20

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| vor | 0 | vor |
| vand | 1 | vand |
| vxor | 2 | vxor |
| minf | 3 | minf |
| maxf | 4 | maxf |
| powf | 5 | powf |
| mod | 6 | mod |
| modui | 7 | modui |
| shli | 8 | shli |
| shrsi | 9 | shrsi |
| shrui | 10 | shrui |
| ldexp | 11 | ldexp |
| ceildivsi | 12 | ceildivsi |
| ceildivui | 13 | ceildivui |
| floordivsi | 14 | floordivsi |
| powi | 15 | powi |
| minnumf | 16 | minnumf |
| maxnumf | 17 | maxnumf |
| atan2 | 18 | atan2 |
| divfhp | 19 | divfhp |
| copysign | 20 | copysign |

### CastMode

**取值范围**：32位无符号整数0~8

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| F32TOI8 | 0 | F32TOI8 |
| F32TOI16 | 1 | F32TOI16 |
| F16TOI8 | 2 | F16TOI8 |
| I64TOI32 | 3 | I64TOI32 |
| I64TOI16 | 4 | I64TOI16 |
| I64TOI8 | 5 | I64TOI8 |
| I32TOI16 | 6 | I32TOI16 |
| I32TOI8 | 7 | I32TOI8 |
| I16TOI8 | 8 | I16TOI8 |

### CompareFn

**取值范围**：32位无符号整数0~9

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| veq | 0 | veq |
| vne | 1 | vne |
| vle | 2 | vle |
| vlt | 3 | vlt |
| vge | 4 | vge |
| vgt | 5 | vgt |
| vule | 6 | vule |
| vult | 7 | vult |
| vuge | 8 | vuge |
| vugt | 9 | vugt |

### CacheModifier

**功能**：缓存修改策略

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| NONE | 1 | none |
| CA | 2 | ca |
| CG | 3 | cg |
| WB | 4 | wb |
| CS | 5 | cs |
| WT | 6 | wt |
| CV | 7 | cv |

### EvictionPolicy

**功能**：缓存淘汰策略

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| NORMAL | 1 | EvictNormal |
| EVICT_FIRST | 2 | EvictFirst |
| EVICT_LAST | 3 | EvictLast |

### FlattenMode

**功能**：HFusion张量扁平化模式

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| Greedy | 1 | Greedy |
| Tidy | 2 | Tidy |

### FusionKind

**功能**：HFusion融合内核分类

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| PureElemwise | 1 | PURE_ELEMWISE |
| AnyPB | 2 | ANY_PB |
| LastAxisPBR | 3 | LAST_AXIS_PBR |
| AnyPBR | 4 | ANY_PBR |
| SingleCube | 5 | SINGLE_CUBE |
| ShallowCV | 6 | SHALLOW_CV |
| ShallowVV | 7 | SHALLOW_VV |
| MixCV | 8 | MIX_CV |
| MixC2 | 9 | MIX_C2 |
| Unknown | 10 | UNKNOWN |

### Dataformat

**功能**：MatMulMx微缩放浮点数据格式

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| FP8E5M2_T | 1 | fp8_e5m2_t |
| FP8E4M3_T | 2 | fp8_e4m3_t |
| FP4E2M1_T | 3 | fp4_e2m1_t |

### OutputMode

**功能**：HFusion输出布局模式

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| Multiple | 1 | Multiple |
| Single | 2 | Single |
| SingleAggressive | 3 | SingleAggressive |

### PaddingOption

**功能**：填充选项，指定带边界检查（boundaryCheck）的load类操作对越界元素的填充方式，语义与Triton `tt.load`的`padding`选项一致。

- `PAD_ZERO`：越界元素填充为0。
- `PAD_NAN`：越界元素填充为NaN（适用于浮点类型）。

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| PAD_ZERO | 1 | zero |
| PAD_NAN | 2 | nan |

### CumOpType

**功能**：累积运算类型

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| UNDEFINED | 0 | undefined |
| CUMSUM | 1 | cumsum |
| CUMPROD | 2 | cumprod |
| CUMMAX | 3 | cummax |
| CUMMIN | 4 | cummin |

### MmMapMode

**取值范围**：32位无符号整数0~1

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| CoreOp | 0 | core_op |
| MacroInstr | 1 | macro_instr |

### ReduceWithIndexKind

**取值范围**：32位无符号整数0~3

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| MIN | 0 | min |
| MAX | 1 | max |
| MINUI | 2 | minui |
| MAXUI | 3 | maxui |

### RoundMode

**取值范围**：32位无符号整数0~6

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| RINT | 0 | rint |
| ROUND | 1 | round |
| FLOOR | 2 | floor |
| CEIL | 3 | ceil |
| TRUNC | 4 | trunc |
| ODD | 5 | odd |
| TRUNCWITHOVERFLOW | 6 | truncwithoverflow |

### TaylerMode

**取值范围**：32位无符号整数0~1

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| SIN | 0 | sin |
| ATAN | 1 | atan |

### TernaryFn

**取值范围**：32位无符号整数0~1

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| select | 0 | select |
| fma | 1 | fma |

### TypeFn

**取值范围**：32位无符号整数0~2

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| cast_signed | 0 | cast_signed |
| cast_unsigned | 1 | cast_unsigned |
| bitcast | 2 | bitcast |

### UnaryFn

**取值范围**：32位无符号整数0~26

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| relu | 0 | relu |
| sqrt | 1 | sqrt |
| rsqrt | 2 | rsqrt |
| rec | 3 | rec |
| vnot | 4 | vnot |
| tanh | 5 | tanh |
| sin | 6 | sin |
| cos | 7 | cos |
| atan | 8 | atan |
| tan | 9 | tan |
| absi | 10 | absi |
| erf | 11 | erf |
| log2 | 12 | log2 |
| log10 | 13 | log10 |
| log1p | 14 | log1p |
| exp2 | 15 | exp2 |
| expm1 | 16 | expm1 |
| ilogb | 17 | ilogb |
| asin | 18 | asin |
| acos | 19 | acos |
| asinh | 20 | asinh |
| acosh | 21 | acosh |
| atanh | 22 | atanh |
| sinh | 23 | sinh |
| lgamma | 24 | lgamma |
| cosh | 25 | cosh |
| nearbyint | 26 | nearbyint |

### UnsignedMode

**取值范围**：32位无符号整数0~3

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| SI2SI | 0 | si2si |
| SI2UI | 1 | si2ui |
| UI2SI | 2 | ui2si |
| UI2UI | 3 | ui2ui |
