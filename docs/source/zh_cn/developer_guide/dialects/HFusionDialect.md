# hfusion方言

Hybrid Fusion（HFusion）方言。

## 操作定义

### hfusion.arange (hfusion::ArangeOp)

**功能：** 生成序列，与标准arange存在差异，支持偏移量（默认值0）与多维场景并配套多维步长。偏移量、步长定义规则与内存描述符保持一致。

三维arange取值计算公式：
`arange[i, j, k] = offset + stride[0] * i + stride[1] * j + stride[2] * k`

**特性：** `AttrSizedOperandSegments`、`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`

**接口：** `DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `offset` | index类型偏移值 |
| `strides` | 变长index类型步长数组 |
| `init` | 任意类型带形状初始化张量 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `result_tensor` | 任意类型带形状输出张量 |

### hfusion.assert (hfusion::AssertOp)

**功能：** 设备端调试断言。接收字符串提示信息与标量/张量判断条件。

**语法：**

```mlir
operation ::= `hfusion.assert` $msg attr-dict $cond `:` type($cond)
```

**属性：**

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `msg` | `::mlir::StringAttr` | 字符串提示文本 |

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `cond` | 整数或任意类型定维张量 |

### hfusion.atomic_cas (hfusion::AtomicCasOp)

**功能：** 原子比较交换（CAS）操作。包含内存地址、预期旧值、新值三个输入；仅当内存值等于预期旧值时，将内存更新为新值，无论是否更新均返回内存原始值。

**约束：** 输入、输出秩与元素类型必须完全一致。

**参数说明：**

- `src0`：预期旧值
- `src1`：待写入新值
- `dst`：全局内存目标地址

**语法：**

```mlir
operation ::= `hfusion.atomic_cas` attr-dict `ins` `(` $input `:` type($input) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`->` type($output)^)?
```

**示例：**

```mlir
hfusion.atomic_cas ins(%src0, %src1 : memref<?xf32>, memref<?xf32>) outs(%dst : memref<?xf32>)
%result = hfusion.atomic_cas ins(%src0, %src1 : tensor<?xf32>, tensor<?xf32>) outs(%dst : tensor<?xf32>) -> tensor<?xf32>
```

**特性：** `SameOperandsAndResultRank`
**接口：** `MemoryEffectOpInterface`

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 变长张量/内存视图输入 |
| `dst` | 目标张量/内存视图 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 输出张量/内存视图 |

### hfusion.atomic_rmw (hfusion::AtomicRMWOp)

**功能：** 原子读改写操作。流程：读取内存原值 → 根据原子类型执行计算 → 写入新值并返回原值，全程不可中断。

**约束：** 输入、输出内存视图秩与元素类型必须完全一致。

**参数说明：**

- `src`：计算新值
- `dst`：全局内存目标地址

**语法：**

```mlir
operation ::= `hfusion.atomic_rmw` attr-dict `ins` `(` $input `:` type($input) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `atomic_kind` `=` $atomic_kind
              (`->` type($output)^)?
```

**示例：**

```mlir
hfusion.atomic_rmw ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) atomic_kind = <add>
%result = hfusion.atomic_rmw ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) atomic_kind = <or> -> tensor<?xf32>
```

**特性：** `SameOperandsAndResultRank`
**接口：** `MemoryEffectOpInterface`

**属性：**

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `atomic_kind` | `::mlir::hfusion::AtomicKindAttr` | 原子操作类型，可选值：none、add、max、min、and、or、xor、cas、xchg、umax、umin |

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 输入张量/内存视图 |
| `dst` | 目标张量/内存视图 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 输出张量/内存视图 |

### hfusion.atomic_xchg (hfusion::AtomicXchgOp)

**功能：** 原子交换操作。流程：读取内存原值 → 写入新值 → 返回原值，全程不可中断。

**约束：** 输入、输出内存视图秩与元素类型必须完全一致。

**参数说明：**

- `src`：待写入新值
- `dst`：全局内存目标地址
- `mask`：可选掩码元素

**语法：**

```mlir
operation ::= `hfusion.atomic_xchg` attr-dict `ins` `(` $input `:` type($input) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`mask` `(` $mask^ `:` type($mask) `)`)?
              (`->` type($output)^)?
```

**示例：**

```mlir
hfusion.atomic_xchg ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) mask(%m : memref<?xi1>)
%result = hfusion.atomic_xchg ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) mask(%m : memref<?xi1>) -> tensor<?xf32>
```

**特性：** `SameOperandsAndResultRank`

**接口：** `MemoryEffectOpInterface`

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 输入张量/内存视图 |
| `dst` | 目标张量/内存视图 |
| `mask` | 可选掩码张量/内存视图 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 输出张量/内存视图 |

### hfusion.barrier (hfusion::BarrierOp)

**功能：** 同步单个计算核内所有流水线。

**语法：**

```mlir
operation ::= `hfusion.barrier` attr-dict
```

### hfusion.bitcast (hfusion::BitcastOp)

**功能：** 逐元素执行比特类型转换。

**特性：** `AttrSizedOperandSegments`、`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`
**接口：** `DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `inputs` | 变长任意类型输入 |
| `outputs` | 变长任意带形状输出张量 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `result_tensors` | 变长任意类型定维输出张量 |

### hfusion.cast (hfusion::CastOp)

**功能：** 逐元素执行数值类型转换。

**特性：** `AttrSizedOperandSegments`、`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`
**接口：** `DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**属性：**

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `round_mode` | `::mlir::hfusion::RoundModeAttr` | 舍入模式：RINT、ROUND、FLOOR、CEIL、TRUNC、ODD |
| `enable_overflow` | `::mlir::BoolAttr` | 是否开启溢出检测 |
| `cast` | `::mlir::hfusion::TypeFnAttr` | 转换类型：cast_signed、cast_unsigned、bitcast |

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `inputs` | 变长任意类型输入 |
| `outputs` | 变长任意带形状输出张量 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `result_tensors` | 变长任意类型定维输出张量 |

### hfusion.compare (hfusion::CompareOp)

**功能：** 逐元素执行比较运算，不会对输入做数值类型提升转换。

**特性：** `AttrSizedOperandSegments`、`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`
**接口：** `DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**属性：**

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `compare_fn` | `::mlir::hfusion::CompareFnAttr` | 比较函数：veq、vne、vle、vlt、vge、vgt、vule、vult、vuge、vugt |

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `inputs` | 变长任意类型输入 |
| `outputs` | 变长任意带形状输出张量 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `result_tensors` | 变长任意类型定维输出张量 |

### hfusion.cumprod (hfusion::CumprodOp)

**功能：** 在指定维度计算张量累积乘积，reverse控制累积方向，当前仅支持单维度累积。

**语法：**

```mlir
operation ::= `hfusion.cumprod` $input attr-dict `:` type($input) `cum_dims` `=` $cum_dims `reverse` `=` $reverse `->` type($output)
```

**特性：** `AlwaysSpeculatableImplTrait`、`SameOperandsAndResultRank`
**接口：** `ConditionallySpeculatable`、`NoMemoryEffect`
**内存效应：** `MemoryEffects::Effect{}`

**属性：**

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `cum_dims` | `::mlir::DenseI64ArrayAttr` | 累积维度数组，维度序号升序排列 |
| `reverse` | `::mlir::BoolAttr` | 是否反向累积 |

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 定维张量，支持bfloat16、16/32位浮点、8/16/32/64位无符号整数 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 与输入同类型、同秩输出张量 |

### hfusion.cumsum (hfusion::CumsumOp)

**功能：** 在指定维度计算张量累积和，reverse控制累积方向，当前仅支持单维度累积。

**语法：**

```mlir
operation ::= `hfusion.cumsum` $input attr-dict `:` type($input) `cum_dims` `=` $cum_dims `reverse` `=` $reverse `->` type($output)
```

**特性：** `AlwaysSpeculatableImplTrait`、`SameOperandsAndResultRank`
**接口：** `ConditionallySpeculatable`、`NoMemoryEffect`
**内存效应：** `MemoryEffects::Effect{}`

**属性：**

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `cum_dims` | `::mlir::DenseI64ArrayAttr` | 累积维度数组，维度序号升序排列 |
| `reverse` | `::mlir::BoolAttr` | 是否反向累积 |

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 定维张量，支持bfloat16、16/32位浮点、8/16/32/64位无符号整数 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 与输入同类型、同秩输出张量 |

### hfusion.deinterleave (hfusion::DeinterleaveOp)

**功能：** 对输入张量最后一维解交织，拆分两组元素：偶数索引、奇数索引，输入最后一维长度必须为2的倍数。
channelIndex控制输出：-1输出两组、0仅输出偶数通道、1仅输出奇数通道。

**语法：**

```mlir
operation ::= `hfusion.deinterleave` $input custom<HFusionDeinterleave>($channelIndex) attr-dict `:` type($input) `->` type($output)
```

**约束：** 输入张量最后一维尺寸必须为2的倍数。

**特性：** `AlwaysSpeculatableImplTrait`、`Commutative`、`SameOperandsAndResultRank`
**接口：** `ConditionallySpeculatable`、`NoMemoryEffect`、`ReifyRankedShapedTypeOpInterface`
**内存效应：** `MemoryEffects::Effect{}`

**属性：**

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `channelIndex` | `::mlir::IntegerAttr` | 64位无符号整数，通道输出控制标识 |

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 任意类型定维输入张量 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 变长任意类型定维输出张量 |

### hfusion.elemwise_binary (hfusion::ElemwiseBinaryOp)

**功能：** 逐元素二元运算，自动将输入数值提升至输出/累加器数据类型。

**特性：** `AttrSizedOperandSegments`、`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`
**接口：** `DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**属性：**

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `fun` | `::mlir::hfusion::BinaryFnAttr` | 二元运算函数：vor、vand、vxor、minf、maxf、powf、mod、modui、shli、shrsi、shrui、ldexp、ceildivsi、ceildivui、floordivsi、powi、minnumf、maxnumf |
| `cast` | `::mlir::hfusion::TypeFnAttr` | 类型转换规则：cast_signed、cast_unsigned、bitcast |

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `inputs` | 变长任意类型输入 |
| `outputs` | 变长任意带形状输出张量 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `result_tensors` | 变长任意类型定维输出张量 |

### hfusion.elemwise_unary (hfusion::ElemwiseUnaryOp)

**功能：** 逐元素一元运算，自动将输入数值提升至输出/累加器数据类型。

**特性：** `AttrSizedOperandSegments`、`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`
**接口：** `DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**属性：**

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `fun` | `::mlir::hfusion::UnaryFnAttr` | 一元运算函数：relu、sqrt、rsqrt、rec、vnot、tanh、sin、cos、atan、tan、absi、erf、log2、log10、log1p、exp2、expm1、ilogb |
| `cast` | `::mlir::hfusion::TypeFnAttr` | 类型转换规则：cast_signed、cast_unsigned、bitcast |

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `inputs` | 变长任意类型输入 |
| `outputs` | 变长任意带形状输出张量 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `result_tensors` | 变长任意类型定维输出张量 |

### hfusion.flip (hfusion::FlipOp)

**功能：** 沿指定维度翻转张量，当前仅支持最后一维。

**语法：**

```mlir
operation ::= `hfusion.flip` $input attr-dict `:` type($input)
              `flip_axis` `=` $flip_axis
              `->` type($output)
```

**约束：** 仅支持沿张量最后一维执行翻转。

**特性：** `AlwaysSpeculatableImplTrait`、`Commutative`
**接口：** `ConditionallySpeculatable`、`NoMemoryEffect`
**内存效应：** `MemoryEffects::Effect{}`

**属性：**

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `flip_axis` | `::mlir::IntegerAttr` | 64位无符号整数，待翻转维度序号 |

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 任意类型定维输入张量 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 与输入同类型、同秩输出张量 |

### hfusion.gather (hfusion::GatherOp)

**功能：** 沿指定轴从源张量收集元素，非收集维度形状与输入保持一致，对齐triton.language.gather语义。

**特性：** `SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`
**接口：** `BiShengIRAggregatedOpInterface`、`DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**属性：**

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `axis` | `::mlir::IntegerAttr` | 64位无符号整数，收集维度序号 |

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `src` | 任意带形状源张量 |
| `index` | 任意带形状索引张量 |
| `init` | 任意带形状初始化输出张量 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `result` | 变长任意类型输出张量 |

### hfusion.group_matmul (hfusion::GroupMatmulOp)

**功能：** 分组矩阵乘法，用于MoE场景，为每个专家权重与对应Token执行矩阵乘。

**特性：** `AttrSizedOperandSegments`、`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`
**接口：** `DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `inputs` | 变长任意类型输入 |
| `outputs` | 变长任意带形状输出张量 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `result_tensors` | 变长任意类型定维输出张量 |

### hfusion.histogram (hfusion::HistogramOp)

**功能：** 整数张量直方图统计，支持可选掩码，仅统计掩码为true的元素；输出为一维张量，长度等于分箱数量，分箱数为编译期常量。

**语法：**

```mlir
operation ::= `hfusion.histogram` $input `,` $num_bins (`,` $mask^)? attr-dict `:` type($input) (`,` type($mask)^)? `->` type($output)
```

**接口：** `BiShengIRAggregatedOpInterface`

**属性：**

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `num_bins` | `::mlir::IntegerAttr` | 64位无符号整数，直方图分箱总数 |

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 定维整数张量，支持8/16/32/64位无符号整数 |
| `mask` | 可选1比特定维掩码张量 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 一维整数张量，支持32/64位无符号整数 |

### hfusion.interleave (hfusion::InterleaveOp)

**功能：** 沿最后一维交织多个张量元素，当前仅支持2个输入张量；所有输入形状、秩必须完全一致。

**语法：**

```mlir
operation ::= `hfusion.interleave` $input attr-dict `:` type($input) `->` type($output)
```

**约束：** 仅支持2个输入张量，全部输入张量秩、形状必须完全相同。

**特性：** `AlwaysSpeculatableImplTrait`、`Commutative`、`SameOperandsAndResultRank`
**接口：** `ConditionallySpeculatable`、`NoMemoryEffect`、`ReifyRankedShapedTypeOpInterface`
**内存效应：** `MemoryEffects::Effect{}`

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 变长任意类型定维输入张量 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 任意类型定维交织输出张量 |

### hfusion.isfinite (hfusion::IsFiniteOp)

**功能：** 判断浮点张量每个元素是否为有限值（非NaN、非正负无穷）。

**语法：**

```mlir
operation ::= `hfusion.isfinite` $input attr-dict `:` type($input) `->` type($output)
```

**特性：** `AlwaysSpeculatableImplTrait`、`SameOperandsAndResultRank`
**接口：** `BiShengIRAggregatedOpInterface`、`ConditionallySpeculatable`、`NoMemoryEffect`
**内存效应：** `MemoryEffects::Effect{}`

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 定维浮点张量，支持bfloat16、16/32位浮点 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 1比特定维布尔张量 |

### hfusion.isinf (hfusion::IsInfOp)

**功能：** 判断浮点张量每个元素是否为正负无穷。

**语法：**

```mlir
operation ::= `hfusion.isinf` $input attr-dict `:` type($input) `->` type($output)
```

**特性：** `AlwaysSpeculatableImplTrait`、`SameOperandsAndResultRank`
**接口：** `ConditionallySpeculatable`、`NoMemoryEffect`
**内存效应：** `MemoryEffects::Effect{}`

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 定维浮点张量，支持bfloat16、16/32位浮点 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 1比特定维布尔张量 |

### hfusion.isnan (hfusion::IsNanOp)

**功能：** 判断浮点张量每个元素是否为NaN。

**语法：**

```mlir
operation ::= `hfusion.isnan` $input attr-dict `:` type($input) `->` type($output)
```

**特性：** `AlwaysSpeculatableImplTrait`、`SameOperandsAndResultRank`
**接口：** `ConditionallySpeculatable`、`NoMemoryEffect`
**内存效应：** `MemoryEffects::Effect{}`

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `input` | 定维浮点张量，支持bfloat16、16/32位浮点 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `output` | 1比特定维布尔张量 |

### hfusion.load (hfusion::LoadOp)

**功能：** 逐元素读取张量数据，不执行数值类型转换。

**特性：** `AttrSizedOperandSegments`、`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`
**接口：** `DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `inputs` | 变长任意类型输入 |
| `outputs` | 变长任意带形状输出张量 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `result_tensors` | 变长任意类型定维输出张量 |

### hfusion.mulext (hfusion::MulExtOp)

**功能：** 有符号整数扩展乘法，输入N位整数，输出两组N位结果：乘积低半段、乘积高半段；低半段等价普通乘法结果。

**语法：**

```mlir
operation ::= `hfusion.mulext` $lhs `,` $rhs attr-dict `:` type($lhs)
```

**特性：** `AlwaysSpeculatableImplTrait`、`Commutative`
**接口：** `ConditionallySpeculatable`、`InferTypeOpInterface`、`NoMemoryEffect`
**内存效应：** `MemoryEffects::Effect{}`

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `lhs` | signless-integer-like类型 |
| `rhs` | signless-integer-like类型 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `low` | 乘积低半段，signless-integer-like类型 |
| `high` | 乘积高半段，signless-integer-like类型 |

### hfusion.print (hfusion::PrintOp)

**功能：** 设备端调试打印，接收前缀字符串与标量/张量，支持十六进制输出开关。

**语法：**

```mlir
operation ::= `hfusion.print` $prefix attr-dict $arg `:` type($arg)
```

**属性：**

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `prefix` | `::mlir::StringAttr` | 打印前缀文本 |
| `hex` | `::mlir::BoolAttr` | 是否以十六进制打印 |

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `arg` | 整数、浮点或任意类型定维张量 |

### hfusion.reduce_with_index (hfusion::ReduceWithIndexOp)

**功能：** 带索引的最大/最小值规约运算，仅支持单规约维度。两种使用模式：输入+索引张量输出结果与索引；仅输入张量自动生成索引。tie_break_left控制相等值取最左/最右索引。

**特性：** `AttrSizedOperandSegments`、`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`
**接口：** `DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**属性：**

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `reduce_kind` | `::mlir::hfusion::ReduceWithIndexKindAttr` | 规约类型（最大/最小，含无符号分支） |
| `tie_break_left` | `::mlir::BoolAttr` | 等值时是否取左侧索引 |
| `dimensions` | `::mlir::DenseI64ArrayAttr` | 规约维度数组，升序排列 |

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `inputs` | 变长任意带形状输入张量 |
| `inits` | 变长任意带形状初始化张量 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `result` | 变长任意类型输出张量 |

### hfusion.select (hfusion::SelectOp)

**功能：** 根据首个二元条件操作数选择对应数值。

**特性：** `AttrSizedOperandSegments`、`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`
**接口：** `DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `inputs` | 变长任意类型输入 |
| `outputs` | 变长任意带形状输出张量 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `result_tensors` | 变长任意类型定维输出张量 |

### hfusion.sort (hfusion::SortOp)

**功能：** 沿指定轴排序张量，输出排序数值与对应索引。

**约束：**

1. 输入向量和输出向量必须具有相同的秩。
2. 当前仅支持尾部轴排序。

**参数说明：**

- `src`：待排序的张量/memref
- `dst_value`：用于存储排序后值的张量/memref
- `dst_index`：用于存储与dst_value对应的索引的张量/memref
- `descending`：决定按升序还是降序排序。默认为false，即升序
- `sort_axis`：待排序的轴

**语法：**

```mlir
operation ::= `hfusion.sort` attr-dict `ins` `(` $src `:` type($src) `)`
              `descending` `=` $descending
              `sort_axis` `=` $sort_axis
              (`->` type($result)^)?
```

**示例：**

```mlir
%result = hfusion.sort ins(%src : tensor<?xf32>) descending = true sort_axis = 0 -> tensor<?xf32>
```

**特性：** `SameOperandsAndResultRank`

**属性：**

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `descending` | `::mlir::BoolAttr` | 是否降序排列 |
| `sort_axis` | `::mlir::IntegerAttr` | 64位无符号整数，排序维度序号 |

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `src` | 待排序张量/内存视图 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `result` | 变长任意类型定维输出张量 |

### hfusion.store (hfusion::StoreOp)

**功能：** 逐元素存储张量数据，不执行数值类型转换，支持原子写入模式。

**特性：** `AttrSizedOperandSegments`、`SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`、`SingleBlock`
**接口：** `DestinationStyleOpInterface`、`LinalgStructuredInterface`、`MemoryEffectOpInterface`、`ReifyRankedShapedTypeOpInterface`

**属性：**

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `atomic_kind` | `::mlir::hfusion::AtomicKindAttr` | 原子操作类型，可选值：none、add、max、min、and、or、xor、cas、xchg、umax、umin |

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `inputs` | 变长任意类型输入 |
| `outputs` | 变长任意带形状输出张量 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `result_tensors` | 变长任意类型定维输出张量 |

### hfusion.symbolic_dim (hfusion::SymbolicDimOp)

**功能：** 通过符号名称引用符号维度，返回index类型数值。

**语法：**

```mlir
operation ::= `hfusion.symbolic_dim` $symbolName attr-dict `:` type($result)
```

**特性：** `AlwaysSpeculatableImplTrait`
**接口：** `ConditionallySpeculatable`、`InferTypeOpInterface`、`NoMemoryEffect`
**内存效应：** `MemoryEffects::Effect{}`

**属性：**

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `symbolName` | `::mlir::SymbolRefAttr` | 符号维度引用 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `result` | index类型维度值 |

## 属性

### AtomicKindAttr

**语法：**

```mlir
#hfusion.atomic_kind<
  ::mlir::hfusion::AtomicKind   # value
>
```

**功能：** 原子操作类型属性，32位无符号整数取值范围0~10。
枚举取值：none、add、max、min、and、or、xor、cas、xchg、umax、umin

**参数：**

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hfusion::AtomicKind` | AtomicKind枚举值 |

### BinaryFnAttr

**语法：**

```mlir
#hfusion.binary_fn<
  ::mlir::hfusion::BinaryFn   # value
>
```

**功能：** 二元逐元素运算函数属性，32位无符号整数取值范围0~17。
枚举取值：vor、vand、vxor、minf、maxf、powf、mod、modui、shli、shrsi、shrui、ldexp、ceildivsi、ceildivui、floordivsi、powi、minnumf、maxnumf

**参数：**

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hfusion::BinaryFn` | BinaryFn枚举值 |

### CompareFnAttr

**语法：**

```mlir
#hfusion.compare_fn<
  ::mlir::hfusion::CompareFn   # value
>
```

**功能：** 逐元素比较函数属性，32位无符号整数取值范围0~9。
枚举取值：veq、vne、vle、vlt、vge、vgt、vule、vult、vuge、vugt

**参数：**

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hfusion::CompareFn` | CompareFn枚举值 |

### BindSubBlockAttr

**语法：** `#hfusion.bind_sub_block`
**功能：** 标记绑定子块专用操作。

### FusionKindAttr

**语法：**

```mlir
#hfusion.fusion_kind<
  ::mlir::hfusion::FusionKind   # fusion_kind
>
```

**功能：** HFusion融合内核类型标识属性。

**参数：**

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| fusion_kind | `::mlir::hfusion::FusionKind` | FusionKind枚举值 |

### InsertSliceSourceIndexAttr

**语法：** `#hfusion.insert_slice_source_index`
**功能：** 标记concat操作中作为insert_slice输入的操作数序号。

### MultiBufferAttr

**语法：** `#hfusion.multi_buffer`
**功能：** 目标操作多缓冲配置属性。

### ReduceComposeAttr

**语法：** `#hfusion.reduce_composed`
**功能：** 标记组合式规约运算。

### ReduceWithIndexKindAttr

**语法：**

```mlir
#hfusion.reduce_with_index_kind<
  ::mlir::hfusion::ReduceWithIndexKind   # reduce_with_index_kind
>
```

**功能：** 带索引规约运算类型属性。

**参数：**

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| reduce_with_index_kind | `::mlir::hfusion::ReduceWithIndexKind` | ReduceWithIndexKind枚举值 |

### ReturnOperandNumAttr

**语法：** `#hfusion.return_operand_num`
**功能：** 指定当前参数对应函数返回值下标。

### StrideAlignDimsAttr

**语法：** `#hfusion.stride_align_dims`
**功能：** 标记需要步长对齐的维度。

### StrideAlignValueInByteAttr

**语法：** `#hfusion.stride_align_value_in_byte`
**功能：** 步长对齐字节数值配置属性。

### RoundModeAttr

**语法：**

```mlir
#hfusion.round_mode<
  ::mlir::hfusion::RoundMode   # value
>
```

**功能：** 数值转换舍入模式属性，32位无符号整数取值范围0~6。
枚举说明：

- RINT：四舍五入到最近偶数
- ROUND：四舍五入远离零
- FLOOR：向负无穷取整
- CEIL：向正无穷取整
- TRUNC：向零截断
- ODD：冯·诺依曼奇数舍入

**参数：**

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hfusion::RoundMode` | RoundMode枚举值 |

### TernaryFnAttr

**语法：**

```mlir
#hfusion.ternary_fn<
  ::mlir::hfusion::TernaryFn   # value
>
```

**功能：** 三元运算函数属性，仅支持select。

**参数：**

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hfusion::TernaryFn` | TernaryFn枚举值 |

### TypeFnAttr

**语法：**

```mlir
#hfusion.type_fn<
  ::mlir::hfusion::TypeFn   # value
>
```

**功能：** 类型转换规则属性，32位无符号整数取值0~2。
枚举取值：cast_signed、cast_unsigned、bitcast

**参数：**

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hfusion::TypeFn` | TypeFn枚举值 |

### UnaryFnAttr

**语法：**

```mlir
#hfusion.unary_fn<
  ::mlir::hfusion::UnaryFn   # value
>
```

**功能：** 一元逐元素运算函数属性，32位无符号整数取值0~17。
枚举取值：relu、sqrt、rsqrt、rec、vnot、tanh、sin、cos、atan、tan、absi、erf、log2、log10、log1p、exp2、expm1、ilogb

**参数：**

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hfusion::UnaryFn` | UnaryFn枚举值 |

## 枚举

### AtomicKind

**取值范围：** 32位无符号整数0~10

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| NONE | `0` | none |
| ADD | `1` | add |
| MAX | `2` | max |
| MIN | `3` | min |
| AND | `4` | and |
| OR | `5` | or |
| XOR | `6` | xor |
| CAS | `7` | cas |
| XCHG | `8` | xchg |
| UMAX | `9` | umax |
| UMIN | `10` | umin |

### BinaryFn

**取值范围：** 32位无符号整数0~17

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| vor | `0` | vor |
| vand | `1` | vand |
| vxor | `2` | vxor |
| minf | `3` | minf |
| maxf | `4` | maxf |
| powf | `5` | powf |
| mod | `6` | mod |
| modui | `7` | modui |
| shli | `8` | shli |
| shrsi | `9` | shrsi |
| shrui | `10` | shrui |
| ldexp | `11` | ldexp |
| ceildivsi | `12` | ceildivsi |
| ceildivui | `13` | ceildivui |
| floordivsi | `14` | floordivsi |
| powi | `15` | powi |
| minnumf | `16` | minnumf |
| maxnumf | `17` | maxnumf |

### CastMode

**取值范围：** 32位无符号整数0~8

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| F32TOI8 | `0` | F32TOI8 |
| F32TOI16 | `1` | F32TOI16 |
| F16TOI8 | `2` | F16TOI8 |
| I64TOI32 | `3` | I64TOI32 |
| I64TOI16 | `4` | I64TOI16 |
| I64TOI8 | `5` | I64TOI8 |
| I32TOI16 | `6` | I32TOI16 |
| I32TOI8 | `7` | I32TOI8 |
| I16TOI8 | `8` | I16TOI8 |

### CompareFn

**取值范围：** 32位无符号整数0~9

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| veq | `0` | veq |
| vne | `1` | vne |
| vle | `2` | vle |
| vlt | `3` | vlt |
| vge | `4` | vge |
| vgt | `5` | vgt |
| vule | `6` | vule |
| vult | `7` | vult |
| vuge | `8` | vuge |
| vugt | `9` | vugt |

### FlattenMode

**功能：** HFusion张量扁平化模式

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| Greedy | `1` | Greedy |
| Tidy | `2` | Tidy |

### FusionKind

**功能：** HFusion融合内核分类

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| PureElemwise | `1` | PURE_ELEMWISE |
| AnyPB | `2` | ANY_PB |
| LastAxisPBR | `3` | LAST_AXIS_PBR |
| AnyPBR | `4` | ANY_PBR |
| SingleCube | `5` | SINGLE_CUBE |
| ShallowCV | `6` | SHALLOW_CV |
| ShallowVV | `7` | SHALLOW_VV |
| MixCV | `8` | MIX_CV |
| MixC2 | `9` | MIX_C2 |
| Unknown | `10` | UNKNOWN |

### OutputMode

**功能：** HFusion输出布局模式

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| Multiple | `1` | Multiple |
| Single | `2` | Single |
| SingleAggressive | `3` | SingleAggressive |

### CumOpType

**功能：** 累积运算类型

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| UNDEFINED | `0` | undefined |
| CUMSUM | `1` | cumsum |
| CUMPROD | `2` | cumprod |

### MmMapMode

**取值范围：** 32位无符号整数0~1

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| CoreOp | `0` | core_op |
| MacroInstr | `1` | macro_instr |

### ReduceWithIndexKind

**取值范围：** 32位无符号整数0~3

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| MIN | `0` | min |
| MAX | `1` | max |
| MINUI | `2` | minui |
| MAXUI | `3` | maxui |

### RoundMode

**取值范围：** 32位无符号整数0~6

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| RINT | `0` | rint |
| ROUND | `1` | round |
| FLOOR | `2` | floor |
| CEIL | `3` | ceil |
| TRUNC | `4` | trunc |
| ODD | `5` | odd |
| TRUNCWITHOVERFLOW | `6` | truncwithoverflow |

### TaylorMode

**取值范围：** 32位无符号整数0~1

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| SIN | `0` | sin |
| ATAN | `1` | atan |

### TernaryFn

**取值范围：** 32位无符号整数仅0

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| select | `0` | select |

### TypeFn

**取值范围：** 32位无符号整数0~2

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| cast_signed | `0` | cast_signed |
| cast_unsigned | `1` | cast_unsigned |
| bitcast | `2` | bitcast |

### UnaryFn

**取值范围：** 32位无符号整数0~17

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| relu | `0` | relu |
| sqrt | `1` | sqrt |
| rsqrt | `2` | rsqrt |
| rec | `3` | rec |
| vnot | `4` | vnot |
| tanh | `5` | tanh |
| sin | `6` | sin |
| cos | `7` | cos |
| atan | `8` | atan |
| tan | `9` | tan |
| absi | `10` | absi |
| erf | `11` | erf |
| log2 | `12` | log2 |
| log10 | `13` | log10 |
| log1p | `14` | log1p |
| exp2 | `15` | exp2 |
| expm1 | `16` | expm1 |
| ilogb | `17` | ilogb |
