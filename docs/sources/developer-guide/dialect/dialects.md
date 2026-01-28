# 'hfusion' 方言

多维融合（HFusion）方言。


## Operations

### `hfusion.arange` (hfusion::ArangeOp)

与经典的 arange 定义略有不同，增加了偏移量（默认为 0）和多维支持（这也引入了步长）。偏移量和步长的定义类似于 memref 描述符。

给定一个 3D arange 操作，每个位置的值将是：
`arange[i, j, k] = offset + stride[0] * i + stride[1] * j + stride[2] * k`

Traits: `AttrSizedOperandSegments`, `SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`, `SingleBlock`

Interfaces: `DestinationStyleOpInterface`, `LinalgStructuredInterface`, `MemoryEffectOpInterface`, `ReifyRankedShapedTypeOpInterface`

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `offset` | 索引
| `strides` | 可变数量的索引
| `init` | 任意类型的形状值

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result_tensor` | 任意类型的形状值


### `hfusion.assert` (hfusion::AssertOp)

_用于调试的设备端断言_

语法：

```
operation ::= `hfusion.assert` $msg attr-dict $cond `:` type($cond)
```

`hfusion.assert` 接受一个字符串字面量 `msg` 和一个应该被断言的标量或张量参数。

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>msg</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `cond` | 整数或任意类型的张量


### `hfusion.atomic_cas` (hfusion::AtomicCasOp)

_原子比较并交换（CAS）操作_

语法：

```
operation ::= `hfusion.atomic_cas` attr-dict `ins` `(` $input `:` type($input) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`->` type($output)^)?
```

比较并交换（CAS）是一个原子操作，由三个操作数组成：
内存位置（V）、期望的旧值（A）、新值（B）。
操作的语义是：只有当内存位置 V 的值等于期望的旧值 A 时，V 的值才会更新为 B。
无论是否更新，操作都会返回 V 的原始值。

约束：
  1. 输入和输出必须具有相同的秩和相同的元素类型。

参数：
  * `src0`: 期望的旧值
  * `src1`: 新值
  * `dst`: GM 中的内存位置

示例：
```mlir
hfusion.atomic_cas ins(%src0, %src1 : memref<?xf32>, memref<?xf32>) outs(%dst : memref<?xf32>)
%result = hfusion.atomic_cas ins(%src0, %src1 : tensor<?xf32>, tensor<?xf32>) outs(%dst : tensor<?xf32>) -> tensor<?xf32>
```

Traits: `SameOperandsAndResultRank`

Interfaces: `MemoryEffectOpInterface`

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `input` | 可变数量的张量或内存引用
| `dst` | 张量或内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `output` | 张量或内存引用


### `hfusion.atomic_rmw` (hfusion::AtomicRMWOp)

_原子 RMW 操作_

语法：

```
operation ::= `hfusion.atomic_rmw` attr-dict `ins` `(` $input `:` type($input) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `atomic_kind` `=` $atomic_kind
              (`->` type($output)^)?
```

原子 RMW 是一个由三个步骤组成的原子操作：
1. 读取指定内存地址的当前值
2. 根据 atomic_kind 属性执行操作
3. 返回之前读取的旧值
整个过程是原子的，即在操作期间不会被其他线程中断。

约束：
  1. 输入 memref 和输出 memref 必须具有相同的秩和相同的元素类型。

参数：
  * `src`: 新值
  * `dst`: GM 中的内存位置

示例：
```mlir
hfusion.atomic_rmw ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) atomic_kind = <add>
%result = hfusion.atomic_rmw ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) atomic_kind = <or> -> tensor<?xf32>
```

Traits: `SameOperandsAndResultRank`

Interfaces: `MemoryEffectOpInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>atomic_kind</code></td><td>::mlir::hfusion::AtomicKindAttr</td><td><details><summary>allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10</summary>{{% markdown %}}Enum cases:
* none (`NONE`)
* add (`ADD`)
* max (`MAX`)
* min (`MIN`)
* and (`AND`)
* or (`OR`)
* xor (`XOR`)
* cas (`CAS`)
* xchg (`XCHG`)
* umax (`UMAX`)
* umin (`UMIN`){{% /markdown %}}</details></td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `input` | 张量或内存引用
| `dst` | 张量或内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `output` | 张量或内存引用


### `hfusion.atomic_xchg` (hfusion::AtomicXchgOp)

_原子交换操作_

语法：

```
operation ::= `hfusion.atomic_xchg` attr-dict `ins` `(` $input `:` type($input) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`mask` `(` $mask^ `:` type($mask) `)`)?
              (`->` type($output)^)?
```

原子交换是一个由三个步骤组成的原子操作：
1. 读取指定内存地址的当前值
2. 将新值写入内存地址
3. 返回之前读取的旧值
整个过程是原子的，即在操作期间不会被其他线程中断。

约束：
  1. 输入 memref 和输出 memref 必须具有相同的秩和相同的元素类型。

参数：
  * `src`: 新值
  * `dst`: GM 中的内存位置
  * `mask` : 掩码元素

示例：
```mlir
hfusion.atomic_xchg ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) mask(%m : memref<?xi1>)
%result = hfusion.atomic_xchg ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) mask(%m : memref<?xi1>) -> tensor<?xf32>
```

Traits: `SameOperandsAndResultRank`

Interfaces: `MemoryEffectOpInterface`

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `input` | 张量或内存引用
| `dst` | 张量或内存引用
| `mask` | 张量或内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `output` | 张量或内存引用


### `hfusion.barrier` (hfusion::BarrierOp)

_同步核心的所有流水线_

语法：

```
operation ::= `hfusion.barrier` attr-dict
```

"barrier" 操作同步核心的所有流水线。


### `hfusion.bitcast` (hfusion::BitcastOp)

_逐元素应用 bitcast 函数_

Traits: `AttrSizedOperandSegments`, `SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`, `SingleBlock`

Interfaces: `DestinationStyleOpInterface`, `LinalgStructuredInterface`, `MemoryEffectOpInterface`, `ReifyRankedShapedTypeOpInterface`

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `inputs` | 可变数量的任意类型
| `outputs` | 可变数量的任意类型形状值

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result_tensors` | 可变数量的任意类型张量


### `hfusion.cast` (hfusion::CastOp)

_逐元素应用 cast 函数_

Traits: `AttrSizedOperandSegments`, `SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`, `SingleBlock`

Interfaces: `DestinationStyleOpInterface`, `LinalgStructuredInterface`, `MemoryEffectOpInterface`, `ReifyRankedShapedTypeOpInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>round_mode</code></td><td>::mlir::hfusion::RoundModeAttr</td><td><details><summary>allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6</summary>{{% markdown %}}
    - RINT: round to nearest, tie to even (c language rint)
    - ROUND: round to nearest, tie away from zero (c language round)
    - FLOOR: round to minus infinity (c language floor)
    - CEIL: round to positive infinity (c language ceil)
    - TRUNC: round to zero (c language trunc)
    - ODD: round to odd (Von Neumann rounding)
  {{% /markdown %}}</details></td></tr>
<tr><td><code>enable_overflow</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>cast</code></td><td>::mlir::hfusion::TypeFnAttr</td><td><details><summary>allowed 32-bit signless integer cases: 0, 1, 2</summary>{{% markdown %}}Enum cases:
* cast_signed (`cast_signed`)
* cast_unsigned (`cast_unsigned`)
* bitcast (`bitcast`){{% /markdown %}}</details></td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `inputs` | 可变数量的任意类型
| `outputs` | 可变数量的任意类型形状值

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result_tensors` | 可变数量的任意类型张量


### `hfusion.compare` (hfusion::CompareOp)

_逐元素应用比较函数 fun_

不对输入操作数执行数值类型转换。
Traits: `AttrSizedOperandSegments`, `SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`, `SingleBlock`

Interfaces: `DestinationStyleOpInterface`, `LinalgStructuredInterface`, `MemoryEffectOpInterface`, `ReifyRankedShapedTypeOpInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>compare_fn</code></td><td>::mlir::hfusion::CompareFnAttr</td><td><details><summary>allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9</summary>{{% markdown %}}Enum cases:
* veq (`veq`)
* vne (`vne`)
* vle (`vle`)
* vlt (`vlt`)
* vge (`vge`)
* vgt (`vgt`)
* vule (`vule`)
* vult (`vult`)
* vuge (`vuge`)
* vugt (`vugt`){{% /markdown %}}</details></td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `inputs` | 可变数量的任意类型
| `outputs` | 可变数量的任意类型形状值

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result_tensors` | 可变数量的任意类型张量


### `hfusion.cumprod` (hfusion::CumprodOp)

_计算输入张量某个维度上的累积乘积_

语法：

```
operation ::= `hfusion.cumprod` $input attr-dict `:` type($input) `cum_dims` `=` $cum_dims `reverse` `=` $reverse `->` type($output)
```

cumsum 操作计算输入张量某个维度上的累积乘积。reverse 参数指定累积乘积的方向。
目前仅支持一个累积维度。

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultRank`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>cum_dims</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute should be in increasing order</td></tr>
<tr><td><code>reverse</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `input` | bfloat16 类型或 16 位浮点数或 32 位浮点数或 8 位无符号整数或 16 位无符号整数或 32 位无符号整数或 64 位无符号整数的张量

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `output` | bfloat16 类型或 16 位浮点数或 32 位浮点数或 8 位无符号整数或 16 位无符号整数或 32 位无符号整数或 64 位无符号整数的张量


### `hfusion.cumsum` (hfusion::CumsumOp)

_计算输入张量某个维度上的累积和_

语法：

```
operation ::= `hfusion.cumsum` $input attr-dict `:` type($input) `cum_dims` `=` $cum_dims `reverse` `=` $reverse `->` type($output)
```

cumsum 操作计算输入张量某个维度上的累积和。reverse 参数指定累积和的方向。
目前仅支持一个累积维度。

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultRank`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>cum_dims</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute should be in increasing order</td></tr>
<tr><td><code>reverse</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `input` | bfloat16 类型或 16 位浮点数或 32 位浮点数或 8 位无符号整数或 16 位无符号整数或 32 位无符号整数或 64 位无符号整数的张量

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `output` | bfloat16 类型或 16 位浮点数或 32 位浮点数或 8 位无符号整数或 16 位无符号整数或 32 位无符号整数或 64 位无符号整数的张量


### `hfusion.deinterleave` (hfusion::DeinterleaveOp)

_通过解交错输入张量构造两个张量_

语法：

```
operation ::= `hfusion.deinterleave` $input custom<HFusionDeinterleave>($channelIndex) attr-dict `:` type($input) `->` type($output)
```

解交错操作从单个输入张量构造两个张量。第一个结果张量包含偶数索引的元素，第二个包含奇数索引的元素（两个索引都指最后一个维度的索引）。它约束输入的最后一个维度大小必须是 2 的倍数。

channelIndex 属性控制输出行为：
* -1: 输出所有通道（返回两个张量，一个包含偶数索引，一个包含奇数索引）
* 0: 仅输出通道 0（偶数索引）
* 1: 仅输出通道 1（奇数索引）

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultRank`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ReifyRankedShapedTypeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>channelIndex</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `input` | 任意类型的张量

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `output` | 可变数量的任意类型张量


### `hfusion.elemwise_binary` (hfusion::ElemwiseBinaryOp)

_逐元素应用二元函数 fun_

对输入操作数执行数值类型转换，将其提升为与累加器/输出相同的数据类型。
Traits: `AttrSizedOperandSegments`, `SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`, `SingleBlock`

Interfaces: `DestinationStyleOpInterface`, `LinalgStructuredInterface`, `MemoryEffectOpInterface`, `ReifyRankedShapedTypeOpInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>fun</code></td><td>::mlir::hfusion::BinaryFnAttr</td><td><details><summary>allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17</summary>{{% markdown %}}Enum cases:
* vor (`vor`)
* vand (`vand`)
* vxor (`vxor`)
* minf (`minf`)
* maxf (`maxf`)
* powf (`powf`)
* mod (`mod`)
* modui (`modui`)
* shli (`shli`)
* shrsi (`shrsi`)
* shrui (`shrui`)
* ldexp (`ldexp`)
* ceildivsi (`ceildivsi`)
* ceildivui (`ceildivui`)
* floordivsi (`floordivsi`)
* powi (`powi`)
* minnumf (`minnumf`)
* maxnumf (`maxnumf`){{% /markdown %}}</details></td></tr>
<tr><td><code>cast</code></td><td>::mlir::hfusion::TypeFnAttr</td><td><details><summary>allowed 32-bit signless integer cases: 0, 1, 2</summary>{{% markdown %}}Enum cases:
* cast_signed (`cast_signed`)
* cast_unsigned (`cast_unsigned`)
* bitcast (`bitcast`){{% /markdown %}}</details></td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `inputs` | 可变数量的任意类型
| `outputs` | 可变数量的任意类型形状值

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result_tensors` | 可变数量的任意类型张量


### `hfusion.elemwise_unary` (hfusion::ElemwiseUnaryOp)

_逐元素应用一元函数 fun_

对输入操作数执行数值类型转换，将其提升为与累加器/输出相同的数据类型。
Traits: `AttrSizedOperandSegments`, `SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`, `SingleBlock`

Interfaces: `DestinationStyleOpInterface`, `LinalgStructuredInterface`, `MemoryEffectOpInterface`, `ReifyRankedShapedTypeOpInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>fun</code></td><td>::mlir::hfusion::UnaryFnAttr</td><td><details><summary>allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17</summary>{{% markdown %}}Enum cases:
* relu (`relu`)
* sqrt (`sqrt`)
* rsqrt (`rsqrt`)
* rec (`rec`)
* vnot (`vnot`)
* tanh (`tanh`)
* sin (`sin`)
* cos (`cos`)
* atan (`atan`)
* tan (`tan`)
* absi (`absi`)
* erf (`erf`)
* log2 (`log2`)
* log10 (`log10`)
* log1p (`log1p`)
* exp2 (`exp2`)
* expm1 (`expm1`)
* ilogb (`ilogb`){{% /markdown %}}</details></td></tr>
<tr><td><code>cast</code></td><td>::mlir::hfusion::TypeFnAttr</td><td><details><summary>allowed 32-bit signless integer cases: 0, 1, 2</summary>{{% markdown %}}Enum cases:
* cast_signed (`cast_signed`)
* cast_unsigned (`cast_unsigned`)
* bitcast (`bitcast`){{% /markdown %}}</details></td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `inputs` | 可变数量的任意类型
| `outputs` | 可变数量的任意类型形状值

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result_tensors` | 可变数量的任意类型张量


### `hfusion.flip` (hfusion::FlipOp)

_沿维度 dim 翻转张量 x_

语法：

```
operation ::= `hfusion.flip` $input attr-dict `:` type($input)
              `flip_axis` `=` $flip_axis
              `->` type($output)
```

沿维度 dim 翻转张量 x。
目前仅支持最后一个维度。

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>flip_axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `input` | 任意类型的张量

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `output` | 任意类型的张量


### `hfusion.gather` (hfusion::GatherOp)

将 src 张量的一个轴收集到不同的形状中，除了收集轴外其他维度形状相同。对应于 triton.language.gather。

给定 src:tensor<16x16> 和 index:tensor<16x4>，axis = 1，该操作等价于：
```
for i in 0 to 16 {
  for j in 0 to 4 {       // Can be tiled without consequence
    for k in 0 to 16 {    // Cannot be tiled without result potentially
                          //   becoming partial, define as gather axis
      output[i][j] = (index[i][j] == k) ? src[i][k] : output[i][j];
    }
  }
}
```

Traits: `SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`, `SingleBlock`

Interfaces: `BiShengIRAggregatedOpInterface`, `DestinationStyleOpInterface`, `LinalgStructuredInterface`, `MemoryEffectOpInterface`, `ReifyRankedShapedTypeOpInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 任意类型的形状值
| `index` | 任意类型的形状值
| `init` | 任意类型的形状值

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hfusion.group_matmul` (hfusion::GroupMatmulOp)

_在专家权重和 token 嵌入之间执行分组矩阵乘法。
对于每个专家，将其权重矩阵与其分配的 tokens 相乘。_

Traits: `AttrSizedOperandSegments`, `SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`, `SingleBlock`

Interfaces: `DestinationStyleOpInterface`, `LinalgStructuredInterface`, `MemoryEffectOpInterface`, `ReifyRankedShapedTypeOpInterface`

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `inputs` | 可变数量的任意类型
| `outputs` | 可变数量的任意类型形状值

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result_tensors` | 可变数量的任意类型张量


### `hfusion.histogram` (hfusion::HistogramOp)

_计算整数张量的直方图，支持可选的掩码_

语法：

```
operation ::= `hfusion.histogram` $input `,` $num_bins (`,` $mask^)? attr-dict `:` type($input) (`,` type($mask)^)? `->` type($output)
```

对于输入张量的每个元素，递增输出直方图中对应的 bin。bin 的数量是编译时常量（`I64Attr`），输出必须是长度为 num_bins 的一维张量。
如果提供了掩码张量，则只计算 mask[i] = true 的元素。

Interfaces: `BiShengIRAggregatedOpInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>num_bins</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `input` | 8 位无符号整数或 16 位无符号整数或 32 位无符号整数或 64 位无符号整数的张量
| `mask` | 1 位无符号整数的张量

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `output` | 32 位无符号整数或 64 位无符号整数的张量


### `hfusion.interleave` (hfusion::InterleaveOp)

_通过交错 n 个输入张量构造一个张量。
      目前仅支持 n = 2。_

语法：

```
operation ::= `hfusion.interleave` $input attr-dict `:` type($input) `->` type($output)
```

沿最后一个维度交错 n 个张量的值。
N 个张量必须具有相同的形状。
输入张量和输出张量必须具有相同的秩。

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultRank`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ReifyRankedShapedTypeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `input` | 可变数量的任意类型张量

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `output` | 任意类型的张量


### `hfusion.isfinite` (hfusion::IsFiniteOp)

_计算浮点类型张量的元素是否为有限值_

语法：

```
operation ::= `hfusion.isfinite` $input attr-dict `:` type($input) `->` type($output)
```

计算浮点类型张量的元素是否为有限值（即不是 NaN 或无穷大）。

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultRank`

Interfaces: `BiShengIRAggregatedOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `input` | bfloat16 类型或 16 位浮点数或 32 位浮点数的张量

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `output` | 1 位无符号整数的张量


### `hfusion.isinf` (hfusion::IsInfOp)

_计算浮点类型张量的元素是否等于正无穷或负无穷_

语法：

```
operation ::= `hfusion.isinf` $input attr-dict `:` type($input) `->` type($output)
```

计算浮点类型张量的元素是否为无穷大。
正无穷和负无穷都可以。

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultRank`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `input` | bfloat16 类型或 16 位浮点数或 32 位浮点数的张量

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `output` | 1 位无符号整数的张量


### `hfusion.isnan` (hfusion::IsNanOp)

_计算浮点类型张量的元素是否为 NAN_

语法：

```
operation ::= `hfusion.isnan` $input attr-dict `:` type($input) `->` type($output)
```

计算浮点类型张量的元素是否为 NAN。

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultRank`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `input` | bfloat16 类型或 16 位浮点数或 32 位浮点数的张量

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `output` | 1 位无符号整数的张量


### `hfusion.load` (hfusion::LoadOp)

_逐元素加载张量_

不对输入操作数执行数值类型转换。
Traits: `AttrSizedOperandSegments`, `SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`, `SingleBlock`

Interfaces: `DestinationStyleOpInterface`, `LinalgStructuredInterface`, `MemoryEffectOpInterface`, `ReifyRankedShapedTypeOpInterface`

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `inputs` | 可变数量的任意类型
| `outputs` | 可变数量的任意类型形状值

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result_tensors` | 可变数量的任意类型张量


### `hfusion.mulext` (hfusion::MulExtOp)

_扩展有符号整数乘法操作_

语法：

```
operation ::= `hfusion.mulext` $lhs `,` $rhs attr-dict `:` type($lhs)
```

对符号扩展的操作数执行 (2*N) 位乘法。返回两个 N 位结果：乘积的低半部分和高半部分。低半部分的值与使用相同操作数的常规乘法的结果相同。

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `lhs` | signless-integer-like
| `rhs` | signless-integer-like

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `low` | signless-integer-like
| `high` | signless-integer-like


### `hfusion.print` (hfusion::PrintOp)

_用于调试的设备端打印_

语法：

```
operation ::= `hfusion.print` $prefix attr-dict $arg `:` type($arg)
```

`hfusion.print` 接受一个字符串字面量 `prefix` 和一个应该被打印的标量或张量参数。可选参数 `hex` 配置是否以十六进制格式打印。

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>prefix</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>hex</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `arg` | 整数或浮点数或任意类型的张量


### `hfusion.reduce_with_index` (hfusion::ReduceWithIndexOp)

_带索引的最大/最小归约_

使用 max/min 对 AnyShaped 执行归约操作。
它支持两种模式：(1) 接受输入（AnyShaped）和索引（AnyShaped），产生结果输入和索引；(2) 接受输入，产生结果输入和索引。
目前仅支持一个归约维度。
`tie_break_left` 显示是查找最左侧索引还是最右侧索引。

Traits: `AttrSizedOperandSegments`, `SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`, `SingleBlock`

Interfaces: `DestinationStyleOpInterface`, `LinalgStructuredInterface`, `MemoryEffectOpInterface`, `ReifyRankedShapedTypeOpInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>reduce_kind</code></td><td>::mlir::hfusion::ReduceWithIndexKindAttr</td><td><details><summary></summary>{{% markdown %}}
    The kind of reduce with index.
  {{% /markdown %}}</details></td></tr>
<tr><td><code>tie_break_left</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>dimensions</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute should be in increasing order</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `inputs` | 可变数量的任意类型形状值
| `inits` | 可变数量的任意类型形状值

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hfusion.select` (hfusion::SelectOp)

_根据作为第一个操作数提供的二元条件选择一个值_

Traits: `AttrSizedOperandSegments`, `SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`, `SingleBlock`

Interfaces: `DestinationStyleOpInterface`, `LinalgStructuredInterface`, `MemoryEffectOpInterface`, `ReifyRankedShapedTypeOpInterface`

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `inputs` | 可变数量的任意类型
| `outputs` | 可变数量的任意类型形状值

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result_tensors` | 可变数量的任意类型张量


### `hfusion.sort` (hfusion::SortOp)

_排序操作_

语法：

```
operation ::= `hfusion.sort` attr-dict `ins` `(` $src `:` type($src) `)`
              `descending` `=` $descending
              `sort_axis` `=` $sort_axis
              (`->` type($result)^)?
```

按升序或降序对 `src` 的排序轴进行排序，并输出排序后的值和对应的索引。

约束：
  1. 输入向量和输出向量必须具有相同的秩。
  2. 目前仅支持尾部轴排序。

参数：
  * `src`: 要排序的张量/memref
  * `dst_value`: 存储排序值的张量/memref
  * `dst_index`: 存储对应于 dst_value 的索引的张量/memref
  * `descending`: 确定是按升序还是降序排序。默认为 false，表示升序
  * `sort_axis`: 要排序的轴

示例：
  ```mlir
  %result = hfusion.sort ins(%src : tensor<?xf32>) descending = true sort_axis = 0 -> tensor<?xf32>
  ```

Traits: `SameOperandsAndResultRank`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>descending</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>sort_axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 张量或内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hfusion.store` (hfusion::StoreOp)

_逐元素存储张量_

不对输入操作数执行数值类型转换。
Traits: `AttrSizedOperandSegments`, `SingleBlockImplicitTerminator<mlir::linalg::YieldOp>`, `SingleBlock`

Interfaces: `DestinationStyleOpInterface`, `LinalgStructuredInterface`, `MemoryEffectOpInterface`, `ReifyRankedShapedTypeOpInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>atomic_kind</code></td><td>::mlir::hfusion::AtomicKindAttr</td><td><details><summary>allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10</summary>{{% markdown %}}Enum cases:
* none (`NONE`)
* add (`ADD`)
* max (`MAX`)
* min (`MIN`)
* and (`AND`)
* or (`OR`)
* xor (`XOR`)
* cas (`CAS`)
* xchg (`XCHG`)
* umax (`UMAX`)
* umin (`UMIN`){{% /markdown %}}</details></td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `inputs` | 可变数量的任意类型
| `outputs` | 可变数量的任意类型形状值

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result_tensors` | 可变数量的任意类型张量


### `hfusion.symbolic_dim` (hfusion::SymbolicDimOp)

_返回索引的符号维度引用_

语法：

```
operation ::= `hfusion.symbolic_dim` $symbolName attr-dict `:` type($result)
```

"symbolic_dim" 操作通过名称（通过符号属性）引用符号维度，并将其作为 `index` 类型的值返回。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>symbolName</code></td><td>::mlir::SymbolRefAttr</td><td>symbol reference attribute</td></tr>
</table>

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 索引


## Attributes

### AtomicKindAttr

allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

语法：

```
#hfusion.atomic_kind<
  ::mlir::hfusion::AtomicKind   # value
>
```

枚举情况：
* none (`NONE`)
* add (`ADD`)
* max (`MAX`)
* min (`MIN`)
* and (`AND`)
* or (`OR`)
* xor (`XOR`)
* cas (`CAS`)
* xchg (`XCHG`)
* umax (`UMAX`)
* umin (`UMIN`)
#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hfusion::AtomicKind` | 类型为 AtomicKind |

### BinaryFnAttr

allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17

语法：

```
#hfusion.binary_fn<
  ::mlir::hfusion::BinaryFn   # value
>
```

枚举情况：
* vor (`vor`)
* vand (`vand`)
* vxor (`vxor`)
* minf (`minf`)
* maxf (`maxf`)
* powf (`powf`)
* mod (`mod`)
* modui (`modui`)
* shli (`shli`)
* shrsi (`shrsi`)
* shrui (`shrui`)
* ldexp (`ldexp`)
* ceildivsi (`ceildivsi`)
* ceildivui (`ceildivui`)
* floordivsi (`floordivsi`)
* powi (`powi`)
* minnumf (`minnumf`)
* maxnumf (`maxnumf`)
#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hfusion::BinaryFn` | 类型为 BinaryFn |

### CompareFnAttr

allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

语法：

```
#hfusion.compare_fn<
  ::mlir::hfusion::CompareFn   # value
>
```

枚举情况：
* veq (`veq`)
* vne (`vne`)
* vle (`vle`)
* vlt (`vlt`)
* vge (`vge`)
* vgt (`vgt`)
* vule (`vule`)
* vult (`vult`)
* vuge (`vuge`)
* vugt (`vugt`)
#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hfusion::CompareFn` | 类型为 CompareFn |

### BindSubBlockAttr

语法: `#hfusion.bind_sub_block`

用于绑定子块的特定操作。

### FusionKindAttr

语法:

```
#hfusion.fusion_kind<
  ::mlir::hfusion::FusionKind   # fusion_kind
>
```

HFusion 融合内核类型。

#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| fusion_kind | `::mlir::hfusion::FusionKind` | 类型为 FusionKind |

### InsertSliceSourceIndexAttr

语法: `#hfusion.insert_slice_source_index`

指定 concat 操作中哪个操作数是 insert_slice 源

### MultiBufferAttr

语法: `#hfusion.multi_buffer`

用于目标操作的 HFusion 多缓冲区属性。

### ReduceComposeAttr

语法: `#hfusion.reduce_composed`

HFusion 归约组合。

### ReduceWithIndexKindAttr

语法:

```
#hfusion.reduce_with_index_kind<
  ::mlir::hfusion::ReduceWithIndexKind   # reduce_with_index_kind
>
```

带索引的归约类型。

#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| reduce_with_index_kind | `::mlir::hfusion::ReduceWithIndexKind` | 类型为 ReduceWithIndexKind |

### ReturnOperandNumAttr

语法: `#hfusion.return_operand_num`

指定这在函数返回中对应于哪个操作数

### StrideAlignDimsAttr

语法: `#hfusion.stride_align_dims`

HFusion 步长对齐维度。

### StrideAlignValueInByteAttr

语法: `#hfusion.stride_align_value_in_byte`

HFusion 步长对齐值（字节）。

### RoundModeAttr

allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6

语法:

```
#hfusion.round_mode<
  ::mlir::hfusion::RoundMode   # value
>
```

- RINT: round to nearest, tie to even (c language rint)
- ROUND: round to nearest, tie away from zero (c language round)
- FLOOR: round to minus infinity (c language floor)
- CEIL: round to positive infinity (c language ceil)
- TRUNC: round to zero (c language trunc)
- ODD: round to odd (Von Neumann rounding)

#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hfusion::RoundMode` | 类型为 RoundMode |

### TernaryFnAttr

allowed 32-bit signless integer cases: 0

语法:

```
#hfusion.ternary_fn<
  ::mlir::hfusion::TernaryFn   # value
>
```

枚举情况：
* select (`select`)
#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hfusion::TernaryFn` | 类型为 TernaryFn |

### TypeFnAttr

allowed 32-bit signless integer cases: 0, 1, 2

语法:

```
#hfusion.type_fn<
  ::mlir::hfusion::TypeFn   # value
>
```

枚举情况：
* cast_signed (`cast_signed`)
* cast_unsigned (`cast_unsigned`)
* bitcast (`bitcast`)
#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hfusion::TypeFn` | 类型为 TypeFn |

### UnaryFnAttr

allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17

语法:

```
#hfusion.unary_fn<
  ::mlir::hfusion::UnaryFn   # value
>
```

枚举情况：
* relu (`relu`)
* sqrt (`sqrt`)
* rsqrt (`rsqrt`)
* rec (`rec`)
* vnot (`vnot`)
* tanh (`tanh`)
* sin (`sin`)
* cos (`cos`)
* atan (`atan`)
* tan (`tan`)
* absi (`absi`)
* erf (`erf`)
* log2 (`log2`)
* log10 (`log10`)
* log1p (`log1p`)
* exp2 (`exp2`)
* expm1 (`expm1`)
* ilogb (`ilogb`)
#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hfusion::UnaryFn` | 类型为 UnaryFn |

## Enums

### AtomicKind

allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

#### 枚举值:

| 符号 | 值 | 字符串 |
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

allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17

#### 枚举值:

| 符号 | 值 | 字符串 |
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

allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8

#### 枚举值:

| 符号 | 值 | 字符串 |
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

allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

#### 枚举值:

| 符号 | 值 | 字符串 |
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

HFusion 展平模式

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| Greedy | `1` | Greedy |
| Tidy | `2` | Tidy |

### FusionKind

HFusion 融合内核类型

#### 枚举值:

| 符号 | 值 | 字符串 |
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

HFusion 输出模式

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| Multiple | `1` | Multiple |
| Single | `2` | Single |
| SingleAggressive | `3` | SingleAggressive |

### CumOpType

HFusion 累积操作类型

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| UNDEFINED | `0` | undefined |
| CUMSUM | `1` | cumsum |
| CUMPROD | `2` | cumprod |

### MmMapMode

allowed 32-bit signless integer cases: 0, 1

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| CoreOp | `0` | core_op |
| MacroInstr | `1` | macro_instr |

### ReduceWithIndexKind

allowed 32-bit signless integer cases: 0, 1

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| MIN | `0` | min |
| MAX | `1` | max |

### RoundMode

allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| RINT | `0` | rint |
| ROUND | `1` | round |
| FLOOR | `2` | floor |
| CEIL | `3` | ceil |
| TRUNC | `4` | trunc |
| ODD | `5` | odd |
| TRUNCWITHOVERFLOW | `6` | truncwithoverflow |

### TaylerMode

allowed 32-bit signless integer cases: 0, 1

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| SIN | `0` | sin |
| ATAN | `1` | atan |

### TernaryFn

allowed 32-bit signless integer cases: 0

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| select | `0` | select |

### TypeFn

allowed 32-bit signless integer cases: 0, 1, 2

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| cast_signed | `0` | cast_signed |
| cast_unsigned | `1` | cast_unsigned |
| bitcast | `2` | bitcast |

### UnaryFn

allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17

#### 枚举值:

| 符号 | 值 | 字符串 |
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

---

# 'hacc' 方言

异构异步计算调用（HACC）方言。


---

# 'hivm' 方言

HIVM（混合智能虚拟机）方言。


## Operations

### `hivm.hir.atomic_cas` (hivm::AtomicCasOp)

_原子比较并交换（CAS）操作_

语法：

```
operation ::= `hivm.hir.atomic_cas` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`->` type($result_tensor)^)?
```

比较并交换（CAS）是一个原子操作，由三个操作数组成：
内存位置（V）、期望的旧值（A）、新值（B）。
操作的语义是：只有当内存位置 V 的值等于期望的旧值 A 时，V 的值才会更新为 B。
无论是否更新，操作都会返回 V 的原始值。

约束：
  1. 输入 memref 和输出 memref 必须具有相同的秩和相同的元素类型。

参数：
  * `src0`: 期望的旧值
  * `src1`: 新值
  * `dst`: GM 中的内存位置

示例：
```mlir
hivm.hir.atomic_cas ins(%src0, %src1 : memref<?xf32>, memref<?xf32>) outs(%dst : memref<?xf32>)
%result = hivm.hir.atomic_cas ins(%src0, %src1 : tensor<?xf32>, tensor<?xf32>) outs(%dst : tensor<?xf32>) -> tensor<?xf32>
```

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的张量或内存引用
| `dst` | 张量或内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result_tensor` | 张量或内存引用


### `hivm.hir.atomic_rmw` (hivm::AtomicRMWOp)

_原子 RMW 操作_

语法：

```
operation ::= `hivm.hir.atomic_rmw` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `atomic_kind` `=` $atomic_kind
              (`->` type($result_tensor)^)?
```

原子交换是一个由三个步骤组成的原子操作：
1. 读取指定内存地址的当前值
2. 根据 atomic_kind 属性对值执行操作
3. 返回之前读取的旧值
整个过程是原子的，即在操作期间不会被其他线程中断。

约束：
  1. 输入 memref 和输出 memref 必须具有相同的秩和相同的元素类型。

参数：
  * `src`: 新值
  * `dst`: GM 中的内存位置

示例：
```mlir
hivm.hir.atomic_rmw ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) atomic_kind = <add>
%result = hivm.hir.atomic_rmw ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) atomic_kind = <or> -> tensor<?xf32>
```

Interfaces: `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `InferCoreTypeInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>atomic_kind</code></td><td>::mlir::hivm::AtomicKindAttr</td><td><details><summary>Atomic Operation Kind for StoreOp</summary>{{% markdown %}}
    HIVM atomic store kind attribute.
  {{% /markdown %}}</details></td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 任意类型
| `dst` | 张量或内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result_tensor` | 张量或内存引用


### `hivm.hir.atomic_xchg` (hivm::AtomicXchgOp)

_原子交换操作_

语法：

```
operation ::= `hivm.hir.atomic_xchg` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`mask` `(` $mask^ `:` type($mask) `)`)?
              (`->` type($result_tensor)^)?
```

原子交换是一个由三个步骤组成的原子操作：
1. 读取指定内存地址的当前值
2. 将新值写入内存地址
3. 返回之前读取的旧值
整个过程是原子的，即在操作期间不会被其他线程中断。

约束：
  1. 输入 memref 和输出 memref 必须具有相同的秩和相同的元素类型。

参数：
  * `src`: 新值
  * `dst`: GM 中的内存位置
  * `mask`: 掩码元素

示例：
```mlir
hivm.hir.atomic_xchg ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>)
%result = hivm.hir.atomic_cas ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) -> tensor<?xf32>
```

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 任意类型
| `dst` | 张量或内存引用
| `mask` | 张量或内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result_tensor` | 张量或内存引用


### `hivm.hir.batchMmadL1` (hivm::BatchMmadL1Op)

_从 L1 内存层次结构输入的批量矩阵乘加操作_

语法：

```
operation ::= `hivm.hir.batchMmadL1` attr-dict `ins` `(`
              $a
              `,` $b
              `,` $init_condition
              `,` $real_m
              `,` $real_k
              `,` $real_n
              (`,` $per_channel_bias^)?
              `:`
              type($a)
              `,` type($b)
              `,` type($init_condition)
              `,` type($real_m)
              `,` type($real_k)
              `,` type($real_n)
              (`,` type($per_channel_bias)^)? `)`
              `outs` `(` $c `:` type($c) `)`
              (`sync_related_args` `(` $sync_related_args^ `:` type($sync_related_args) `)`)?
              (`unit_flag` `[` $unit_flag_mode^ (`,` $unit_flag_cond^)? `]`)?
              (`->` type($result_tensors)^)?
```

计算逻辑是：

```
C = C + A x B + (optional) channel_bias
```

注意：A、B 和 C 矩阵的秩必须为三，其中第 0 维是批次维度。

Traits: `AttrSizedOperandSegments`, `CubeCoreTypeTrait`, `MacroOpPipeTrait<PIPE::PIPE_MTE1, PIPE::PIPE_M>`, `MacroOpTrait`

Interfaces: `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `HIVMUnitFlagEnabledInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>a_transpose</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>b_transpose</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>enable_HF32</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>unit_flag_mode</code></td><td>::mlir::hivm::UnitFlagAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM unit flag attribute for synchronization.
  {{% /markdown %}}</details></td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `a` | 张量或内存引用
| `b` | 张量或内存引用
| `init_condition` | 1-bit signless integer
| `real_m` | 索引
| `real_k` | 索引
| `real_n` | 索引
| `c` | 张量或内存引用
| `sync_related_args` | 可变数量的 64 位无符号整数
| `unit_flag_cond` | 1-bit signless integer
| `per_channel_bias` | 张量或内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result_tensors` | 可变数量的任意类型张量


### `hivm.hir.bitcast` (hivm::BitcastOp)

_在不改变数据的情况下重新解释形状值的位_

语法：

```
operation ::= `hivm.hir.bitcast` $src `:` type($src) `->` type($result) attr-dict
```

`bitcast` 操作将 tensor/memref 从一种元素类型转换为另一种元素类型，同时保留底层位表示。该操作要求：

1. 输入和输出具有相同的形状（2x3 != 3x2）
2. 相同的总位宽（element_bitwidth * num_elements）
3. 相同的内存布局/步长（对于 memref）

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultShape`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 任意类型

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 任意类型


### `hivm.hir.convert_layout` (hivm::ConvertLayoutOp)

_HIVM 布局转换操作_

语法：

```
operation ::= `hivm.hir.convert_layout` $source attr-dict `:` functional-type(operands, results)
```

`convert_layout` 操作将具有一种布局的 memref 转换为另一种布局。数据不会被复制或修改。

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultElementType`

Interfaces: `ConditionallySpeculatable`, `InferCoreTypeInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ViewLikeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>srcLayout</code></td><td>::mlir::hivm::DataLayoutAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM data layout mapping attribute. Maps to DOTA_ND, DOTB_ND, DOTC_ND, zN, nZ and ND.
      - `transpose`: Indicates that the layout is transposed.
                     Only valid and must be present for DOTA_ND and DOTB_ND layout.
  {{% /markdown %}}</details></td></tr>
<tr><td><code>dstLayout</code></td><td>::mlir::hivm::DataLayoutAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM data layout mapping attribute. Maps to DOTA_ND, DOTB_ND, DOTC_ND, zN, nZ and ND.
      - `transpose`: Indicates that the layout is transposed.
                     Only valid and must be present for DOTA_ND and DOTB_ND layout.
  {{% /markdown %}}</details></td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `source` | 任意类型的有秩或无秩内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 任意类型的有秩或无秩内存引用


### `hivm.hir.copy` (hivm::CopyOp)

_HIVM 数据复制操作_

语法：

```
operation ::= `hivm.hir.copy` `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              attr-dict
              (`pad_mode` `=` $pad_mode^)?
              (`pad_value` `=` $pad_value^ `:` type($pad_value))?
              (`collapse_reassociation` `=` $collapse_reassociation^)?
              (`->` type($result_tensor)^)?
```

在本地内存层次结构之间复制数据。
目前支持：
  - UB 到 UB
  - UB 到 L1（适用于 Ascend910_95 系列）

示例：
```mlir
hivm.hir.copy ins(%src : memref<16x16xf16, #hivm.address_space<ub>>) outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
```

约束：
- `src` 和 `dst` 应具有相同的元素类型。
- 如果未设置 `pad_mode`，`src` 和 `dst` 的形状应该相同。
- 仅支持左填充。
- `pad_value` 应与 `src` 和 `dst` 具有相同的元素类型。

### 非连续重关联重塑
`hivm.hir.copy` 还支持将非连续数据复制到连续存储，反之亦然。
这可以看作是数据的"展开"或"折叠"。`collapse_reassociation` 属性用于指定哪些轴被折叠在一起。
例如：
```mlir
hivm.hir.copy ins(%src : memref<32x4xbf16, strided<[16, 1]>>) outs(%dst : memref<32x4xbf16, strided<[4, 1]>>)
  collapse_reassociation = [[0, 1]]
```
表示第 0 和第 1 轴被连续折叠。

Traits: `AlwaysSpeculatableImplTrait`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`

Interfaces: `ConditionallySpeculatable`, `CopyOpInterface`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `InferCoreTypeInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>pad_mode</code></td><td>::mlir::hivm::PadModeAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM pad mode attribute.
  {{% /markdown %}}</details></td></tr>
<tr><td><code>collapse_reassociation</code></td><td>::mlir::ArrayAttr</td><td>Array of 64-bit integer array attributes</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 张量或内存引用
| `dst` | 张量或内存引用
| `pad_value` | 任意类型

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result_tensor` | 任意类型的张量


### `hivm.hir.create_sync_block_lock` (hivm::CreateSyncBlockLockOp)

_创建同步块锁操作_

语法：

```
operation ::= `hivm.hir.create_sync_block_lock` (`from` $lockArg^)?
              attr-dict `:` (`from` type($lockArg)^ `to`)? type($memref)
```

`create_sync_block_lock` 操作分配一个锁内存区域，用于使锁和解锁之间的代码在块之间按顺序执行。
示例：

```mlir
  hivm.hir.create_sync_block_lock() : memref<1xi64>
  hivm.hir.create_sync_block_lock() from %arg : from memref<?xi8> to memref<1xi64>
```

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `lockArg` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `memref` | 任意类型的内存引用


### `hivm.hir.custom` (hivm::CustomOp)

_自定义操作是用户编写自己的自定义实现的通用操作接口。

    场景：
      1. 现有操作无法满足所需功能。
      2. 现有操作可以满足功能，但整体性能不是最优的。
      3. 需要私有操作。_

自定义操作的通用接口，其中：
  - name : 唯一的操作名称。

           注意：有一些保留名称，通常以 "__builtin" 开头。
                  编译器会将这些内置函数链接到自包含的模板库，
                  该库随 bishengir-compile 一起提供。

                  对于普通名称/情况，用户需要指定实现位置/编译命令（TODO），
                  以及所有必要的信息。

           可用的内置名称：
             "__builtin_gather_load"

  - inputs : 输入参数。
  - outputs : 输出结果，指定的 "init" 操作数，它们作为操作的初始值
              或操作结果将被写入的初始位置。

为了快速动态地适应未来的增强，自定义操作依赖于属性来检索必要的信息，所需的信息包括：
  - CoreType : 在哪个核心类型上执行，参考 TCoreTypeAttr。
  - Pipe     : 在哪个管道上执行，参考 PipeAttr。
  - VFMode   : 在向量单元上运行的模式，参考 VFModeAttr。
               当核心类型是 cube 时，此属性会被忽略。

               注意：对于内置函数，用户可以指定这些信息或不指定，
                      编译器将帮助检查正确性并进行规范化。

TODO:
  - Impl : 用户提供的实现。
  - Multi Pipe : 自定义操作想要使用多个管道，这在 HIVM 的上下文中是 MacroOp。

Traits: `AttrSizedOperandSegments`, `SinglePipeOpTrait`

Interfaces: `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `InferCoreTypeInterface`, `MemoryEffectOpInterface (MemoryEffectOpInterface)`, `MemoryEffectsOpInterface`, `OpPipeInterface`

Effects: `MemoryEffects::Effect{MemoryEffects::Read on ::mlir::SideEffects::DefaultResource, MemoryEffects::Write on ::mlir::SideEffects::DefaultResource}`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>name</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `inputs` | 可变数量的任意类型
| `outputs` | 可变数量的任意类型

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `results` | 可变数量的任意类型


### `hivm.hir.dcci` (hivm::DCCIOp)

_Hivm dcci 操作_

语法：

```
operation ::= `hivm.hir.dcci` attr-dict `(` $mode `,` $dataCacheKind (`,` $ptr^ `:` type($ptr))? `)`
```

此操作清理（写回）并使一个缓存行或整个数据缓存无效

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>mode</code></td><td>::mlir::hivm::DCCIModeAttr</td><td><details><summary>hivm dcci mode</summary>{{% markdown %}}HIVM DCCI mode attribute.{{% /markdown %}}</details></td></tr>
<tr><td><code>dataCacheKind</code></td><td>::mlir::hivm::DataCacheKindAttr</td><td><details><summary>hivm data cache kind</summary>{{% markdown %}}HIVM data cache kind attribute.{{% /markdown %}}</details></td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `ptr` | 任意类型的内存引用


### `hivm.hir.debug` (hivm::DebugOp)

_设备端调试_

语法：

```
operation ::= `hivm.hir.debug` attr-dict $arg `:` type($arg)
```

Interfaces: `InferCoreTypeInterface`, `MemoryEffectOpInterface (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{MemoryEffects::Read on ::mlir::SideEffects::DefaultResource, MemoryEffects::Write on ::mlir::SideEffects::DefaultResource}`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>debugtype</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>prefix</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>hex</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>tcoretype</code></td><td>::mlir::hivm::TCoreTypeAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM op core type attribute.
  {{% /markdown %}}</details></td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `arg` | integer or floating-point or Tensor or Memref


### `hivm.hir.finish_debug` (hivm::FinishDebugOp)

_完成设备端调试函数_

语法：

```
operation ::= `hivm.hir.finish_debug` attr-dict
```

Traits: `CubeVectorCoreTypeTrait`


### `hivm.hir.fixpipe` (hivm::FixpipeOp)

_从 L0C 到其他内存层次结构的 HIVM 数据复制操作_

语法：

```
operation ::= `hivm.hir.fixpipe` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`unit_flag` `[` $unit_flag_mode^ (`,` $unit_flag_cond^)? `]`)?
              (`->` type($result_tensor)^)?
```

Fixpipe 是从 L0C 到其他内存层次结构执行数据移动的流水线，
具有预阶段量化、预阶段 ReLU、逐元素加法、后阶段 ReLU、后阶段量化的即时固定功能。
目前支持：
  - L0C 到 OUT
  - L0C 到 L1
  - L0C 到 UB（适用于 Ascend910_95 系列）

此外，Fixpipe 还能够进行布局转换。

Traits: `AlwaysSpeculatableImplTrait`, `CubeCoreTypeTrait`, `OpPipeTrait<PIPE::PIPE_FIX>`, `SinglePipeOpTrait`

Interfaces: `ConditionallySpeculatable`, `CopyOpInterface`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `HIVMUnitFlagEnabledInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>enable_nz2nd</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>pre_quant</code></td><td>::mlir::hivm::FixpipePreQuantModeAttr</td><td><details><summary>HIVM fixpipe pre_quant mode</summary>{{% markdown %}}HIVM fixpipe pre_quant mode{{% /markdown %}}</details></td></tr>
<tr><td><code>pre_relu</code></td><td>::mlir::hivm::FixpipePreReluModeAttr</td><td><details><summary>HIVM fixpipe pre_relu mode</summary>{{% markdown %}}HIVM fixpipe pre_relu mode{{% /markdown %}}</details></td></tr>
<tr><td><code>channel_split</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>unit_flag_mode</code></td><td>::mlir::hivm::UnitFlagAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM unit flag attribute for synchronization.
  {{% /markdown %}}</details></td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 任意类型的形状值
| `dst` | 任意类型的形状值
| `unit_flag_cond` | 1-bit signless integer

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result_tensor` | 任意类型的张量


### `hivm.hir.get_block_idx` (hivm::GetBlockIdxOp)

_获取用于并行化的当前设备线程的块索引_

语法：

```
operation ::= `hivm.hir.get_block_idx` attr-dict `->` type($result)
```

此操作获取当前设备线程的块索引。
此操作将被降级为 `GetBlockIdxInstrOp`。

Traits: `AlwaysSpeculatableImplTrait`, `CubeVectorCoreTypeTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 64 位无符号整数


### `hivm.hir.get_block_num` (hivm::GetBlockNumOp)

_获取用于并行化的当前设备线程的块数量_

语法：

```
operation ::= `hivm.hir.get_block_num` attr-dict `->` type($result)
```

此操作获取当前设备线程的块数量。
此操作将被降级为 `GetBlockNumInstrOp`。

Traits: `AlwaysSpeculatableImplTrait`, `CubeVectorCoreTypeTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 64 位无符号整数


### `hivm.hir.get_sub_block_idx` (hivm::GetSubBlockIdxOp)

_获取用于并行化的当前设备线程的子块索引_

语法：

```
operation ::= `hivm.hir.get_sub_block_idx` attr-dict `->` type($result)
```

此操作获取当前设备线程的子块索引。
此操作将被降级为 GetSubBlockIdxInstrOp。

Traits: `AlwaysSpeculatableImplTrait`, `CubeVectorCoreTypeTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 64 位无符号整数


### `hivm.hir.get_sub_block_num` (hivm::GetSubBlockNumOp)

_获取用于并行化的当前设备线程的子块数量_

语法：

```
operation ::= `hivm.hir.get_sub_block_num` attr-dict `->` type($result)
```

此操作获取当前设备线程的子块数量。
此操作将被降级为 GetSubBlockNumInstrOp。

Traits: `AlwaysSpeculatableImplTrait`, `CubeVectorCoreTypeTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 64 位无符号整数


### `hivm.hir.get_sys_cnt` (hivm::GetSysCntOp)

_获取当前设备的系统计数_

语法：

```
operation ::= `hivm.hir.get_sys_cnt` attr-dict `->` type($result)
```

此操作获取当前设备的系统计数。
此操作将被降级为 `GetSysCntInstrOp`。

Traits: `AlwaysSpeculatableImplTrait`, `CubeVectorCoreTypeTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 64 位无符号整数


### `hivm.hir.init_debug` (hivm::InitDebugOp)

_初始化设备端调试函数_

语法：

```
operation ::= `hivm.hir.init_debug` attr-dict
```

Traits: `CubeVectorCoreTypeTrait`


### `hivm.hir.load` (hivm::LoadOp)

_HIVM 数据加载操作_

语法：

```
operation ::= `hivm.hir.load` `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              attr-dict
              (`pad_mode` `=` $pad_mode^)?
              (`pad_value` `=` $pad_value^ `:` type($pad_value))?
              (`left_padding_num` `=` $left_padding_num^ `:` type($left_padding_num))?
              (`init_out_buffer` `=` $init_out_buffer^ )?
              (`right_padding_num` `=` $right_padding_num^ `:` type($right_padding_num))?
              (`init_condition` `=` $init_condition^ `:` type($init_condition))?
              (`may_implicit_transpose_with_last_axis` `=` $may_implicit_transpose_with_last_axis^ )?
              (`->` type($result_tensor)^)?
```

从全局内存加载数据到本地缓冲区。
目前仅支持加载到统一缓冲区。

示例：
```mlir
hivm.load ins(%src : memref<16x16xf16, #hivm.address_space<gm>>) outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
```

约束：
- `src` 和 `dst` 应具有相同的元素类型。
- 如果未设置 `pad_mode`，`src` 和 `dst` 的形状应该相同。
- 支持左填充和右填充。
- `pad_value` 应与 `src` 和 `dst` 具有相同的元素类型。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `OpPipeTrait<PIPE::PIPE_MTE2>`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`

Interfaces: `BiShengIRAggregatedOpInterface`, `ConditionallySpeculatable`, `CopyOpInterface`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `InferCoreTypeInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>pad_mode</code></td><td>::mlir::hivm::PadModeAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM pad mode attribute.
  {{% /markdown %}}</details></td></tr>
<tr><td><code>init_out_buffer</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>may_implicit_transpose_with_last_axis</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 张量或内存引用
| `dst` | 张量或内存引用
| `pad_value` | 任意类型
| `left_padding_num` | 索引
| `right_padding_num` | 任意类型
| `init_condition` | 任意类型

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result_tensor` | 任意类型的张量


### `hivm.hir.load_scalar` (hivm::LoadScalarOp)

_Hivm 加载标量_

语法：

```
operation ::= `hivm.hir.load_scalar` attr-dict $addr `:` type($addr) `->` type($result)
```

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `addr` | LLVM pointer type

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | integer or floating-point


### `hivm.hir.matmul` (hivm::MatmulOp)

_从全局内存输入的 HIVM 矩阵乘法操作_

语法：

```
operation ::= `hivm.hir.matmul` attr-dict `ins` `(` $a `,` $b `:` type($a) `,` type($b) `)`
              `outs` `(` $c `:` type($c) `)`
              (`tiling_params` `=` $tilingParams^ `:` type($tilingParams) ) ?
              (`bias` `=` $bias^ `:` type($bias) )?
              (`descale` `=` $descale^ `:` type($descale))?
              (`a_transpose` $aTranspose^)?
              (`b_transpose` $bTranspose^)?
              (`descale_mode` `=` $descaleMode^)?
              (`block_sizes` `(` $blockSizes^ `:` type($blockSizes) `)`)?
              (`process_sizes` `(` $processSizes^ `:` type($processSizes) `)`)?
              (`swizzle_offset` `=` $swizzleOffset^ `:` type($swizzleOffset) )?
              (`swizzle_direction` `=` $swizzleDirection^ `:` type($swizzleDirection))?
              (`epilogue_p_tiles` `=` $epiloguePTiles^ `:` type($epiloguePTiles))?
              (`->` type($result)^)?
```

此操作从全局内存中获取三个分块矩阵作为参数：
  - `A`（ranked type）：一个 `m x k` 矩阵
  - `B`（ranked type）：一个 `k x n` 矩阵
  - `C`（ranked type）：一个 `m x n` 矩阵

其他参数包括：
  - `block_sizes`: 在 L1 内存层次结构上处理的 m、n 和 k 维度的数据大小
  - `process_sizes`: 在 L0 内存层次结构上处理的 m、n 和 k 维度的数据大小
  - （可选）`swizzle_offset`: swizzle 调度的连续块数量
  - （可选）`swizzle_direction`: swizzle 调度的块方向
  - （可选）`epilogue_p_tiles`: 一次处理附加操作的块数量

执行的操作表示为 `C = A * B`。
如果存在 `a_transpose` 或 `b_transpose`，则相应的操作数以转置方式加载。

可选地，此操作接受以下参数：
  - `bias`（ranked type）：偏置值，形状为 `n` 的向量
  - `descale`: 反量化值。支持 3 种类型：
  - `DescaleNull` : 无 descale。
  - `DescalePerChannel`: `descale` 的形状等于 `n`。
  - `DescalePerTensor`: `descale` 的形状等于 `1`。

执行的操作表示为 `C = descale * (A * B + bias)`。

Traits: `AttrSizedOperandSegments`, `MacroOpPipeTrait<PIPE::PIPE_MTE2, PIPE::PIPE_MTE3>`, `MacroOpTrait`

Interfaces: `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `InferCoreTypeInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>aTranspose</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>bTranspose</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>descaleMode</code></td><td>::mlir::hivm::DescaleModeAttr</td><td><details><summary>descale mode for matmul</summary>{{% markdown %}}
    HIVM descale mode attribute for matmul op.
  {{% /markdown %}}</details></td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `a` | 任意类型的形状值
| `b` | 任意类型的形状值
| `tilingParams` | 任意类型的形状值
| `bias` | 任意类型的形状值
| `descale` | 任意类型的形状值
| `blockSizes` | 可变数量的 64 位无符号整数
| `processSizes` | 可变数量的 64 位无符号整数
| `swizzleOffset` | 64 位无符号整数
| `swizzleDirection` | 64 位无符号整数
| `epiloguePTiles` | 64 位无符号整数
| `c` | 任意类型的形状值

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.mix_group_matmul` (hivm::MixGroupMatmulOp)

_从全局内存输入的 HIVM（Mix）矩阵组乘法操作_

语法：

```
operation ::= `hivm.hir.mix_group_matmul` attr-dict `ins` `(` $a `,` $b `,` $tokens_per_expert `:` type($a) `,` type($b) `,` type($tokens_per_expert) `)`
              (`post_vector_func_ins` `(` $postVecFuncIns^ `:` type($postVecFuncIns) `)`) ?
              (`post_vector_func_outs` `(` $postVecFuncOuts^ `:` type($postVecFuncOuts) `)`) ?
              (`workspace_ins` `(` $workspaceIns^ `:` type($workspaceIns) `)`) ?
              `outs` `(` $c `:` type($c) `)`
              (`tiling_params` `=` $tilingParams^ `:` type($tilingParams) ) ?
              (`comm_params` `=` $commParams^ `:` type($commParams) ) ?
              (`bias` `=` $bias^ `:` type($bias) )?
              (`descale` `=` $descale^ `:` type($descale))?
              (`a_transpose` $aTranspose^)?
              (`b_transpose` $bTranspose^)?
              (`descale_mode` `=` $descaleMode^)?
              (`block_sizes` `(` $blockSizes^ `:` type($blockSizes) `)`)?
              (`process_sizes` `(` $processSizes^ `:` type($processSizes) `)`)?
              (`swizzle_offset` `=` $swizzleOffset^ `:` type($swizzleOffset) )?
              (`swizzle_direction` `=` $swizzleDirection^ `:` type($swizzleDirection))?
              (`epilogue_p_tiles` `=` $epiloguePTiles^ `:` type($epiloguePTiles))?
              (`->` type($result)^)?
```

此操作从全局内存中获取三个分块矩阵作为参数：
  - `A`（ranked type）：一个 `m x k` 矩阵
  - `B`（ranked type）：一个 `k x n` 矩阵
  - `C`（ranked type）：一个 `m x n` 矩阵

其他参数包括：
  - `block_sizes`: 在 L1 内存层次结构上处理的 m、n 和 k 维度的数据大小
  - `process_sizes`: 在 L0 内存层次结构上处理的 m、n 和 k 维度的数据大小
  - （可选）`swizzle_offset`: swizzle 调度的连续块数量
  - （可选）`swizzle_direction`: swizzle 调度的块方向
  - （可选）`epilogue_p_tiles`: 一次处理附加操作的块数量

执行的操作表示为 `C = A * B`。
如果存在 `a_transpose` 或 `b_transpose`，则相应的操作数以转置方式加载。

可选地，此操作接受以下参数：
  - `bias`（ranked type）：偏置值，形状为 `n` 的向量
  - `descale`: 反量化值。支持 3 种类型：
  - `DescaleNull` : 无 descale。
  - `DescalePerChannel`: `descale` 的形状等于 `n`。
  - `DescalePerTensor`: `descale` 的形状等于 `1`。

执行的操作表示为 `C = descale * (A * B + bias)`。

此操作还支持与后向量函数的 tile 级融合（因此它是 Mix 操作）
`tokens_per_expert` 指定如何将矩阵乘法分配给不同的专家
`post_vector_func_ins` 用于指定参数。
`post_vector_func_outs` 用于指定输出。
`comm_params` 用于指定融合通信操作时的通信相关参数（例如
  拓扑、通信器、组等）。

Traits: `AttrSizedOperandSegments`, `MacroOpPipeTrait<PIPE::PIPE_MTE2, PIPE::PIPE_MTE3>`, `MacroOpTrait`

Interfaces: `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `InferCoreTypeInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>aTranspose</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>bTranspose</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>descaleMode</code></td><td>::mlir::hivm::DescaleModeAttr</td><td><details><summary>descale mode for matmul</summary>{{% markdown %}}
    HIVM descale mode attribute for matmul op.
  {{% /markdown %}}</details></td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `a` | 任意类型的形状值
| `b` | 任意类型的形状值
| `tokens_per_expert` | 任意类型的形状值
| `postVecFuncIns` | 可变数量的任意类型形状值
| `postVecFuncOuts` | 可变数量的任意类型形状值
| `workspaceIns` | 可变数量的任意类型形状值
| `tilingParams` | 任意类型的形状值
| `commParams` | 任意类型的形状值
| `bias` | 任意类型的形状值
| `descale` | 任意类型的形状值
| `blockSizes` | 可变数量的 64 位无符号整数
| `processSizes` | 可变数量的 64 位无符号整数
| `swizzleOffset` | 64 位无符号整数
| `swizzleDirection` | 64 位无符号整数
| `epiloguePTiles` | 64 位无符号整数
| `c` | 任意类型的形状值

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.mix_matmul` (hivm::MixMatmulOp)

_从全局内存输入的 HIVM（Mix）矩阵乘法操作_

语法：

```
operation ::= `hivm.hir.mix_matmul` attr-dict `ins` `(` $a `,` $b `:` type($a) `,` type($b) `)`
              (`post_vector_func_ins` `(` $postVecFuncIns^ `:` type($postVecFuncIns) `)`) ?
              (`workspace_ins` `(` $workspaceIns^ `:` type($workspaceIns) `)`) ?
              `outs` `(` $c `:` type($c) `)`
              (`tiling_params` `=` $tilingParams^ `:` type($tilingParams) ) ?
              (`comm_params` `=` $commParams^ `:` type($commParams) ) ?
              (`bias` `=` $bias^ `:` type($bias) )?
              (`descale` `=` $descale^ `:` type($descale))?
              (`a_transpose` $aTranspose^)?
              (`b_transpose` $bTranspose^)?
              (`descale_mode` `=` $descaleMode^)?
              (`block_sizes` `(` $blockSizes^ `:` type($blockSizes) `)`)?
              (`process_sizes` `(` $processSizes^ `:` type($processSizes) `)`)?
              (`swizzle_offset` `=` $swizzleOffset^ `:` type($swizzleOffset) )?
              (`swizzle_direction` `=` $swizzleDirection^ `:` type($swizzleDirection))?
              (`epilogue_p_tiles` `=` $epiloguePTiles^ `:` type($epiloguePTiles))?
              (`->` type($result)^)?
```

此操作从全局内存中获取三个分块矩阵作为参数：
  - `A`（ranked type）：一个 `m x k` 矩阵
  - `B`（ranked type）：一个 `k x n` 矩阵
  - `C`（ranked type）：一个 `m x n` 矩阵

其他参数包括：
  - `block_sizes`: 在 L1 内存层次结构上处理的 m、n 和 k 维度的数据大小
  - `process_sizes`: 在 L0 内存层次结构上处理的 m、n 和 k 维度的数据大小
  - （可选）`swizzle_offset`: swizzle 调度的连续块数量
  - （可选）`swizzle_direction`: swizzle 调度的块方向
  - （可选）`epilogue_p_tiles`: 一次处理附加操作的块数量

执行的操作表示为 `C = A * B`。
如果存在 `a_transpose` 或 `b_transpose`，则相应的操作数以转置方式加载。

可选地，此操作接受以下参数：
  - `bias`（ranked type）：偏置值，形状为 `n` 的向量
  - `descale`: 反量化值。支持 3 种类型：
  - `DescaleNull` : 无 descale。
  - `DescalePerChannel`: `descale` 的形状等于 `n`。
  - `DescalePerTensor`: `descale` 的形状等于 `1`。

执行的操作表示为 `C = descale * (A * B + bias)`。

此操作还支持与后向量函数的 tile 级融合（因此它是 Mix 操作）。
`post_vector_func_ins` 用于指定参数。
`comm_params` 用于指定融合通信操作时的通信相关参数（例如
  拓扑、通信器、组等）。

Traits: `AttrSizedOperandSegments`, `MacroOpPipeTrait<PIPE::PIPE_MTE2, PIPE::PIPE_MTE3>`, `MacroOpTrait`

Interfaces: `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `InferCoreTypeInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>aTranspose</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>bTranspose</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>descaleMode</code></td><td>::mlir::hivm::DescaleModeAttr</td><td><details><summary>descale mode for matmul</summary>{{% markdown %}}
    HIVM descale mode attribute for matmul op.
  {{% /markdown %}}</details></td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `a` | 任意类型的形状值
| `b` | 任意类型的形状值
| `postVecFuncIns` | 可变数量的任意类型形状值
| `workspaceIns` | 可变数量的任意类型形状值
| `tilingParams` | 任意类型的形状值
| `commParams` | 任意类型的形状值
| `bias` | 任意类型的形状值
| `descale` | 任意类型的形状值
| `blockSizes` | 可变数量的 64 位无符号整数
| `processSizes` | 可变数量的 64 位无符号整数
| `swizzleOffset` | 64 位无符号整数
| `swizzleDirection` | 64 位无符号整数
| `epiloguePTiles` | 64 位无符号整数
| `c` | 任意类型的形状值

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.mmadL1` (hivm::MmadL1Op)

_从 L1 内存层次结构输入的矩阵乘加操作_

语法：

```
operation ::= `hivm.hir.mmadL1` attr-dict `ins` `(`
              $a
              `,` $b
              `,` $init_condition
              `,` $real_m
              `,` $real_k
              `,` $real_n
              (`,` $per_channel_bias^)?
              `:`
              type($a)
              `,` type($b)
              `,` type($init_condition)
              `,` type($real_m)
              `,` type($real_k)
              `,` type($real_n)
              (`,` type($per_channel_bias)^)? `)`
              `outs` `(` $c `:` type($c) `)`
              (`sync_related_args` `(` $sync_related_args^ `:` type($sync_related_args) `)`)?
              (`unit_flag` `[` $unit_flag_mode^ (`,` $unit_flag_cond^)? `]`)?
              (`->` type($result_tensors)^)?
```

计算逻辑是：

```
C = C + A x B + (optional) channel_bias
```

注意：A、B 和 C 矩阵的秩必须为二。

Traits: `AttrSizedOperandSegments`, `CubeCoreTypeTrait`, `MacroOpPipeTrait<PIPE::PIPE_MTE1, PIPE::PIPE_M>`, `MacroOpTrait`

Interfaces: `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `HIVMUnitFlagEnabledInterface`, `MemoryEffectsOpInterface`, `OpLayoutInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>a_transpose</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>b_transpose</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>enable_HF32</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>unit_flag_mode</code></td><td>::mlir::hivm::UnitFlagAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM unit flag attribute for synchronization.
  {{% /markdown %}}</details></td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `a` | 张量或内存引用
| `b` | 张量或内存引用
| `init_condition` | 1-bit signless integer
| `real_m` | 索引
| `real_k` | 索引
| `real_n` | 索引
| `c` | 张量或内存引用
| `sync_related_args` | 可变数量的 64 位无符号整数
| `unit_flag_cond` | 1-bit signless integer
| `per_channel_bias` | 张量或内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result_tensors` | 可变数量的任意类型张量


### `hivm.hir.nd2nz` (hivm::ND2NZOp)

_带即时 ND 到 NZ 布局转换的 HIVM 数据复制操作_

语法：

```
operation ::= `hivm.hir.nd2nz` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`init_out_buffer` `=` $init_out_buffer^ )?
              (`pad_value` `=` $pad_value^ `:` type($pad_value))?
              (`init_condition` `=` $init_condition^ `:` type($init_condition))?
              (`->` type($result_tensor)^)?
```

- `dst_continuous`: 如果存在，表示源数据在目标缓冲区中连续存储。
  必须设置此选项才能将此操作转换为库函数调用。
约束：
- 如果 `init_out_buffer` 为 true，`pad_value` 应该有值。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CubeCoreTypeTrait`, `OpPipeTrait<PIPE::PIPE_MTE2>`, `SinglePipeOpTrait`

Interfaces: `BiShengIRAggregatedOpInterface`, `ConditionallySpeculatable`, `CopyOpInterface`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>dst_continuous</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
<tr><td><code>init_out_buffer</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 任意类型的形状值
| `dst` | 任意类型的形状值
| `pad_value` | 任意类型
| `init_condition` | 任意类型

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result_tensor` | 可变数量的任意类型张量


### `hivm.hir.nz2nd` (hivm::NZ2NDOp)

_从 L1 到全局内存的带 NZ2ND 转换的 HIVM 数据复制操作_

语法：

```
operation ::= `hivm.hir.nz2nd` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`->` type($result_tensor)^)?
```

NZ2ND 执行从 L1 到 OUT 的数据移动，并进行 NZ2ND 转换。
Traits: `AlwaysSpeculatableImplTrait`, `CubeCoreTypeTrait`, `OpPipeTrait<PIPE::PIPE_MTE3>`, `SinglePipeOpTrait`

Interfaces: `ConditionallySpeculatable`, `CopyOpInterface`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 张量或内存引用
| `dst` | 张量或内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result_tensor` | 任意类型的张量


### `hivm.hir.pipe_barrier` (hivm::PipeBarrierOp)

_Hivm 管道屏障_

语法：

```
operation ::= `hivm.hir.pipe_barrier` `[` $pipe `]` attr-dict
```

Interfaces: `InferCoreTypeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>pipe</code></td><td>::mlir::hivm::PipeAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM Op pipe attribute.
  {{% /markdown %}}</details></td></tr>
</table>


### `hivm.hir.pointer_cast` (hivm::PointerCastOp)

_在特定 i64 地址处的 HIVM 指针转换操作_

语法：

```
operation ::= `hivm.hir.pointer_cast` `(`$addrs `)` (`[` $dynamicSizes^`]`)? attr-dict `:` type($result)
```

特定的 i64 地址存储在 `$addrs` 中，它是可变的。

约束：
1. 每个地址的类型应该是 i64。
2. addrs 应该至少有一个地址。

示例：
```mlir
%addr = arith.constant 1234 : i64
%tmp = hivm.hir.pointer_cast(%addr) : memref<32xf32>

%addr2 = arith.constant 1600 : i64
%addr3 = arith.constant 3200 : i64
%tmp2 = hivm.hir.pointer_cast(%addr, %addr2) : memref<32xf32>
%tmp3 = hivm.hir.pointer_cast(%addr, %addr2, %addr3) : memref<32xf32>
```

Traits: `AttrSizedOperandSegments`, `CubeVectorCoreTypeTrait`

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `addrs` | 可变数量的 64 位无符号整数
| `dynamicSizes` | 可变数量的索引

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 任意类型的内存引用


### `hivm.hir.set_ffts_base_addr` (hivm::SetFFTSBaseAddrOp)

_设置 ffts 同步机制的基础地址_

语法：

```
operation ::= `hivm.hir.set_ffts_base_addr` attr-dict $ffts_base_addr
```

Traits: `CubeVectorCoreTypeTrait`

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `ffts_base_addr` | 64 位无符号整数


### `hivm.hir.set_flag` (hivm::SetFlagOp)

_Hivm 设置标志_

语法：

```
operation ::= `hivm.hir.set_flag` `[`
              $set_pipe
              `,` $wait_pipe
              `,` custom<EventID>($static_event_id, $dynamic_event_id)
              `]` attr-dict
```

Interfaces: `InferCoreTypeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>set_pipe</code></td><td>::mlir::hivm::PipeAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM Op pipe attribute.
  {{% /markdown %}}</details></td></tr>
<tr><td><code>wait_pipe</code></td><td>::mlir::hivm::PipeAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM Op pipe attribute.
  {{% /markdown %}}</details></td></tr>
<tr><td><code>static_event_id</code></td><td>::mlir::hivm::EventAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM event attribute for synchronization.
  {{% /markdown %}}</details></td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `dynamic_event_id` | 64 位无符号整数


### `hivm.hir.set_mask_norm` (hivm::SetMaskNormOp)

_Hivm 设置掩码范数_

语法：

```
operation ::= `hivm.hir.set_mask_norm` attr-dict
```


### `hivm.hir.store` (hivm::StoreOp)

_HIVM 数据存储操作_

语法：

```
operation ::= `hivm.hir.store` `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              attr-dict
              (`atomic` `=` $atomic_kind^)?
              (`->` type($result_tensor)^)?
```

将本地缓冲区上的数据存储到全局内存。
目前仅支持存储统一缓冲区上的数据。

示例：
```mlir
hivm.store ins(%src : memref<16x16xf16, #hivm.address_space<ub>>) outs(%dst : memref<16x16xf16, #hivm.address_space<gm>>)
```

约束：
- `src` 和 `dst` 应具有相同的元素类型。
- 如果设置了 `atomic_kind`，类型是 `add`、`max`、`min` 之一。

Traits: `AlwaysSpeculatableImplTrait`, `OpPipeTrait<PIPE::PIPE_MTE3>`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`

Interfaces: `ConditionallySpeculatable`, `CopyOpInterface`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `InferCoreTypeInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>atomic_kind</code></td><td>::mlir::hivm::AtomicKindAttr</td><td><details><summary>Atomic Operation Kind for StoreOp</summary>{{% markdown %}}
    HIVM atomic store kind attribute.
  {{% /markdown %}}</details></td></tr>
<tr><td><code>may_implicit_transpose_with_last_axis</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 张量或内存引用
| `dst` | 张量或内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result_tensor` | 任意类型的张量


### `hivm.hir.sync_block` (hivm::SyncBlockOp)

_不同内核之间的 Hivm 同步块_

语法：

```
operation ::= `hivm.hir.sync_block` attr-dict `[` $sync_block_mode (`,` $flag_id^)?`]`
              (`ffts_base_addr` `=` $ffts_base_addr^)?
              (`tcube_pipe` `=` $tcube_pipe^)?
              (`tvector_pipe` `=` $tvector_pipe^)?
```

有以下同步块模式：
  - ALL_CUBE : 所有 cube 同步到同一点。
               `tcube_pipe` 需要设置为 cube 核心正在等待的管道。
  - ALL_VECTOR : 所有 vector 同步到同一点。
               `tvector_pipe` 需要设置为 vector 核心正在等待的管道。
  - ALL_SUB_VECTOR : 所有子 vector 核心同步到同一点。
  - BARRIER_CUBE : 用于 cube-cube 同步，它将被降级为 barrie.pipe_all，并且只会被复制到 aic 内核。
  - BARRIER_VECTOR : 用于 cube-cube 同步，它将被降级为 barrie.pipe_all，并且只会被复制到 aiv 内核。
  - ALL : 所有 aic/aiv 同步到同一点。
          `tvector_pipe` 需要设置为 vector 核心正在等待的管道。

注意：
  - SyncBlockOp 只能在数据移动到 gm 之后使用。
  - 在 Ascend910B 中必须设置 `$ffts_base_addr`。每次 FFTS 从所有子块收集一个特定的 `$flag_id` 时，FFTS 会将标志 ID 设置回组中的块以进行同步。

Interfaces: `InferCoreTypeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>sync_block_mode</code></td><td>::mlir::hivm::SyncBlockModeAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM synchronization block mode attribute.
  {{% /markdown %}}</details></td></tr>
<tr><td><code>flag_id</code></td><td>::mlir::IntegerAttr</td><td><details><summary>An Attribute containing a integer value</summary>{{% markdown %}}
    Syntax:

    ```
    integer-attribute ::= (integer-literal ( `:` (index-type | integer-type) )?)
                          | `true` | `false`
    ```

    An integer attribute is a literal attribute that represents an integral
    value of the specified integer or index type. `i1` integer attributes are
    treated as `boolean` attributes, and use a unique assembly format of either
    `true` or `false` depending on the value. The default type for non-boolean
    integer attributes, if a type is not specified, is signless 64-bit integer.

    Examples:

    ```mlir
    10 : i32
    10    // : i64 is implied here.
    true  // A bool, i.e. i1, value.
    false // A bool, i.e. i1, value.
    ```
  {{% /markdown %}}</details></td></tr>
<tr><td><code>tcube_pipe</code></td><td>::mlir::hivm::PipeAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM Op pipe attribute.
  {{% /markdown %}}</details></td></tr>
<tr><td><code>tvector_pipe</code></td><td>::mlir::hivm::PipeAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM Op pipe attribute.
  {{% /markdown %}}</details></td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `ffts_base_addr` | 64 位无符号整数


### `hivm.hir.sync_block_lock` (hivm::SyncBlockLockOp)

_同步块锁操作_

语法：

```
operation ::= `hivm.hir.sync_block_lock` attr-dict `lock_var` `(` $lock_var `:` type($lock_var) `)`
```

sync_block_lock 操作将不会释放，直到 lock_var 等于块索引。
示例：

```mlir
  hivm.hir.sync_block_lock lock_var(%lock : memref<1xi64>)
```

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `lock_var` | 1D memref of 64-bit signless integer values


### `hivm.hir.sync_block_set` (hivm::SyncBlockSetOp)

_Hivm 设置块同步_

语法：

```
operation ::= `hivm.hir.sync_block_set` attr-dict `[` $tcore_type `,` $tpipe `,` $pipe`]`
              `flag` `=` custom<FlagID>($static_flag_id, $dynamic_flag_id)
              (`ffts_base_addr` `=` $ffts_base_addr^)?
              (`sync_instr_mode` `=` $tsync_instr_mode^)?
```

Traits: `AttrSizedOperandSegments`

Interfaces: `InferCoreTypeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>tcore_type</code></td><td>::mlir::hivm::TCoreTypeAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM op core type attribute.
  {{% /markdown %}}</details></td></tr>
<tr><td><code>tpipe</code></td><td>::mlir::hivm::PipeAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM Op pipe attribute.
  {{% /markdown %}}</details></td></tr>
<tr><td><code>pipe</code></td><td>::mlir::hivm::PipeAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM Op pipe attribute.
  {{% /markdown %}}</details></td></tr>
<tr><td><code>static_flag_id</code></td><td>::mlir::IntegerAttr</td><td><details><summary>An Attribute containing a integer value</summary>{{% markdown %}}
    Syntax:

    ```
    integer-attribute ::= (integer-literal ( `:` (index-type | integer-type) )?)
                          | `true` | `false`
    ```

    An integer attribute is a literal attribute that represents an integral
    value of the specified integer or index type. `i1` integer attributes are
    treated as `boolean` attributes, and use a unique assembly format of either
    `true` or `false` depending on the value. The default type for non-boolean
    integer attributes, if a type is not specified, is signless 64-bit integer.

    Examples:

    ```mlir
    10 : i32
    10    // : i64 is implied here.
    true  // A bool, i.e. i1, value.
    false // A bool, i.e. i1, value.
    ```
  {{% /markdown %}}</details></td></tr>
<tr><td><code>tsync_instr_mode</code></td><td>::mlir::hivm::SyncBlockInstrModeAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM synchronization block instruction mode attribute.
  {{% /markdown %}}</details></td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `dynamic_flag_id` | 64 位无符号整数
| `ffts_base_addr` | 64 位无符号整数


### `hivm.hir.sync_block_unlock` (hivm::SyncBlockUnlockOp)

_同步块解锁操作_

语法：

```
operation ::= `hivm.hir.sync_block_unlock` attr-dict `lock_var` `(` $lock_var `:` type($lock_var) `)`
```

`sync_block_lock` 操作将增加并释放 lock_var。
示例：

```mlir
  hivm.hir.sync_block_unlock lock_var(%lock : memref<1xi64>)
```

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `lock_var` | 1D memref of 64-bit signless integer values


### `hivm.hir.sync_block_wait` (hivm::SyncBlockWaitOp)

_Hivm 等待块同步_

语法：

```
operation ::= `hivm.hir.sync_block_wait` attr-dict `[` $tcore_type `,` $tpipe `,` $pipe`]`
              `flag` `=` custom<FlagID>($static_flag_id, $dynamic_flag_id)
```

Interfaces: `InferCoreTypeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>tcore_type</code></td><td>::mlir::hivm::TCoreTypeAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM op core type attribute.
  {{% /markdown %}}</details></td></tr>
<tr><td><code>tpipe</code></td><td>::mlir::hivm::PipeAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM Op pipe attribute.
  {{% /markdown %}}</details></td></tr>
<tr><td><code>pipe</code></td><td>::mlir::hivm::PipeAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM Op pipe attribute.
  {{% /markdown %}}</details></td></tr>
<tr><td><code>static_flag_id</code></td><td>::mlir::IntegerAttr</td><td><details><summary>An Attribute containing a integer value</summary>{{% markdown %}}
    Syntax:

    ```
    integer-attribute ::= (integer-literal ( `:` (index-type | integer-type) )?)
                          | `true` | `false`
    ```

    An integer attribute is a literal attribute that represents an integral
    value of the specified integer or index type. `i1` integer attributes are
    treated as `boolean` attributes, and use a unique assembly format of either
    `true` or `false` depending on the value. The default type for non-boolean
    integer attributes, if a type is not specified, is signless 64-bit integer.

    Examples:

    ```mlir
    10 : i32
    10    // : i64 is implied here.
    true  // A bool, i.e. i1, value.
    false // A bool, i.e. i1, value.
    ```
  {{% /markdown %}}</details></td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `dynamic_flag_id` | 64 位无符号整数


### `hivm.hir.vabs` (hivm::VAbsOp)

_逐元素向量绝对值操作_

语法：

```
operation ::= `hivm.hir.vabs` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
  1. 输入/init 操作数和结果具有相同的元素类型。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型形状值
| `dst` | 可变数量的任意类型形状值
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vadd` (hivm::VAddOp)

_逐元素二元向量加法操作_

语法：

```
operation ::= `hivm.hir.vadd` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
  1. 输入/init 操作数和结果具有相同的元素类型。
  2. 支持向量-向量和向量-标量操作。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `CommutativeOpTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型
| `dst` | 可变数量的任意类型形状值
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vand` (hivm::VAndOp)

_逐元素二元向量与操作_

语法：

```
operation ::= `hivm.hir.vand` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
  1. 输入/init 操作数和结果具有相同的元素类型。
  2. 仅支持向量-向量操作。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `CommutativeOpTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`, `VectorOnlyTrait<1>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型
| `dst` | 可变数量的任意类型形状值
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.varange` (hivm::VArangeOp)

_向量 Arange 操作_

语法：

```
operation ::= `hivm.hir.varange` attr-dict
              (`offset` `[` $offset^ `]`)?
              `strides` `[` $strides `]`
              `outs` `(` $dst `:` type($dst) `)`
              (`->` type($result)^)?
```

根据步长和偏移量填充一个向量，范围为 0,1,2...。
例如 offset = 1, strides = [1, 2], tensor/memref shape = [2x4xi32]，
结果是 [[1, 3, 5, 7,
          2, 4, 6, 8]]。

约束：
  1. 必须至少有一个步长。
  2. 默认偏移量为 0。

示例：
```mlir
hivm.hir.varange offset[%o] strides[%s0, %s1] outs(%dst : memref<32xf32>)
%result = hivm.hir.varange offset[%o] strides[%s0, %s1] outs(%dst : tensor<32xf32>)
                            -> tensor<32xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `VectorCoreTypeTrait`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `dst` | 张量或内存引用
| `offset` | 索引
| `strides` | 可变数量的索引

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 任意类型的张量


### `hivm.hir.vbrc` (hivm::VBrcOp)

_向量广播操作_

语法：

```
operation ::= `hivm.hir.vbrc` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast_dims` `=` $broadcast_dims^)?
              (`->` type($result)^)?
```

根据广播轴数组广播一个向量或标量。

约束：
  1. 输入向量和输出向量必须具有相同的秩和相同的元素类型。
  2. 对于输入操作数，被广播轴的大小必须为 1。
  3. 对于向量输入，广播索引数组不能为空。
  4. 对于标量输入，广播索引数组*必须*为空。
  5. 广播索引数组不能大于输入向量的秩。
  6. 广播索引必须在 `[0, RankOfSrcVec)` 范围内。
  7. 对于 i1 类型，需要确保 dst 的尾部轴与 16 对齐，
     否则会有内存踩踏的风险

示例：
```mlir
// Scalar broadcast
hivm.hir.vbrc ins(%src : i32) outs(%dst : memref<?xi32>)
// Vector broadcast
hivm.hir.vbrc ins(%src : memref<1xi32>) outs(%dst : memref<?xi32>) broadcast_dims = [0]
%result = hivm.hir.vbrc ins(%src : tensor<1xi32>) outs(%dst : tensor<?xi32>) broadcast_dims = [0] -> tensor<?xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `CollapsibleConsecutiveTargetDimsTrait`, `SameOperandsElementType`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`

Interfaces: `BiShengIRAggregatedOpInterface`, `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `InferCoreTypeInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>broadcast_dims</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 任意类型
| `dst` | 张量或内存引用
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vcast` (hivm::VCastOp)

_逐元素向量类型转换操作_

语法：

```
operation ::= `hivm.hir.vcast` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`round_mode` `=` $round_mode^)?
              (`cast` `=` $cast^)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
   1. 支持以下转换：

   | src  | dst  | roundingmode                                      |
   |------|------|---------------------------------------------------|
   | f32  | f32  | round, rint, floor, ceil, trunc                   |
   | f32  | f16  | round, rint, floor, ceil, trunc, odd              |
   | f32  | i64  | round, rint, floor, ceil, trunc                   |
   | f32  | i32  | round, rint, floor, ceil, trunc                   |
   | f32  | i16  | round, rint, floor, ceil, trunc                   |
   | f32  | s64  | round, rint, floor, ceil, trunc                   |
   | f32  | bf16 | round, rint, floor, ceil, trunc                   |
   | f16  | f32  | rint                                              |
   | f16  | i32  | round, rint, floor, ceil, trunc                   |
   | f16  | i16  | round, rint, floor, ceil, trunc                   |
   | f16  | i8   | round, rint, floor, ceil, trunc                   |
   | f16  | ui8  | round, rint, floor, ceil, trunc                   |
   | f16  | i4   | round, rint, floor, ceil, trunc                   |
   | bf16 | f32  | rint                                              |
   | bf16 | i32  | round, rint, floor, ceil, trunc                   |
   | ui8  | f16  | rint                                              |
   | i8   | f16  | rint                                              |
   | i8   | i1   | rint                                              |
   | i16  | f16  | round, rint, floor, ceil, trunc                   |
   | i16  | f32  | rint                                              |
   | i32  | f32  | round, rint, floor, ceil, trunc                   |
   | i32  | i64  | rint                                              |
   | i32  | i16  | rint                                              |
   | i64  | i32  | rint                                              |
   | i64  | f32  | round, rint, floor, ceil, trunc                   |
   | i4   | f16  | rint                                              |
   | i1   | f16  | rint                                              |
   | i1   | f32  | rint                                              |

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>round_mode</code></td><td>::mlir::hivm::RoundModeAttr</td><td><details><summary>Round Mode for VCastOp</summary>{{% markdown %}}
    - RINT: round to nearest, tie to even (c language rint)
    - ROUND: round to nearest, tie away from zero (c language round)
    - FLOOR: round to minus infinity (c language floor)
    - CEIL: round to positive infinity (c language ceil)
    - TRUNC: round to zero (c language trunc)
    - ODD: round to odd (Von Neumann rounding)
  {{% /markdown %}}</details></td></tr>
<tr><td><code>cast</code></td><td>::mlir::hivm::TypeFnAttr</td><td><details><summary>Cast for VCastOp</summary>{{% markdown %}}
    HIVM cast attribute.
  {{% /markdown %}}</details></td></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型
| `dst` | 可变数量的任意类型形状值
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vcmp` (hivm::VCmpOp)

_逐元素二元向量比较操作_

语法：

```
operation ::= `hivm.hir.vcmp` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`compare_mode` `=` $compare_mode^)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

比较两个源向量的元素。如果比较结果为 true，
`dst` 的对应位为 1 或 8。

附加约束：
  1. 输入向量和输出向量必须具有相同的秩
  2. `dst` 的元素类型必须为 bool
  3. 输入是仅向量的。
  4. 支持以下数据类型：

   |    compare mode   |       element type      |
   |-------------------|-------------------------|
   | GE/GT/LE/LT/NE/EQ | f16, f32, i16, i32, i64 |

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>compare_mode</code></td><td>::mlir::hivm::CompareModeAttr</td><td><details><summary>Compare Mode for VCmpOp</summary>{{% markdown %}}
    HIVM compare mode attribute.
  {{% /markdown %}}</details></td></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型
| `dst` | 可变数量的任意类型形状值

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vconcat` (hivm::VConcatOp)

_向量连接操作_

语法：

```
operation ::= `hivm.hir.vconcat` `dim` `(` $dim `)` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`->` type($result)^)?
```

concat 操作从输入张量的可变列表构造一个张量，沿静态维度号连接。所有输入和结果类型必须共享相同的秩。

`dim` 指定要连接的维度。结果中连接维度的大小必须等于输入沿该维度的尺寸之和。输入和结果中的所有其他维度必须具有相同的大小。

示例：
```mlir
hivm.hir.vconcat dim(1) ins(%0, %1 : tensor<136x2048xf32>, tensor<136x2048xf32>)
                        outs(%2 : tensor<136x4096xf32>) -> tensor<136x4096xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

Interfaces: `BiShengIRAggregatedOpInterface`, `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>dim</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型
| `dst` | 张量或内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vcos` (hivm::VCosOp)

_逐元素向量余弦操作_

语法：

```
operation ::= `hivm.hir.vcos` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
  1. 输入/init 操作数和结果具有相同的元素类型。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型
| `dst` | 可变数量的任意类型形状值

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vcumprod` (hivm::VCumprodOp)

_向量累积乘积操作_

语法：

```
operation ::= `hivm.hir.vcumprod` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `cum_dims` `=` $cum_dims
              `reverse` `=` $reverse
              (`->` type($result)^)?
```

计算 `src` 沿指定轴的每个元素的累积乘积。沿指定轴在 cumprod 输出中的每个元素包含从第一个元素到原始 `src` 中当前位置的所有元素的乘积。

约束：
  1. 输入向量和输出向量必须具有相同的秩和相同的元素类型。

参数：
  * `src`: 要计算累积乘积的张量/memref
  * `dst`: 存储元素的张量/memref
  * `cum_dims`: 指定要计算累积乘积的维度。
  * `reverse`: 指定累积乘积的方向。

示例：
```mlir
hivm.hir.vcumprod ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) cum_dims : [0] reverse = true
%result = hivm.hir.vcumprod ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) cum_dims : [0] reverse = true -> tensor<?xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>cum_dims</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute should be in increasing order</td></tr>
<tr><td><code>reverse</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 张量或内存引用
| `dst` | 张量或内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vcumsum` (hivm::VCumsumOp)

_向量累积和操作_

语法：

```
operation ::= `hivm.hir.vcumsum` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `cum_dims` `=` $cum_dims
              `reverse` `=` $reverse
              (`->` type($result)^)?
```

计算 `src` 沿指定轴的每个元素的累积和。沿指定轴在 cumsum 输出中的每个元素包含从第一个元素到原始 `src` 中当前位置的所有元素的和。

约束：
  1. 输入向量和输出向量必须具有相同的秩和相同的元素类型。

参数：
  * `src`: 要计算累积和的张量/memref
  * `dst`: 存储元素的张量/memref
  * `cum_dims`: 指定要计算累积和的维度。
  * `reverse`: 指定累积和的方向。

示例：
```mlir
hivm.hir.vcumsum ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) cum_dims : [0] reverse = true
%result = hivm.hir.vcumsum ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) cum_dims : [0] reverse = true -> tensor<?xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>cum_dims</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute should be in increasing order</td></tr>
<tr><td><code>reverse</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 张量或内存引用
| `dst` | 张量或内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vdeinterleave` (hivm::VDeinterleaveOp)

_向量解交错操作_

语法：

```
operation ::= `hivm.hir.vdeinterleave` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`channel_num` `=` $channel_num^)?
              (`index_mode` `=` $index_mode^)?
              (`->` type($result)^)?
```

沿最后一个维度解交错一个张量。
张量的最后一个维度大小必须是 `channel_num` 的倍数。

Traits: `AlwaysSpeculatableImplTrait`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

Interfaces: `BiShengIRAggregatedOpInterface`, `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>channel_num</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>index_mode</code></td><td>::mlir::hivm::DeinterleaveModeAttr</td><td><details><summary>HIVM deinterleave mode</summary>{{% markdown %}}HIVM deinterleave index mode{{% /markdown %}}</details></td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 张量或内存引用
| `dst` | 可变数量的张量或内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vdiv` (hivm::VDivOp)

_逐元素二元向量除法操作_

语法：

```
operation ::= `hivm.hir.vdiv` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
  1. 输入/init 操作数和结果具有相同的元素类型。
  2. 仅支持向量-向量操作。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型
| `dst` | 可变数量的任意类型形状值
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.verf` (hivm::VErfOp)

_逐元素向量误差函数操作_

语法：

```
operation ::= `hivm.hir.verf` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
  1. 输入/init 操作数和结果具有相同的元素类型。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型
| `dst` | 可变数量的任意类型形状值

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vexp` (hivm::VExpOp)

_逐元素向量指数操作_

语法：

```
operation ::= `hivm.hir.vexp` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
  1. 输入/init 操作数和结果具有相同的元素类型。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型形状值
| `dst` | 可变数量的任意类型形状值
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vflip` (hivm::VFlipOp)

_向量翻转操作_

语法：

```
operation ::= `hivm.hir.vflip` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `flip_axis` `=` $flip_axis
              (`->` type($result)^)?
```

沿最后一个维度翻转张量。

Traits: `AlwaysSpeculatableImplTrait`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>flip_axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 张量或内存引用
| `dst` | 张量或内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vgather` (hivm::VGatherOp)

_向量收集操作_

语法：

```
operation ::= `hivm.hir.vgather` attr-dict `ins` `(` $src `:` type($src) `)`
              `indices` `(` $indices `:` type($indices) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`->` type($result)^)?
```

根据给定的索引从张量/memref 中检索元素，
并将这些元素存储在另一个张量/memref 中。
收集轴是最后一个维度。

参数：
  * `src`: 要从中收集元素的张量/memref
  * `indices`: 要从 src 收集的元素的索引
  * `dst`: 存储元素的张量/memref
  * `temp_buffer`: gather 操作所需的额外内存

Traits: `AlwaysSpeculatableImplTrait`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 张量或内存引用
| `indices` | 张量或内存引用
| `dst` | 张量或内存引用
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vinterleave` (hivm::VInterleaveOp)

_向量交错操作_

语法：

```
operation ::= `hivm.hir.vinterleave` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `interleave_channel_nums` `=` $interleave_channel_nums
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`->` type($result)^)?
```

沿最后一个维度交错 `N` 个张量的值。
N 个张量必须具有相同的形状。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>interleave_channel_nums</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型
| `dst` | 张量或内存引用
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vln` (hivm::VLnOp)

_逐元素向量自然对数操作_

语法：

```
operation ::= `hivm.hir.vln` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
  1. 输入/init 操作数和结果具有相同的元素类型。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型形状值
| `dst` | 可变数量的任意类型形状值
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vmax` (hivm::VMaxOp)

_逐元素二元向量最大值操作_

语法：

```
operation ::= `hivm.hir.vmax` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
  1. 输入/init 操作数和结果具有相同的元素类型。
  2. 支持向量-向量和向量-标量操作。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `CommutativeOpTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型
| `dst` | 可变数量的任意类型形状值
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vmin` (hivm::VMinOp)

_逐元素二元向量最小值操作_

语法：

```
operation ::= `hivm.hir.vmin` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
  1. 输入/init 操作数和结果具有相同的元素类型。
  2. 支持向量-向量和向量-标量操作。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `CommutativeOpTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型
| `dst` | 可变数量的任意类型形状值
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vmod` (hivm::VModOp)

_逐元素向量取模操作_

语法：

```
operation ::= `hivm.hir.vmod` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
  1. 输入/init 操作数和结果具有相同的元素类型。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型
| `dst` | 可变数量的任意类型形状值

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vmul` (hivm::VMulOp)

_逐元素二元向量乘法操作_

语法：

```
operation ::= `hivm.hir.vmul` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
  1. 输入/init 操作数和结果具有相同的元素类型。
  2. 支持向量-向量和向量-标量操作。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `CommutativeOpTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型
| `dst` | 可变数量的任意类型形状值
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vmulext` (hivm::VMulExtOp)

_逐元素二元向量乘法操作，计算最高有效 32 位_

语法：

```
operation ::= `hivm.hir.vmulext` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
  1. 输入/init 操作数和结果具有相同的元素类型。
  2. 支持向量-向量操作。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型
| `dst` | 可变数量的任意类型形状值

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vmulextended` (hivm::VMulextendedOp)

_向量扩展乘法操作_

语法：

```
operation ::= `hivm.hir.vmulextended` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`->` type($result)^)?
```

对两个张量执行 vmul。获取高 16 位和低 16 位。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的张量或内存引用
| `dst` | 可变数量的张量或内存引用
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vnot` (hivm::VNotOp)

_逐元素向量非操作_

语法：

```
operation ::= `hivm.hir.vnot` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
  1. 输入/init 操作数和结果具有相同的元素类型。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型形状值
| `dst` | 可变数量的任意类型形状值
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vor` (hivm::VOrOp)

_逐元素二元向量或操作_

语法：

```
operation ::= `hivm.hir.vor` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
  1. 输入/init 操作数和结果具有相同的元素类型。
  2. 仅支持向量-向量操作。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `CommutativeOpTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`, `VectorOnlyTrait<1>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型
| `dst` | 可变数量的任意类型形状值
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vpad` (hivm::VPadOp)

_向量填充操作_

语法：

```
operation ::= `hivm.hir.vpad` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `low` `` custom<DynamicIndexList>($low, $static_low)
              `high` `` custom<DynamicIndexList>($high, $static_high)
              `pad_value` $pad_value `:` type($pad_value)
              (`->` type($result)^)?
```

填充输入操作数。操作语义类似于
`tensor.pad`。

参数：
  * `src`: 要填充值的张量/memref
  * `dst`: 为缓冲化保留
  * `pad_value`: 要填充的值
  * `low`: 沿每个维度开始处的填充长度
  * `high`: 沿每个维度结束处的填充长度

示例：
```mlir
hivm.hir.vpad ins(%src : tensor<2x16xf32>) outs(%dst: tensor<?x16xf32>)
              low[%first_dim_low, 0] high[%first_dim_high, 0]
              pad_value %pad_value : f32
                -> tensor<?x16xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

Interfaces: `BiShengIRAggregatedOpInterface`, `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>static_low</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>static_high</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 张量或内存引用
| `dst` | 张量或内存引用
| `pad_value` | 任意类型
| `low` | 可变数量的索引
| `high` | 可变数量的索引

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vpow` (hivm::VPowOp)

_逐元素二元向量幂操作_

语法：

```
operation ::= `hivm.hir.vpow` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
  1. 输入/init 操作数和结果具有相同的元素类型。
  2. 支持向量-向量和向量-标量操作。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`, `VectorOnlyTrait<1>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型
| `dst` | 可变数量的任意类型形状值
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vrec` (hivm::VRecOp)

_逐元素向量倒数操作_

语法：

```
operation ::= `hivm.hir.vrec` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
  1. 输入/init 操作数和结果具有相同的元素类型。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型形状值
| `dst` | 可变数量的任意类型形状值
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vreduce` (hivm::VReduceOp)

_向量归约操作_

语法：

```
operation ::= `hivm.hir.vreduce` attr-dict $arith `ins` `(` $src `:` type($src) `)`
              (`indices` `(` $indices^ `:` type($indices) `)`)?
              `outs` `(` $dst `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              `reduce_dims` `=` $reduce_dims
              (`->` type($result)^)?
```

根据归约轴数组归约源向量的一个或多个轴，从初始值开始。

约束：
  1. 输入向量和输出向量必须具有相同的秩和相同的元素类型。
  2. 对于输出操作数，归约轴的大小必须为 1。
  3. 归约索引数组不能为空，也不能大于输入向量的秩。
  4. 归约索引必须在 `[0, RankOfDstVec)` 范围内。

示例：
```mlir
hivm.hir.vreduce <add> ins(%src : memref<?xf32>) outs(%dst : memref<1xf32>) reduce_dims : [1]
%result = hivm.hir.vreduce <max> ins(%src : tensor<?xf32>) outs(%dst : tensor<1xf32>) reduce_dims : [0] -> tensor<1xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

Interfaces: `BiShengIRAggregatedOpInterface`, `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>arith</code></td><td>::mlir::hivm::ReduceOpAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM reduction arithmetic operation attribute.
  {{% /markdown %}}</details></td></tr>
<tr><td><code>reduce_dims</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 张量或内存引用
| `dst` | 可变数量的张量或内存引用
| `temp_buffer` | 任意类型的内存引用
| `indices` | 张量或内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vrelu` (hivm::VReluOp)

_逐元素向量修正线性单元操作_

语法：

```
operation ::= `hivm.hir.vrelu` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
  1. 输入/init 操作数和结果具有相同的元素类型。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型形状值
| `dst` | 可变数量的任意类型形状值
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vrsqrt` (hivm::VRsqrtOp)

_逐元素向量倒数平方根操作_

语法：

```
operation ::= `hivm.hir.vrsqrt` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
  1. 输入/init 操作数和结果具有相同的元素类型。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型形状值
| `dst` | 可变数量的任意类型形状值
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vsel` (hivm::VSelOp)

_逐元素向量选择操作_

语法：

```
operation ::= `hivm.hir.vsel` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

根据二元 `condition` 向量从两个源向量中选择元素。
如果指示符的对应位为 1，则选择 `src0`。否则，
选择 `src1`。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<3>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型
| `dst` | 可变数量的任意类型形状值
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vshl` (hivm::VShLOp)

_逐元素二元向量左移操作_

语法：

```
operation ::= `hivm.hir.vshl` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
  1. 输入向量和结果具有相同的元素类型。
  2. 仅支持向量-标量操作。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `ScalarOnlyHWTrait<1>`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型
| `dst` | 可变数量的任意类型形状值
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vshr` (hivm::VShROp)

_逐元素二元向量右移操作_

语法：

```
operation ::= `hivm.hir.vshr` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`round` `:` $round^ )?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
  1. 输入向量和结果具有相同的元素类型。
  2. 仅支持向量-标量操作。
  3. 如果 `round` 设置为 true，则在算术右移期间应用舍入。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `ScalarOnlyHWTrait<1>`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>round</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型
| `dst` | 可变数量的任意类型形状值
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vsin` (hivm::VSinOp)

_逐元素向量正弦操作_

语法：

```
operation ::= `hivm.hir.vsin` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
  1. 输入/init 操作数和结果具有相同的元素类型。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型
| `dst` | 可变数量的任意类型形状值

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vsort` (hivm::VSortOp)

_向量排序操作_

语法：

```
operation ::= `hivm.hir.vsort` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `descending` `=` $descending
              `sort_axis` `=` $sort_axis
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`->` type($result)^)?
```

按升序或降序对 `src` 的排序轴进行排序，并输出排序后的值和对应的索引。

约束：
  1. 输入向量和输出向量必须具有相同的秩。
  2. 目前仅支持尾部轴排序。

参数：
  * `src`: 要排序的张量/memref
  * `dst_value`: 存储排序值的张量/memref
  * `dst_index`: 存储对应于 dst_value 的索引的张量/memref
  * `descending`: 确定是按升序还是降序排序。默认为 false，表示升序
  * `sort_axis`: 要排序的轴

示例：
```mlir
hivm.hir.vsort ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) descending = true sort_axis = 0
%result = hivm.hir.vsort ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) descending = true sort_axis = 0 -> tensor<?xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `VectorCoreTypeTrait`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>descending</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>sort_axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 张量或内存引用
| `dst` | 可变数量的张量或内存引用
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vsqrt` (hivm::VSqrtOp)

_逐元素向量平方根操作_

语法：

```
operation ::= `hivm.hir.vsqrt` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
  1. 输入/init 操作数和结果具有相同的元素类型。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型形状值
| `dst` | 可变数量的任意类型形状值
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vsub` (hivm::VSubOp)

_逐元素二元向量减法操作_

语法：

```
operation ::= `hivm.hir.vsub` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
  1. 输入/init 操作数和结果具有相同的元素类型。
  2. 支持向量-向量和向量-标量操作。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型
| `dst` | 可变数量的任意类型形状值
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vtanh` (hivm::VTanhOp)

_逐元素向量双曲正切操作_

语法：

```
operation ::= `hivm.hir.vtanh` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
  1. 输入/init 操作数和结果具有相同的元素类型。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型形状值
| `dst` | 可变数量的任意类型形状值
| `temp_buffer` | 任意类型的内存引用

#### 结果:
hivm.hir.vxor
| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vtranspose` (hivm::VTransposeOp)

_向量转置操作_

语法：

```
operation ::= `hivm.hir.vtranspose` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`permutation` `=` $permutation^)?
              (`disable_align` `=` $disable_align^)?
              (`->` type($result)^)?
```

根据给定的 `permutation` 置换 `src` 的维度。换句话说：
  `dim(dst, i) = dim(src, permutation[i])`。

约束：
  1. 输入向量和输出向量必须具有相同的秩和相同的元素类型。

示例：
```mlir
 hivm.hir.vtranspose ins(%src : memref<32x8xf32>) outs(%dst : memref<8x32xf32>) permutation = [1, 0]
 %result = hivm.hir.vtranspose ins(%src : tensor<32x8xf32>) outs(%dst: tensor<8x32xf32>) permutation = [1, 0] -> tensor<8x32xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

Interfaces: `BiShengIRAggregatedOpInterface`, `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>permutation</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>disable_align</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 张量或内存引用
| `dst` | 张量或内存引用
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.vxor` (hivm::VXorOp)

_逐元素二元向量异或操作_

语法：

```
operation ::= `hivm.hir.vxor` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

*来自逐元素 N 元向量操作模板：*

此操作对 N 个操作数执行逐元素操作并产生单个结果。
它可以在过程中执行转置或广播（但不能同时执行）。

通用约束：
  1. 遵循 DestinationStyleOpInterface。
  2. 输入操作数的数量为 N；输出/结果的数量为一个。
  3. 输入/init 操作数和结果具有相同的秩。
  4. 第一个输入是仅向量的。

附加约束：
  1. 输入/init 操作数和结果具有相同的元素类型。
  2. 仅支持向量-向量操作。

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `CommutativeOpTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`, `VectorOnlyTrait<1>`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>transpose</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
<tr><td><code>broadcast</code></td><td>::mlir::DenseI64ArrayAttr</td><td>i64 dense array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 可变数量的任意类型
| `dst` | 可变数量的任意类型形状值
| `temp_buffer` | 任意类型的内存引用

#### 结果:

| 结果 | 描述 |
| :----: | ----------- |
| `result` | 可变数量的任意类型张量


### `hivm.hir.wait_flag` (hivm::WaitFlagOp)

_Hivm 等待标志_

语法：

```
operation ::= `hivm.hir.wait_flag` `[`
              $set_pipe
              `,` $wait_pipe
              `,` custom<EventID>($static_event_id, $dynamic_event_id)
              `]` attr-dict
```


Interfaces: `InferCoreTypeInterface`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>set_pipe</code></td><td>::mlir::hivm::PipeAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM Op pipe attribute.
  {{% /markdown %}}</details></td></tr>
<tr><td><code>wait_pipe</code></td><td>::mlir::hivm::PipeAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM Op pipe attribute.
  {{% /markdown %}}</details></td></tr>
<tr><td><code>static_event_id</code></td><td>::mlir::hivm::EventAttr</td><td><details><summary></summary>{{% markdown %}}
    HIVM event attribute for synchronization.
  {{% /markdown %}}</details></td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `dynamic_event_id` | 64 位无符号整数


## Attributes

### AddressSpaceAttr

语法：

```
#hivm.address_space<
  ::mlir::hivm::AddressSpace   # address_space
>
```

HIVM 地址空间映射属性。映射到 GM、L1、L0A、L0B、L0C 和 UB。

#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| address_space | `::mlir::hivm::AddressSpace` | 类型为 AddressSpace |

### AlignKindAttr

对齐类型信息

语法：

```
#hivm.align_kind<
  ::mlir::hivm::AlignKind   # value
>
```

HIVM 对齐类型属性。
#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hivm::AlignKind` | 类型为 AlignKind |

### AllocAlignDimsAttr

语法: `#hivm.alloc_align_dims`

HIVM 分配对齐维度。

### AllocAlignValueInByteAttr

语法: `#hivm.alloc_align_value_in_byte`

HIVM 分配对齐值（字节）。

### AtomicKindAttr

原子操作类型（用于 StoreOp）

语法：

```
#hivm.atomic_kind<
  ::mlir::hivm::AtomicKind   # value
>
```

HIVM 原子存储类型属性。

#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hivm::AtomicKind` | 类型为 AtomicKind |

### AxisKindAttr

hivm 操作轴类型信息

语法：

```
#hivm.axis_kind<
  ::mlir::hivm::AxisKind   # value
>
```

HIVM 操作轴类型属性。
#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hivm::AxisKind` | 类型为 AxisKind |

### HIVMBlockMappingAttr

语法：

```
#hivm.block<
  std::optional<int32_t>   # order
>
```


#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| order | `std::optional<int32_t>` |  |

### CompareModeAttr

VCmpOp 的比较模式

语法：

```
#hivm.compare_mode<
  ::mlir::hivm::CompareMode   # value
>
```

HIVM 比较模式属性。

#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hivm::CompareMode` | 类型为 CompareMode |

### DCCIModeAttr

hivm dcci 模式

语法：

```
#hivm.DCCIMode<
  ::mlir::hivm::DCCIMode   # value
>
```

HIVM DCCI 模式属性。
#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hivm::DCCIMode` | 类型为 DCCIMode |

### DataCacheKindAttr

hivm 数据缓存类型

语法：

```
#hivm.DataCacheKind<
  ::mlir::hivm::DataCacheKind   # value
>
```

HIVM 数据缓存类型属性。
#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hivm::DataCacheKind` | 类型为 DataCacheKind |

### DataLayoutAttr

语法：

```
#hivm.data_layout<
  ::mlir::hivm::DataLayout,   # data_layout
  std::optional<bool>,   # transpose
  std::optional<DenseI64ArrayAttr>   # fractalSizes
>
```

HIVM 数据布局映射属性。映射到 DOTA_ND、DOTB_ND、DOTC_ND、zN、nZ 和 ND。
  - `transpose`: 指示布局是转置的。
                 仅对 DOTA_ND 和 DOTB_ND 布局有效且必须存在。

#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| data_layout | `::mlir::hivm::DataLayout` | 类型为 DataLayout |
| transpose | `std::optional<bool>` |  |
| fractalSizes | `std::optional<DenseI64ArrayAttr>` |  |

### DeinterleaveModeAttr

HIVM 解交错模式

语法：

```
#hivm.deinterleave_mode<
  ::mlir::hivm::DeinterleaveMode   # value
>
```

HIVM 解交错索引模式
#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hivm::DeinterleaveMode` | 类型为 DeinterleaveMode |

### DescaleModeAttr

matmul 的 descale 模式

语法：

```
#hivm.descale_mode<
  ::mlir::hivm::DescaleMode   # value
>
```

HIVM matmul 操作的 descale 模式属性。

#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hivm::DescaleMode` | 类型为 DescaleMode |

### DisableAutoInjectBlockSyncAttr

语法: `#hivm.disable_auto_inject_block_sync`

禁用自动注入块同步，跳过块同步注入。

### EventAttr

语法：

```
#hivm.event<
  ::mlir::hivm::EVENT   # event
>
```

用于同步的 HIVM 事件属性。

#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| event | `::mlir::hivm::EVENT` | 类型为 EVENT |

### FixpipePreQuantModeAttr

HIVM fixpipe pre_quant 模式

语法：

```
#hivm.fixpipe_pre_quant_mode<
  ::mlir::hivm::FixpipePreQuantMode   # value
>
```

HIVM fixpipe pre_quant 模式
#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hivm::FixpipePreQuantMode` | 类型为 FixpipePreQuantMode |

### FixpipePreReluModeAttr

HIVM fixpipe pre_relu 模式

语法：

```
#hivm.fixpipe_pre_relu_mode<
  ::mlir::hivm::FixpipePreReluMode   # value
>
```

HIVM fixpipe pre_relu 模式
#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hivm::FixpipePreReluMode` | 类型为 FixpipePreReluMode |

### HIVMFuncDynMemrefArgsAttr

语法: `#hivm.func_dyn_memref_args`

HIVM FuncDynMemrefArgs 用于标记函数的动态
memref 参数的索引数组。

### InsertSliceSourceIndexAttr

语法: `#hivm.insert_slice_source_index`

指定 vconcat 操作中哪个操作数是 insert_slice 源

### MultiBufferAttr

语法: `#hivm.multi_buffer`

HIVM 多缓冲区属性。

### PadModeAttr

语法：

```
#hivm.padmode<
  ::mlir::hivm::PadMode   # padmode
>
```

HIVM 填充模式属性。

#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| padmode | `::mlir::hivm::PadMode` | 类型为 PadMode |

### ParallelLoopAttr

语法: `#hivm.parallel_loop`

指定标记的循环可以并行运行。

### PipeAttr

语法：

```
#hivm.pipe<
  ::mlir::hivm::PIPE   # pipe
>
```

HIVM 操作管道属性。

#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| pipe | `::mlir::hivm::PIPE` | 类型为 PIPE |

### ReduceOpAttr

语法：

```
#hivm.reduce_op<
  ::mlir::hivm::ReduceOperation   # reduce_op
>
```

HIVM 归约算术操作属性。

#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| reduce_op | `::mlir::hivm::ReduceOperation` | 类型为 ReduceOperation |

### RoundModeAttr

VCastOp 的舍入模式

语法：

```
#hivm.round_mode<
  ::mlir::hivm::RoundMode   # value
>
```

- RINT: 舍入到最近，平局取偶（C 语言 rint）
- ROUND: 舍入到最近，平局远离零（C 语言 round）
- FLOOR: 舍入到负无穷（C 语言 floor）
- CEIL: 舍入到正无穷（C 语言 ceil）
- TRUNC: 舍入到零（C 语言 trunc）
- ODD: 舍入到奇数（Von Neumann 舍入）

#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hivm::RoundMode` | 类型为 RoundMode |

### StorageAlignedAttr

语法: `#hivm.storage_aligned`

如果模块标记了此属性，则表示此模块中所有设备函数内的所有操作都已对齐。
如果函数标记了此属性，则表示此函数中的所有操作都已对齐。

### StrideAlignDimsAttr

语法: `#hivm.stride_align_dims`

HIVM 步长对齐维度。

### StrideAlignValueInByteAttr

语法: `#hivm.stride_align_value_in_byte`

HIVM 步长对齐值（字节）。

### HIVMSubBlockMappingAttr

语法：

```
#hivm.sub_block<
  ::mlir::hivm::MappingId   # sub_block
>
```

用于 mix func 的 cv 块维度比率的 HIVM 子块映射属性。

#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| sub_block | `::mlir::hivm::MappingId` | 类型为 MappingId |

### SyncBlockInstrModeAttr

语法：

```
#hivm.sync_block_instr_mode<
  ::mlir::hivm::SyncBlockInstrMode   # sync_instr_mode
>
```

HIVM 同步块指令模式属性。

#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| sync_instr_mode | `::mlir::hivm::SyncBlockInstrMode` | 类型为 SyncBlockInstrMode |

### SyncBlockModeAttr

语法：

```
#hivm.sync_block_mode<
  ::mlir::hivm::SyncBlockMode   # sync_mode
>
```

HIVM 同步块模式属性。

#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| sync_mode | `::mlir::hivm::SyncBlockMode` | 类型为 SyncBlockMode |

### TCoreTypeAttr

语法：

```
#hivm.tcore_type<
  ::mlir::hivm::TCoreType   # tcoretype
>
```

HIVM 操作核心类型属性。

#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| tcoretype | `::mlir::hivm::TCoreType` | 类型为 TCoreType |

### TCoreTypeMarkerAttr

语法：

```
#hivm.tcore_type_marker<
  ::mlir::hivm::TCoreType   # tcoretype
>
```

HIVM 操作核心类型标记属性。

#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| tcoretype | `::mlir::hivm::TCoreType` | 类型为 TCoreType |

### TFuncCoreTypeAttr

语法：

```
#hivm.func_core_type<
  ::mlir::hivm::TFuncCoreType   # funcCoreType
>
```

HIVM 函数核心类型属性。

#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| funcCoreType | `::mlir::hivm::TFuncCoreType` | 类型为 TFuncCoreType |

### TModuleCoreTypeAttr

语法：

```
#hivm.module_core_type<
  ::mlir::hivm::TModuleCoreType   # moduleCoreType
>
```

HIVM 模块核心类型属性。

如果模块内的所有函数都具有 `AIV` 函数核心类型，则模块核心类型为 `AIV`。

如果模块内的所有函数都具有 `AIC` 函数核心类型，则模块核心类型为 `AIC`。

否则，模块核心类型为 `MIX`。

#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| moduleCoreType | `::mlir::hivm::TModuleCoreType` | 类型为 TModuleCoreType |

### TPartOfMixAttr

语法: `#hivm.part_of_mix`

HIVM 函数是 mix 内核的一部分。

### TypeFnAttr

VCastOp 的转换

语法：

```
#hivm.cast<
  ::mlir::hivm::TypeFn   # value
>
```

HIVM 转换属性。

#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hivm::TypeFn` | 类型为 TypeFn |

### UnitFlagAttr

语法：

```
#hivm.unit_flag<
  ::mlir::hivm::UNIT_FLAG   # unit_flag
>
```

用于同步的 HIVM 单元标志属性。

#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| unit_flag | `::mlir::hivm::UNIT_FLAG` | 类型为 UNIT_FLAG |

### UnlikelyConditionAttr

语法: `#hivm.unlikely_condition`

指定标记的条件不太可能评估为 true。

### VFModeAttr

HIVM VF 模式

语法：

```
#hivm.vf_mode<
  ::mlir::hivm::VFMode   # value
>
```

HIVM VF 模式属性。

#### 参数:

| 参数 | C++ 类型 | 描述 |
| :-------: | :-------: | ----------- |
| value | `::mlir::hivm::VFMode` | 类型为 VFMode |


## Enums

### AddressSpace

HIVM 地址空间

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| Zero | `0` | zero |
| GM | `1` | gm |
| L1 | `2` | cbuf |
| L0A | `3` | ca |
| L0B | `4` | cb |
| L0C | `5` | cc |
| UB | `6` | ub |

### AlignKind

对齐类型信息

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| ALIGN | `0` | align |
| UNALIGNED | `1` | unaligned |
| UNKNOWN | `2` | unknown |

### AtomicKind

原子操作类型（用于 StoreOp）

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| NONE | `0` | none |
| ADD | `1` | add |
| MAX | `2` | max |
| MIN | `3` | min |
| AND | `4` | and |
| OR | `5` | or |
| XOR | `6` | xor |
| CAS | `7` | or |
| XCHG | `8` | xor |
| UMAX | `9` | umax |
| UMIN | `10` | umin |

### AxisKind

hivm 操作轴类型信息

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| FIRST | `0` | first |
| MIDDLE | `1` | middle |
| LAST | `2` | last |

### CompareMode

VCmpOp 的比较模式

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| EQ | `0` | eq |
| NE | `1` | ne |
| LT | `2` | lt |
| GT | `3` | gt |
| GE | `4` | ge |
| LE | `5` | le |

### DCCIMode

hivm dcci 模式

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| SINGLE_CACHE_LINE | `0` | single_cache_line |
| ALL_CACHE_LINES | `1` | all_cache_lines |

### DataCacheKind

hivm 数据缓存类型

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| ALL | `0` | all |
| UB | `1` | ub |
| OUT | `2` | out |
| ATOMIC | `3` | atomic |

### DataLayout

HIVM 数据布局

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| DOTA_ND | `1` | dotA_ND |
| DOTB_ND | `2` | dotB_ND |
| DOTC_ND | `3` | dotC_ND |
| nZ | `4` | nZ |
| zN | `5` | zN |
| ND | `6` | ND |

### DeinterleaveMode

HIVM 解交错模式

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| CHANNEL_0 | `0` | CHANNEL_0 |
| CHANNEL_1 | `1` | CHANNEL_1 |
| ALL_CHANNELS | `999` | ALL_CHANNELS |

### DescaleMode

matmul 的 descale 模式

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| DescaleNull | `0` | DescaleNull |
| DescalePerChannel | `1` | DescalePerChannel |
| DescalePerTensor | `2` | DescalePerTensor |

### EVENT

用于同步的事件 ID

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| EVENT_ID0 | `0` | EVENT_ID0 |
| EVENT_ID1 | `1` | EVENT_ID1 |
| EVENT_ID2 | `2` | EVENT_ID2 |
| EVENT_ID3 | `3` | EVENT_ID3 |
| EVENT_ID4 | `4` | EVENT_ID4 |
| EVENT_ID5 | `5` | EVENT_ID5 |
| EVENT_ID6 | `6` | EVENT_ID6 |
| EVENT_ID7 | `7` | EVENT_ID7 |

### FixpipePreQuantMode

HIVM fixpipe pre_quant 模式

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| NO_QUANT | `0` | NO_QUANT |
| S322I8 | `9` | S322I8 |
| F322F16 | `1` | F322F16 |
| F322BF16 | `16` | F322BF16 |

### FixpipePreReluMode

HIVM fixpipe pre_relu 模式

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| NO_RELU | `0` | NO_RELU |
| NORMAL_RELU | `1` | NORMAL_RELU |
| LEAKY_RELU | `2` | LEAKY_RELU |
| P_RELU | `3` | P_RELU |

### IteratorType

HIVM 结构化操作迭代器类型

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| kParallel | `0` | parallel |
| kBroadcast | `1` | broadcast |
| kTranspose | `2` | transpose |
| kReduction | `3` | reduction |
| kInterleave | `4` | interleave |
| kDeinterleave | `5` | deinterleave |
| kInverse | `6` | inverse |
| kPad | `7` | pad |
| kConcat | `8` | concat |
| kGather | `9` | gather |
| kCumulative | `10` | cumulative |
| kOpaque | `99` | opaque |

### MatmulBiasMode

本地 matmul 操作的偏置模式

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| NoBias | `0` | NoBias |
| PerChannelAdd | `1` | PerChannelAdd |
| PerChannelAddWithSplitK | `2` | PerChannelAddWithSplitK |
| ElementwiseCrossLoopAdd | `4` | ElementwiseCrossLoopAdd |
| ElementwiseAdd | `3` | ElementwiseAdd |

### MemPlanMode

内存规划模式

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| LOCAL_MEM_PLAN | `0` | LOCAL_MEM_PLAN |
| GLOBAL_WORKSPACE_PLAN | `1` | GLOBAL_WORKSPACE_PLAN |

### PadMode

LoadOp 的填充模式

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| PadNull | `0` | PadNull |
| PadFirstElem | `1` | PadFirstElem |
| PadValue | `2` | PadValue |

### PIPE

HIVM 操作管道

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| PIPE_S | `0` | PIPE_S |
| PIPE_V | `1` | PIPE_V |
| PIPE_M | `2` | PIPE_M |
| PIPE_MTE1 | `3` | PIPE_MTE1 |
| PIPE_MTE2 | `4` | PIPE_MTE2 |
| PIPE_MTE3 | `5` | PIPE_MTE3 |
| PIPE_ALL | `6` | PIPE_ALL |
| PIPE_MTE4 | `7` | PIPE_MTE4 |
| PIPE_MTE5 | `8` | PIPE_MTE5 |
| PIPE_V2 | `9` | PIPE_V2 |
| PIPE_FIX | `10` | PIPE_FIX |
| VIRTUAL_PIPE_MTE2_L1A | `11` | VIRTUAL_PIPE_MTE2_L1A |
| VIRTUAL_PIPE_MTE2_L1B | `12` | VIRTUAL_PIPE_MTE2_L1B |
| PIPE_NUM | `13` | PIPE_NUM |
| PIPE_UNASSIGNED | `99` | PIPE_UNASSIGNED |

### ReduceOperation

VReduceOp 的归约类型

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| sum | `1` | sum |
| prod | `2` | prod |
| max | `3` | max |
| min | `4` | min |
| max_with_index_left | `5` | max_with_index_left |
| max_with_index_right | `6` | max_with_index_right |
| min_with_index_left | `7` | min_with_index_left |
| min_with_index_right | `8` | min_with_index_right |
| any | `9` | any |
| all | `10` | all |
| xori | `11` | xori |
| ori | `12` | ori |
| andi | `13` | andi |
| none | `0` | none |

### RoundMode

VCastOp 的舍入模式

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| RINT | `0` | rint |
| ROUND | `1` | round |
| FLOOR | `2` | floor |
| CEIL | `3` | ceil |
| TRUNC | `4` | trunc |
| ODD | `5` | odd |
| TRUNCWITHOVERFLOW | `6` | truncwithoverflow |

### SyncBlockInstrMode

HIVM SyncBlockInstrMode

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| INTER_BLOCK_SYNCHRONIZATION | `0` | INTER_BLOCK_SYNCHRONIZATION |
| INTER_SUBBLOCK_SYNCHRONIZATION | `1` | INTER_SUBBLOCK_SYNCHRONIZATION |
| INTRA_BLOCK_SYNCHRONIZATION | `2` | INTRA_BLOCK_SYNCHRONIZATION |

### SyncBlockMode

HIVM SyncBlockMode

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| ALL_CUBE | `0` | ALL_CUBE |
| ALL_VECTOR | `1` | ALL_VECTOR |
| ALL_SUB_VECTOR | `2` | ALL_SUB_VECTOR |
| BARRIER_CUBE | `3` | BARRIER_CUBE |
| BARRIER_VECTOR | `4` | BARRIER_VECTOR |
| ALL | `5` | ALL |

### TCoreType

HIVM 操作核心类型

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| CUBE | `1` | CUBE |
| VECTOR | `2` | VECTOR |
| CUBE_OR_VECTOR | `3` | CUBE_OR_VECTOR |
| CUBE_AND_VECTOR | `4` | CUBE_AND_VECTOR |

### TFuncCoreType

HIVM 函数核心类型

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| AIC | `1` | AIC |
| AIV | `2` | AIV |
| MIX | `3` | MIX |
| AIC_OR_AIV | `4` | AIC_OR_AIV |

### TModuleCoreType

HIVM 模块核心类型

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| AIC | `1` | AIC |
| AIV | `2` | AIV |
| MIX | `3` | MIX |

### TypeFn

VCastOp 的转换

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| cast_signed | `0` | cast_signed |
| cast_unsigned | `1` | cast_unsigned |
| bitcast | `2` | bitcast |

### UNIT_FLAG

用于同步的单元标志模式

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| DISABLED | `0` | DISABLED |
| RESERVED | `1` | RESERVED |
| ENABLED_WITHOUT_UPDATE | `2` | ENABLED_WITHOUT_UPDATE |
| ENABLED_WITH_UPDATE | `3` | ENABLED_WITH_UPDATE |
| ENABLED_ONLY_LAST_ITER | `4` | ENABLED_ONLY_LAST_ITER |
| ENABLED_ONLY_FIRST_ITER | `5` | ENABLED_ONLY_FIRST_ITER |
| ENABLED_ONLY_FIRST_AND_LAST_ITERS | `6` | ENABLED_ONLY_FIRST_AND_LAST_ITERS |

### VFMode

HIVM VF 模式

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| SIMD | `0` | SIMD |
| SIMT | `1` | SIMT |
| MIX | `2` | MIX |

### MappingId

用于循环映射的映射 ID

#### 枚举值:

| 符号 | 值 | 字符串 |
| :----: | :---: | ------ |
| DimX | `0` | x |

---


# 'annotation' 方言

用于标记操作的注解方言，用于为某个操作定义一些额外的属性：

```mlir
// Mark extra attrs
annotation.mark %a { attr-dict } : f64
```


## Operations

### `annotation.mark` (annotation::MarkOp)

_使用键值属性对来注解一个 IR 值_

语法：

```
operation ::= `annotation.mark` $src attr-dict
              (`keys` `=` $keys^)?
              (`values` `=` `[`$values^`:`type($values) `]`)?
              `:`type($src)
```

`annotation.mark` 操作可用于使用一组键值属性对来注解一个 IR 值。

该值可以是静态的（即属性）或动态的（即 IR 值）。

示例：

```mlir
annotation.mark %target keys = ["key"] values = [%val]
annotation.mark %target {key : val}
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `MemoryEffectOpInterface (MemoryEffectOpInterface)`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{MemoryEffects::Write on ::mlir::SideEffects::DefaultResource}`, `MemoryEffects::Effect{}`

#### 属性:

<table>
<tr><th>属性</th><th>MLIR 类型</th><th>描述</th></tr>
<tr><td><code>keys</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
</table>

#### 操作数:

| 操作数 | 描述 |
| :-----: | ----------- |
| `src` | 任意类型
| `values` | 可变数量的任意类型

---