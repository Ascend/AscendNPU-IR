# mathExt方言

扩展标准数学运算能力的方言，提供浮点数指数/分数部分分解等特殊数学操作，支持标量、张量和向量类型的逐元素计算。

## 操作定义

### mathExt.ilogb (::mlir::mathExt::IlogbOp)

**功能：** 计算浮点数绝对值的以2为底对数的向下取整（等效于提取IEEE 754浮点数的指数部分）。

**语法：**

```mlir
operation ::= `mathExt.ilogb` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

**示例：**

```mlir
%0 = mathExt.ilogb %x : f32
%1 = mathExt.ilogb %tensor : tensor<4xf64>
%2 = mathExt.ilogb %vec fastmath<contract> : vector<8xf16>
```

**特性：** `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

**操作数：**

| 操作数    | 说明         |
| :------: | ------------ |
| `operand` | 浮点类型（标量/Tensor/向量） |

**结果：**

| 结果     | 说明         |
| :------: | ------------ |
| `result` | 浮点类型（与操作数同类型） |

### mathExt.ldep (::mlir::mathExt::LdexpOp)

**功能：** 计算浮点数的分数部分与指数部分的归一化组合（等效于`x * (ilogb(x) + 1)^(-1)`）。

**语法：**

```mlir
operation ::= `mathExt.ldep` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

**示例：**

```mlir
%0 = mathExt.ldep %x, %y : f32
%1 = mathExt.ldep %tensor1, %tensor2 : tensor<4xf64>
%2 = mathExt.ldep %vec1, %vec2 fastmath<nnan,ninf> : vector<8xf16>
```

**特性：** `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

**操作数：**

| 操作数   | 说明         |
| :------: | ------------ |
| `lhs`    | 浮点类型（标量/Tensor/向量） |
| `rhs`    | 浮点类型（与`lhs`同类型） |

**结果：**

| 结果     | 说明         |
| :------: | ------------ |
| `result` | 浮点类型（与操作数同类型） |

## 枚举

### CmpFPredicate

**功能：** 定义浮点数比较操作的谓词类型，控制NaN处理和比较语义。

**取值范围：** 64位无符号整数（0~15）

| 枚举符号      | 数值   | 标识字符串 |
| :-----------: | :----: | ---------- |
| AlwaysFalse   | `0`    | false      |
| OEQ           | `1`    | oeq        |
| OGT           | `2`    | ogt        |
| OGE           | `3`    | oge        |
| OLT           | `4`    | olt        |
| OLE           | `5`    | ole        |
| ONE           | `6`    | one        |
| ORD           | `7`    | ord        |
| UEQ           | `8`    | ueq        |
| UGT           | `9`    | ugt        |
| UGE           | `10`   | uge        |
| ULT           | `11`   | ult        |
| ULE           | `12`   | ule        |
| UNE           | `13`   | une        |
| UNO           | `14`   | uno        |
| AlwaysTrue    | `15`   | true       |

### CmpIPredicate

**功能：** 定义整数比较操作的谓词类型，区分有符号/无符号比较逻辑。

**取值范围：** 64位无符号整数（0~9）

| 枚举符号 | 数值   | 标识字符串 |
| :------: | :----: | ---------- |
| eq       | `0`    | eq         |
| ne       | `1`    | ne         |
| slt      | `2`    | slt        |
| sle      | `3`    | sle        |
| sgt      | `4`    | sgt        |
| sge      | `5`    | sge        |
| ult      | `6`    | ult        |
| ule      | `7`    | ule        |
| ugt      | `8`    | ugt        |
| uge      | `9`    | uge        |

### IntegerOverflowFlags

**功能：** 标记整数溢出行为的语义属性（如nsw表示No Signed Wrap）。

| 枚举符号 | 数值   | 标识字符串 |
| :------: | :----: | ---------- |
| none     | `0`    | none       |
| nsw      | `1`    | nsw        |
| nuw      | `2`    | nuw        |

### RoundingMode

**功能：** 定义浮点运算的舍入模式，控制精度截断行为。

| 枚举符号           | 数值   | 标识字符串          |
| :----------------: | :----: | ------------------- |
| to_nearest_even    | `0`    | to_nearest_even     |
| downward           | `1`    | downward            |
| upward             | `2`    | upward              |
| toward_zero        | `3`    | toward_zero         |
| to_nearest_away    | `4`    | to_nearest_away     |

### AtomicRMWKind

**功能：** 定义原子读-修改-写操作的计算类型。

**取值范围：** 64位无符号整数（0~14）

| 枚举符号    | 数值   | 标识字符串  |
| :---------: | :----: | ----------- |
| addf        | `0`    | addf        |
| addi        | `1`    | addi        |
| assign      | `2`    | assign      |
| maximumf    | `3`    | maximumf    |
| maxs        | `4`    | maxs        |
| maxu        | `5`    | maxu        |
| minimumf    | `6`    | minimumf    |
| mins        | `7`    | mins        |
| minu        | `8`    | minu        |
| mulf        | `9`    | mulf        |
| muli        | `10`   | muli        |
| ori         | `11`   | ori         |
| andi        | `12`   | andi        |
| maxnumf     | `13`   | maxnumf     |
| minnumf     | `14`   | minnumf     |

### FastMathFlags

**功能：** 定义浮点运算的优化标志集合，控制NaN/无穷大等特殊值的处理策略。

| 枚举符号   | 数值    | 标识字符串 |
| :--------: | :-----: | ---------- |
| none       | `0`     | none       |
| reassoc    | `1`     | reassoc    |
| nnan       | `2`     | nnan       |
| ninf       | `4`     | ninf       |
| nsz        | `8`     | nsz        |
| arcp       | `16`    | arcp       |
| contract   | `32`    | contract   |
| afn        | `64`    | afn        |
| fast       | `127`   | fast       |
