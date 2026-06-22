# `symbol` 方言

`symbol` 方言用于定义与动态形状相关的操作，表达张量维度与符号量之间的映射与约束关系。

## 操作定义

### `symbol.bind_symbolic_shape` (symbol::BindSymbolicShapeOp)

**功能：** 通过符号索引的仿射映射，将形状表达式绑定到张量。该操作为张量绑定形状表达式以描述其动态维度的计算规则，接收一组 SSA 符号值，与仿射映射中声明的局部符号按顺序一一对应；仿射映射包含每个维度对应的仿射形状表达式，表达式的自变量均来自传入的符号。

**示例：**

```mlir
symbol.bind_symbolic_shape %arg0, [%0, %1], affine_map<()[s0, s1] -> (s0, s1, 3)> : tensor<?x?x3xf32>
symbol.bind_symbolic_shape %out0, [%0, %1, %2], affine_map<()[s0, s1, s2] -> (s0, s1 * 2 + s2, 3)> : tensor<?x?x3xf32>
```

**属性：**

| 属性名 | MLIR 类型 | 说明 |
| :-----: | ----------- | ---- |
| `shape_expressions` | `::mlir::AffineMapAttr` | 封装 AffineMap 对象的仿射映射属性<br><br>语法：<br>`affine-map-attribute ::= affine_map<affine-map>`<br><br>示例：<br>`affine_map<(d0) -> (d0)>`<br>`affine_map<(d0, d1, d2) -> (d0, d1)>` |

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `operand` | 待绑定形状的目标值，支持任意类型 |
| `shape_symbols` | 变长 index 类型符号列表，与仿射映射中的符号按顺序一一对应 |

### `symbol.symbolic_int` (symbol::SymbolicIntOp)

**功能：** 表示带范围约束的符号整数。该操作定义一个具名的符号整数值，最终以 `index` 类型返回，通过 `min_val` 与 `max_val` 属性指定符号值的闭区间取值边界，符号名称由符号属性声明，通常用于表示张量的动态维度，或其他带有已知取值约束的符号整数量。

**示例：**

```mlir
%0 = symbol.symbolic_int @s0 {min_val = 5, max_val = 10} : index
%1 = symbol.symbolic_int @s1 {min_val = 2, max_val = 20} : index
%2 = symbol.symbolic_int @s2 [%0, %1], affine_map<()[s1, s2] -> (s1 * s2)> {min_val = 2, max_val = 20} : index
```

**特性：** `AlwaysSpeculatableImplTrait`

**接口：** `ConditionallySpeculatable`、`NoMemoryEffect`、`OpAsmOpInterface`

**内存效应：** `MemoryEffects::Effect{}`

**属性：**

| 属性名 | MLIR 类型 | 说明 |
| :-----: | ----------- | ---- |
| `symbol_name` | `::mlir::FlatSymbolRefAttr` | 扁平符号引用属性 |
| `min_val` | `::mlir::IntegerAttr` | 64 位无符号整数属性，定义符号取值的下界（闭区间） |
| `max_val` | `::mlir::IntegerAttr` | 64 位无符号整数属性，定义符号取值的上界（闭区间） |
| `int_expressions` | `::mlir::AffineMapAttr` | 封装 AffineMap 对象的仿射映射属性<br><br>语法：<br>`affine-map-attribute ::= affine_map<affine-map>`<br><br>示例：<br>`affine_map<(d0) -> (d0)>`<br>`affine_map<(d0, d1, d2) -> (d0, d1)>` |

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `int_symbols` | 变长 index 类型符号列表 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `result` | index 类型的符号整数值 |
