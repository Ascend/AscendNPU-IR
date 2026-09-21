# BiShengIR SCF Loop TransformOps

本文档列出 BiShengIR SCF Loop Transform Ops。

## 操作定义

### transform.loop.for_to_forall (transform::ForToForallOp)

**功能**：将给定的单个`scf.for`操作转换为`scf.forall`操作。

**语法**：

```mlir
operation ::= `transform.loop.for_to_forall` $for_op attr-dict `:` functional-type(operands, results)
```

表示给定的`for_op`（单个`scf.for`操作）应被转换为`scf.forall`操作。

如果`mapping`非空，它将作为属性添加到生成的`scf.forall`上。

注意，给定循环要么不产生结果，要么只产生由`tensor.insert_slice`操作生成的值。在后一种情况下，insert slice将被映射为`tensor.parallel_insert_slice`操作。

如果`annotate_only`为true，该transform op只会向目标循环句柄添加`map_for_to_forall`属性。如果存在`mapping`，也会一并作为属性添加。

**返回模式**：

成功时，结果句柄与`scf.forall`操作关联。

以下情况该操作产生确定失败（definite failure）：

- payload操作数量不为一。
- payload操作不是`scf.for`操作。
- 存在mapping属性时，其数量不为一。
- `scf.for`操作要么没有yield结果，要么只yield由`tensor.insert_slice`操作生成的值。

**特性**：`MemoryEffectOpInterface`, `TransformOpInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `mapping` | `::mlir::ArrayAttr` | 设备映射数组属性 |
| `annotate_only` | `::mlir::BoolAttr` | 布尔属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `for_op` | `TransformHandleTypeInterface`实例 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `forallOp` | `TransformHandleTypeInterface`实例 |

### transform.loop.fuse_nested_siblings (transform::LoopFuseNestedSiblingsOp)

**功能**：融合直接嵌套在目标循环内的兄弟`scf.for`循环。

**语法**：

```mlir
operation ::= `transform.loop.fuse_nested_siblings` $target attr-dict `:` functional-type(operands, results)
```

收集直接嵌套在目标`scf.for`循环体内的`scf.for`循环，将迭代次数相同且相互独立的兄弟循环分组，并将每组融合为单个循环。

如果设置了`recursive`，则对每个新融合得到的循环内部嵌套的循环再次应用融合，依此类推，直到无法继续融合内部兄弟循环为止。

**特性**：`MemoryEffectOpInterface`, `TransformOpInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `recursive` | `::mlir::BoolAttr` | 布尔属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `target` | `TransformHandleTypeInterface`实例 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `transformed` | `TransformHandleTypeInterface`实例 |

### transform.loop.normalize (transform::LoopNormalizeOp)

**功能**：规范化给定的`scf.for`循环。

**语法**：

```mlir
operation ::= `transform.loop.normalize` $target attr-dict `:` functional-type(operands, results)
```

在保持迭代次数不变的前提下，使目标`scf.for`操作的步长为1。目前仅支持规范化具有静态非零下界的循环。

**特性**：`MemoryEffectOpInterface`, `TransformOpInterface`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `target` | `TransformHandleTypeInterface`实例 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `transformed` | `TransformHandleTypeInterface`实例 |

### transform.loop.tile (transform::LoopTileOp)

**功能**：按给定大小对目标循环进行分块（tile）。

表示给定的`target`循环应按给定大小进行分块。该变换会生成一个循环嵌套。

目前分块仅限于`scf.ForOp`，且分块大小的数量只能为一（即仅支持将一个循环分为两个）。

分块大小可以在变换时已知，此时应通过`static_size`属性提供；也可以在变换时未知，此时分块值必须由payload IR计算，并通过`dynamic_sizes`提供计算该值的操作句柄。当大小不是静态已知时，`static_sizes`属性中对应的条目必须设置为`ShapedType::kDynamic`。`dynamic_sizes`中只需提供动态大小，即句柄数量应与`static_sizes`属性中`ShapedType::kDynamic`值的数量一致。静态大小为`0`表示该维度不应被分块，不会为这类维度生成循环。如果所有分块大小都为`0`，该变换实际上就是空操作。

该op返回生成循环的句柄。循环数量为静态已知非零的分块大小数量。

**返回模式**：

成功时，结果句柄与分块后的循环列表关联。

该操作仅支持对`scf.ForOp`分块，否则产生确定失败。输入分块大小的数量只能为一，否则产生确定失败。

如果`dynamic_sizes`句柄关联的payload操作列表大小与`target`句柄关联的列表大小不同，该操作产生可抑制失败（silenceable failure）。

如果任一操作的分块内部实现失败，则产生确定失败。

**特性**：`MemoryEffectOpInterface`, `TransformOpInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `static_sizes` | `::mlir::DenseI64ArrayAttr` | i64稠密数组属性 |
| `is_npart_mode` | `::mlir::BoolAttr` | 布尔属性 |
| `is_reorder_mode` | `::mlir::BoolAttr` | 布尔属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `target` | `TransformHandleTypeInterface`实例 |
| `dynamic_sizes` | 任意transform句柄或参数的可变参数列表 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `loops` | `TransformHandleTypeInterface`实例的可变参数列表 |

### transform.parallel_loop.tile (transform::ParallelLoopTileOp)

**功能**：按给定大小对并行循环进行分块（tile）。

**特性**：`MemoryEffectOpInterface`, `TransformOpInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `static_sizes` | `::mlir::ArrayAttr` | 64位整数数组属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `target` | `TransformHandleTypeInterface`实例 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `loops` | `TransformHandleTypeInterface`实例的可变参数列表 |
