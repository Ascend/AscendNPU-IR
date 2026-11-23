# HFusion TransformOps

本文档列出 HFusion Transform Ops。

## 操作定义

### transform.structured.cache_read (transform::CacheReadOp)

**功能**：为读取方创建原始张量的缓存读取。

**语法**：

```mlir
operation ::= `transform.structured.cache_read` $targets attr-dict `:` functional-type(operands, results)
```

为由`targets`操作句柄给定的值创建缓存读取。例如，给定输入IR：

```mlir
"some_op"(%a) : tensor<16x16xf16> -> ()
```

如果`targets`句柄指向`%a`，变换后的IR为：

```mlir
%empty = tensor.empty() : tensor<16x16xf16>
%cached = linalg.copy ins(%a : tensor<16x16xf16>) outs(%empty : tensor<16x16xf16>)
"some_op"(%cached) : tensor<16x16xf16> -> ()
```

`targets`操作句柄可关联一个或多个payload IR值，缓存读取将逐一执行。

**返回模式**：

返回句柄指向缓存值的定义操作。该操作仅读取`targets`句柄。

**特性**：`MemoryEffectOpInterface`, `TransformOpInterface`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `targets` | `TransformValueHandleTypeInterface`实例 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `cached` | `TransformHandleTypeInterface`实例 |

### transform.structured.cache_write (transform::CacheWriteOp)

**功能**：在存入张量之前，为原始张量创建缓存写入。

**语法**：

```mlir
operation ::= `transform.structured.cache_write` $targets attr-dict `:` functional-type(operands, results)
```

为由`targets`操作句柄给定的值创建缓存写入。`targets`的payload值应为linalg或hfusion操作的SSA结果。

例如，给定输入IR：

```mlir
%a = "some_op"(%init) : (tensor<16x16xf16>) -> tensor<16x16xf16>
```

如果`targets`句柄指向`%a`，变换后的IR为：

```mlir
%a = "some_op"(%init) : (tensor<16x16xf16>) -> tensor<16x16xf16>
%empty = tensor.empty() : tensor<16x16xf16>
%cached = linalg.copy ins(%a : tensor<16x16xf16>) outs(%empty : tensor<16x16xf16>)
```

如果可选属性`$cache_write_to_output_init`设置为true，将使用目标绑定的init操作数来存储缓存结果。*注意，这仅在该操作不读取init操作数时有效。*

```mlir
%empty = tensor.empty() : tensor<16x16xf16>
%a = "some_op"(%empty) : (tensor<16x16xf16>) -> tensor<16x16xf16>
%cached = linalg.copy ins(%a : tensor<16x16xf16>) outs(%init : tensor<16x16xf16>)
```

如果可选属性`output_only`设置为true，缓存结果将只替换`func.return`操作中的原始操作数。

沿用上述示例，变换后的IR为：

```mlir
%a = "some_op"(%init) : (tensor<16x16xf16>) -> tensor<16x16xf16>
%empty = tensor.empty() : tensor<16x16xf16>
%cached = linalg.copy ins(%a : tensor<16x16xf16>) outs(%empty : tensor<16x16xf16>)
"some_use"(%a) : tensor<16x16xf16> -> ()
// code omitted
func.return %cached : tensor<16x16xf16>
```

`targets`操作句柄可关联一个或多个payload IR值，缓存写入将逐一执行。

**返回模式**：

返回句柄指向缓存值的定义操作。该操作仅读取`targets`句柄。

**特性**：`MemoryEffectOpInterface`, `TransformOpInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `output_only` | `::mlir::BoolAttr` | 布尔属性 |
| `cache_write_to_output_init` | `::mlir::BoolAttr` | 布尔属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `targets` | `TransformValueHandleTypeInterface`实例 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `cached` | `TransformHandleTypeInterface`实例 |

### transform.structured.multi_buffer (transform::MultiBufferOp)

**功能**：标记操作数以在HFusion中执行多缓冲优化。

**语法**：

```mlir
operation ::= `transform.structured.multi_buffer` $target (`factor` `=` $factor^)? attr-dict `:` type($target)
```

表示给定的`target`应被标记为在HFusion中执行多缓冲优化。使用`annotation::MarkOp`进行标记，并设置值为`factor`的"hfusion.multi_buffer"属性。factor表示启用的缓冲区数量，可选，默认为2。

目标句柄的payload操作必须已定义且具有结果，否则该目标句柄不会被标记。一个句柄可指向多个payload操作。

具有一个结果的payload操作示例：

```mlir
%0 = linalg.exp ins(%arg0: tensor<4x8x16xf32>) outs(%arg1: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
```

如果上述linalg.exp被匹配以执行多缓冲：

```mlir
%0 = transform.structured.match ops{["linalg.exp"]} in %arg0 : (!transform.any_op) -> !transform.any_op
transform.structured.multi_buffer %0 : !transform.any_op, !transform.any_op
```

变换后：

```mlir
%0 = linalg.exp ins(%arg0 : tensor<4x8x16xf32>) outs(%arg1 : tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
annotation.mark %0 {hfusion.multi_buffer = 2 : i32} : tensor<4x8x16xf32>
```

具有两个结果的payload操作示例：

```mlir
%0, %1 = "test.source"() : () ->  (memref<1xf32>, memref<2xf32>)
```

如果上述test.source被匹配以执行多缓冲，则变换后：

```mlir
%0:2 = "test.source"() : () -> (memref<1xf32>, memref<2xf32>)
annotation.mark %0#0 {hfusion.multi_buffer = 2 : i32} : memref<1xf32>
annotation.mark %0#1 {hfusion.multi_buffer = 2 : i32} : memref<2xf32>
```

使用factor的示例：

```mlir
%0 = transform.structured.match ops{["linalg.exp"]} in %arg0 : (!transform.any_op) -> !transform.any_op
transform.structured.multi_buffer %0 factor = 3 : !transform.any_op
```

**返回模式**：

该操作仅读取目标句柄。

**特性**：`MemoryEffectOpInterface`, `TransformOpInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `factor` | `::mlir::IntegerAttr` | 32位无符号整数属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `target` | `TransformHandleTypeInterface`实例的可变参数列表 |

### transform.structured.set_buffer_size (transform::SetBufferSizeOp)

**功能**：设置目标操作数的分配大小。

**语法**：

```mlir
operation ::= `transform.structured.set_buffer_size` $target
              (`static_buffer_sizes` `=` $static_buffer_sizes^)?
              `unit_mode` `=` $unit_mode
              (`reference_type` `=` $reference_type^)?
              attr-dict
              `:` type($target)
```

表示给定的`target`应被设置为期望的大小。如果目标是静态形状的，则不执行任何操作。

如果`target`是`memref.alloc`或`memref.alloca`操作，它将被替换为i8元素类型、空布局映射的一维连续memref。其大小由`static_buffer_sizes`和`unit_mode`决定。

否则，该操作的结果会通过`annotation.mark`操作标注大小信息。注意，只有形状类型的结果才能被标注。

有两种`unit_mode`：

- "per_byte"：memref大小等于所提供的缓冲区大小。
- "per_element"：memref大小等于所提供的缓冲区大小乘以目标值元素类型的大小。

提供`reference_type`时，缓冲区大小会乘以目标值元素类型大小再除以`reference_type`（必须是int或float类型）的大小。例如：

```text
  static buffer sizes = [1000]
  unit mode = per_byte
  reference type = i8
  target value's element type = f32
  final buffer size = 1000 * sizeof(f32) / sizeof(i8) = 4000 bytes
```

新的alloc会被view为原始alloc的形状。

输入的`target`与`static_buffer_sizes`应具有相同的数量。每个`target`操作句柄可关联多个payload操作，它们都会被设置为相同的缓冲区大小。

注意，如果`target`已具有静态形状，该transform op为空操作。

**返回模式**：

该操作消耗目标句柄。

**特性**：`MemoryEffectOpInterface`, `TransformOpInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `static_buffer_sizes` | `::mlir::DenseI64ArrayAttr` | i64稠密数组属性 |
| `unit_mode` | `mlir::transform::SetBufferSizeModeAttr` | `SetBufferSizeOp`的缓冲区大小设置模式，枚举值：`per_byte`（`kPerByte`）、`per_element`（`kPerElement`） |
| `reference_type` | `::mlir::TypeAttr` | 任意类型属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `target` | `TransformHandleTypeInterface`实例的可变参数列表 |
