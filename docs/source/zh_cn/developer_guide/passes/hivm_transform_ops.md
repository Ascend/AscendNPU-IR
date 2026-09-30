# HIVM TransformOps

本文档列出 HIVM Transform Ops。

## 操作定义

### transform.hivm.vectorize (transform::HIVMVectorizeOp)

**功能**：向量化实现了`VectorizableOpInterface`的HIVM操作。

**语法**：

```mlir
operation ::= `transform.hivm.vectorize` $target (`vector_sizes` `=` $static_vector_sizes^)? attr-dict
              `:` type($target)
```

向量化每个payload操作，这些操作必须实现`VectorizableOpInterface`。

如果省略`vector_sizes`，则使用与`hivm-vectorize-ops`相同的VL打包策略（256字节向量长度）从操作形状推断大小。显式指定静态大小可跳过该推断。

**返回模式**：

如果payload操作不可向量化、无法计算向量大小或`vectorize()`失败，该操作产生可抑制失败（silenceable failure）。

**特性**：`MemoryEffectOpInterface`, `TransformOpInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `static_vector_sizes` | `::mlir::DenseI64ArrayAttr` | i64稠密数组属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `target` | `TransformHandleTypeInterface`实例 |

### transform.hivm.map_forall_to_blocks (transform::MapForallToHIVMBlocks)

**功能**：将目标句柄给定的每个`scf.forall`映射到HIVM块操作。

**语法**：

```mlir
operation ::= `transform.hivm.map_forall_to_blocks` $target attr-dict `:` functional-type($target, $result)
```

该操作将目标句柄给定的每个`scf.forall`映射到HIVM块操作。映射是一对一的，`scf.forall`的归纳变量会被重写为hivm块索引操作。

**限制**：

- 目前不支持非规范化的`scf.forall`操作或具有动态迭代次数的操作。
- 目前仅支持**已buffer化**的`scf.forall`。
- 目前仅支持分布到**至多1个维度**的`scf.forall`。

**返回模式**：

如果`target`句柄指向任何不满足上述要求的`scf.forall`，该变换确定失败。返回句柄指向块索引操作。该操作消耗目标句柄。

**特性**：`MemoryEffectOpInterface`, `TransformOpInterface`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `target` | `TransformHandleTypeInterface`实例 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | `TransformHandleTypeInterface`实例 |
