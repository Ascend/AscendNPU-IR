# `annotation` 方言

`annotation` 方言用于提供注解操作，可为指定操作附加扩展属性。

```mlir
// 标记额外属性
annotation.mark %a { attr-dict } : f64
```

## 操作定义

### `annotation.mark` (annotation::MarkOp)

**功能：** 使用键值对形式的属性对IR值添加注解。注解取值分为两种形式，静态取值通过内联属性字典定义，动态取值通过IR运行时值传入。

**语法：**

```mlir
operation ::= `annotation.mark` $src attr-dict
              (`keys` `=` $keys^)?
              (`values` `=` `[`$values^`:`type($values) `]`)?
              `:`type($src)
```

**示例：**

```mlir
annotation.mark %target keys = ["key"] values = [%val]
annotation.mark %target {key : val}
```

**特性：** `AlwaysSpeculatableImplTrait`

**接口：** `ConditionallySpeculatable`、`MemoryEffectOpInterface`、`NoMemoryEffect`

**内存效应：** `MemoryEffects::Effect{MemoryEffects::Write on ::mlir::SideEffects::DefaultResource}`、`MemoryEffects::Effect{}`

**属性：**

| 属性名 | MLIR 类型 | 说明 |
| :-----: | ----------- | ---- |
| `keys` | `::mlir::ArrayAttr` | 字符串数组属性 |

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `src` | 待注解IR值，支持任意类型 |
| `values` | 变长操作数，支持任意类型 |
