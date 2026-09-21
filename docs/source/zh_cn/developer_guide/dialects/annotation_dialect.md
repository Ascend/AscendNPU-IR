# annotation方言

`annotation`方言用于提供注解操作，可为指定操作附加扩展属性。

```mlir
// 标记额外属性
annotation.mark %a { attr-dict } : f64
```

## 操作定义

### annotation.mark (annotation::MarkOp)

**功能**：使用键值对形式的属性对IR值添加注解。注解取值分为两种形式，静态取值通过内联属性字典定义，动态取值通过IR运行时值传入。

**语法**：

```mlir
operation ::= `annotation.mark` $src attr-dict
              (`keys` `=` $keys^)?
              (`values` `=` `[`$values^`:`type($values) `]`)?
              `:`type($src)
```

**示例**：

```mlir
annotation.mark %target keys = ["key"] values = [%val]
annotation.mark %target {key : val}
```

**特性**：`AlwaysSpeculatableImplTrait`

**接口**：`ConditionallySpeculatable`、`MemoryEffectOpInterface`、`NoMemoryEffect`

**内存效应**：`MemoryEffects::Effect{MemoryEffects::Write on ::mlir::SideEffects::DefaultResource}`、`MemoryEffects::Effect{}`

**属性**：

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `keys` | `::mlir::ArrayAttr` | 字符串数组属性 |

**操作数**：

| 操作数 | 说明 |
| :-----: | ----------- |
| `src` | 待注解IR值，支持任意类型 |
| `values` | 变长操作数，支持任意类型 |

## 属性

### EffectModeAttr

**语法**：

```mlir
#annotation.effect_mode<
  ::mlir::annotation::EffectMode   # value
>
```

**功能**：annotation内存效应模式属性。

**参数**：

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| value | `::mlir::annotation::EffectMode` | EffectMode枚举值 |

### FilterPassesAttr

**语法**：

```mlir
#annotation.filter_passes<
  StringAttr   # passes
>
```

**功能**：限制作用于被注解操作的Pass范围。附加到操作后，Pass管理器仅执行命令行参数名出现在逗号分隔`passes`列表中的Pass，其余Pass对该操作一律静默跳过。

**示例**：

```mlir
func.func @foo() attributes {
    annotation.filter_passes = #annotation.filter_passes<"canonicalize,cse">
} { ... }
```

**参数**：

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| passes | `StringAttr` | 逗号分隔的Pass参数名 |

## 枚举

### EffectMode

annotation内存效应模式枚举

| 枚举符号 | 数值 | 字符串标识 |
| :----: | :---: | ------ |
| Write | 0 | write |
| Read | 1 | read |
