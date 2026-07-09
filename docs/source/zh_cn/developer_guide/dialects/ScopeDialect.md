# scope 方言

Scope方言用于定义代码作用域区域。

## 操作定义

### scope.return (scope::ReturnOp)

**功能：** `scope.scope`区域内部的终结返回操作。

**语法：**

```mlir
operation ::= `scope.return` attr-dict ($results^ `:` type($results))?
```

**示例：**

```mlir
scope.return
```

**特性：** `AlwaysSpeculatableImplTrait`、`HasParent<ScopeOp>`、`ReturnLike`、`Terminator`

**接口：** `ConditionallySpeculatable`、`NoMemoryEffect`、`RegionBranchTerminatorOpInterface`

**内存效应：** `MemoryEffects::Effect{}`

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `results` | 变长任意类型返回值 |

### scope.scope (scope::ScopeOp)

**功能：** 定义一段独立操作区域的作用域容器。

**示例：**

```mlir
scope.scope : () -> () {
    scope.return
} {tcore_type = #hivm.tcore_type<CUBE>, ...}

scope.scope : () -> () {
    scope.return
}
```

**特性：** `NoRegionArguments`、`RecursiveMemoryEffects`、`SingleBlockImplicitTerminator<scope::ReturnOp>`、`SingleBlock`
**接口：** `RegionBranchOpInterface`

**属性：**

| 属性名 | MLIR类型 | 说明 |
| :-----: | ----------- | ---- |
| `no_inline` | `::mlir::UnitAttr` | 单元属性，标记禁止内联 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `results` | 变长任意类型输出值 |
