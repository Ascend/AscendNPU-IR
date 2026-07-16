# scope方言Passes

## -inline-scope

**功能：** 内联ScopeOp中的作用域区域。若`scope.scope`未携带`no_inline`属性，则将作用域内的算子移出至其父区域。

**转换示例：**

转换前：

```mlir
func.func @test() {
  scope.scope : () -> () {
    ...
    scope.return
  } {no_inline}
  scope.scope : () -> () {
    <inlinable_operations>
    scope.return
  }
  return
}
```

转换后：

```mlir
func.func @test() {
  scope.scope : () -> () {
    ...
    scope.return
  } {no_inline}
  <inlinable_operations>
  return
}
```

**选项：**

- `-force-inline`：忽略`no_inline`属性，强制内联所有作用域。

## -outline-scope

**功能：** 外提ScopeOp中的作用域区域，将`scope.scope`转换为独立的`func.func`函数。

**转换示例：**

转换前：

```mlir
module {
  func.func @test() {
    scope.scope : () -> () {
      ...
      scope.return
    } {tcore_type = #hivm.tcore_type<CUBE>, ...}
    scope.scope : () -> () {
      ...
      scope.return
    } {tcore_type = #hivm.tcore_type<VECTOR>, ...}
    return
  }
}
```

转换后：

```mlir
module {
  func.func @test_scope_0() attributes {tcore_type = #hivm.tcore_type<CUBE>, ...} {
    ...
    return
  }
  func.func @test_scope_1() attributes {tcore_type = #hivm.tcore_type<VECTOR>, ...} {
    ...
    return
  }
  func.func @test() {
    call @test_scope_scope_scope_0() : () -> ()
    call @test_scope_scope_scope_1() : () -> ()
    return
  }
}
```
