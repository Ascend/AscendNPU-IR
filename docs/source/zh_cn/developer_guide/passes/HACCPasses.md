# hacc 方言 Passes

## `-hacc-append-device-spec`

**功能：** 为编译目标追加设备规格信息。

**选项：**

- `-target`：指定设备目标名称。

## `-hacc-rename-func`

**功能：** 基于属性重命名函数。该 Pass 根据 `hacc.rename_func` 属性重命名函数，并同步更新模块内所有对该函数的调用引用。

**转换示例：**

转换前：

```mlir
func.func @bar() attributes {hacc.rename_func = #hacc.rename_func<@foo>} {
  return
}

func.func @caller() {
  func.call @bar() : () -> ()
  return
}
```

转换后：

```mlir
func.func @foo() {
  return
}

func.func @caller() {
  func.call @foo() : () -> ()
  return
}
```

**使用限制：**

- 目标函数名不得与模块内已有函数重名。
