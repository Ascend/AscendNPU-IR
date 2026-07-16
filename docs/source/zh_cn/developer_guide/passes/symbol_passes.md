# symbol方言Passes

## -encoding-to-symbol

**功能：** 将tensor编码转换为bind_symbolic_shape符号绑定。

## -erase-symbol

**功能：** 擦除符号标记。该Pass从函数中移除符号信息，用于无需符号分析时清理符号化Shape标注。

## -propagate-symbol

**功能：** 沿算子传播符号信息。该Pass通过替换`tensor.dim`的使用，传播绑定在函数参数上的符号，使tensor动态维度之间的关联更加显式。

**执行逻辑：**

1. 为未绑定的函数参数绑定`symbolic_int`，确保每个动态维度都由唯一符号表示。
2. 通过`reifyResultShapes`为每个算子的tensor结果绑定`bind_symbolic_shape`，动态输出维度由`tensor.dim`或携带`tensor.dim`操作数的`affine.apply`表示。
3. 将`tensor.dim`替换为`symbolic_int`完成符号传播，确保所有动态维度完成符号化。

**转换示例：**

转换前：

```mlir
%dim0 = tensor.dim %arg0, %c0 : tensor<?x640xf16>
%out0 = tensor.empty(%dim0) : tensor<?x640xf16>
%add0 = linalg.elemwise_binary ins(%arg0, %arg0) outs(%out0)
%concat = tensor.concat dim(0) %add0, %add0 : tensor<?x640xf16>
%dim1 = tensor.dim %concat, %c0 : tensor<?x640xf16>
%out1 = tensor.empty(%dim1) : tensor<?x640xf16>
%add1 = linalg.elemwise_binary ins(%concat, %concat) outs(%out1)
```

转换后：

```mlir
%S0 = symbol.symbolic_int @S0 : index
%S1 = symbol.symbolic_int @S1 [%S0, %S0], affine_map<()[s0, s1] -> (s0 + s1)>
symbol.bind_symbolic_shape %arg0, [%S0]

%out0 = tensor.empty(%S0) : tensor<?x640xf16>
symbol.bind_symbolic_shape %0, [%S0]
%add0 = linalg.elemwise_binary ins(%arg0, %arg0) outs(%out0)
symbol.bind_symbolic_shape %add0, [%S0]

// concat的结果维度将被具体化为`affine.apply`操作。
// 此Pass在`%S1`中保留仿射映射
%concat = tensor.concat dim(0) %add0, %add0 : tensor<?x640xf16>
symbol.bind_symbolic_shape %concat, [%S1]

%out1 = tensor.empty(%S1) : tensor<?x640xf16>
symbol.bind_symbolic_shape %out1, [%S1]
%add1 = linalg.elemwise_binary ins(%concat, %concat) outs(%out1)
symbol.bind_symbolic_shape %add1, [%S1]
```

## -symbol-to-encoding

**功能：** 将bind_symbolic_shape符号绑定转换回tensor编码。

## -unfold-symbolic-int

**功能：** 展开符号整数，用具体值替换`symbol.symbolic_int`操作。
该Pass依据bind_symbolic_shape替换所有符号整数，所有symbolic_int都应可通过其首次出现的bind_symbolic_shape完成替换。本Pass不执行符号具体化操作，如需执行具体化请参考propagate-symbol Pass。

**约束条件：**

1. symbolic_int必须可通过其绑定的首个bind_symbolic_shape完成替换。
2. symbolic_int绑定的首个bind_symbolic_shape必须具备单位仿射映射。
   示例：`symbol.bind_symbolic_shape %arg0, [%S0, %S1], affine_map<()[s0, s1] -> (s0, 640, s1)> : tensor<?x640x?xf16>`
3. symbolic_int绑定的首个bind_symbolic_shape不能绑定在`tensor.empty`上。

**无效示例：**

```mlir
symbol.bind_symbolic_shape %arg0, [%S0, %S1], affine_map<()[s0, s1] -> (s0/2, s0/s1, s1+1)> : tensor<?x640x?xf16>
%empty = tensor.empty(%S2, %S3) : tensor<?x?xf16>
symbol.bind_symbolic_shape %empty, [%S2, %S3], affine_map<()[s0, s1] -> (s0, s1)> : tensor<?x?xf16>
```

**转换示例：**

转换前：

```mlir
func.func @test_already_bind_symbol_0(%arg0: tensor<?x640x?xf16>) -> tensor<?x640x?xf16> {
  %S0 = symbol.symbolic_int @S0 {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : index
  %S1 = symbol.symbolic_int @S1 {max_val = 9223372036854775807 : i64, min_val = 0 : i64} : index
  symbol.bind_symbolic_shape %arg0, [%S0, %S1], affine_map<()[s0, s1] -> (s0, 640, s1)> : tensor<?x640x?xf16>
}
```

转换后：

```mlir
func.func @test_already_bind_symbol_0(%arg0: tensor<?x640x?xf16>) -> tensor<?x640x?xf16> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x640x?xf16>
  %c2 = arith.constant 2 : index
  %dim_0 = tensor.dim %arg0, %c2 : tensor<?x640x?xf16>
}
```
