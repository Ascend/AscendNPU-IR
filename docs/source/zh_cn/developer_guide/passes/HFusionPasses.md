# hfusion 方言 Passes

## -adapt-triton-kernel

**功能：** 适配Triton Kernel编译流程。

**选项：**

- `-hivmc-version`：指定hivmc版本以解决向后兼容性。

## -hfusion-add-ffts-addr

**功能：** 向函数参数与注解中添加FFTS基地址。

**选项：**

- `-force-add-ffts-addr`：强制将FFTS基地址插入到指定参数位置；默认值-1表示不插入，0表示插入至首个参数位置。

## -hfusion-auto-schedule

**功能：** 对融合后的Kernel执行自动调度。

**选项：**

- `-block-dim`：设置使用的Block数量。
- `-enable-auto-multi-buffer`：启用自动多缓冲优化。
- `-enable-deterministic-computing`：启用确定性计算。
- `-max-buffer-count-tuning`：开启maxBufferCnt参数调优。
- `-enable-count-buffer-dma-opt`：开启后，DMA操作占用的缓冲区不会被Vector操作复用。
- `-enable-manage-host-resources`：启用Host端函数的资源管理。
- `-cube-tiling-tuning`：开启Cube Tiling参数调优。
- `-external-tiling-func-path`：自动引入外部Tiling函数。
- `-enable-symbol-analysis`：启用Tiling与融合阶段的符号分析。

## -hfusion-cache-io

**功能：** 缓存输入与输出参数。

## -hfusion-cache-io-for-return-arg

**功能：** 对直接返回的参数执行缓存处理。

## -hfusion-compose-multi-reduce

**功能：** 组合多个归约操作，执行合并优化。

**选项：**

- `-max-compose`：单个操作可合并的最大归约数，-1表示无限制。
- `-max-dist-diff`：与共同祖先节点的最大距离差值。
- `-aggressive`：激进模式，在Shape松散匹配时自动尝试插入Reshape操作。

## -hfusion-constantize-tiling-data

**功能：** 在Tiling函数与设备函数之间传播常量Tiling数据。

**核心修改：**

- 将常量Tiling数据内联到设备函数中。
- 从Tiling函数中移除常量Tiling数据。
- 从设备函数的入参中移除常量Tiling数据，并同步修改对应调用点。
- 从设备函数的调用方逐层移除常量Tiling数据，向上递归至所有上层调用者。

**约束说明：**

- 共享同一Tiling函数的所有设备函数，其Tiling数据参数的顺序必须完全一致。
- 设备函数入参中的Tiling参数，与Tiling函数的返回值顺序必须完全一致。

**转换示例：**

转换前：

```mlir
func.func @tiling_func(%arg0: tensor<?x?xf16>) -> (i64, i64)
attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %ret0 = "some_calculation"() : () -> i64
  %ret1 = arith.constant 42: i64
  return %ret0, %ret1: i64, i64
}

func.func @device_kernel_tiling_0(%arg0: tensor<?x?xf16>,
                                  %arg1: i64 {hacc.tiling_data},
                                  %arg2: i64 {hacc.tiling_data}) -> tensor<?x?xf16>
attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_func = "tiling_func"} {
  "some_use"(%arg1) : (i64) -> ()
  "some_use"(%arg2) : (i64) -> ()
  %ret0 = "some_op"(%arg0) : (tensor<?x?xf16>) -> tensor<?x?xf16>
  return %ret0 : tensor<?x?xf16>
}

func.func @device_kernel_tiling_1(%arg0: tensor<?x?xf16>,
                                   %arg1: i64 {hacc.tiling_data},
                                   %arg2: i64 {hacc.tiling_data}) -> tensor<?x?xf16>
attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_func = "tiling_func"} {
  "some_use"(%arg1) : (i64) -> ()
  "some_use"(%arg2) : (i64) -> ()
  %ret0 = "some_op"(%arg0) : (tensor<?x?xf16>) -> tensor<?x?xf16>
  return %ret0 : tensor<?x?xf16>
}

func.func @main(%arg0: tensor<?x?xf16>,
                 %arg1: i64 {hacc.tiling_data},
                 %arg2: i64 {hacc.tiling_data}) -> tensor<?x?xf16>
attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %0 = arith.index_castui %arg1 : i64 to index
  %1 = scf.index_switch %0 -> tensor<?x?xf16>
  case 1 {
    %2 = func.call @device_kernel_tiling_1(%arg0, %arg1, %arg2) : (tensor<?x?xf16>, i64, i64) -> tensor<?x?xf16>
    scf.yield %2 : tensor<?x?xf16>
  }
  case 0 {
    %2 = func.call @device_kernel_tiling_0(%arg0, %arg1, %arg2): (tensor<?x?xf16>, i64, i64) -> tensor<?x?xf16>
    scf.yield %2 : tensor<?x?xf16>
  }
  default {
    %false = arith.constant false
    cf.assert %false, "Invalid tiling key"
    %2 = ub.poison : tensor<?x?xf16>
    scf.yield %2 : tensor<?x?xf16>
  }
  return %1 : tensor<?x?xf16>
}
```

转换后：

```mlir
func.func @tiling_func(%arg0: tensor<?x?xf16>) -> (i64)
attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %ret0 = "some_calculation"() : () -> i64
  return %ret0: i64
}

func.func @device_kernel_tiling_0(%arg0: tensor<?x?xf16>,
                                   %arg1: i64 {hacc.tiling_data}) -> tensor<?x?xf16>
attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_func = "tiling_func"} {
  "some_use"(%arg1) : (i64) -> ()
  %arg2 = arith.constant 32 : i64
  "some_use"(%arg2) : (i64) -> ()
  %ret0 = "some_op"(%arg0) : (tensor<?x?xf16>) -> tensor<?x?xf16>
  return %ret0 : tensor<?x?xf16>
}

func.func @device_kernel_tiling_1(%arg0: tensor<?x?xf16>,
                                   %arg1: i64 {hacc.tiling_data}) -> tensor<?x?xf16>
attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_func = "tiling_func"} {
  "some_use"(%arg1) : (i64) -> ()
  %arg2 = arith.constant 32 : i64
  "some_use"(%arg2) : (i64) -> ()
  %ret0 = "some_op"(%arg0) : (tensor<?x?xf16>) -> tensor<?x?xf16>
  return %ret0 : tensor<?x?xf16>
}

func.func @main(%arg0: tensor<?x?xf16>,
                 %arg1: i64 {hacc.tiling_data}) -> tensor<?x?xf16>
attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %0 = arith.index_castui %arg1 : i64 to index
  %1 = scf.index_switch %0 -> tensor<?x?xf16>
  case 1 {
    %2 = func.call @device_kernel_tiling_1(%arg0, %arg1) : (tensor<?x?xf16>, i64) -> tensor<?x?xf16>
    scf.yield %2 : tensor<?x?xf16>
  }
  case 0 {
    %2 = func.call @device_kernel_tiling_0(%arg0, %arg1): (tensor<?x?xf16>, i64) -> tensor<?x?xf16>
    scf.yield %2 : tensor<?x?xf16>
  }
  default {
    %false = arith.constant false
    cf.assert %false, "Invalid tiling key"
    %2 = ub.poison : tensor<?x?xf16>
    scf.yield %2 : tensor<?x?xf16>
  }
  return %1 : tensor<?x?xf16>
}
```

## -hfusion-convert-generic-to-named

**功能：** 将linalg通用算子转换为linalg命名算子与hfusion命名算子。

## -hfusion-decompose

**功能：** 分解所有实现了`AggregatedOpInterface`接口的算子。

**选项：**

- `-hfusion-decompose-phase`：指定执行的分解阶段。

## -hfusion-decompose-multi

**功能：** 将组合算子拆解为多个独立的单算子。

## -hfusion-downgrade-fp64

**功能：** 将fp64精度的常量降级为fp32精度。

## -hfusion-drop-symbols

**功能：** 从算子中移除Ranked Tensor符号标记。

## -hfusion-eliminate-duplicate-funcs

**功能：** 消除融合后产生的重复函数。

## -hfusion-flatten-ops

**功能：** 对linalg与hfusion算子执行展平处理。

**选项：**

- `-flatten-mode`：设置展平模式，tidy模式会对全函数做全局分析。
- `-skip-host`：是否跳过Host端函数的处理。
- `-multi-dynamic-shape`：是否合并多个动态Shape。

## -hfusion-fold-symbolic-dim

**功能：** 将`tensor.dim`的源操作数替换为`hfusion::SymbolicDimOp`。

## -hfusion-fuse-ops

**功能：** 基于HFusion框架融合Tensor上的算子。

**选项：**

- `-output-mode`：函数输出提取模式，默认为multi，可选single、single-aggr。
- `-fusion-mode`：按标签区分融合类型。
- `-always-inline`：对提取出的函数启用强制内联。
- `-move-out-to-param`：是否将输出Tensor转为入参形式。
- `-max-horizontal-fusion-size`：允许的最大水平（无依赖）融合数，-1表示无限制尝试水平融合。
- `-multi-kernel`：关闭时强制将计算图融合为单个Kernel；开启时可拆分为多个Kernel。
- `-enable-symbol-analysis`：启用符号方言（Symbol Dialect）分析。

## -hfusion-hoist-tensor-empty

**功能：** 将`tensor.empty`提升为函数入参，并合并为统一参数。该Pass会把函数内所有`tensor.empty`操作整合为单个函数入参。

## -hfusion-infer-func-fusion-kind

**功能：** 自动推断函数的融合类型。

## -hfusion-infer-out-shapes

**功能：** 为Kernel生成输出Tensor的Shape推导函数。

## -hfusion-inline-brc

**功能：** 内联广播类算子。

## -hfusion-legalize-bf16

**功能：** 将BF16类型统一规范化为FP32类型。

## -hfusion-legalize-bool

**功能：** 输入侧将int8转换为int1，输出侧将int1转换回int8。

## -hfusion-normalize-ops

**功能：** 对Hfusion算子执行归一化处理。

## -hfusion-normalize-slice-ops

**功能：** 对Slice算子执行归一化处理。

**选项：**

- `-skip-aligned-slice`：针对对齐的Slice，跳过FoldInsertSliceToConcat优化模式。

## -hfusion-outline-single-op

**功能：** 将单个linalg算子提取为独立Kernel。

**选项：**

- `-move-out-to-param`：是否将输出Tensor转为入参形式。

## -hfusion-pack-tiling-data

**功能：** 将动态Tiling信息打包封装为结构体。

**选项：**

- `-include-symbols`：指定转换生效的符号列表，以逗号分隔；为空时默认对所有函数生效。
- `-emit-get-tiling-struct-size-function`：开启后生成一个主机函数，返回Tiling数据总量（i64类型）。
- `-pack-tiling-key`：开启时将Tiling Key一同打包进结构体；关闭时Tiling Key直接写入指针。

## -hfusion-recache-io

**功能：** 对IO执行二次缓存。

## -hfusion-reorder-ops

**功能：** 按广度优先（BFS）顺序重排算子。

## -hfusion-simplify-ops

**功能：** 简化算子表达式。

## -hfusion-tensor-results-to-out-params

**功能：** 将Tensor返回值转为函数输出参数。

**选项：**

- `-include-symbols`：指定转换生效的符号列表，以逗号分隔；为空时默认对所有函数生效。
- `-enable-manage-host-resources`：启用Host端函数的资源管理。

## -hfusion-unfold-symbolic-dim

**功能：** 将`hfusion::SymbolicDimOp`替换为对应的符号参数。

## -hfusion-wrap-host-func

**功能：** 为指定的Host端相关函数创建包装器，覆盖主机端Tiling函数、Shape推导函数等场景。

**选项：**

- `-remove-unused-arguments`：是否移除主机包装函数中未使用的参数。
