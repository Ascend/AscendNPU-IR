# hfusion方言Passes

## -adapt-triton-kernel

**功能**：适配Triton Kernel编译流程。

**选项**：

- `-hivmc-version`：指定hivmc版本以解决向后兼容性。

## -hfusion-add-ffts-addr

**功能**：向函数参数与注解中添加FFTS基地址。

**选项**：

- `-force-add-ffts-addr`：强制将FFTS基地址插入到指定参数位置；默认值-1表示不插入，0表示插入至首个参数位置。

## -hfusion-auto-schedule

**功能**：对融合后的Kernel执行自动调度。

**选项**：

- `-block-dim`：设置使用的Block数量。
- `-enable-auto-multi-buffer`：启用自动多缓冲优化。
- `-enable-deterministic-computing`：启用确定性计算。
- `-max-buffer-count-tuning`：开启maxBufferCnt参数调优。
- `-enable-count-buffer-dma-opt`：开启后，DMA操作占用的缓冲区不会被Vector操作复用。
- `-enable-manage-host-resources`：启用Host端函数的资源管理。
- `-cube-tiling-tuning`：开启Cube Tiling参数调优。
- `-external-tiling-func-path`：自动引入外部Tiling函数。
- `-enable-symbol-analysis`：启用Tiling与融合阶段的符号分析。

## -hfusion-auto-vectorize

**功能**：面向所有linalg命名算子的自动向量化。

**选项**：

- `-vector-length`：向量长度，单位为字节。
- `-peel-loops`：尝试剥离分块（Tile）循环。
- `-max-vectorize-axes`：可向量化的最内层轴的最大数量；默认值-1表示不限制向量化轴数。
- `-tree-reduce`：使用树形归约（Tree Reduce）。
- `-restrict-to-func-names`：非空时，仅处理名称在该列表中的函数，而非模块内所有可向量化函数；供AutoVectorizeV2的Legacy回退路径使用，将处理范围限定在向量化失败的函数上，避免重新处理整个模块。

## -hfusion-auto-vectorize-v2

**功能**：对所有linalg命名算子执行分块、融合与向量化。

**选项**：

- `-vector-length`：向量长度，单位为字节。
- `-max-fused-ops`：单个融合节点中允许的最大算子数量。
- `-enable-vf-stack-limit`：启用保守的向量函数栈空间限制。
- `-enable-multiple-consumer-fusion`：在AutoVectorizeV2中启用多消费者融合。
- `-emit-transform-sequence`：生成AutoVectorizeV2的Payload标签与Transform序列，但不实际应用。
- `-tree-reduce`：使用树形归约（Tree Reduce）。
- `-enable-cross-if-fusion`：在AutoVectorizeV2中启用跨带区域算子（Region-bearing Op）的融合。

## -hfusion-auto-vectorize-verifier

**功能**：在自动向量化之后校验向量操作。

**选项**：

- `-verify-free-vector-region`：校验向量操作均嵌套在提取出的向量区域内。
- `-verify-free-vector-func`：校验向量函数之外没有残留的向量操作。

## -hfusion-cache-io

**功能**：缓存输入与输出参数。

## -hfusion-cache-io-for-return-arg

**功能**：对直接返回的参数执行缓存处理。

## -hfusion-compose-multi-reduce

**功能**：组合多个归约操作，执行合并优化。

**选项**：

- `-max-compose`：单个操作可合并的最大归约数，-1表示无限制。
- `-max-dist-diff`：与共同祖先节点的最大距离差值。
- `-aggressive`：激进模式，在Shape松散匹配时自动尝试插入Reshape操作。

## -hfusion-constantize-tiling-data

**功能**：在Tiling函数与设备函数之间传播常量Tiling数据。

**核心修改**：

- 将常量Tiling数据内联到设备函数中。
- 从Tiling函数中移除常量Tiling数据。
- 从设备函数的入参中移除常量Tiling数据，并同步修改对应调用点。
- 从设备函数的调用方逐层移除常量Tiling数据，向上递归至所有上层调用者。

**约束说明**：

- 共享同一Tiling函数的所有设备函数，其Tiling数据参数的顺序必须完全一致。
- 设备函数入参中的Tiling参数，与Tiling函数的返回值顺序必须完全一致。

**转换示例**：

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

**功能**：将linalg通用算子转换为linalg命名算子与hfusion命名算子。

## -hfusion-decompose

**功能**：分解所有实现了`AggregatedOpInterface`接口的算子。

**选项**：

- `-hfusion-decompose-phase`：指定执行的分解阶段。

## -hfusion-decompose-multi

**功能**：将组合算子拆解为多个独立的单算子。

## -hfusion-downgrade-fp64

**功能**：将fp64精度的常量降级为fp32精度。

## -hfusion-drop-symbols

**功能**：从算子中移除Ranked Tensor符号标记。

## -hfusion-eliminate-duplicate-funcs

**功能**：消除融合后产生的重复函数。

## -hfusion-flatten-ops

**功能**：对linalg与hfusion算子执行展平处理。

**选项**：

- `-flatten-mode`：设置展平模式，tidy模式会对全函数做全局分析。
- `-skip-host`：是否跳过Host端函数的处理。
- `-multi-dynamic-shape`：是否合并多个动态Shape。

## -hfusion-fold-extract-insert-pair

**功能**：折叠Extract与Insert操作对。

## -hfusion-fold-symbolic-dim

**功能**：将`tensor.dim`的源操作数替换为`hfusion::SymbolicDimOp`。

## -hfusion-fold-unit-dims

**功能**：移除Tensor上Linalg算子单位维度（长度为1的维度）的扩展版本。

## -hfusion-fuse-ops

**功能**：基于HFusion框架融合Tensor上的算子。

**选项**：

- `-output-mode`：函数输出提取模式，默认为multi，可选single、single-aggr。
- `-fusion-mode`：按标签区分融合类型。
- `-always-inline`：对提取出的函数启用强制内联。
- `-move-out-to-param`：是否将输出Tensor转为入参形式。
- `-max-horizontal-fusion-size`：允许的最大水平（无依赖）融合数，-1表示无限制尝试水平融合。
- `-multi-kernel`：关闭时强制将计算图融合为单个Kernel；开启时可拆分为多个Kernel。
- `-enable-symbol-analysis`：启用符号方言（Symbol Dialect）分析。

## -hfusion-generalize

**功能**：将命名算子泛化为通用（Generic）算子。

## -hfusion-generic-unroller

**功能**：对不可向量化的linalg算子提供展开降阶（当前仅支持`linalg.reduce`）。

## -hfusion-hoist-tensor-empty

**功能**：将`tensor.empty`提升为函数入参，并合并为统一参数。该Pass会把函数内所有`tensor.empty`操作整合为单个函数入参。

## -hfusion-infer-func-fusion-kind

**功能**：自动推断函数的融合类型。

## -hfusion-infer-out-shapes

**功能**：为Kernel生成输出Tensor的Shape推导函数。

## -hfusion-inline-brc

**功能**：内联广播类算子。

## -hfusion-legalize-bf16

**功能**：将BF16类型统一规范化为FP32类型。

## -hfusion-legalize-bool

**功能**：输入侧将int8转换为int1，输出侧将int1转换回int8。

## -hfusion-legalize-fp8

**功能**：将FP8类型统一规范化为FP32类型。

## -hfusion-legalize-scalar

**功能**：将标量算子规范化为Tensor形式。

## -hfusion-merge-vf

**功能**：合并VF（Vector Function）函数。

**选项**：

- `-merge-level`：合并级别。0：不合并；1：仅合并无依赖关系的VF；2：合并所有VF。
- `-merge-vf-num-limit`：允许合并的VF最大数量。

## -hfusion-normalize-ops

**功能**：对Hfusion算子执行归一化处理。

## -hfusion-normalize-slice-ops

**功能**：对Slice算子执行归一化处理。

**选项**：

- `-skip-aligned-slice`：针对对齐的Slice，跳过FoldInsertSliceToConcat优化模式。

## -hfusion-optimize-scalar-transfers

**功能**：优化Tensor操作之间的标量传输。

## -hfusion-outline-single-op

**功能**：将单个linalg算子提取为独立Kernel。

**选项**：

- `-move-out-to-param`：是否将输出Tensor转为入参形式。

## -hfusion-pack-tiling-data

**功能**：将动态Tiling信息打包封装为结构体。

**选项**：

- `-include-symbols`：指定转换生效的符号列表，以逗号分隔；为空时默认对所有函数生效。
- `-emit-get-tiling-struct-size-function`：开启后生成一个主机函数，返回Tiling数据总量（i64类型）。
- `-pack-tiling-key`：开启时将Tiling Key一同打包进结构体；关闭时Tiling Key直接写入指针。

## -hfusion-pre-vectorization-fusion

**功能**：在向量化之前执行逐元素（Elemwise）算子融合与泛化。

**选项**：

- `-enable-triton-compile`：启用Triton编译。
- `-max-fused-elementwise-ops`：可合并进单个融合`linalg.generic`的逐元素算子最大数量（-1表示无限制）。
- `-enable-vf-stack-limit`：启用保守的向量函数栈空间限制。

## -hfusion-prepare-i1-nx1-for-vectorization

**功能**：在向量化之前，对带有扩展i1输入的Nx1 `linalg.generic`执行维度折叠（Collapse）。

## -hfusion-pull-slice-into-vector-function

**功能**：将Tensor切片（Slice）操作下沉至VF被调函数中，以减少缓冲区拷贝。

## -hfusion-recache-io

**功能**：对IO执行二次缓存。

## -hfusion-remove-cache-io

**功能**：移除IO缓存。

## -hfusion-remove-mask-from-unaligned-reduction-loop

**功能**：移除非对齐归约循环中的掩码（Mask）。

## -hfusion-reorder-ops

**功能**：按广度优先（BFS）顺序重排算子。

## -hfusion-simplify-ops

**功能**：简化算子表达式。

## -hfusion-simplify-vf-arg

**功能**：简化并合法化VF函数的参数。

## -hfusion-tensor-results-to-out-params

**功能**：将Tensor返回值转为函数输出参数。

**选项**：

- `-include-symbols`：指定转换生效的符号列表，以逗号分隔；为空时默认对所有函数生效。
- `-enable-manage-host-resources`：启用Host端函数的资源管理。

## -hfusion-unfold-symbolic-dim

**功能**：将`hfusion::SymbolicDimOp`替换为对应的符号参数。

## -hfusion-unroll-and-forward

**功能**：在VF内完全展开小迭代次数的`scf.for`循环，并前递`transfer_write`到`transfer_read`的链路，使中间结果保留在向量寄存器中（无需经过UB往返）。

该Pass分为两步：

1. 在`hivm.vector_function`内，完全展开具有常量下界、上界、步长、且迭代次数在[2, max-unroll]范围内的`scf.for`；
2. 将`vector.transfer_read`沿展开后的`extract_slice(insert_slice(...transfer_write...))`链路前递至源向量，消除UB实体化。

第2步比RemoveRedundantWriteAndReadPair更严格：它会沿`extract_slice(insert_slice)`链路（仅当其在展开后紧邻出现时）回溯，直至到达匹配常量偏移处的`transfer_write`定义。

**选项**：

- `-max-unroll`：允许完全展开的最大迭代次数（用于限制代码膨胀）。
- `-max-body-ops`：允许展开的循环体最大操作数；超过该数量的循环体即使匹配链路模式也不展开，以防止代码膨胀。

## -hfusion-uplift-while-to-for

**功能**：在条件允许时将`scf.while`循环提升为`scf.for`循环。

## -hfusion-vectorize-ops

**功能**：对hfusion算子执行向量化。

**选项**：

- `-for-manual-scope`：仅对手工Scope函数启用hfusion向量化。

## -hfusion-wrap-host-func

**功能**：为指定的Host端相关函数创建包装器，覆盖主机端Tiling函数、Shape推导函数等场景。

**选项**：

- `-remove-unused-arguments`：是否移除主机包装函数中未使用的参数。

## -loop-invariant-promotion

**功能**：提升循环中的`transfer_read`、`transfer_write`。

将循环携带Tensor中循环不变的`transfer_read`、`transfer_write`子集提升为循环携带向量，随后前递常量填充：若某个向量函数仅向`tensor.empty`写入Splat常量，则在其读取方处将该函数替换为常量本身，从而被填充的Tensor不再需要独占的UB缓冲区。

## -outline-vector-function

**功能**：外提向量函数（Vector Function）。

## -remove-redundant-write-and-read-pair

**功能**：移除冗余的写读操作对。

## -tree-reduce-v2

**功能**：树形归约V2（Tree-reduce-v2）。

**选项**：

- `-enable-ra`：在TreeReduceV2中启用RA（沿维度0归约）。
- `-enable-ar`：在TreeReduceV2中启用AR（沿维度1归约）。
- `-only-marked`：仅重写被标记为直接寄存器归约的循环。
- `-direct-register-ra`：使用直接的平衡寄存器树RA降阶方式。
- `-only-legacy-scope`：仅重写被选定采用Legacy树形策略的模块。
