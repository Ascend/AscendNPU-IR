# hivm方言

HIVM（Hybrid Intelligence Virtual Machine）方言，定义了用于异构计算的核心操作、属性与类型系统，涵盖数据搬运、矩阵运算、向量运算及同步机制。

## 操作定义

### hivm.hir.atomic_cas (hivm::AtomicCasOp)

**功能：** 执行原子比较并交换（CAS）操作。

**语法：**

```mlir
operation ::= `hivm.hir.atomic_cas` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`->` type($result_tensor)^)?
```

**示例：**

```mlir
hivm.hir.atomic_cas ins(%src0, %src1 : memref<?xf32>, memref<?xf32>) outs(%dst : memref<?xf32>)
%result = hivm.hir.atomic_cas ins(%src0, %src1 : tensor<?xf32>, tensor<?xf32>) outs(%dst : tensor<?xf32>) -> tensor<?xf32>
```

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result_tensor` | 类型为Tensor或Memref |

### hivm.hir.atomic_rmw (hivm::AtomicRMWOp)

**功能：** 执行原子读-修改-写操作。

**语法：**

```mlir
operation ::= `hivm.hir.atomic_rmw` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `atomic_kind` `=` $atomic_kind
              (`->` type($result_tensor)^)?
```

**示例：**

```mlir
hivm.hir.atomic_rmw ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) atomic_kind = <add>
%result = hivm.hir.atomic_rmw ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) atomic_kind = <or> -> tensor<?xf32>
```

**特性：** `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `InferCoreTypeInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 任意类型 |
| `dst` | 类型为Tensor或Memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result_tensor` | 类型为Tensor或Memref |

### hivm.hir.atomic_xchg (hivm::AtomicXchgOp)

**功能：** 执行原子交换操作。

**语法：**

```mlir
operation ::= `hivm.hir.atomic_xchg` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`mask` `(` $mask^ `:` type($mask) `)`)?
              (`->` type($result_tensor)^)?
```

**示例：**

```mlir
hivm.hir.atomic_xchg ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>)
%result = hivm.hir.atomic_cas ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) -> tensor<?xf32>
```

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 任意类型 |
| `dst` | 类型为Tensor或Memref |
| `mask` | 类型为Tensor或Memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result_tensor` | 类型为Tensor或Memref |

### hivm.hir.batchMmadL1 (hivm::BatchMmadL1Op)

**功能：** 从L1内存层级执行批处理矩阵乘加操作，支持批处理维度。

**语法：**

```mlir
operation ::= `hivm.hir.batchMmadL1` attr-dict `ins` `(`
              $a
              `,` $b
              `,` $init_condition
              `,` $real_m
              `,` $real_k
              `,` $real_n
              (`,` $per_channel_bias^)?
              `:`
              type($a)
              `,` type($b)
              `,` type($init_condition)
              `,` type($real_m)
              `,` type($real_k)
              `,` type($real_n)
              (`,` type($per_channel_bias)^)? `)`
              `outs` `(` $c `:` type($c) `)`
              (`sync_related_args` `(` $sync_related_args^ `:` type($sync_related_args) `)`)?
              (`unit_flag` `[` $unit_flag_mode^ (`,` $unit_flag_cond^)? `]`)?
              (`->` type($result_tensors)^)?
```

**示例：**

```mlir
hivm.hir.batchMmadL1 ins(%A, %B, %init, %m, %k, %n : memref<2x32x64xf16>, memref<2x64x32xf16>, i1, index, index, index) outs(%C : memref<2x32x32xf16>)
```

**特性：** `AttrSizedOperandSegments`, `CubeCoreTypeTrait`, `MacroOpPipeTrait<PIPE::PIPE_MTE1, PIPE::PIPE_M>`, `MacroOpTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `a` | 类型为Tensor或Memref |
| `b` | 类型为Tensor或Memref |
| `init_condition` | 1位无符号整数 |
| `real_m` | index类型 |
| `real_k` | index类型 |
| `real_n` | index类型 |
| `c` | 类型为Tensor或Memref |
| `sync_related_args` | 可变参数，64位无符号整数 |
| `unit_flag_cond` | 1位无符号整数 |
| `per_channel_bias` | 类型为Tensor或Memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result_tensors` | 可变参数，任意类型的ranked tensor |

### hivm.hir.bitcast (hivm::BitcastOp)

**功能：** 在不改变底层位表示的前提下，重新解释有形状值的位模式，进行元素类型转换。

**语法：**

```mlir
operation ::= `hivm.hir.bitcast` $src `:` type($src) `->` type($result) attr-dict
```

**示例：**

```mlir
%res = hivm.hir.bitcast %src : memref<4xf32> -> memref<4xi32>
```

**特性：** `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultShape`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 任意类型 |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 任意类型 |

### hivm.hir.convert_layout (hivm::ConvertLayoutOp)

**功能：** 将memref从一种数据布局转换为另一种布局，不复制或修改数据。

**语法：**

```mlir
operation ::= `hivm.hir.convert_layout` $source attr-dict `:` functional-type(operands, results)
```

**示例：**

```mlir
%res = hivm.hir.convert_layout %src {srcLayout = #hivm.data_layout<ND>, dstLayout = #hivm.data_layout<zN>} : memref<32x32xf16> -> memref<32x32xf16>
```

**特性：** `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultElementType`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `source` | 任意类型值的ranked或unranked memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 任意类型值的ranked或unranked memref |

### hivm.hir.copy (hivm::CopyOp)

**功能：** 在本地内存层级之间拷贝数据，支持非连续数据的重关联重塑。

**语法：**

```mlir
operation ::= `hivm.hir.copy` `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              attr-dict
              (`pad_mode` `=` $pad_mode^)?
              (`pad_value` `=` $pad_value^ `:` type($pad_value))?
              (`collapse_reassociation` `=` $collapse_reassociation^)?
              (`->` type($result_tensor)^)?
```

**示例：**

```mlir
hivm.hir.copy ins(%src : memref<16x16xf16, #hivm.address_space<ub>>) outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |
| `pad_value` | 任意类型 |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result_tensor` | 任意类型的ranked tensor |

### hivm.hir.create_sync_block_lock (hivm::CreateSyncBlockLockOp)

**功能：** 分配一块锁内存区域，用于确保锁与解锁之间的代码在块间按顺序执行。

**语法：**

```mlir
operation ::= `hivm.hir.create_sync_block_lock` (`from` $lockArg^)?
              attr-dict `:` (`from` type($lockArg)^ `to`)? type($memref)
```

**示例：**

```mlir
hivm.hir.create_sync_block_lock() : memref<1xi64>
hivm.hir.create_sync_block_lock() from %arg : from memref<?xi8> to memref<1xi64>
```

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `lockArg` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `memref` | 任意类型值的memref |

### hivm.hir.custom (hivm::CustomOp)

**功能：** 通用自定义操作接口，供用户编写内置操作无法满足或性能非最优的私有实现。

**语法：**

```mlir
operation ::= `hivm.hir.custom` $name attr-dict `ins` `(` $inputs `:` type($inputs) `)` `outs` `(` $outputs `:` type($outputs) `)`
```

**示例：**

```mlir
hivm.hir.custom "__builtin_gather_load" ins(%src : memref<256xf32>) outs(%dst : memref<256xf32>)
```

**特性：** `AttrSizedOperandSegments`, `SinglePipeOpTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `inputs` | 可变参数，任意类型 |
| `outputs` | 可变参数，任意类型 |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `results` | 可变参数，任意类型 |

### hivm.hir.dcci (hivm::DCCIOp)

**功能：** 清理（写回）并使一个缓存行或整个数据缓存失效。

**语法：**

```mlir
operation ::= `hivm.hir.dcci` attr-dict `(` $mode `,` $dataCacheKind (`,` $ptr^ `:` type($ptr))? `)`
```

**示例：**

```mlir
hivm.hir.dcci(#hivm.DCCIMode<single_cache_line>, #hivm.DataCacheKind<ub>, %ptr : memref<1024xf32>)
```

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `ptr` | 任意类型值的memref |

### hivm.hir.debug (hivm::DebugOp)

**功能：** 设备端调试操作，用于输出调试信息。

**语法：**

```mlir
operation ::= `hivm.hir.debug` attr-dict $arg `:` type($arg)
```

**示例：**

```mlir
hivm.hir.debug %arg {debugtype = "print", prefix = "value"} : f32
```

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `arg` | 整数、浮点数、Tensor或Memref |

### hivm.hir.finish_debug (hivm::FinishDebugOp)

**功能：** 设备端调试的结束函数。

**语法：**

```mlir
operation ::= `hivm.hir.finish_debug` attr-dict
```

**示例：**

```mlir
hivm.hir.finish_debug
```

**特性：** `CubeVectorCoreTypeTrait`

### hivm.hir.fixpipe (hivm::FixpipeOp)

**功能：** 从L0C到其他内存层级的数据搬移操作，支持前级量化、前级ReLU、逐元素加法、后级ReLU、后级量化及布局变换。

**语法：**

```mlir
operation ::= `hivm.hir.fixpipe` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`unit_flag` `[` $unit_flag_mode^ (`,` $unit_flag_cond^)? `]`)?
              (`->` type($result_tensor)^)?
```

**示例：**

```mlir
hivm.hir.fixpipe ins(%src : memref<16x16xf16, #hivm.address_space<l0c>>) outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>) {pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>}
```

**特性：** `AlwaysSpeculatableImplTrait`, `CubeCoreTypeTrait`, `OpPipeTrait<PIPE::PIPE_FIX>`, `SinglePipeOpTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 任意类型值的shaped类型 |
| `dst` | 任意类型值的shaped类型 |
| `unit_flag_cond` | 1位无符号整数 |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result_tensor` | 任意类型的ranked tensor |

### hivm.hir.get_block_idx (hivm::GetBlockIdxOp)

**功能：** 获取当前设备线程用于并行化的block索引。

**语法：**

```mlir
operation ::= `hivm.hir.get_block_idx` attr-dict `->` type($result)
```

**示例：**

```mlir
%idx = hivm.hir.get_block_idx -> i64
```

**特性：** `AlwaysSpeculatableImplTrait`, `CubeVectorCoreTypeTrait`

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 64位无符号整数 |

### hivm.hir.get_block_num (hivm::GetBlockNumOp)

**功能：** 获取当前设备线程用于并行化的block数量。

**语法：**

```mlir
operation ::= `hivm.hir.get_block_num` attr-dict `->` type($result)
```

**示例：**

```mlir
%num = hivm.hir.get_block_num -> i64
```

**特性：** `AlwaysSpeculatableImplTrait`, `CubeVectorCoreTypeTrait`

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 64位无符号整数 |

### hivm.hir.get_sub_block_idx (hivm::GetSubBlockIdxOp)

**功能：** 获取当前设备线程用于并行化的子块索引。

**语法：**

```mlir
operation ::= `hivm.hir.get_sub_block_idx` attr-dict `->` type($result)
```

**示例：**

```mlir
%sub_idx = hivm.hir.get_sub_block_idx -> i64
```

**特性：** `AlwaysSpeculatableImplTrait`, `CubeVectorCoreTypeTrait`

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 64位无符号整数 |

### hivm.hir.get_sub_block_num (hivm::GetSubBlockNumOp)

**功能：** 获取当前设备线程用于并行化的子块数量。

**语法：**

```mlir
operation ::= `hivm.hir.get_sub_block_num` attr-dict `->` type($result)
```

**示例：**

```mlir
%sub_num = hivm.hir.get_sub_block_num -> i64
```

**特性：** `AlwaysSpeculatableImplTrait`, `CubeVectorCoreTypeTrait`

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 64位无符号整数 |

### hivm.hir.get_sys_cnt (hivm::GetSysCntOp)

**功能：** 获取当前设备的系统计数。

**语法：**

```mlir
operation ::= `hivm.hir.get_sys_cnt` attr-dict `->` type($result)
```

**示例：**

```mlir
%cnt = hivm.hir.get_sys_cnt -> i64
```

**特性：** `AlwaysSpeculatableImplTrait`, `CubeVectorCoreTypeTrait`

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 64位无符号整数 |

### hivm.hir.init_debug (hivm::InitDebugOp)

**功能：** 设备端调试的初始化函数。

**语法：**

```mlir
operation ::= `hivm.hir.init_debug` attr-dict
```

**示例：**

```mlir
hivm.hir.init_debug
```

**特性：** `CubeVectorCoreTypeTrait`

### hivm.hir.load (hivm::LoadOp)

**功能：** 将数据从全局内存加载到本地缓冲区，支持填充模式与隐式转置。

**语法：**

```mlir
operation ::= `hivm.hir.load` `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              attr-dict
              (`pad_mode` `=` $pad_mode^)?
              (`pad_value` `=` $pad_value^ `:` type($pad_value))?
              (`left_padding_num` `=` $left_padding_num^ `:` type($left_padding_num))?
              (`init_out_buffer` `=` $init_out_buffer^ )?
              (`right_padding_num` `=` $right_padding_num^ `:` type($right_padding_num))?
              (`init_condition` `=` $init_condition^ `:` type($init_condition))?
              (`may_implicit_transpose_with_last_axis` `=` $may_implicit_transpose_with_last_axis^ )?
              (`->` type($result_tensor)^)?
```

**示例：**

```mlir
hivm.load ins(%src : memref<16x16xf16, #hivm.address_space<gm>>) outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `OpPipeTrait<PIPE::PIPE_MTE2>`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |
| `pad_value` | 任意类型 |
| `left_padding_num` | index类型 |
| `right_padding_num` | 任意类型 |
| `init_condition` | 任意类型 |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result_tensor` | 任意类型的ranked tensor |

### hivm.hir.load_scalar (hivm::LoadScalarOp)

**功能：** 从LLVM指针地址加载标量值。

**语法：**

```mlir
operation ::= `hivm.hir.load_scalar` attr-dict $addr `:` type($addr) `->` type($result)
```

**示例：**

```mlir
%val = hivm.hir.load_scalar %addr : !llvm.ptr<f32> -> f32
```

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `addr` | LLVM指针类型 |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 整数或浮点数 |

### hivm.hir.matmul (hivm::MatmulOp)

**功能：** 从全局内存执行矩阵乘法操作，支持转置、偏置与反量化。

**语法：**

```mlir
operation ::= `hivm.hir.matmul` attr-dict `ins` `(` $a `,` $b `:` type($a) `,` type($b) `)`
              `outs` `(` $c `:` type($c) `)`
              (`tiling_params` `=` $tilingParams^ `:` type($tilingParams) ) ?
              (`bias` `=` $bias^ `:` type($bias) )?
              (`descale` `=` $descale^ `:` type($descale))?
              (`a_transpose` $aTranspose^)?
              (`b_transpose` $bTranspose^)?
              (`descale_mode` `=` $descaleMode^)?
              (`block_sizes` `(` $blockSizes^ `:` type($blockSizes) `)`)?
              (`process_sizes` `(` $processSizes^ `:` type($processSizes) `)`)?
              (`swizzle_offset` `=` $swizzleOffset^ `:` type($swizzleOffset) )?
              (`swizzle_direction` `=` $swizzleDirection^ `:` type($swizzleDirection))?
              (`epilogue_p_tiles` `=` $epiloguePTiles^ `:` type($epiloguePTiles))?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.matmul ins(%A, %B : memref<32x64xf16>, memref<64x32xf16>) outs(%C : memref<32x32xf32>) {a_transpose, block_sizes = [16, 16, 16]}
```

**特性：** `AttrSizedOperandSegments`, `MacroOpPipeTrait<PIPE::PIPE_MTE2, PIPE::PIPE_MTE3>`, `MacroOpTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `a` | 任意类型值的shaped类型 |
| `b` | 任意类型值的shaped类型 |
| `tilingParams` | 任意类型值的shaped类型 |
| `bias` | 任意类型值的shaped类型 |
| `descale` | 任意类型值的shaped类型 |
| `blockSizes` | 可变参数，64位无符号整数 |
| `processSizes` | 可变参数，64位无符号整数 |
| `swizzleOffset` | 64位无符号整数 |
| `swizzleDirection` | 64位无符号整数 |
| `epiloguePTiles` | 64位无符号整数 |
| `c` | 任意类型值的shaped类型 |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.mix_group_matmul (hivm::MixGroupMatmulOp)

**功能：** 执行分组矩阵乘法，支持按专家分配token并与后向量函数融合。

**语法：**

```mlir
operation ::= `hivm.hir.mix_group_matmul` attr-dict `ins` `(` $a `,` $b `,` $tokens_per_expert `:` type($a) `,` type($b) `,` type($tokens_per_expert) `)`
              (`post_vector_func_ins` `(` $postVecFuncIns^ `:` type($postVecFuncIns) `)`) ?
              (`post_vector_func_outs` `(` $postVecFuncOuts^ `:` type($postVecFuncOuts) `)`) ?
              (`workspace_ins` `(` $workspaceIns^ `:` type($workspaceIns) `)`) ?
              `outs` `(` $c `:` type($c) `)`
              (`tiling_params` `=` $tilingParams^ `:` type($tilingParams) ) ?
              (`comm_params` `=` $commParams^ `:` type($commParams) ) ?
              (`bias` `=` $bias^ `:` type($bias) )?
              (`descale` `=` $descale^ `:` type($descale))?
              (`a_transpose` $aTranspose^)?
              (`b_transpose` $bTranspose^)?
              (`descale_mode` `=` $descaleMode^)?
              (`block_sizes` `(` $blockSizes^ `:` type($blockSizes) `)`)?
              (`process_sizes` `(` $processSizes^ `:` type($processSizes) `)`)?
              (`swizzle_offset` `=` $swizzleOffset^ `:` type($swizzleOffset) )?
              (`swizzle_direction` `=` $swizzleDirection^ `:` type($swizzleDirection))?
              (`epilogue_p_tiles` `=` $epiloguePTiles^ `:` type($epiloguePTiles))?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.mix_group_matmul ins(%A, %B, %tokens : memref<128x64xf16>, memref<64x32xf16>, memref<8xi32>) outs(%C : memref<128x32xf32>)
```

**特性：** `AttrSizedOperandSegments`, `MacroOpPipeTrait<PIPE::PIPE_MTE2, PIPE::PIPE_MTE3>`, `MacroOpTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `a` | 任意类型值的shaped类型 |
| `b` | 任意类型值的shaped类型 |
| `tokens_per_expert` | 任意类型值的shaped类型 |
| `postVecFuncIns` | 可变参数，任意类型值的shaped类型 |
| `postVecFuncOuts` | 可变参数，任意类型值的shaped类型 |
| `workspaceIns` | 可变参数，任意类型值的shaped类型 |
| `tilingParams` | 任意类型值的shaped类型 |
| `commParams` | 任意类型值的shaped类型 |
| `bias` | 任意类型值的shaped类型 |
| `descale` | 任意类型值的shaped类型 |
| `blockSizes` | 可变参数，64位无符号整数 |
| `processSizes` | 可变参数，64位无符号整数 |
| `swizzleOffset` | 64位无符号整数 |
| `swizzleDirection` | 64位无符号整数 |
| `epiloguePTiles` | 64位无符号整数 |
| `c` | 任意类型值的shaped类型 |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.mix_matmul (hivm::MixMatmulOp)

**功能：** 执行矩阵乘法，支持与后向量函数进行tile级融合。

**语法：**

```mlir
operation ::= `hivm.hir.mix_matmul` attr-dict `ins` `(` $a `,` $b `:` type($a) `,` type($b) `)`
              (`post_vector_func_ins` `(` $postVecFuncIns^ `:` type($postVecFuncIns) `)`) ?
              (`workspace_ins` `(` $workspaceIns^ `:` type($workspaceIns) `)`) ?
              `outs` `(` $c `:` type($c) `)`
              (`tiling_params` `=` $tilingParams^ `:` type($tilingParams) ) ?
              (`comm_params` `=` $commParams^ `:` type($commParams) ) ?
              (`bias` `=` $bias^ `:` type($bias) )?
              (`descale` `=` $descale^ `:` type($descale))?
              (`a_transpose` $aTranspose^)?
              (`b_transpose` $bTranspose^)?
              (`descale_mode` `=` $descaleMode^)?
              (`block_sizes` `(` $blockSizes^ `:` type($blockSizes) `)`)?
              (`process_sizes` `(` $processSizes^ `:` type($processSizes) `)`)?
              (`swizzle_offset` `=` $swizzleOffset^ `:` type($swizzleOffset) )?
              (`swizzle_direction` `=` $swizzleDirection^ `:` type($swizzleDirection))?
              (`epilogue_p_tiles` `=` $epiloguePTiles^ `:` type($epiloguePTiles))?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.mix_matmul ins(%A, %B : memref<32x64xf16>, memref<64x32xf16>) outs(%C : memref<32x32xf32>) {a_transpose}
```

**特性：** `AttrSizedOperandSegments`, `MacroOpPipeTrait<PIPE::PIPE_MTE2, PIPE::PIPE_MTE3>`, `MacroOpTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `a` | 任意类型值的shaped类型 |
| `b` | 任意类型值的shaped类型 |
| `postVecFuncIns` | 可变参数，任意类型值的shaped类型 |
| `workspaceIns` | 可变参数，任意类型值的shaped类型 |
| `tilingParams` | 任意类型值的shaped类型 |
| `commParams` | 任意类型值的shaped类型 |
| `bias` | 任意类型值的shaped类型 |
| `descale` | 任意类型值的shaped类型 |
| `blockSizes` | 可变参数，64位无符号整数 |
| `processSizes` | 可变参数，64位无符号整数 |
| `swizzleOffset` | 64位无符号整数 |
| `swizzleDirection` | 64位无符号整数 |
| `epiloguePTiles` | 64位无符号整数 |
| `c` | 任意类型值的shaped类型 |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.mmadL1 (hivm::MmadL1Op)

**功能：** 从L1内存层级执行矩阵乘加操作。

**语法：**

```mlir
operation ::= `hivm.hir.mmadL1` attr-dict `ins` `(`
              $a
              `,` $b
              `,` $init_condition
              `,` $real_m
              `,` $real_k
              `,` $real_n
              (`,` $per_channel_bias^)?
              `:`
              type($a)
              `,` type($b)
              `,` type($init_condition)
              `,` type($real_m)
              `,` type($real_k)
              `,` type($real_n)
              (`,` type($per_channel_bias)^)? `)`
              `outs` `(` $c `:` type($c) `)`
              (`sync_related_args` `(` $sync_related_args^ `:` type($sync_related_args) `)`)?
              (`unit_flag` `[` $unit_flag_mode^ (`,` $unit_flag_cond^)? `]`)?
              (`->` type($result_tensors)^)?
```

**示例：**

```mlir
hivm.hir.mmadL1 ins(%A, %B, %init, %m, %k, %n : memref<32x64xf16>, memref<64x32xf16>, i1, index, index, index) outs(%C : memref<32x32xf16>)
```

**特性：** `AttrSizedOperandSegments`, `CubeCoreTypeTrait`, `MacroOpPipeTrait<PIPE::PIPE_MTE1, PIPE::PIPE_M>`, `MacroOpTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `a` | 类型为Tensor或Memref |
| `b` | 类型为Tensor或Memref |
| `init_condition` | 1位无符号整数 |
| `real_m` | index类型 |
| `real_k` | index类型 |
| `real_n` | index类型 |
| `c` | 类型为Tensor或Memref |
| `sync_related_args` | 可变参数，64位无符号整数 |
| `unit_flag_cond` | 1位无符号整数 |
| `per_channel_bias` | 类型为Tensor或Memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result_tensors` | 可变参数，任意类型的ranked tensor |

### hivm.hir.nd2nz (hivm::ND2NZOp)

**功能：** 执行即时ND到NZ布局变换的数据拷贝操作。

**语法：**

```mlir
operation ::= `hivm.hir.nd2nz` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`init_out_buffer` `=` $init_out_buffer^ )?
              (`pad_value` `=` $pad_value^ `:` type($pad_value))?
              (`init_condition` `=` $init_condition^ `:` type($init_condition))?
              (`->` type($result_tensor)^)?
```

**示例：**

```mlir
hivm.hir.nd2nz ins(%src : memref<32x32xf16>) outs(%dst : memref<32x32xf16>) {dst_continuous}
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CubeCoreTypeTrait`, `OpPipeTrait<PIPE::PIPE_MTE2>`, `SinglePipeOpTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 任意类型值的shaped类型 |
| `dst` | 任意类型值的shaped类型 |
| `pad_value` | 任意类型 |
| `init_condition` | 任意类型 |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result_tensor` | 可变参数，任意类型的ranked tensor |

### hivm.hir.nz2nd (hivm::NZ2NDOp)

**功能：** 从L1到全局内存执行NZ2ND转换的数据拷贝操作。

**语法：**

```mlir
operation ::= `hivm.hir.nz2nd` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`->` type($result_tensor)^)?
```

**示例：**

```mlir
hivm.hir.nz2nd ins(%src : memref<32x32xf16, #hivm.address_space<l1>>) outs(%dst : memref<32x32xf16, #hivm.address_space<gm>>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `CubeCoreTypeTrait`, `OpPipeTrait<PIPE::PIPE_MTE3>`, `SinglePipeOpTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result_tensor` | 任意类型的ranked tensor |

### hivm.hir.pipe_barrier (hivm::PipeBarrierOp)

**功能：** 在指定pipe上插入屏障，用于同步。

**语法：**

```mlir
operation ::= `hivm.hir.pipe_barrier` `[` $pipe `]` attr-dict
```

**示例：**

```mlir
hivm.hir.pipe_barrier [#hivm.pipe<PIPE_V>]
```

### hivm.hir.pointer_cast (hivm::PointerCastOp)

**功能：** 将指定的64位整型地址转换为memref。

**语法：**

```mlir
operation ::= `hivm.hir.pointer_cast` `(`$addrs `)` (`[` $dynamicSizes^`]`)? attr-dict `:` type($result)
```

**示例：**

```mlir
%addr = arith.constant 1234 : i64
%tmp = hivm.hir.pointer_cast(%addr) : memref<32xf32>

%addr2 = arith.constant 1600 : i64
%addr3 = arith.constant 3200 : i64
%tmp2 = hivm.hir.pointer_cast(%addr, %addr2) : memref<32xf32>
%tmp3 = hivm.hir.pointer_cast(%addr, %addr2, %addr3) : memref<32xf32>
```

**特性：** `AttrSizedOperandSegments`, `CubeVectorCoreTypeTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `addrs` | 可变参数，64位无符号整数 |
| `dynamicSizes` | 可变参数，index类型 |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 任意类型值的memref |

### hivm.hir.set_ffts_base_addr (hivm::SetFFTSBaseAddrOp)

**功能：** 设置FFTS同步机制的基础地址。

**语法：**

```mlir
operation ::= `hivm.hir.set_ffts_base_addr` attr-dict $ffts_base_addr
```

**示例：**

```mlir
hivm.hir.set_ffts_base_addr %base_addr
```

**特性：** `CubeVectorCoreTypeTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `ffts_base_addr` | 64位无符号整数 |

### hivm.hir.set_flag (hivm::SetFlagOp)

**功能：** 设置同步标志。

**语法：**

```mlir
operation ::= `hivm.hir.set_flag` `[`
              $set_pipe
              `,` $wait_pipe
              `,` custom<EventID>($static_event_id, $dynamic_event_id)
              `]` attr-dict
```

**示例：**

```mlir
hivm.hir.set_flag [#hivm.pipe<PIPE_V>, #hivm.pipe<PIPE_M>, #hivm.event<EVENT_ID0>]
```

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `dynamic_event_id` | 64位无符号整数 |

### hivm.hir.set_mask_norm (hivm::SetMaskNormOp)

**功能：** 设置掩码归一化模式。

**语法：**

```mlir
operation ::= `hivm.hir.set_mask_norm` attr-dict
```

**示例：**

```mlir
hivm.hir.set_mask_norm
```

### hivm.hir.store (hivm::StoreOp)

**功能：** 将本地缓冲区数据存储到全局内存，支持原子操作。

**语法：**

```mlir
operation ::= `hivm.hir.store` `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              attr-dict
              (`atomic` `=` $atomic_kind^)?
              (`->` type($result_tensor)^)?
```

**示例：**

```mlir
hivm.store ins(%src : memref<16x16xf16, #hivm.address_space<ub>>) outs(%dst : memref<16x16xf16, #hivm.address_space<gm>>)
hivm.store ins(%src : memref<16x16xf16, #hivm.address_space<ub>>) outs(%dst : memref<16x16xf16, #hivm.address_space<gm>>) atomic = #hivm.atomic_kind<add>
```

**特性：** `AlwaysSpeculatableImplTrait`, `OpPipeTrait<PIPE::PIPE_MTE3>`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result_tensor` | 任意类型的ranked tensor |

### hivm.hir.sync_block (hivm::SyncBlockOp)

**功能：** 在不同内核间执行块同步，支持多种同步模式。

**语法：**

```mlir
operation ::= `hivm.hir.sync_block` attr-dict `[` $sync_block_mode (`,` $flag_id^)?`]`
              (`ffts_base_addr` `=` $ffts_base_addr^)?
              (`tcube_pipe` `=` $tcube_pipe^)?
              (`tvector_pipe` `=` $tvector_pipe^)?
```

**示例：**

```mlir
hivm.hir.sync_block [#hivm.sync_block_mode<ALL>] {tvector_pipe = #hivm.pipe<PIPE_V>}
```

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `ffts_base_addr` | 64位无符号整数 |

### hivm.hir.sync_block_lock (hivm::SyncBlockLockOp)

**功能：** 等待锁变量等于当前block索引，实现块间同步。

**语法：**

```mlir
operation ::= `hivm.hir.sync_block_lock` attr-dict `lock_var` `(` $lock_var `:` type($lock_var) `)`
```

**示例：**

```mlir
hivm.hir.sync_block_lock lock_var(%lock : memref<1xi64>)
```

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `lock_var` | 64位无符号整数的1D memref |

### hivm.hir.sync_block_set (hivm::SyncBlockSetOp)

**功能：** 设置块同步的同步点。

**语法：**

```mlir
operation ::= `hivm.hir.sync_block_set` attr-dict `[` $tcore_type `,` $tpipe `,` $pipe`]`
              `flag` `=` custom<FlagID>($static_flag_id, $dynamic_flag_id)
              (`ffts_base_addr` `=` $ffts_base_addr^)?
              (`sync_instr_mode` `=` $tsync_instr_mode^)?
```

**示例：**

```mlir
hivm.hir.sync_block_set [#hivm.tcore_type<CUBE>, #hivm.pipe<PIPE_M>, #hivm.pipe<PIPE_V>] flag = #hivm.event<EVENT_ID0>
```

**特性：** `AttrSizedOperandSegments`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `dynamic_flag_id` | 64位无符号整数 |
| `ffts_base_addr` | 64位无符号整数 |

### hivm.hir.sync_block_unlock (hivm::SyncBlockUnlockOp)

**功能：** 递增并释放锁变量。

**语法：**

```mlir
operation ::= `hivm.hir.sync_block_unlock` attr-dict `lock_var` `(` $lock_var `:` type($lock_var) `)`
```

**示例：**

```mlir
hivm.hir.sync_block_unlock lock_var(%lock : memref<1xi64>)
```

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `lock_var` | 64位无符号整数的1D memref |

### hivm.hir.sync_block_wait (hivm::SyncBlockWaitOp)

**功能：** 等待指定块同步标志。

**语法：**

```mlir
operation ::= `hivm.hir.sync_block_wait` attr-dict `[` $tcore_type `,` $tpipe `,` $pipe`]`
              `flag` `=` custom<FlagID>($static_flag_id, $dynamic_flag_id)
```

**示例：**

```mlir
hivm.hir.sync_block_wait [#hivm.tcore_type<VECTOR>, #hivm.pipe<PIPE_V>, #hivm.pipe<PIPE_M>] flag = #hivm.event<EVENT_ID1>
```

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `dynamic_flag_id` | 64位无符号整数 |

### hivm.hir.vabs (hivm::VAbsOp)

**功能：** 逐元素计算向量的绝对值。

**语法：**

```mlir
operation ::= `hivm.hir.vabs` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vabs ins(%src : memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型值的shaped类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vadd (hivm::VAddOp)

**功能：** 逐元素执行二元向量加法运算。

**语法：**

```mlir
operation ::= `hivm.hir.vadd` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vadd ins(%src0, %src1 : memref<32xf32>, memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `CommutativeOpTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vand (hivm::VAndOp)

**功能：** 逐元素执行二元向量按位与运算。

**语法：**

```mlir
operation ::= `hivm.hir.vand` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vand ins(%src0, %src1 : memref<32xi32>, memref<32xi32>) outs(%dst : memref<32xi32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `CommutativeOpTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`, `VectorOnlyTrait<1>`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.varange (hivm::VArangeOp)

**功能：** 根据步长和偏移量生成等差序列填充向量。

**语法：**

```mlir
operation ::= `hivm.hir.varange` attr-dict
              (`offset` `[` $offset^ `]`)?
              `strides` `[` $strides `]`
              `outs` `(` $dst `:` type($dst) `)`
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.varange offset[%o] strides[%s0, %s1] outs(%dst : memref<32xf32>)
%result = hivm.hir.varange offset[%o] strides[%s0, %s1] outs(%dst : tensor<32xf32>) -> tensor<32xf32>
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `VectorCoreTypeTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `dst` | 类型为Tensor或Memref |
| `offset` | index类型 |
| `strides` | 可变参数，index类型 |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 任意类型的ranked tensor |

### hivm.hir.vbrc (hivm::VBrcOp)

**功能：** 将向量或标量沿指定维度进行广播。

**语法：**

```mlir
operation ::= `hivm.hir.vbrc` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast_dims` `=` $broadcast_dims^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vbrc ins(%src : i32) outs(%dst : memref<?xi32>)
hivm.hir.vbrc ins(%src : memref<1xi32>) outs(%dst : memref<?xi32>) broadcast_dims = [0]
%result = hivm.hir.vbrc ins(%src : tensor<1xi32>) outs(%dst : tensor<?xi32>) broadcast_dims = [0] -> tensor<?xi32>
```

**特性：** `AlwaysSpeculatableImplTrait`, `CollapsibleConsecutiveTargetDimsTrait`, `SameOperandsElementType`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 任意类型 |
| `dst` | 类型为Tensor或Memref |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vcast (hivm::VCastOp)

**功能：** 逐元素执行向量类型转换，支持多种舍入模式。

**语法：**

```mlir
operation ::= `hivm.hir.vcast` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`round_mode` `=` $round_mode^)?
              (`cast` `=` $cast^)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vcast ins(%src : memref<32xf32>) outs(%dst : memref<32xi32>) {round_mode = #hivm.round_mode<rint>}
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vcmp (hivm::VCmpOp)

**功能：** 逐元素执行二元向量比较，结果存入bool向量。

**语法：**

```mlir
operation ::= `hivm.hir.vcmp` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`compare_mode` `=` $compare_mode^)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vcmp ins(%src0, %src1 : memref<32xf32>, memref<32xf32>) outs(%dst : memref<32xi1>) {compare_mode = #hivm.compare_mode<GE>}
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vconcat (hivm::VConcatOp)

**功能：** 沿指定维度拼接多个向量。

**语法：**

```mlir
operation ::= `hivm.hir.vconcat` `dim` `(` $dim `)` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vconcat dim(1) ins(%0, %1 : tensor<136x2048xf32>, tensor<136x2048xf32>) outs(%2 : tensor<136x4096xf32>) -> tensor<136x4096xf32>
```

**特性：** `AlwaysSpeculatableImplTrait`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 类型为Tensor或Memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vcos (hivm::VCosOp)

**功能：** 逐元素计算向量的余弦值。

**语法：**

```mlir
operation ::= `hivm.hir.vcos` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vcos ins(%src : memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vcumprod (hivm::VCumprodOp)

**功能：** 沿指定维度计算向量的累积乘积，支持反向方向。

**语法：**

```mlir
operation ::= `hivm.hir.vcumprod` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `cum_dims` `=` $cum_dims
              `reverse` `=` $reverse
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vcumprod ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) cum_dims : [0] reverse = true
%result = hivm.hir.vcumprod ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) cum_dims : [0] reverse = true -> tensor<?xf32>
```

**特性：** `AlwaysSpeculatableImplTrait`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vcumsum (hivm::VCumsumOp)

**功能：** 沿指定维度计算向量的累积和，支持反向方向。

**语法：**

```mlir
operation ::= `hivm.hir.vcumsum` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `cum_dims` `=` $cum_dims
              `reverse` `=` $reverse
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vcumsum ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) cum_dims : [0] reverse = true
%result = hivm.hir.vcumsum ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) cum_dims : [0] reverse = true -> tensor<?xf32>
```

**特性：** `AlwaysSpeculatableImplTrait`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vdeinterleave (hivm::VDeinterleaveOp)

**功能：** 沿最后一个维度对向量进行解交织。

**语法：**

```mlir
operation ::= `hivm.hir.vdeinterleave` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`channel_num` `=` $channel_num^)?
              (`index_mode` `=` $index_mode^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vdeinterleave ins(%src : memref<32x16xf32>) outs(%dst0, %dst1 : memref<32x8xf32>, memref<32x8xf32>) {channel_num = 2}
```

**特性：** `AlwaysSpeculatableImplTrait`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 可变参数，类型为Tensor或Memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vdiv (hivm::VDivOp)

**功能：** 逐元素执行二元向量除法运算。

**语法：**

```mlir
operation ::= `hivm.hir.vdiv` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vdiv ins(%src0, %src1 : memref<32xf32>, memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.verf (hivm::VErfOp)

**功能：** 逐元素计算向量的误差函数。

**语法：**

```mlir
operation ::= `hivm.hir.verf` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.verf ins(%src : memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vexp (hivm::VExpOp)

**功能：** 逐元素计算向量的指数值。

**语法：**

```mlir
operation ::= `hivm.hir.vexp` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vexp ins(%src : memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型值的shaped类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vflip (hivm::VFlipOp)

**功能：** 沿指定轴翻转向量元素顺序。

**语法：**

```mlir
operation ::= `hivm.hir.vflip` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `flip_axis` `=` $flip_axis
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vflip ins(%src : memref<32x16xf32>) outs(%dst : memref<32x16xf32>) flip_axis = 1
```

**特性：** `AlwaysSpeculatableImplTrait`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vgather (hivm::VGatherOp)

**功能：** 根据索引从源向量中收集元素。

**语法：**

```mlir
operation ::= `hivm.hir.vgather` attr-dict `ins` `(` $src `:` type($src) `)`
              `indices` `(` $indices `:` type($indices) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vgather ins(%src : memref<32x16xf32>) indices(%idx : memref<32x8xi32>) outs(%dst : memref<32x8xf32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `indices` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vinterleave (hivm::VInterleaveOp)

**功能：** 沿最后一个维度交织多个向量的值。

**语法：**

```mlir
operation ::= `hivm.hir.vinterleave` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `interleave_channel_nums` `=` $interleave_channel_nums
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vinterleave ins(%src0, %src1 : memref<32x8xf32>, memref<32x8xf32>) outs(%dst : memref<32x16xf32>) interleave_channel_nums = 2
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 类型为Tensor或Memref |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vln (hivm::VLnOp)

**功能：** 逐元素计算向量的自然对数。

**语法：**

```mlir
operation ::= `hivm.hir.vln` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vln ins(%src : memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型值的shaped类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vmax (hivm::VMaxOp)

**功能：** 逐元素计算两个向量的最大值。

**语法：**

```mlir
operation ::= `hivm.hir.vmax` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vmax ins(%src0, %src1 : memref<32xf32>, memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `CommutativeOpTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vmin (hivm::VMinOp)

**功能：** 逐元素计算两个向量的最小值。

**语法：**

```mlir
operation ::= `hivm.hir.vmin` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vmin ins(%src0, %src1 : memref<32xf32>, memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `CommutativeOpTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vmod (hivm::VModOp)

**功能：** 逐元素计算向量取模运算。

**语法：**

```mlir
operation ::= `hivm.hir.vmod` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vmod ins(%src0, %src1 : memref<32xi32>, memref<32xi32>) outs(%dst : memref<32xi32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vmul (hivm::VMulOp)

**功能：** 逐元素执行二元向量乘法运算。

**语法：**

```mlir
operation ::= `hivm.hir.vmul` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vmul ins(%src0, %src1 : memref<32xf32>, memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `CommutativeOpTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vmulext (hivm::VMulExtOp)

**功能：** 逐元素执行二元向量乘法，并计算高32位结果。

**语法：**

```mlir
operation ::= `hivm.hir.vmulext` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vmulext ins(%src0, %src1 : memref<32xi32>, memref<32xi32>) outs(%dst : memref<32xi32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vmulextended (hivm::VMulextendedOp)

**功能：** 对两个tensor执行向量乘法，同时获取高16位和低16位结果。

**语法：**

```mlir
operation ::= `hivm.hir.vmulextended` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vmulextended ins(%src0, %src1 : memref<32xi16>, memref<32xi16>) outs(%dst_hi, %dst_lo : memref<32xi16>, memref<32xi16>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，类型为Tensor或Memref |
| `dst` | 可变参数，类型为Tensor或Memref |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vnot (hivm::VNotOp)

**功能：** 逐元素执行向量按位非运算。

**语法：**

```mlir
operation ::= `hivm.hir.vnot` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vnot ins(%src : memref<32xi32>) outs(%dst : memref<32xi32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型值的shaped类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vor (hivm::VOrOp)

**功能：** 逐元素执行二元向量按位或运算。

**语法：**

```mlir
operation ::= `hivm.hir.vor` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vor ins(%src0, %src1 : memref<32xi32>, memref<32xi32>) outs(%dst : memref<32xi32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `CommutativeOpTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`, `VectorOnlyTrait<1>`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vpad (hivm::VPadOp)

**功能：** 对输入向量进行填充，类似`tensor.pad`语义。

**语法：**

```mlir
operation ::= `hivm.hir.vpad` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `low` `` custom<DynamicIndexList>($low, $static_low)
              `high` `` custom<DynamicIndexList>($high, $static_high)
              `pad_value` $pad_value `:` type($pad_value)
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vpad ins(%src : tensor<2x16xf32>) outs(%dst: tensor<?x16xf32>) low[%first_dim_low, 0] high[%first_dim_high, 0] pad_value %pad_value : f32 -> tensor<?x16xf32>
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |
| `pad_value` | 任意类型 |
| `low` | 可变参数，index类型 |
| `high` | 可变参数，index类型 |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vpow (hivm::VPowOp)

**功能：** 逐元素执行向量幂运算（指数为标量或向量）。

**语法：**

```mlir
operation ::= `hivm.hir.vpow` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vpow ins(%src0, %src1 : memref<32xf32>, memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`, `VectorOnlyTrait<1>`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vrec (hivm::VRecOp)

**功能：** 逐元素计算向量的倒数。

**语法：**

```mlir
operation ::= `hivm.hir.vrec` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vrec ins(%src : memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型值的shaped类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vreduce (hivm::VReduceOp)

**功能：** 沿指定轴对向量进行规约操作。

**语法：**

```mlir
operation ::= `hivm.hir.vreduce` attr-dict $arith `ins` `(` $src `:` type($src) `)`
              (`indices` `(` $indices^ `:` type($indices) `)`)?
              `outs` `(` $dst `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              `reduce_dims` `=` $reduce_dims
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vreduce <add> ins(%src : memref<?xf32>) outs(%dst : memref<1xf32>) reduce_dims : [1]
%result = hivm.hir.vreduce <max> ins(%src : tensor<?xf32>) outs(%dst : tensor<1xf32>) reduce_dims : [0] -> tensor<1xf32>
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 可变参数，类型为Tensor或Memref |
| `temp_buffer` | 任意类型值的memref |
| `indices` | 类型为Tensor或Memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vrelu (hivm::VReluOp)

**功能：** 逐元素计算向量的ReLU激活函数。

**语法：**

```mlir
operation ::= `hivm.hir.vrelu` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vrelu ins(%src : memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型值的shaped类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vrsqrt (hivm::VRsqrtOp)

**功能：** 逐元素计算向量倒数平方根。

**语法：**

```mlir
operation ::= `hivm.hir.vrsqrt` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vrsqrt ins(%src : memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型值的shaped类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vsel (hivm::VSelOp)

**功能：** 根据条件向量逐元素选择两个源向量中的值。

**语法：**

```mlir
operation ::= `hivm.hir.vsel` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vsel ins(%cond, %src0, %src1 : memref<32xi1>, memref<32xf32>, memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<3>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vshl (hivm::VShLOp)

**功能：** 逐元素执行向量左移运算（标量移位量）。

**语法：**

```mlir
operation ::= `hivm.hir.vshl` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vshl ins(%src_vec, %shift : memref<32xi32>, i32) outs(%dst : memref<32xi32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `ScalarOnlyHWTrait<1>`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vshr (hivm::VShROp)

**功能：** 逐元素执行向量右移运算（标量移位量），支持算术右移舍入。

**语法：**

```mlir
operation ::= `hivm.hir.vshr` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`round` `:` $round^ )?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vshr ins(%src_vec, %shift : memref<32xi32>, i32) outs(%dst : memref<32xi32>) {round}
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `ScalarOnlyHWTrait<1>`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vsin (hivm::VSinOp)

**功能：** 逐元素计算向量的正弦值。

**语法：**

```mlir
operation ::= `hivm.hir.vsin` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vsin ins(%src : memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vsort (hivm::VSortOp)

**功能：** 对向量的指定轴进行排序，输出排序后的值和对应索引。

**语法：**

```mlir
operation ::= `hivm.hir.vsort` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `descending` `=` $descending
              `sort_axis` `=` $sort_axis
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vsort ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) descending = true sort_axis = 0
%result = hivm.hir.vsort ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) descending = true sort_axis = 0 -> tensor<?xf32>
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `VectorCoreTypeTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 可变参数，类型为Tensor或Memref |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vsqrt (hivm::VSqrtOp)

**功能：** 逐元素计算向量的平方根。

**语法：**

```mlir
operation ::= `hivm.hir.vsqrt` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vsqrt ins(%src : memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型值的shaped类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vsub (hivm::VSubOp)

**功能：** 逐元素执行二元向量减法运算。

**语法：**

```mlir
operation ::= `hivm.hir.vsub` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vsub ins(%src0, %src1 : memref<32xf32>, memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vtanh (hivm::VTanhOp)

**功能：** 逐元素计算向量的双曲正切值。

**语法：**

```mlir
operation ::= `hivm.hir.vtanh` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vtanh ins(%src : memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vtranspose (hivm::VTransposeOp)

**功能：** 根据指定的排列对向量维度进行转置。

**语法：**

```mlir
operation ::= `hivm.hir.vtranspose` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`permutation` `=` $permutation^)?
              (`disable_align` `=` $disable_align^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vtranspose ins(%src : memref<32x8xf32>) outs(%dst : memref<8x32xf32>) permutation = [1, 0]
%result = hivm.hir.vtranspose ins(%src : tensor<32x8xf32>) outs(%dst: tensor<8x32xf32>) permutation = [1, 0] -> tensor<8x32xf32>
```

**特性：** `AlwaysSpeculatableImplTrait`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vxor (hivm::VXorOp)

**功能：** 逐元素执行二元向量按位异或运算。

**语法：**

```mlir
operation ::= `hivm.hir.vxor` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例：**

```mlir
hivm.hir.vxor ins(%src0, %src1 : memref<32xi32>, memref<32xi32>) outs(%dst : memref<32xi32>)
```

**特性：** `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`, `VectorOnlyTrait<1>`

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型值的shaped类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果：**

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.wait_flag (hivm::WaitFlagOp)

**功能：** 等待指定同步标志。

**语法：**

```mlir
operation ::= `hivm.hir.wait_flag` `[`
              $set_pipe
              `,` $wait_pipe
              `,` custom<EventID>($static_event_id, $dynamic_event_id)
              `]` attr-dict
```

**示例：**

```mlir
hivm.hir.wait_flag [#hivm.pipe<PIPE_M>, #hivm.pipe<PIPE_V>, #hivm.event<EVENT_ID0>]
```

**操作数：**

| 操作数 | 说明 |
| :----: | ---- |
| `dynamic_event_id` | 64位无符号整数 |

## 属性

### AddressSpaceAttr

**功能：** 定义HIVM地址空间映射属性，映射到GM、L1、L0A、L0B、L0C和UB。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| address_space | `::mlir::hivm::AddressSpace` | 类型为AddressSpace的枚举 |

### AlignKindAttr

**功能：** 定义对齐方式信息。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::AlignKind` | 类型为AlignKind的枚举 |

### AllocAlignDimsAttr

**功能：** 定义分配对齐维度信息。

无参数。

### AllocAlignValueInByteAttr

**功能：** 定义按字节对齐的分配大小信息。

无参数。

### AtomicKindAttr

**功能：** 定义StoreOp的原子操作类型。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::AtomicKind` | 类型为AtomicKind的枚举 |

### AxisKindAttr

**功能：** 定义轴类型信息。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::AxisKind` | 类型为AxisKind的枚举 |

### HIVMBlockMappingAttr

**功能：** 定义块映射信息。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| order | `std::optional<int32_t>` | 顺序标识 |

### CompareModeAttr

**功能：** 定义VCmpOp的比较模式。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::CompareMode` | 类型为CompareMode的枚举 |

### DCCIModeAttr

**功能：** 定义DCCI模式。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::DCCIMode` | 类型为DCCIMode的枚举 |

### DataCacheKindAttr

**功能：** 定义数据缓存类型。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::DataCacheKind` | 类型为DataCacheKind的枚举 |

### DataLayoutAttr

**功能：** 定义HIVM数据布局映射属性，包括DOTA_ND、DOTB_ND、DOTC_ND、zN、nZ和ND，支持转置标识。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| data_layout | `::mlir::hivm::DataLayout` | 类型为DataLayout的枚举 |
| transpose | `std::optional<bool>` | 转置标识 |
| fractalSizes | `std::optional<DenseI64ArrayAttr>` | 分形尺寸 |

### DeinterleaveModeAttr

**功能：** 定义解交织索引模式。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::DeinterleaveMode` | 类型为DeinterleaveMode的枚举 |

### DescaleModeAttr

**功能：** 定义MatmulOp的反量化模式。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::DescaleMode` | 类型为DescaleMode的枚举 |

### DisableAutoInjectBlockSyncAttr

**功能：** 禁用自动注入块同步。

无参数。

### EventAttr

**功能：** 定义同步事件属性。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| event | `::mlir::hivm::EVENT` | 类型为EVENT的枚举 |

### FixpipePreQuantModeAttr

**功能：** 定义Fixpipe前级量化模式。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::FixpipePreQuantMode` | 类型为FixpipePreQuantMode的枚举 |

### FixpipePreReluModeAttr

**功能：** 定义Fixpipe前级ReLU模式。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::FixpipePreReluMode` | 类型为FixpipePreReluMode的枚举 |

### HIVMFuncDynMemrefArgsAttr

**功能：** 标记函数中动态memref参数的索引数组。

无参数。

### InsertSliceSourceIndexAttr

**功能：** 指定VConcatOp中的插入片段的源操作数。

无参数。

### MultiBufferAttr

**功能：** 定义多缓冲属性。

无参数。

### PadModeAttr

**功能：** 定义填充模式属性。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| padmode | `::mlir::hivm::PadMode` | 类型为PadMode的枚举 |

### ParallelLoopAttr

**功能：** 标记可并行执行的循环。

无参数。

### PipeAttr

**功能：** 定义操作所属的pipe。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| pipe | `::mlir::hivm::PIPE` | 类型为PIPE的枚举 |

### ReduceOpAttr

**功能：** 定义规约操作的算术类型。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| reduce_op | `::mlir::hivm::ReduceOperation` | 类型为ReduceOperation的枚举 |

### RoundModeAttr

**功能：** 定义VCastOp的舍入模式。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::RoundMode` | 类型为RoundMode的枚举 |

### StorageAlignedAttr

**功能：** 标记模块或函数内所有操作已对齐。

无参数。

### StrideAlignDimsAttr

**功能：** 定义步长对齐维度信息。

无参数。

### StrideAlignValueInByteAttr

**功能：** 定义按字节对齐的步长大小信息。

无参数。

### HIVMSubBlockMappingAttr

**功能：** 定义混合函数中子块映射关系。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| sub_block | `::mlir::hivm::MappingId` | 类型为MappingId的枚举 |

### SyncBlockInstrModeAttr

**功能：** 定义同步块指令模式。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| sync_instr_mode | `::mlir::hivm::SyncBlockInstrMode` | 类型为SyncBlockInstrMode的枚举 |

### SyncBlockModeAttr

**功能：** 定义同步块模式。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| sync_mode | `::mlir::hivm::SyncBlockMode` | 类型为SyncBlockMode的枚举 |

### TCoreTypeAttr

**功能：** 定义操作的核心类型。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| tcoretype | `::mlir::hivm::TCoreType` | 类型为TCoreType的枚举 |

### TCoreTypeMarkerAttr

**功能：** 定义操作的核心类型标记。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| tcoretype | `::mlir::hivm::TCoreType` | 类型为TCoreType的枚举 |

### TFuncCoreTypeAttr

**功能：** 定义函数的核心类型。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| funcCoreType | `::mlir::hivm::TFuncCoreType` | 类型为TFuncCoreType的枚举 |

### TModuleCoreTypeAttr

**功能：** 定义模块的核心类型，根据模块内函数类型自动推断为AIC、AIV或MIX。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| moduleCoreType | `::mlir::hivm::TModuleCoreType` | 类型为TModuleCoreType的枚举 |

### TPartOfMixAttr

**功能：** 标记函数是混合内核的一部分。

无参数。

### TypeFnAttr

**功能：** 定义VCastOp的转换类型。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::TypeFn` | 类型为TypeFn的枚举 |

### UnitFlagAttr

**功能：** 定义单元标志同步模式。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| unit_flag | `::mlir::hivm::UNIT_FLAG` | 类型为UNIT_FLAG的枚举 |

### UnlikelyConditionAttr

**功能：** 标记条件分支大概率不成立。

无参数。

### VFModeAttr

**功能：** 定义向量单元运行模式。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::VFMode` | 类型为VFMode的枚举 |

## 枚举

### AddressSpace

**功能：** 定义HIVM地址空间。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| Zero | 0 | zero |
| GM | 1 | gm |
| L1 | 2 | cbuf |
| L0A | 3 | ca |
| L0B | 4 | cb |
| L0C | 5 | cc |
| UB | 6 | ub |

### AlignKind

**功能：** 定义对齐方式信息。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| ALIGN | 0 | align |
| UNALIGNED | 1 | unaligned |
| UNKNOWN | 2 | unknown |

### AtomicKind

**功能：** 定义StoreOp的原子操作类型。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| NONE | 0 | none |
| ADD | 1 | add |
| MAX | 2 | max |
| MIN | 3 | min |
| AND | 4 | and |
| OR | 5 | or |
| XOR | 6 | xor |
| CAS | 7 | or |
| XCHG | 8 | xor |
| UMAX | 9 | umax |
| UMIN | 10 | umin |

### AxisKind

**功能：** 定义轴类型信息。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| FIRST | 0 | first |
| MIDDLE | 1 | middle |
| LAST | 2 | last |

### CompareMode

**功能：** 定义VCmpOp的比较模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| EQ | 0 | eq |
| NE | 1 | ne |
| LT | 2 | lt |
| GT | 3 | gt |
| GE | 4 | ge |
| LE | 5 | le |

### DCCIMode

**功能：** 定义DCCI模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| SINGLE_CACHE_LINE | 0 | single_cache_line |
| ALL_CACHE_LINES | 1 | all_cache_lines |

### DataCacheKind

**功能：** 定义数据缓存类型。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| ALL | 0 | all |
| UB | 1 | ub |
| OUT | 2 | out |
| ATOMIC | 3 | atomic |

### DataLayout

**功能：** 定义HIVM数据布局类型。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| DOTA_ND | 1 | dotA_ND |
| DOTB_ND | 2 | dotB_ND |
| DOTC_ND | 3 | dotC_ND |
| nZ | 4 | nZ |
| zN | 5 | zN |
| ND | 6 | ND |

### DeinterleaveMode

**功能：** 定义解交织索引模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| CHANNEL_0 | 0 | CHANNEL_0 |
| CHANNEL_1 | 1 | CHANNEL_1 |
| ALL_CHANNELS | 999 | ALL_CHANNELS |

### DescaleMode

**功能：** 定义MatmulOp的反量化模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| DescaleNull | 0 | DescaleNull |
| DescalePerChannel | 1 | DescalePerChannel |
| DescalePerTensor | 2 | DescalePerTensor |

### EVENT

**功能：** 定义同步事件ID。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| EVENT_ID0 | 0 | EVENT_ID0 |
| EVENT_ID1 | 1 | EVENT_ID1 |
| EVENT_ID2 | 2 | EVENT_ID2 |
| EVENT_ID3 | 3 | EVENT_ID3 |
| EVENT_ID4 | 4 | EVENT_ID4 |
| EVENT_ID5 | 5 | EVENT_ID5 |
| EVENT_ID6 | 6 | EVENT_ID6 |
| EVENT_ID7 | 7 | EVENT_ID7 |

### FixpipePreQuantMode

**功能：** 定义Fixpipe前级量化模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| NO_QUANT | 0 | NO_QUANT |
| S322I8 | 9 | S322I8 |
| F322F16 | 1 | F322F16 |
| F322BF16 | 16 | F322BF16 |

### FixpipePreReluMode

**功能：** 定义Fixpipe前级ReLU模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| NO_RELU | 0 | NO_RELU |
| NORMAL_RELU | 1 | NORMAL_RELU |
| LEAKY_RELU | 2 | LEAKY_RELU |
| P_RELU | 3 | P_RELU |

### IteratorType

**功能：** 定义结构化的迭代器类型。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| kParallel | 0 | parallel |
| kBroadcast | 1 | broadcast |
| kTranspose | 2 | transpose |
| kReduction | 3 | reduction |
| kInterleave | 4 | interleave |
| kDeinterleave | 5 | deinterleave |
| kInverse | 6 | inverse |
| kPad | 7 | pad |
| kConcat | 8 | concat |
| kGather | 9 | gather |
| kCumulative | 10 | cumulative |
| kOpaque | 99 | opaque |

### MatmulBiasMode

**功能：** 定义局部MatmulOp的偏置模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| NoBias | 0 | NoBias |
| PerChannelAdd | 1 | PerChannelAdd |
| PerChannelAddWithSplitK | 2 | PerChannelAddWithSplitK |
| ElementwiseCrossLoopAdd | 4 | ElementwiseCrossLoopAdd |
| ElementwiseAdd | 3 | ElementwiseAdd |

### MemPlanMode

**功能：** 定义内存规划模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| LOCAL_MEM_PLAN | 0 | LOCAL_MEM_PLAN |
| GLOBAL_WORKSPACE_PLAN | 1 | GLOBAL_WORKSPACE_PLAN |

### PadMode

**功能：** 定义LoadOp的填充模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| PadNull | 0 | PadNull |
| PadFirstElem | 1 | PadFirstElem |
| PadValue | 2 | PadValue |

### PIPE

**功能：** 定义HIVM操作所属的pipe。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| PIPE_S | 0 | PIPE_S |
| PIPE_V | 1 | PIPE_V |
| PIPE_M | 2 | PIPE_M |
| PIPE_MTE1 | 3 | PIPE_MTE1 |
| PIPE_MTE2 | 4 | PIPE_MTE2 |
| PIPE_MTE3 | 5 | PIPE_MTE3 |
| PIPE_ALL | 6 | PIPE_ALL |
| PIPE_MTE4 | 7 | PIPE_MTE4 |
| PIPE_MTE5 | 8 | PIPE_MTE5 |
| PIPE_V2 | 9 | PIPE_V2 |
| PIPE_FIX | 10 | PIPE_FIX |
| VIRTUAL_PIPE_MTE2_L1A | 11 | VIRTUAL_PIPE_MTE2_L1A |
| VIRTUAL_PIPE_MTE2_L1B | 12 | VIRTUAL_PIPE_MTE2_L1B |
| PIPE_NUM | 13 | PIPE_NUM |
| PIPE_UNASSIGNED | 99 | PIPE_UNASSIGNED |

### ReduceOperation

**功能：** 定义VReduceOp的规约操作类型。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| sum | 1 | sum |
| prod | 2 | prod |
| max | 3 | max |
| min | 4 | min |
| max_with_index_left | 5 | max_with_index_left |
| max_with_index_right | 6 | max_with_index_right |
| min_with_index_left | 7 | min_with_index_left |
| min_with_index_right | 8 | min_with_index_right |
| any | 9 | any |
| all | 10 | all |
| xori | 11 | xori |
| ori | 12 | ori |
| andi | 13 | andi |
| none | 0 | none |

### RoundMode

**功能：** 定义VCastOp的舍入模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| RINT | 0 | rint |
| ROUND | 1 | round |
| FLOOR | 2 | floor |
| CEIL | 3 | ceil |
| TRUNC | 4 | trunc |
| ODD | 5 | odd |
| TRUNCWITHOVERFLOW | 6 | truncwithoverflow |

### SyncBlockInstrMode

**功能：** 定义同步块指令模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| INTER_BLOCK_SYNCHRONIZATION | 0 | INTER_BLOCK_SYNCHRONIZATION |
| INTER_SUBBLOCK_SYNCHRONIZATION | 1 | INTER_SUBBLOCK_SYNCHRONIZATION |
| INTRA_BLOCK_SYNCHRONIZATION | 2 | INTRA_BLOCK_SYNCHRONIZATION |

### SyncBlockMode

**功能：** 定义同步块模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| ALL_CUBE | 0 | ALL_CUBE |
| ALL_VECTOR | 1 | ALL_VECTOR |
| ALL_SUB_VECTOR | 2 | ALL_SUB_VECTOR |
| BARRIER_CUBE | 3 | BARRIER_CUBE |
| BARRIER_VECTOR | 4 | BARRIER_VECTOR |
| ALL | 5 | ALL |

### TCoreType

**功能：** 定义操作的核心类型。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| CUBE | 1 | CUBE |
| VECTOR | 2 | VECTOR |
| CUBE_OR_VECTOR | 3 | CUBE_OR_VECTOR |
| CUBE_AND_VECTOR | 4 | CUBE_AND_VECTOR |

### TFuncCoreType

**功能：** 定义函数的核心类型。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| AIC | 1 | AIC |
| AIV | 2 | AIV |
| MIX | 3 | MIX |
| AIC_OR_AIV | 4 | AIC_OR_AIV |

### TModuleCoreType

**功能：** 定义模块的核心类型。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| AIC | 1 | AIC |
| AIV | 2 | AIV |
| MIX | 3 | MIX |

### TypeFn

**功能：** 定义VCastOp的转换类型。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| cast_signed | 0 | cast_signed |
| cast_unsigned | 1 | cast_unsigned |
| bitcast | 2 | bitcast |

### UNIT_FLAG

**功能：** 定义单元标志同步模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| DISABLED | 0 | DISABLED |
| RESERVED | 1 | RESERVED |
| ENABLED_WITHOUT_UPDATE | 2 | ENABLED_WITHOUT_UPDATE |
| ENABLED_WITH_UPDATE | 3 | ENABLED_WITH_UPDATE |
| ENABLED_ONLY_LAST_ITER | 4 | ENABLED_ONLY_LAST_ITER |
| ENABLED_ONLY_FIRST_ITER | 5 | ENABLED_ONLY_FIRST_ITER |
| ENABLED_ONLY_FIRST_AND_LAST_ITERS | 6 | ENABLED_ONLY_FIRST_AND_LAST_ITERS |

### VFMode

**功能：** 定义向量单元运行模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| SIMD | 0 | SIMD |
| SIMT | 1 | SIMT |
| MIX | 2 | MIX |

### MappingId

**功能：** 定义循环映射标识。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| DimX | 0 | x |
