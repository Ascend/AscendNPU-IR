# hivm方言

HIVM（Hybrid Intelligence Virtual Machine）方言，定义了用于异构计算的核心操作、属性与类型系统，涵盖数据搬运、矩阵运算、向量运算及同步机制。

## 操作定义

### hivm.hir.anchor (hivm::AnchorOp)

**功能**：用于记录IR位置信息的标记操作。

**语法**：

```mlir
operation ::= `hivm.hir.anchor` attr-dict
```

Anchor操作作为标记，用于记录IR中的位置信息，携带唯一且单调递增的`id`。

先将其插入IR，待其他pass修改IR后再检查其位置，用于分析IR发生的变化。

Anchor对内部资源声明写效应，因此规范化、CSE、DCE、LICM无法移除或重排它们。

**特性**：`MemoryEffectOpInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `id` | `::mlir::IntegerAttr` | 64位无符号整数属性 |
| `tcore_type` | `::mlir::hivm::TCoreTypeAttr` | HIVM操作核类型属性 |

### hivm.hir.atomic_cas (hivm::AtomicCasOp)

**功能**：执行原子比较并交换（CAS）操作。

**语法**：

```mlir
operation ::= `hivm.hir.atomic_cas` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`->` type($result_tensor)^)?
```

**示例**：

```mlir
hivm.hir.atomic_cas ins(%src0, %src1 : memref<?xf32>, memref<?xf32>) outs(%dst : memref<?xf32>)
%result = hivm.hir.atomic_cas ins(%src0, %src1 : tensor<?xf32>, tensor<?xf32>) outs(%dst : tensor<?xf32>) -> tensor<?xf32>
```

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result_tensor` | 类型为Tensor或Memref |

### hivm.hir.atomic_rmw (hivm::AtomicRMWOp)

**功能**：执行原子读-修改-写操作。

**语法**：

```mlir
operation ::= `hivm.hir.atomic_rmw` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `atomic_kind` `=` $atomic_kind
              (`->` type($result_tensor)^)?
```

**示例**：

```mlir
hivm.hir.atomic_rmw ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) atomic_kind = <add>
%result = hivm.hir.atomic_rmw ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) atomic_kind = <or> -> tensor<?xf32>
```

**特性**：`DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `InferCoreTypeInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 任意类型 |
| `dst` | 类型为Tensor或Memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result_tensor` | 类型为Tensor或Memref |

### hivm.hir.atomic_xchg (hivm::AtomicXchgOp)

**功能**：执行原子交换操作。

**语法**：

```mlir
operation ::= `hivm.hir.atomic_xchg` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`mask` `(` $mask^ `:` type($mask) `)`)?
              (`->` type($result_tensor)^)?
```

**示例**：

```mlir
hivm.hir.atomic_xchg ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>)
%result = hivm.hir.atomic_cas ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) -> tensor<?xf32>
```

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 任意类型 |
| `dst` | 类型为Tensor或Memref |
| `mask` | 类型为Tensor或Memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result_tensor` | 类型为Tensor或Memref |

### hivm.hir.batchMmadL1 (hivm::BatchMmadL1Op)

**功能**：从L1内存层级执行批处理矩阵乘加操作，支持批处理维度。

**语法**：

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

**示例**：

```mlir
hivm.hir.batchMmadL1 ins(%A, %B, %init, %m, %k, %n : memref<2x32x64xf16>, memref<2x64x32xf16>, i1, index, index, index) outs(%C : memref<2x32x32xf16>)
```

**特性**：`AttrSizedOperandSegments`, `CubeCoreTypeTrait`, `MacroOpPipeTrait<PIPE::PIPE_MTE1, PIPE::PIPE_M>`, `MacroOpTrait`

**操作数**：

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

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result_tensors` | 可变参数，任意类型的ranked tensor |

### hivm.hir.bitcast (hivm::BitcastOp)

**功能**：在不改变底层位表示的前提下，重新解释有形状值的位模式，进行元素类型转换。

**语法**：

```mlir
operation ::= `hivm.hir.bitcast` $src `:` type($src) `->` type($result) attr-dict
```

**示例**：

```mlir
%res = hivm.hir.bitcast %src : memref<4xf32> -> memref<4xi32>
```

**特性**：`AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultShape`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 任意类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 任意类型 |

### hivm.hir.Conv1dL1 (hivm::Conv1DL1Op)

**功能**：输入来自L1内存层级的一维卷积算子。

**语法**：

```mlir
operation ::= `hivm.hir.Conv1dL1` attr-dict `ins` `(`
              $input `,` $weight
              `,` $init_condition
              (`,` $bias^)?
              `:` type($input) `,` type($weight)
              `,` type($init_condition)
              (`,` type($bias)^)?
              `)`
              `outs` `(` $init `:` type($init) `)`
              (`sync_related_args`
              `(` $sync_related_args^ `:` type($sync_related_args) `)`
              )?
              (`->` type($result_tensors)^)?
```

**特性**：`DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpLayoutInterface`, `OpPipeInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `stride` | `::mlir::IntegerAttr` | 32位无符号整数属性 |
| `padding` | `::mlir::Attribute` | 任意属性 |
| `dilation` | `::mlir::IntegerAttr` | 32位无符号整数属性 |
| `groups` | `::mlir::IntegerAttr` | 32位无符号整数属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `input` | 类型为Tensor或Memref |
| `weight` | 类型为Tensor或Memref |
| `bias` | 类型为Tensor或Memref |
| `init` | 类型为Tensor或Memref |
| `init_condition` | 1位无符号整数 |
| `sync_related_args` | 可变参数，64位无符号整数 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result_tensors` | 可变参数，任意类型的ranked tensor |

### hivm.hir.Conv2dL1 (hivm::Conv2DL1Op)

**功能**：输入来自L1内存层级的二维卷积算子。

**语法**：

```mlir
operation ::= `hivm.hir.Conv2dL1` attr-dict `ins` `(`
              $input `,` $weight
              `,` $init_condition
              (`,` $bias^)?
              `:` type($input) `,` type($weight)
              `,` type($init_condition)
              (`,` type($bias)^)?
              `)`
              `outs` `(` $init `:` type($init) `)`
              (`sync_related_args`
              `(` $sync_related_args^ `:` type($sync_related_args) `)`
              )?
              (`->` type($result_tensors)^)?
```

**特性**：`DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpLayoutInterface`, `OpPipeInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `stride` | `::mlir::Attribute` | 任意属性 |
| `padding` | `::mlir::Attribute` | 任意属性 |
| `dilation` | `::mlir::Attribute` | 任意属性 |
| `groups` | `::mlir::IntegerAttr` | 32位无符号整数属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `input` | 类型为Tensor或Memref |
| `weight` | 类型为Tensor或Memref |
| `bias` | 类型为Tensor或Memref |
| `init` | 类型为Tensor或Memref |
| `init_condition` | 1位无符号整数 |
| `sync_related_args` | 可变参数，64位无符号整数 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result_tensors` | 可变参数，任意类型的ranked tensor |

### hivm.hir.Conv3dL1 (hivm::Conv3DL1Op)

**功能**：输入来自L1内存层级的三维卷积算子。

**语法**：

```mlir
operation ::= `hivm.hir.Conv3dL1` attr-dict `ins` `(`
              $input `,` $weight
              `,` $init_condition
              (`,` $bias^)?
              `:` type($input) `,` type($weight)
              `,` type($init_condition)
              (`,` type($bias)^)?
              `)`
              `outs` `(` $init `:` type($init) `)`
              (`sync_related_args`
              `(` $sync_related_args^ `:` type($sync_related_args) `)`
              )?
              (`->` type($result_tensors)^)?
```

**特性**：`DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `stride` | `::mlir::Attribute` | 任意属性 |
| `padding` | `::mlir::Attribute` | 任意属性 |
| `dilation` | `::mlir::Attribute` | 任意属性 |
| `groups` | `::mlir::IntegerAttr` | 32位无符号整数属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `input` | 类型为Tensor或Memref |
| `weight` | 类型为Tensor或Memref |
| `bias` | 类型为Tensor或Memref |
| `init` | 类型为Tensor或Memref |
| `init_condition` | 1位无符号整数 |
| `sync_related_args` | 可变参数，64位无符号整数 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result_tensors` | 可变参数，任意类型的ranked tensor |

### hivm.hir.convert_layout (hivm::ConvertLayoutOp)

**功能**：将memref从一种数据布局转换为另一种布局，不复制或修改数据。

**语法**：

```mlir
operation ::= `hivm.hir.convert_layout` $source attr-dict `:` functional-type(operands, results)
```

**示例**：

```mlir
%res = hivm.hir.convert_layout %src {srcLayout = #hivm.data_layout<ND>, dstLayout = #hivm.data_layout<zN>} : memref<32x32xf16> -> memref<32x32xf16>
```

**特性**：`AlwaysSpeculatableImplTrait`, `SameOperandsAndResultElementType`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `source` | 任意类型值的ranked或unranked memref |
| `output_shape` | 可变参数，index类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 任意类型值的ranked或unranked memref |

### hivm.hir.copy (hivm::CopyOp)

**功能**：在本地内存层级之间拷贝数据，支持非连续数据的重关联重塑。

**语法**：

```mlir
operation ::= `hivm.hir.copy` `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              attr-dict
              (`pad_mode` `=` $pad_mode^)?
              (`pad_value` `=` $pad_value^ `:` type($pad_value))?
              (`collapse_reassociation` `=` $collapse_reassociation^)?
              (`->` type($result_tensor)^)?
```

**示例**：

```mlir
hivm.hir.copy ins(%src : memref<16x16xf16, #hivm.address_space<ub>>) outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
```

### 非连续重关联重塑

`hivm.hir.copy`还支持将非连续数据拷贝到连续存储，以及反向拷贝，可视为数据的"展开"或"折叠"。`collapse_reassociation`属性用于指定哪些轴被折叠在一起。例如：

```mlir
hivm.hir.copy ins(%src : memref<32x4xbf16, strided<[16, 1]>>) outs(%dst : memref<32x4xbf16, strided<[4, 1]>>)
  collapse_reassociation = [[0, 1]]
```

表示第0轴和第1轴被连续地折叠在一起。

**特性**：`AlwaysSpeculatableImplTrait`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |
| `pad_value` | 任意类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result_tensor` | 任意类型的ranked tensor |

### hivm.hir.create_sync_block_lock (hivm::CreateSyncBlockLockOp)

**功能**：分配一块锁内存区域，用于确保锁与解锁之间的代码在块间按顺序执行。

**语法**：

```mlir
operation ::= `hivm.hir.create_sync_block_lock` (`from` $lockArg^)?
              attr-dict `:` (`from` type($lockArg)^ `to`)? type($memref)
```

**示例**：

```mlir
hivm.hir.create_sync_block_lock() : memref<1xi64>
hivm.hir.create_sync_block_lock() from %arg : from memref<?xi8> to memref<1xi64>
```

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `lockArg` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `memref` | 任意类型值的memref |

### hivm.hir.custom (hivm::CustomOp)

**功能**：通用自定义操作接口，供用户编写内置操作无法满足或性能非最优的私有实现。

**语法**：

```mlir
operation ::= `hivm.hir.custom` $name attr-dict `ins` `(` $inputs `:` type($inputs) `)` `outs` `(` $outputs `:` type($outputs) `)`
```

**示例**：

```mlir
hivm.hir.custom "__builtin_gather_load" ins(%src : memref<256xf32>) outs(%dst : memref<256xf32>)
```

**特性**：`AttrSizedOperandSegments`, `SinglePipeOpTrait`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `inputs` | 可变参数，任意类型 |
| `outputs` | 可变参数，任意类型 |
| `temp_buffers` | 可变参数，任意类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `results` | 可变参数，任意类型 |

### hivm.hir.custom_macro (hivm::CustomMacroOp)

**功能**：通用自定义操作接口，供用户编写自己的自定义实现。适用场景：现有操作无法满足所需功能；现有操作虽能满足功能但整体性能非最优；需要私有操作。

自定义操作的通用接口，其中：

- `name`：唯一操作名。注意：内置保留名通常以"__builtin"开头，编译器会将这些内置操作链接到bishengir-compile自带的模板库；对于普通名称，用户需要指定实现位置、编译命令及所有其他必要信息。
- `inputs`：输入参数。
- `outputs`：输出结果，即指定的"init"操作数，作为操作结果的初始值或操作结果的写入位置。

为快速扩展，自定义操作（custom op）通过属性获取所需信息：

- `CoreType`：在哪个核类型上执行，参见`TCoreTypeAttr`。
- `VFMode`：向量单元的运行模式，参见`VFModeAttr`。`requiresVFMode()`为true时必需。
- `InPipe`：在哪个输入流水线上执行，参见`PipeAttr`。
- `OutPipe`：在哪个输出流水线上执行，参见`PipeAttr`。
- `sync_event_slots`：可选的同步槽`(set_pipe, wait_pipe, macro_sync)`。`macro_sync`为`internal`时（默认），宏仅需GraphSyncSolver分配的事件ID，不在边界注入set_flag、wait_flag；为`wait`时kernel在该槽上调用wait_flag；为`set`时调用set_flag。`wait`或`set`时`set_pipe`和`wait_pipe`必需，`internal`时可选。可选的event字段固定一个静态事件ID。
- `sync_related_args`：由GraphSyncSolver填充的i64操作数，每个`sync_event_slots`条目对应一个。

**特性**：`DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `InferCoreTypeInterface`, `MemoryEffectOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `name` | `::mlir::StringAttr` | 字符串属性 |
| `no_side_effect` | `::mlir::UnitAttr` | unit属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `inputs` | 可变参数，任意类型 |
| `outputs` | 可变参数，任意类型 |
| `temp_buffers` | 可变参数，任意类型 |
| `sync_related_args` | 可变参数，64位无符号整数 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `results` | 可变参数，任意类型 |

### hivm.hir.dcci (hivm::DCCIOp)

**功能**：清理（写回）并使一个缓存行或整个数据缓存失效。

**语法**：

```mlir
operation ::= `hivm.hir.dcci` attr-dict `(` $mode `,` $dataCacheKind (`,` $ptr^ `:` type($ptr))? `)`
```

**示例**：

```mlir
hivm.hir.dcci(#hivm.DCCIMode<single_cache_line>, #hivm.DataCacheKind<ub>, %ptr : memref<1024xf32>)
```

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `ptr` | 任意类型值的memref |

### hivm.hir.debug (hivm::DebugOp)

**功能**：设备端调试操作，用于输出调试信息。

**语法**：

```mlir
operation ::= `hivm.hir.debug` attr-dict $arg `:` type($arg)
```

**示例**：

```mlir
hivm.hir.debug %arg {debugtype = "print", prefix = "value"} : f32
```

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `arg` | 整数、浮点数、Tensor或Memref |

### hivm.hir.embedding_gather (hivm::EmbeddingGatherOp)

**功能**：使用gather语义的embedding查找操作。

**语法**：

```mlir
operation ::= `hivm.hir.embedding_gather` `ins` `(` $src `:` type($src) `,` $index `:` type($index) `,`
              $bound `:` type($bound) `,`
              `[` $offsets `:` type($offsets) `]` `,`
              `[` $numels `:` type($numels) `]` `)`
              `outs` `(` $dst `:` type($dst) `)`
              attr-dict
              (`->` type($result)^)?
```

执行embedding查找（又称embedding表查找或embedding gather）。输入为全局内存中的2D embedding表和1D或2D索引tensor，按索引gather对应的embedding向量，生成2D或3D输出tensor。

语义等价：`result[b][i][d] = src[index[b][i]][d]`（带边界检查）。

其中：`b`为batch维度，`i`为序列（位置）维度，`d`为embedding特征维度。

**特性**：`DestinationStyleOpInterface`, `HIVMCoreTypeInterface`, `MemoryEffectOpInterface`, `OpPipeInterface`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 任意类型值的memref |
| `index` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |
| `bound` | 32位或64位无符号整数 |
| `offsets` | 可变参数，32位或64位无符号整数 |
| `numels` | 可变参数，32位或64位无符号整数 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，类型为Tensor或Memref |

### hivm.hir.finish_debug (hivm::FinishDebugOp)

**功能**：设备端调试的结束函数。

**语法**：

```mlir
operation ::= `hivm.hir.finish_debug` attr-dict
```

**示例**：

```mlir
hivm.hir.finish_debug
```

**特性**：`CubeVectorCoreTypeTrait`

### hivm.hir.fixpipe (hivm::FixpipeOp)

**功能**：从L0C到其他内存层级的数据搬移操作，支持前级量化、前级ReLU、逐元素加法、后级ReLU、后级量化及布局变换。

**语法**：

```mlir
operation ::= `hivm.hir.fixpipe` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`unit_flag` `[` $unit_flag_mode^ (`,` $unit_flag_cond^)? `]`)?
              (`->` type($result_tensor)^)?
```

**示例**：

```mlir
hivm.hir.fixpipe ins(%src : memref<16x16xf16, #hivm.address_space<l0c>>) outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>) {pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>}
```

### 属性

#### dma_mode

HIVM从L0C到目标的数据搬移模式，有三个取值：NZ2DN、NZ2ND和NZ2NZ（normal）。

#### dual_dst_mode

HIVM双目标模式控制。仅当启用nz2nd或normal数据搬移模式、且正在执行从L0C到UB的数据搬移时，才能启用双目标模式。仅支持Ascend 950PR&950DT系列产品。

#### c0_pad_en

为目标分形布局启用硬件C0 padding（配合int8通道合并使用，否则奇数个C0块会留下未合并的尾部）。默认为true（启用padding）。

**特性**：`AlwaysSpeculatableImplTrait`, `CubeCoreTypeTrait`, `OpPipeTrait<PIPE::PIPE_FIX>`, `SinglePipeOpTrait`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 任意类型值的shaped类型 |
| `dst` | 任意类型值的shaped类型 |
| `unit_flag_cond` | 1位无符号整数 |
| `quant_scale` | 浮点类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result_tensor` | 任意类型的ranked tensor |

### hivm.hir.free_lock_var (hivm::FreeLockVarOp)

**功能**：当控制流跳过sync_block解锁时释放锁变量。

**语法**：

```mlir
operation ::= `hivm.hir.free_lock_var` attr-dict `lock_var` `(` $lock_var `:` type($lock_var) `)`
```

lower到模板`free_lock_var`，执行一对sync_block_lock与sync_block_unlock，使lock_var按与正常操作相同的屏障和内存语义推进。当控制流可能跳过受保护的加锁、解锁区域时，在return前插入该操作。

**示例**：

```mlir
  hivm.hir.free_lock_var lock_var(%lock : memref<1xi64>)
```

**特性**：`HIVMCoreTypeInterface`, `InferCoreTypeInterface`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `lock_var` | 64位无符号整数值的1D memref |

### hivm.hir.gather_load (hivm::GatherLoadOp)

**功能**：执行稀疏内存加载，支持可选的mask与回退值。

**语法**：

```mlir
operation ::= `hivm.hir.gather_load` `ins` `(` $base `:` type($base) `,` $indices `:` type($indices) `,`
              $burst_len `:` type($burst_len)
              (`,` $mask `:` type($mask)^)? (`,` $other `:` type($other)^)? `)`
              `outs` `(` $dst `:` type($dst) `)`
              attr-dict
              (`->` type($result)^)?
```

该操作输入源内存buffer和偏移tensor，按指定偏移位置从源中gather元素生成输出tensor，支持mask并为被mask掉的位置提供回退值。参数含义如下：

- `base`：待加载数据的基地址。
- `indices`：待加载数据的索引。
- `burst_len`：表示索引值在最内层维度上的连续性。
- `mask`：（可选）加载mask，控制哪些位置实际加载、哪些位置不加载。除block pointer情形外均有意义。
- `other`：（可选）对应mask元素为false时的默认（回退）值。除block pointer情形外均有意义。
- `dst`：指定输出形状和类型的目标buffer或tensor。
- `cache`：（可选）枚举hint，控制NPU L1如何缓存从全局内存加载的数据。
- `evict`：（可选）枚举hint，控制缓存空间满时数据被移出NPU缓存的优先级。
- `isVolatile`：（可选）标记为volatile的load操作不允许以任何方式被优化。

**特性**：`DestinationStyleOpInterface`, `HIVMCoreTypeInterface`, `InferTypeOpInterface`, `MemoryEffectOpInterface`, `OpPipeInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `cache` | `::mlir::hivm::CacheModifierAttr` | 缓存修改策略 |
| `evict` | `::mlir::hivm::EvictionPolicyAttr` | 共享内存的逐出策略 |
| `isVolatile` | `::mlir::BoolAttr` | 布尔属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `base` | 任意类型值的memref |
| `indices` | 32位或64位无符号整数值的ranked tensor |
| `burst_len` | 32位或64位无符号整数 |
| `mask` | 1位无符号整数值的ranked tensor |
| `other` | 任意类型的ranked tensor |
| `dst` | 类型为Tensor或Memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 任意类型的ranked tensor |

### hivm.hir.gatherT (hivm::GatherTOp)

**功能**：沿指定轴按索引tensor从输入tensor中选取元素的gather操作。

**语法**：

```mlir
operation ::= `hivm.hir.gatherT` `ins` `(` $src `:` type($src) `,`
              $index `:` type($index) `,`
              $bound `:` type($bound) `,`
              $dim `:` type($dim) `,`
              `[` $src_stride `:` type($src_stride) `]` `,`
              `[` $index_shape `:` type($index_shape) `]` `,`
              `[` $offsets `:` type($offsets) `]` `)`
              `outs` `(` $dst `:` type($dst) `)`
              attr-dict
              (`->` type($result)^)?
```

该操作输入源内存GM buffer和UB上的索引tensor，按索引位置加偏移从源中gather元素，生成UB输出tensor。支持1D-5D。

输出tensor各位置的语义：

- 1D：`dst[i] = src[index[i]]`
- 2D：`dim=0`时`dst[i][j] = src[index[i][j]][j]`；`dim=1`时`dst[i][j] = src[i][index[i][j]]`
- 3D：`dim=0`时`dst[i][j][k] = src[index[i][j][k]][j][k]`；`dim=1`时`dst[i][j][k] = src[i][index[i][j][k]][k]`；`dim=2`时`dst[i][j][k] = src[i][j][index[i][j][k]]`

其中：

- `src`：源内存GM buffer。
- `index`：指定src中位置的UB tensor。
- `dst`：指定输出形状和类型的目标UB tensor。
- `bound`：gather维度的大小。
- `dim`：gather操作的维度。
- `src_stride`：源tensor的stride。
- `index_shape`：索引tensor的shape。
- `offsets`：dst buffer中的偏移。

**特性**：`DestinationStyleOpInterface`, `HIVMCoreTypeInterface`, `MemoryEffectOpInterface`, `OpPipeInterface`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 任意类型值的memref |
| `index` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |
| `bound` | 32位或64位无符号整数 |
| `dim` | 32位或64位无符号整数 |
| `src_stride` | 可变参数，32位或64位无符号整数 |
| `index_shape` | 可变参数，32位或64位无符号整数 |
| `offsets` | 可变参数，32位或64位无符号整数 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 类型为Tensor或Memref |

### hivm.hir.get_block_idx (hivm::GetBlockIdxOp)

**功能**：获取当前设备线程用于并行化的block索引。

**语法**：

```mlir
operation ::= `hivm.hir.get_block_idx` attr-dict `->` type($result)
```

**示例**：

```mlir
%idx = hivm.hir.get_block_idx -> i64
```

**特性**：`AlwaysSpeculatableImplTrait`, `CubeVectorCoreTypeTrait`

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 64位无符号整数 |

### hivm.hir.get_block_num (hivm::GetBlockNumOp)

**功能**：获取当前设备线程用于并行化的block数量。

**语法**：

```mlir
operation ::= `hivm.hir.get_block_num` attr-dict `->` type($result)
```

**示例**：

```mlir
%num = hivm.hir.get_block_num -> i64
```

**特性**：`AlwaysSpeculatableImplTrait`, `CubeVectorCoreTypeTrait`

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 64位无符号整数 |

### hivm.hir.get_sub_block_idx (hivm::GetSubBlockIdxOp)

**功能**：获取当前设备线程用于并行化的子块索引。

**语法**：

```mlir
operation ::= `hivm.hir.get_sub_block_idx` attr-dict `->` type($result)
```

**示例**：

```mlir
%sub_idx = hivm.hir.get_sub_block_idx -> i64
```

**特性**：`AlwaysSpeculatableImplTrait`, `CubeVectorCoreTypeTrait`

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 64位无符号整数 |

### hivm.hir.get_sub_block_num (hivm::GetSubBlockNumOp)

**功能**：获取当前设备线程用于并行化的子块数量。

**语法**：

```mlir
operation ::= `hivm.hir.get_sub_block_num` attr-dict `->` type($result)
```

**示例**：

```mlir
%sub_num = hivm.hir.get_sub_block_num -> i64
```

**特性**：`AlwaysSpeculatableImplTrait`, `CubeVectorCoreTypeTrait`

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 64位无符号整数 |

### hivm.hir.get_sys_cnt (hivm::GetSysCntOp)

**功能**：获取当前设备的系统计数。

**语法**：

```mlir
operation ::= `hivm.hir.get_sys_cnt` attr-dict `->` type($result)
```

**示例**：

```mlir
%cnt = hivm.hir.get_sys_cnt -> i64
```

**特性**：`AlwaysSpeculatableImplTrait`, `CubeVectorCoreTypeTrait`

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 64位无符号整数 |

### hivm.hir.index_put (hivm::IndexPutOp)

**功能**：通过SIMT模板将值赋给目标tensor中指定位置的IndexPut操作。

**语法**：

```mlir
operation ::= `hivm.hir.index_put` `ins` `(` $dst `:` type($dst) `,`
              $index `:` type($index) `,`
              $value `:` type($value) `,`
              $scatter_dim `:` type($scatter_dim) `,`
              $bound `:` type($bound) `,`
              `[` $end_offset `:` type($end_offset) `]` `,`
              `[` $start_offset `:` type($start_offset) `]` `,`
              `[` $dst_stride `:` type($dst_stride) `]` `)`
              attr-dict
```

该操作按索引UB tensor指定的位置加偏移，沿指定的scatter维度将value UB tensor中的值写入GM上的dst buffer，使用SIMT模板。支持2D-5D。

目标tensor各位置的语义：

- 2D：`dim=0`时`dst[index[i]][offset[1] + j] = value[i][j]`
- 3D：`dim=0`时`dst[index[i]][offset[1] + j][offset[2] + k] = value[i][j][k]`；`dim=1`时`dst[offset[0] + i][index[j]][offset[2] + k] = value[i][j][k]`
- 4D：`dim=0`时`dst[index[i]][offset[1] + j][offset[2] + k][offset[3] + l] = value[i][j][k][l]`；`dim=1`时`dst[offset[0] + i][index[j]][offset[2] + k][offset[3] + l] = value[i][j][k][l]`；`dim=2`时`dst[offset[0] + i][offset[1] + j][index[k]][offset[3] + l] = value[i][j][k][l]`

其中：

- `dst`：GM上目标tensor的指针。
- `index`：UB上的索引tensor tile指针，指定目标tensor中的位置；索引不得重复，且必须包含沿指定`scatter_dim`维度在目标tensor边界内的有效索引。
- `value`：UB上的值tile指针。
- `scatter_dim`：应用索引的gather维度。
- `bound`：索引值的上界。
- `end_offset`：value tile的结束偏移，index tile的结束偏移为元组中第`dim`个值。
- `start_offset`：value tile的起始偏移，index tile的起始偏移为元组中第`dim`个值。
- `dst_stride`：目标全局tensor的stride。

**特性**：`DestinationStyleOpInterface`, `HIVMCoreTypeInterface`, `MemoryEffectOpInterface`, `OpPipeInterface`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `dst` | 任意类型值的memref |
| `index` | 类型为Tensor或Memref |
| `value` | 类型为Tensor或Memref |
| `scatter_dim` | 32位或64位无符号整数 |
| `bound` | 32位或64位无符号整数 |
| `end_offset` | 可变参数，32位或64位无符号整数 |
| `start_offset` | 可变参数，32位或64位无符号整数 |
| `dst_stride` | 可变参数，32位或64位无符号整数 |

### hivm.hir.indirect_load (hivm::IndirectLoadOp)

**功能**：执行带mask与回退值的间接内存加载。

**语法**：

```mlir
operation ::= `hivm.hir.indirect_load` `ins` `(` $src `:` type($src) `,` $offsets `:` type($offsets)
              (`,` $mask^ `:` type($mask))?
              (`,` $other^ `:` type($other))? `)`
              `outs` `(` $dst `:` type($dst) `)`
              attr-dict
              (`->` type($result)^)?
```

该操作输入源内存buffer和偏移tensor，按指定偏移位置从源中gather元素生成输出tensor，支持mask并为被mask掉的位置提供回退值。支持1D-5D。

输出tensor各位置的语义：

- 1D：`dst[i] = mask[i] ? src[offsets[i]] : other[i]`
- 2D：`dst[i][j] = mask[i][j] ? src[offsets[i][j]] : other[i][j]`
- 3D：`dst[i][j][k] = mask[i][j][k] ? src[offsets[i][j][k]] : other[i][j][k]`
- 4D：`dst[i][j][k][l] = mask[i][j][k][l] ? src[offsets[i][j][k][l]] : other[i][j][k][l]`
- 5D：`dst[i][j][k][l][m] = mask[i][j][k][l][m] ? src[offsets[i][j][k][l][m]] : other[i][j][k][l][m]`

其中：

- `src`：加载的源内存buffer。
- `offsets`：指定源buffer中位置的索引。
- `mask`：控制加载哪些元素的布尔mask。
- `other`：mask为false时使用的回退值。
- `dst`：指定输出形状和类型的目标tensor。
- `isVolatile`：默认为true。仅当分析证明该indirect load的GM地址不会被任何可能的操作写入时才可设为false。

该操作适用于稀疏数据访问模式以及带条件加载语义的gather操作。

**特性**：`DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `isVolatile` | `::mlir::BoolAttr` | 布尔属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 任意类型值的memref |
| `offsets` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |
| `mask` | 类型为Tensor或Memref |
| `other` | 类型为Tensor或Memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 类型为Tensor或Memref |

### hivm.hir.indirect_store (hivm::IndirectStoreOp)

**功能**：通过SIMT模板执行间接内存存储。

**语法**：

```mlir
operation ::= `hivm.hir.indirect_store` `ins` `(` $src `:` type($src) `,` $offsets `:` type($offsets)
              (`,` $mask^ `:` type($mask))? `)`
              `outs` `(` $dst `:` type($dst) `)`
              attr-dict
```

该操作输入UB上的源tensor和偏移tensor，通过SIMT模板按指定偏移位置将源中的元素存入目标GM buffer，支持mask以条件控制存储哪些元素。支持1D-5D。

源tensor各位置的语义：

- 1D：`if (mask[i]) dst[offsets[i]] = src[i]`
- 2D：`if (mask[i][j]) dst[offsets[i][j]] = src[i][j]`
- 3D：`if (mask[i][j][k]) dst[offsets[i][j][k]] = src[i][j][k]`
- 4D：`if (mask[i][j][k][l]) dst[offsets[i][j][k][l]] = src[i][j][k][l]`
- 5D：`if (mask[i][j][k][l][m]) dst[offsets[i][j][k][l][m]] = src[i][j][k][l][m]`

其中：

- `src`：UB上包含待存储值的源tensor。
- `offsets`：指定目标buffer中位置的索引。
- `dst`：存入GM的目标内存buffer。
- `mask`：可选的布尔mask，控制存储哪些元素。

未提供mask时，源tensor中的所有元素都会存储到目标buffer的对应偏移位置。

**特性**：`DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `dst` | 任意类型值的memref |
| `offsets` | 类型为Tensor或Memref |
| `src` | 类型为Tensor或Memref |
| `mask` | 类型为Tensor或Memref |

### hivm.hir.init_debug (hivm::InitDebugOp)

**功能**：设备端调试的初始化函数。

**语法**：

```mlir
operation ::= `hivm.hir.init_debug` attr-dict
```

**示例**：

```mlir
hivm.hir.init_debug
```

**特性**：`CubeVectorCoreTypeTrait`

### hivm.hir.l12bt (hivm::L12BTOp)

**功能**：从L1到BiasTable Buffer的数据拷贝操作。

**语法**：

```mlir
operation ::= `hivm.hir.l12bt` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `n` `=` $n
              (`->` type($result_tensor)^)?
```

从L1拷贝数据到BiasTable Buffer。

**特性**：`ConditionallySpeculatable`, `CopyOpInterface`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |
| `n` | index类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result_tensor` | 任意类型的ranked tensor |

### hivm.hir.l12l0 (hivm::L12L0Op)

**功能**：从L1到L0A或L0B Buffer的数据拷贝操作。

**语法**：

```mlir
operation ::= `hivm.hir.l12l0` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `{`
              `k_part_idx` `=` $k_part_idx `,`
              `k_part` `=` $k_part `,`
              `k_part_ceil` `=` $k_part_ceil `,`
              `k_part_loop` `=` $k_part_loop `,`
              `k_part_actual` `=` $k_part_actual `,`
              `m` `=` $m
              (`,` `is_transposed` `=` $is_transposed^)?
              `}`
              (`->` type($result_tensor)^)?
```

从L1拷贝数据到L0A或L0B buffer。目标由dst地址空间区分：

- L0A：`#hivm.address_space<ca>`
- L0B：`#hivm.address_space<cb>`

**特性**：`ConditionallySpeculatable`, `CopyOpInterface`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `k_part_idx` | `::mlir::IntegerAttr` | 64位无符号整数属性 |
| `k_part` | `::mlir::IntegerAttr` | 64位无符号整数属性 |
| `k_part_ceil` | `::mlir::IntegerAttr` | 64位无符号整数属性 |
| `k_part_loop` | `::mlir::IntegerAttr` | 64位无符号整数属性 |
| `k_part_actual` | `::mlir::IntegerAttr` | 64位无符号整数属性 |
| `m` | `::mlir::IntegerAttr` | 64位无符号整数属性 |
| `is_transposed` | `::mlir::BoolAttr` | 布尔属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result_tensor` | 任意类型的ranked tensor |

### hivm.hir.l12ub (hivm::L12UBOp)

**功能**：从L1到UB内存的数据拷贝操作，带NZ2ND转换。

**语法**：

```mlir
operation ::= `hivm.hir.l12ub` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`->` type($result_tensor)^)?
```

NZ2ND执行从L1到UB的数据搬移，并进行NZ2ND转换。

**特性**：`ConditionallySpeculatable`, `CopyOpInterface`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result_tensor` | 任意类型的ranked tensor |

### hivm.hir.load (hivm::LoadOp)

**功能**：将数据从全局内存加载到本地缓冲区，支持填充模式与隐式转置。

**语法**：

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

**示例**：

```mlir
hivm.load ins(%src : memref<16x16xf16, #hivm.address_space<gm>>) outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `OpPipeTrait<PIPE::PIPE_MTE2>`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |
| `pad_value` | 任意类型 |
| `left_padding_num` | index类型 |
| `right_padding_num` | 任意类型 |
| `init_condition` | 任意类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result_tensor` | 任意类型的ranked tensor |

### hivm.hir.load_scalar (hivm::LoadScalarOp)

**功能**：从LLVM指针地址加载标量值。

**语法**：

```mlir
operation ::= `hivm.hir.load_scalar` attr-dict $addr `:` type($addr) `->` type($result)
```

**示例**：

```mlir
%val = hivm.hir.load_scalar %addr : !llvm.ptr<f32> -> f32
```

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `addr` | LLVM指针类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 整数或浮点数 |

### hivm.hir.load_scale (hivm::LoadMXScaleOp)

**功能**：加载MX scale tensor的HIVM数据搬移操作。

**语法**：

```mlir
operation ::= `hivm.hir.load_scale` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`is_transposed` `=` $is_transposed^)?
              (`->` type($result_tensor)^)?
```

使用MX矩阵乘所需的专用数据布局，将MX scale tensor从全局内存加载到L1。

**特性**：`ConditionallySpeculatable`, `CopyOpInterface`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `is_transposed` | `::mlir::BoolAttr` | 布尔属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result_tensor` | 任意类型的ranked tensor |

### hivm.hir.local_load (hivm::LocalLoadOp)

**功能**：从unified buffer加载tensor。

**语法**：

```mlir
operation ::= `hivm.hir.local_load` `ins` `(` $addr `:` type($addr) `)`
              attr-dict
              `->` type($result)
```

该操作输入UB上的源内存buffer，从中加载数据生成输出tensor。仅用于数据从SIMD传输到SIMT的场景。

**特性**：`MemoryEffectOpInterface`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `addr` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 任意类型的ranked tensor |

### hivm.hir.local_store (hivm::LocalStoreOp)

**功能**：将tensor存储到unified buffer。

**语法**：

```mlir
operation ::= `hivm.hir.local_store` `ins` `(` $addr `:` type($addr) `,` $data `:` type($data) `)`
              attr-dict
```

该操作输入UB上的内存buffer，并将tensor存入该buffer。仅用于数据从SIMT传输到SIMD的场景。

**特性**：`MemoryEffectOpInterface`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `addr` | 任意类型值的memref |
| `data` | 任意类型的ranked tensor |

### hivm.hir.matmul (hivm::MatmulOp)

**功能**：从全局内存执行矩阵乘法操作，支持转置、偏置与反量化。

**语法**：

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

**示例**：

```mlir
hivm.hir.matmul ins(%A, %B : memref<32x64xf16>, memref<64x32xf16>) outs(%C : memref<32x32xf32>) {a_transpose, block_sizes = [16, 16, 16]}
```

**特性**：`AttrSizedOperandSegments`, `MacroOpPipeTrait<PIPE::PIPE_MTE2, PIPE::PIPE_MTE3>`, `MacroOpTrait`

**操作数**：

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

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.mix_group_matmul (hivm::MixGroupMatmulOp)

**功能**：执行分组矩阵乘法，支持按专家分配token并与后向量函数融合。

**语法**：

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

**示例**：

```mlir
hivm.hir.mix_group_matmul ins(%A, %B, %tokens : memref<128x64xf16>, memref<64x32xf16>, memref<8xi32>) outs(%C : memref<128x32xf32>)
```

**特性**：`AttrSizedOperandSegments`, `MacroOpPipeTrait<PIPE::PIPE_MTE2, PIPE::PIPE_MTE3>`, `MacroOpTrait`

**操作数**：

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

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.mix_matmul (hivm::MixMatmulOp)

**功能**：执行矩阵乘法，支持与后向量函数进行tile级融合。

**语法**：

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

**示例**：

```mlir
hivm.hir.mix_matmul ins(%A, %B : memref<32x64xf16>, memref<64x32xf16>) outs(%C : memref<32x32xf32>) {a_transpose}
```

**特性**：`AttrSizedOperandSegments`, `MacroOpPipeTrait<PIPE::PIPE_MTE2, PIPE::PIPE_MTE3>`, `MacroOpTrait`

**操作数**：

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

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.mmadL0 (hivm::MmadL0Op)

**功能**：输入来自L0内存层级的矩阵乘加操作。

**语法**：

```mlir
operation ::= `hivm.hir.mmadL0` attr-dict `ins` `(`
              $a
              `,` $b
              `,` $real_m
              `,` $real_k
              `,` $real_n
              `,` $k_direction_align
              `,` $is_with_bias
              `,` $init_condition
              `:`
              type($a)
              `,` type($b)
              `,` type($real_m)
              `,` type($real_k)
              `,` type($real_n)
              `,` type($k_direction_align)
              `,` type($is_with_bias)
              `,` type($init_condition) `)`
              `outs` `(` $c `:` type($c) `)`
              (`unit_flag_mode` `(` $unit_flag_mode^ `)` )?
              (`unit_flag_cond` `(` $unit_flag_cond^ `)` )?
              (`->` type($result_tensors)^)?
```

计算逻辑为：`C = C + A x B +（可选）bias`。

**特性**：`DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMLocalMatmulBaseOpInterface`, `HIVMLocalMatmulL0OpInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `HIVMUnitFlagEnabledInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `enable_HF32` | `::mlir::UnitAttr` | unit属性 |
| `enable_I4` | `::mlir::UnitAttr` | unit属性 |
| `unit_flag_mode` | `::mlir::ArrayAttr` | unit-flag模式数组属性 |
| `unit_flag_group_id` | `::mlir::IntegerAttr` | 64位无符号整数属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `a` | 类型为Tensor或Memref |
| `b` | 类型为Tensor或Memref |
| `real_m` | index类型 |
| `real_k` | index类型 |
| `real_n` | index类型 |
| `c` | 类型为Tensor或Memref |
| `k_direction_align` | 1位无符号整数 |
| `is_with_bias` | 1位无符号整数 |
| `init_condition` | 1位无符号整数 |
| `unit_flag_cond` | 可变参数，1位无符号整数 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result_tensors` | 可变参数，任意类型的ranked tensor |

### hivm.hir.mmadL1 (hivm::MmadL1Op)

**功能**：从L1内存层级执行矩阵乘加操作。

**语法**：

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

**示例**：

```mlir
hivm.hir.mmadL1 ins(%A, %B, %init, %m, %k, %n : memref<32x64xf16>, memref<64x32xf16>, i1, index, index, index) outs(%C : memref<32x32xf16>)
```

**特性**：`AttrSizedOperandSegments`, `CubeCoreTypeTrait`, `MacroOpPipeTrait<PIPE::PIPE_MTE1, PIPE::PIPE_M>`, `MacroOpTrait`

**操作数**：

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

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result_tensors` | 可变参数，任意类型的ranked tensor |

### hivm.hir.mmadmxL1 (hivm::MmadMxL1Op)

**功能**：输入来自L1内存层级的矩阵乘加操作。

**语法**：

```mlir
operation ::= `hivm.hir.mmadmxL1` attr-dict
              (`a_transpose` $a_transpose^)?
              (`b_transpose` $b_transpose^)?
              `ins` `(`
              $a
              `,` $b
              `,` $scaleA
              `,` $scaleB
              `,` $init_condition
              `,` $real_m
              `,` $real_k
              `,` $real_n
              (`,` $per_channel_bias^)?
              `:`
              type($a)
              `,` type($b)
              `,` type($scaleA)
              `,` type($scaleB)
              `,` type($init_condition)
              `,` type($real_m)
              `,` type($real_k)
              `,` type($real_n)
              (`,` type($per_channel_bias)^)?
              `)`
              `outs` `(` $c `:` type($c) `)`
              (`sync_related_args` `(` $sync_related_args^ `:` type($sync_related_args) `)`)?
              (`->` type($result_tensors)^)?
```

计算逻辑为：`C = C + MX(A, ScaleA) x MX(B, ScaleB) +（可选）channel_bias`。

注意：A、B、C矩阵的rank必须为2。

**特性**：`DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMLocalMatmulBaseOpInterface`, `HIVMLocalMatmulL1OpInterface`, `HIVMLocalMatmulLikeOpInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`, `OpWithLayoutInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `lhsFormat` | `::mlir::IntegerAttr` | 允许的32位无符号整数值：1、2、3。枚举值：`fp8_e5m2_t`（`FP8E5M2_T`）、`fp8_e4m3_t`（`FP8E4M3_T`）、`fp4_e2m1_t`（`FP4E2M1_T`） |
| `rhsFormat` | `::mlir::IntegerAttr` | 允许的32位无符号整数值：1、2、3。枚举值：`fp8_e5m2_t`（`FP8E5M2_T`）、`fp8_e4m3_t`（`FP8E4M3_T`）、`fp4_e2m1_t`（`FP4E2M1_T`） |
| `a_transpose` | `::mlir::UnitAttr` | unit属性 |
| `b_transpose` | `::mlir::UnitAttr` | unit属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `a` | f8E4M3FN、f8E5M2、8位无符号整数或8位signless整数值的shaped类型 |
| `b` | f8E4M3FN、f8E5M2、8位无符号整数或8位signless整数值的shaped类型 |
| `scaleA` | 8位无符号整数或8位signless整数值的shaped类型 |
| `scaleB` | 8位无符号整数或8位signless整数值的shaped类型 |
| `init_condition` | 1位无符号整数 |
| `real_m` | index类型 |
| `real_k` | index类型 |
| `real_n` | index类型 |
| `c` | 浮点值的shaped类型 |
| `sync_related_args` | 可变参数，64位无符号整数 |
| `per_channel_bias` | 类型为Tensor或Memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result_tensors` | 可变参数，浮点值的shaped类型 |

### hivm.hir.multi_buffer_counter (hivm::MultiBufferCounterOp)

**功能**：具现化的多buffer循环计数器锚点。

**语法**：

```mlir
operation ::= `hivm.hir.multi_buffer_counter` attr-dict `->` type($counter)
```

该操作为`scf.for`或`scf.while`循环锚定逐迭代的多buffer计数器，由MultiBufferLoopAdapter引入，供所有需要槽位轮换的使用者共享。后续lowering pass会将其重写为函数作用域的`memref.alloca`计数器，外加循环体头部的load和循环体尾部的递增、store。

**特性**：`InferTypeOpInterface`

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `counter` | 64位无符号整数 |

### hivm.hir.nchw2nc1hwc0 (hivm::NCHW2NC1HWC0Op)

**功能**：将GM中的分组NCHW数据转换为L1中的NC1HWC0数据。

**语法**：

```mlir
operation ::= `hivm.hir.nchw2nc1hwc0` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
```

将全局内存中形状为`[N, G*C, H, W]`的连续源转换为L1中形状为`[N, G*C1, H, W, C0]`的连续目标，其中`C1 = ceil(C / C0)`且`C0 * sizeof(element_type) = 32`字节。通道padding按组独立进行。

仅支持`f16`、`bf16`和`f32`元素类型。`groups`属性指定分组通道变换的组数。

**特性**：`ConditionallySpeculatable`, `CopyOpInterface`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `groups` | `::mlir::IntegerAttr` | 64位无符号整数属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 任意类型值的memref |
| `dst` | 任意类型值的memref |

### hivm.hir.nd2nz (hivm::ND2NZOp)

**功能**：执行即时ND到NZ布局变换的数据拷贝操作。

**语法**：

```mlir
operation ::= `hivm.hir.nd2nz` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`init_out_buffer` `=` $init_out_buffer^ )?
              (`pad_value` `=` $pad_value^ `:` type($pad_value))?
              (`init_condition` `=` $init_condition^ `:` type($init_condition))?
              (`->` type($result_tensor)^)?
```

**示例**：

```mlir
hivm.hir.nd2nz ins(%src : memref<32x32xf16>) outs(%dst : memref<32x32xf16>) {dst_continuous}
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CubeCoreTypeTrait`, `OpPipeTrait<PIPE::PIPE_MTE2>`, `SinglePipeOpTrait`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 任意类型值的shaped类型 |
| `dst` | 任意类型值的shaped类型 |
| `pad_value` | 任意类型 |
| `init_condition` | 任意类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result_tensor` | 可变参数，任意类型的ranked tensor |

### hivm.hir.nz2nd (hivm::NZ2NDOp)

**功能**：从L1到全局内存执行NZ2ND转换的数据拷贝操作。

**语法**：

```mlir
operation ::= `hivm.hir.nz2nd` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`->` type($result_tensor)^)?
```

**示例**：

```mlir
hivm.hir.nz2nd ins(%src : memref<32x32xf16, #hivm.address_space<l1>>) outs(%dst : memref<32x32xf16, #hivm.address_space<gm>>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `CubeCoreTypeTrait`, `OpPipeTrait<PIPE::PIPE_MTE3>`, `SinglePipeOpTrait`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result_tensor` | 任意类型的ranked tensor |

### hivm.hir.pipe_barrier (hivm::PipeBarrierOp)

**功能**：在指定pipe上插入屏障，用于同步。

**语法**：

```mlir
operation ::= `hivm.hir.pipe_barrier` `[` $pipe `]` attr-dict
```

**示例**：

```mlir
hivm.hir.pipe_barrier [#hivm.pipe<PIPE_V>]
```

### hivm.hir.pointer_cast (hivm::PointerCastOp)

**功能**：将指定的64位整型地址转换为memref。

**语法**：

```mlir
operation ::= `hivm.hir.pointer_cast` `(`$addrs `)` (`[` $dynamicSizes^`]`)? attr-dict `:` type($result)
```

**示例**：

```mlir
%addr = arith.constant 1234 : i64
%tmp = hivm.hir.pointer_cast(%addr) : memref<32xf32>

%addr2 = arith.constant 1600 : i64
%addr3 = arith.constant 3200 : i64
%tmp2 = hivm.hir.pointer_cast(%addr, %addr2) : memref<32xf32>
%tmp3 = hivm.hir.pointer_cast(%addr, %addr2, %addr3) : memref<32xf32>
```

**特性**：`AttrSizedOperandSegments`, `CubeVectorCoreTypeTrait`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `addrs` | 可变参数，64位无符号整数 |
| `dynamicSizes` | 可变参数，index类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 任意类型值的memref |

### hivm.hir.scatter_store (hivm::ScatterStoreOp)

**功能**：执行稀疏内存存储，支持可选mask。

**语法**：

```mlir
operation ::= `hivm.hir.scatter_store` `ins` `(` $indices `:` type($indices) `,`
              $data `:` type($data) `,` $burst_len `:` type($burst_len)
              (`,` $mask `:` type($mask)^)? `)`
              `outs` `(` $base `:` type($base) `)`
              attr-dict
              (`->` type($result)^)?
```

该操作输入UB上的源tensor和偏移tensor，按指定偏移位置将源中的元素存入目标GM buffer，支持mask以条件控制存储哪些元素。参数含义与gather_load相同。

**特性**：`DestinationStyleOpInterface`, `HIVMCoreTypeInterface`, `MemoryEffectOpInterface`, `OpPipeInterface`, `StoreLikeOpInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `cache` | `::mlir::hivm::CacheModifierAttr` | 缓存修改策略 |
| `evict` | `::mlir::hivm::EvictionPolicyAttr` | 共享内存的逐出策略 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `indices` | 32位或64位无符号整数值的ranked tensor |
| `data` | 任意类型的ranked tensor |
| `burst_len` | 32位或64位无符号整数 |
| `mask` | 1位无符号整数值的ranked tensor |
| `base` | 类型为Tensor或Memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 任意类型的ranked tensor |

### hivm.hir.scatterT (hivm::ScatterTOp)

**功能**：通过SIMT模板将value tensor中的所有值按index_tile tensor指定的索引写入dst的scatter操作。

**语法**：

```mlir
operation ::= `hivm.hir.scatterT` `ins` `(` $dst `:` type($dst) `,`
              $value `:` type($value)`,`
              $index_tile `:` type($index_tile)`,`
              $index_boundary `:` type($index_boundary)`,`
              $dim `:` type($dim)`,`
              `[` $dst_stride `:` type($dst_stride) `]` `,`
              `[` $index_shape `:` type($index_shape) `]` `,`
              `[` $offsets `:` type($offsets) `]` `)`
              attr-dict
```

该scatter操作按index_tile UB tensor指定的位置加偏移，将UB tensor中的值赋给GM上的dst buffer，使用SIMT模板。支持2D-5D。

以3D为例：`dim=0`时`Dst[index_tile[i][j][k]][j][k] = value[i][j][k]`；`dim=1`时`Dst[i][index_tile[i][j][k]][k] = value[i][j][k]`；`dim=2`时`Dst[i][j][index_tile[i][j][k]] = value[i][j][k]`。

其中：

- `dst`：目标tensor GM内存buffer的基指针。
- `value`：UB上待存储的值tensor。
- `index_tile`：UB上指定目标tensor中位置的索引tensor，索引不应重复。
- `index_boundary`：索引的上界。
- `dim`：执行赋值操作的维度。
- `dst_stride`：目标stride元组。
- `index_shape`：索引tensor的shape。
- `offsets`：index tile各维度上的偏移元组。

**特性**：`DestinationStyleOpInterface`, `HIVMCoreTypeInterface`, `MemoryEffectOpInterface`, `OpPipeInterface`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `dst` | 任意类型值的memref |
| `value` | 类型为Tensor或Memref |
| `index_tile` | 类型为Tensor或Memref |
| `index_boundary` | 32位或64位无符号整数 |
| `dim` | 32位或64位无符号整数 |
| `dst_stride` | 可变参数，32位或64位无符号整数 |
| `index_shape` | 可变参数，32位或64位无符号整数 |
| `offsets` | 可变参数，32位或64位无符号整数 |

### hivm.hir.set_atomic (hivm::SetAtomicOp)

**功能**：HIVM设置原子操作。

**语法**：

```mlir
operation ::= `hivm.hir.set_atomic` `kind` `=` $kind `[` `type` `=` $type `]` attr-dict
```

该操作为写入全局内存的任何数据启用原子操作。设置`kind = #hivm.atomic_kind<NONE>`可复位原子操作。

支持的原子操作数据类型包括：F32、F16、S16、S32、S8、BF16。

支持的原子操作类型包括：ADD、MAX、MIN。

**示例**：

```mlir
hivm.hir.set_atomic kind= #hivm.atomic_kind<ADD> type = i32
// all subsequent store-like operations will enable atomic store with addition
hivm.hir.store ins(...) outs(...)
...
hivm.hir.set_atomic kind= #hivm.atomic_kind<NONE>
hivm.hir.fixpipe ins(...) outs(...)
```

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `kind` | `::mlir::hivm::AtomicKindAttr` | StoreOp的原子操作类型，HIVM原子store类型属性 |
| `type` | `::mlir::TypeAttr` | 任意类型属性 |

### hivm.hir.set_ctrl (hivm::SetCtrlOp)

**功能**：HIVM设置ctrl操作。

**语法**：

```mlir
operation ::= `hivm.hir.set_ctrl` $enable `at` `ctrl``[` $idx `]` attr-dict
```

该操作通过将特定位置0或置1来配置控制位。

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `enable` | `::mlir::BoolAttr` | 布尔属性 |
| `idx` | `::mlir::IntegerAttr` | 64位无符号整数属性 |

### hivm.hir.set_ffts_base_addr (hivm::SetFFTSBaseAddrOp)

**功能**：设置FFTS同步机制的基础地址。

**语法**：

```mlir
operation ::= `hivm.hir.set_ffts_base_addr` attr-dict $ffts_base_addr
```

**示例**：

```mlir
hivm.hir.set_ffts_base_addr %base_addr
```

**特性**：`CubeVectorCoreTypeTrait`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `ffts_base_addr` | 64位无符号整数 |

### hivm.hir.set_flag (hivm::SetFlagOp)

**功能**：设置同步标志。

**语法**：

```mlir
operation ::= `hivm.hir.set_flag` `[`
              $set_pipe
              `,` $wait_pipe
              `,` custom<EventID>($static_event_id, $dynamic_event_id)
              `]` attr-dict
```

**示例**：

```mlir
hivm.hir.set_flag [#hivm.pipe<PIPE_V>, #hivm.pipe<PIPE_M>, #hivm.event<EVENT_ID0>]
```

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `dynamic_event_id` | 64位无符号整数 |

### hivm.hir.set_mask_norm (hivm::SetMaskNormOp)

**功能**：设置掩码归一化模式。

**语法**：

```mlir
operation ::= `hivm.hir.set_mask_norm` attr-dict
```

**示例**：

```mlir
hivm.hir.set_mask_norm
```

### hivm.hir.store (hivm::StoreOp)

**功能**：将本地缓冲区数据存储到全局内存，支持原子操作。

**语法**：

```mlir
operation ::= `hivm.hir.store` `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              attr-dict
              (`atomic` `=` $atomic_kind^)?
              (`->` type($result_tensor)^)?
```

**示例**：

```mlir
hivm.store ins(%src : memref<16x16xf16, #hivm.address_space<ub>>) outs(%dst : memref<16x16xf16, #hivm.address_space<gm>>)
hivm.store ins(%src : memref<16x16xf16, #hivm.address_space<ub>>) outs(%dst : memref<16x16xf16, #hivm.address_space<gm>>) atomic = #hivm.atomic_kind<add>
```

**特性**：`AlwaysSpeculatableImplTrait`, `OpPipeTrait<PIPE::PIPE_MTE3>`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result_tensor` | 任意类型的ranked tensor |

### hivm.hir.stride_load (hivm::StrideLoadOp)

**功能**：执行带标量padding的跨步内存加载。

**语法**：

```mlir
operation ::= `hivm.hir.stride_load` `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `offset` `(` $offset `:` type($offset) `)`
              `other` `(` $other `:` type($other) `)`
              `strides` `(` `[` $stride `:` type($stride) `]` `)`
              `numels` `(` `[` $numel `:` type($numel) `]` `)`
              attr-dict
              (`->` type($result)^)?
```

该操作输入源GM buffer、一个线性基偏移、各维度物理stride及逻辑有效元素个数，按以下规则加载生成目标tensor或memref：

`dst[i, ...] = all(dim_index < numel[dim]) ? src[offset + sum(dim_index * stride[dim])] : other`

该操作面向SIMT Stride Load，当前实现支持1D-3D。

**特性**：`DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 任意类型值的memref |
| `dst` | 类型为Tensor或Memref |
| `offset` | 32位或64位无符号整数 |
| `other` | 任意类型 |
| `stride` | 可变参数，32位或64位无符号整数 |
| `numel` | 可变参数，32位或64位无符号整数 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 类型为Tensor或Memref |

### hivm.hir.stride_store (hivm::StrideStoreOp)

**功能**：执行跨步内存存储。

**语法**：

```mlir
operation ::= `hivm.hir.stride_store` `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `offset` `(` $offset `:` type($offset) `)`
              `strides` `(` `[` $stride `:` type($stride) `]` `)`
              `numels` `(` `[` $numel `:` type($numel) `]` `)`
              attr-dict
```

该操作输入目标GM buffer、源tensor或memref、一个线性基偏移、各维度物理stride及逻辑有效元素个数，对所有满足`all(dim_index < numel[dim])`的位置按以下规则存储源元素：

`dst[offset + sum(dim_index * stride[dim])] = src[i, ...]`

该操作面向SIMT Stride Store，当前实现支持1D-3D。

**特性**：`DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `dst` | 任意类型值的memref |
| `src` | 类型为Tensor或Memref |
| `offset` | 32位或64位无符号整数 |
| `stride` | 可变参数，32位或64位无符号整数 |
| `numel` | 可变参数，32位或64位无符号整数 |

### hivm.hir.sync_block (hivm::SyncBlockOp)

**功能**：在不同内核间执行块同步，支持多种同步模式。

**语法**：

```mlir
operation ::= `hivm.hir.sync_block` attr-dict `[` $sync_block_mode (`,` $flag_id^)?`]`
              (`ffts_base_addr` `=` $ffts_base_addr^)?
              (`tcube_pipe` `=` $tcube_pipe^)?
              (`tvector_pipe` `=` $tvector_pipe^)?
```

**示例**：

```mlir
hivm.hir.sync_block [#hivm.sync_block_mode<ALL>] {tvector_pipe = #hivm.pipe<PIPE_V>}
```

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `ffts_base_addr` | 64位无符号整数 |

### hivm.hir.sync_block_lock (hivm::SyncBlockLockOp)

**功能**：等待锁变量等于当前block索引，实现块间同步。

**语法**：

```mlir
operation ::= `hivm.hir.sync_block_lock` attr-dict `lock_var` `(` $lock_var `:` type($lock_var) `)`
```

**示例**：

```mlir
hivm.hir.sync_block_lock lock_var(%lock : memref<1xi64>)
```

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `lock_var` | 64位无符号整数的1D memref |

### hivm.hir.sync_block_set (hivm::SyncBlockSetOp)

**功能**：设置块同步的同步点。

**语法**：

```mlir
operation ::= `hivm.hir.sync_block_set` attr-dict `[` $tcore_type `,` $tpipe `,` $pipe`]`
              `flag` `=` custom<FlagID>($static_flag_id, $dynamic_flag_id)
              (`ffts_base_addr` `=` $ffts_base_addr^)?
              (`sync_instr_mode` `=` $tsync_instr_mode^)?
```

**示例**：

```mlir
hivm.hir.sync_block_set [#hivm.tcore_type<CUBE>, #hivm.pipe<PIPE_M>, #hivm.pipe<PIPE_V>] flag = #hivm.event<EVENT_ID0>
```

**特性**：`AttrSizedOperandSegments`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `dynamic_flag_id` | 64位无符号整数 |
| `ffts_base_addr` | 64位无符号整数 |

### hivm.hir.sync_block_unlock (hivm::SyncBlockUnlockOp)

**功能**：递增并释放锁变量。

**语法**：

```mlir
operation ::= `hivm.hir.sync_block_unlock` attr-dict `lock_var` `(` $lock_var `:` type($lock_var) `)`
```

**示例**：

```mlir
hivm.hir.sync_block_unlock lock_var(%lock : memref<1xi64>)
```

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `lock_var` | 64位无符号整数的1D memref |

### hivm.hir.sync_block_wait (hivm::SyncBlockWaitOp)

**功能**：等待指定块同步标志。

**语法**：

```mlir
operation ::= `hivm.hir.sync_block_wait` attr-dict `[` $tcore_type `,` $tpipe `,` $pipe`]`
              `flag` `=` custom<FlagID>($static_flag_id, $dynamic_flag_id)
```

**示例**：

```mlir
hivm.hir.sync_block_wait [#hivm.tcore_type<VECTOR>, #hivm.pipe<PIPE_V>, #hivm.pipe<PIPE_M>] flag = #hivm.event<EVENT_ID1>
```

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `dynamic_flag_id` | 64位无符号整数 |

### hivm.hir.vabs (hivm::VAbsOp)

**功能**：逐元素计算向量的绝对值。

**语法**：

```mlir
operation ::= `hivm.hir.vabs` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vabs ins(%src : memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型值的shaped类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vadd (hivm::VAddOp)

**功能**：逐元素执行二元向量加法运算。

**语法**：

```mlir
operation ::= `hivm.hir.vadd` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vadd ins(%src0, %src1 : memref<32xf32>, memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `CommutativeOpTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vand (hivm::VAndOp)

**功能**：逐元素执行二元向量按位与运算。

**语法**：

```mlir
operation ::= `hivm.hir.vand` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vand ins(%src0, %src1 : memref<32xi32>, memref<32xi32>) outs(%dst : memref<32xi32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `CommutativeOpTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`, `VectorOnlyTrait<1>`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.varange (hivm::VArangeOp)

**功能**：根据步长和偏移量生成等差序列填充向量。

**语法**：

```mlir
operation ::= `hivm.hir.varange` attr-dict
              (`offset` `[` $offset^ `]`)?
              `strides` `[` $strides `]`
              `outs` `(` $dst `:` type($dst) `)`
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.varange offset[%o] strides[%s0, %s1] outs(%dst : memref<32xf32>)
%result = hivm.hir.varange offset[%o] strides[%s0, %s1] outs(%dst : tensor<32xf32>) -> tensor<32xf32>
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `VectorCoreTypeTrait`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `dst` | 类型为Tensor或Memref |
| `offset` | index类型 |
| `strides` | 可变参数，index类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 任意类型的ranked tensor |

### hivm.hir.vatan (hivm::VAtanOp)

**功能**：逐元素计算向量反正切值。

**语法**：

```mlir
operation ::= `hivm.hir.vatan` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

该操作对N个操作数执行逐元素运算并产生单个结果，过程中可执行转置或广播（二者不可同时进行）。

公共约束：

1. 遵循`DestinationStyleOpInterface`。
2. 输入操作数数量为N，输出（结果）数量为一。
3. 输入（init）操作数与结果rank相同。
4. 第一个输入仅为向量。

附加约束：

1. 输入（init）操作数与结果元素类型相同。

**特性**：`ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`, `VectorizableOpInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `transpose` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性 |
| `broadcast` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vbrc (hivm::VBrcOp)

**功能**：将向量或标量沿指定维度进行广播。

**语法**：

```mlir
operation ::= `hivm.hir.vbrc` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast_dims` `=` $broadcast_dims^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vbrc ins(%src : i32) outs(%dst : memref<?xi32>)
hivm.hir.vbrc ins(%src : memref<1xi32>) outs(%dst : memref<?xi32>) broadcast_dims = [0]
%result = hivm.hir.vbrc ins(%src : tensor<1xi32>) outs(%dst : tensor<?xi32>) broadcast_dims = [0] -> tensor<?xi32>
```

**特性**：`AlwaysSpeculatableImplTrait`, `CollapsibleConsecutiveTargetDimsTrait`, `SameOperandsElementType`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 任意类型 |
| `dst` | 类型为Tensor或Memref |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vcast (hivm::VCastOp)

**功能**：逐元素执行向量类型转换，支持多种舍入模式。

**语法**：

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

**示例**：

```mlir
hivm.hir.vcast ins(%src : memref<32xf32>) outs(%dst : memref<32xi32>) {round_mode = #hivm.round_mode<rint>}
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vcmp (hivm::VCmpOp)

**功能**：逐元素执行二元向量比较，结果存入bool向量。

**语法**：

```mlir
operation ::= `hivm.hir.vcmp` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`compare_mode` `=` $compare_mode^)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vcmp ins(%src0, %src1 : memref<32xf32>, memref<32xf32>) outs(%dst : memref<32xi1>) {compare_mode = #hivm.compare_mode<GE>}
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vconcat (hivm::VConcatOp)

**功能**：沿指定维度拼接多个向量。

**语法**：

```mlir
operation ::= `hivm.hir.vconcat` `dim` `(` $dim `)` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vconcat dim(1) ins(%0, %1 : tensor<136x2048xf32>, tensor<136x2048xf32>) outs(%2 : tensor<136x4096xf32>) -> tensor<136x4096xf32>
```

**特性**：`AlwaysSpeculatableImplTrait`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 类型为Tensor或Memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vcos (hivm::VCosOp)

**功能**：逐元素计算向量的余弦值。

**语法**：

```mlir
operation ::= `hivm.hir.vcos` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vcos ins(%src : memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vcummax (hivm::VCummaxOp)

**功能**：向量累积最大值操作。

**语法**：

```mlir
operation ::= `hivm.hir.vcummax` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              `cum_dims` `=` $cum_dims
              `reverse` `=` $reverse
              (`->` type($result)^)?
```

沿`src`的指定轴计算每个元素的累积最大值。cummax输出中指定轴上的每个元素包含原始`src`中从第一个元素到当前位置所有元素的最大值。

约束：

1. 输入向量与输出向量必须rank相同且元素类型相同。

参数：

- `src`：计算累积最大值的tensor或memref。
- `dst`：存储元素的tensor或memref。
- `cum_dims`：指定计算累积最大值的维度。

**示例**：

```mlir
hivm.hir.vcummax ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) cum_dims : [0]
%result = hivm.hir.vcummax ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) cum_dims : [0] -> tensor<?xf32>
```

**特性**：`ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`, `VectorizableOpInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `cum_dims` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性，应按递增顺序排列 |
| `reverse` | `::mlir::BoolAttr` | 布尔属性 |
| `propagate_nan` | `::mlir::BoolAttr` | 布尔属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vcummin (hivm::VCumminOp)

**功能**：向量累积最小值操作。

**语法**：

```mlir
operation ::= `hivm.hir.vcummin` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              `cum_dims` `=` $cum_dims
              `reverse` `=` $reverse
              (`->` type($result)^)?
```

沿`src`的指定轴计算每个元素的累积最小值。cummin输出中指定轴上的每个元素包含原始`src`中从第一个元素到当前位置所有元素的最小值。

约束：

1. 输入向量与输出向量必须rank相同且元素类型相同。

参数：

- `src`：计算累积最小值的tensor或memref。
- `dst`：存储元素的tensor或memref。
- `cum_dims`：指定计算累积最小值的维度。

**示例**：

```mlir
hivm.hir.vcummin ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) cum_dims : [0]
%result = hivm.hir.vcummin ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) cum_dims : [0] -> tensor<?xf32>
```

**特性**：`ConditionallySpeculatable`, `DestinationStyleOpInterface`, `ExtraBufferOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`, `VectorizableOpInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `cum_dims` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性，应按递增顺序排列 |
| `reverse` | `::mlir::BoolAttr` | 布尔属性 |
| `propagate_nan` | `::mlir::BoolAttr` | 布尔属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vcumprod (hivm::VCumprodOp)

**功能**：沿指定维度计算向量的累积乘积，支持反向方向。

**语法**：

```mlir
operation ::= `hivm.hir.vcumprod` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `cum_dims` `=` $cum_dims
              `reverse` `=` $reverse
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vcumprod ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) cum_dims : [0] reverse = true
%result = hivm.hir.vcumprod ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) cum_dims : [0] reverse = true -> tensor<?xf32>
```

**特性**：`AlwaysSpeculatableImplTrait`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vcumsum (hivm::VCumsumOp)

**功能**：沿指定维度计算向量的累积和，支持反向方向。

**语法**：

```mlir
operation ::= `hivm.hir.vcumsum` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `cum_dims` `=` $cum_dims
              `reverse` `=` $reverse
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vcumsum ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) cum_dims : [0] reverse = true
%result = hivm.hir.vcumsum ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) cum_dims : [0] reverse = true -> tensor<?xf32>
```

**特性**：`AlwaysSpeculatableImplTrait`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vdeinterleave (hivm::VDeinterleaveOp)

**功能**：沿最后一个维度对向量进行解交织。

**语法**：

```mlir
operation ::= `hivm.hir.vdeinterleave` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`channel_num` `=` $channel_num^)?
              (`index_mode` `=` $index_mode^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vdeinterleave ins(%src : memref<32x16xf32>) outs(%dst0, %dst1 : memref<32x8xf32>, memref<32x8xf32>) {channel_num = 2}
```

**特性**：`AlwaysSpeculatableImplTrait`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 可变参数，类型为Tensor或Memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vdiv (hivm::VDivOp)

**功能**：逐元素执行二元向量除法运算。

**语法**：

```mlir
operation ::= `hivm.hir.vdiv` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vdiv ins(%src0, %src1 : memref<32xf32>, memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.verf (hivm::VErfOp)

**功能**：逐元素计算向量的误差函数。

**语法**：

```mlir
operation ::= `hivm.hir.verf` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.verf ins(%src : memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vexp (hivm::VExpOp)

**功能**：逐元素计算向量的指数值。

**语法**：

```mlir
operation ::= `hivm.hir.vexp` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vexp ins(%src : memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型值的shaped类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vexp2 (hivm::VExp2Op)

**功能**：逐元素计算向量以2为底的指数值。

**语法**：

```mlir
operation ::= `hivm.hir.vexp2` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

该操作对N个操作数执行逐元素运算并产生单个结果，过程中可执行转置或广播（二者不可同时进行）。

公共约束：

1. 遵循`DestinationStyleOpInterface`。
2. 输入操作数数量为N，输出（结果）数量为一。
3. 输入（init）操作数与结果rank相同。
4. 第一个输入仅为向量。

附加约束：

1. 输入（init）操作数与结果元素类型相同。
2. 支持的元素类型为f16和f32。
3. 该操作为仅规范化（normalize-only）的源形式，应由HIVM normalize进行lower。

**特性**：`ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`, `VectorizableOpInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `transpose` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性 |
| `broadcast` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vexpm1 (hivm::VExpM1Op)

**功能**：逐元素计算向量exp(x) - 1的值。

**语法**：

```mlir
operation ::= `hivm.hir.vexpm1` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

该操作对N个操作数执行逐元素运算并产生单个结果，过程中可执行转置或广播（二者不可同时进行）。

公共约束：

1. 遵循`DestinationStyleOpInterface`。
2. 输入操作数数量为N，输出（结果）数量为一。
3. 输入（init）操作数与结果rank相同。
4. 第一个输入仅为向量。

附加约束：

1. 输入（init）操作数与结果元素类型相同。

**特性**：`ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`, `VectorizableOpInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `transpose` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性 |
| `broadcast` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vflip (hivm::VFlipOp)

**功能**：沿指定轴翻转向量元素顺序。

**语法**：

```mlir
operation ::= `hivm.hir.vflip` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `flip_axis` `=` $flip_axis
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vflip ins(%src : memref<32x16xf32>) outs(%dst : memref<32x16xf32>) flip_axis = 1
```

**特性**：`AlwaysSpeculatableImplTrait`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vgather (hivm::VGatherOp)

**功能**：根据索引从源向量中收集元素。

**语法**：

```mlir
operation ::= `hivm.hir.vgather` attr-dict `ins` `(` $src `:` type($src) `)`
              `indices` `(` $indices `:` type($indices) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vgather ins(%src : memref<32x16xf32>) indices(%idx : memref<32x8xi32>) outs(%dst : memref<32x8xf32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `indices` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vgathermask (hivm::VGatherMaskOp)

**功能**：向量Gather Mask操作。

**语法**：

```mlir
operation ::= `hivm.hir.vgathermask` attr-dict `ins` `(` $src `:` type($src) `)`
              `mask` `(` $mask `:` type($mask) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`->` type($result)^)?
```

从`src`tensor中gather对应`mask`tensor元素为True的元素，结果写入预分配的`dst`tensor或memref（InPlace语义），并将`dst`tensor作为`result`返回以保持函数式兼容。`dst`、`result`tensor与`src`tensor具有相同的形状和元素类型。

参数：

- `src`：从中gather元素的tensor或memref。
- `mask`：过滤元素的mask tensor或memref（非零即选择src元素）。
- `dst`：存储gather所得元素的预分配tensor或memref（InPlace写入）。

结果：

- `result`：同`dst`（为兼容而做的函数式返回）。
- `dst_size`：实际从src gather的元素个数。

**示例**：

```mlir
result_data, result_size = vgathermask ins (%src : tensor<16xf32>) mask (%mask : tensor<16xi1>) outs (%dst_data, %dst_size : tensor<16xf32>, tensor<1xi64>)
```

**特性**：`ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`, `VectorizableOpInterface`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `mask` | 类型为Tensor或Memref |
| `dst` | 可变参数，类型为Tensor或Memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vilogb (hivm::VIlogbOp)

**功能**：逐元素执行向量ilogb运算。

**语法**：

```mlir
operation ::= `hivm.hir.vilogb` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

该操作对N个操作数执行逐元素运算并产生单个结果，过程中可执行转置或广播（二者不可同时进行）。

公共约束：

1. 遵循`DestinationStyleOpInterface`。
2. 输入操作数数量为N，输出（结果）数量为一。
3. 输入（init）操作数与结果rank相同。
4. 第一个输入仅为向量。

附加约束：

1. 输入（init）操作数与结果元素类型相同。

**特性**：`ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`, `VectorizableOpInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `transpose` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性 |
| `broadcast` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vinterleave (hivm::VInterleaveOp)

**功能**：沿最后一个维度交织多个向量的值。

**语法**：

```mlir
operation ::= `hivm.hir.vinterleave` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `interleave_channel_nums` `=` $interleave_channel_nums
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vinterleave ins(%src0, %src1 : memref<32x8xf32>, memref<32x8xf32>) outs(%dst : memref<32x16xf32>) interleave_channel_nums = 2
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 类型为Tensor或Memref |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.visinf (hivm::VIsInfOp)

**功能**：逐元素判断向量元素是否为无穷。

**语法**：

```mlir
operation ::= `hivm.hir.visinf` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`->` type($result)^)?
```

该操作对N个操作数执行逐元素运算并产生单个结果，过程中可执行转置或广播（二者不可同时进行）。

公共约束：

1. 遵循`DestinationStyleOpInterface`。
2. 输入操作数数量为N，输出（结果）数量为一。
3. 输入（init）操作数与结果rank相同。
4. 第一个输入仅为向量。

若输入元素为正或负无穷则返回1，否则返回0。

附加约束：

1. 输入（init）操作数与结果元素类型相同。
2. `dst`的元素类型必须为bool（i1）。

**特性**：`ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`, `VectorizableOpInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `transpose` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性 |
| `broadcast` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型值的shaped类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.visnan (hivm::VIsNanOp)

**功能**：逐元素判断向量元素是否为NaN。

**语法**：

```mlir
operation ::= `hivm.hir.visnan` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`->` type($result)^)?
```

该操作对N个操作数执行逐元素运算并产生单个结果，过程中可执行转置或广播（二者不可同时进行）。

公共约束：

1. 遵循`DestinationStyleOpInterface`。
2. 输入操作数数量为N，输出（结果）数量为一。
3. 输入（init）操作数与结果rank相同。
4. 第一个输入仅为向量。

若输入元素为NaN（Not a Number）则返回1，否则返回0。

附加约束：

1. 输入（init）操作数与结果元素类型相同。
2. `dst`的元素类型必须为bool（i1）。

**特性**：`ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`, `VectorizableOpInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `transpose` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性 |
| `broadcast` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型值的shaped类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vldexp (hivm::VLdexpOp)

**功能**：逐元素执行二元向量ldexp运算。

**语法**：

```mlir
operation ::= `hivm.hir.vldexp` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

该操作对N个操作数执行逐元素运算并产生单个结果，过程中可执行转置或广播（二者不可同时进行）。

公共约束：

1. 遵循`DestinationStyleOpInterface`。
2. 输入操作数数量为N，输出（结果）数量为一。
3. 输入（init）操作数与结果rank相同。
4. 第一个输入仅为向量。

附加约束：

1. 输入（init）操作数与结果元素类型相同。
2. 该操作为仅规范化（normalize-only）的源形式，应在代码生成前由HIVM normalize进行lower。

**特性**：`ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`, `VectorizableOpInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `transpose` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性 |
| `broadcast` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vln (hivm::VLnOp)

**功能**：逐元素计算向量的自然对数。

**语法**：

```mlir
operation ::= `hivm.hir.vln` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vln ins(%src : memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型值的shaped类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vlog10 (hivm::VLog10Op)

**功能**：逐元素计算向量以10为底的对数值。

**语法**：

```mlir
operation ::= `hivm.hir.vlog10` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

该操作对N个操作数执行逐元素运算并产生单个结果，过程中可执行转置或广播（二者不可同时进行）。

公共约束：

1. 遵循`DestinationStyleOpInterface`。
2. 输入操作数数量为N，输出（结果）数量为一。
3. 输入（init）操作数与结果rank相同。
4. 第一个输入仅为向量。

附加约束：

1. 输入（init）操作数与结果元素类型相同。

**特性**：`ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`, `VectorizableOpInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `transpose` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性 |
| `broadcast` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vlog1p (hivm::VLog1pOp)

**功能**：逐元素计算向量log(1 + x)的值。

**语法**：

```mlir
operation ::= `hivm.hir.vlog1p` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

该操作对N个操作数执行逐元素运算并产生单个结果，过程中可执行转置或广播（二者不可同时进行）。

公共约束：

1. 遵循`DestinationStyleOpInterface`。
2. 输入操作数数量为N，输出（结果）数量为一。
3. 输入（init）操作数与结果rank相同。
4. 第一个输入仅为向量。

附加约束：

1. 输入（init）操作数与结果元素类型相同。

**特性**：`ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`, `VectorizableOpInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `transpose` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性 |
| `broadcast` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vlog2 (hivm::VLog2Op)

**功能**：逐元素计算向量以2为底的对数值。

**语法**：

```mlir
operation ::= `hivm.hir.vlog2` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

该操作对N个操作数执行逐元素运算并产生单个结果，过程中可执行转置或广播（二者不可同时进行）。

公共约束：

1. 遵循`DestinationStyleOpInterface`。
2. 输入操作数数量为N，输出（结果）数量为一。
3. 输入（init）操作数与结果rank相同。
4. 第一个输入仅为向量。

附加约束：

1. 输入（init）操作数与结果元素类型相同。

**特性**：`ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`, `VectorizableOpInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `transpose` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性 |
| `broadcast` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vmax (hivm::VMaxOp)

**功能**：逐元素计算两个向量的最大值。

**语法**：

```mlir
operation ::= `hivm.hir.vmax` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vmax ins(%src0, %src1 : memref<32xf32>, memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `CommutativeOpTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vmin (hivm::VMinOp)

**功能**：逐元素计算两个向量的最小值。

**语法**：

```mlir
operation ::= `hivm.hir.vmin` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vmin ins(%src0, %src1 : memref<32xf32>, memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `CommutativeOpTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vmod (hivm::VModOp)

**功能**：逐元素计算向量取模运算。

**语法**：

```mlir
operation ::= `hivm.hir.vmod` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vmod ins(%src0, %src1 : memref<32xi32>, memref<32xi32>) outs(%dst : memref<32xi32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vmodui (hivm::VModUIOp)

**功能**：逐元素执行向量无符号整数取模运算。

**语法**：

```mlir
operation ::= `hivm.hir.vmodui` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

该操作对N个操作数执行逐元素运算并产生单个结果，过程中可执行转置或广播（二者不可同时进行）。

公共约束：

1. 遵循`DestinationStyleOpInterface`。
2. 输入操作数数量为N，输出（结果）数量为一。
3. 输入（init）操作数与结果rank相同。
4. 第一个输入仅为向量。

附加约束：

1. 输入（init）操作数与结果元素类型相同。
2. 对于I1（bool），由于布尔值的x % y恒为0，规范化产生fill(0)。
3. 对于I8，规范化将两个操作数拓宽到I16，计算无符号余数，再将结果转换回I8。

**特性**：`ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`, `VectorizableOpInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `transpose` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性 |
| `broadcast` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vmul (hivm::VMulOp)

**功能**：逐元素执行二元向量乘法运算。

**语法**：

```mlir
operation ::= `hivm.hir.vmul` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vmul ins(%src0, %src1 : memref<32xf32>, memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `CommutativeOpTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vmulext (hivm::VMulExtOp)

**功能**：逐元素执行二元向量乘法，并计算高32位结果。

**语法**：

```mlir
operation ::= `hivm.hir.vmulext` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vmulext ins(%src0, %src1 : memref<32xi32>, memref<32xi32>) outs(%dst : memref<32xi32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vmulextended (hivm::VMulextendedOp)

**功能**：对两个tensor执行向量乘法，同时获取高16位和低16位结果。

**语法**：

```mlir
operation ::= `hivm.hir.vmulextended` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vmulextended ins(%src0, %src1 : memref<32xi16>, memref<32xi16>) outs(%dst_hi, %dst_lo : memref<32xi16>, memref<32xi16>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，类型为Tensor或Memref |
| `dst` | 可变参数，类型为Tensor或Memref |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vmulextui (hivm::VMulExtUiOp)

**功能**：逐元素执行二元向量无符号整数乘法，计算结果的最高32位。

**语法**：

```mlir
operation ::= `hivm.hir.vmulextui` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

该操作对N个操作数执行逐元素运算并产生单个结果，过程中可执行转置或广播（二者不可同时进行）。

公共约束：

1. 遵循`DestinationStyleOpInterface`。
2. 输入操作数数量为N，输出（结果）数量为一。
3. 输入（init）操作数与结果rank相同。
4. 第一个输入仅为向量。

附加约束：

1. 输入（init）操作数与结果元素类型相同。
2. 支持Vector-Vector运算。

**特性**：`ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `ImplByScalarOpInterface`, `MemoryEffectsOpInterface`, `OpPipeInterface`, `VectorizableOpInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `transpose` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性 |
| `broadcast` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vnot (hivm::VNotOp)

**功能**：逐元素执行向量按位非运算。

**语法**：

```mlir
operation ::= `hivm.hir.vnot` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vnot ins(%src : memref<32xi32>) outs(%dst : memref<32xi32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型值的shaped类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vor (hivm::VOrOp)

**功能**：逐元素执行二元向量按位或运算。

**语法**：

```mlir
operation ::= `hivm.hir.vor` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vor ins(%src0, %src1 : memref<32xi32>, memref<32xi32>) outs(%dst : memref<32xi32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `CommutativeOpTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`, `VectorOnlyTrait<1>`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vpad (hivm::VPadOp)

**功能**：对输入向量进行填充，类似`tensor.pad`语义。

**语法**：

```mlir
operation ::= `hivm.hir.vpad` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `low` `` custom<DynamicIndexList>($low, $static_low)
              `high` `` custom<DynamicIndexList>($high, $static_high)
              `pad_value` $pad_value `:` type($pad_value)
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vpad ins(%src : tensor<2x16xf32>) outs(%dst: tensor<?x16xf32>) low[%first_dim_low, 0] high[%first_dim_high, 0] pad_value %pad_value : f32 -> tensor<?x16xf32>
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |
| `pad_value` | 任意类型 |
| `low` | 可变参数，index类型 |
| `high` | 可变参数，index类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vpow (hivm::VPowOp)

**功能**：逐元素执行向量幂运算（指数为标量或向量）。

**语法**：

```mlir
operation ::= `hivm.hir.vpow` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vpow ins(%src0, %src1 : memref<32xf32>, memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`, `VectorOnlyTrait<1>`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vrec (hivm::VRecOp)

**功能**：逐元素计算向量的倒数。

**语法**：

```mlir
operation ::= `hivm.hir.vrec` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vrec ins(%src : memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型值的shaped类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vreduce (hivm::VReduceOp)

**功能**：沿指定轴对向量进行规约操作。

**语法**：

```mlir
operation ::= `hivm.hir.vreduce` attr-dict $arith `ins` `(` $src `:` type($src) `)`
              (`indices` `(` $indices^ `:` type($indices) `)`)?
              `outs` `(` $dst `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              `reduce_dims` `=` $reduce_dims
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vreduce <add> ins(%src : memref<?xf32>) outs(%dst : memref<1xf32>) reduce_dims : [1]
%result = hivm.hir.vreduce <max> ins(%src : tensor<?xf32>) outs(%dst : tensor<1xf32>) reduce_dims : [0] -> tensor<1xf32>
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 可变参数，类型为Tensor或Memref |
| `temp_buffer` | 任意类型值的memref |
| `indices` | 类型为Tensor或Memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vrelu (hivm::VReluOp)

**功能**：逐元素计算向量的ReLU激活函数。

**语法**：

```mlir
operation ::= `hivm.hir.vrelu` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vrelu ins(%src : memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型值的shaped类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vrsqrt (hivm::VRsqrtOp)

**功能**：逐元素计算向量倒数平方根。

**语法**：

```mlir
operation ::= `hivm.hir.vrsqrt` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vrsqrt ins(%src : memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型值的shaped类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vsel (hivm::VSelOp)

**功能**：根据条件向量逐元素选择两个源向量中的值。

**语法**：

```mlir
operation ::= `hivm.hir.vsel` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vsel ins(%cond, %src0, %src1 : memref<32xi1>, memref<32xf32>, memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<3>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vshl (hivm::VShLOp)

**功能**：逐元素执行向量左移运算（标量移位量）。

**语法**：

```mlir
operation ::= `hivm.hir.vshl` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vshl ins(%src_vec, %shift : memref<32xi32>, i32) outs(%dst : memref<32xi32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `ScalarOnlyHWTrait<1>`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vshr (hivm::VShROp)

**功能**：逐元素执行向量右移运算（标量移位量），支持算术右移舍入。

**语法**：

```mlir
operation ::= `hivm.hir.vshr` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`round` `:` $round^ )?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vshr ins(%src_vec, %shift : memref<32xi32>, i32) outs(%dst : memref<32xi32>) {round}
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `ScalarOnlyHWTrait<1>`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vsin (hivm::VSinOp)

**功能**：逐元素计算向量的正弦值。

**语法**：

```mlir
operation ::= `hivm.hir.vsin` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vsin ins(%src : memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vsort (hivm::VSortOp)

**功能**：对向量的指定轴进行排序，输出排序后的值和对应索引。

**语法**：

```mlir
operation ::= `hivm.hir.vsort` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `descending` `=` $descending
              `sort_axis` `=` $sort_axis
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vsort ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) descending = true sort_axis = 0
%result = hivm.hir.vsort ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) descending = true sort_axis = 0 -> tensor<?xf32>
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `VectorCoreTypeTrait`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 可变参数，类型为Tensor或Memref |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vsqrt (hivm::VSqrtOp)

**功能**：逐元素计算向量的平方根。

**语法**：

```mlir
operation ::= `hivm.hir.vsqrt` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vsqrt ins(%src : memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型值的shaped类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vsub (hivm::VSubOp)

**功能**：逐元素执行二元向量减法运算。

**语法**：

```mlir
operation ::= `hivm.hir.vsub` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vsub ins(%src0, %src1 : memref<32xf32>, memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `BroadcastableOTF`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vtan (hivm::VTanOp)

**功能**：逐元素计算向量正切值。

**语法**：

```mlir
operation ::= `hivm.hir.vtan` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

该操作对N个操作数执行逐元素运算并产生单个结果，过程中可执行转置或广播（二者不可同时进行）。

公共约束：

1. 遵循`DestinationStyleOpInterface`。
2. 输入操作数数量为N，输出（结果）数量为一。
3. 输入（init）操作数与结果rank相同。
4. 第一个输入仅为向量。

附加约束：

1. 输入（init）操作数与结果元素类型相同。

**特性**：`ConditionallySpeculatable`, `DestinationStyleOpInterface`, `FlattenInterface`, `HIVMCoreTypeInterface`, `HIVMStructuredOpInterface`, `HIVMStructuredOp`, `MemoryEffectsOpInterface`, `OpPipeInterface`, `VectorizableOpInterface`

**属性**：

| 属性 | 类型 | 说明 |
| :--: | :--: | ---- |
| `transpose` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性 |
| `broadcast` | `::mlir::DenseI64ArrayAttr` | i64 dense数组属性 |

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vtanh (hivm::VTanhOp)

**功能**：逐元素计算向量的双曲正切值。

**语法**：

```mlir
operation ::= `hivm.hir.vtanh` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vtanh ins(%src : memref<32xf32>) outs(%dst : memref<32xf32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<1>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vtranspose (hivm::VTransposeOp)

**功能**：根据指定的排列对向量维度进行转置。

**语法**：

```mlir
operation ::= `hivm.hir.vtranspose` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`permutation` `=` $permutation^)?
              (`disable_align` `=` $disable_align^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vtranspose ins(%src : memref<32x8xf32>) outs(%dst : memref<8x32xf32>) permutation = [1, 0]
%result = hivm.hir.vtranspose ins(%src : tensor<32x8xf32>) outs(%dst: tensor<8x32xf32>) permutation = [1, 0] -> tensor<8x32xf32>
```

**特性**：`AlwaysSpeculatableImplTrait`, `OpPipeTrait<PIPE::PIPE_V>`, `SinglePipeOpTrait`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 类型为Tensor或Memref |
| `dst` | 类型为Tensor或Memref |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.vxor (hivm::VXorOp)

**功能**：逐元素执行二元向量按位异或运算。

**语法**：

```mlir
operation ::= `hivm.hir.vxor` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
```

**示例**：

```mlir
hivm.hir.vxor ins(%src0, %src1 : memref<32xi32>, memref<32xi32>) outs(%dst : memref<32xi32>)
```

**特性**：`AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `CollapsibleConsecutiveTargetDimsTrait`, `ElementwiseNaryOpTrait<2>`, `HIVMOpSameOperandsAndResultRank`, `OpPipeTrait<PIPE::PIPE_V>`, `SameOperandsElementType`, `SinglePipeOpTrait`, `TransposableOTF`, `UniformReassociationFlattenTrait`, `VectorCoreTypeTrait`, `VectorOnlyTrait<0>`, `VectorOnlyTrait<1>`

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `src` | 可变参数，任意类型值的shaped类型 |
| `dst` | 可变参数，任意类型值的shaped类型 |
| `temp_buffer` | 任意类型值的memref |

**结果**：

| 结果 | 说明 |
| :--: | ---- |
| `result` | 可变参数，任意类型的ranked tensor |

### hivm.hir.wait_flag (hivm::WaitFlagOp)

**功能**：等待指定同步标志。

**语法**：

```mlir
operation ::= `hivm.hir.wait_flag` `[`
              $set_pipe
              `,` $wait_pipe
              `,` custom<EventID>($static_event_id, $dynamic_event_id)
              `]` attr-dict
```

**示例**：

```mlir
hivm.hir.wait_flag [#hivm.pipe<PIPE_M>, #hivm.pipe<PIPE_V>, #hivm.event<EVENT_ID0>]
```

**操作数**：

| 操作数 | 说明 |
| :----: | ---- |
| `dynamic_event_id` | 64位无符号整数 |

## 属性

### DisableSizeAlignForCastAttr

**功能**：若函数带有该属性，则禁用VCast分配大小对齐，并使用新的临时buffer方案（基于对齐后的src大小）；不带该属性时使用原方案（大小对齐 + 基于src分配的临时buffer）。

**参数说明**：不涉及。

### AIC_BITCODEAttr

**功能**：定义HIVM AIC bitcode路径属性（字符串）。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| path | `StringAttr` | AIC bitcode路径 |

### AIV_BITCODEAttr

**功能**：定义HIVM AIV bitcode路径属性（字符串）。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| path | `StringAttr` | AIV bitcode路径 |

### AddressSpaceAttr

**功能**：定义HIVM地址空间映射属性，映射到GM、L1、L0A、L0B、L0C和UB。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| address_space | `::mlir::hivm::AddressSpace` | 类型为AddressSpace的枚举 |

### AlignKindAttr

**功能**：定义对齐方式信息。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::AlignKind` | 类型为AlignKind的枚举 |

### AllocAlignDimsAttr

**功能**：定义分配对齐维度信息。

**参数说明**：不涉及。

### AllocAlignValueInByteAttr

**功能**：定义按字节对齐的分配大小信息。

**参数说明**：不涉及。

### AllowFlattenAttr

**功能**：标记一个scope，即使启用了skip-scope，其body对hfusion-flatten-ops、propagate-reshape也是安全的。供软件atomic_xchg分解使用，使临界区内的多rank tensor可以折叠，以降低UB压力。

**参数说明**：不涉及。

### AtomicKindAttr

**功能**：定义StoreOp的原子操作类型。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::AtomicKind` | 类型为AtomicKind的枚举 |

### AxisKindAttr

**功能**：定义轴类型信息。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::AxisKind` | 类型为AxisKind的枚举 |

### BackupFunctionAttr

**功能**：HIVM backup函数属性。

**参数说明**：不涉及。

### HIVMBlockMappingAttr

**功能**：定义块映射信息。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| order | `std::optional<int32_t>` | 顺序标识 |

### CVPipelinedLoopAttr

**功能**：标记一个循环为CV流水线循环。

**参数说明**：不涉及。

### CVPipelinedMultiBufferAttr

**功能**：标记一个`memref.alloc`为cv流水线变换产生的多buffer存储。首个维度为槽位维度（每个流水线阶段一个槽位）；兄弟槽位可能被其他流水线阶段并发使用，因此任何由聚合的`hivm.hir.nd2nz`（或类似操作）派生的预初始化、pad操作必须只针对当前槽位tile（即驱动消费者操作的那个subview），绝不能针对整个alloc。

**参数说明**：不涉及。

### CacheModifierAttr

**功能**：定义缓存修改策略。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| policy | `::mlir::hivm::CacheModifier` | 类型为CacheModifier的枚举 |

### CompareModeAttr

**功能**：定义VCmpOp的比较模式。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::CompareMode` | 类型为CompareMode的枚举 |

### DCCIModeAttr

**功能**：定义DCCI模式。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::DCCIMode` | 类型为DCCIMode的枚举 |

### DataCacheKindAttr

**功能**：定义数据缓存类型。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::DataCacheKind` | 类型为DataCacheKind的枚举 |

### DataLayoutAttr

**功能**：定义HIVM数据布局映射属性，包括DOTA_ND、DOTB_ND、DOTC_ND、zN、nZ和ND，支持转置标识。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| data_layout | `::mlir::hivm::DataLayout` | 类型为DataLayout的枚举 |
| transpose | `std::optional<bool>` | 转置标识 |
| fractalSizes | `std::optional<DenseI64ArrayAttr>` | 分形尺寸 |

### DeinterleaveModeAttr

**功能**：定义解交织索引模式。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::DeinterleaveMode` | 类型为DeinterleaveMode的枚举 |

### DescaleModeAttr

**功能**：定义MatmulOp的反量化模式。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::DescaleMode` | 类型为DescaleMode的枚举 |

### DisableAutoInjectBlockSyncAttr

**功能**：禁用自动注入块同步。

**参数说明**：不涉及。

### EnableSavingUbAttr

**功能**：若模块带有该属性，表示尽可能使用内存占用更小的操作，例如enablevcg vector reduce。

**参数说明**：不涉及。

### EventAttr

**功能**：定义同步事件属性。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| event | `::mlir::hivm::EVENT` | 类型为EVENT的枚举 |

### EvictionPolicyAttr

**功能**：定义共享内存的逐出策略。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| policy | `::mlir::hivm::EvictionPolicy` | 类型为EvictionPolicy的枚举 |

### FixpipeDMAModeAttr

**功能**：定义HIVM Fixpipe DMA模式（int32_t存储）。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::FixpipeDMAMode` | 类型为FixpipeDMAMode的枚举 |

### FixpipeDualDstModeAttr

**功能**：HIVM fixpipe双目标模式控制。对于L0C中的M x N矩阵，支持以下模式：`NO_DUAL`（单目标模式，将整个矩阵写入SUB BLOCKID指示的目标UB）、`ROW_SPLIT`（沿M维拆分，每个UB写入M/2 x N，M必须为2的倍数）、`COLUMN_SPLIT`（沿N维拆分，每个UB写入M x N/2，N必须为32的倍数）。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| dual_dst_mode | `::mlir::hivm::FixpipeDualDstMode` | 类型为FixpipeDualDstMode的枚举 |

### FixpipePreQuantModeAttr

**功能**：定义Fixpipe前级量化模式。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::FixpipePreQuantMode` | 类型为FixpipePreQuantMode的枚举 |

### FixpipePreReluModeAttr

**功能**：定义Fixpipe前级ReLU模式。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::FixpipePreReluMode` | 类型为FixpipePreReluMode的枚举 |

### FixpipeSubBlockAttr

**功能**：定义HIVM Fixpipe目标AIV子块（0或1）。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::FixpipeSubBlock` | 类型为FixpipeSubBlock的枚举 |

### HIVMFuncDynMemrefArgsAttr

**功能**：标记函数中动态memref参数的索引数组。

**参数说明**：不涉及。

### HOST_BITCODEAttr

**功能**：定义HIVM HOST bitcode路径属性（字符串）。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| path | `StringAttr` | HOST bitcode路径 |

### HasAliaScopesAttr

**功能**：HIVM has alias scopes属性。

**参数说明**：不涉及。

### InlineModeAttr

**功能**：定义HIVM custom op内联模式。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::InlineMode` | 类型为InlineMode的枚举 |

### InsertSliceSourceIndexAttr

**功能**：指定VConcatOp中的插入片段的源操作数。

**参数说明**：不涉及。

### kInsertedCopyAttr

**功能**：标记插入的copy。

**参数说明**：不涉及。

### kInsertedFixpipeAttr

**功能**：标记插入的fixpipe。

**参数说明**：不涉及。

### kInsertedLoadAttr

**功能**：标记插入的load。

**参数说明**：不涉及。

### kInsertedStoreAttr

**功能**：标记插入的store。

**参数说明**：不涉及。

### kInsertedTensorAttr

**功能**：标记插入的tensor。

**参数说明**：不涉及。

### IsDistributedAttr

**功能**：指定操作与分布式相关。

**参数说明**：不涉及。

### IteratorTypeAttr

**功能**：定义HIVM结构化操作的迭代器类型。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::IteratorType` | 类型为IteratorType的枚举 |

### MIX_AIC_BITCODEAttr

**功能**：定义HIVM MIX AIC bitcode路径属性（字符串）。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| path | `StringAttr` | MIX AIC bitcode路径 |

### MIX_AIV_BITCODEAttr

**功能**：定义HIVM MIX AIV bitcode路径属性（字符串）。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| path | `StringAttr` | MIX AIV bitcode路径 |

### MatmulLimitedInCubeAttr

**功能**：表示matmul操作可以仅在AIC中执行，将绕过可能因硬件约束而把matmul分解为向量加法的编译器分析。

**参数说明**：不涉及。

### MaxPreloadNumAttr

**功能**：指定scope操作的最大preload数量。

**参数说明**：不涉及。

### HIVMMemoryUniqueAttr

**功能**：HIVM规划唯一内存。

**参数说明**：不涉及。

### MemoryEffectAttr

**功能**：定义用于simt vf的HIVM内存效应属性。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| effect | `::mlir::hivm::MemoryEffect` | 类型为MemoryEffect的枚举 |

### MultiBufferAttr

**功能**：定义多缓冲属性。

**参数说明**：不涉及。

### PadModeAttr

**功能**：定义填充模式属性。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| padmode | `::mlir::hivm::PadMode` | 类型为PadMode的枚举 |

### PaddingOptionAttr

**功能**：定义填充选项。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| option | `::mlir::hivm::PaddingOption` | 类型为PaddingOption的枚举 |

### ParallelLoopAttr

**功能**：标记可并行执行的循环。

**参数说明**：不涉及。

### PipeAttr

**功能**：定义操作所属的pipe。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| pipe | `::mlir::hivm::PIPE` | 类型为PIPE的枚举 |

### PreloadLocalBufferAttr

**功能**：指定相关操作为preload local buffer。

**参数说明**：不涉及。

### PreloadNumAttr

**功能**：指定scope操作的preload数量。

**参数说明**：不涉及。

### PreloadWorkspaceAttr

**功能**：指定相关操作为preload workspace。

**参数说明**：不涉及。

### ReduceOpAttr

**功能**：定义规约操作的算术类型。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| reduce_op | `::mlir::hivm::ReduceOperation` | 类型为ReduceOperation的枚举 |

### RemainInL0CAttr

**功能**：表示操作的结果应保留在L0C buffer中，由NormalizeMatmul pass用于buffer复用优化；整数参数指定应保留在L0C的结果索引。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| resultIndex | `unsigned` | 保留在L0C的结果索引 |

### RoundModeAttr

**功能**：定义VCastOp的舍入模式。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::RoundMode` | 类型为RoundMode的枚举 |

### SharedMemoryAttr

**功能**：定义用于simt vf的共享内存。

**参数说明**：不涉及。

### SimtMemScopeHintAttr

**功能**：在地址空间具现化之前，为SIMT VF参数提供内存scope hint。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| addressSpace | `::mlir::hivm::AddressSpace` | 类型为AddressSpace的枚举 |

### SkipStrideAlignForVLoadAttr

**功能**：标记一个alloc，其下游vload的buffer必须跳过stride对齐，因为由此产生的行间间隙会破坏vldsx1（128个元素连续flat-pointer加载）。

**参数说明**：不涉及。

### StorageAlignedAttr

**功能**：标记模块或函数内所有操作已对齐。

**参数说明**：不涉及。

### StrideAlignDimsAttr

**功能**：定义步长对齐维度信息。

**参数说明**：不涉及。

### StrideAlignValueInByteAttr

**功能**：定义按字节对齐的步长大小信息。

**参数说明**：不涉及。

### HIVMSubBlockMappingAttr

**功能**：定义混合函数中子块映射关系。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| sub_block | `::mlir::hivm::MappingId` | 类型为MappingId的枚举 |

### SyncBlockInstrModeAttr

**功能**：定义同步块指令模式。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| sync_instr_mode | `::mlir::hivm::SyncBlockInstrMode` | 类型为SyncBlockInstrMode的枚举 |

### SyncBlockLockOrderingAttr

**功能**：定义同步块锁操作的排序方式。选择有序令牌环锁或无序（Lamport bakery）锁，作用于sync_block_lock、sync_block_unlock、free_lock_var操作（默认为ordered）。旧IR可改用unit属性`hivm.sync_block_lock_unordered`，辅助函数会将其视为无序。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::SyncBlockLockOrdering` | 类型为SyncBlockLockOrdering的枚举 |

### SyncBlockLockUnorderedAttr

**功能**：标记应使用无序锁（Lamport bakery，按到达顺序）而非有序令牌环的create_sync_block_lock、sync_block_lock、sync_block_unlock、free_lock_var操作。

**参数说明**：不涉及。

### SyncBlockLockWithSubblockAttr

**功能**：标记在设备模板中使用subblock感知的块索引的sync_block_lock、sync_block_unlock和free_lock_var操作。

**参数说明**：不涉及。

### SyncBlockModeAttr

**功能**：定义同步块模式。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| sync_mode | `::mlir::hivm::SyncBlockMode` | 类型为SyncBlockMode的枚举 |

### SyncEventSlotAttr

**功能**：CustomMacroOp的一个`sync_event_slots`条目。

`macro_sync`为`internal`（默认）时，宏仅需通过`sync_related_args`传入一个已分配的事件ID，GraphSyncSolver不插入边界set_flag、wait_flag操作；为`wait`时，宏内核在（set_pipe, wait_pipe）上调用wait_flag，GraphSyncSolver在宏之前注入set_flag；为`set`时，宏在内部set，GraphSyncSolver在宏之后注入wait_flag。`wait`或`set`时`set_pipe`和`wait_pipe`必需，`internal`时可选。可选的event属性为该槽固定一个静态硬件事件ID。

**示例**：

```mlir
// internal (default): GSS allocates event id; pipes optional
#hivm.sync_event_slot<internal>
#hivm.sync_event_slot<#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_MTE1>>
// wait: macro waits on (set_pipe, wait_pipe); GSS may inject set_flag
#hivm.sync_event_slot<#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_MTE1>, wait>
// set: macro sets on (set_pipe, wait_pipe); GSS injects wait_flag after macro
#hivm.sync_event_slot<#hivm.pipe<PIPE_M>, #hivm.pipe<PIPE_MTE2>, set>
#hivm.sync_event_slot<#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_MTE1>, internal, <EVENT_ID1>>
```

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| set_pipe | `PipeAttr` | set_flag所在流水线 |
| wait_pipe | `PipeAttr` | wait_flag所在流水线 |
| macro_sync | `SyncEventSlotMacroSync` | 宏同步方式 |
| event | `EventAttr` | 静态硬件事件ID |

### TCoreRatioAttr

**功能**：mix内核上的CV配比。仅在MIX内核上有效，支持1:2、0:1、1:0、1:1和1:2。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| cube | `int32_t` | cube核配比 |
| vector | `int32_t` | vector核配比 |

### TCoreTypeAttr

**功能**：定义操作的核心类型。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| tcoretype | `::mlir::hivm::TCoreType` | 类型为TCoreType的枚举 |

### TCoreTypeMarkerAttr

**功能**：定义操作的核心类型标记。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| tcoretype | `::mlir::hivm::TCoreType` | 类型为TCoreType的枚举 |

### TFuncCoreTypeAttr

**功能**：定义函数的核心类型。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| funcCoreType | `::mlir::hivm::TFuncCoreType` | 类型为TFuncCoreType的枚举 |

### TModuleCoreTypeAttr

**功能**：定义模块的核心类型，根据模块内函数类型自动推断为AIC、AIV或MIX。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| moduleCoreType | `::mlir::hivm::TModuleCoreType` | 类型为TModuleCoreType的枚举 |

### TPartOfMixAttr

**功能**：标记函数是混合内核的一部分。

**参数说明**：不涉及。

### HIVMTightlyCoupledBufferAttr

**功能**：定义HIVM cv紧耦合buffer。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| id | `std::optional<int32_t>` | 标识 |

### TileMixCubeNumAttr

**功能**：指定mix内核中将被tiling的cube循环数量。

**参数说明**：不涉及。

### TypeFnAttr

**功能**：定义VCastOp的转换类型。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::TypeFn` | 类型为TypeFn的枚举 |

### UnitFlagAttr

**功能**：定义单元标志同步模式。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| unit_flag | `::mlir::hivm::UNIT_FLAG` | 类型为UNIT_FLAG的枚举 |

### UnlikelyConditionAttr

**功能**：标记条件分支大概率不成立。

**参数说明**：不涉及。

### UnsignedModeAttr

**功能**：表示无符号或有符号转换模式（VCastOp的无符号模式）。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::UnsignedMode` | 类型为UnsignedMode的枚举 |

### VectorFunctionAttr

**功能**：HIVM vector函数属性。

**参数说明**：不涉及。

### VFModeAttr

**功能**：定义向量单元运行模式。

| 参数 | C++ 类型 | 说明 |
| :--: | :------: | ---- |
| value | `::mlir::hivm::VFMode` | 类型为VFMode的枚举 |

## 枚举

### AddressSpace

**功能**：定义HIVM地址空间。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| Zero | 0 | zero |
| GM | 1 | gm |
| L1 | 2 | cbuf |
| L0A | 3 | ca |
| L0B | 4 | cb |
| L0C | 5 | cc |
| UB | 6 | ub |
| SSBUF | 11 | ssbuf |
| FixBUF | 7 | fixbuf |
| BiasBUF | 12 | biasbuf |

### AlignKind

**功能**：定义对齐方式信息。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| ALIGN | 0 | align |
| UNALIGNED | 1 | unaligned |
| UNKNOWN | 2 | unknown |

### AtomicKind

**功能**：定义StoreOp的原子操作类型。

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

**功能**：定义轴类型信息。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| FIRST | 0 | first |
| MIDDLE | 1 | middle |
| LAST | 2 | last |

### CacheModifier

**功能**：定义缓存修改策略。允许的32位无符号整数值：1、2、3、4、5、6、7。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| NONE | 1 | none |
| CA | 2 | ca |
| CG | 3 | cg |
| WB | 4 | wb |
| CS | 5 | cs |
| WT | 6 | wt |
| CV | 7 | cv |

### CompareMode

**功能**：定义VCmpOp的比较模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| EQ | 0 | eq |
| NE | 1 | ne |
| LT | 2 | lt |
| GT | 3 | gt |
| GE | 4 | ge |
| LE | 5 | le |

### DCCIMode

**功能**：定义DCCI模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| SINGLE_CACHE_LINE | 0 | single_cache_line |
| ALL_CACHE_LINES | 1 | all_cache_lines |

### DataCacheKind

**功能**：定义数据缓存类型。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| ALL | 0 | all |
| UB | 1 | ub |
| OUT | 2 | out |
| ATOMIC | 3 | atomic |

### DataLayout

**功能**：定义HIVM数据布局类型。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| DOTA_ND | 1 | dotA_ND |
| DOTB_ND | 2 | dotB_ND |
| DOTC_ND | 3 | dotC_ND |
| nZ | 4 | nZ |
| zN | 5 | zN |
| ND | 6 | ND |
| Fractal | 7 | Fractal |
| NCHW | 8 | NCHW |
| NC1HWC0 | 9 | NC1HWC0 |
| C1HWNC0 | 10 | C1HWNC0 |
| SCALEA_ND | 11 | SCALEA_ND |
| SCALEB_DN | 12 | SCALEB_DN |
| SCALEA_zZ | 13 | SCALEA_zZ |
| SCALEB_nN | 14 | SCALEB_nN |

### DeinterleaveMode

**功能**：定义解交织索引模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| CHANNEL_0 | 0 | CHANNEL_0 |
| CHANNEL_1 | 1 | CHANNEL_1 |
| ALL_CHANNELS | 999 | ALL_CHANNELS |

### DescaleMode

**功能**：定义MatmulOp的反量化模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| DescaleNull | 0 | DescaleNull |
| DescalePerChannel | 1 | DescalePerChannel |
| DescalePerTensor | 2 | DescalePerTensor |

### EVENT

**功能**：定义同步事件ID。

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

### EvictionPolicy

**功能**：定义共享内存的逐出策略。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| EvictNormal | 1 | EvictNormal |
| EvictFirst | 2 | EvictFirst |
| EvictLast | 3 | EvictLast |

### FixpipeDMAMode

**功能**：定义HIVM Fixpipe DMA模式（int32_t存储）。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| NZ2ND | 0 | nz2nd |
| NZ2DN | 1 | nz2dn |
| NZ2NZ | 2 | normal |

### FixpipeDualDstMode

**功能**：定义HIVM fixpipe双目标模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| NO_DUAL | 0 | NO_DUAL |
| ROW_SPLIT | 1 | ROW_SPLIT |
| COLUMN_SPLIT | 2 | COLUMN_SPLIT |

### FixpipePreQuantMode

**功能**：定义Fixpipe前级量化模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| NO_QUANT | 0 | NO_QUANT |
| S322I8 | 9 | S322I8 |
| F322F16 | 1 | F322F16 |
| QF322F32_PRE | 15 | QF322F32_PRE |
| F322BF16 | 16 | F322BF16 |

### FixpipePreReluMode

**功能**：定义Fixpipe前级ReLU模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| NO_RELU | 0 | NO_RELU |
| NORMAL_RELU | 1 | NORMAL_RELU |
| LEAKY_RELU | 2 | LEAKY_RELU |
| P_RELU | 3 | P_RELU |

### FixpipeSubBlock

**功能**：定义HIVM fixpipe目标AIV子块。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| SUB_BLOCK_0 | 0 | sub_block_0 |
| SUB_BLOCK_1 | 1 | sub_block_1 |

### InlineMode

**功能**：定义HIVM custom op内联模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| AlwaysInline | 0 | always_inline |
| NoInline | 1 | no_inline |

### IteratorType

**功能**：定义结构化的迭代器类型。

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

### HIVMMatmulDataformat

**功能**：定义HIVM matmul数据格式。允许的32位无符号整数值：1、2、3。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| FP8E5M2_T | 1 | fp8_e5m2_t |
| FP8E4M3_T | 2 | fp8_e4m3_t |
| FP4E2M1_T | 3 | fp4_e2m1_t |

### MatmulBiasMode

**功能**：定义局部MatmulOp的偏置模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| NoBias | 0 | NoBias |
| PerChannelAdd | 1 | PerChannelAdd |
| PerChannelAddWithSplitK | 2 | PerChannelAddWithSplitK |
| ElementwiseCrossLoopAdd | 4 | ElementwiseCrossLoopAdd |
| ElementwiseAdd | 3 | ElementwiseAdd |
| PostPerChannelAddWithSplitK | 5 | PostPerChannelAddWithSplitK |
| MMInitPerChannelAddWithSplitK | 6 | MMInitPerChannelAddWithSplitK |
| ZeroInitNoAccumulation | 7 | ZeroInitNoAccumulation |
| ReuseL0C | 8 | ReuseL0C |

### MemPlanMode

**功能**：定义内存规划模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| LOCAL_MEM_PLAN | 0 | LOCAL_MEM_PLAN |
| GLOBAL_WORKSPACE_PLAN | 1 | GLOBAL_WORKSPACE_PLAN |

### MemoryEffect

**功能**：定义HIVM内存效应。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| READ | 0 | read |
| WRITE | 1 | write |
| READ_WRITE | 2 | read_write |

### PadMode

**功能**：定义LoadOp的填充模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| PadNull | 0 | PadNull |
| PadFirstElem | 1 | PadFirstElem |
| PadValue | 2 | PadValue |

### PaddingOption

**功能**：定义填充选项。允许的32位无符号整数值：1、2。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| PAD_ZERO | 1 | zero |
| PAD_NAN | 2 | nan |

### PIPE

**功能**：定义HIVM操作所属的pipe。

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

**功能**：定义VReduceOp的规约操作类型。

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
| max_with_index | 9 | max_with_index |
| min_with_index | 10 | min_with_index |
| any | 11 | any |
| all | 12 | all |
| xori | 13 | xori |
| ori | 14 | ori |
| andi | 15 | andi |
| none | 0 | none |

### RoundMode

**功能**：定义VCastOp的舍入模式。

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

**功能**：定义同步块指令模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| INTER_BLOCK_SYNCHRONIZATION | 0 | INTER_BLOCK_SYNCHRONIZATION |
| INTER_SUBBLOCK_SYNCHRONIZATION | 1 | INTER_SUBBLOCK_SYNCHRONIZATION |
| INTRA_BLOCK_SYNCHRONIZATION | 2 | INTRA_BLOCK_SYNCHRONIZATION |

### SyncBlockLockOrdering

**功能**：定义同步块锁操作的排序方式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| Ordered | 0 | ordered |
| Unordered | 1 | unordered |

### SyncBlockLockPipelinePhase

**功能**：定义optimize-hivm中sync-block-lock流水线的阶段。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| Prepare | 0 | prepare |
| Finalize | 1 | finalize |

### SyncBlockMode

**功能**：定义同步块模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| ALL_CUBE | 0 | ALL_CUBE |
| ALL_VECTOR | 1 | ALL_VECTOR |
| ALL_SUB_VECTOR | 2 | ALL_SUB_VECTOR |
| BARRIER_CUBE | 3 | BARRIER_CUBE |
| BARRIER_VECTOR | 4 | BARRIER_VECTOR |
| ALL | 5 | ALL |

### SyncEventSlotMacroSync

**功能**：表示custom macro内核是wait、set还是仅需事件ID。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| wait | 0 | wait |
| set | 1 | set |
| internal | 2 | internal |

### TCoreType

**功能**：定义操作的核心类型。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| CUBE | 1 | CUBE |
| VECTOR | 2 | VECTOR |
| CUBE_OR_VECTOR | 3 | CUBE_OR_VECTOR |
| CUBE_AND_VECTOR | 4 | CUBE_AND_VECTOR |

### TFuncCoreType

**功能**：定义函数的核心类型。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| AIC | 1 | AIC |
| AIV | 2 | AIV |
| MIX | 3 | MIX |
| AIC_OR_AIV | 4 | AIC_OR_AIV |

### TModuleCoreType

**功能**：定义模块的核心类型。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| AIC | 1 | AIC |
| AIV | 2 | AIV |
| MIX | 3 | MIX |

### TypeFn

**功能**：定义VCastOp的转换类型。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| cast_signed | 0 | cast_signed |
| cast_unsigned | 1 | cast_unsigned |
| bitcast | 2 | bitcast |

### UNIT_FLAG

**功能**：定义单元标志同步模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| DISABLED | 0 | DISABLED |
| RESERVED | 1 | RESERVED |
| ENABLED_WITHOUT_UPDATE | 2 | ENABLED_WITHOUT_UPDATE |
| ENABLED_WITH_UPDATE | 3 | ENABLED_WITH_UPDATE |
| ENABLED_ONLY_LAST_ITER | 4 | ENABLED_ONLY_LAST_ITER |
| ENABLED_ONLY_FIRST_ITER | 5 | ENABLED_ONLY_FIRST_ITER |
| ENABLED_ONLY_FIRST_AND_LAST_ITERS | 6 | ENABLED_ONLY_FIRST_AND_LAST_ITERS |

### UnsignedMode

**功能**：定义VCastOp的无符号模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| SI2SI | 0 | si2si |
| SI2UI | 1 | si2ui |
| UI2SI | 2 | ui2si |
| UI2UI | 3 | ui2ui |

### VFMode

**功能**：定义向量单元运行模式。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| SIMD | 0 | SIMD |
| SIMT | 1 | SIMT |
| MIX | 2 | MIX |

### MappingId

**功能**：定义循环映射标识。

| 枚举符号 | 数值 | 标识字符串 |
| :------: | :--: | ---------- |
| DimX | 0 | x |
