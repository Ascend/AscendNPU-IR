# hivm方言Passes

## -arith-vector-mask-analyze

**功能**：分析arith、vector掩码，消除select算子。

## -auto-blockify-parallel-loop

**功能**：当逻辑块数量大于物理块数量时，在块维度启用自动循环。

## -auto-scope

**功能**：为gather_load与scatter_store创建scope。

## -compose-collapse-expand

**功能**：组合collapse与expand操作。

## -convert-non-contiguous-reshape-to-copy

**功能**：为非连续且可能重关联的reshape操作生成copy算子。

## -convert-to-hivm-op

**功能**：将其他方言的算子转换为HIVM算子。

将`memref.copy`、`bufferization.materialize_in_destination`转为`hivm.hir.load`、`hivm.hir.store`、`hivm.hir.copy`；传递pad值（`VBrc`或`pad_const`）、左填充（`offset % num_per_block`）、`eviction_policy`（默认`EvictFirst`），并保留master侧的`may_implicit_transpose_with_last_axis`等属性。

跳过Host函数；在`hivm.vector_function`中不对`memref.copy`做load、store转换（仍可转为`hivm.copy`）；在`scf.forall`中跳过`memref.copy`、`materialize_in_destination`；在`hivm.vector_function`中跳过`materialize_in_destination`。

## -create-preload

**功能**：为CV流水线创建preload。

## -cv-pipelining

**功能**：为多缓冲mix-cv操作的Cube与Vector核心开启流水线优化。

**选项**：

- `-enable-auto-balance`：在流水线执行期间启用向量子任务负载均衡。

## -enable-hivmc-compatible-print

**功能**：启用HIVMC版本兼容的IR打印。

启用HIVMC版本兼容IR打印后，`Operation::print(OpAsmPrinter)`会从moduleOp读取hivmc版本属性，并打印对应hivmc版本兼容的MLIR。

## -hivm-add-ffts-to-syncblocksetop

**功能**：将FFTS（arg0）添加到SyncBlockSetOp算子中。

## -hivm-aggregated-decompose-op

**功能**：分解实现了hivm AggregatedOpInterface接口的hivm算子。

**选项**：

- `-decompose-phase`：指定执行的分解阶段。

## -hivm-align-alloc-size

**功能**：自动对齐特殊hivm算子所需的memref.alloc内存大小，满足访问对齐要求。

部分特殊hivm算子的访问大小必须对齐到硬件单元粒度，该Pass会调整对应memref.alloc的分配大小，避免越界访问。

## -hivm-alloc-extra-buffer

**功能**：为有需求的算子分配额外的临时缓冲区。

## -hivm-annotate-vf-alias

**功能**：在VF内标注别名信息。

## -hivm-auto-infer-buffer-size

**功能**：自动推断缓冲区大小。

该Pass通过插入annotation.mark算子完成缓冲区大小的推断。

## -hivm-bind-sub-block

**功能**：执行分块处理并绑定子块。

## -hivm-bind-sync-block-lock-arg

**功能**：将携带hacc.syncblocklock属性的函数参数绑定到CreateSyncBlockLockOp算子。

## -hivm-bind-workspace-arg

**功能**：将携带hacc.workspace属性的函数参数绑定到AllocWorkspaceOp算子。

## -hivm-bubble-up-extract-slice

**功能**：上提extract-slice算子。

## -hivm-canonicalization-pipeline

**功能**：在单个函数上运行HIVM规范化（canonicalization）Pass流水线。

## -hivm-clone-tensor-empty

**功能**：根据hivm算子将输出克隆为不同的空张量。

该Pass会为hivm算子的输出克隆生成独立的tensor.empty。

## -hivm-combine-optimized-convert-layout

**功能**：折叠部分convert layout模式，并将其转换为优化形式。

## -hivm-constantize-buffer-size

**功能**：尝试将动态形状的缓冲区转换为常量形态。

该Pass通过对原始形状取上界，将动态形状缓冲区转换为静态常量。转换成功后会创建新的静态形状alloc，并通过subview映射回原始形状，供后续流程使用。

## -hivm-convert-layout-to-transpose

**功能**：将convert layout转换为transpose分解。

**选项**：

- `-use-3d-transpose`：对ND到Fractal的转换使用分阶段的3D transpose分解。

## -hivm-cross-core-gss

**功能**：跨核图同步求解。

**选项**：

- `-block-all-sync`：为HIVM injectBlockSync启用全量块同步注入。
- `-disable-auto-inject-block-sync`：切换自动set、wait插入逻辑，始终保留SetFFTSBaseAddrOp。
- `-always-use-pipe-s`：wait_flag操作始终使用标量流水线作为等待流水线。
- `-use-different-multibuffer-flag-ids`：多缓冲反向同步对使用不同的flag-id。
- `-enable-cv-patterns`：启用跨核同步的CV模式。
- `-solver-version`：选择SyncSolver实现。
- `-round-robin-event-ids`：在基于内存的架构上，当默认策略反复分配相同ID时，按程序顺序重试EventId分配。
- `-force-is-mem-based`：强制使用基于内存的AI核心架构模式。
- `-force-is-reg-based`：强制使用基于寄存器的AI核心架构模式。

## -hivm-decompose-op

**功能**：依据硬件能力将复合hivm算子拆解为多个基础hivm算子。

该Pass会根据硬件支持能力拆分复合算子，例如硬件不支持直接的f32转i8类型转换时，会拆解为f32转f16、f16转i8两步转换操作。

动态场景下，会为额外分配的缓冲区创建携带`buffer_size_in_byte`属性的`annotation.markOp`，该属性值与原始算子的src或dst操作数对应大小一致。

此外，会展开标量算子`arith.fptoui`（f32转i64），跳过`hivm.vector_function`内核；并在寄存器架构（`isRegBasedArch`）上跳过`reduce-init`播种逻辑。

## -hivm-delayed-cross-core-gss

**功能**：延迟式跨核GSS自动同步Pass。

在mix kernel拆分之后解析跨核同步。该Pass要求每个目标kernel组包含：

1. 一个备份mix函数，其anchor保留了拆分前IR的位置信息；
2. 对应的拆分后cube与vector函数。

对每一组这样的三元组，该Pass会移除已有的块同步操作，根据匹配的anchor位置重建基于区间的读写信息，运行延迟式跨核同步求解器，并将求解得到的同步操作物化回mix、cube与vector函数中。

**选项**：

- `-block-all-sync`：为HIVM injectBlockSync启用全量块同步注入。
- `-always-use-pipe-s`：wait_flag操作始终使用标量流水线作为等待流水线。
- `-use-different-multibuffer-flag-ids`：多缓冲反向同步对使用不同的flag-id。
- `-enable-cv-patterns`：启用跨核同步的CV模式。
- `-solver-version`：选择SyncSolver实现。
- `-force-is-mem-based`：强制使用基于内存的AI核心架构模式。
- `-force-is-reg-based`：强制使用基于寄存器的AI核心架构模式。

## -hivm-enable-multi-buffer

**功能**：为算子启用多缓冲优化。

对于标记了`hivm.multi_buffer`属性的算子，该Pass会为其开启多缓冲机制。

## -hivm-enable-stride-align

**功能**：依据步幅对齐标记调整memref的分配对齐方式。

该Pass会根据`storage_align`注解标记，重新分配memref内存以满足对齐要求。

## -hivm-flatten-module

**功能**：将嵌套的子module展平到单个顶层MLIR module中。

## -hivm-flatten-ops

**功能**：展平HIVM算子。

## -hivm-fuse-transpose-into-load

**功能**：借助DMA在线转置能力，将`linalg.transpose`融合进`hivm.hir.load`。

## -hivm-graph-sync-solver

**功能**：执行图同步求解处理。

**选项**：

- `-enable-unit-flag`：启用同步的单元标志模式。

## -hivm-hoist-tightly-coupled-alloc

**功能**：将yield出的tightly-coupled alloc上提到外层区域（Regbase支持）。

保证CV tightly-coupled buffer的multi-buffer锚点在AIC、AIV两侧一致。

## -hivm-infer-data-layout

**功能**：推断HIVM算子的数据布局。

## -hivm-infer-func-core-type

**功能**：推断每个函数对应的核心类型。

在Ascend 310B、V300上跳过mix-kernel分析并强制模块归属AIV域，同时为mmadL1上层父函数标记AIC。gather和scatter算子归类至Vector核心类型。

## -hivm-infer-mem-scope

**功能**：推断HIVM算子的内存作用域。

## -hivm-infer-vf-mode

**功能**：推断算子的VF模式。

## -hivm-init-entry-kernel

**功能**：在入口Kernel的起始位置插入set_mask_norm() 调用。

## -hivm-inject-block-sync

**功能**：自动注入块同步操作。

**选项**：

- `-block-all-sync`：为HIVM injectBlockSync启用全量块同步注入。
- `-assume-alive-loops`：假定所有循环（forOp、whileOp）至少执行一次。
- `-disable-auto-inject-block-sync`：切换自动set、wait插入逻辑，始终保留SetFFTSBaseAddrOp。

## -hivm-inject-sync

**功能**：自动注入同步操作。

**选项**：

- `-sync-mode`：同步注入模式，默认为正常注入模式。
- `-enable-unit-flag`：启用同步的单元标志模式。
- `-assume-alive-loops`：假定所有循环（forOp、whileOp）至少执行一次。

## -hivm-inline-fixpipe

**功能**：将算子转换为HIVM Fixpipe算子。

## -hivm-inline-load-copy

**功能**：内联复制类load操作。

## -hivm-inline-otf-broadcast

**功能**：内联OTF广播操作。

## -hivm-inline-otf-load-store

**功能**：即时内联Load与Store操作。

## -hivm-insert-anchors-and-backup

**功能**：插入anchor操作并备份mix函数。

预处理Pass，用于在mix函数中记录位置信息：

1. 递归遍历每个mix `func.func`的所有block，在每个原有非终结操作之前以及每个block终结操作之前各插入一个`hivm.anchor`。anchor在每个函数内分配单调递增的`id`。
2. 将每个处理过的函数克隆为带有专用备份属性的备份函数。启用`strip-body`时，会从备份中移除所有非anchor、非控制流的工作算子，并将其存活结果替换为`ub.poison`，为后续分析保留结构骨架。

其他Pass应跳过被标记为备份的函数。

**选项**：

- `-cleanup`：移除该Pass此前生成的anchor与备份函数。
- `-insert-anchor-ops-before-all`：在所有操作之前插入anchor；未启用时，仅在可能涉及跨核内存的操作之前插入。
- `-insert-anchor-ops-before-mem-effect-ops`：在具有内存效应的操作之前插入anchor。
- `-insert-anchor-only-before-cube-ops`：仅在cube算子之前插入anchor。
- `-insert-anchor-only-before-vector-ops`：仅在vector算子之前插入anchor。
- `-insert-anchor-before-cube-and-vector-ops`：仅在cube与vector算子之前插入anchor。

## -hivm-insert-convert-layout

**功能**：为matmul操作插入convert layout。

## -hivm-insert-cv-tight-coupled-buffer

**功能**：为mix cv插入CV紧耦合缓冲区。

**选项**：

- `-only-insert-tightly-coupled-buffer`：仅插入紧耦合缓冲区。

## -hivm-insert-fixpipe

**功能**：插入HIVM Fixpipe算子。

## -hivm-insert-free-lock-var-before-return

**功能**：在return之前插入FreeLockVarOp，防止控制流跳过sync_block_lock或sync_block_unlock时发生死锁。

当SyncBlockLock与SyncBlockUnlock位于条件分支（如else块）中时，部分块可能不会执行它们，导致lock_var计数不匹配并引发死锁。该Pass在每个return之前插入FreeLockVarOp，每个FreeLockVarOp会降级为一对sync_block_lock与sync_block_unlock。

## -hivm-insert-infer-sync-block-lock-num-and-init-func

**功能**：为主机端插入同步块锁数量推断与初始化的回调函数。

该Pass会统计全部静态同步块锁的数量并完成初始化，同时创建主机回调函数返回该数值。

## -hivm-insert-infer-task-type-func

**功能**：推断模块的任务类型，并生成对应的主机端返回函数。

该Pass会检测模块类型，包括CubeVectorMix、CubeOnly、VectorOnly、Unknown四类，同时生成主机端函数`<original_func>_infer_task_type_function`，返回编码了任务类型的i8常量，并标注对应的HACC主机函数属性。

## -hivm-insert-infer-workspace-size-func

**功能**：为主机端插入工作空间大小推断的回调函数。

该Pass会在plan-workspace Pass执行后统计总静态工作空间大小，并创建主机回调函数返回该数值。

## -hivm-insert-init-and-finish-for-debug

**功能**：插入调试用的init与finish操作。

## -hivm-insert-l12ub-for-debug

**功能**：插入调试用的l12ub操作。

## -hivm-insert-load-store-for-mix-cv

**功能**：为mix cv场景插入load与store操作。

## -hivm-insert-load-store-for-scalar

**功能**：为跨越vector与cube核心的标量值插入显式的load、store边界。

## -hivm-insert-nz2nd-for-debug

**功能**：插入调试用的nz2nd操作。

## -hivm-insert-vf-mode-func

**功能**：为主机端插入VF模式推断的回调函数。

推断函数的VF模式（SIMD、SIMT、MIX），并创建主机回调函数返回该VF模式。

## -hivm-lift-lowest-stride

**功能**：抬升hivm操作数的最低维步幅。

对于绝大多数hivm结构化算子，若最后一个维度不连续，则会抬升操作数的最低维步幅，使最后一维变为连续。例外算子：MacroOp、VArangeOp。

例如：类型为`memref<16xf16, strided<[8]>>`的操作数，经过该Pass处理后变为`memref<16x1xf32, strided<[8, 1]>>`，最后一维满足连续要求。

## -hivm-lift-zero-rank

## -hivm-lower-create-sync-block-lock

**功能**：将CreateSyncBlockLockOp降级为ViewOp。

## -hivm-lower-multi-buffer-counter

**功能**：将HIVM多缓冲计数anchor降级为memref状态。

该Pass将`hivm.hir.multi_buffer_counter`降级为多缓冲槽位轮换所使用的、基于alloca的具体计数器表示。

## -hivm-lower-to-loops

**功能**：将hivm算子降级为循环实现。

将选定的HIVM算子降级为标量`scf.for`嵌套。触发降级的判定条件包含：数据类型为i64、硬件不兼容标量操作数、向量函数模式为SIMT（`hivm.vf_mode = SIMT`）；针对寄存器架构，仅对几何不合法的`index-reduce`算子执行降级。经SIMT降级后的循环会打上`map_for_to_forall`标记。

## -hivm-map-forall-to-blocks

**功能**：将forall循环映射到hivm块。

该Pass将每个scf.forall算子一对一映射到HIVM块算子，并将scf.forall的归纳变量重写为hivm block idx算子。

## -hivm-mark-disable-load

**功能**：标记需要禁用dcache的memref.load操作。

## -hivm-mark-multi-buffer

**功能**：为HIVM算子标记多缓冲属性。

当enable-auto选项开启时，该Pass会自动为hivm算子标记多缓冲属性；作用域为L0C的缓冲区不会被标记。若enable-auto关闭，则不执行任何操作。

**选项**：

- `-enable-auto`：自动标记多缓冲。
- `-limit-auto-multi-buffer-only-for-local-buffer`：禁用工作空间的多缓冲标记。
- `-limit-auto-multi-buffer-of-local-buffer`：限制本地缓冲区的自动多缓冲行为。
- `-limit-mix-auto-multi-buffer-buffer`：在cube、vector上禁用多缓冲或不做限制。
- `-set-workspace-multibuffer`：覆盖工作空间的多缓冲数量配置。

## -hivm-mark-real-core-type

**功能**：使用core-type属性标记标量算子。

**选项**：

- `-remove-core-type-attrs`：移除所有核心类型属性，开启后该Pass变为清理Pass。

## -hivm-mark-stride-align

**功能**：自动为hivm算子的操作数标注stride_align标记。

该Pass会遍历所有hivm算子，为其memref类型操作数自动添加storage_align注解标记。

## -hivm-mark-sync-block-lock-with-subblock

**功能**：在mix模块中，为不在`limit_sub_block_id0`条件if内的sync_block_lock与sync_block_unlock算子打上`sync_block_lock_with_subblock`标记。

当模块为mix类型时，该Pass会标记那些不在带`limit_sub_block_id0`属性的`scf.if`内的sync_block_lock与sync_block_unlock算子。被标记的算子会获得`sync_block_lock_with_subblock`属性，表示其应使用基于subblockid的block_idx计算方式（`get_block_idx * get_subblocknum + get_subblockid`）。

## -hivm-mark-tightly-coupled-buffer

**功能**：为L1、UB alloc标记tightly-coupled-buffer id（RegBase支持）。

在`-hivm-split-mix-kernel`之前于MIX函数上分配id，使AIC、AIV克隆继承相同id。

## -hivm-memref-alloc-to-alloca

**功能**：将局部AllocOp转换为AllocaOp。

该Pass会将所有非全局内存空间的memref.alloc替换为memref.alloca。

## -hivm-normalize-bitwise-select

**功能**：规范化hivm位运算select算子。

## -hivm-normalize-convops

**功能**：规范化hivm卷积算子。

## -hivm-normalize-matmul

**功能**：规范化hivm矩阵乘算子。

## -hivm-normalize-ops

**功能**：规范化HIVM。

**选项**：

- `-enable-high-precision`：在HIVM中启用sin、cos的高精度计算。

## -hivm-normalize-to-tensor

**功能**：规范化`bufferization.to_tensor`算子。

在bufferization之前对`bufferization.to_tensor`算子进行规范化处理：

1. 当memref带有hivm地址空间属性时，在`to_tensor`之前插入`memref.memory_space_cast`，将memref操作数携带的hivm内存空间信息隔离开，避免地址空间属性进入张量bufferization路径。
2. 为toTensorOp添加copyOp并替换其所有使用者，将memref级别的写操作（如`hivm.hir.load`）暴露给张量级别的分析。这使memref级别的写对One-Shot Bufferization分析可见，从而支持RAW冲突检测与正确的copy插入。

## -hivm-opt-func-output

**功能**：在bufferization完成后优化函数输出。

该Pass会尝试移除不必要的地址返回操作。

## -hivm-opt-single-point

**功能**：通过标量运算优化单点hivm算子。

该Pass借助标量操作完成单点hivm算子的优化，包括`vdiv`和无符号`vmax`、`vmin`等逐元素算子。

## -hivm-outline-alloc-in-VF

**功能**：将VF中静态形状的memref.alloc外提（outline）。

## -hivm-outline-copy-in-VF

**功能**：当操作数为VF参数时，将VF中的`hivm.load`改写为`hivm.copy`。

## -hivm-partition-and-bind-sub-block

**功能**：将`{sub_block = n}`的`scope.scope`区域降级为操作数并行的守卫（guard）。

**选项**：

- `-pin-free-nodes`：仅用于测试：将所有自由算子固定到sub-block 0，而不是在两个AIV子核之间做负载均衡。

## -hivm-plan-memory

**功能**：为HIVM算子执行内存规划。

**选项**：

- `-mem-plan-mode`：内存规划模式，默认为LOCAL_MEM_PLAN。
- `-enable-global-workspace-reuse`：启用全局工作空间复用，默认关闭。
- `-restrict-inplace-as-isa`：限制内存就地操作与ISA保持一致，默认关闭。

## -hivm-plan-memory-regbase

**功能**：在A5上为HIVM算子执行内存规划。

**选项**：

- `-mem-plan-mode`：内存规划模式，默认为LOCAL_MEM_PLAN。
- `-enable-global-workspace-reuse`：启用全局工作空间复用，默认关闭。
- `-enable-print-memory-allocated-size`：打印已分配内存大小，默认关闭。
- `-restrict-inplace-as-isa`：限制内存就地操作与ISA保持一致，默认关闭。
- `-simt-vf-dynamic-size`：SIMT VF的动态UB大小（KB），默认216。
- `-disable-tightly-coupled-buffer-reuse`：禁用紧耦合缓冲区复用，默认关闭。
- `-disable-vf-reachable-check`：禁用VF可达性检查，默认关闭。
- `-plan-memory-strategy`：存储条目重排序的内存规划策略。

## -hivm-pre-mark-stride-align

**功能**：步幅对齐预分析，标记必须跳过对齐处理的alloc。

该Pass会遍历所有hivm算子，为其memref类型操作数添加无需对齐的注解标记。

## -hivm-propagate-convert-layout

**功能**：为matmul操作传播convert layout。

**选项**：

- `-allow-agnostic-ops`：允许布局无关算子传播reshape。当前默认禁用，因为其他Pass尚不支持fractal布局操作。
- `-enable-elementwise-propagate`：启用逐元素convert layout传播模式。

## -hivm-recognize-deinterleave-op

**功能**：识别解交错算子，优化非连续内存访问。

该Pass通过解交错方式优化非连续的内存访问行为。

## -hivm-recognize-discontinuous-store

**功能**：为UB到GM的store构造32B对齐视图。

该Pass通过expand_shape+VBrcOp+subview+collapse的组合，为最后一维连续的UB源操作数构造32B对齐视图，使store能够以硬件可映射的最后一维步幅写入最后一维不连续的GM目标。

## -hivm-reduce-rank-subview

**功能**：通过subview实现降维处理。

## -hivm-remove-copy-ops

**功能**：移除VF函数中冗余的copy操作。

## -hivm-remove-layout-annotation

**功能**：移除HIVM数据布局注解。

## -hivm-set-buffer-size

## -hivm-simt-vf-sub-tiling

**功能**：将过大的SIMT VF分块拆分为串行执行的更小分块。

**选项**：

- `-max-tile-size`：所选SIMT分块维度上每个分块的最大尺寸。

## -hivm-sink-op-to-consumer-in-loop

**功能**：将算子下沉至循环内的消费者处。

## -hivm-split-mix-kernel

**功能**：将Mix设备函数拆分为AICube与AIVector两个独立函数。

该Pass会将mix kernel拆分为独立的AICube kernel和AIVector kernel，并将父模块标记为Mix模块。

**注意事项**：

- 若在主机函数内调用Mix kernel，会为最终的Kernel启动生成函数声明；当前不支持在设备函数内调用Mix kernel。
- 若存在混合核`scf.if`，请先显式运行`-hivm-split-mixed-if-conditionals`（独立Pass，不在默认pipeline中）。
- 在Regbase芯片上，还需先运行`-hivm-mark-tightly-coupled-buffer`、`-hivm-hoist-tightly-coupled-alloc`。

**转换示例**：

转换前：

```mlir
func (workspace) attribute {tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>} {
  t = cube_op ins() outs(workspace)
  ... = vector_op ins(t) ...
}
```

转换后：

```mlir
func (workspace) attribute {tcore_type = #hivm.tcore_type<CUBE>} {
  t = cube_op ins() outs(workspace)
  annotation.mark t // mark to avoid dce
}

func (workspace) attribute {tcore_type = #hivm.tcore_type<VECTOR>} {
  ... = vector_op ins(workspace) ...
}
```

## -hivm-split-mixed-if-conditionals

**功能**：将混合核类型的`scf.if`拆分为按核类型的if链。

与架构无关的独立Pass（不在默认HIVM pipeline中）。当MIX kernel含混合核`scf.if`时，请在`-hivm-split-mix-kernel`之前显式运行；结果会带`hivm.cube_only`或`hivm.vec_only`标记。

## -hivm-sub-block-guard-cleanup

**功能**：清理带返回值的操作数并行sub-block守卫。

操作数并行sub-block划分的第3阶段，在bufferization与DropEquivalentBufferResults之后运行。通过三个单一职责的步骤清理每个带返回值的`scf.if (get_sub_block_idx() == n) -> memref { ...; yield %out } else { yield %init }`：

1. 将仅含元数据的memref视图（expand_shape、collapse_shape、cast）从then分支的yield中上提，使then输出为裸alloc（此前是独立的pre-bufferize Pass）；
2. 折叠bufferization在then块中留下的冗余UB到UB copy，包括scope输出端的协调copy与只读输入端的暂存copy；
3. 将then分支自身的输出alloc上提到守卫之上并丢弃else分支，使守卫不再携带返回值，死掉的else-init alloc随后被DCE消除。

守卫通过`get_sub_block_idx() == const`谓词进行结构化匹配（bufferization会丢弃可丢弃属性，但会保留该索引域条件）；此处只有划分产生的守卫仍携带返回值。

## -hivm-sync-block-hoisting

**功能**：若syncblock的lock与unlock操作位于scf.for或scf.while循环内，则将其上提到父区域。

## -hivm-tensor-copy-insertion

**功能**：运行One-Shot分析并插入张量copy以解决bufferization冲突。

该Pass运行One-Shot Bufferize分析（不执行实际bufferization），并在检测到冲突（如RAW）的位置插入`bufferization.alloc_tensor`副本。它等价于`one-shot-bufferize`的analyze + insertTensorCopies阶段，但不含最终的bufferization步骤。

在`one-shot-bufferize`（使用`copy-before-write`或默认的analyze+copy路径）之前运行该Pass，可以让第二轮分析捕获第一轮copy插入所暴露的冲突。

**选项**：

- `-allow-return-allocs-from-loops`：允许从循环中return或yield新的分配。
- `-allow-unknown-ops`：允许输入IR中包含未知（不可bufferize）的算子。
- `-analysis-heuristic`：控制分析期间IR遍历的启发式策略。
- `-bufferize-function-boundaries`：对函数边界执行bufferize（实验性）。

## -hivm-tile-batchmm-into-loop

**功能**：将批量矩阵乘分块，生成在批处理维度上迭代的循环。

## -hivm-vectorize-ops

**功能**：利用VL打包对向量函数中的HIVM算子执行向量化。

## -hivm-vf-operand-substitution

**功能**：通过调用点操作数替换，将VF输入缓冲区复用为其输出。

## -infer-simt-vf-memory-effect

**功能**：推断SIMT VF函数参数的内存效应。

## -infer-simt-vf-memory-scope-hint

**功能**：推断SIMT VF参数的内存作用域提示。

## -insert-alloc-base-placeholder

**功能**：插入带有占位尺寸的`memref.alloc`占位符。

## -insert-memory-semantic-for-simtvf

**功能**：为SIMT VF插入内存语义。

## -insert-workspace-for-mix-cv

**功能**：为mix cv场景插入工作空间。

## -legalize-bool-for-simtvf

**功能**：合法化SIMT VF中使用的bool值。

## -mark-simt-scope-no-inline

**功能**：为SIMT scope标记no_inline，使inline-scope保持其独立不内联。

SIMT scope会被SIMD、SIMT混合流水线降级为独立的SIMT向量函数，必须保持独立。该Pass为每个`vector_mode = "simt"`的scope标记`no_inline`，防止通用的inline-scope Pass将其合并回调用者。

## -materialize-simt-vf-memory-scope

**功能**：在拆分后的SIMT模块内物化显式的内存作用域。

## -split-simt-module

**功能**：为每个SIMT VF拆分SIMT模块。

## -tile-cube-vector-loop

**功能**：在本地缓冲区上对cube与vector循环执行分块。

该Pass会在本地缓冲区维度对cube和vector操作做二次分块，主要目的包括：

1. 减少高开销的核间同步次数。
2. 支持更大的分块尺寸。

**选项**：

- `-tile-mix-vector-loop`：mix kernel分块后向量循环的迭代次数。
- `-tile-mix-cube-loop`：mix kernel分块后cube循环的迭代次数。

## -triton-global-kernel-args-to-hivm-op

**功能**： 将 Triton 全局 kernel 的 program_id 参数转换为 HIVM 操作

## -write-back-shared

**功能**：为SIMT VF中表示已用共享内存的`memref.alloc`设置实际大小。
