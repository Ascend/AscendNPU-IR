# 自动同步

Auto-Sync是AscendNPU IR（HIVM）编译器的自动同步插入功能，用于自动为共享数据或资源的生产者与消费者插入同步操作，确保正确的执行顺序。设计目标：正确性（避免数据竞争和顺序错误）与最小开销（仅插入必要的同步，安全时复用硬件事件）。

## 硬件背景

**AICore架构**：

AICore架构详情可查阅官方文档：[基本架构](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html)。

**HIVM同步操作说明**：

同步操作定义于`bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td`。以下从MLIR使用角度（操作数/属性）进行描述，而非汇编语法。

**核内同步（Normal-Sync）**：

- `hivm.set_flag`

  操作数/属性：`set_pipe`、`wait_pipe`、事件ID（`static_event_id` 和/或 `dynamic_event_id`）

  运行逻辑：在`set_pipe`上执行，等该pipe上所有前序指令执行完毕后再执行。执行时触发对应事件ID。

- `hivm.wait_flag`

  操作数/属性：`set_pipe`、`wait_pipe`、事件ID（`static_event_id` 和/或 `dynamic_event_id`）

  运行逻辑：在`wait_pipe`上执行。阻塞其后所有指令，直到匹配的事件ID被触发。

- `hivm.pipe_barrier`

  操作数/属性：`pipe`

  运行逻辑：对指定pipe的屏障操作。阻塞`pipe`上的所有后续指令，直到该pipe上所有前序指令执行完毕。

**跨核同步（Block-Sync，块内）**：

- `hivm.sync_block_set`

  操作数/属性：`tcore_type`、`tpipe`、`pipe`、flag ID（`static_flag_id` 和/或 `dynamic_flag_id`）、可选`ffts_base_addr`、`tsync_instr_mode`

  参数说明：

    - `tcore_type`：目标核类型（vector/cube）
    - `tpipe`、`pipe`：目标核上的set/wait pipe
    - flag ID：静态和/或动态flag
    - `ffts_base_addr`：内存型架构（如Ascend910B）上通常需要
    - `tsync_instr_mode`：默认值为`INTRA_BLOCK_SYNCHRONIZATION`

  执行逻辑：在目标核的`tpipe`（set pipe）上执行，需等该pipe上所有前序指令执行完毕后再执行。执行完成后设置对应flag ID。

- `hivm.sync_block_wait`

  操作数/属性：`tcore_type`、`tpipe`、`pipe`、flag ID（`static_flag_id` 和/或 `dynamic_flag_id`）、`tsync_instr_mode`

  参数说明：

    - `tcore_type`：目标核类型（vector/cube）
    - `tpipe`、`pipe`：目标核上的set/wait pipe
    - `tsync_instr_mode`：默认值为`INTRA_BLOCK_SYNCHRONIZATION`

  执行逻辑：在目标核`tcore_type`的`pipe`（wait pipe）上执行，阻塞该pipe上所有后续指令，直到匹配的flag ID被触发。

部分流程还会用到（不一定由auto-sync直接生成）：`hivm.sync_block`（多模式块屏障）、`hivm.anchor`（延迟跨核分析用的位置标记），以及sync-block lock/unlock相关辅助操作。

## 算法原理

### AutoSync

代码库提供两类Auto-Sync解决方案。管线通过编译选项在二者之间选择（见[接口说明](#接口说明)）。

- `GraphSyncSolver` / `CrossCoreGSS` / `DelayedCrossCoreGSS`（默认方案）

  使用基于图的算法分析冲突、选择set/wait（或barrier）对、分配event/flag ID，并生成HIVM同步操作。这是**默认**路径（`--enable-hivm-graph-sync-solver`与`--enable-hivm-cross-core-gss`默认均为`true`）。在RegBase上，延迟跨核GSS默认也会启用。

- `InjectSync` / `InjectBlockSync`（回退方案）

  使用多个Pass插入所需同步、移动/删除冗余同步，并通过活跃性分析分配flag ID/event ID。在禁用图同步，或barrier-all / block-all调试模式强制走inject路径时使用。

在Triton-Ascend中，也可通过`sync_solver=True`选择图同步求解器路径。

### GraphSyncSolver

![alt text](../../../images/developer_guide/auto_sync1.png)

**目的**：主要的核内自动同步方案。使用基于图的算法判断何时插入set/wait成对操作（或pipe barrier），并分配event IDs。支持unit-flag模式以及SyncSolver V1/V2。

**源码**：

- 头文件：`bishengir/include/bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/`
- 实现：`bishengir/lib/Dialect/HIVM/Transforms/GraphSyncSolver/`
  （`GraphSyncSolver.cpp`、`SyncSolverBase.cpp`、`SyncSolverV1.cpp`、`SyncSolverV2.cpp`、`SyncSolverIR.cpp`、`SyncSolverIRTranslator.cpp`、`SyncSolverCodeGen.cpp`、`GraphSolver.cpp`、`GraphSolverBase.cpp`、`GraphSolverUnitFlag.cpp`、`EventIdSolver.cpp`、`MemInfo.cpp`、`CustomMacroSync.cpp`、`Utility.cpp`，以及tester相关文件）

**执行阶段**：

1. IRTranslator：从输入函数构建Sync-IR（函数、作用域、循环、条件、读写操作）。
2. Solver（`SyncSolverV1` / `SyncSolverV2`，默认**v2**）：收集冲突对（生产者–消费者），在图可达性模型下进行对选择与排序，分配/复用event IDs；可选启用unit-flag与custom-macro预留。
3. CodeGenerator：生成`hivm.set_flag` / `hivm.wait_flag` / `hivm.pipe_barrier`。

### CrossCoreGSS

**目的**：为MIX内核（cube和vector）插入块级（块内）跨核同步操作：`sync_block_set`、`sync_block_wait`（特殊模式下也可能生成相关block sync形态）。

**源码**：`CrossCoreGSS.cpp`；复用GraphSyncSolver中的`IRTranslator`、`SyncSolver`和`CodeGenerator`。

**工作原理**：

- 与核内GSS相同的求解栈，配置为`CROSS_CORE_SYNC`。
- 仅在MIX内核上运行（非Host、非纯AIC/AIV）。
- 在内存型架构上，当内核参数存在FFTS基址时插入`SetFFTSBaseAddrOp`。
- 支持CV pattern、multibuffer flag-id策略、内存型架构上的round-robin event-id重试，以及block-all模式。

### DelayedCrossCoreGSS

**目的**（RegBase管线）：在mix内核拆分**之后**再求解跨核同步；通过anchor与备份mix函数，使位置相关的读写信息在cube/vector拆分后仍然可用。

**源码**：`DelayedCrossCoreGSS.cpp`；配套Pass为`InsertAnchorsAndBackup`（`InsertAnchorsAndBackup.cpp`，Pass名`hivm-insert-anchors-and-backup`）。

**工作原理**：

1. **Step 1**（拆分前）：运行CrossCoreGSS（通常关闭CV pattern），再执行`InsertAnchorsAndBackup`插入`hivm.anchor`并克隆备份mix函数。
2. **Step 2**（拆分后）：`DelayedCrossCoreGSS`匹配备份mix与拆分后的cube/vector函数，清除旧的块内同步，基于anchor重建区间读写信息，求解并将同步物化回mix/cube/vector；最后清理anchor与备份函数。

当`--enable-hivm-cross-core-gss`与`--enable-hivm-delayed-cross-core-gss`同时为true时启用（RegBase编译面上二者默认均为`true`）。

### InjectSync

![alt text](../../../images/developer_guide/auto_sync0.png)

**目的**：回退的核内同步方案。使用内存依赖分析、同步分析、event-id分配以及冗余同步清理（移动/删除冗余同步）插入`set_flag` / `wait_flag` / `pipe_barrier`。

**源码**：

- 头文件：`bishengir/include/bishengir/Dialect/HIVM/Transforms/InjectSync/`
- 实现：`bishengir/lib/Dialect/HIVM/Transforms/InjectSync/`
  （`InjectSync.cpp`、`MemoryDependentAnalyzer.cpp`、`SyncAnalysis.cpp`、`SyncEventIdAllocation.cpp`、`IRTranslator.cpp`、`SyncCodegen.cpp`、`MoveSyncState.cpp`、`RemoveRedundantSync.cpp`、`SyncCommon.cpp`、`SyncDebug.cpp`）

**执行阶段**：

1. IRTranslator：从输入函数构建Sync-IR（复合元素、循环、条件、内存操作）。
2. SyncAnalyzer：对每对冲突操作，插入一对set_flag/wait_flag；若两操作属于同一pipe，则插入`pipe_barrier`。
3. MoveSyncState：在保持语义的前提下，重新定位同步操作以减少停顿。
4. RemoveRedundantSync：删除冗余同步对。
5. SyncEventIdAllocation：分配静态或动态event IDs；在安全时进行复用。
6. SyncCodegen：生成`hivm.set_flag` / `hivm.wait_flag` / `hivm.pipe_barrier`。

barrier-all调试模式（`--enable-hivm-inject-barrier-all-sync`）会在相关内存效应操作前插入`pipe_barrier(PIPE_ALL)`，而不是走正常分析路径。

### InjectBlockSync

**目的**：回退的块级（块内）跨核同步方案，为MIX内核插入`sync_block_set`、`sync_block_wait`（block-all模式下为`sync_block`）。

**源码**：`bishengir/lib/Dialect/HIVM/Transforms/InjectBlockSync.cpp`、`bishengir/include/bishengir/Dialect/HIVM/Transforms/InjectBlockSync.h`

**行为**：

- 仅在MIX内核（非Host、非纯AIC/AIV）上运行。
- 当内核参数中存在FFTS基址时，插入`SetFFTSBaseAddrOp`（即使禁用自动插入set/wait，仍会执行该步骤）。
- 模式（由选项和融合类型控制）：
    - InjectAllBlockSync：在相关交接处插入块同步（`--enable-hivm-inject-block-all-sync`）。
    - InjectBlockShallowSync：面向`ShallowCV`融合，在matmul / mix-matmul / call等位置插入同步。
    - InjectBlockMixSync：面向完整mix场景，通过`SyncBlockIRTranslator`构建块同步IR，然后依次执行SyncAnalyzer（`BLOCKSYNC`）、MoveSyncState、RemoveRedundantSync、SyncEventIdAllocation、SyncCodegen。

## 接口说明

**命令行选项**：

这些通常在编译器驱动中接入（例如`bishengir-compile`）；具体映射关系请参见`bishengir/include/bishengir/Tools/bishengir-compile/Options.td`、HIVM `Passes.td`，以及`bishengir/lib/Dialect/HIVM/Pipelines/`下的管线实现。

| 标志 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--enable-hivm-graph-sync-solver` | bool | true | 使用GraphSyncSolver替代InjectSync进行核内自动同步 |
| `--enable-hivm-cross-core-gss` | bool | true | 使用CrossCoreGSS（或DelayedCrossCoreGSS）替代InjectBlockSync进行跨核自动同步 |
| `--enable-hivm-delayed-cross-core-gss` | bool | true | 在RegBase上，当跨核GSS启用时走延迟跨核GSS（anchors + 拆分后求解） |
| `--hivm-sync-solver-version` | string | v2 | 选择图同步求解器实现版本（`v1`或`v2`） |
| `--disable-auto-inject-block-sync` | bool | false | 禁用自动块级set/wait插入（InjectBlockSync / CrossCoreGSS / DelayedCrossCoreGSS）；FFTS基址设置仍可能执行 |
| `--disable-hivm-auto-inject-sync` | bool | false | 完全禁用核内自动同步（GraphSyncSolver与InjectSync） |
| `--enable-hivm-inject-barrier-all-sync` | bool | false | 强制InjectSync走barrier-all模式（同时覆盖GraphSyncSolver选择）；用于诊断自动同步失败 |
| `--enable-hivm-inject-block-all-sync` | bool | false | 强制块同步走block-all插入（InjectBlockSync / CrossCoreGSS / DelayedCrossCoreGSS） |
| `--enable-hivm-unit-flag-sync` | bool | false* | 对支持的操作启用unit-flag同步。*在Ascend950/RegBase上，除非显式指定该标志，否则默认启用 |
| `--enable-hivm-assume-alive-loops` | bool | false | 假设`for`/`while`循环至少执行一次（影响InjectSync / InjectBlockSync分析） |

管线选择摘要：

- 核内：若启用`--enable-hivm-graph-sync-solver`且非barrier-all，则使用GraphSyncSolver；否则使用InjectSync（除非设置`--disable-hivm-auto-inject-sync`）。
- 跨核：若禁用块同步则跳过；若禁用跨核GSS则使用InjectBlockSync；若启用延迟GSS（RegBase）则走DelayedCrossCoreGSS流程；否则使用CrossCoreGSS。

## 使用约束

- **硬件顺序模型**：Auto-Sync通过插入HIVM同步操作（`hivm.set_flag` / `hivm.wait_flag`、`hivm.pipe_barrier`，以及（如适用）`hivm.sync_block_set` / `hivm.sync_block_wait` / `hivm.sync_block`）来组织执行顺序。该顺序用cores与pipes，以及event/flag id来表示。
- **正确性基于可行性校验**：对于求解器流程，只有当候选同步约束在基于图的可达性/顺序模型下仍保持可行时才会接受（避免死锁或过度约束导致调度失败）。
- **块级同步覆盖范围**：跨核块级同步面向MIX内核（cube/vector handoff）；在非MIX流程（Host或纯AIC/AIV）中不会应用InjectBlockSync / CrossCoreGSS / DelayedCrossCoreGSS。
- **架构差异**：内存型架构的mix块同步需要FFTS基址；RegBase可在拆分后使用延迟跨核GSS；unit-flag默认值随目标架构变化。
- **可选功能模式**：unit-flag同步、SyncSolver V1/V2、barrier-all / block-all调试模式，以及通过编译选项显式回退到InjectSync/InjectBlockSync。
- **验证需求**：检查生成的操作是否满足方言验证规范；成对的set/wait必须共享相同的event/flag id，并且core/pipe端点兼容。
