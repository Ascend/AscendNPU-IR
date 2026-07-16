# 自动同步（Auto-Sync）

Auto-Sync是AscendNPU IR（HIVM）编译器的自动同步插入功能，用于自动为共享数据或资源的生产者与消费者插入同步操作，确保正确的执行顺序。设计目标：正确性（避免数据竞争和顺序错误）与最小开销（仅插入必要的同步，安全时复用硬件事件）。

## 硬件背景

**AICore架构**：

AICore架构详情可查阅官方文档：[基本架构](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html)。

**HIVM同步操作说明**：

同步操作定义于`HIVMIR/HIVMSynchronizationOps.td`。以下从MLIR使用角度（操作数/属性）进行描述，而非汇编语法。

**核内同步（Normal-Sync）**：

- `hivm.set_flag`
  操作数/属性：`set_pipe`、`wait_pipe`、`flag_id`
  运行逻辑：在`set_pipe`上执行，等该pipe上所有前序指令执行完毕后再执行。执行时触发`flag_id`。

- `hivm.wait_flag`
  操作数/属性：`set_pipe`、`wait_pipe`、`flag_id`
  运行逻辑：在`wait_pipe`上执行。阻塞其后所有指令，直到`flag_id`被触发。

- `hivm.pipe_barrier`
  操作数/属性：`pipe`
  运行逻辑：对指定pipe的屏障操作。阻塞`pipe`上的所有后续指令，直到所有前序指令执行完毕。

**跨核同步（Block-Sync，块内）**：

- `hivm.sync_block_set`
  操作数/属性：`tcore_type`、`tpipe`、`pipe`、`sync_instr_mode`、`event_id`

  参数说明：

    - `tcore_type`：目标核类型（vector/cube）
    - `tpipe`、`pipe`：目标核上的set/wait pipe
    - `sync_instr_mode`：默认值为`INTRA_BLOCK_SYNCHRONIZATION`

  执行逻辑：在目标核的`tpipe`（set_pipe）上执行，需等同一核的core.pipe上所有前序指令执行完毕后再执行。执行完成后设置`event_id`。

- `hivm.sync_block_wait`
  操作数/属性：`tcore_type`、`tpipe`、`pipe`、`sync_instr_mode`、`event_id`

  参数说明：

    - `tcore_type`：目标核类型（vector/cube）
    - `tpipe`、`pipe`：目标核上的set/wait pipe
    - `sync_instr_mode`：默认值为`INTRA_BLOCK_SYNCHRONIZATION`

  执行逻辑：在pipe（pipe_wait）上执行，会阻塞目标核`tcore_type`的`pipe`上所有后续指令，直到所有前序指令执行完毕。

## 算法原理

### AutoSync

代码库提供两种Auto-Sync解决方案：

- `Inject-Sync` / `Inject-Block-Sync` Passes
  使用多个Pass插入所需同步操作、移除冗余同步，并使用活跃性分析分配flag ID/event ID。它是默认启用的主要方案。

- `Graph-Sync-Solver` / `Cross-Core-GSS` Passes
  使用基于图的算法分析输入代码结构并插入所需同步操作。它仍是可选功能，可通过`-hivm-enable-graph-sync-solver=true`（或Triton-Ascend中`sync_solver=True`）启用。

### InjectSync

![alt text](../../../images/developer_guide/auto_sync0.png)

**目的：** 使用内存依赖分析、同步分析、event-id分配以及冗余同步清理（移动/删除冗余同步）在核内插入同步（`set_flag` / `wait_flag`）。

**源码**：

- 头文件：`include/../InjectSync/`
- 实现：`lib/../InjectSync/`（例如`InjectSync.cpp`、`MemoryDependentAnalyzer.cpp`、`SyncAnalysis.cpp`、`SyncEventIdAllocation.cpp`、`IRTranslator.cpp`、`SyncCodegen.cpp`、`MoveSyncState.cpp`、`RemoveRedundantSync.cpp`、`SyncCommon.cpp`）

**执行阶段**：

1. IRTranslator：从输入函数构建Sync-IR（复合元素、循环、条件、内存操作）。
2. SyncAnalyzer：对每对冲突操作，插入一对set_flag/wait_flag；若两操作属于同一pipe，则插入barrier(pipe)。
3. MoveSyncState：在保持语义的前提下，重新定位同步操作以减少停顿。
4. RemoveRedundantSync：删除冗余同步对。
5. SyncEventIdAllocation：分配静态或动态event IDs；在安全时进行复用。
6. SyncCodegen：生成`hivm.set_flag` / `hivm.wait_flag` / `hivm.barrier`

### InjectBlockSync

**目的：** 为MIX内核（cube和vector）插入块级（块内）跨核同步操作：`sync_block_set`、`sync_block_wait`。

**源码：** `InjectBlockSync.cpp` `InjectBlockSync.h`

**行为**：

- 仅在MIX内核（非host、非纯AIC/AIV）上运行。
- 当内核参数中存在FFTS基址时，插入`SetFFTSBaseAddrOp`。
- 三种模式（由选项和融合类型控制）：
    - InjectAllBlockSync：在每个`LoadOp`和`StoreOp`（cube/vector交接处）前后插入块同步。
    - InjectBlockMixSync：面向完整mix场景，通过`SyncBlockIRTranslator`构建块同步IR，然后依次执行SyncAnalyzer（BLOCKSYNC模式）、MoveSyncState、RemoveRedundantSync、SyncEventIdAllocation、SyncCodegen。

### GraphSyncSolver

![alt text](../../../images/developer_guide/auto_sync1.png)

**目的：** 作为Inject-Sync方案的替代方案；使用基于图的算法判断何时插入set/wait成对操作，并分配event IDs。

**源码**：

- 头文件：`include/../GraphSyncSolver/`
- 实现：`lib/../GraphSyncSolver/`（`GraphSyncSolver.cpp`、`SyncSolver.cpp`、`SyncSolverIR.cpp`、`SyncSolverIRTranslator.cpp`、`SyncSolverCodeGen.cpp`、`GraphSolver.cpp`、`EventIdSolver.cpp`、`Utility.cpp`、`SyncSolverTest.cpp`、`SyncSolverTester.cpp`）

**执行阶段**：

1. IRTranslator：从输入函数构建Sync-IR（函数、作用域、循环、条件、读写操作）。
2. Solver：收集冲突对（生产者–消费者对），执行对选择与排序；可选地复用冲突对以节省event IDs。
3. CodeGenerator：将求解结果转换回MLIR，生成`hivm.set_flag` / `hivm.wait_flag` / `hivm.barrier`。

### CrossCoreGSS

**目的：** 为MIX内核（cube和vector）插入块级（块内）跨核同步操作：`sync_block_set`、`sync_block_wait`。

**源码：** `CrossCoreGSS.h` `CrossCoreGSS.cpp`；复用GraphSyncSolver中的`IRTranslator`、`Solver`和`CodeGenerator`。

**工作原理**：与核内GSS Pass相同，但它处理跨核内存操作。

## 接口说明

**命令行选项**：

这些通常在编译器驱动中接入（例如`bishengir-hivm-compile`）；具体映射关系请参见`Passes.td`以及`bishengir/lib/Tools/`下的工具。

| 标志                                    | 类型 | 默认值 | 说明                                                         |
| --------------------------------------- | ---- | ------ | ------------------------------------------------------------ |
| `--disable-auto-inject-block-sync`      | bool | false  | 禁用自动块级set/wait插入（InjectBlockSync / CrossCoreGSS）   |
| `--disable-hivm-auto-inject-sync`       | bool | false  | 禁用InjectSync（核内同步）                                   |
| `--enable-hivm-inject-barrier-all-sync` | bool | false  | 使InjectSync插入barrier(all) 指令（在自动同步失败时有用）    |
| `--enable-hivm-inject-block-all-sync`   | bool | false  | 使InjectBlockSync插入block(all) 指令（在自动同步失败时有用） |
| `--enable-hivm-unit-flag-sync`          | bool | false  | 启用unit-flag同步特性                                        |
| `--enable-hivm-graph-sync-solver`       | bool | false  | 使用GraphSyncSolver/CrossCoreGSS替代InjectSync/InjectBlockSync进行核内/跨核自动同步 |

## 使用约束

- **硬件顺序模型：** Auto-Sync通过插入HIVM同步操作（`hivm.set_flag` / `hivm.wait_flag`、`hivm.pipe_barrier`，以及（如适用）`hivm.sync_block_set` / `hivm.sync_block_wait`）来组织执行顺序。该顺序用cores与pipes，以及event/flag id来表示。
- **正确性基于可行性校验：** 对于求解器流程（Graph Sync Solver），只有当候选同步约束在基于图的可达性/顺序模型下仍保持可行时才会接受（避免死锁或过度约束导致调度失败）。
- **块级同步覆盖范围：** 跨核块级同步（`sync_block_set` / `sync_block_wait`）面向MIX内核（cube/vector handoff）；在非MIX流程（host或纯AIC/AIV）中不会应用InjectBlockSync/CrossCoreGSS。
- **可选功能模式：** 当支持的操作具备适用条件时，可启用unit-flag同步作为替代方案；也可以通过编译选项选择基于图的求解器，而不是使用InjectSync/InjectBlockSync。
- **验证需求：** 检查生成的操作是否满足方言验证规范；`set_flag` / `wait_flag`必须共享相同的event/flag id，并且core/pipe端点兼容。
