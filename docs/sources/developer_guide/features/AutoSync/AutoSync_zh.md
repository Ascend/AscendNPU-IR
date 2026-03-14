# 自动同步（Auto-Sync）

Auto-Sync 是 AscendNPU-IR（HIVM）编译器的自动同步插入功能，用于自动为共享数据或资源的生产者与消费者插入同步操作，确保正确的执行顺序。设计目标：**正确性**（避免数据竞争和顺序错误）与**最小开销**（仅插入必要的同步，安全时复用硬件事件）。

---

## 1. AICore 架构

<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html>

---

## 1. HIVM 同步操作（MLIR）

同步操作定义于 `HIVMIR/HIVMSynchronizationOps.td`，以下从 **MLIR 使用角度**（操作数/属性）进行描述，而非汇编语法。

### 1.1 核内同步（Normal-Sync）

- **`hivm.set_flag`**
  操作数/属性：`set_pipe`、`wait_pipe` 和 `flag_id`
  在 `set_pipe` 上等待该 pipe 所有前序指令执行完毕后执行。
  执行时触发 `flag_id`。

- **`hivm.wait_flag`**
  操作数/属性：`set_pipe`、`wait_pipe` 和 `flag_id`
  在 `wait_pipe` 上执行。
  阻塞其后所有指令，直到 `flag_id` 被触发。

- **`hivm.pipe_barrier`**
  操作数/属性：`pipe`
  对指定 pipe 的屏障操作。
  阻塞 `pipe` 上的所有后续指令，直到所有前序指令执行完毕。

### 1.2 跨核同步（Block-Sync，块内）

- **`hivm.sync_block_set`**
  操作数/属性：
  - `tcore_type`：目标核类型（vector/cube）
  - `tpipe`、`pipe`：目标核上的 set/wait pipe
  - `sync_instr_mode`（默认 `INTRA_BLOCK_SYNCHRONIZATION`）
  - `event_id`

  在目标核的 `tpipe`（set_pipe）上，等待同一核的所有前序指令完成后执行。
  设置 `event_id`。

- **`hivm.sync_block_wait`**
  操作数/属性：
  - `tcore_type`：目标核类型（vector/cube）
  - `tpipe`、`pipe`：目标核上的 set/wait pipe
  - `sync_instr_mode`（默认 `INTRA_BLOCK_SYNCHRONIZATION`）
  - `event_id`

  在目标核的 `pipe`（wait pipe）上执行。
  阻塞该核 `pipe` 上的所有后续指令，直到所有前序指令执行完毕。

---

## 2. 自动同步方案概述

代码库提供 **两种** 自动同步方案：

### **`Inject-Sync / Inject-Block-Sync`**

通过多个 Pass 插入所需同步操作、移除冗余同步，并利用活跃性分析分配 flag-id/event-id。
这是默认启用的主要方案。

### **`Graph-Sync-Solver / Cross-Core-GSS`**

使用基于图的算法分析输入代码结构并插入所需同步操作。
可选功能，通过 `-hivm-enable-graph-sync-solver=true` 命令行选项或 triton-ascend 的 `sync_solver=True` 选项启用。

---

## 3. InjectSync（主要的核内自动同步 Pass）

![alt text](0.png)

**目的：** 通过内存依赖分析、同步分析、事件 ID 分配和冗余同步清理（移动/删除），插入核级（核内）同步操作（`set_flag` / `wait_flag`）。

**源码位置：**

- 头文件：`include/../InjectSync/`
- 实现：`lib/../InjectSync/`（如 `InjectSync.cpp`、`MemoryDependentAnalyzer.cpp`、`SyncAnalysis.cpp`、`SyncEventIdAllocation.cpp`、`IRTranslator.cpp`、`SyncCodegen.cpp`、`MoveSyncState.cpp`、`RemoveRedundantSync.cpp`、`SyncCommon.cpp`）

**执行阶段：**

1. **IRTranslator**：
   从输入函数构建 Sync-IR（复合元素、循环、条件、内存操作）。
2. **SyncAnalyzer**：
   对每对冲突操作，插入一对 set_flag/wait_flag；若两操作属于同一 pipe，则插入 barrier(pipe)。
3. **MoveSyncState**：
   在保持语义的前提下，重新定位同步操作以减少流水线停顿。
4. **RemoveRedundantSync**：
   删除冗余的同步对。
5. **SyncEventIdAllocation**：
   分配静态或动态事件 ID，安全时进行复用。
6. **SyncCodegen**：
   生成 `hivm.set_flag` / `hivm.wait_flag` / `hivm.barrier`。

---

## 4. InjectBlockSync（块级同步 Pass）

**目的：** 为 **MIX** 内核（cube 和 vector 核）插入块级（块内）跨核同步操作：`sync_block_set`、`sync_block_wait`。

**源码：** `InjectBlockSync.cpp` `InjectBlockSync.h`

**行为：**

- 仅在 **MIX** 内核（非 host、非纯 AIC/AIV）上运行。
- 当内核参数中存在 FFTS 基址时，插入 `SetFFTSBaseAddrOp`。
- 三种模式（由选项和融合类型控制）：
  - **InjectAllBlockSync** — 在每个 `LoadOp` 和 `StoreOp`（cube/vector 交接处）前后插入块同步。
  - **InjectBlockMixSync** — 完整 mix 模式：通过 `SyncBlockIRTranslator` 构建块同步 IR，然后依次执行 SyncAnalyzer（BLOCKSYNC 模式）、MoveSyncState、RemoveRedundantSync、SyncEventIdAllocation、SyncCodegen。

---

## 5. GraphSyncSolver（基于求解器的核内 Pass）

![alt text](1.png)

**目的：** InjectSync 方案的替代实现，使用基于图的算法决定何时插入 set/wait 对并分配事件 ID（通常比 InjectSync 具有更好的复用效果）。

**源码：**

- 头文件：`include/../GraphSyncSolver/`
- 实现：`lib/../GraphSyncSolver/`（`GraphSyncSolver.cpp`、`SyncSolver.cpp`、`SyncSolverIR.cpp`、`SyncSolverIRTranslator.cpp`、`SyncSolverCodeGen.cpp`、`GraphSolver.cpp`、`EventIdSolver.cpp`、`Utility.cpp`、`SyncSolverTest.cpp`、`SyncSolverTester.cpp`）

**执行阶段：**

1. **IRTranslator**：
   从输入函数构建 Sync-IR（函数、作用域、循环、条件、读写操作）。
2. **Solver**：
   收集冲突对（生产者–消费者对），执行对选择与排序，可选地复用冲突对以节省事件 ID。
3. **CodeGenerator**：
   将求解结果转换回 MLIR，生成 `hivm.set_flag` / `hivm.wait_flag` / `hivm.barrier`。

---

## 6. CrossCoreGSS（跨核同步）

**目的：** 为 **MIX** 内核（cube 和 vector 核）插入块级（块内）跨核同步操作：`sync_block_set`、`sync_block_wait`。

**源码：** `CrossCoreGSS.h` `CrossCoreGSS.cpp`；复用 GraphSyncSolver 中的 `IRTranslator`、`Solver` 和 `CodeGenerator`。

**工作原理：**

- 与核内 GSS Pass 相同，但处理跨核内存操作。

---

## 7. Pass 选项与命令行标志

### 7.1 全局命令行标志（编译工具）

这些标志通常在编译器驱动中（如 `bishengir-hivm-compile`）配置，具体映射关系参见 `Passes.td` 及 `bishengir/lib/Tools/` 下的工具。

| 标志 | 类型 | 默认值 | 说明 |
| ---- | ---- | ------- | ----------- |
| `--disable-auto-inject-block-sync` | bool | false | 禁用块级 set/wait 的自动插入（InjectBlockSync / CrossCoreGSS）。 |
| `--disable-hivm-auto-inject-sync` | bool | false | 禁用 InjectSync（核内同步）。 |
| `--enable-hivm-inject-barrier-all-sync` | bool | false | 使 InjectSync 插入 barrier(all) 指令（在自动同步失败时有用）。 |
| `--enable-hivm-inject-block-all-sync` | bool | false | 使 InjectBlockSync 插入 block(all) 指令（在自动同步失败时有用）。 |
| `--enable-hivm-unit-flag-sync` | bool | false | 启用 unit-flag 同步特性。 |
| `--enable-hivm-graph-sync-solver` | bool | false | 使用 GraphSyncSolver/CrossCoreGSS 替代 InjectSync/InjectBlockSync 进行核内/跨核自动同步。 |

---

## 8. 扩展与调试

- **InjectSync：** 从 `InjectSync.cpp` 入手，按分析 → 分配 → 代码生成 → 移动 → 删除的流程进行。内存与同步决策：`MemoryDependentAnalyzer`、`SyncAnalysis`；事件 ID：`SyncEventIdAllocation`；生成：`SyncCodegen`、`IRTranslator`；清理：`MoveSyncState`、`RemoveRedundantSync`。使用 `SyncDebug` 进行日志记录。
- **InjectBlockSync：** 入口在 `InjectBlockSync.cpp`，块 IR 由 `SyncBlockIRTranslator` 构建，其余流水线在 BLOCKSYNC 模式下与 InjectSync 共享。
- **GraphSyncSolver / CrossCoreGSS：** 入口分别在 `GraphSyncSolver.cpp` 和 `CrossCoreGSS.cpp`。检查 `IRTranslator`（同步 IR 构建）、`Solver`（冲突选择）、`SyncSolverCodeGen`（MLIR 生成）。跨核行为通过 translator、solver 和 codegen 中的 `SyncMode::CROSS_CORE_SYNC` 控制（块 set/wait，无 barrier）。

**验证：** 检查生成的操作是否满足方言验证规范；set 和 wait 必须共享相同的 event/flag ID 且具有兼容的 pipe。

---
