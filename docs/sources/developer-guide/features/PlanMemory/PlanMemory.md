# PlanMemoryPass

本文介绍 HIVM 中的 **PlanMemory** 变换（`PlanMemoryPass`），包括硬件背景、算法原理、接口说明和约束能力。

---

## 1. 硬件背景

内存分配主要是为了分配昇腾芯片上的内存，包括 UB、L1、L0C 等。
由于昇腾硬件指令在执行的时候没有显式给每个buffer指定具体地址，因此内存分配需要通过软件实现。
内存分配需要在有限的 Local Memory下，根据每个申请buffer自身的size，以及AscendNPU IR运行过程中产生的multi-buffer、temp buffer这些额外内存，为每个内存缓冲区分配合适的内存起始地址。通过算法，在尽可能地保障性能的情况下，提高内存利用率。

片上内存分配主要是在Cube（矩阵）计算单元和Vector（矢量）计算单元所涉及的存储单元上进行内存分配。Cube访问的存储单元中，L0A存储左矩阵，L0B存储右矩阵，左右矩阵会从L1 Buffer搬入L0A和L0B，L0C存储矩阵乘的结果和中间结果，内存分配主要在L1和L0C存储单元上进行内存分配。Vector访问的存储单元是UB(Unified Buffer)内存，存储着向量计算的输入和输出，内存分配也需要在UB为不同的Buffer分配内存。

除了片上内存，PlanMemory还会进行少量的Workspace的内存分配，主要用于CV场景。如果涉及Cube计算完成后Vector单元继续参与运算，则需要将Cube运算结果从L0C搬出，临时保存在Workspace空间，再搬入UB进行Vector运算。

---

## 2. 算法原理

### 2.1. Pass 概述

**对 buffer 进行内存分配**，根据 生命区间 与 内存复用 的规则将 `memref.alloc`（本地分配）或 `memref_ext.alloc_workspace`（全局 workspace 分配）替换为地址偏移的形式，实现内存地址的分配和buffer复用。

#### 2.1.1 相关术语说明

- **BufferLife**：某 buffer 的「生命区间」——从第一次被写入（gen）到最后一次被读（kill）。若两个 buffer 的生命区间不重叠，它们可以共用同一块内存；PlanMemory 据此计算每个 alloc buffer 的地址偏移，使生命区间不重叠的 buffer 可以共享内存。
- **Alias**：当两个数据本质来源于同一个数据的时候，这两个数据就属于alias（别名）关系，如 `subview` 前后的数据。
- **Inplace**：某 op 的**输出**可以写在**输入**的存储位置上（覆盖写），从而少一次 alloc。例如 vcast 从 f16 转到 i16（等宽），输出可复用输入 buffer。PlanMemory 会识别这类 op，给输出分配与输入相同的地址偏移（或满足硬件 inplace 约束的规则）。
- **地址偏移 / pointer_cast**：内存分配后不再生成「独立 alloc」，而是生成 `hivm.hir.pointer_cast(offset)`：offset 为本 buffer 在该内存空间上的字节偏移量。
- **为何有两处 PlanMemory**：**pre bufferization** 阶段对 `memref_ext.alloc_workspace` 做 **GLOBAL_WORKSPACE_PLAN**，此时 IR 仍是 tensor 为主，本 pass 负责在 workspace 上分配内存地址。**post bufferization** 阶段对 `memref.alloc`（本地 alloc）做 **LOCAL_MEM_PLAN**，在 UB、L0C 等本地空间上进行内存地址分配。两处内存分配的对象和内存空间来源不同。

---

### 2.2. 两种内存分配在流水线中的位置

`mem-plan-mode`包括local-mem-plan和global-work-space-plan，分别表示片上Local内存分配和Workspace内存分配。

| 模式 | 作用对象 | 说明 |
|------|----------|------|
| **LOCAL_MEM_PLAN** | `memref.alloc` | 对本地 alloc buffer 进行内存分配 |
| **GLOBAL_WORKSPACE_PLAN** | `memref_ext.alloc_workspace` | 在 workspace 上分配内存，供 CV 优化使用 |

- 在 **hivmPreBufferizationOptimizationPipeline** 中：

```
... → CVPipelining → TileCubeVectorLoop → PlanMemory (GLOBAL_WORKSPACE_PLAN) → InjectBlockSync → ...
```

在 CVPipelining、TileCubeVectorLoop 之后，InjectBlockSync 之前运行。此时 workspace 已插入并绑定，PlanMemory 负责在 workspace 上分配起始地址。

- 在 **hivmPostBufferizationOptimizationPipeline** 中：

```
... → MarkMultiBuffer → PlanMemory (LOCAL_MEM_PLAN) → InjectBlockSync → ...
```
还有一次 PlanMemory (LOCAL_MEM_PLAN)，针对 bufferization 后的 `memref.alloc`进行Local内存起始地址的分配,在 MarkMultiBuffer 之后，InjectBlockSync 之前运行。

---

### 2.3. 实现原理（代码结构）

**源文件**：`bishengir/lib/Dialect/HIVM/Transforms/PlanMemory.cpp`

主流程包含：

- **MemLivenessAnalysis** 进行每个 Buffer 的生命区间分析；
- **MemPlan** 基于上述生命区间的分析，对不同 Buffer 进行内存分配；
- **populateBufferAddressToAllocOp** 使用`hivm.hir.pointer_cast(offset)`，将分配完成的内存起始地址写回到对应的 Buffer 上进行指示。

### 2.3.1 MemLivenessAnalysis

主流程包含：
- **Liveness**：通过社区 Liveness 类分析各个节点的活跃性。
- **RecursionIR**：遍历 IR（含 scf.for、scf.if、scf.while），收集每个 op 的 **gen**（生成了哪些 buffer）与 **kill**（哪些 buffer 在此处最后一次被读），用于计算每个 Buffer 的生命区间 BufferLife。
- **GenerateBufferLife**：根据 gen/kill 计算每个 buffer 的 **lifetime**（从第一次写到最后一次读的区间）。如果两个 buffer 的 lifetime 不重叠，即可共享内存。
- **InitializeInplacePairList**：基于Alias关系，识别可 **inplace** 的 op（输出可写在输入的存储位置上，复用同一块 buffer）；内存分配时会为这类 op 的输入输出分配相同内存起始地址。

#### 2.3.2 MemPlan

内存分配有两种模式：顺序分配 和 可复用分配。当所有 Buffer 内存相加都能在对应的 Memory Scope（内存空间，如 UB、L1等）分的下时，无需复杂算法，可以直接进行顺序分配。当所有 Buffer 内存相加，超出对应的 Memory Scope 大小时，需由算法分析可复用内存的 Buffer，确保内存在复用的同时，避免冲突产生精度问题。

可复用分配 包括**Inplace复用**和**三级分配复用**。

主流程包含：
 - **GenerateStorageEntry**：基于生命区间分析的结果生成各个内存节点信息。
 - **PlanLocalMemAddress / PlanWorkSpaceMemAddress**：基于硬件可 **Inplace** 的 op 和位宽等约束，分析不同 Buffer 的**Inplace**复用关系；通过 顺序分配 或 三级分配策略 进行内存地址的分配。
 - **UpdateBuffer2Offsets**：更新内存分配后每个 Buffer 的地址偏移信息。

Inplace 复用条件：
1. Memory Scope相同，例如同为 UB 。
2. `A = B + C`场景，A的kill节点是C的gen节点。
3. 符合硬件约束。

三级分配复用：从高level到低level策略尝试分配内存，分不下时会降level进行回滚重试。
 - **level2**: 基于流水类型的优先复用策略，非DMA的相同流水优先复用。
 - **level1**: 相同Loop下，如果 SingleBuffer 复用 DoubleBuffer ，SingleBuffer 自动转 DoubleBuffer。
 - **level0**: 如果两块 Buffer 的生命区间不重叠，内存可以直接复用。

#### 2.3.3 变换

对 `alloc_workspace`（GLOBAL_WORKSPACE_PLAN） 和 `memref.alloc`（LOCAL_MEM_PLAN）计算地址偏移量，替换为 `hivm.hir.pointer_cast(offset)`。

---

### 2.4. 测试用例

**文件**：`bishengir/test/Dialect/HIVM/plan-memory.mlir`

**典型 CHECK**：
```mlir
// CHECK-NOT: memref.alloc()
// CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
// CHECK: {{.*}} = hivm.hir.pointer_cast(%[[CONST0]])
```

---

## 3. 接口说明

| 选项 | 默认值 | 说明 |
|--------|--------|--------|
| `-mem-plan-mode=global-work-space-plan` | false | CV 流水线中使用 `GLOBAL_WORKSPACE_PLAN` |
| `enable-global-workspace-reuse` | false | 启用 workspace 内 buffer 复用 |
| `restrict-inplace-as-isa` | false | 限制 inplace 规则以匹配 ISA 行为 |

---

## 4. 约束能力

1. PlanMemory pass的内存分配是基于生命区间的分析来保障的。当内存尽可能复用的情况下，内存还是分不下时，会上报对应的Memory Scope overflow的error，比如`UB overflow`。

2. PlanMemory 针对不同的内存空间都有容量(SIZE)限制与地址偏移对齐(ALIGN)的粒度约束（见下表），确保分配结果满足硬件要求。GLOBAL_WORKSPACE_PLAN 模式下，workspace 来自 GM，无 Local Memory 的容量限制。

    | 内存空间 | SIZE | ALIGN |
    |------|--------|------|
    | UB | 192KB | 32字节 |
    | L1 | 512KB | 32字节 |
    | L0C | 128KB | 512字节 |
