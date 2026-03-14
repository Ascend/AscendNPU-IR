# 内存管理

本文介绍 HIVM 中的 **PlanMemory** 变换（`PlanMemoryPass`），包括硬件背景、算法原理、接口说明和约束能力。

---

## 1. 硬件背景

昇腾硬件片上内存使用Buffer机制，主要包含Cube（矩阵）计算单元和Vector（矢量）计算单元所涉及的存储单元。软件需要显式控制内存地址，并确保操作地址的对齐。

以 Atlas A2 训练系列产品 / Atlas A2 推理系列产品 为例，硬件架构图<sup>[1]</sup>如下：
![](./HardwareStructure.png)

各Buffer对齐要求：

| Buffer | 对齐要求 | 功能 |
|-----|-----|-----|
| Unified Buffer (UB) | 32字节对齐 | 通用缓存空间，主要用于向量和标量运算 |
| L1 Buffer | 32字节对齐 | 暂存feature map等卷积使用到的数据 |
| L0A Buffer | 512字节对齐 | 暂存矩阵运算的左矩阵(feature map) |
| L0B Buffer | 512字节对齐 | 暂存矩阵运算的右矩阵(weight) |
| L0C Buffer | 512字节对齐 | 暂存矩阵运算的中间结果和输出矩阵 |
| BT Buffer | 64字节对齐 | BiasTable Buffer，存放矩阵运算中的Bias |
| FP Buffer | 64字节对齐 | Fixpipe Buffer，存放量化参数、Relu参数等 |

---

## 2. 算法原理

### 2.1. 软件背景

针对昇腾片上内存，由于输入IR中`memref.alloc`的Buffer仅包含变量名称、需要的内存空间大小，不包含地址信息，因此AscendNPU IR内存管理模块（PlanMemory）需要根据Buffer的**生命区间**分配合适的内存地址，防止运算过程中出现内存覆写导致精度问题。为了尽可能在有限的内存空间下分配完所有的Buffer，PlanMemory会基于「IR语义」和「硬件支持」进行**内存复用**。同时为了避免引入不必要的数据依赖影响性能，PlanMemory提供了三级分配内存算法，在尽量保障性能的情况下，提高内存利用率。

片上内存分配主要是在Cube（矩阵）计算单元和Vector（矢量）计算单元所涉及的存储单元（包括 UB、L1、L0C 等）上进行内存分配。Cube访问的存储单元中，L0A存储左矩阵，L0B存储右矩阵，左右矩阵会从L1 Buffer搬入L0A和L0B，L0C存储矩阵乘的结果和中间结果，内存分配主要在L1和L0C存储单元上进行内存分配。Vector访问的存储单元是UB(Unified Buffer)内存，存储着向量计算的输入和输出，内存分配也需要在UB为不同的Buffer分配内存。

除了片上内存，PlanMemory还会进行少量的Workspace的内存分配(`memref_ext.alloc_workspace`)，主要用于CV场景。如果涉及Cube计算完成后Vector单元继续参与运算，则需要将Cube运算结果从L0C搬出，临时保存在Workspace空间，再搬入UB进行Vector运算。片外空间交给框架Rumtime统一申请管理，因此算子需要反馈所需的Workspace空间大小。

### 2.2 相关术语说明

- **BufferLife**：某Buffer的「生命区间」——从第一次被写入（gen）到最后一次被读（kill）。若两个Buffer的生命区间不重叠，它们可以共用同一块内存；PlanMemory据此计算每个alloc Buffer的地址偏移，使生命区间不重叠的Buffer可以共享内存。
- **Alias**：当两个数据本质来源于同一个数据的时候，这两个数据就属于alias（别名）关系，如 `subview` 前后的数据。
- **Inplace复用**：某op的**输出**可以写在**输入**的存储位置上（覆盖写），从而少一次alloc。例如`vcast`从`f16`转到`i16`（等宽），输出可复用输入Buffer。PlanMemory会识别这类op，给输出分配与输入相同的地址偏移（或满足硬件inplace约束的规则）。
- **地址偏移 / pointer_cast**：内存分配后不再生成「独立alloc」，而是生成 `hivm.hir.pointer_cast(offset)`（offset 为本Buffer在该内存空间上的字节偏移量）。

---

### 2.3. 实现原理

**源文件**：`bishengir/lib/Dialect/HIVM/Transforms/PlanMemory.cpp`

主流程包含：

- **生命区间分析** ：对IR中的每个Buffer进行gen和kill的分析；
- **内存分配**：基于上述生命区间的分析，对不同Buffer进行内存地址分配；
- **OP变换**：使用`hivm.hir.pointer_cast(offset)`替换原`alloc`，将分配完成的内存起始地址写回到对应的Buffer上进行指示。

### 2.3.1 生命区间分析

主流程包含：
1. 通过社区`Liveness`类分析各个节点的活跃性。
2. **遍历IR**（含 scf.for、scf.if、scf.while），收集每个op的**gen**（生成了哪些Buffer）与 **kill**（哪些Buffer在此处最后一次被读），用于计算每个Buffer的生命区间BufferLife。
3. 根据gen/kill计算每个Buffer的**lifetime**（从第一次写到最后一次读的区间）。如果两个Buffer的lifetime不重叠，即可共享内存。
4. 基于Alias关系，识别出一部分可**inplace**的Buffer，分配相同的内存起始地址。

#### 2.3.2 内存分配

内存分配有两种模式：顺序分配 和 可复用分配。当所有Buffer内存相加都能在对应的Memory Scope（内存空间，如 UB、L1等）分的下时，无需复杂算法，可以直接进行顺序分配。当所有Buffer内存相加，超出对应的Memory Scope大小时，需由算法分析可复用内存的Buffer，确保内存在复用的同时，避免冲突产生精度问题。

可复用分配 包括**Inplace复用**和**三级分配复用**。

##### 2.3.2.1 Inplace复用

Inplace复用条件：
1. Memory Scope相同，例如同为 UB 。
2. `A = B + C`场景，A的kill节点是C的gen节点。
3. 符合硬件约束。

##### 2.3.2.2 三级分配复用

从高level到低level策略尝试分配内存，分不下时会降level进行回滚重试。

[1] **Level2**

硬件多流水单元，PIPE并行起来是关键，不同流水间内存复用会引入额外的数据依赖，导致的流水冲突会影响到流水效率。

Level2：基于流水类型的优先复用策略，会使非DMA的相同流水优先复用（比如Vector与Vector指令的Buffer优先复用）。

举例：

```
Shared A [A0, A1]
Shared B [B]
Shared C [C]
Shared D [D0, D1]
Loop i:
  // sync
  op1(A0, A1) // DMA OP, Double Buffer
  op2(B)      // Vector OP
  op3(C)      // Vector OP
  op4(D0, D1) // DMA OP, Double Buffer
```

开启Double Buffer后，A/D为DMA PIPE涉及的内存，各自申请了2片内存。当Shared Memory内存资源有限时，C与A复用会引入 op1 (DMA PIPE) 和 op4 (Vector PIPE)之间额外的依赖，导致 MTE_PIPE 与 V_PIPE 无法并行进行计算，从而影响流水性能。

使用Level2的内存分配策略后，C与B复用内存，两者同为Vector指令，V_PIPE本就只能串行执行，因此Vector指令之间复用对流水无额外影响。

使用Level2前后的流水效果对比如下：
![](./Level2.png)

 - 优点：同流水PIPE复用不引入PIPE间额外依赖，整体算子性能更好。
 - 缺点：可复用解空间小，内存复用的成功概率低。

[2] **Level1**

DoubleBuffer通过并行化数据加载和计算，大幅提升计算性能并减少等待时间，是算子性能优化的核心技术点。复杂场景下内存资源紧张，一旦出现Single Buffer复用Double Buffer，会使得Double Buffer无法并行，算子流水被打断导致算子性能差。

Level1：相同Loop下，如果SingleBuffer复用DoubleBuffer，SingleBuffer自动转DoubleBuffer。

举例：

```
Shared A [A0, A1]
Shared B [B]
Shared C [C]
Shared D [D0, D1]
Loop i:
  // sync
  op1(A0, A1) // DMA OP, Double Buffer
  op2(B)      // Vector OP
  op3(C)      // Vector OP
  op4(D0, D1) // DMA OP, Double Buffer
```

开启Double Buffer后，A/D为DMA PIPE涉及的内存，各自申请了2片内存。当Shared Memory内存资源有限时，C的Single Buffer与A的一个Buffer复用会使得op1使用A0内存时需要等待op3释放C(A0)内存，因此会导致流水打断，影响性能。

使用Level1的内存分配策略后，C复用Double Buffer会自动开启Double Buffer，op1使用A0内存时op3使用的是C1(A1)内存，因此op1无需等待op3，依然可以流水并行。

使用Level1前后的流水效果对比如下：
![](./Level1.png)

 - 优点：避免Double Buffer场景下流水被打断，流水性能更好。
 - 缺点：额外开启Double Buffer，需要一片额外的内存，会降低整体内存复用成功的概率。

[3] **level0**

Level0：如果两块 Buffer 的生命区间不重叠，内存可以直接复用。

使用Level0前后的内存使用情况对比如下：
![](./Level0.png)

 - 优点：能够尽可能地复用内存，内存可复用概率高。
 - 缺点：完全无视硬件并行流水，不合理的复用会导致算子性能差。

#### 2.3.3 OP变换

计算完成所有Buffer的地址后，将 `memref_ext.alloc_workspace`（GLOBAL_WORKSPACE_PLAN） 和 `memref.alloc`（LOCAL_MEM_PLAN）替换为 `hivm.hir.pointer_cast(offset)`，指示Buffer的内存起始地址。

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
| `enable-global-workspace-reuse` | false | 启用 workspace 内 Buffer 复用 |
| `restrict-inplace-as-isa` | false | 限制 inplace 规则以匹配 ISA 行为 |

---

## 4. 约束能力

**用户需要保证「同一时间所申请的Buffer总大小」小于等于「实际硬件内存空间大小」。**

> 【注】每个Buffer的实际占用空间会自动进行字节对齐。对齐大小见 [1 硬件背景](#1-硬件背景) 。

反之，PlanMemory pass会编译Fail，并上报对应的Memory Scope overflow的error，比如`UB overflow`，如下图所示。
![](./UBOverflowError.png)

---

[1] 图片来源：昇腾社区Ascend C算子开发文档(https://www.hiascend.com/doc_center/source/zh/canncommercial/850/opdevg/Ascendcopdevg/figure/zh-cn_image_0000002502735896.png)
