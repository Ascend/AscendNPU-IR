# 向量布局分析

本文介绍HIVMAVE中的向量布局分析变换（AnalyzeVectorLayoutPass），包括硬件背景、算法原理、测试用例、接口说明和使用约束。

## 硬件布局规划

昇腾Vector计算单元采用SIMD架构，一条指令同时处理向量寄存器中的多个元素。向量寄存器的物理容量固定，但同一寄存器中的逻辑元素并不只有一种排布方式。元素位宽、元素之间的间隔以及指令实际使用的数据位宽，都会影响数据在寄存器中的组织方式。

以256字节向量寄存器为例，HIVMAVE使用`VecMemType`描述向量数据的布局。主要布局如下：

| 布局 | 排布方式 | 每个元素的有效占位 | 单个向量寄存器可容纳的逻辑元素数 |
|-----|-----|-----|-----|
| `B8` | 8-bit元素紧密排列 | 8 bit | 256 |
| `B8_2VL` | 8-bit元素间隔排列，间距为2 | 16 bit | 128 |
| `B8_4VL` | 8-bit元素间隔排列，间距为4 | 32 bit | 64 |
| `B16` | 16-bit元素紧密排列 | 16 bit | 128 |
| `B16_2VL` | 16-bit元素间隔排列，间距为2 | 32 bit | 64 |
| `B32` | 32-bit元素紧密排列 | 32 bit | 64 |

`B8_2VL`、`B8_4VL`和`B16_2VL`虽然保存的逻辑元素仍分别是8-bit或16-bit，但每个元素占用的有效槽位更宽。间隔布局能够与更宽元素的紧密布局建立无损的对应关系：

```text
B8_2VL  <-> B16
B8_4VL  <-> B32
B16_2VL <-> B32
```

这种对应关系是硬件类型转换指令的基础。例如，8-bit到16-bit的扩展不能将任意紧密排列的`B8`数据直接转换为`B16`，源数据需要先处于`B8_2VL`布局，使每个8-bit元素位于对应16-bit槽位中。类似地，16-bit到32-bit的扩展要求源数据采用`B16_2VL`布局。

### Load/Store布局

Vector Load/Store指令不仅负责搬运数据，还负责在内存和向量寄存器之间完成数据排布转换。HIVMAVE中的Load/Store分发模式与硬件指令的dist模式对应：

- `NORM`：数据按照元素位宽紧密装载或存储；
- `UNPK`：Load时将内存中的紧密数据展开为寄存器中的间隔布局；
- `PK`：Store时将寄存器中的间隔布局压缩为内存中的紧密数据；
- `INTLV`和`DINTLV`：搬运或计算过程中完成交织和解交织；
- `PB8`、`PB16`、`PB32`：按照8-bit、16-bit或32-bit粒度生成谓词数据。

因此，逻辑向量类型`vector<Nxi8>`不能单独决定应该使用哪一种Load/Store指令。若结果布局是`B8`，Load通常使用`NORM`；若结果布局是`B8_2VL`或`B8_4VL`，则需要使用相应的`PK`或`PK4`分发形式。Store也需要根据向量布局和Mask布局选择匹配的模式。

### 类型转换布局

AVE硬件的类型转换指令按照寄存器槽位进行转换，并通过`part`等参数选择通道。位宽转换不是只改变MLIR类型中的元素类型，也会改变元素在寄存器中的有效排布：

```text
trunc方向（高位宽转换为低位宽）：
B32      -- trunc 32->8  --> B8_4VL
B32      -- trunc 32->16 --> B16_2VL
B16      -- trunc 16->8  --> B8_2VL

ext方向（低位宽转换为高位宽）：
B8_2VL  -- ext 8->16  --> B16
B8_4VL  -- ext 8->32  --> B32
B16_2VL -- ext 16->32 --> B32
```

如果源数据的布局不满足转换指令要求，编译器需要在转换前后插入`INTLV`或`DINTLV`，否则转换结果可能发生槽位重叠或数据错位。因此，类型转换操作的布局是正确性约束，而不是单纯的性能优化信息。

### Mask布局

AVE使用谓词寄存器保存Mask。Mask的逻辑元素类型通常是`i1`，但其控制粒度需要与数据指令的有效位宽匹配：

| 数据布局 | 指令实际处理粒度 | Mask布局 |
|-----|-----|-----|
| `B8` | 8 bit | `B8` |
| `B16` | 16 bit | `B16` |
| `B32` | 32 bit | `B32` |
| `B8_2VL` | 16 bit | `B16` |
| `B8_4VL` | 32 bit | `B32` |
| `B16_2VL` | 32 bit | `B32` |

例如，`B8_2VL`的数据虽然元素类型为8-bit，但硬件以16-bit槽位处理，因此对应的Mask必须采用`B16`粒度。若Mask仍按`B8`粒度解释，谓词位与数据lane之间的对应关系会错误。

## 相关术语说明

| 术语 | 含义 |
|-----|-----|
| VecMemType | 向量值在Vector寄存器中的物理布局状态，定义于`HIVMAVEAttrs.td`。| 
| VectorLayoutAttr | 附加在`VectorType`上的`ave.vector_layout`属性，内部保存`VecMemTypeAttr`。例如：`#ave.vector_layout<{mem = #ave.vec_mem_type<b16>}>`。|
| FunctionDistType | 操作所使用的功能类型或分发类型，以`functionType`属性保存。它描述操作需要采用的硬件指令变体或额外布局处理方式。|
| 有效位宽 | 硬件指令实际处理一个元素槽位时使用的位宽。对于间隔布局，有效位宽可能大于逻辑元素位宽，例如`B8_2VL`的有效位宽为16 bit。|
| 密集布局 | 元素连续排列的布局，包括`B8`、`B16`和`B32`。|
| 稀疏布局 | 元素之间存在间隔的布局，包括`B8_2VL`、`B8_4VL`和`B16_2VL`。|
| INTLV/DINTLV | 布局重排操作。INTLV将数据转换为更稀疏的交错排布，DINTLV将数据转换为更密集的排布。|

## 算法原理

### 软件背景

前序转换会将标准Vector/Arith操作转换为HIVMAVE操作。此时，IR中的`VectorType`通常只记录逻辑形状和元素类型，例如`vector<128xi8>`，并没有记录元素在硬件向量寄存器中的排布方式。

后续的AVE规范化和指令降低需要这些信息完成以下工作：

- 根据向量结果的布局选择Load的`NORM`、`UNPK`或`UNPK4`模式；
- 根据Store输入的布局选择`NORM`、`PK`或`PK4`模式；
- 为Mask选择正确的谓词粒度；
- 为ext/trunc选择正确的硬件转换形式和通道；
- 在源布局和目标布局不一致时插入`INTLV`或`DINTLV`；
- 保证同一数据流上的多个消费者看到一致的布局。

如果为每个操作独立选择布局，前后操作之间很容易出现不匹配。例如，Load可以产生`B8`、`B8_2VL`或`B8_4VL`中的任意一种布局，但后续ext、compare和Store对输入布局的要求不同。一个操作的选择会影响其所有输入和输出，最终需要在整个函数范围内统一求解。

AnalyzeVectorLayout将布局推导建模为约束满足问题。它从函数末端的向量结果开始，枚举可能的布局，然后沿数据流反向传播约束。每个操作根据自己的硬件语义产生一个或多个合法候选；候选在传播过程中不断合并，发生冲突时被删除。最终选出的解同时满足数据布局、Mask粒度、类型转换和控制流边界的约束。

### 求解流程

1. 初始化解空间：收集没有用户的向量结果，并根据元素位宽枚举初始布局；
2. 逆序遍历操作：从函数末端向前处理所有包含向量操作数或向量结果的操作；
3. 操作约束求解：通过`TypeSwitch`分派到操作类型对应的求解函数；
4. 合并候选解：将当前操作产生的输入状态合并到已有状态，发现冲突时丢弃候选；
5. 应用求解结果：为向量结果类型添加布局属性，为操作添加`functionType`属性，并重写特殊操作。

### 操作求解规则

#### 操作分派

`solveProblem(Operation *)`使用`TypeSwitch`根据操作类型分派求解函数。主要分派关系如下：

| 操作类型 | 求解函数 | 主要处理内容 |
|-----|-----|-----|
| `VFLoadOp` | `solveProblem(VFLoadOp)` | 根据结果位宽和布局选择`NORM`、`PK`、`PK4`或谓词Load形式 |
| `VFMaskedStoreOp` | `solveProblem(VFMaskedStoreOp)` | 同时约束数据向量和Mask布局 |
| `VFStoreWithStrideOp` | `solveProblem(VFStoreWithStrideOp)` | 根据间隔布局选择`NORM`、`DINTLV2`或`DINTLV4` |
| `VFPgeOp`、`VFPltOp` | `solvePregGenProblem` | 根据谓词结果状态选择`PB8`、`PB16`或`PB32` |
| 逐元素Vector操作 | `solveMaskProblem` | 约束数据操作数、结果和Mask布局 |
| `VFCmpOp` | `solveProblem(VFCmpOp)` | 根据比较数据位宽和Mask结果状态约束左右输入 |
| `VFTruncFOp`、`VFTruncIOp` | `solveTruncProblem` | 处理高位宽到低位宽的布局转换 |
| `VFExtFOp`、`VFExtUIOp`、`VFExtSIOp` | `solveExtProblem` | 处理低位宽到高位宽的布局转换 |
| 其他类型转换操作 | `solveTypeConvertProblem` | 按源、结果位宽关系转发到ext、trunc或Mask求解 |
| `VFGatherOp` | `solveProblem(VFGatherOp)` | 约束index向量和Mask布局 |
| `VFInterleaveOp`、`VFDeInterleaveOp` | `solveLayoutChangeProblem` | 处理布局变化和交织方向 |
| `scf.for` | `solveProblem(scf::ForOp)` | 约束循环迭代参数、初始值和循环结果 |
| `scf.yield` | `solveProblem(scf::YieldOp)` | 将循环结果状态传播到yield操作数 |
| `func.call` | `solveProblem(func::CallOp)` | 处理调用边界的向量状态组合 |
| 其他含向量结果的操作 | `solveProblemDefault` | 默认假设向量操作数和第一个向量结果状态一致 |

未在`TypeSwitch`中显式处理的操作会进入默认分支。默认分支要求操作至少包含一个向量结果，否则无法建立约束；如果存在多个向量操作数，则将第一个向量结果的状态传播到所有向量操作数。

#### Load、Store和Mask求解

对于`VFLoadOp`，结果布局直接决定Load功能类型：

| 元素位宽 | 结果布局 | 功能类型 |
|-----|-----|-----|
| 1 bit | `B8`、`B16`、`B32` | `PB8`、`PB16`、`PB32` |
| 8 bit | `B8`、`B8_2VL`、`B8_4VL` | `NORM`、`PK`、`PK4` |
| 16 bit | `B16`、`B16_2VL` | `NORM`、`PK` |
| 32 bit | `B32` | `NORM` |

`VFMaskedStoreOp`除了约束数据向量外，还必须约束Mask：

| 数据元素位宽 | 数据布局 | Mask布局 | 功能类型 |
|-----|-----|-----|-----|
| 1 bit | `B8`、`B16`、`B32` | 与数据布局相同 | `PB8`、`PB16`、`PB32` |
| 8 bit | `B8` | `B8` | `NORM` |
| 8 bit | `B8_2VL` | `B16` | `PK` |
| 8 bit | `B8_4VL` | `B32` | `PK4` |
| 16 bit | `B16` | `B16` | `NORM` |
| 16 bit | `B16_2VL` | `B32` | `PK` |
| 32 bit | `B32` | `B32` | `NORM` |

对于逐元素Mask操作，非Mask向量操作数通常要求与结果使用相同布局，Mask操作数则按照结果布局对应的有效位宽选择状态。`VFCmpOp`有所不同：它的结果本身是`vector<...xi1>`，因此左右输入的布局由输入元素位宽和结果Mask状态共同决定。例如，8-bit输入在结果为`B16`时需要使用`B8_2VL`，在结果为`B32`时需要使用`B8_4VL`。

#### 类型转换求解

`solveTruncProblem()`按照源元素位宽和结果状态查找合法组合。通常使用`EVEN`功能类型，表示从对应宽槽位中提取有效通道：

| 源元素位宽 | 源布局 | 结果布局 | Mask布局 | 功能类型 |
|-----|-----|-----|-----|-----|
| 32 bit | `B32` | `B16_2VL` | `B32` | `EVEN` |
| 32 bit | `B32` | `B8_4VL` | `B32` | `EVEN` |
| 16 bit | `B16` | `B8_2VL` | `B16` | `EVEN` |
| 16 bit | `B16_2VL` | `B8_4VL` | `B32` | `EVEN` |

当Trunc操作带有`Layout_Change`属性时，表示其来自特定的VSSTB优化路径，求解器会使用`C2C`功能类型处理列到列转换，而不是普通的`EVEN`路径。

`solveExtProblem()`处理反方向的位宽扩展：

| 源元素位宽 | 源布局 | 结果布局 | Mask布局 | 功能类型 |
|-----|-----|-----|-----|-----|
| 8 bit | `B8_2VL` | `B16` | `B16` | `EVEN` |
| 8 bit | `B8_4VL` | `B16_2VL` | `B32` | `EVEN` |
| 8 bit | `B8_4VL` | `B32` | `B32` | `EVEN` |
| 16 bit | `B16_2VL` | `B32` | `B32` | `EVEN` |
| 32 bit | `B32` | `B32` | `B32` | `EVEN` |

其他整数和浮点类型转换由`solveTypeConvertProblem()`按位宽关系处理：源位宽大于结果位宽时按Trunc求解，源位宽小于结果位宽时按Ext求解，位宽相同时按Mask操作处理。

#### Interleave和DeInterleave求解

`solveLayoutChangeProblem()`处理`VFInterleaveOp`和`VFDeInterleaveOp`。首先要求两个结果的状态一致，然后根据`layout_change`属性确定约束方向：

| `layout_change` | 含义 | 典型约束 |
|-----|-----|-----|
| 无属性 | 原始Interleave/DeInterleave操作 | 源状态和结果状态相同，或选择`INTLV2/4` |
| `UNCHANGED` | 布局不改变 | 委托给默认求解，源状态等于结果状态 |
| `DENSE` | 从稀疏布局变为密集布局 | 结果为`B16`时，源为`B16_2VL` |
| `SPARSE` | 从密集布局变为稀疏布局 | 结果为`B16_2VL`时，源为`B16` |

典型映射如下：

```text
DENSE：
B16_2VL -> B16
B8_4VL  -> B8_2VL
B8_2VL  -> B8

SPARSE：
B16 -> B16_2VL
B8  -> B8_2VL
B8_2VL -> B8_4VL
```

无属性的原始Interleave操作中，若8-bit结果为`B8_2VL`或`B8_4VL`，分别选择`INTLV2`或`INTLV4`；DeInterleave操作则对应选择`DINTLV2`或`DINTLV4`。这里的`DENSE`和`SPARSE`描述的是相对于输入的变化方向，而不是结果布局的绝对名称。

#### 循环和函数调用求解

对于`scf.for`，循环区域迭代参数和循环结果表示同一个跨迭代传递的数据，必须保持相同布局：

1. 如果iterArg和forOp result都已有状态，则检查两者是否一致；
2. 如果只知道iterArg状态，将该状态传播给initArg；
3. 如果只知道forOp result状态，将结果状态传播给initArg；
4. `scf.yield`将forOp结果状态传播到对应的yield operand。

应用解时，Pass会重建`scf.for`，将原循环体合并到新循环体中，以便使用带布局的迭代参数和结果类型。

对于`func.call`，如果调用结果包含向量，求解器使用默认规则传播状态；如果调用没有向量结果但包含向量参数，则为每个向量参数枚举所有可能状态组合。应用解时，Pass在调用输入侧插入无布局的`VectorLayoutCastOp`，并在向量结果侧按照元素位宽补充默认布局，从而避免调用操作边界的类型属性不一致。

## 编译选项

| 选项 | 说明 |
|-----|-----|
| `-analyze-vector-layout` | 在`func::FuncOp`粒度执行向量布局分析 |
| `-remove-vector-layout-attr` | 移除布局属性和`VectorLayoutCastOp`，不属于AnalyzeVectorLayout本身的选项 |

AnalyzeVectorLayout没有额外的参数。调试时可以通过LLVM debug输出查看操作求解和候选解数量：

```bash
build/bin/bishengir-opt -analyze-vector-layout \
  -debug-only=analyze-vector-layout \
  --mlir-disable-threading=true input.mlir
```

## 使用约束

### 运行前提

用户需要保证：运行AnalyzeVectorLayout前，相关标准Vector和Arith操作已经完成HIVMAVE lowering，并且输入函数中的向量操作属于当前求解器支持的操作类型和布局组合。

### 已知限制

当前实现主要支持1、8、16和32 bit元素。64 bit元素在部分状态判断中按32 bit处理，但不代表所有64 bit向量操作组合都受到完整支持。

同一向量值的多个消费者必须能够同时满足布局约束。如果不同消费者要求互相冲突的`VecMemType`，且数据流中没有合法的布局转换操作，候选解会被删除。循环迭代参数与循环结果也必须能够保持一致布局，否则`scf.for`对应的候选解无效。

操作未完成HIVMAVE lowering、操作类型未加入`solveProblem`的`TypeSwitch`、当前操作不支持该`VecMemType`组合、多个消费者存在冲突，或者输入元素位宽不受支持时，Pass会调用`signalPassFailure()`并输出`No Solve`。

### 错误诊断

失败诊断会输出以下信息：

- 失败操作的位置和完整操作；
- 操作码、操作数类型和结果类型；
- 失败前剩余的候选解数量；
- 失败候选中已经记录的输入状态；
- 可能的原因和建议处理方式。

典型错误信息如下：

```text
No Solve
========== Vector Layout Analysis Failure ==========
Location: ...
Operation: ...
Opcode: ave.hir.vtrc
Operand types:
  vector<64xf32>
  vector<64xi1>
Result types:
  vector<64xf16>
Candidates in the solution space: 0
Possible causes and solutions:
  1. The operation may not have been lowered to the HIVMAVE dialect before VectorLayout analysis.
  2. The operation type may not be handled in solveProblem TypeSwitch.
  3. The specific VecMemType combination is not supported by this op.
  4. Conflicting layout requirements from multiple consumers.
  5. Unsupported element bitwidth (only 1/8/16/32 are supported).
======================================================
```

若需要支持新的HIVMAVE操作或新的布局组合，应在对应的`solveProblem()`函数中增加正向约束，明确操作数、结果和Mask之间的状态关系，并补充成功和失败测试。不能通过跳过该操作、默认放宽约束或将分析失败改写为pattern匹配失败来规避布局问题。
