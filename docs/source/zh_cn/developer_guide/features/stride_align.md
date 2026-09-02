# 存储对齐

本文介绍HIVM中的存储对齐（Stride Alignment）机制，包括 `hivm-pre-mark-stride-align`、`hivm-mark-stride-align` 和 `hivm-enable-stride-align` 三个Pass的硬件背景、算法原理、接口说明和使用约束。

存储对齐分为"标记"（mark）与"使能"(enable)两个阶段：标记阶段分析每个HIVM算子的memref操作数，确定需要对齐的维度和对齐字节数，并以 `annotation.mark` 注解的形式写入IR；使能阶段读取这些注解，将标记信息传播到根分配点（`memref.alloc`），对分配的内存形状进行填充（padding），使各行起始地址满足硬件对齐要求，再通过 `memref.subview` 切回原始逻辑形状。

## 硬件背景

昇腾硬件的片上内存（UB、L1、L0C等）对数据访问有严格的地址对齐要求。各类Buffer的对齐要求与功能如下：

| Buffer | 对齐要求 | 功能 |
|-----|-----|-----|
| Unified Buffer (UB) | 32字节对齐 | 通用缓存空间，主要用于向量和标量运算 |
| L1 Buffer | 32字节对齐 | 暂存feature map等卷积使用到的数据 |
| L0A Buffer | 512字节对齐 | 暂存矩阵运算的左矩阵（feature map） |
| L0B Buffer | 512字节对齐 | 暂存矩阵运算的右矩阵（weight） |
| L0C Buffer | 512字节对齐 | 暂存矩阵运算的中间结果和输出矩阵 |
| BT Buffer | 64字节对齐 | BiasTable Buffer，存放矩阵运算中的Bias |
| FP Buffer | 64字节对齐 | Fixpipe Buffer，存放量化参数、Relu参数等 |

除了Buffer自身的起始地址需要对齐外，当memref的最低维（last dim）不连续时——即次低维（sub-tail dim）的stride不等于最低维的size——硬件在逐行访问数据时，每一行的起始地址也需要满足对齐约束。例如，对于 `memref<3x13xi8, strided<[16, 1]>>`，最低维size为13但stride为16，最低维不连续；16字节的行宽不满足硬件对齐要求（32字节），就需要在分配时将最低维填充到对齐大小32。

DMA搬运指令和Vector计算指令对stride有各自的约束：

- DMA指令（`copy_gm_to_ub`、`copy_ub_to_gm`）要求被搬运的UB memref各行起始地址对齐到32字节（或硬件对应的对齐粒度）。
- Fixpipe（L0C→UB搬运）有更复杂的M/N轴对齐约束，取决于数据位宽、ChannelSplit模式、NZ2ND/NZ2DN转换模式等。
- `vsstb`（stride store with broadcast）指令建议stride对齐到 (512\*N + 32) 字节，以避免Bank Conflict。

## 相关术语说明

| 术语 | 含义 |
|-----|-----|
| stride_align_dims / stride_align_value_in_byte | 两个注解属性，分别记录需要对齐的维度索引（从0开始）和对齐字节数。通过 `annotation.mark` op附加到memref Value上。例如 `{hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>}` 表示对第1维的stride对齐到32字节。|
| last discontinuous dim（最后不连续维度） | 从最低维（最右侧维度）向高维方向扫描，第一个stride≠size的维度。该维度的stride即为行宽，需要满足对齐要求。如果最低维stride已经是unit stride（即stride==size，最低维连续），则需要继续向高维查找。|
| 对齐传播（Align Propagation） | 标记可能落在 `memref.subview`、`memref.cast` 等view-like op的结果上，而非直接落在 `memref.alloc` 上。对齐传播将标记沿着view链向上回溯到根分配点，同时向下传播到叶子操作数，并在同一算子的多个操作数间取并集，确保所有相关Buffer使用一致的对齐信息。|
| SkipStrideAlignForVLoad | 预标记阶段（PreMarkStrideAlign）在根alloc上设置的跳过标记。某些Buffer由DMA加载后会在Vector Function中被 `vlds`（仅支持连续加载）读取，如果对其做stride对齐会产生行间间隙，`vlds` 无法处理非连续布局，因此需要跳过。|

## 算法原理

### 软件背景

在PlanMemory完成地址分配之前，输入IR中的 `memref.alloc` 仅声明了逻辑形状和所需内存大小，不包含地址信息。如果某个Buffer的逻辑形状在次低维未对齐（如 `memref<37x5x3xi32, strided<[15, 3, 1]>>`，次低维stride=15），DMA模版实现中转为2D处理时，各行的起始地址不对齐，会导致功能异常。

存储对齐机制通过以下方式解决该问题：

1. 在标记阶段，分析每个HIVM算子的memref操作数，找出"最后一个不连续维度"（last discontinuous dim），即从最低维向上第一个stride≠size的维度。该维度的stride就是需要被对齐的行宽。
2. 在使能阶段，将根 `memref.alloc` 的对应维度填充到满足硬件对齐要求的size（如将 `memref<37x5x3xi32>` 的最低维从3填充到8，使stride=8\*4字节=32字节满足对齐），再通过 `memref.subview` 切回原始逻辑形状。

这样，底层分配的内存布局满足硬件对齐约束，而上层算子仍按原始逻辑形状访问数据。

### Pass Pipeline

三个Pass在Pipeline中按以下顺序执行：

```text
hivm-align-alloc-size          // 调整特殊算子的alloc大小
  └─ hivm-pre-mark-stride-align   // [仅reg-based] 预分析：标记需跳过对齐的alloc
  └─ hivm-mark-stride-align       // 标记阶段：为算子操作数添加stride_align注解
hivm-enable-stride-align        // 使能阶段：传播注解 + 重分配内存
```

- 标记阶段在FuncOp级别运行（`func::FuncOp`），可并行处理不同函数。
- 使能阶段在ModuleOp级别运行（`mlir::ModuleOp`），因为对齐传播可能跨 `func.call` 边界影响Vector Function被调用者，FuncOp并行嵌套会引发数据竞争。

## hivm-pre-mark-stride-align

### 功能概述

该Pass仅对reg-based架构（如Ascend 950PR/Ascend 950DT）生效，是标记阶段的前置分析步骤。

某些Buffer由DMA（`hivm.hir.load`）从GM搬运到UB后，会在Vector Function（通过 `func.call` 调用的VF）中被 `vector.transfer_read`（即vload）读取。如果对这些Buffer做stride对齐，会在各行之间插入填充间隙，使UB中的memref布局变为非连续。而 `vlds` 指令要求连续元素的扁平指针加载，无法处理非连续布局。

因此，PreMarkStrideAlign提前检测这类Buffer，在其根alloc上添加 `hivm.skip_stride_align_for_vload` 注解，供后续MarkStrideAlign跳过。

### 核心流程

1. 遍历函数内的所有 `hivm.hir.load`（DMA加载）操作。
2. 对每个load的结果Buffer，通过数据流追踪（经过 `memref.subview`、`memref.cast` 等view-like op，以及 `func.call` 调用边界），判断其是否最终流向某个VF被调用者中的 `vector.transfer_read`。
3. 若确认流向vload，则回溯到根alloc（`memref.alloc`），在其后插入 `annotation.mark` 并设置 `hivm.skip_stride_align_for_vload` 属性。

> 注：当VF被调用者仅有声明（declaration，无函数体，如SplitSimtModule后的情况）时，无法判断参数是否被vload，因此跳过该Buffer的标记。

## hivm-mark-stride-align

### 功能概述

该Pass遍历所有HIVM结构化算子（`HIVMStructuredOp`），自动分析其memref操作数的布局，找出最后不连续维度，并为UB空间的操作数创建 `annotation.mark` 注解，记录对齐维度和对齐字节数。

### 核心流程

#### 1. 通用结构化算子标记

对每个 `HIVMStructuredOp`，执行以下步骤：

1. 跳过非Buffer化（tensor类型）或仅含GM操作数的算子（GM无需对齐）。
2. 收集算子的memref操作数类型，跳过全rank-0或全shape为1的操作数。
3. 判断是否为UB DMA操作（`hivm.hir.load`/`store`），DMA操作的对齐判断规则与普通计算op不同。
4. 根据架构走不同分析路径：
   - **reg-based（A5/950）**：由于A5的 `hfusion-flatten` 已合并轴，直接将memref视为已flatten的状态，用 `getLastDiscontinuousDimRegBased` 查找最后不连续维度。对Fixpipe有专门的对齐约束计算。
   - **非reg-based（A2/A3）**：通过 `FlattenInterface` 的 `getFlattened` 获取flatten后的关联组和类型，用 `getLastDiscontinuousDim` 在flatten后的类型上查找，再映射回原始维度。
5. 取UB空间的操作数（`getTargetSpaceOperands(UB)`），调用 `markAlignedDim` 为每个操作数创建对齐注解。

#### 2. 最后不连续维度查找

`getLastDiscontinuousDim` 的查找逻辑：

- 如果任何memref的最低维stride不为1（最低维不连续），则最低维就是对齐目标维度。
- 如果最低维stride为1（最低维连续），则在更高维中查找最后一个stride≠size的维度。
- 对UB DMA操作（`copy_gm_to_ub`等），在Ascend 950PR/Ascend 950DT上只考虑最低维的stride，如果次低维已对齐且最低维无tail-jump问题，则无需对齐。
- 对1D memref，不存在不连续维度，返回 `nullopt`。

#### 3. Fixpipe特殊处理

Fixpipe（L0C→UB搬运）的对齐约束更为复杂，取决于：

| 参数 | 说明 |
|------|------|
| DataWidthType | 数据位宽：B4(4bit)、B8、B16、B32 |
| ChannelSplit | 是否通道分割（CS_N/CS_Y） |
| NZ2ND | 是否NZ→ND转换 |
| NZ2DN | 是否NZ→DN转换 |
| LoopEnhance | 是否循环增强 |
| DualDstMode | 双目的模式：N、SplitN、SplitM |

N轴对齐要求（部分示例）：

| 数据位宽 | NZ2ND | 对齐元素数 |
|----------|-------|-----------|
| B16 | ND_N | 16 |
| B8 | ND_N | 16 |
| B8 | ND_Y | 32 |
| B4 | ND_Y | 64 |
| B32 | ND_N | 16 |
| B32 | ND_Y | 8 |

M轴对齐要求（部分示例）：

| 数据位宽 | NZ2DN | 对齐元素数 |
|----------|-------|-----------|
| B16 | DN_Y | 16 |
| B8 | DN_Y | 32 |
| B4 | DN_Y | 64 |
| B32 | DN_N(SplitM) | 2 |
| B32 | DN_Y | 8 |

此外，对3D-dot的Fixpipe，还需对第0维（tiling维度）进行对齐。

#### 4. vsstb Bank Conflict避免

当VF函数中存在 `vector.transfer_write` 且适合stride store（`vsstb`）时，MarkStrideAlign会对 `func.call` 的对应参数额外添加更高的对齐要求，以避免Bank Conflict。对齐大小计算为：

```text
alignSize = shape[subTailDim] * 16 / gcd(shape[subTailDim], 16) + 1
```

其中 `subTailDim = rank - 2`（倒数第二维），16 = 512字节 / 32字节（对齐粒度与最低维size的比值）。最终对齐字节数需乘以数据类型宽度。

#### 5. rank-reducing subview修复

当Store操作的UB操作数经过trailing-unit rank-reduced subview（如 `memref<3x1x1xi32>` → `memref<1x1xi32>`）时，对齐维度可能落在被丢弃的维度上。此时需要将标记目标从subview结果回溯到subview的source，并调整对齐维度索引。

#### 6. Vector Function参数标记（仅reg-based）

对 `func.call` 的参数，如果尚未被标记且属于本地Buffer，则走与结构化算子相同的最后不连续维度分析，为参数添加对齐注解。同时跳过被PreMarkStrideAlign标记为 `SkipStrideAlignForVLoad` 的参数。

### 标记效果示例

```text
// 标记前：UB alloc的最低维不连续
%alloc = memref.alloc() : memref<37x5x3xi32, #hivm.address_space<ub>>
hivm.hir.load ins(%gm : ...) outs(%alloc : memref<37x5x3xi32, strided<[15, 3, 1]>, ...>)

// 标记后：在第2维（stride=15）添加32字节对齐注解
annotation.mark %alloc {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>}
```

## hivm-enable-stride-align

### 功能概述

该Pass读取标记阶段产生的 `stride_align` 注解，通过多轮传播将注解汇总到根分配点，然后对 `memref.alloc`/`memref.alloca` 进行形状填充（padding），使各行起始地址满足硬件对齐要求，最后通过 `memref.subview` 切回原始逻辑形状。

### 核心流程

1. 规范化对齐信息（NormalizeAlignInfoPattern）
   对每个 `annotation.mark` 上的 `stride_align_dims` 和 `stride_align_value_in_byte` 属性进行排序，确保维度索引按升序排列，便于后续传播时的并集和比较操作。

2. 向上传播到根Alloc（PropagateAlignUpToRootAllocationPattern）
   将落在 `memref.subview`、`memref.cast`、`memref.collapse_shape`、`memref.expand_shape`、`memref.reshape`、`memref.view` 等view-like op结果上的对齐标记，沿着定义链向上回溯到根 `memref.alloc`/`memref.alloca`，并将对齐信息附加到根alloc上。

   > 约束：memref不应由SCF op返回（如 `scf.if` yield memref），这类控制流应在标记阶段前由其他pass下沉。

3. 向下传播到叶子操作数（PropagateAlignDownToLeafOperandsPattern）
   将根alloc上的对齐信息向下传播到所有直接或间接使用该alloc的叶子操作数。同时为缺少标记的alloc补充创建 `annotation.mark`（`AddAlignAnnotationMarkForAlloc`）。

4. 算子操作数间传播（PropagateAlignAmongOperationOperands）
   对每个HIVM Vector算子和 `hivm.hir.copy`，收集其UB空间非临时操作数的对齐信息，取并集后传播到未标记或标记不完全的操作数。例如：

   ```text
   // op有三个UB操作数A, B, C，A和B有对齐标记
   // 传播后C也获得相同的对齐标记
   %A = ... {stride_align_dims=[1], value=[32]}
   %B = ... {stride_align_dims=[1], value=[32]}
   hivm.hir.vadd ins(%A, %B) outs(%C : ...)  // C获得 {stride_align_dims=[1], value=[32]}
   ```

   Copy操作的第一操作数（源）不参与传播，仅传播到目的操作数。

5. 迭代至不动点
   步骤3→4→2反复迭代，直到对齐信息不再变化或达到最大迭代次数（10次）。若超过最大迭代仍未收敛，报错退出。

6. 清除注解（RemoveAlignMarkPattern）
   传播完成后，移除所有 `annotation.mark` 上的 `stride_align_dims` 和 `stride_align_value_in_byte` 属性，保留空的mark op（或由后续pass清理）。该步骤只清除挂在 `annotation.mark` 上的传播中间信息，不会清除已经传播到 `memref.alloc`/`memref.alloca` 上的对齐属性；后续 `EnableAlignAllocation` 仍依赖alloc上的对齐属性完成内存重分配。

7. 使能对齐分配（EnableAlignAllocation）
   对每个带有 `stride_align_dims` 属性的 `memref.alloc`/`memref.alloca`：

   - 7.1 收集对齐单元（`collectAlignUnits`）：将对齐字节数转换为对齐元素数（除以元素位宽），考虑架构差异。
   - 7.2 计算对齐后形状（`calculateAlignedShape`）：对每个需要对齐的维度，将其size向上取整到对齐单元的倍数（`AlignUp`）。同时计算原始逻辑形状（sub_shape）。
   - 7.3 创建新的alloc，类型为对齐后的形状和identity layout。
   - 7.4 创建 `memref.subview`，从对齐后的alloc切出原始逻辑形状。
   - 7.5 调用 `replaceAndPropagateMemRefType` 将原alloc的所有使用替换为新subview，并传播memref类型变化。
   - 7.6 同步更新 `buffer_size_in_byte` 注解（按对齐后与对齐前的静态体积比缩放）。
   - 7.7 删除原alloc。

   若对齐后形状与原始形状相同（无需padding），仅移除对齐属性，不创建新alloc。

8. AIC→AIV对齐传播（仅Ascend 950PR/Ascend 950DT）
   对于TightlyCoupledBuffer（紧密耦合Buffer），AIC和AIV共享同一物理内存。如果AIC侧的alloc做了stride对齐，AIV侧的对应alloc也需要相同对齐。`AddStrideAlignInfoForAiv` 通过TightlyCoupledBuffer的ID匹配AIC和AIV的alloc，将对齐信息从AIC传播到AIV。

9. 后处理
   - `handlePropagateFailure`：处理传播失败（unrealized_conversion_cast）的op。
   - `materializeRemainingStaticUBLayoutCasts`（仅reg-based）：将stride-align产生的静态UB layout-changing cast物化。
   - 设置 `hivm.storage_aligned` 属性（仅非reg-based）：标记函数已完成存储对齐。

### 使能效果示例

```text
// 使能前：alloc带对齐标记
%alloc = memref.alloc() {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<37x5x3xi32, #hivm.address_space<ub>>
// 第1维size=3，stride需要32字节对齐 → 32/4(i32)=8元素 → 最低维填充到8

// 使能后：分配对齐后的形状，再subview切回原始形状
%aligned_alloc = memref.alloc() : memref<37x5x8xi32, #hivm.address_space<ub>>
%subview = memref.subview %aligned_alloc[0, 0, 0] [37, 5, 3] [40, 8, 1] : memref<37x5x8xi32, ...> to memref<37x5x3xi32, strided<[40, 8, 1]>, ...>
// stride从[15,3,1]变为[40,8,1]，各行起始地址满足32字节对齐
```

## 编译选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `enable-hivm-auto-storage-align` | true | Pipeline选项，控制是否在 `alignStoragePipeline` 中执行标记和使能阶段 |

三个Pass本身不暴露独立的命令行选项，均通过 `enableHIVMAutoStorageAlign` 选项统一控制。关闭该选项后，`hivm-pre-mark-stride-align`、`hivm-mark-stride-align` 不执行，但 `hivm-enable-stride-align` 仍会执行（此时因无标记注解，不会产生实际的内存重分配）。

## 使用约束

1. **需在Buffer化之后运行**：标记阶段要求算子已通过 `hasPureBufferSemantics()` 检查，即输入IR已完成tensor→memref的Bufferization。
2. **需在Flatten之后运行**：reg-based架构要求 `hfusion-flatten` 已完成轴合并；非reg-based架构通过 `FlattenInterface::getFlattened` 在标记时获取flatten信息。
3. **memref不应由SCF op返回**：如 `scf.if` yield memref 的场景，对齐传播无法处理，需在标记阶段前由其他pass将控制流下沉。
4. **使能阶段可能增大内存占用**：对齐填充会使alloc的物理尺寸增大（对齐维度size变大），可能影响后续PlanMemory的内存分配。用户需保证对齐后的总内存需求不超过硬件空间上限。
5. **传播收敛性**：操作数间传播最大迭代10次，若未收敛会编译失败。通常合理的IR结构不会触发此限制。
6. **vsstb与FlattenOps的已知冲突**：vsstb的bank conflict对齐可能将连续形状变为非连续形状，与依赖连续形状特征的 `memref.collapse_shape`/`memref.reshape` 产生冲突，可能导致后续FlattenOps编译错误。
