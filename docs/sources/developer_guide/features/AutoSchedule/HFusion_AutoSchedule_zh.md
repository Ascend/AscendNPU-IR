# AutoSchedule

## HFusion AutoSchedule 自动融合与调度框架设计说明

HFusion 是 Bisheng IR 上针对算子融合和自动调度的高层框架，其中 **AutoSchedule** 模块负责在确定融合单元后，自动生成面向 Ascend NPU 的高效执行 schedule。设计目标包括：

- **自动化**：根据融合模式与算子特征，自动选择合适的调度策略与 Tiling 方案。
- **可扩展**：提供统一的调度基类与内核信息抽象，便于新增自定义调度策略。
- **高性能**：动态 shape 支持、reduce 多核并行等优化能力。
- **工程化**：依托 MLIR Transform Dialect，将 schedule 描述为可复用、可解释的 transform 序列。

---

### 1. 硬件背景

Ascend NPU 采用多级存储架构：全局内存（GM）容量大但访问延迟高，片上内存（如 L1、UB）延迟低但容量有限。为充分利用片上内存带宽并减少对全局内存的访问，需要：

- **最大化片上内存利用率**：通过大范围算子融合，将多个算子合并进同一内核执行，使中间结果在片上缓存中复用，减少 GM 读写次数。
- **满足硬件访存约束**：片上内存（如 UB）的访问需满足硬件要求，例如 **UB 访问需 32 字节对齐**（stride-align），否则会导致访存异常或性能下降。
- **适配 Tiling 与循环结构**：在 Tiling 与循环生成时，需考虑 stride/size/tile 对齐约束，确保生成的内核满足硬件规格。

AutoSchedule 的设计正是针对上述硬件特性，通过自动化调度与 Tiling 策略，在保证合法性的前提下最大化片上内存的利用效率。

---

### 2. 算法原理

AutoSchedule 的核心算法围绕 **大范围算子融合 + 轴映射驱动的循环生成 + Transform Dialect 融合执行** 展开：

1. **Dimension Analyzer 轴映射分析**
   - 通过 `DimensionAnalyzer` 分析 kernel 内各 op 相对于 anchor 的轴映射关系（`getCommonAxis`、`getNormalizedInterchange` 等）。
   - 建立各张量维度与 anchor 维度的对应关系，支持 broadcast、reduce、transpose 等复杂模式下的轴对齐分析。
   - 为后续 Tiling 与循环结构设计提供维度级别的精确信息。

2. **循环生成与算子融合**
   - 基于轴映射分析结果，为融合图生成统一循环结构，并将各 op 通过 MLIR Transform Dialect 的 `fuseIntoContaining`、`fuseLoops`、`coalesceLoops` 等原语融合进同一循环。
   - 在融合过程中显式考虑硬件架构约束：如 stride-align（32 字节对齐）、size-align、tile-align 等，确保生成的 IR 满足 Ascend NPU 的访存规范。

3. **Tiling 计算与选择**
   - 通过 `StmtExprBuilder` 与 `Expr` 系统，结合静态/动态 shape，生成多种候选 Tiling 方案（`TilingCases`）。
   - 在 Tiling 计算中应用 `getStrideAlignments()`、`getTileAlignments()` 等约束，对相关维度执行 `alignTo(alignment)`，输出满足 stride-align 等要求的合法 Tiling。
   - 根据代价、对齐等因素选择最优 `TilingKey`，并据此构造调度描述。

4. **Transform Dialect 解释执行**
   - 调度器不直接改写 IR，而是构造 Transform Dialect 程序；`AutoScheduleInterpreter` 将其翻译为具体 Transform 操作序列并应用到目标 IR 上，实现 schedule 的实际生效。

---

### 3. 接口说明

#### 3.1 代码位置

- **头文件（接口与抽象）**：`bishengir/include/bishengir/Dialect/HFusion/Transforms/AutoSchedule/`
- **实现文件**：`bishengir/lib/Dialect/HFusion/Transforms/AutoSchedule/`

#### 3.2 核心接口与抽象

| 类型       | 名称               | 说明                                                                 |
| ---------- | ------------------ | -------------------------------------------------------------------- |
| 基类       | `SchedulerBase`    | 所有调度器的抽象基类，封装统一调度主流程                             |
| 调度器     | `PureElemwiseScheduler` | 纯元素级算子融合策略                                                 |
| 调度器     | `AnyPBRScheduler`  | Pointwise/Broadcast/Reduce 等复杂融合的通用策略                      |
| 内核描述   | `KernelInfo`       | 融合内核的 IO、维度、对齐需求、多核能力等统一描述                     |
| 对齐接口   | `getStrideAlignments()` | 返回 stride 对齐约束（维度索引, 对齐粒度），如 32 字节对齐            |
| 对齐接口   | `getSizeAlignments()`  | 返回 size 维度对齐约束                                               |
| 对齐接口   | `getTileAlignments()`  | 返回 tile 维度对齐约束                                               |
| 轴分析     | `DimensionAnalyzer`| 提供 `getCommonAxis`、`getInterchange`、`getNormalizedInterchange` 等 |
| 调度原语   | `cacheRead` / `cacheWrite` | IO 缓存                                                              |
| 调度原语   | `tileUsingFor` / `tileUsingForAll` / `tileReductionUsingFor` | Tiling |
| 调度原语   | `fuseLoops` / `fuseIntoContaining` / `coalesceLoops` | 循环融合与合并     |
| 调度原语   | `setBufferSize`    | 资源约束                                                             |

#### 3.3 策略选择与调用链

- **Pass 入口**：AutoSchedule Pass 在 HFusion pipeline 中被触发，得到待处理的 `func::FuncOp` 与其融合信息。
- **策略选择**：在 `AutoScheduleBase.cpp::applySchedule()` 中根据 `FusionKind` 选择调度器：
  - `FusionKind::PureElemwise` → `PureElemwiseScheduler`
  - `FusionKind::AnyPB` / `FusionKind::LastAxisPBR` / `FusionKind::AnyPBR` → `AnyPBRScheduler`
- **主流程**：`runPreScheduleProcedure()` → `runScheduleProcedure()`（含 `calculateTilingImpl()`、`createScheduleImpl()`）→ `runPostScheduleProcedure()` → Transform Dialect 应用。

---

### 4. 约束能力

AutoSchedule 在调度与 Tiling 过程中显式处理以下约束：

| 约束类型      | 说明                                                                 | 相关接口 / 实现 |
| ------------- | -------------------------------------------------------------------- | ---------------- |
| **Stride 对齐** | UB 等片上内存访问需满足 32 字节对齐，避免非对齐访存                  | `getStrideAlignments()`，`calculateTilingImpl()` 中对维度 `alignTo()` |
| **Size 对齐**   | 部分 op（如 transpose、concat、cast）要求 tile/size 满足特定对齐    | `getSizeAlignments()` |
| **Tile 对齐**   | stride 与 size 约束的综合，作用于 Tiling 方案                        | `getTileAlignments()` |
| **Reduce 轴**   | reduce、broadcast、extract_slice、transpose 等 op 有最低维对齐要求  | `KernelInfo` 中各 op 的 stride-align 逻辑 |
| **片上 buffer** | 多 buffer 共存的容量与分配受 L1/UB 等限制                            | `setBufferSize`，`maxBufferCnt` |
| **多核 reduce** | 需满足特定条件才可启用多核并行 reduce                                | `analyzeMultiCoreReduceInfo()` |

这些约束在 `KernelInfo::getStrideAlignments()` 及具体调度器的 `calculateTilingImpl()` 中统一应用，保证生成 schedule 的合法性与硬件兼容性。

---

### 5. 架构总览

#### 5.1 核心组成模块

##### 调度基类与策略实现

- **SchedulerBase**：所有具体调度器的抽象基类（`AutoScheduleBase.h`），封装统一的调度主流程。
- **具体策略调度器**：
  - **PureElemwiseScheduler**：纯元素级算子融合策略（`PureElemwiseSchedule.h/cpp`）。
  - **AnyPBRScheduler**：面向 AnyPBR: Pointwise/Broadcast/Reduce 等 op 的通用策略（`AnyPBRSchedule.h/cpp`）。

##### 内核与 Tiling 抽象

- **KernelInfo**：融合内核的统一描述（`KernelInfo.h`），记录 IO、维度、对齐需求、多核能力等。
- **Tiling 抽象与工具**（`TilingUtils.h/cpp`）：
  - **TilingInfo**、**TilingStruct**、**TilingData**：描述单个或多种候选 Tiling 方案。
  - **Expr** / **StmtExprBuilder**：构建依赖静态 / 动态 shape 的 Tiling 表达式。

##### 调度操作实现

- **ScheduleOperations.cpp**：封装一组可复用的基础调度原语，包括：
    - IO 缓存：`cacheRead` / `cacheWrite`
    - Tiling：`tileUsingFor` / `tileUsingForAll` / `tileReductionUsingFor`
    - 循环变换：`fuseLoops` / `fuseIntoContaining` / `coalesceLoops`
    - 资源约束：`setBufferSize` 等。

##### 调度解释执行

- **AutoScheduleInterpreter.cpp**：将高层调度器构造的调度描述转换为 Transform Dialect 操作并应用到目标 IR 上，实现 schedule 的实际生效。

#### 5.2 策略选择与调用链

AutoSchedule 的整体调用链可以概括为：

- **Pass 入口**
  - AutoSchedule Pass 在 HFusion pipeline 中被触发，得到待处理的 `func::FuncOp` 与其融合信息。

- **策略选择与调度器构造**
  - 在 `AutoScheduleBase.cpp::applySchedule()` 中，根据融合类型 `FusionKind` 选择对应调度器：
    - `FusionKind::PureElemwise` → `PureElemwiseScheduler`
    - `FusionKind::AnyPB` / `FusionKind::LastAxisPBR` / `FusionKind::AnyPBR` → `AnyPBRScheduler`
  - 通过 `std::make_unique<...>(funcOp)` 构造实际调度器实例。

- **调度主流程（`SchedulerBase::runOnOperation()`）**
  - **前处理阶段（`runPreScheduleProcedure()`）**：
    - IO cache 插入、融合图结构与合法性分析。
    - 调用虚函数 `analyzeAndVerifyKernelImpl()` 由具体策略进行内核分析与限制检查。
  - **调度生成阶段（`runScheduleProcedure()`）**：
    - 调用虚函数 `calculateTilingImpl()`，生成 `TilingComputeFn`，计算候选 Tiling 方案。
    - 选择合适的 `TilingKey`（例如基于代价、对齐等因素）。
    - 调用虚函数 `createScheduleImpl()` 生成针对该 TilingKey 的调度描述。
    - 通过 `applyScheduleImpl()` 将调度描述交给 Transform 解释器执行。
  - **后处理阶段（`runPostScheduleProcedure()`）**：
    - 视策略需求进行结构优化后处理、统计信息收集等。

- **Transform Dialect 应用**
  - `AutoScheduleInterpreter` 解析调度描述，将其翻译为 Transform Dialect 操作序列，并对原始 HFusion IR 进行变换。

#### 5.3 关键数据结构

##### KernelInfo（内核信息描述）
  - 用于抽象单个融合内核的结构与约束，典型信息包括：
    - 输入 / 输出张量及其 shape / layout。
    - 各算子在融合图中的拓扑关系。
    - stride 对齐、size 对齐、tile 对齐等硬件友好约束。
    - 是否支持多核 reduce 以及可并行的维度信息。
  - 针对特定融合模式，可扩展派生类（如 `AnyPBRKernelInfo`）以添加模式特有的分析结果。

##### Tiling 描述（`TilingUtils.h`）

- **TilingData**：表示单个维度的 Tiling 参数，可为常量或表达式。
- **TilingStruct** / **TilingCases**：描述一组完整 Tiling 方案，以及多候选方案集合。
- **Expr** / **StmtExprBuilder**：
    - `DimSymbol`：对动态维度的符号抽象。
    - `Expr`：可进行加减乘除等运算，表达"维度 / 因子"、"对齐到某个粒度"等逻辑。
    - `StmtExprBuilder`：负责从 IR 中的 shape 信息、常量等构建 `Expr`，生成 host 侧可执行的 Tiling 函数。

##### ValueHandle 系列
  - 对 MLIR 中的 `Value`、函数参数、命名值等进行统一封装，提供统一接口访问与处理。
  - 常见类型包括 `NamedValueHandle`、`FuncArgHandle` 等。

---

### 6. 调度策略实现概述

#### 6.1 PureElemwise 调度策略

- **适用场景**：算子图主要由逐元素算子组成，无复杂 broadcast/reduce 结构。
- **实现位置**：`PureElemwiseSchedule.h/cpp`。
- **策略特点**：
  - 以规整 loop 结构为目标，采用简单、规则的 Tiling；
  - 更关注融合后的访存连续性与多级 cache 友好性；
  - 通过 `calculateTilingImpl()` 和 `createScheduleImpl()` 完成 Tiling 计算和调度原语串接。

#### 6.2 AnyPBR 调度策略（AnyPBRScheduler）

- **适用场景**：包含 broadcast、reduce等复杂模式的融合子图。
- **实现位置**：`AnyPBRSchedule.h/cpp`。
- **核心能力**：
  - **复杂 Tiling 计算**：
    - 在 `calculateTilingImpl()` 中综合考虑：
      - Kernel 的 stride 对齐要求；
      - 动态 shape 符号；
      - reduce 轴、broadcast 轴等特殊维度。
    - 通过 `StmtExprBuilder` 构建表达式，生成多种 Tiling 方案（`TilingCases`）。
  - **多核 reduce 分析与利用**：
    - `analyzeMultiCoreReduceInfo()` 分析是否满足多核 reduce 条件（见第 7.3 小节）。
  - **调度构造**：
    - 在 `createScheduleImpl()` 中，根据选定的 `TilingKey` 使能具体的调度策略，综合考虑：
      - 片上空间分配大小；
      - 对不同类型轴使能不同的 Tiling 策略；
      - loop fuse / coalesce；
      - 绑定多核等。

---

### 7. 关键优化能力

本节围绕 **stride-align 对齐优化、动态 shape 支持、reduce 多核并行** 三个方面，概述其设计与在 AutoSchedule 流程中的作用。

#### 7.1 Stride-Align 内存对齐优化

- **优化目标**：防止出现非对齐内存UB访问。
- **接口与数据来源**：
  - `KernelInfo::getStrideAlignments()`（`KernelInfo.h`）：
    - 返回一组 `(维度索引, 对齐粒度)`，描述某些维度在访存时需要对齐到的最小步长。
  - 其他相关接口：
    - `getSizeAlignments()`：size 维度对齐约束；
    - `getTileAlignments()`：tile 维度对齐约束。
- **在调度策略中的使用（以 AnyPBR 为例）**：
  - 在 `AnyPBRSchedule.cpp::calculateTilingImpl()` 中：
    - 先基于问题规模生成初始 Tiling 方案；
    - 遍历 `KernelInfo::getStrideAlignments()` 和 `getTileAlignments()` 中的信息，将相关维度 Tiling 大小通过 `alignTo(alignment)` 向上对齐；
    - 最终输出已满足 stride-align 约束的 `TilingCases`。
- **位置与时机**：
  - stride-align 处理发生在 **Tiling 计算阶段**，即 `SchedulerBase::runScheduleProcedure()` 中调用 `calculateTilingImpl()` 时；

#### 7.2 动态 Shape 支持

- **问题与要求**：
  - 在实际业务场景中，部分维度（如 batch size、高宽等）在编译期并非常量；
  - AutoSchedule 需要支持在 **运行时** 根据实际输入 shape 计算合适的 Tiling 参数。
- **表达式系统设计**：
  - `TilingUtils.h` 中的 **Expr** / **DimSymbol** / **StmtExprBuilder** 构成一套轻量表达式框架：
    - **DimSymbol**：
      - 表示某一维度的符号，例如 `N`、`H`、`W` 等；
      - 可由 `StmtExprBuilder::createDimSymbolExpr()` 等接口创建。
    - **Expr**：
      - 支持基本算术运算，可以表示诸如 `N / 4`、`min(N, 64)`、`(H * W) / factor` 等表达式。
    - **StmtExprBuilder**：
      - 负责从 IR 中的 shape 信息、常量等构建 `Expr`；
      - 在 host 侧生成具体的 Tiling 计算语句。
- **Host Tiling 函数生成与执行**：
  - `calculateTilingImpl()` 返回的 `TilingComputeFn` 通常是一个 lambda 或可调用对象；
  - 在 host 上执行该函数时，已知实际输入 shape，即可将 `DimSymbol` 映射为具体数值，求值出最终 Tiling；
  - 对于完全静态 shape，表达式可以在编译期直接折叠为常量。
- **配置与扩展**：
  - AutoSchedule 通过如 `AutoScheduleOptions::enableSymbolAnalysis` 等选项使能符号等价性分析，用于优化动态 Tiling 的编译

#### 7.3 多核 Reduce

- 多核 reduce 通过 `analyzeMultiCoreReduceInfo()` 进行分析，当 kernel 与 pattern 满足所需条件时启用（参见相关文档）。

---

### 8. 扩展开发指南：自定义调度策略

本节介绍如何在 HFusion AutoSchedule 框架中，**新增一种融合模式及其调度策略**，包括策略类实现、内核信息扩展以及注册流程。

#### 8.1 定义新的 FusionKind

- 在 HFusion 的枚举定义（如 `HFusionEnums.td`）中，新增一个融合类型枚举，例如：`FusionKind::MyKind`。
- 在融合分析与 pattern 匹配阶段，确保能识别并产出对应 `FusionKind::MyKind` 的融合单元，以便后续 AutoSchedule 正确选择调度器。

#### 8.2 继承 `SchedulerBase` 实现自定义调度器

- 在 `bishengir/include/bishengir/Dialect/HFusion/Transforms/AutoSchedule/` 下新增头文件（如 `MySchedule.h`），定义调度器类：

```cpp
class MyScheduler : public SchedulerBase {
public:
  explicit MyScheduler(func::FuncOp funcOpIn)
      : SchedulerBase(funcOpIn, FusionKind::MyKind) {}

  // 1. 内核分析与验证
  LogicalResult analyzeAndVerifyKernelImpl() override;

  // 2. Tiling 计算（静态 / 动态 shape）
  TilingComputeFn calculateTilingImpl() override;

  // 3. 调度创建（基于 Transform Dialect 原语）
  LogicalResult createScheduleImpl(TilingKey key,
                                   OpBuilder &opBuilder) override;

  // 4. 可选：前后处理扩展
  LogicalResult runPreScheduleProcedure(OpBuilder &opBuilder) override;
  LogicalResult runPostScheduleProcedure(OpBuilder &opBuilder) override;
};
```

- 在 `bishengir/lib/Dialect/HFusion/Transforms/AutoSchedule/` 下新增实现文件（如 `MySchedule.cpp`），完成各虚函数实现：

- **analyzeAndVerifyKernelImpl()**
  - 结合 `KernelInfoCollector` 收集内核信息（可选择复用现有 `KernelInfo` 或新增自定义子类）。
  - 检查融合图是否符合该策略假设（如算子类型、形状关系等）。

- **calculateTilingImpl()**
  - 构造并返回 `TilingComputeFn`：
    - 使用 `StmtExprBuilder` 构造静态 / 动态维度表达式；
    - 引入 stride-align、tile-align 等约束；
    - 为不同场景（小规模 / 大规模 / 高维 / 低维等）生成多套 `TilingCases` 以供选择。

- **createScheduleImpl(TilingKey key, OpBuilder &opBuilder)**
  - 根据选定的 `TilingKey`，依次调用调度原语：
    - IO 缓存：`cacheRead` / `cacheWrite`；
    - Tiling：`tileUsingFor` / `tileUsingForAll` / `tileReductionUsingFor`；
    - 融合与循环优化：`fuseLoops`、`fuseIntoContaining`、`coalesceLoops`；
  - 保证生成的 Transform 序列语义正确且与 `KernelInfo` 中的分析结果一致。

- **runPreScheduleProcedure()** / **runPostScheduleProcedure()**（可选）
  - 在通用流程基础上添加策略特有的前后处理逻辑，例如：
    - 特殊的 pattern 归一化；
    - 调度结果校验与统计输出。

#### 8.3 扩展 `KernelInfo`（可选）

若新策略需要额外的结构化信息，可通过继承 `KernelInfo` 扩展：

```cpp
class MyKernelInfo : public KernelInfo {
public:
  MyKernelInfo(MLIRContext *ctx)
      : KernelInfo(FusionKind::MyKind, ctx) {}

  static bool classof(const KernelInfo *T) {
    return T->getFusionKind() == FusionKind::MyKind;
  }

  // 在此添加特定融合模式所需的附加字段与查询接口
};
```

同时，在 `KernelInfoCollector` 实现中增加对 `FusionKind::MyKind` 的处理逻辑，构造并填充 `MyKernelInfo` 实例，以便调度器在 `analyzeAndVerifyKernelImpl()` 和 `calculateTilingImpl()` 中使用。

#### 8.4 将新策略注册到 AutoSchedule 框架

在 `AutoScheduleBase.cpp::applySchedule()` 中，向 `switch (fusionKind)` 语句新增分支：

```cpp
case FusionKind::MyKind:
  scheduler = std::make_unique<MyScheduler>(funcOp);
  break;
```

确保对应的 `MySchedule.cpp` 已被纳入构建系统并链接进 HFusion Transform 模块后，即可在 pipeline 中使用该新策略。

#### 8.5 调度原语（Schedule API）使用速览

在 `createScheduleImpl()` 内，可直接复用框架提供的调度 API（位于 `ScheduleOperations.cpp`）：

- **IO 缓存与 buffer 管理**
  - `cacheRead`
  - `cacheWrite`
  - `setBufferSize`
- **Tiling 与循环结构控制**
  - `tileUsingFor`
  - `tileUsingForAll`
  - `tileReductionUsingFor`
- **循环融合与合并**
  - `fuseLoops`
  - `fuseIntoContaining`
  - `coalesceLoops`
- **多核绑定**
  - `bindLoopToMulticore`
  - 并可参考 AnyPBR 策略中的 `getMultiCoreNum` 等函数确定核数配置。

通过组合上述调度原语，开发者可以在统一架构下，为新融合模式实现灵活且高效的 schedule 策略。

---

### 9. 内部机制简要说明

#### 9.1 ValueHandle 抽象体系

- 通过 `ValueHandle` 系列类型，将 MLIR 中不同来源（Value、Argument、命名值等）的对象统一抽象；
- 提供一致的访问与操作接口，使调度器代码避免直接耦合底层 IR 细节；
- 有助于在调度描述构造与 Transform 解释阶段维护代码简洁性与可维护性。

#### 9.2 Transform Dialect 集成与解释执行

- AutoSchedule 并不直接在调度器内部改写算子 IR，而是构造一段 **Transform Dialect 程序**；
- `AutoScheduleInterpreter.cpp` 负责：
  - 接收由各调度器生成的调度描述；
  - 将其翻译成具体的 Transform Dialect 操作序列；
  - 将这些操作应用到目标 `func::FuncOp` 上，完成实际 IR 变换；
- 这样的设计使得：
  - 调度逻辑与 IR 变换细节解耦；
  - 调度过程可追踪（可打印 Transform 程序）、可调试、可复用。

#### 9.3 Tiling 计算框架

- 通过 `TilingInfo` + `Expr` 系统，将 **维度大小、对齐规则、动态 shape** 等统一抽象为表达式。
- 对于静态 shape：
  - 表达式可以在编译期求值，折叠为常量 Tiling 参数；
- 对于动态 shape：
  - 在 host 生成的 Tiling 函数中，根据实际输入 shape 求值；
  - 同一套表达式既服务于静态场景，也兼容动态场景，减少代码分叉。
