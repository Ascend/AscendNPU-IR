# Automatic Fusion and Scheduling

## Framework Overview

**HFusion** is a high-level framework on **Bisheng IR** for operator fusion and automatic scheduling, in which the **AutoSchedule** module is responsible for automatically generating efficient execution schedules for **Ascend NPU** after the fusion unit is determined.

### Design Goals

- Automation: Automatically select the appropriate scheduling policy and Tiling scheme based on the fusion pattern and operator characteristics.
- Extensibility: Provide a unified scheduling base class and kernel information abstraction to facilitate the addition of custom scheduling policies.
- High performance: Optimization capabilities such as dynamic shape support and multi-core parallel reduce.
- Engineering: Leverage the MLIR Transform Dialect to describe the schedule as a reusable and interpretable transform sequence.

### Hardware Background

Ascend NPU adopts a multi-level memory architecture: global memory (GM) offers large capacity but high access latency, while on-chip memory (such as L1 and UB) offers low latency but limited capacity. To fully utilize the on-chip memory bandwidth and reduce accesses to global memory, the following are required:

- **Maximize on-chip memory utilization**: Through large-scale operator fusion, multiple operators are merged into the same kernel for execution, so that intermediate results are reused in the on-chip cache and the number of GM read/write operations is reduced.
- **Satisfy hardware memory access constraints**: Accesses to on-chip memory (such as UB) must meet hardware requirements. For example, UB accesses require 32-byte alignment (`stride-align`); otherwise, memory access exceptions or performance degradation may occur.
- **Adapt Tiling and loop structures**: During Tiling and loop generation, the `stride`/`size`/`tile` alignment constraints must be considered to ensure that the generated kernels meet hardware specifications.

The design of AutoSchedule targets the hardware characteristics described above. Through automated scheduling and Tiling policies, it maximizes on-chip memory utilization efficiency while ensuring legality.

### Core Algorithm Flow

The AutoSchedule core algorithm consists of three parts: large-scale operator fusion, axis-mapping-driven loop generation, and Transform Dialect fusion execution. The overall flow is as follows:

1. **Dimension Analyzer axis mapping analysis**
   - The `DimensionAnalyzer` analyzes the axis mapping relationship of each op in the kernel relative to the anchor (`getCommonAxis`, `getNormalizedInterchange`, and so on).
   - It establishes the correspondence between each tensor dimension and the anchor dimension, supporting axis alignment analysis under complex patterns such as broadcast, reduce, and transpose.
   - It provides precise dimension-level information for subsequent Tiling and loop structure design.

2. **Loop generation and operator fusion**
   - Based on the axis mapping analysis results, a unified loop structure is generated for the fusion graph, and each op is fused into the same loop through primitives such as `fuseIntoContaining`, `fuseLoops`, and `coalesceLoops` of the MLIR Transform Dialect.
   - Hardware architecture constraints are explicitly considered during fusion, such as stride-align (32-byte alignment), size-align, and tile-align, ensuring that the generated IR satisfies the memory access specifications of the Ascend NPU.

3. **Tiling computation and selection**
   - Through the StmtExprBuilder and the Expr system, combined with static/dynamic shapes, multiple candidate Tiling schemes (TilingCases) are generated.
   - In Tiling computation, constraints such as `getStrideAlignments()` and `getTileAlignments()` are applied, and `alignTo(alignment)` is executed on the relevant dimensions to output valid Tiling that satisfies requirements such as stride-align.
   - The optimal `TilingKey` is selected based on factors such as cost and alignment, and the scheduling description is constructed accordingly.

4. **Transform Dialect Interpretation and Execution**

   The scheduler does not directly rewrite the IR; instead, it constructs a Transform Dialect program. The AutoScheduleInterpreter translates it into a concrete Transform operation sequence and applies it to the target IR, thereby making the schedule take effect.

## Code Interfaces and Basic Components

**Code Location**:

- **Header files (interfaces and abstractions)**: `bishengir/include/bishengir/Dialect/HFusion/Transforms/AutoSchedule/`
- **Implementation files**: `bishengir/lib/Dialect/HFusion/Transforms/AutoSchedule/`

**Core Interfaces and Abstractions**:

| Type       | Name               | Description                                                                 |
| ---------- | ------------------ | -------------------------------------------------------------------- |
| Base class       | `SchedulerBase`    | Abstract base class of all schedulers, encapsulating the unified scheduling main flow                             |
| Scheduler     | `PureElemwiseScheduler` | Pure element-wise operator fusion policy                                                 |
| Scheduler     | `AnyPBRScheduler`  | General policy for complex fusion such as `Pointwise`/`Broadcast`/`Reduce`                |
| Kernel description   | `KernelInfo`       | Unified description of the fused kernel's IO, dimensions, alignment requirements, and multi-core capabilities                     |
| Alignment interface   | `getStrideAlignments()` | Returns `stride` alignment constraints (dimension index, alignment granularity), such as 32-byte alignment            |
| Alignment interface   | `getSizeAlignments()`  | Returns `size` dimension alignment constraints                                               |
| Alignment interface   | `getTileAlignments()`  | Returns `tile` dimension alignment constraints                                               |
| Axis analysis     | `DimensionAnalyzer` | Provides `getCommonAxis`, `getInterchange`, `getNormalizedInterchange`, etc. |
| Scheduling primitive   | `cacheRead` / `cacheWrite` | IO cache                                                              |
| Scheduling primitive   | `tileUsingFor` / `tileUsingForAll` / `tileReductionUsingFor` | Tiling |
| Scheduling primitive   | `fuseLoops` / `fuseIntoContaining` / `coalesceLoops` | Loop fusion and merging     |
| Scheduling primitive   | `setBufferSize`    | Resource constraint                                                             |

**Policy Selection and Call Chain**:

- **Pass entry**: The AutoSchedule Pass is triggered in the HFusion pipeline and obtains the `func::FuncOp` to be processed and its fusion information.
- **Policy selection**: In `AutoScheduleBase.cpp::applySchedule()`, the scheduler is selected based on `FusionKind`:
    - `FusionKind::PureElemwise` → `PureElemwiseScheduler`
    - `FusionKind::AnyPB` / `FusionKind::LastAxisPBR` / `FusionKind::AnyPBR` → `AnyPBRScheduler`
- **Main flow**: `runPreScheduleProcedure()` → `runScheduleProcedure()` (including `calculateTilingImpl()` and `createScheduleImpl()`) → `runPostScheduleProcedure()` → Transform Dialect application.

## Usage Constraints

AutoSchedule explicitly handles the following constraints during scheduling and Tiling:

| Constraint Type      | Description                                                                 | Related Interface / Implementation |
| ------------- | -------------------------------------------------------------------- | ---------------- |
| **Stride Alignment** | On-chip memory access such as UB must satisfy 32-byte alignment to avoid unaligned access.                  | `getStrideAlignments()`, `alignTo()` on dimensions in `calculateTilingImpl()` |
| **Size Alignment**   | Some ops (such as `transpose`, `concat`, and `cast`) require tile/size to satisfy specific alignment. | `getSizeAlignments()` |
| **Tile Alignment**   | A combination of stride and size constraints, applied to the Tiling scheme.                        | `getTileAlignments()` |
| **Reduce Axis**   | Ops such as `reduce`, `broadcast`, `extract_slice`, and `transpose` have minimum-dimension alignment requirements. | The stride-align logic of each op in `KernelInfo` |
| **On-chip Buffer** | The capacity and allocation of multiple coexisting buffers are limited by L1/UB and other constraints.                            | `setBufferSize`, `maxBufferCnt` |
| **Multi-core Reduce** | Specific conditions must be met before multi-core parallel `reduce` can be enabled.                              | `analyzeMultiCoreReduceInfo()` |

These constraints are uniformly applied in `KernelInfo::getStrideAlignments()` and in the `calculateTilingImpl()` of specific schedulers, ensuring the legality and hardware compatibility of the generated schedule.

### Architecture Overview

#### Four Core Modules

**1. Scheduling Base Class and Policy Implementation**

- **SchedulerBase**: The abstract base class of all concrete schedulers (`AutoScheduleBase.h`), which encapsulates the unified scheduling main flow.

- Concrete policy schedulers:
    - `PureElemwiseScheduler`: The pure element-wise operator fusion policy (`PureElemwiseSchedule.h/cpp`).
    - `AnyPBRScheduler`: A general policy for AnyPBR (operators such as `Pointwise`/`Broadcast`/`Reduce`) (`AnyPBRSchedule.h/cpp`).

**2. Kernel and Tiling Abstraction**

- **KernelInfo**: The unified description of a fused kernel (`KernelInfo.h`), which records I/O, dimensions, alignment requirements, multi-core capabilities, and so on.

- Tiling abstraction and utilities (`TilingUtils.h/cpp`):
    - **TilingInfo**, **TilingStruct**, and **TilingData**: Describe a single or multiple candidate tiling schemes.
    - Expr / StmtExprBuilder: Build Tiling expressions that depend on static/dynamic shapes.

**III. Scheduling Operation Implementation**

`ScheduleOperations.cpp`: Encapsulates a set of reusable basic scheduling primitives, including:

- IO cache: `cacheRead` / `cacheWrite`

- Tiling: `tileUsingFor` / `tileUsingForAll` / `tileReductionUsingFor`

- Loop transformation: `fuseLoops` / `fuseIntoContaining` / `coalesceLoops`

- Resource constraints: `setBufferSize`, etc.

**IV. Scheduling Interpretation and Execution**

`AutoScheduleInterpreter.cpp`: Converts the scheduling description constructed by the high-level scheduler into Transform Dialect operations and applies them to the target IR, thereby making the schedule take effect.

#### Policy Selection and Call Chain

The overall call chain of AutoSchedule can be summarized as follows:

1. **Pass entry**

   The AutoSchedule Pass is triggered in the HFusion pipeline, obtaining the `func::FuncOp` to be processed and its fusion information.

2. **Policy selection and scheduler construction**

   - In `AutoScheduleBase.cpp::applySchedule()`, the corresponding scheduler is selected based on the fusion type `FusionKind`:
       - `FusionKind::PureElemwise` → `PureElemwiseScheduler`
       - `FusionKind::AnyPB` / `FusionKind::LastAxisPBR` / `FusionKind::AnyPBR` → `AnyPBRScheduler`

   - The actual scheduler instance is constructed through `std::make_unique<...>(funcOp)`.

3. **Main scheduling flow (`SchedulerBase::runOnOperation()`)**

   - **Pre-processing stage (`runPreScheduleProcedure()`)**:
       - IO cache insertion, fusion graph structure and validity analysis.
       - Calls the virtual function `analyzeAndVerifyKernelImpl()` to perform kernel analysis and constraint checking for the specific policy.
   - **Schedule generation stage (`runScheduleProcedure()`)**:
       - Calls the virtual function `calculateTilingImpl()` to generate `TilingComputeFn` and compute candidate Tiling schemes.
       - Selects an appropriate `TilingKey` (for example, based on factors such as cost and alignment).
       - Calls the virtual function `createScheduleImpl()` to generate the schedule description for that `TilingKey`.
       - Passes the schedule description to the Transform interpreter for execution through `applyScheduleImpl()`.

   - **Post-processing stage (`runPostScheduleProcedure()`)**:

       Performs structural optimization post-processing, statistics collection, and so on as required by the policy.

4. **Transform Dialect Application**

   `AutoScheduleInterpreter` parses the scheduling description, translates it into a Transform Dialect operation sequence, and transforms the original HFusion IR.

#### Key Data Structures

**1. KernelInfo (Kernel Information Description)**

- Used to abstract the structure and constraints of a single fused kernel. Typical information includes:
    - Input/output tensors and their shape/layout.
    - The topological relationships of each operator in the fusion graph.
    - Hardware-friendly constraints such as `stride` alignment, `size` alignment, and `tile` alignment.
    - Whether multi-core `reduce` is supported, along with the parallelizable dimension information.

- For specific fusion patterns, derived classes (such as `AnyPBRKernelInfo`) can be extended to add pattern-specific analysis results.

**2. Tiling Description (TilingUtils.h)**

- **TilingData**: Represents the Tiling parameters of a single dimension, which can be a constant or an expression.

- **TilingStruct** / **TilingCases**: Describe a complete set of Tiling schemes and a collection of multiple candidate schemes.

- **Expr** / **StmtExprBuilder**:
    - DimSymbol: A symbolic abstraction of dynamic dimensions.
    - Expr: Supports operations such as addition, subtraction, multiplication, and division, and expresses logic such as "dimension/factor" and "alignment to a certain granularity".
    - StmtExprBuilder: Responsible for constructing Expr from shape information, constants, and other elements in the IR, and generating host-side executable Tiling functions.

**3. ValueHandle Series**

- Provides unified encapsulation of `Value`, function parameters, named values, and other elements in MLIR, offering a unified interface for access and processing.

- Common types include `NamedValueHandle`, `FuncArgHandle`, and others.

### Scheduling Policy Implementation Overview

#### PureElemwise Scheduling Policy

- **Applicable scenario**: The operator graph mainly consists of element-wise operators, without complex broadcast/reduce structures.
- **Implementation location**: `PureElemwiseSchedule.h/cpp`.
- **Policy characteristics**:
    - Aims at a regular loop structure and adopts simple, regular tiling;
    - Focuses more on the memory access continuity and multi-level cache friendliness after fusion;
    - Completes tiling computation and scheduling primitive concatenation through `calculateTilingImpl()` and `createScheduleImpl()`.

#### AnyPBR Scheduling Policy

- **Applicable scenarios**: Fused subgraphs that include complex patterns such as `broadcast` and `reduce`.

- **Implementation location**: `AnyPBRSchedule.h/cpp`.

- **Core capabilities**:

    - **Complex Tiling computation**:

        - In `calculateTilingImpl()`, the following are considered comprehensively:
            - The stride alignment requirements of the Kernel
            - Dynamic shape symbols
            - Special dimensions such as reduce axes and broadcast axes
        - Builds expressions through `StmtExprBuilder` to generate multiple Tiling schemes (TilingCases).

    - **Scheduling construction**:

        In `createScheduleImpl()`, the specific scheduling policy is enabled based on the selected `TilingKey`, with the following factors considered:
        - On-chip space allocation size;
        - Enabling different Tiling policies for different types of axes;
        - loop fuse / coalesce;
        - Binding multiple cores, etc.

### Three Key Optimization Capabilities

This section outlines the design and roles of stride-align alignment optimization, dynamic shape support, and multi-core reduce parallelism in the AutoSchedule process.

**1. Stride-Align Memory Alignment Optimization**

**Optimization Objective**: Prevent unaligned memory UB access.

**Interfaces and Data Sources**:

- `KernelInfo::getStrideAlignments()` (`KernelInfo.h`): Returns a set of (dimension index, alignment granularity), describing the minimum stride to which certain dimensions must be aligned during memory access.
- `getSizeAlignments()`: Size dimension alignment constraints.
- `getTileAlignments()`: Tile dimension alignment constraints.

**Usage in the Scheduling Policy (Taking AnyPBR as an Example)**:

The implementation is located in `AnyPBRSchedule.cpp::calculateTilingImpl()`, and the specific process is as follows:

1. Generate an initial Tiling scheme based on the problem size;
2. Traverse the information in `KernelInfo::getStrideAlignments()` and `getTileAlignments()`, and align the Tiling size of the related dimensions upward through `alignTo(alignment)`;
3. Output the TilingCases that already satisfy the stride-align constraints.

**Location and timing**:

The stride-align processing occurs during the Tiling calculation phase, that is, when `calculateTilingImpl()` is called in `SchedulerBase::runScheduleProcedure()`.

**II. Dynamic Shape Support**

**Problem and requirements**:

- In actual business scenarios, some dimensions (such as batch size, height, and width) are not constants at compile time;
- AutoSchedule needs to support calculating appropriate Tiling parameters at runtime based on the actual input shape.

**Expression system design**:

The `Expr` / `DimSymbol` / `StmtExprBuilder` in `TilingUtils.h` constitute a lightweight expression framework:

- DimSymbol:
    - Represents the symbol of a dimension, for example, `N`, `H`, `W`, and so on;
    - Can be created through interfaces such as `StmtExprBuilder::createDimSymbolExpr()`.
- Expr: Supports basic arithmetic operations and can represent expressions such as `N / 4`, `min(N, 64)`, and `(H * W) / factor`.
- StmtExprBuilder:
    - Is responsible for building Expr from shape information, constants, and so on in the IR;
    - Generates specific Tiling computation statements on the Host side.

**Host Tiling function generation and execution**:

- The `TilingComputeFn` returned by `calculateTilingImpl()` is usually a lambda or a callable object;
- When this function is executed on the host, the actual input shape is known, so `DimSymbol` can be mapped to concrete values to evaluate the final Tiling.
- For a fully static shape, the expression can be folded directly into a constant at compile time.

**Configuration and Extension**:

AutoSchedule enables symbolic equivalence analysis through options such as `AutoScheduleOptions::enableSymbolAnalysis`, which is used for Tiling optimization under dynamic shapes.

**3. Multi-Core Reduce**

Multi-core reduce is analyzed through `analyzeMultiCoreReduceInfo()` and is enabled when the kernel and pattern meet the required conditions (see the related documentation).

### Custom Scheduling Policy Development Guide

This section describes how to add a new fusion mode and its scheduling policy in the HFusion AutoSchedule framework, including the policy class implementation, kernel information extension, and registration process.

1. **Define a new FusionKind**:

   - Add a new fusion type enumeration in the HFusion enumeration definition (such as `HFusionEnums.td`), for example: `FusionKind::MyKind`.

   - During the fusion analysis and pattern matching phase, ensure that fusion units corresponding to `FusionKind::MyKind` can be recognized and produced, so that AutoSchedule can correctly select the scheduler later.

2. **Inherit SchedulerBase to implement a custom scheduler**:

   Add a new header file (such as `MySchedule.h`) under `bishengir/include/bishengir/Dialect/HFusion/Transforms/AutoSchedule/`, and define the scheduler class:

   ```cpp
   class MyScheduler : public SchedulerBase {
   public:
     explicit MyScheduler(func::FuncOp funcOpIn)
         : SchedulerBase(funcOpIn, FusionKind::MyKind) {}
   
     // 1. Analyze and verify the kernel.
     LogicalResult analyzeAndVerifyKernelImpl() override;
   
     // 2. Compute Tiling (static/dynamic shape).
     TilingComputeFn calculateTilingImpl() override;
   
     // 3. Create the schedule (based on Transform Dialect primitives).
     LogicalResult createScheduleImpl(TilingKey key,
                                      OpBuilder &opBuilder) override;
   
     // 4. Optionally extend pre-/post-processing.
     LogicalResult runPreScheduleProcedure(OpBuilder &opBuilder) override;
     LogicalResult runPostScheduleProcedure(OpBuilder &opBuilder) override;
   };
   ```

   Add an implementation file (for example, `MySchedule.cpp`) under `bishengir/lib/Dialect/HFusion/Transforms/AutoSchedule/` to implement each virtual function:

   - **analyzeAndVerifyKernelImpl()**

     Collect kernel information with `KernelInfoCollector`. You can reuse the existing `KernelInfo` or add a custom subclass. Also check whether the operator types and shape relationships in the fusion graph conform to the assumptions of this policy.

   - **calculateTilingImpl()**

     Construct and return a `TilingComputeFn` callable object: use `StmtExprBuilder` to construct static/dynamic dimension expressions; introduce constraints such as stride-align and tile-align; and generate multiple sets of TilingCases for selection across different scenarios such as small-scale, large-scale, high-dimensional, and low-dimensional cases.

   - **createScheduleImpl(TilingKey key, OpBuilder &opBuilder)**

     Based on the selected `TilingKey`, invoke the scheduling primitives in sequence:

     - IO cache: `cacheRead` / `cacheWrite`;
     - Tiling: `tileUsingFor` / `tileUsingForAll` / `tileReductionUsingFor`;
     - Fusion and loop optimization: `fuseLoops`, `fuseIntoContaining`, `coalesceLoops`;

     Ensure that the generated Transform sequence is semantically correct and consistent with the analysis results in `KernelInfo`.

   - **runPreScheduleProcedure()** / **runPostScheduleProcedure()** (optional)

     Add policy-specific pre- and post-processing logic on top of the common flow, for example, special pattern normalization, scheduling result verification, and statistical output.

3. **Extend KernelInfo (optional)**:

   If the new policy requires additional structured information, it can be extended by inheriting from `KernelInfo`:

   ```cpp
   class MyKernelInfo : public KernelInfo {
   public:
     MyKernelInfo(MLIRContext *ctx)
         : KernelInfo(FusionKind::MyKind, ctx) {}
   
     static bool classof(const KernelInfo *T) {
       return T->getFusionKind() == FusionKind::MyKind;
     }
   
     // Add the additional fields and query interfaces required by the specific fusion mode here.
   };
   ```

   In addition, add the processing logic for `FusionKind::MyKind` in the `KernelInfoCollector` implementation to construct and populate a `MyKernelInfo` instance, so that the scheduler can use it in `analyzeAndVerifyKernelImpl()` and `calculateTilingImpl()`.

4. **Register the new policy with the AutoSchedule framework**

   In `AutoScheduleBase.cpp::applySchedule()`, add a new branch to the `switch (fusionKind)` statement:

   ```cpp
   case FusionKind::MyKind:
     scheduler = std::make_unique<MyScheduler>(funcOp);
     break;
   ```

After ensuring that the corresponding `MySchedule.cpp` is included in the build system and linked into the HFusion Transform module, the new policy can be used in the pipeline.

**Scheduling primitive (Schedule API) usage overview**:

Within `createScheduleImpl()`, the scheduling APIs provided by the framework (located in `ScheduleOperations.cpp`) can be directly reused:

- IO cache and buffer management

  `cacheRead`, `cacheWrite`, `setBufferSize`

- Tiling and loop structure control

  `tileUsingFor`, `tileUsingForAll`, `tileReductionUsingFor`

- Loop fusion and merging

  `fuseLoops`, `fuseIntoContaining`, `coalesceLoops`

- Multi-core binding

  `bindLoopToMulticore`, where the core count configuration can be determined by referring to functions such as `getMultiCoreNum` in the AnyPBR policy.

By combining the scheduling primitives described above, developers can implement flexible and efficient schedule policies for new fusion modes under a unified architecture.

### Brief Description of Internal Mechanisms

**ValueHandle Abstraction System**:

The framework uses the `ValueHandle` family of types to uniformly abstract various source objects in MLIR, including Value, Argument, and named values. This system provides unified access and operation interfaces, so that scheduler code does not need to be directly coupled to the underlying IR details. At the same time, it keeps the code concise and easy to maintain in both the scheduling description construction phase and the Transform interpretation and execution phase.

**Transform Dialect Integration and Interpreted Execution**:

AutoSchedule does not directly rewrite the operator IR inside the scheduler. Instead, it first constructs a Transform Dialect program, which is processed by `AutoScheduleInterpreter.cpp` as follows:

1. Receive the scheduling descriptions generated by each scheduler.
2. Translate the scheduling descriptions into a concrete Transform Dialect operation sequence.
3. Apply these operations to the target `func::FuncOp` to complete the actual IR transformation.

This design decouples the scheduling logic from the IR transformation details. The scheduling flow supports printing the Transform program for easy tracing and debugging, and the scheduling logic is reusable.

**Tiling Computation Framework**:

The framework combines **TilingInfo** with the **Expr** expression system to perform unified expression abstraction over dimension sizes, alignment rules, and dynamic shapes.

- Static shape scenario: Expressions can be evaluated at the compilation stage and folded into constant Tiling parameters.
- Dynamic shape scenario: In the Tiling function generated on the Host side, evaluation is completed in real time based on the actual input shape.

  The same set of expressions serves both static and dynamic scenarios, reducing code branching.
