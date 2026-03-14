# AutoSchedule

## HFusion AutoSchedule: Design for Automatic Fusion and Scheduling

HFusion is a high-level framework on Bisheng IR for operator fusion and automatic scheduling. The **AutoSchedule** module generates efficient schedules for Ascend NPU once fusion units are fixed. Design goals include:

- **Automation**: Choose scheduling and Tiling strategies from fusion patterns and operator traits.
- **Extensibility**: Common scheduler base and kernel abstractions for new strategies.
- **Performance**: Dynamic shape, multi-core reduce, and related optimizations.
- **Engineering**: Schedules expressed as reusable, interpretable Transform Dialect sequences.

---

### Hardware Background

The Ascend NPU uses a multi-level memory hierarchy: global memory (GM) has large capacity but high access latency, while on-chip memory (e.g., L1, UB) has lower latency but limited capacity. To leverage on-chip memory bandwidth and reduce global memory traffic:

- **Maximize on-chip memory utilization**: Through large-scale operator fusion, multiple ops are merged into the same kernel, so intermediate results are reused in on-chip buffers and GM read/write counts are reduced.
- **Satisfy hardware access constraints**: On-chip memory (e.g., UB) access must meet hardware requirements; for example, **UB access requires 32-byte alignment** (stride-align). Otherwise, misaligned access can cause runtime errors or performance degradation.
- **Fit Tiling and loop structure**: When generating Tiling and loop nests, stride/size/tile alignment constraints must be respected so that the resulting kernel complies with hardware specifications.

AutoSchedule is designed around these hardware characteristics: it uses automated scheduling and Tiling strategies to maximize on-chip memory utilization while preserving legality.

---

### Algorithm Principle

The core algorithm centers on **large-scale operator fusion + dimension-mapping-driven loop generation + Transform Dialect fusion execution**:

1. **Dimension Analyzer axis mapping**
   - `DimensionAnalyzer` analyzes the axis mapping of each op in the kernel relative to the anchor (`getCommonAxis`, `getNormalizedInterchange`, etc.).
   - It establishes the correspondence between tensor dimensions and anchor dimensions, supporting broadcast, reduce, transpose, and other patterns.
   - This provides dimension-level information for subsequent Tiling and loop construction.

2. **Loop generation and op fusion**
   - Based on the axis mapping, a unified loop structure is generated for the fusion graph, and ops are fused into the same loop via MLIR Transform Dialect primitives such as `fuseIntoContaining`, `fuseLoops`, and `coalesceLoops`.
   - During fusion, hardware constraints are explicitly enforced: stride-align (32-byte alignment), size-align, tile-align, etc., so the generated IR meets Ascend NPU memory access rules.

3. **Tiling computation and selection**
   - Using `StmtExprBuilder` and the `Expr` system, together with static or dynamic shape, multiple candidate Tiling schemes (`TilingCases`) are generated.
   - `getStrideAlignments()` and `getTileAlignments()` are applied during Tiling; dimensions are adjusted with `alignTo(alignment)` to produce valid Tiling that satisfies stride-align and related constraints.
   - A best `TilingKey` is chosen (e.g., by cost or alignment), and the schedule description is built accordingly.

4. **Transform Dialect interpretation**
   - The scheduler does not modify IR directly; it builds a Transform Dialect program. `AutoScheduleInterpreter` translates it into concrete Transform operations and applies them to the target IR so the schedule takes effect.

---

### Interface Description

#### Code location

- **Headers (APIs and abstractions)**: `bishengir/include/bishengir/Dialect/HFusion/Transforms/AutoSchedule/`
- **Implementation**: `bishengir/lib/Dialect/HFusion/Transforms/AutoSchedule/`

#### Core interfaces and abstractions

| Type       | Name                   | Description                                                                 |
| ---------- | ---------------------- | --------------------------------------------------------------------------- |
| Base class | `SchedulerBase`        | Abstract base for all schedulers, encapsulating the common scheduling flow  |
| Scheduler  | `PureElemwiseScheduler`| Pure elementwise fusion strategy                                            |
| Scheduler  | `AnyPBRScheduler`      | Generic strategy for Pointwise/Broadcast/Reduce and similar fusion patterns  |
| Kernel     | `KernelInfo`           | Unified description of fused kernel IO, dimensions, alignment, multi-core   |
| Alignment  | `getStrideAlignments()`| Returns stride alignment constraints (dim index, unit), e.g. 32-byte align   |
| Alignment  | `getSizeAlignments()`  | Returns size dimension alignment constraints                                 |
| Alignment  | `getTileAlignments()`  | Returns tile dimension alignment constraints                                 |
| Analysis   | `DimensionAnalyzer`    | `getCommonAxis`, `getInterchange`, `getNormalizedInterchange`, etc.          |
| Primitive  | `cacheRead` / `cacheWrite` | IO cache                                                                |
| Primitive  | `tileUsingFor` / `tileUsingForAll` / `tileReductionUsingFor` | Tiling      |
| Primitive  | `fuseLoops` / `fuseIntoContaining` / `coalesceLoops` | Loop fusion and coalesce         |
| Primitive  | `setBufferSize`        | Resource constraints                                                        |

#### Strategy selection and call chain

- **Pass entry**: The AutoSchedule pass is invoked in the HFusion pipeline and receives the `func::FuncOp` and fusion information.
- **Strategy selection**: In `AutoScheduleBase.cpp::applySchedule()`, the scheduler is chosen by `FusionKind`:
  - `FusionKind::PureElemwise` → `PureElemwiseScheduler`
  - `FusionKind::AnyPB` / `FusionKind::LastAxisPBR` / `FusionKind::AnyPBR` → `AnyPBRScheduler`
- **Main flow**: `runPreScheduleProcedure()` → `runScheduleProcedure()` (including `calculateTilingImpl()`, `createScheduleImpl()`) → `runPostScheduleProcedure()` → Transform Dialect application.

---

### Constraints and Capabilities

AutoSchedule explicitly handles the following constraints during scheduling and Tiling:

| Constraint       | Description                                                                  | API / Implementation |
| ---------------- | ---------------------------------------------------------------------------- | -------------------- |
| **Stride align** | UB and other on-chip memory access must be 32-byte aligned                   | `getStrideAlignments()`, `alignTo()` in `calculateTilingImpl()` |
| **Size align**   | Some ops (e.g., transpose, concat, cast) require tile/size alignment         | `getSizeAlignments()` |
| **Tile align**   | Combination of stride and size constraints, applied to Tiling schemes        | `getTileAlignments()` |
| **Reduce axis**  | Reduce, broadcast, extract_slice, transpose, etc. impose lowest-dim alignment| `KernelInfo` per-op stride-align logic |
| **On-chip buffer** | Buffer allocation limited by L1/UB capacity and `maxBufferCnt`           | `setBufferSize`, `maxBufferCnt` |
| **Multi-core reduce** | Multi-core parallel reduce only when specific conditions hold         | `analyzeMultiCoreReduceInfo()` |

These constraints are applied uniformly in `KernelInfo::getStrideAlignments()` and scheduler-specific `calculateTilingImpl()` to ensure the generated schedule is valid and hardware-compatible.

---

### Architecture overview

#### Core components

##### Scheduler base and strategies

- **SchedulerBase**: Abstract base for all schedulers (`AutoScheduleBase.h`), encapsulating the common scheduling flow.
- **Concrete strategy schedulers**:
  - **PureElemwiseScheduler**: Pure elementwise fusion (`PureElemwiseSchedule.h/cpp`).
  - **AnyPBRScheduler**: Generic strategy for AnyPBR (Pointwise/Broadcast/Reduce) and similar ops (`AnyPBRSchedule.h/cpp`).

##### Kernel and tiling abstraction

- **KernelInfo**: Unified description of a fused kernel (`KernelInfo.h`), including IO, dimensions, alignment, and multi-core capability.
- **Tiling abstraction and utilities** (`TilingUtils.h/cpp`):
  - **TilingInfo**, **TilingStruct**, **TilingData**: Describe a single or multiple candidate tiling schemes.
  - **Expr** / **StmtExprBuilder**: Build tiling expressions that depend on static or dynamic shape.

##### Schedule operations

- **ScheduleOperations.cpp**: Implements reusable schedule primitives, including:
    - IO cache: `cacheRead` / `cacheWrite`
    - Tiling: `tileUsingFor` / `tileUsingForAll` / `tileReductionUsingFor`
    - Loop transforms: `fuseLoops` / `fuseIntoContaining` / `coalesceLoops`
    - Resource constraints: `setBufferSize`, etc.

##### Schedule interpretation

- **AutoScheduleInterpreter.cpp**: Converts the high-level schedule description produced by the scheduler into Transform Dialect operations and applies them to the target IR so the schedule takes effect.

#### Strategy selection and call chain

The overall call chain is:

- **Pass entry**
  - The AutoSchedule pass is invoked in the HFusion pipeline and receives the `func::FuncOp` and fusion information to process.

- **Strategy selection and scheduler construction**
  - In `AutoScheduleBase.cpp::applySchedule()`, the scheduler is chosen by fusion kind `FusionKind`:
    - `FusionKind::PureElemwise` → `PureElemwiseScheduler`
    - `FusionKind::AnyPB` / `FusionKind::LastAxisPBR` / `FusionKind::AnyPBR` → `AnyPBRScheduler`
  - The scheduler instance is created with `std::make_unique<...>(funcOp)`.

- **Main scheduling flow (`SchedulerBase::runOnOperation()`)**
  - **Pre** (`runPreScheduleProcedure()`): Insert IO cache, analyze fusion graph and legality; call `analyzeAndVerifyKernelImpl()` for strategy-specific kernel analysis and checks.
  - **Schedule** (`runScheduleProcedure()`): Call `calculateTilingImpl()` to get `TilingComputeFn` and candidate tiling; choose a `TilingKey` (e.g. by cost or alignment); call `createScheduleImpl()` to build the schedule for that key; pass the schedule to the Transform interpreter via `applyScheduleImpl()`.
  - **Post** (`runPostScheduleProcedure()`): Optional structure cleanup and statistics.

- **Transform Dialect application**
  - `AutoScheduleInterpreter` parses the schedule description, translates it into a sequence of Transform Dialect operations, and applies them to the HFusion IR.

#### Key data structures

##### KernelInfo (kernel description)
  - Abstracts the structure and constraints of a single fused kernel. Typical information includes:
    - Input/output tensors and their shape/layout.
    - Topology of ops in the fusion graph.
    - Stride, size, and tile alignment constraints for hardware.
    - Whether multi-core reduce is supported and which dimensions can be parallelized.
  - For specific fusion patterns, derived classes (e.g. `AnyPBRKernelInfo`) can add pattern-specific analysis.

##### Tiling (`TilingUtils.h`)

- **TilingData**: Tiling parameters for a single dimension (constant or expression).
- **TilingStruct** / **TilingCases**: A full tiling scheme and sets of candidate schemes.
- **Expr** / **StmtExprBuilder**:
    - `DimSymbol`: Symbol for a dynamic dimension.
    - `Expr`: Arithmetic (e.g. dimension/factor, align-to granularity).
    - `StmtExprBuilder`: Builds `Expr` from IR shape and constants and generates the host-side tiling function.

##### ValueHandle
  - Uniform wrapper for MLIR `Value`, function arguments, and named values for consistent access and manipulation.
  - Common types include `NamedValueHandle`, `FuncArgHandle`, etc.

---

### Scheduling strategies

#### PureElemwise scheduling strategy

- **Use case**: Graphs that are mostly elementwise ops, without complex broadcast/reduce.
- **Implementation location**: `PureElemwiseSchedule.h/cpp`.
- **Strategy characteristics**:
  - Aims at regular loop structure with simple, regular tiling.
  - Emphasizes contiguous access and multi-level cache friendliness after fusion.
  - `calculateTilingImpl()` and `createScheduleImpl()` perform tiling and schedule primitive chaining.

#### AnyPBR scheduling strategy (AnyPBRScheduler)

- **Use case**: Fused subgraphs containing broadcast, reduce, and similar patterns.
- **Implementation location**: `AnyPBRSchedule.h/cpp`.
- **Capabilities**:
  - **Tiling**:
    - In `calculateTilingImpl()`, considers stride alignment, dynamic shape symbols, reduce/broadcast axes, etc.; uses `StmtExprBuilder` to build expressions and produce multiple `TilingCases`.
  - **Multi-core reduce analysis and utilization**:
    - `analyzeMultiCoreReduceInfo()` determines whether multi-core reduce conditions are met (see §7.3).
  - **Schedule construction**:
    - In `createScheduleImpl()`, for the chosen `TilingKey`, enables concrete scheduling strategies, considering on-chip buffer allocation, axis-specific tiling, loop fuse/coalesce, and multi-core binding.

---

### Main optimizations

This section summarizes **stride-align**, **dynamic shape**, and **multi-core reduce** and their role in AutoSchedule.

#### Stride-align memory alignment optimization

- **Goal**: Avoid unaligned UB access.
- **APIs and data source**:
  - `KernelInfo::getStrideAlignments()` (`KernelInfo.h`): Returns (dimension index, alignment) pairs describing the minimum stride alignment for certain dimensions during memory access.
  - Related APIs: `getSizeAlignments()` for size alignment; `getTileAlignments()` for tile alignment.
- **Usage in scheduling strategy (AnyPBR example)**:
  - In `AnyPBRSchedule.cpp::calculateTilingImpl()`:
    - Generate initial tiling from problem size.
    - Iterate over `getStrideAlignments()` and `getTileAlignments()`, and adjust relevant dimension tiling sizes with `alignTo(alignment)`.
    - Output stride-aligned `TilingCases`.
- **When**: Stride-align is applied during **tiling**, i.e. when `calculateTilingImpl()` is called inside `runScheduleProcedure()`.

#### Dynamic shape support

- **Problem and requirements**:
  - Some dimensions (e.g., batch size, spatial size) may not be constant at compile time.
  - AutoSchedule must support computing suitable tiling parameters at **runtime** from actual input shapes.
- **Expr system design**:
  - **Expr** / **DimSymbol** / **StmtExprBuilder** in `TilingUtils.h` form a lightweight expression framework:
    - **DimSymbol**: Symbol for a dimension (e.g., N, H, W); can be created via `StmtExprBuilder::createDimSymbolExpr()`.
    - **Expr**: Supports basic arithmetic, e.g., `N/4`, `min(N,64)`, `(H*W)/factor`.
    - **StmtExprBuilder**: Builds `Expr` from IR shape and constants; generates host-side tiling computation statements.
- **Host tiling function generation and execution**:
  - `TilingComputeFn` from `calculateTilingImpl()` is typically a lambda or callable.
  - When executed on the host with concrete shapes, `DimSymbol`s are bound to values and tiling is computed.
  - For fully static shapes, expressions fold to constants at compile time.
- **Configuration and extension**:
  - Options such as `AutoScheduleOptions::enableSymbolAnalysis` enable symbolic equivalence analysis for dynamic shape tiling optimization.

#### Multi-core reduce

- Multi-core reduce is analyzed (e.g. via `analyzeMultiCoreReduceInfo()`) and applied when the kernel and pattern satisfy the required conditions (see dedicated documentation).

---

### Extending AutoSchedule: custom strategy

This section describes how to **add a new fusion pattern and its scheduling strategy** in the HFusion AutoSchedule framework, including scheduler implementation, kernel info extension, and registration.

#### Define a new FusionKind

- In the HFusion enum definition (e.g. `HFusionEnums.td`), add a new fusion kind, e.g. `FusionKind::MyKind`.
- In fusion analysis and pattern matching, ensure that fusion units with this kind are produced so that AutoSchedule can select the corresponding scheduler.

#### Custom scheduler (inherit SchedulerBase)

- Add a header (e.g. `MySchedule.h`) under `AutoSchedule/` and define the scheduler class:

```cpp
class MyScheduler : public SchedulerBase {
public:
  explicit MyScheduler(func::FuncOp funcOpIn)
      : SchedulerBase(funcOpIn, FusionKind::MyKind) {}

  // 1. Kernel analysis and verification
  LogicalResult analyzeAndVerifyKernelImpl() override;

  // 2. Tiling computation (static / dynamic shape)
  TilingComputeFn calculateTilingImpl() override;

  // 3. Schedule creation (Transform Dialect primitives)
  LogicalResult createScheduleImpl(TilingKey key,
                                 OpBuilder &opBuilder) override;

  // 4. Optional: pre/post extensions
  LogicalResult runPreScheduleProcedure(OpBuilder &opBuilder) override;
  LogicalResult runPostScheduleProcedure(OpBuilder &opBuilder) override;
};
```

- Add implementation (e.g. `MySchedule.cpp`) and implement the overrides:

- **analyzeAndVerifyKernelImpl()**
  - Use `KernelInfoCollector` to gather kernel info (reuse an existing `KernelInfo` or add a custom subclass).
  - Check that the fusion graph matches the strategy (op types, shape relations, etc.).

- **calculateTilingImpl()**
  - Build and return `TilingComputeFn`: use `StmtExprBuilder` for static/dynamic dimension expressions; apply stride-align and tile-align; generate multiple `TilingCases` for different scenarios (e.g. small/large, different ranks) for selection.

- **createScheduleImpl(TilingKey key, OpBuilder &opBuilder)**
  - For the chosen `TilingKey`, call schedule primitives in order: IO cache (`cacheRead`/`cacheWrite`), tiling (`tileUsingFor`/`tileUsingForAll`/`tileReductionUsingFor`), loop transforms (`fuseLoops`, `fuseIntoContaining`, `coalesceLoops`). Ensure the generated Transform sequence is correct and consistent with `KernelInfo`.

- **runPreScheduleProcedure()** / **runPostScheduleProcedure()** (optional)
  - Add strategy-specific pre/post logic, e.g. pattern normalization, schedule validation, or statistics.

#### Extend KernelInfo (optional)

If the new strategy needs extra structured information, extend `KernelInfo` by subclassing:

```cpp
class MyKernelInfo : public KernelInfo {
public:
  MyKernelInfo(MLIRContext *ctx)
      : KernelInfo(FusionKind::MyKind, ctx) {}

  static bool classof(const KernelInfo *T) {
    return T->getFusionKind() == FusionKind::MyKind;
  }

  // Add fields and accessors required by this fusion pattern
};
```

In `KernelInfoCollector`, add handling for `FusionKind::MyKind` to construct and fill `MyKernelInfo` so the scheduler can use it in `analyzeAndVerifyKernelImpl()` and `calculateTilingImpl()`.

#### Register the strategy

In `AutoScheduleBase.cpp::applySchedule()`, add a branch:

```cpp
case FusionKind::MyKind:
  scheduler = std::make_unique<MyScheduler>(funcOp);
  break;
```

Ensure `MySchedule.cpp` is in the build and linked into the HFusion Transform module; the new strategy is then available in the pipeline.

#### Schedule primitives (Schedule API) quick reference

Inside `createScheduleImpl()`, you can reuse the schedule APIs in `ScheduleOperations.cpp`:

- **IO cache and buffer management**
  - `cacheRead`
  - `cacheWrite`
  - `setBufferSize`
- **Tiling and loop structure control**
  - `tileUsingFor`
  - `tileUsingForAll`
  - `tileReductionUsingFor`
- **Loop fuse and coalesce**
  - `fuseLoops`
  - `fuseIntoContaining`
  - `coalesceLoops`
- **Multi-core binding**
  - `bindLoopToMulticore`
  - See AnyPBR for `getMultiCoreNum` and similar to determine core count configuration.

By combining these primitives, you can implement flexible and efficient schedule strategies for new fusion patterns within the same framework.

---

### Internal mechanisms (brief)

#### ValueHandle abstraction

- The `ValueHandle` family provides a uniform abstraction over MLIR values, function arguments, and named values from different sources.
- They offer a single interface for access and manipulation so scheduler code does not depend on low-level IR details.
- This helps keep the code concise and maintainable in both schedule construction and Transform interpretation.

#### Transform Dialect integration and interpretation

- AutoSchedule does not modify operator IR directly inside the scheduler; it builds a **Transform Dialect program**.
- `AutoScheduleInterpreter.cpp` is responsible for:
  - Receiving the schedule description produced by the scheduler;
  - Translating it into a sequence of Transform Dialect operations;
  - Applying these operations to the target `func::FuncOp` to perform the actual IR transformation.
- This design keeps schedule logic decoupled from IR details and makes the schedule traceable (e.g., by printing the Transform program), debuggable, and reusable.

#### Tiling computation framework

- The `TilingInfo` and `Expr` system unify **dimension size**, **alignment rules**, and **dynamic shape** as expressions.
- For static shape:
  - Expressions can be evaluated at compile time and folded into constant tiling parameters.
- For dynamic shape:
  - They are evaluated in the host tiling function with concrete input shapes;
  - The same expressions serve both static and dynamic cases, reducing code duplication.
