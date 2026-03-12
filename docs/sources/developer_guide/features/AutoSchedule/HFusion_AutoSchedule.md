# AutoSchedule

## HFusion AutoSchedule: design for automatic fusion and scheduling

### 1. Background and goals

HFusion is a high-level framework on Bisheng IR for operator fusion and automatic scheduling. The **AutoSchedule** module generates efficient schedules for Ascend NPU once fusion units are fixed. Its design goals include:

- **Automation**: Choose scheduling and tiling strategies from fusion patterns and operator traits.
- **Extensibility**: Common scheduler base and kernel abstractions for new strategies.
- **Performance**: Dynamic shape, multi-core reduce, and other optimizations.
- **Engineering**: Schedules expressed as reusable, interpretable Transform Dialect sequences.

AutoSchedule code lives in:

- **Headers (APIs and abstractions)**: `bishengir/include/bishengir/Dialect/HFusion/Transforms/AutoSchedule/`
- **Implementation**: `bishengir/lib/Dialect/HFusion/Transforms/AutoSchedule/`

---

### 2. Architecture overview

#### 2.1 Core components

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

#### 2.2 Strategy selection and call chain

The overall call chain is:

- **Pass entry**
  - The AutoSchedule pass is invoked in the HFusion pipeline and receives the `func::FuncOp` and fusion information to process.

- **Strategy selection and scheduler construction**
  - In `AutoScheduleBase.cpp::applySchedule()`, the scheduler is chosen by fusion kind `FusionKind`:
    - `FusionKind::PureElemwise` → `PureElemwiseScheduler`
    - `FusionKind::AnyPB` / `FusionKind::LastAxisPBR` / `FusionKind::AnyPBR` → `AnyPBRScheduler`
  - The scheduler instance is created with `std::make_unique<...>(funcOp)`.

- **Main scheduling flow (`SchedulerBase::runOnOperation()`)**:
  - **Pre** (`runPreScheduleProcedure()`): Insert IO cache, analyze fusion graph and legality; call `analyzeAndVerifyKernelImpl()` for strategy-specific kernel analysis and checks.
  - **Schedule** (`runScheduleProcedure()`): Call `calculateTilingImpl()` to get `TilingComputeFn` and candidate tiling; choose a `TilingKey` (e.g. by cost or alignment); call `createScheduleImpl()` to build the schedule for that key; pass the schedule to the Transform interpreter via `applyScheduleImpl()`.
  - **Post** (`runPostScheduleProcedure()`): Optional structure cleanup and statistics.

- **Transform Dialect application**
  - `AutoScheduleInterpreter` parses the schedule description, translates it into a sequence of Transform Dialect operations, and applies them to the HFusion IR.

#### 2.3 Key data structures

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

---

### 3. Scheduling strategies

#### 3.1 PureElemwise

- **Use case**: Graphs that are mostly elementwise ops, without complex broadcast/reduce.
- **Location**: `PureElemwiseSchedule.h/cpp`.
- **Strategy**: Aims at regular loop structure with simple, regular tiling; emphasizes contiguous access and multi-level cache friendliness after fusion; `calculateTilingImpl()` and `createScheduleImpl()` perform tiling and schedule construction.

#### 3.2 AnyPBR (AnyPBRScheduler)

- **Use case**: Fused subgraphs containing broadcast, reduce, and similar patterns.
- **Location**: `AnyPBRSchedule.h/cpp`.
- **Capabilities**:
  - **Tiling**: In `calculateTilingImpl()`, considers stride alignment, dynamic shape symbols, reduce/broadcast axes, etc.; uses `StmtExprBuilder` to build expressions and produce multiple `TilingCases`.
  - **Multi-core reduce**: `analyzeMultiCoreReduceInfo()` determines whether multi-core reduce conditions are met (see §4.3).
  - **Schedule construction**: In `createScheduleImpl()`, for the chosen `TilingKey`, applies buffer sizes, axis-specific tiling, loop fuse/coalesce, and multi-core binding.

---

### 4. Main optimizations

This section summarizes **stride-align**, **dynamic shape**, and **multi-core reduce** and their role in AutoSchedule.

#### 4.1 Stride-align

- **Goal**: Avoid unaligned UB access.
- **APIs**: `KernelInfo::getStrideAlignments()` returns (dimension index, alignment) pairs; `getSizeAlignments()`, `getTileAlignments()` give size and tile alignment constraints.
- **Usage**: In `AnyPBRSchedule.cpp::calculateTilingImpl()`, initial tiling is generated from problem size; then dimensions from `getStrideAlignments()` and `getTileAlignments()` are adjusted with `alignTo(alignment)`; the result is stride-aligned `TilingCases`.
- **When**: Stride-align is applied during **tiling**, i.e. when `calculateTilingImpl()` is called inside `runScheduleProcedure()`.

#### 4.2 Dynamic shape

- **Need**: Dimensions such as batch size or spatial size may be unknown at compile time; tiling must be computed at runtime from actual input shapes.
- **Expr system** (`TilingUtils.h`): `DimSymbol` for a dimension (e.g. N, H, W); `Expr` for arithmetic like `N/4`, `min(N,64)`; `StmtExprBuilder` builds `Expr` from IR shapes and constants and generates the host tiling code.
- **Execution**: `TilingComputeFn` from `calculateTilingImpl()` is run on the host with concrete shapes so `DimSymbol`s are bound to values and tiling is computed; fully static shapes fold to constants at compile time.
- **Options**: e.g. `AutoScheduleOptions::enableSymbolAnalysis` to enable symbolic analysis for dynamic tiling.

#### 4.3 Multi-core reduce

- Multi-core reduce is analyzed (e.g. via `analyzeMultiCoreReduceInfo()`) and applied when the kernel and pattern satisfy the required conditions (see dedicated documentation).

---

### 5. Extending AutoSchedule: custom strategy

#### 5.1 Define a new FusionKind

- In the HFusion enum definition (e.g. `HFusionEnums.td`), add a new fusion kind, e.g. `FusionKind::MyKind`.
- In fusion analysis and pattern matching, ensure that fusion units with this kind are produced so that AutoSchedule can select the corresponding scheduler.

#### 5.2 Custom scheduler (inherit SchedulerBase)

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

#### 5.3 Extend KernelInfo (optional)

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

#### 5.4 Register the strategy

In `AutoScheduleBase.cpp::applySchedule()`, add a branch:

```cpp
case FusionKind::MyKind:
  scheduler = std::make_unique<MyScheduler>(funcOp);
  break;
```

Ensure `MySchedule.cpp` is in the build and linked into the HFusion Transform module; the new strategy is then available in the pipeline.

#### 5.5 Schedule primitives (Schedule API) quick reference

Inside `createScheduleImpl()` you can use the schedule APIs in `ScheduleOperations.cpp`:

- **IO cache and buffer**
  - `cacheRead`, `cacheWrite`, `setBufferSize`
- **Tiling and loop structure**
  - `tileUsingFor`, `tileUsingForAll`, `tileReductionUsingFor`
- **Loop fuse and coalesce**
  - `fuseLoops`, `fuseIntoContaining`, `coalesceLoops`
- **Multi-core**
  - `bindLoopToMulticore`; see AnyPBR for `getMultiCoreNum` and similar for core count configuration.

By combining these primitives, you can implement flexible and efficient schedule strategies for new fusion patterns within the same framework.

---

### 6. Internal mechanisms (brief)

#### 6.1 ValueHandle

- The `ValueHandle` family provides a uniform abstraction over MLIR values, function arguments, and named values.
- They offer a single interface for access and manipulation so scheduler code does not depend on low-level IR details and stays maintainable in both schedule construction and Transform interpretation.

#### 6.2 Transform Dialect integration and interpretation

- AutoSchedule does not modify operator IR directly inside the scheduler; it builds a **Transform Dialect program**.
- `AutoScheduleInterpreter.cpp` receives the schedule description from the scheduler, translates it into a sequence of Transform Dialect operations, and applies them to the target `func::FuncOp` to perform the actual IR transformation.
- This keeps schedule logic decoupled from IR details and makes the schedule traceable (e.g. by printing the Transform program), debuggable, and reusable.

#### 6.3 Tiling framework

- The `TilingInfo` and `Expr` system unify **dimension size**, **alignment rules**, and **dynamic shape** as expressions.
- For static shape, expressions can be evaluated at compile time and folded into constant tiling parameters.
- For dynamic shape, they are evaluated in the host tiling function with concrete input shapes; the same expressions support both static and dynamic cases and reduce code duplication.
