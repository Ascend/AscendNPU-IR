# Cube and Vector Optimization

## Hardware Background

This document provides a macro-level overview of the overall Cube-Vector (CV) optimization process in AscendNPU IR. CV optimization targets the NPU hardware of Atlas A2 series products and Atlas A3 series products, focusing on the collaborative work of two core units, Cube (matrix multiplication unit) and Vector (vector operation unit). It performs a series of transformations at the HIVM (Huawei Intermediate Representation Virtual Machine) layer to improve the execution efficiency of mixed kernels (Mix Kernel).

### Terminology and Background Knowledge (Read Before Reading)

The following terms appear repeatedly throughout the CV documentation. It is recommended to establish a basic understanding of them before reading the details of each pass.

| Terminology | Meaning | Supplementary Notes |
|------|------|----------|
| **HIVM** | Huawei Intermediate Virtual Machine | A dialect in AscendNPU IR that carries NPU-oriented operators (such as mmadL1, fixpipe, and vadd) and control flow. |
| **IR** | Intermediate Representation | The abstract representation between source code and machine code in a compiler. This repository uses MLIR (Multi-Level IR), and the IR is organized in SSA form. |
| **Bufferization** | The process of converting the tensor abstraction into concrete memory (memref) | In the "pre-bufferization" phase, the IR is still dominated by tensors (logical multi-dimensional arrays); after bufferization, memrefs (memory references with addresses/layouts) are introduced. Most CV passes run in the pre-bufferization phase, so expressions such as "tensor.empty" and "tensor slice" are common in this document. |
| **tensor vs memref** | tensor: a logical multi-dimensional array without an explicit address; memref: a memory region with a base address, strides, and shape | In the CV flow, the "output" of fixpipe is often first represented as a tensor and later materialized into concrete memory through a workspace (memref) or bufferization. |
| **Workspace** | A contiguous block of memory allocated on GM at runtime and passed in as a kernel parameter | Used to store intermediate results between Cube and Vector (such as fixpipe output). Multiple intermediate buffers can be allocated on the same workspace by offset, with offsets computed by PlanMemory, thereby reusing one large buffer and reducing total occupancy. |
| **Liveness** | The lifetime of a buffer from "definition/first use" to "last use" | PlanMemory uses liveness to determine whether two buffers may be alive at the same time; if they do not overlap, they can be allocated the same base address with different offsets to achieve reuse. |
| **Inplace** | The output of an operation is written directly to the storage location of the input, reusing the same buffer | For example, when vcast converts from f16 to i16 (equal width), the output can overwrite the input, reducing allocations. PlanMemory identifies ops that can be inplace and performs the corresponding offset allocation. |
| **AIC / AIV** | AIC: a sub-kernel dominated by Cube; AIV: a sub-kernel dominated by Vector | After the Mix kernel is split, AIC mainly executes on the Cube core and AIV mainly on the Vector core; the two transfer data through fixpipe, DMA, and so on, with the call order coordinated by the Host or the scheduler. |
| **Host / Device** | Host: runs on the CPU<br>Device: runs on the NPU | The Mix kernel belongs to the device side.<br>In this document, "Mix can only be called by Host" means: in a kernel call initiated from the Host, a mix entry can be called, whereas another kernel on the device cannot directly call a mix function (current convention). |
| **CC / CV / VC / VV** | The two letters respectively denote the "preceding compute unit" and the "following compute unit": C=Cube, V=Vector | For example, CV means Cube computation followed by Vector computation via fixpipe/load;<br>CC means two Cube segments connected via fixpipe+load. Used to describe the matching modes of InsertWorkSpaceForMixCV. |

**About "pre-bufferization" and "post-bufferization"**:

Most CV-related passes run in the pre-bufferization phase (`hivmPreBufferizationOptimizationPipeline`), where the IR is still dominated by tensors and contains control flow such as `scf.for`. Bufferization converts tensors into memrefs and determines the physical layout; after that, there are optimizations in the post-bufferization phase (such as another round of PlanMemory targeting `memref.alloc`). Understanding "perform CV structural transformation first, then materialize memory" helps understand the pass order.

### Chip Architecture

Ascend NPU adopts a heterogeneous computing architecture, which mainly includes:

| Component | Description | Typical Specifications (Example) |
|------|------|-------------------|
| **Cube** | Matrix multiplication unit that executes matrix operations such as `mmadL1` and `batchMmadL1`. | 24 AI Cores |
| **Vector** | Vector operation unit that executes vector operations such as `vadd`, `vcast`, and `vreduce`. | 48 AI Cores |
| **L0C** | Cube output buffer that stores matrix multiplication results. | 128 KB |
| **L0A/L0B** | Cube input buffer. | 64 KB |
| **UB** | Unified buffer, the main memory for Vector operations. | 256 KB |
| **GM** | Global memory. | External DDR |

fixpipe is the data transfer channel between Cube and Vector. The underlying Cube and Vector architectures of Ascend chips are separated. Different chip versions have different interaction paths. For example, for the 910 series, after Cube computation completes, the results are transferred from L0C to GM through fixpipe for subsequent Vector operations. In the IR, this is represented by the `hivm.hir.fixpipe` operator; at the hardware level, it corresponds to a dedicated L0C-to-UB data path that can simultaneously perform type conversion, quantization, and so on (controlled by attributes such as `pre_quant` and `pre_relu` of fixpipe). The chip architecture of the 910 series is as follows:
![V220 Architecture](../../../images/developer_guide/cvarch.png)

## Algorithm Principles

### createNormalizeMatmulPass

- **Function**: Normalizes the M/K/N dimensions, init conditions, and per-channel add forms of `hivm.hir.mmadL1` and `hivm.hir.batchMmadL1`.
- **Purpose**: Unifies the IR form of matmul to facilitate matching and transformation by subsequent passes such as fixpipe insertion and tiling.
- **Typical Transformation**: Inlines the bias in the vbrc + vadd form into the init of mmadL1; extracts the actual M/K/N and replaces constants; handles the PerChannel scenario.
- **Typical Scenario**: `elementwise` accumulation.

Before Transformation:

```mlir
%2 = ops // not 0 const
%3 = hivm.hir.mmadL1 ins(*)
       outs(%2 : tensor<16x32xf32>) -> tensor<16x32xf32>
```

After Transformation:

```mlir
%2 = ops
%3 = tensor.empty() : tensor<16x32xf32>
%4 = hivm.hir.mmadL1 ins(*)
        outs(%3 : tensor<16x32xf32>) -> tensor<16x32xf32>
%5 = hivm.hir.vadd ins(%2, %4: tensor<1x32xf32>) outs(%2 : tensor<16x32xf32>)
```

### createInlineFixpipePass

- **Function**: Inserts `hivm.hir.fixpipe` between **mmadL1**/**batchMmadL1** and **store**, fusing **store**+**vcast** and other operations into the quantization/activation options of **fixpipe**.
- **Purpose**: Explicitly expresses the data movement from Cube to Vector, providing clear insertion points for subsequent workspace allocation and load/store insertion.
- **Typical Transformation**: Inserts **fixpipe** on the use chain from the **mmadL1** result to **store**; fuses **vcast(f32->f16)** and similar operations into the `pre_quant = F322F16` option of **fixpipe**.
- **Typical Scenario**: Pure Cube-to-Store.

Before Transformation:

```mlir
mmadL1 -> store
```

After Transformation:

```mlir
mmadL1 -> fixpipe
```

InlineFixpipe is responsible for inserting **fixpipe**. Based on the newly inserted **fixpipe**, it attempts to inline operations such as `hivm.vcast`/`hivm.vrelu`/`hivm.store`.

### createTileBatchMMIntoLoopPass

- **Function**: Expands `hivm.hir.batchMmadL1` along the batch dimension into an `scf.for` loop, where each iteration executes a single `mmadL1` and fixpipe.
- **Purpose**: Splits the batch dimension into a loop so that load/fixpipe/store can be accessed by batch index, facilitating workspace management and pipelining.
- **Typical Transformation**: TileBatchMMIntoLoop expands batchMmadL1 into a loop.

Before Transformation:

```mlir
batchmmadL1 a : [batch, m, k], b[batch, k, n]
fixpipe workspace : [batch, m, n]
```

After Transformation:

```mlir
for batch_idx in range(batch):
  mmadL1(extract_slice(a), extract_slice(b))
  fixpipe(extract_slice(workspace))
```

### createInsertLoadStoreForMixCVPass

- **Function**: Inserts `load`/`store` at the Cube-Vector junction so that data flows correctly between tensors and the global workspace.
- **Purpose**: Ensures correct data transfer between CV operations.
- **Typical Transformation**: batchMmadL1 + fixpipe is rewritten as mmadL1 + fixpipe inside a loop, with extract_slice / insert_slice applied to the inputs/outputs.
- **Typical Scenario**: Cube-Vector mixed (CV mode).

Before Transformation:

```mlir
mmadL1
fixpipe
vadd
```

After Transformation:

```mlir
mmadL1
fixpipe
load
vadd
```

### createInsertWorkSpaceForMixCVPass

- **Function**: Replaces `tensor.empty` with `memref_ext.alloc_workspace` at Cube-Vector intersection points (CC/CV/VC/VV).
- **Purpose**: Changes intermediate buffers such as fixpipe outputs and store outputs to be allocated from the global workspace, enabling sharing across iterations and cores.
- **Matching Modes**: CC (`mmadL1`→`fixpipe`→`load`→`mmadL1`), CV (`mmadL1`→`fixpipe`→`load`→`vector`), VC (`vector`→`store`→`load`→`mmadL1`), VV (`vector`→`store`→`load`→`vector`).
- **Typical Scenario**: Cube-Vector mixed (CV mode).

Before Transformation:

```mlir
%1 = mmadL1
%2 = tensor.empty() 
%3 = fixpipe ins(%1) outs(%2)
%4 = load ins(%3)
vadd (%4)
```

After Transformation:

```mlir
%1 = mmadL1
%2 = memref_ext.alloc_workspace()
%3 =  bufferization.to_tensor(%2)
%4 = fixpipe ins(%1) outs(%3)
%5 = load ins(%4)
vadd (%5)
```

### createBindWorkSpaceArgPass

- **Function**: Binds `memref_ext.alloc_workspace` within the function to the function's workspace parameter (`hacc.arg_type = #hacc.arg_type<workspace>`).
- **Purpose**: Unifies the workspace source so that the runtime passes in the workspace pointer through the parameter, enabling multiple kernels to share a single workspace.
- **Prerequisites**: The function must have a workspace parameter; `InsertWorkSpaceForMixCV` has already inserted `alloc_workspace`.
- **Typical Scenario**: Cube-Vector mixed (CV mode).

Before Transformation:

```mlir
func.func @bind_workspace_arg(
              %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
              %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}){
  memref_ext.alloc_workspace() : memref<100xi32>
  return
}
```

After Transformation:

```mlir
func.func @bind_workspace_arg(
              %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
              %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}){
  memref_ext.alloc_workspace() from %arg1 : memref<100xi32>
  return
}
```

### createPlanMemoryPass

- **Function**: In `GLOBAL_WORKSPACE_PLAN` mode, performs memory planning for `memref_ext.alloc_workspace`, replacing the alloc with `hivm.hir.pointer_cast` + offset.
- **Purpose**: On a given workspace base address, allocates offsets according to liveness and inplace rules to maximize reuse and reduce the total workspace size.
- **Typical Transformation**: Multiple alloc_workspace operations are mapped to different offsets of the same workspace; conflicting buffers are allocated different offsets.

Before Transformation:

```mlir
func.func @bind_workspace_arg(
              %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
              %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}){
  memref_ext.alloc_workspace() from %arg1 : memref<100xi32>
  return
}
```

After Transformation:

```mlir
func.func @bind_workspace_arg(
              %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
              %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}){
  memref_ext.alloc_workspace() from %arg1 offset=[0] : memref<100xi32>
  return
}
```

### createSplitMixKernelPass

- **Function**: Splits a mixed kernel into two sub-functions, AIC (Cube-primary) and AIV (Vector-primary), and generates a mix entry.
- **Purpose**: The backend can schedule AIC and AIV to the Cube and Vector cores respectively, facilitating pipelining and synchronization.
- **Typical Transformation**: Traverses the IR according to the core type, placing Cube-related code into AIC and Vector-related code into AIV; uses `annotation.mark` to mark tensors passed across cores.

Before Transformation:

```mlir
func.func @bind_workspace_arg(
              %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
              %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
     hivm.func_core_type = #hivm.func_core_type<MIX>){
  mmadl1
  memref_ext.alloc_workspace() from %arg1 offset=[0] : memref<100xi32>
  fixpipe
  load
  vadd
}
```

After Transformation:

```mlir
func.func @bind_workspace_arg_aic(
              %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
              %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
     hivm.func_core_type = #hivm.func_core_type<AIC>){
  mmadl1
  memref_ext.alloc_workspace() from %arg1 offset=[0] : memref<100xi32>
  fixpipe
}
func.func @bind_workspace_arg_aiv(
              %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
              %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
     hivm.func_core_type = #hivm.func_core_type<AIV>){
  load
  vadd
}
```

## Local Debugging and Testing Methods

**Single Pass Debugging Command**:

```bash
bishengir-opt -hivm-normalize-matmul input.mlir -o output.mlir
bishengir-opt -hivm-inline-fixpipe input.mlir -o output.mlir
bishengir-opt --hivm-tile-batchmm-into-loop input.mlir -o output.mlir
bishengir-opt -insert-workspace-for-mix-cv input.mlir -o output.mlir
bishengir-opt --hivm-bind-workspace-arg input.mlir -o output.mlir
bishengir-opt -hivm-plan-memory -mem-plan-mode=global-work-space-plan input.mlir -o output.mlir
# Explicitly run the standalone split-mixed-if when needed; TCB mark/hoist is only for Ascend950.
bishengir-opt -hivm-split-mixed-if-conditionals -hivm-mark-tightly-coupled-buffer \
  -hivm-hoist-tightly-coupled-alloc -hivm-split-mix-kernel input.mlir -o output.mlir
```

**Test Case Storage Path**:

Currently, all test cases in the library are located under `path-to-ascendnpuir/bishengir/test`. To run a specific pass, search for the corresponding compilation command to find the relevant test file. For example, searching for `hivm-normalize-matmul` will locate the corresponding test file `bishengir/test/Dialect/HIVM/normalize-matmul.mlir`.

**Test Execution Method**:

The specific run command is provided at the top of each test file. For example:

```bash
// RUN: bishengir-opt -hivm-normalize-matmul %s -split-input-file -verify-diagnostics -allow-unregistered-dialect | FileCheck %s
```

Here, `bishengir-opt` and `FileCheck` are both compiled binary executables located under `path-to-ascendnpuir/build/bin`. In the command above, replace `%s` with the corresponding test file `bishengir/test/Dialect/HIVM/normalize-matmul.mlir`.

The output `mlir` is matched against the `CHECK:` sections in the test file. The test succeeds if no `CHECK failed` errors are reported.

## Constraints

- `createPlanMemoryPass` handles the space size at data interaction points. Because it dynamically returns the total space size required by the data, there is no limit on the size.
- `createInlineFixpipePass` can currently inline only three types of ops: `vcast`, `relu`, and `store`.
