# Auto-Sync

Auto-sync is the AscendNPU-IR (HIVM) compiler feature that automatically inserts synchronization operations so producers and consumers of shared data or resources are correctly ordered. Goals: **correctness** (no data races or ordering bugs) and **minimal overhead** (fewest syncs needed, reuse of hardware events when safe).

## Hardware Background

### AICore Architecture

<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html>

### HIVM Synchronization Operations

Synchronization ops are defined in `HIVMIR/HIVMSynchronizationOps.td`. Below they are described in terms of **MLIR usage** (operands/attributes), not assembly syntax.

#### Intra-Core-Sync (Normal-Sync)

- **`hivm.set_flag`**  
  Operands/attributes: `set_pipe`, `wait_pipe` and `flag_id`  
  Executes on `set_pipe` after all previous instructions on that pipe have finished.
  Triggers `flag_id` on execution  

- **`hivm.wait_flag`**  
  Operands/attributes: `set_pipe`, `wait_pipe` and `flag_id`  
  Executes on `wait_pipe`  
  Blocks all following instructions until `flag_id` is triggered

- **`hivm.pipe_barrier`**  
  Operands/attributes: `pipe`  
  Barrier across a given pipe.  
  Block all following instructions on `pipe` until all previous instructions finish.

#### Cross-Core-Sync (Block-Sync) (Intra-Block)

- **`hivm.sync_block_set`**  
  Operands/attributes:  
  - `tcore_type`  target core type (vector/cube)  
  - `tpipe`, `pipe`  (set/wait pipes on target core)  
  - `sync_instr_mode` (default `INTRA_BLOCK_SYNCHRONIZATION`)  
  - `event_id`  

  Executes on `tpipe` (set_pipe) on the `tcore_type` core after all previous instructions on the same core.pipe finish.
  Sets `event_id`

- **`hivm.sync_block_wait`**  
  Operands/attributes:  
  - `tcore_type`  target core type (vector/cube)  
  - `tpipe`, `pipe`  (set/wait pipes on target core)  
  - `sync_instr_mode` (default `INTRA_BLOCK_SYNCHRONIZATION`)  
  - `event_id`  

  Executes on `pipe` (pipe_wait) on the `tcore_type`  
  Block all following instructions on `pipe` on the `tcore_type` core until all previous instructions finish.

## Algorithm principles

### AutoSync solutions overview

The codebase provides **two** auto-sync solutions:  

- **`Inject-Sync/Inject-Block-Sync`** Passes  

  Uses multiple passes to insert needed sync operations, remove redundant ones, and allocate flag IDs/event IDs using liveliness analysis. It is the primary solution enabled by default.

- **`Graph-Sync-Solver/Cross-Core-GSS`** Passes  

  Uses graph-based algorithms to analyze the input code structure and insert needed sync operations. It remains optional and can be enabled via `-hivm-enable-graph-sync-solver=true` (or `sync_solver=True` in Triton-Ascend).

### InjectSync

![alt text](../../../../images/developer_guide/auto_sync0.png)

**Purpose:** Insert core-level (intra-core) synchronization (`set_flag` / `wait_flag`) using memory-dependence analysis, sync analysis, event-id allocation, and cleanup (move/remove redundant syncs).

**Source:**  

- Headers: `include/../InjectSync/`.  
- Implementation: `lib/../InjectSync/` (e.g. `InjectSync.cpp`, `MemoryDependentAnalyzer.cpp`, `SyncAnalysis.cpp`, `SyncEventIdAllocation.cpp`, `IRTranslator.cpp`, `SyncCodegen.cpp`, `MoveSyncState.cpp`, `RemoveRedundantSync.cpp`, `SyncCommon.cpp`).  

**Stages:**  

1. **IRTranslator**:  
   Build Sync-IR from the input function (compound elements, loops, conditions, memory ops).  
2. **SyncAnalyzer**:  
   For each pair of conflicting operations, it inserts a pair of set_flag/wait_flag operations or a barrier(pipe) operations if both operations are of same pipe.
3. **MoveSyncState**:  
   Reposition sync ops to reduce stalls while preserving semantics.  
4. **RemoveRedundantSync**:  
   Drop redundant sync pairs.  
5. **SyncEventIdAllocation**:  
   Assign static or dynamic event IDs; reuse when safe.  
6. **SyncCodegen**:  
   Emit `hivm.set_flag` / `hivm.wait_flag` / `hivm.barrier`

### InjectBlockSync

**Purpose:** Insert block-level (intra-block) (cross-core) synchronization for **MIX** kernels (cube and vector): `sync_block_set`, `sync_block_wait`.

**Source:** `InjectBlockSync.cpp` `InjectBlockSync.h`  

**Behavior**:

- Runs only on **MIX** kernels (not host, not pure AIC/AIV).
- Inserts `SetFFTSBaseAddrOp` when an FFTS base addr kernel argument is present.
- Three modes (controlled by options and fusion kind):
  - **InjectAllBlockSync** — Emit block sync before/after every `LoadOp` and every `StoreOp` (cube/vector handoff).
  - **InjectBlockMixSync** — Full mix: build block sync IR via `SyncBlockIRTranslator`, then run SyncAnalyzer (BLOCKSYNC mode), MoveSyncState, RemoveRedundantSync, SyncEventIdAllocation, SyncCodegen.

### GraphSyncSolver

![alt text](../../../../images/developer_guide/auto_sync1.png)

**Purpose:** Alternative to the Inject-Sync solution; it uses graph-based algorithms to decide when to insert pairs of set/wait operations and assign event IDs.

**Source:**  

- Headers: `include/../GraphSyncSolver/`
- Implementation: `lib/../GraphSyncSolver/` (`GraphSyncSolver.cpp`, `SyncSolver.cpp`, `SyncSolverIR.cpp`, `SyncSolverIRTranslator.cpp`, `SyncSolverCodeGen.cpp`, `GraphSolver.cpp`, `EventIdSolver.cpp`, `Utility.cpp`, `SyncSolverTest.cpp`, `SyncSolverTester.cpp`).  

**Stages**:

1. **IRTranslator**:  
Build Sync-IR from the input function (function, scopes, loops, conditions, rw-operations).  
2. **Solver**:  
Collect conflict pairs (producer–consumer pairs), run pair selection and ordering, optionally reuse conflict pairs to save event IDs.  
3. **CodeGenerator**:  
Translate solver result back to MLIR: emit `hivm.set_flag` / `hivm.wait_flag` / `hivm.barrier`

### CrossCoreGSS

**Purpose:** Insert block-level (intra-block) (cross-core) synchronization for **MIX** kernels (cube and vector): `sync_block_set`, `sync_block_wait`.

**Source:** `CrossCoreGSS.h` `CrossCoreGSS.cpp`; reuses `IRTranslator`, `Solver`, and `CodeGenerator` from GraphSyncSolver.

**How it works**:

- Same as the intra-core GSS pass, but it handles cross-core memory operations.

## Interface description

### Command Line Options

These are typically wired in the compiler driver (e.g. `bishengir-hivm-compile`); see `Passes.td` and tools under `bishengir/lib/Tools/` for exact mapping.

<table>
  <thead>
    <tr>
      <th style="white-space: nowrap;">Flag</th>
      <th>Type</th>
      <th>Default</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="white-space: nowrap;">`--disable-auto-inject-block-sync`</td>
      <td>bool</td>
      <td>false</td>
      <td>Disable automatic block-level set/wait insertion (InjectBlockSync / CrossCoreGSS).</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;">`--disable-hivm-auto-inject-sync`</td>
      <td>bool</td>
      <td>false</td>
      <td>Disable InjectSync (Intra-Core sync).</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;">`--enable-hivm-inject-barrier-all-sync`</td>
      <td>bool</td>
      <td>false</td>
      <td>Make InjectSync inserts barrier(all) instructions (useful auto-sync fails)</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;">`--enable-hivm-inject-block-all-sync`</td>
      <td>bool</td>
      <td>false</td>
      <td>Make InjectBlockSync inserts block(all) instructions (useful auto-sync fails)</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;">`--enable-hivm-unit-flag-sync`</td>
      <td>bool</td>
      <td>false</td>
      <td>Enable unit-flag sync feature.</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;">`--enable-hivm-graph-sync-solver`</td>
      <td>bool</td>
      <td>false</td>
      <td>Use GraphSyncSolver/CrossCoreGSS instead of InjectSync/InjectBlockSync for Intra-Core/Cross-Core auto-sync.</td>
    </tr>
  </tbody>
</table>

## Constraints and capabilities

- **Hardware ordering model:** Auto-sync orders execution by inserting HIVM synchronization ops (`hivm.set_flag` / `hivm.wait_flag`, `hivm.pipe_barrier`, and (when applicable) `hivm.sync_block_set` / `hivm.sync_block_wait`). The ordering is expressed in terms of **cores** and **pipes**, plus event/flag ids.
- **Correctness via feasibility checking:** For the solver-based flow (Graph Sync Solver), candidate sync constraints are accepted only if they remain feasible under a graph-based reachability/ordering model (avoids deadlock or over-constraining schedules).
- **Kernel coverage (block-level sync):** Block-level cross-core sync (`sync_block_set` / `sync_block_wait`) is intended for **MIX** kernels (cube/vector handoff). InjectBlockSync/CrossCoreGSS will not apply to non-MIX flows (host or pure AIC/AIV).
- **Optional feature modes:** Unit-flag sync can be enabled as an alternative pattern for supported operations, and the graph-based solver can be selected instead of InjectSync/InjectBlockSync via compiler options.

- **Verification requirements:** Check that emitted ops satisfy dialect verification; `set_flag` / `wait_flag` must share the same event/flag id and compatible core/pipe endpoints.
