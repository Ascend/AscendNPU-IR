# Multi-Buffer

Multi-buffer (Multi Buffer) is a fundamental performance optimization in AscendNPU IR. It expands a buffer inside a loop body into N physical memory copies, so that the loads and computations of adjacent iterations fall on different memory slots, thereby overlapping multiple originally serial hardware pipelines. When N equals 2, it is also called Double Buffer (ping-pong pipeline), which is the most commonly used configuration.

This document focuses on why multi-buffer delivers performance gains, the forms of these gains at each memory hierarchy level, and the compilation options that control this optimization.

## Hardware Background

The Ascend AI Core contains multiple mutually independent pipelines (PIPEs) that can execute in parallel. An instruction is queued and executed only on the pipeline to which it belongs:

| Pipeline | Typical HIVM Operation | Function |
|-----|-----|-----|
| MTE2 | `hivm.hir.load`, `hivm.hir.nd2nz` | Load from GM into L1 or UB |
| MTE1 | Load phase of `hivm.hir.mmadL1` | Load from L1 into L0A and L0B |
| M (Cube) | Compute phase of `hivm.hir.mmadL1` | Matrix multiplication |
| V (Vector) | Vector operations such as `hivm.hir.vadd` | Vector computation |
| FIX | `hivm.hir.fixpipe` | Move L0C results out, including post-processing such as quantization and ReLU |
| MTE3 | `hivm.hir.store`, `hivm.hir.nz2nd` | Move UB or L1 out to GM |

Each pipeline is executed asynchronously by the hardware, and their ordering is established by synchronization operations such as `hivm.set_flag` and `hivm.wait_flag` inserted by the compiler. For details, see [Auto Sync - Hardware Background](./auto_sync.md#hardware-background). The on-chip buffers at each level (UB, L1, L0A, L0B, L0C) have limited capacity and require explicit address allocation. For details, see [Memory Management - Hardware Background](./plan_memory.md#hardware-background).

The key point is that pipelines have parallel capability, but data dependencies consume that parallelism.

## Sources of Performance Benefits

### Eliminating Inter-Iteration Anti-Dependencies

The typical form of a loop-based operator is "load → compute → store". If the entire loop uses only one buffer, an anti-dependency (WAR, Write-After-Read) forms between iterations: the load of iteration `i+1` must write new data into the buffer, and therefore must wait until the compute of iteration `i` has finished reading that buffer. As a result, the three pipelines—MTE2, V, and MTE3—which could otherwise run in parallel, are forced to execute serially:

```text
Single buffer (1 buffer):
iter 0: [MTE2 load][ V compute ][MTE3 store]
iter 1:                                     [MTE2 load][ V compute ][MTE3 store]
iter 2:                                                                          [MTE2 load]...

Time ≈ number of iterations × (T_load + T_compute + T_store)
```

After multi-buffer is enabled, the buffer is expanded to N pieces, and iteration `i` uses piece `i % N`. Adjacent iterations access different physical addresses, so the WAR dependency between iterations disappears, leaving only the true dependency (RAW) within the same buffer piece:

```text
Double buffer (2 buffers):
iter 0: [MTE2 load buf0][ V compute buf0 ][MTE3 store buf0]
iter 1:                 [MTE2 load buf1][ V compute buf1 ][MTE3 store buf1]
iter 2:                                 [MTE2 load buf0][ V compute buf0 ]...

Time in steady state ≈ number of iterations × max(T_load, T_compute, T_store)
```

Multi-buffer turns "the sum of the durations of all stages" into "the duration of the longest stage", which is essentially a latency-hiding technique that trades memory for time. From this, the criteria for evaluating the benefit can be derived:

- The upper bound of the benefit is determined by how balanced the pipeline is. The benefit is most pronounced when the load and compute durations are comparable; when one pipeline is an absolute bottleneck (for example, when the compute duration is far greater than the load duration), multi-buffer can only hide the duration of the non-bottleneck portion, and the benefit is limited.
- The benefit requires a sufficient number of iterations to be amortized. The first load and the last store outside the steady state cannot be hidden, so when the loop count is small, the benefit is offset by this overhead.
- Increasing N does not increase the benefit proportionally. Increasing N from 1 to 2 is sufficient to eliminate the WAR dependency between adjacent iterations. Further increasing N yields additional benefit only when the pipeline has more stages (for example, a CV software pipeline) or when the duration of a single load fluctuates significantly, while memory usage grows linearly.

### Benefit Patterns at Different Memory Hierarchy Levels

The compiler identifies multi-buffer candidates by memory hierarchy level. Different levels overlap different pipelines, resulting in different benefit patterns:

| Memory Hierarchy | Operation That Triggers the Marker | Overlapped Pipeline | Benefit Pattern |
|-----|-----|-----|-----|
| GM (Workspace) | `hivm.hir.store` and `hivm.hir.fixpipe` that write to Workspace | Cross-core parallelism between Cube core and Vector core | In MIX operators, the Cube computation result is passed to Vector through Workspace for further processing. Multiple Workspace copies allow the two cores to work simultaneously with a one-iteration offset, rather than waiting for each other |
| L1 (cbuf) | `hivm.hir.nd2nz` | MTE2 with MTE1 and M | The load of the left and right matrices for matrix multiplication overlaps with the Cube computation of the current tile |
| L0C | `hivm.hir.fixpipe` | M with FIX | While the previous matrix multiplication result is being moved out of L0C, the next matrix multiplication can begin accumulation |
| UB | `hivm.hir.load` and `hivm.hir.store` | MTE2, V, and MTE3 | For pure Vector operators and the Vector side of MIX operators, the load, compute, and store stages run in a three-stage parallel pipeline |

Among these, multi-buffer at the GM Workspace level is the prerequisite for MIX operators (such as FlashAttention) to achieve Cube and Vector core parallelism. It is used together with [Cube and Vector Software Pipeline Optimization - Hardware Background](./cv_pipelining.md#hardware-background), and the number of software pipeline stages equals the number of Workspace multi-buffers.

### Relationship with Memory Reuse Strategy

The benefits of multi-buffer depend on the premise that different slots do not interfere with each other. When on-chip memory is tight, the Level 1 strategy in [Memory Management - Algorithm Principles](./plan_memory.md#algorithm-principles) protects this premise: if a single-buffer Buffer within the same loop reuses the space of a multi-buffer Buffer, the multi-buffer cannot actually run in parallel and the pipeline is interrupted; therefore, that single buffer is automatically converted to multi-buffer. This also explains why the increase in memory usage after enabling multi-buffer may exceed intuitive expectations.

## Cost and Risk

The cost of multi-buffer is N times the memory usage, which introduces two types of issues that need to be weighed:

- **Memory overflow**: UB, L1, and L0C have limited capacity, and N times the usage may cause PlanMemory allocation to fail and report an overflow. The compiler has an overflow fallback mechanism: on Ascend 950PR/Ascend 950DT, it first disables only the multi-buffer corresponding to the overflowing memory space and retries, and if yielding item by item is ineffective, it then disables the global switch; on Atlas A3 product series and Atlas A2 product series, it disables `--enable-code-motion` and then the multi-buffer global switch in sequence and retries. The fallback ensures successful compilation, but it means the benefits at the corresponding level are lost; in this case, a more appropriate approach is to reduce the tiling block size. Enabling `--enable-tuning-mode` disables this retry behavior, causing the overflow to be exposed directly as a compilation failure.
- **Smaller blocks become slower instead**: Reducing the tiling block size to accommodate multiple buffers lowers the transfer efficiency of a single DMA operation and increases the number of loop iterations and synchronization overhead. Therefore, it is not the case that "the more you enable, the faster it is"; this needs to be confirmed through actual measurements.

In addition, multi-buffer changes the synchronization structure (both the intra-core flag IDs and the cross-core event IDs need to rotate with the slots), which places higher demands on synchronization resolution in complex scenarios such as CV cross-core. If precision anomalies or hangs occur after enabling multi-buffer, you can first use `--enable-auto-multi-buffer=false` to narrow down the scope of the problem. For related troubleshooting methods, see [Debugging and Tuning - Debugging: Tool Classes](../../user_guide/debug_option.md#debugging-tool-classes).

## Algorithm Principle

Enabling multi-buffer consists of four steps, each handled by a different Pass:

1. **Marking**: `-hivm-mark-multi-buffer` identifies candidate buffers by memory hierarchy and generates `annotation.mark {hivm.multi_buffer = N}` at their allocation points. For local buffers, N is fixed at 2; for GM Workspace, N is specified by `--set-workspace-multibuffer`.
2. **Address allocation**: PlanMemory allocates N addresses for marked buffers and generates `hivm.hir.pointer_cast` carrying multiple offsets.
3. **Synchronization insertion**: Automatic synchronization assigns flag IDs and event IDs in a slot-rotating manner, ensuring that producers and consumers of the same slot are matched in pairs.
4. **Slot selection**: `-hivm-enable-multi-buffer` introduces the iteration counter `hivm.hir.multi_buffer_counter`, which takes the modulo of N to select the address actually used in the current iteration.

Example of IR form (using double buffering on UB as an example):

```mlir
// After marking and after PlanMemory allocates addresses: one pointer_cast carries 2 addresses
%p = hivm.hir.pointer_cast(%addr0, %addr1) : memref<1024xf16, #hivm.address_space<ub>>
annotation.mark %p {hivm.multi_buffer = 2 : i32} : memref<1024xf16, #hivm.address_space<ub>>

// After -hivm-enable-multi-buffer: split into 2 single-address pointer_casts, selected by the iteration counter
%p0 = hivm.hir.pointer_cast(%addr0) : memref<1024xf16, #hivm.address_space<ub>>
%p1 = hivm.hir.pointer_cast(%addr1) : memref<1024xf16, #hivm.address_space<ub>>
%counter = hivm.hir.multi_buffer_counter : i64
%slot = arith.remui %counter, %c2_i64 : i64
%is1 = arith.cmpi eq, %slot, %c1_i64 : i64
%active = arith.select %is1, %p1, %p0 : memref<1024xf16, #hivm.address_space<ub>>
```

When troubleshooting whether multi-buffer actually takes effect, search the IR for the `hivm.multi_buffer` marker and the `arith.select` structure described above; the absence of either indicates that the marking stage did not hit any candidate.

## Compilation Options

| Option | Description | Type | Default Value |
|--------|------|------|--------|
| --enable-auto-multi-buffer | Master switch for automatic multi-buffer. When disabled, the following three options do not take effect. | bool | true |
| --limit-auto-multi-buffer-only-for-local-buffer | Restricts multi-buffer to on-chip local buffers (UB, L1, L0C). When set to `true`, skips multi-buffer marking for GM Workspace, that is, disables the CV cross-core pipeline. | bool | false |
| --limit-auto-multi-buffer-of-local-buffer=\<value> | Restricts the multi-buffer scope of local buffers. `no-l0c` means multi-buffer is not enabled for L0C; `no-limit` means no restriction. | enum | no-l0c |
| --limit-auto-multi-buffer-buffer=\<value> | Restricts the side on which multi-buffer takes effect in MIX operators. `only-cube` means only the Cube side (L1, L0C); `only-vector` means only the Vector side (UB); `no-limit` means no restriction. | enum | See the description below. |

The default value of `--limit-auto-multi-buffer-buffer` depends on the target hardware:

- Ascend 950PR/Ascend 950DT: `no-limit`
- Atlas A3 training products/Atlas A3 inference products, Atlas A2 training products/Atlas A2 inference products: `only-cube`

When this option is explicitly passed, the user-specified value takes precedence.

Related multi-buffer count option: `--set-workspace-multibuffer` specifies the multi-buffer count for GM Workspace. It defaults to 2 on Ascend 950PR/Ascend 950DT and to 4 on Atlas A3 product series and Atlas A2 product series.

Configuration example:

```bash
# Disable all multi-buffer for performance comparison or issue localization.
bishengir-compile input.mlir --enable-auto-multi-buffer=false

# Keep only on-chip multi-buffer and disable the CV cross-core pipeline on GM Workspace.
bishengir-compile input.mlir --limit-auto-multi-buffer-only-for-local-buffer=true

# Fully enable on-chip, including L0C
bishengir-compile input.mlir --limit-auto-multi-buffer-of-local-buffer=no-limit

# Enable multi-buffer only for the UB on the Vector side of the MIX operator
bishengir-compile input.mlir --limit-auto-multi-buffer-buffer=only-vector
```

## Future Evolution

The three `limit`-type options above each constrain a portion of the memory hierarchy. Their semantics overlap with one another and are all expressed in a "restriction" manner, which makes them poorly readable: they can neither directly express "enable multi-buffer only for L1" nor specify different buffer counts for different levels.

In the future, the three options `--limit-auto-multi-buffer-only-for-local-buffer`, `--limit-auto-multi-buffer-of-local-buffer`, and `--limit-auto-multi-buffer-buffer` are planned to be deprecated and merged into a single positively-expressed `--multibuffer-mode` option, which directly specifies the multi-buffer count enabled for each of the four levels GM, L1, L0C, and UB:

```bash
--multibuffer-mode="[(gm, 2), (l1, 2), (l0c, 2), (ub, 2)]"
```

Value conventions:

- Greater than 1: multi-buffer is enabled for this level, and the count is the number of physical slots.
- Equal to 1: multi-buffer is disabled for this level (only one buffer exists).
- Equal to 0: invalid configuration, and a compilation error is reported.

For example, to enable multi-buffer only for L1:

```bash
--multibuffer-mode="[(gm, 2), (l1, 2), (l0c, 1), (ub, 1)]"
```

The equivalence between the new and old options is as follows, which can be used to migrate existing configurations. The following table uses Ascend 950PR/Ascend 950DT as the baseline, that is, the target is a MIX operator and the other options remain at their defaults (the GM count takes the default value 2 of `--set-workspace-multibuffer`):

| Configuration | Equivalent `--multibuffer-mode` |
|-----|-----|
| All defaults | `[(gm, 2), (l1, 2), (l0c, 1), (ub, 2)]` |
| `--enable-auto-multi-buffer=false` | `[(gm, 1), (l1, 1), (l0c, 1), (ub, 1)]` |
| `--limit-auto-multi-buffer-only-for-local-buffer=true` | `[(gm, 1), (l1, 2), (l0c, 1), (ub, 2)]` |
| `--limit-auto-multi-buffer-of-local-buffer=no-limit` | `[(gm, 2), (l1, 2), (l0c, 2), (ub, 2)]` |
| `--limit-auto-multi-buffer-buffer=only-cube` | `[(gm, 2), (l1, 2), (l0c, 1), (ub, 1)]` |
| `--limit-auto-multi-buffer-buffer=only-vector` | `[(gm, 2), (l1, 1), (l0c, 1), (ub, 2)]` |

> Note: `--multibuffer-mode` is not yet available. This section is only intended to illustrate the evolution direction. A one-version compatibility period will be retained when the option is switched, and it will be synchronized in the release notes at that time.

## Usage Constraints

- `--limit-auto-multi-buffer-only-for-local-buffer` and `--limit-auto-multi-buffer-buffer` take effect only for functions whose core type is MIX. Pure Cube or pure Vector operators have no GM Workspace handover and no Cube-side/Vector-side division, so configuring these two options does not change their behavior.
- A multi-buffer candidate must be located inside a loop, and all ancestor loops from the Buffer allocation point to the outermost loop must be `scf.for` or `scf.while`. Buffers located inside loops such as `scf.parallel` and `scf.forall` do not satisfy the premise of "rotating one slot per iteration" and will not be marked.
- The number of multi-buffers for a local Buffer is fixed at 2 and cannot be adjusted through compilation options; only the number of GM Workspaces can be configured through `--set-workspace-multibuffer`.
- Users must ensure that the N-fold Buffer usage does not exceed the capacity limit of the corresponding memory space; otherwise, PlanMemory reports an overflow. For the error form and the workaround, see [Memory Management - Usage Constraints](./plan_memory.md#usage-constraints).
