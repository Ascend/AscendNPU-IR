# Storage Alignment

**Applicable Products**:

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 training products/Atlas A3 inference products
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 training products/Atlas A2 inference products
<!-- end id3 -->

This document introduces the stride alignment mechanism in HIVM, including the hardware background, algorithm principles, interface descriptions, and usage constraints of the three passes `hivm-pre-mark-stride-align`, `hivm-mark-stride-align`, and `hivm-enable-stride-align`.

Stride alignment is divided into two phases: "mark" and "enable". In the marking phase, the memref operands of each HIVM operator are analyzed to determine the dimensions that require alignment and the number of alignment bytes, which are then written into the IR in the form of `annotation.mark` annotations. In the enabling phase, these annotations are read, and the marking information is propagated to the root allocation point (`memref.alloc`). The allocated memory shape is padded so that the starting address of each row satisfies the hardware alignment requirement, and then sliced back to the original logical shape through `memref.subview`.

## Hardware Background

The on-chip memory of Ascend hardware (UB, L1, L0C, and so on) imposes strict address alignment requirements on data access. The alignment requirements and functions of each type of buffer are as follows:

| Buffer | Alignment Requirement | Function |
|-----|-----|-----|
| Unified Buffer (UB) | 32-byte alignment | General-purpose cache space, mainly used for vector and scalar operations |
| L1 Buffer | 32-byte alignment | Staging for feature maps and other data used in convolution |
| L0A Buffer | 512-byte alignment | Staging for the left matrix (feature map) of matrix operations |
| L0B Buffer | 512-byte alignment | Staging for the right matrix (weight) of matrix operations |
| L0C Buffer | 512-byte alignment | Staging for intermediate results and the output matrix of matrix operations |
| BT Buffer | 64-byte alignment | BiasTable Buffer, storing the bias in matrix operations |
| FP Buffer | 64-byte alignment | Fixpipe Buffer, storing quantization parameters, ReLU parameters, and so on |

In addition to the alignment of the buffer's own start address, when the lowest dimension (last dim) of a memref is non-contiguous—that is, when the stride of the sub-tail dim is not equal to the size of the lowest dimension—the hardware accesses data row by row, and the start address of each row must also satisfy the alignment constraint. For example, for `memref<3x13xi8, strided<[16, 1]>>`, the lowest dimension has a size of 13 but a stride of 16, so the lowest dimension is non-contiguous; the 16-byte row width does not satisfy the hardware alignment requirement (32 bytes), so the lowest dimension must be padded to the alignment size of 32 at allocation time.

DMA transfer instructions and vector computation instructions each impose their own constraints on stride:

- DMA instructions (`copy_gm_to_ub`, `copy_ub_to_gm`) require that the start address of each row of the transferred UB memref be aligned to 32 bytes (or the hardware-corresponding alignment granularity).
- Fixpipe (L0C→UB transfer) has more complex M/N-axis alignment constraints, depending on the data bit width, ChannelSplit mode, NZ2ND/NZ2DN conversion mode, and so on.
- The `vsstb` (stride store with broadcast) instruction recommends that the stride be aligned to (512\*N + 32) bytes to avoid bank conflicts.

## Related Terms

| Term | Meaning |
|-----|-----|
| stride_align_dims / stride_align_value_in_byte | Two annotation attributes that record the dimension index to be aligned (starting from 0) and the number of alignment bytes, respectively. They are attached to a memref Value via the `annotation.mark` op. For example, `{hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>}` means that the stride of dimension 1 is aligned to 32 bytes. |
| last discontinuous dim | Scanning from the lowest dimension (the rightmost dimension) toward higher dimensions, the first dimension whose stride != size. The stride of this dimension is the row width, which must satisfy the alignment requirement. If the lowest dimension's stride is already a unit stride (that is, stride == size, meaning the lowest dimension is contiguous), the search continues toward higher dimensions. |
| Align Propagation | The mark may fall on the result of view-like ops such as `memref.subview` and `memref.cast`, rather than directly on `memref.alloc`. Alignment propagation traces the mark upward along the view chain to the root allocation point, while also propagating it downward to leaf operands, and takes the union across multiple operands of the same operator, ensuring that all related buffers use consistent alignment information. |
| SkipStrideAlignForVLoad | A skip mark set on the root alloc during the pre-marking phase (PreMarkStrideAlign). Some buffers, after being loaded by DMA, are read in a Vector Function by `vlds` (which supports only contiguous loads). If stride alignment is applied to them, inter-row gaps are introduced, and `vlds` cannot handle a non-contiguous layout, so alignment must be skipped. |

## Algorithm Principles

### Software Background

Before PlanMemory completes address allocation, the `memref.alloc` in the input IR only declares the logical shape and the required memory size, without address information. If the logical shape of a buffer is not aligned in the second-lowest dimension (for example, `memref<37x5x3xi32, strided<[15, 3, 1]>>`, where the stride of the second-lowest dimension is 15), when the DMA template implementation converts it to 2D processing, the start addresses of each row are not aligned, which causes functional exceptions.

The storage alignment mechanism solves this problem in the following ways:

1. In the marking phase, analyze the memref operands of each HIVM operator to find the "last discontinuous dim", that is, the first dimension from the lowest dimension upward whose stride is not equal to its size. The stride of this dimension is the row width that needs to be aligned.
2. In the enabling phase, pad the corresponding dimension of the root `memref.alloc` to a size that meets the hardware alignment requirement (for example, pad the lowest dimension of `memref<37x5x3xi32>` from 3 to 8, so that stride=8\*4 bytes = 32 bytes meeting the alignment), and then slice the memory back to the original logical shape through `memref.subview`.

In this way, the memory layout allocated at the lower level meets the hardware alignment constraints, while the upper-level operators still access data according to the original logical shape.

### Pass Pipeline

The three passes are executed in the pipeline in the following order:

```text
hivm-align-alloc-size          // Adjust the alloc size of special operators
  └─ hivm-pre-mark-stride-align   // [reg-based only] Pre-analysis: mark allocs that skip alignment
  └─ hivm-mark-stride-align       // Marking phase: add stride_align annotations to operator operands
hivm-enable-stride-align        // Enabling phase: propagate annotations + reallocate memory
```

- The marking phase runs at the FuncOp level (`func::FuncOp`) and can process different functions in parallel.
- The enabling phase runs at the ModuleOp level (`mlir::ModuleOp`), because alignment propagation may cross `func.call` boundaries and affect Vector Function callees, and parallel nesting of FuncOp may cause data races.

<!-- npu="950" id4 -->
## hivm-pre-mark-stride-align

> **Note**:
>
> The content in this section applies only to the Ascend 950PR&950DT product series.

### Function Overview

This pass takes effect only on reg-based architectures (such as Ascend 950PR/Ascend 950DT) and serves as the pre-analysis step of the marking phase.

After certain buffers are moved from GM to UB by DMA (`hivm.hir.load`), they are read by `vector.transfer_read` (that is, vload) in a Vector Function (a VF invoked through `func.call`). If stride alignment is applied to these buffers, padding gaps are inserted between rows, making the memref layout in UB non-contiguous. However, the `vlds` instruction requires flat pointer loading of contiguous elements and cannot handle a non-contiguous layout.

Therefore, **PreMarkStrideAlign** detects such buffers in advance and adds the `hivm.skip_stride_align_for_vload` annotation to their root alloc, so that subsequent **MarkStrideAlign** skips them.

### Core Process

1. Traverse all `hivm.hir.load` (DMA load) operations in the function.
2. For the result buffer of each load, determine through dataflow tracking (passing through view-like ops such as `memref.subview` and `memref.cast`, as well as `func.call` call boundaries) whether it ultimately flows to a `vector.transfer_read` in some VF callee.
3. If it is confirmed to flow to a vload, trace back to the root alloc (`memref.alloc`), insert an `annotation.mark` after it, and set the `hivm.skip_stride_align_for_vload` attribute.

> Note: When a VF callee has only a declaration (no function body, such as after SplitSimtModule), it is impossible to determine whether the parameter is used by a vload, so the marking of that buffer is skipped.
<!-- end id4 -->

## hivm-mark-stride-align

### Function Overview

This pass traverses all HIVM structured operators (`HIVMStructuredOp`), automatically analyzes the layout of their memref operands, identifies the last non-contiguous dimension, and creates an `annotation.mark` annotation for operands in the UB space, recording the alignment dimension and the number of aligned bytes.

### Core Process

#### 1. Marking General Structured Operators

For each `HIVMStructuredOp`, perform the following steps:

1. Skip operators that are non-bufferized (tensor type) or contain only GM operands (GM does not require alignment).
2. Collect the memref operand types of the operator, and skip operands that are all rank-0 or all shape-1.
3. Determine whether it is a UB DMA operation (`hivm.hir.load`/`store`/`copy`). The alignment determination rule for DMA operations differs from that for ordinary compute ops.
4. Follow different analysis paths based on the architecture:
   <!-- npu="950" id5 -->
   - **reg-based (Ascend 950PR&950DT)**: Because the `hfusion-flatten` of the Ascend 950PR&950DT product series has already merged the axes, the memref is directly treated as already flattened, and `getLastDiscontinuousDimRegBased` is used to find the last non-contiguous dimension. Dedicated alignment constraint computation is performed for Fixpipe.
   <!-- end id5 -->
   <!-- npu="A3,910b" id6 -->
   - **non-reg-based (Atlas A2 products, Atlas A3 products)**: Obtain the flattened association group and type via `getFlattened` of `FlattenInterface`, use `getLastDiscontinuousDim` to search on the flattened type, and then map back to the original dimension.
   <!-- end id6 -->
5. Take the operands in the UB space (`getTargetSpaceOperands(UB)`), and call `markAlignedDim` to create an alignment annotation for each operand.

#### 2. Finding the Last Non-contiguous Dimension

The search logic of `getLastDiscontinuousDim`:

- If the stride of the lowest dimension of any memref is not 1 (the lowest dimension is non-contiguous), the lowest dimension is the alignment target dimension.
- If the stride of the lowest dimension is 1 (the lowest dimension is contiguous), search the higher dimensions for the last dimension whose stride is not equal to its size.
<!-- npu="950" id7 -->
- For UB DMA operations (`copy_gm_to_ub`, etc.), on the Ascend 950PR&950DT product series, only the stride of the lowest dimension is considered. If the second-lowest dimension is already aligned and the lowest dimension has no tail-jump issue, no alignment is required.
<!-- end id7 -->
- For a 1D memref, no non-contiguous dimension exists, and `nullopt` is returned.

#### 3. Fixpipe Special Handling

The alignment constraints of Fixpipe (L0C→UB transfer) are more complex and depend on:

| Parameter | Description |
|------|------|
| DataWidthType | Data bit width: B4 (4 bit), B8, B16, B32 |
| ChannelSplit | Whether channel split (CS_N/CS_Y) |
| NZ2ND | Whether NZ→ND conversion |
| NZ2DN | Whether NZ→DN conversion |
| LoopEnhance | Whether loop enhancement |
| DualDstMode | Dual-destination mode: N, SplitN, SplitM |

N-axis alignment requirements (partial examples):

| Data Bit Width | NZ2ND | Number of Aligned Elements |
|----------|-------|-----------|
| B16 | ND_N | 16 |
| B8 | ND_N | 16 |
| B8 | ND_Y | 32 |
| B4 | ND_Y | 64 |
| B32 | ND_N | 16 |
| B32 | ND_Y | 8 |

M-axis alignment requirements (partial examples):

| Data Bit Width | NZ2DN | Number of Aligned Elements |
|----------|-------|-----------|
| B16 | DN_Y | 16 |
| B8 | DN_Y | 32 |
| B4 | DN_Y | 64 |
| B32 | DN_N(SplitM) | 2 |
| B32 | DN_Y | 8 |

In addition, for the Fixpipe of 3D-dot, alignment must also be performed on dimension 0 (the tiling dimension).

#### 4. vsstb Bank Conflict Avoidance

When a VF function contains `vector.transfer_write` and is suitable for stride store (`vsstb`), MarkStrideAlign adds a higher alignment requirement to the corresponding parameter of `func.call` to avoid bank conflicts. The alignment size is calculated as:

```text
alignSize = shape[subTailDim] * 16 / gcd(shape[subTailDim], 16) + 1
```

where `subTailDim = rank - 2` (the second-to-last dimension), and 16 = 512 bytes / 32 bytes (the ratio of the alignment granularity to the size of the lowest dimension). The final number of aligned bytes must be multiplied by the data type width.

#### 5. rank-reducing subview fix

When the UB operand of a Store operation passes through a trailing-unit rank-reduced subview (for example, `memref<3x1x1xi32>` → `memref<1x1xi32>`), the alignment dimension may fall on a discarded dimension. In this case, the marking target must be traced back from the subview result to the source of the subview, and the alignment dimension index must be adjusted.

<!-- npu="950" id11 -->
#### 6. Vector Function Parameter Marking (reg-based only)

For the parameters of `func.call`, if they have not yet been marked and belong to a local Buffer, the same last non-contiguous dimension analysis as for structured operators is applied, and an alignment annotation is added to the parameters. Parameters marked as `SkipStrideAlignForVLoad` by PreMarkStrideAlign are also skipped.
<!-- end id11 -->

### Marking Effect Example

```text
// Before marking: the lowest dimension of UB alloc is non-contiguous
%alloc = memref.alloc() : memref<37x5x3xi1, #hivm.address_space<ub>>
hivm.hir.load ins(%gm : ...) outs(%alloc : memref<37x5x3xi1, strided<[15, 3, 1]>, ...>)

// After marking: add a 32-byte alignment annotation on dimension 2
annotation.mark %alloc {hivm.stride_align_dims = array<i32: 2>, hivm.stride_align_value_in_byte = array<i32: 32>}
```

## hivm-enable-stride-align

### Function Overview

This pass reads the `stride_align` annotation generated in the marking phase, aggregates the annotation to the root allocation point through multiple rounds of propagation, then performs shape padding on `memref.alloc`/`memref.alloca` so that the starting address of each row meets the hardware alignment requirement, and finally switches back to the original logical shape through `memref.subview`.

### Core Process

1. Normalize alignment information (NormalizeAlignInfoPattern)
   Sort the `stride_align_dims` and `stride_align_value_in_byte` attributes on each `annotation.mark` to ensure dimension indices are in ascending order, facilitating subsequent union and comparison operations during propagation.

2. Propagate upward to the root Alloc (PropagateAlignUpToRootAllocationPattern)
   Trace alignment marks on the results of view-like ops such as `memref.subview`, `memref.cast`, `memref.collapse_shape`, `memref.expand_shape`, `memref.reshape`, and `memref.view` back along the definition chain to the root `memref.alloc`/`memref.alloca`, and attach the alignment information to the root alloc.

   > Constraint: A memref must not be returned by an SCF op (for example, `scf.if` yielding a memref). Such control flow should be sunk by other passes before the marking phase.

3. Propagate downward to leaf operands (PropagateAlignDownToLeafOperandsPattern)
   Propagate the alignment information on the root alloc downward to all leaf operands that directly or indirectly use that alloc. At the same time, create `annotation.mark` for allocs that lack marks (`AddAlignAnnotationMarkForAlloc`).

4. Propagate among operation operands (PropagateAlignAmongOperationOperands)
   For each HIVM Vector operation and `hivm.hir.copy`, collect the alignment information of its non-temporary operands in UB space, take the union, and propagate it to operands that are unmarked or incompletely marked. For example:

   ```text
   // The op has three UB operands A, B, and C, where A and B carry alignment marks.
   // After propagation, C also obtains the same alignment mark.
   %A = ... {stride_align_dims=[1], value=[32]}
   %B = ... {stride_align_dims=[1], value=[32]}
   hivm.hir.vadd ins(%A, %B) outs(%C : ...)  // C obtains {stride_align_dims=[1], value=[32]}
   ```

   The first operand (source) of a Copy operation does not participate in propagation; propagation applies only to the destination operand.

5. Iterate to a fixed point
   Steps 3, 4, and 2 are iterated repeatedly until the alignment information no longer changes or the maximum number of iterations (10) is reached. If convergence is not achieved after exceeding the maximum number of iterations, an error is reported and the process exits.

6. Remove annotations (RemoveAlignMarkPattern)
   After propagation is complete, remove the `stride_align_dims` and `stride_align_value_in_byte` attributes from all `annotation.mark` operations, leaving empty mark ops (or leaving them to be cleaned up by a subsequent pass). This step only clears the intermediate propagation information attached to `annotation.mark`; it does not clear the alignment attributes that have already been propagated to `memref.alloc`/`memref.alloca`. The subsequent `EnableAlignAllocation` still relies on the alignment attributes on alloc to complete memory reallocation.

7. Enable alignment allocation (EnableAlignAllocation)
   For each `memref.alloc`/`memref.alloca` carrying the `stride_align_dims` attribute:

   - 7.1 Collect alignment units (`collectAlignUnits`): Convert the number of alignment bytes to the number of aligned elements (divide by the element bit width), taking architecture differences into account.
   - 7.2 Calculate the aligned shape (`calculateAlignedShape`): For each dimension that requires alignment, round its size up to a multiple of the alignment unit (`AlignUp`). Also compute the original logical shape (sub_shape).
   - 7.3 Create a new alloc whose type is the aligned shape with an identity layout.
   - 7.4 Create a `memref.subview` to slice the original logical shape out of the aligned alloc.
   - 7.5 Call `replaceAndPropagateMemRefType` to replace all uses of the original alloc with the new subview and propagate the memref type change.
   - 7.6 Synchronously update the `buffer_size_in_byte` annotation (scaled by the ratio of the static volume after alignment to that before alignment).
   - 7.7 Delete the original alloc.

   If the aligned shape is identical to the original shape (no padding required), only remove the alignment attribute without creating a new alloc.

8. AIC-to-AIV alignment propagation (supported only on the Ascend 950PR&950DT product series)
   For a TightlyCoupledBuffer, the AIC and AIV share the same physical memory. If the alloc on the AIC side performs stride alignment, the corresponding alloc on the AIV side also requires the same alignment. `AddStrideAlignInfoForAiv` matches the AIC and AIV allocs by the TightlyCoupledBuffer ID and propagates the alignment information from the AIC to the AIV.

9. Post-processing
   - `handlePropagateFailure`: handles ops whose propagation fails (unrealized_conversion_cast).
   - `materializeRemainingStaticUBLayoutCasts` (reg-based only): materializes the static UB layout-changing casts generated by stride-align.
   - Set the `hivm.storage_aligned` attribute (non-reg-based only): marks that the function has completed storage alignment.

### Enabling Effect Example

```text
// Before enabling: alloc with alignment mark
%alloc = memref.alloc() {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<37x5x3xi32, #hivm.address_space<ub>>
// Dimension 2 size=3, stride requires 32-byte alignment → 32/4(i32)=8 elements → pad the lowest dimension to 8

// After enabling: allocate the aligned shape, then subview back to the original shape
%aligned_alloc = memref.alloc() : memref<37x5x8xi32, #hivm.address_space<ub>>
%subview = memref.subview %aligned_alloc[0, 0, 0] [37, 5, 3] [40, 8, 1] : memref<37x5x8xi32, ...> to memref<37x5x3xi32, strided<[40, 8, 1]>, ...>
// stride changes from [15,3,1] to [40,8,1], each row start address satisfies 32-byte alignment
```

## Compilation Options

| Option | Default Value | Description |
|------|--------|------|
| `enable-hivm-auto-storage-align` | true | Pipeline option that controls whether the marking and enabling phases are executed in `alignStoragePipeline` |

The three passes themselves do not expose independent command-line options; they are all controlled uniformly through the `enableHIVMAutoStorageAlign` option. When this option is disabled, `hivm-pre-mark-stride-align` and `hivm-mark-stride-align` are not executed, but `hivm-enable-stride-align` is still executed (in this case, because there is no marking annotation, no actual memory reallocation occurs).

## Usage Constraints

1. **Must run after bufferization**: The marking phase requires operators to have passed the `hasPureBufferSemantics()` check, meaning the input IR has completed tensor-to-memref bufferization.
2. **Must run after flattening**: For reg-based architectures, `hfusion-flatten` must have completed axis merging; for non-reg-based architectures, flatten information is obtained at marking time through `FlattenInterface::getFlattened`.
3. **The enabling phase may increase memory usage**: Alignment padding increases the physical size of the alloc (the aligned dimension size becomes larger), which may affect subsequent memory allocation by PlanMemory. Users must ensure that the total memory requirement after alignment does not exceed the hardware space limit.
4. **Propagation convergence**: Propagation between operands iterates up to 10 times; if it does not converge, compilation fails. Reasonable IR structures typically do not trigger this limit.
5. **Known conflict between vsstb and FlattenOps**: The bank conflict alignment of vsstb may turn contiguous shapes into non-contiguous shapes, conflicting with `memref.collapse_shape`/`memref.reshape`, which rely on contiguous shape characteristics, and may cause subsequent FlattenOps compilation errors.
