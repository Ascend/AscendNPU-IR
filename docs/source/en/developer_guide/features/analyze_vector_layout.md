# Vector Layout Analysis

This document introduces the vector layout analysis transformation (**AnalyzeVectorLayoutPass**) in HIVMAVE, covering the hardware background, algorithm principles, test cases, interface description, and usage constraints.

## Hardware Layout Planning

The Ascend Vector compute unit adopts a SIMD architecture, where a single instruction processes multiple elements in a vector register simultaneously. The physical capacity of a vector register is fixed, but the logical elements in the same register are not arranged in only one way. The element bit width, the stride between elements, and the data bit width actually used by an instruction all affect how data is organized in a register.

Taking a 256-byte vector register as an example, HIVMAVE uses `VecMemType` to describe the layout of vector data. The main layouts are as follows:

| Layout | Arrangement | Effective Occupancy Per Element | Number of Logical Elements Per Vector Register |
|-----|-----|-----|-----|
| `B8` | 8-bit elements packed tightly | 8 bit | 256 |
| `B8_2VL` | 8-bit elements arranged with a stride of 2 | 16 bit | 128 |
| `B8_4VL` | 8-bit elements arranged with a stride of 4 | 32 bit | 64 |
| `B16` | 16-bit elements packed tightly | 16 bit | 128 |
| `B16_2VL` | 16-bit elements arranged with a stride of 2 | 32 bit | 64 |
| `B32` | 32-bit elements packed tightly | 32 bit | 64 |

Although `B8_2VL`, `B8_4VL`, and `B16_2VL` still store logical elements of 8-bit or 16-bit respectively, each element occupies a wider effective slot. A strided layout can establish a lossless correspondence with the packed layout of wider elements:

```text
B8_2VL  <-> B16
B8_4VL  <-> B32
B16_2VL <-> B32
```

This correspondence is the basis of hardware type conversion instructions. For example, an 8-bit to 16-bit extension cannot directly convert arbitrarily packed `B8` data into `B16`; the source data must first be in the `B8_2VL` layout so that each 8-bit element is located in the corresponding 16-bit slot. Similarly, a 16-bit to 32-bit extension requires the source data to adopt the `B16_2VL` layout.

### Load/Store Layout

Vector Load/Store instructions are responsible not only for moving data but also for completing the data layout conversion between memory and vector registers. The Load/Store distribution modes in HIVMAVE correspond to the dist modes of hardware instructions:

- `NORM`: Data is loaded or stored tightly according to the element bit width.
- `UNPK`: During Load, the tightly packed data in memory is expanded into a strided layout in registers.
- `PK`: During Store, the strided layout in registers is compressed into tightly packed data in memory.
- `INTLV` and `DINTLV`: Interleaving and de-interleaving are completed during data movement or computation.
- `PB8`, `PB16`, and `PB32`: Predicate data is generated at 8-bit, 16-bit, or 32-bit granularity.

Therefore, the logical vector type `vector<Nxi8>` alone cannot determine which Load/Store instruction should be used. If the result layout is `B8`, Load typically uses `NORM`; if the result layout is `B8_2VL` or `B8_4VL`, the corresponding `PK` or `PK4` distribution form is required. Store also needs to select a matching mode based on the vector layout and the Mask layout.

### Type Conversion Layout

The type conversion instructions of the AVE hardware perform conversion by register slot and select channels through parameters such as `part`. Bit width conversion does not merely change the element type in the MLIR type; it also changes the effective arrangement of elements in the register:

```text
trunc direction (conversion from a higher bit width to a lower bit width):
B32      -- trunc 32->8  --> B8_4VL
B32      -- trunc 32->16 --> B16_2VL
B16      -- trunc 16->8  --> B8_2VL

ext direction (conversion from a lower bit width to a higher bit width):
B8_2VL  -- ext 8->16  --> B16
B8_4VL  -- ext 8->32  --> B32
B16_2VL -- ext 16->32 --> B32
```

If the layout of the source data does not satisfy the requirements of the conversion instruction, the compiler needs to insert `INTLV` or `DINTLV` before and after the conversion; otherwise, the conversion result may suffer from slot overlap or data misalignment. Therefore, the layout of a type conversion operation is a correctness constraint rather than mere performance optimization information.

### Mask layout

AVE uses predicate registers to store masks. The logical element type of a mask is usually `i1`, but its control granularity must match the effective bit width of the data instruction:

| Data layout | Actual Processing Granularity of the Instruction | Mask Layout |
|-----|-----|-----|
| `B8` | 8 bit | `B8` |
| `B16` | 16 bit | `B16` |
| `B32` | 32 bit | `B32` |
| `B8_2VL` | 16 bit | `B16` |
| `B8_4VL` | 32 bit | `B32` |
| `B16_2VL` | 32 bit | `B32` |

For example, although the element type of `B8_2VL` data is 8-bit, the hardware processes it in 16-bit slots, so the corresponding mask must use the `B16` granularity. If the mask is still interpreted at the `B8` granularity, the correspondence between predicate bits and data lanes will be incorrect.

## Related Terms

| Term | Meaning |
|-----|-----|
| VecMemType | The physical layout status of a vector value in Vector registers, defined in `HIVMAVEAttrs.td`. |
| VectorLayoutAttr | The `ave.vector_layout` attribute attached to a `VectorType`, which internally stores a `VecMemTypeAttr`. For example: `#ave.vector_layout<{mem = #ave.vec_mem_type<b16>}>`. |
| FunctionDistType | The function type or distribution type used by an operation, stored as the `functionType` attribute. It describes the hardware instruction variant or additional layout processing method that the operation needs to adopt. |
| Effective bit width | The bit width used by a hardware instruction when actually processing one element slot. For strided layouts, the effective bit width may be greater than the logical element bit width. For example, the effective bit width of `B8_2VL` is 16 bit. |
| Dense layout | A layout in which elements are arranged contiguously, including `B8`, `B16`, and `B32`. |
| Sparse layout | A layout in which there are gaps between elements, including `B8_2VL`, `B8_4VL`, and `B16_2VL`. |
| INTLV/DINTLV | Layout rearrangement operations. INTLV converts data into a sparser interleaved arrangement, and DINTLV converts data into a denser arrangement. |

## Algorithm Principles

### Software Background

Preceding transformations convert standard Vector/Arith operations into HIVMAVE operations. At this point, the `VectorType` in the IR typically records only the logical shape and element type, for example `vector<128xi8>`, without recording how the elements are arranged in the hardware vector registers.

Subsequent AVE normalization and instruction lowering require this information to accomplish the following tasks:

- Select the `NORM`, `UNPK`, or `UNPK4` mode of Load based on the layout of the vector result;
- Select the `NORM`, `PK`, or `PK4` mode based on the layout of the Store input;
- Select the correct predicate granularity for the Mask;
- Select the correct hardware conversion form and lanes for ext/trunc;
- Insert `INTLV` or `DINTLV` when the source layout and the target layout are inconsistent;
- Ensure that multiple consumers on the same data flow observe a consistent layout.

If the layout is selected independently for each operation, mismatches can easily arise between adjacent operations. For example, Load can produce any one of the `B8`, `B8_2VL`, or `B8_4VL` layouts, but the subsequent ext, compare, and Store operations impose different requirements on the input layout. The choice made by one operation affects all of its inputs and outputs, and ultimately the layout must be resolved uniformly across the entire function.

AnalyzeVectorLayout models layout derivation as a constraint satisfaction problem. It starts from the vector results at the end of the function, enumerates possible layouts, and then propagates constraints backward along the data flow. Each operation produces one or more legal candidates according to its own hardware semantics; candidates are continuously merged during propagation and removed when conflicts occur. The final selected solution simultaneously satisfies the constraints of data layout, Mask granularity, type conversion, and control flow boundaries.

### Solving Process

1. Initialize the solution space: collect vector results that have no users, and enumerate initial layouts based on element bit width;
2. Traverse operations in reverse order: process all operations that contain vector operands or vector results from the end of the function backward;
3. Resolve operation constraints: dispatch to the solution function corresponding to the operation type through `TypeSwitch`;
4. Merge candidate solutions: merge the input status produced by the current operation into the existing status, and discard candidates when conflicts are detected;
5. Apply the solution result: add layout attributes to vector result types, add the `functionType` attribute to operations, and rewrite special operations.

### Operation Solving Rules

#### Operation Dispatch

`solveProblem(Operation *)` uses `TypeSwitch` to dispatch solution functions based on the operation type. The main dispatch relationships are as follows:

| Operation Type | Solution Function | Main Processing Content |
|-----|-----|-----|
| `VFLoadOp` | `solveProblem(VFLoadOp)` | Selects the `NORM`, `PK`, `PK4`, or predicate Load form based on the result bit width and layout |
| `VFMaskedStoreOp` | `solveProblem(VFMaskedStoreOp)` | Constrains the data vector and Mask layout simultaneously |
| `VFStoreWithStrideOp` | `solveProblem(VFStoreWithStrideOp)` | Selects `NORM`, `DINTLV2`, or `DINTLV4` based on the stride layout |
| `VFPgeOp`, `VFPltOp` | `solvePregGenProblem` | Selects `PB8`, `PB16`, or `PB32` based on the predicate result status |
| Element-wise vector operations | `solveMaskProblem` | Constrains the data operands, result, and Mask layout |
| `VFCmpOp` | `solveProblem(VFCmpOp)` | Constrains the left and right inputs based on the comparison data bit width and Mask result status |
| `VFTruncFOp`, `VFTruncIOp` | `solveTruncProblem` | Handles layout conversion from a higher bit width to a lower bit width |
| `VFExtFOp`, `VFExtUIOp`, `VFExtSIOp` | `solveExtProblem` | Handles layout conversion from a lower bit width to a higher bit width |
| Other type conversion operations | `solveTypeConvertProblem` | Forwards to ext, trunc, or Mask solution based on the source and result bit width relationship |
| `VFGatherOp` | `solveProblem(VFGatherOp)` | Constrains the index vector and Mask layout |
| `VFInterleaveOp`, `VFDeInterleaveOp` | `solveLayoutChangeProblem` | Handles layout changes and the interleaving direction |
| `scf.for` | `solveProblem(scf::ForOp)` | Constrains the loop iteration parameters, initial values, and loop results |
| `scf.yield` | `solveProblem(scf::YieldOp)` | Propagates the loop result status to the yield operands |
| `func.call` | `solveProblem(func::CallOp)` | Handles the vector status combination at the call boundary |
| Other operations with vector results | `solveProblemDefault` | Assumes by default that the vector operands and the first vector result have the same status |

Operations that are not explicitly handled in `TypeSwitch` enter the default branch. The default branch requires the operation to contain at least one vector result; otherwise, constraints cannot be established. If multiple vector operands exist, the status of the first vector result is propagated to all vector operands.

#### Load, Store, and Mask Solution

For `VFLoadOp`, the result layout directly determines the Load function type:

| Element Bit Width | Result Layout | Function Type |
|-----|-----|-----|
| 1 bit | `B8`, `B16`, `B32` | `PB8`, `PB16`, `PB32` |
| 8 bit | `B8`, `B8_2VL`, `B8_4VL` | `NORM`, `PK`, `PK4` |
| 16 bit | `B16`, `B16_2VL` | `NORM`, `PK` |
| 32 bit | `B32` | `NORM` |

`VFMaskedStoreOp` must constrain the Mask in addition to the data vector:

| Data Element Bit Width | Data Layout | Mask Layout | Function Type |
|-----|-----|-----|-----|
| 1 bit | `B8`, `B16`, `B32` | Same as the data layout | `PB8`, `PB16`, `PB32` |
| 8 bit | `B8` | `B8` | `NORM` |
| 8 bit | `B8_2VL` | `B16` | `PK` |
| 8 bit | `B8_4VL` | `B32` | `PK4` |
| 16 bit | `B16` | `B16` | `NORM` |
| 16 bit | `B16_2VL` | `B32` | `PK` |
| 32 bit | `B32` | `B32` | `NORM` |

For element-wise Mask operations, non-Mask vector operands are generally required to use the same layout as the result, while the Mask operand selects its status according to the effective bit width corresponding to the result layout. `VFCmpOp` differs: its result is itself `vector<...xi1>`, so the layouts of the left and right inputs are jointly determined by the input element bit width and the result Mask status. For example, an 8-bit input requires `B8_2VL` when the result is `B16`, and `B8_4VL` when the result is `B32`.

#### Type Conversion Solution

`solveTruncProblem()` searches for a valid combination based on the source element bit width and the result status. The `EVEN` function type is typically used, indicating that valid channels are extracted from the corresponding wide slots:

| Source Element Bit Width | Source Layout | Result Layout | Mask Layout | Function Type |
|-----|-----|-----|-----|-----|
| 32 bit | `B32` | `B16_2VL` | `B32` | `EVEN` |
| 32 bit | `B32` | `B8_4VL` | `B32` | `EVEN` |
| 16 bit | `B16` | `B8_2VL` | `B16` | `EVEN` |
| 16 bit | `B16_2VL` | `B8_4VL` | `B32` | `EVEN` |

When a Trunc operation carries the `Layout_Change` attribute, it indicates that the operation originates from a specific VSSTB optimization path. In this case, the solver uses the `C2C` function type to handle column-to-column conversion instead of the ordinary `EVEN` path.

`solveExtProblem()` handles bit width extension in the opposite direction:

| Source Element Bit Width | Source Layout | Result Layout | Mask Layout | Function Type |
|-----|-----|-----|-----|-----|
| 8 bit | `B8_2VL` | `B16` | `B16` | `EVEN` |
| 8 bit | `B8_4VL` | `B16_2VL` | `B32` | `EVEN` |
| 8 bit | `B8_4VL` | `B32` | `B32` | `EVEN` |
| 16 bit | `B16_2VL` | `B32` | `B32` | `EVEN` |
| 32 bit | `B32` | `B32` | `B32` | `EVEN` |

Other integer and floating-point type conversions are handled by `solveTypeConvertProblem()` according to the bit width relationship: when the source bit width is greater than the result bit width, the problem is resolved as Trunc; when the source bit width is smaller than the result bit width, it is resolved as Ext; and when the bit widths are equal, it is handled as a Mask operation.

#### Interleave and DeInterleave Solution

`solveLayoutChangeProblem()` handles `VFInterleaveOp` and `VFDeInterleaveOp`. It first requires that the statuses of the two results be consistent, and then determines the constraint direction based on the `layout_change` attribute:

| `layout_change` | Meaning | Typical Constraint |
|-----|-----|-----|
| No attribute | Original Interleave/DeInterleave operation | The source state and the result state are the same, or `INTLV2/4` is selected |
| `UNCHANGED` | The layout does not change | Delegated to the default solver, where the source state equals the result state |
| `DENSE` | Changes from a sparse layout to a dense layout | When the result is `B16`, the source is `B16_2VL` |
| `SPARSE` | Changes from a dense layout to a sparse layout | When the result is `B16_2VL`, the source is `B16` |

The typical mappings are as follows:

```text
DENSE:
B16_2VL -> B16
B8_4VL  -> B8_2VL
B8_2VL  -> B8

SPARSE:
B16 -> B16_2VL
B8  -> B8_2VL
B8_2VL -> B8_4VL
```

In the original Interleave operation without an attribute, if the 8-bit result is `B8_2VL` or `B8_4VL`, `INTLV2` or `INTLV4` is selected respectively; the DeInterleave operation correspondingly selects `DINTLV2` or `DINTLV4`. Here, `DENSE` and `SPARSE` describe the direction of change relative to the input, rather than the absolute name of the result layout.

#### Loop and Function Call Solution

For `scf.for`, the loop region iteration parameters and the loop result represent the same data passed across iterations and must maintain the same layout:

1. If both the iterArg and the forOp result already have a status, check whether the two are consistent;
2. If only the iterArg status is known, propagate that status to the initArg;
3. If only the forOp result status is known, propagate the result status to the initArg;
4. `scf.yield` propagates the forOp result status to the corresponding yield operand.

When applying the solution, the Pass rebuilds `scf.for` and merges the original loop body into the new loop body so that the iteration parameters and result types with layouts can be used.

For `func.call`, if the call result contains vectors, the solver uses the default rule to propagate the state; if the call has no vector result but contains vector parameters, it enumerates all possible state combinations for each vector parameter. When applying the solution, the Pass inserts a `VectorLayoutCastOp` without a layout on the call input side and supplements the default layout on the vector result side according to the element bit width, thereby avoiding inconsistency of type attributes at the call operation boundary.

## Compilation Options

| Option | Description |
|-----|-----|
| `-analyze-vector-layout` | Performs vector layout analysis at the `func::FuncOp` granularity. |
| `-remove-vector-layout-attr` | Removes layout attributes and `VectorLayoutCastOp`; not an option of AnalyzeVectorLayout itself. |

AnalyzeVectorLayout has no additional parameters. During debugging, you can view the operation solution and the number of candidate solutions through LLVM debug output:

```bash
build/bin/bishengir-opt -analyze-vector-layout \
  -debug-only=analyze-vector-layout \
  --mlir-disable-threading=true input.mlir
```

## Usage Constraints

### Prerequisites

Users must ensure that, before running AnalyzeVectorLayout, the relevant standard Vector and Arith operations have completed HIVMAVE lowering, and that the vector operations in the input function belong to the operation types and layout combinations currently supported by the solver.

### Known Limitations

The current implementation mainly supports 1, 8, 16, and 32 bit elements. 64 bit elements are treated as 32 bit in some status judgments, but this does not mean that all 64 bit vector operation combinations are fully supported.

Multiple consumers of the same vector value must be able to satisfy the layout constraints simultaneously. If different consumers require conflicting `VecMemType` values and there is no valid layout conversion operation in the data flow, the candidate solutions are removed. The loop iteration parameters and loop results must also be able to maintain a consistent layout; otherwise, the candidate solutions corresponding to `scf.for` are invalid.

When an operation has not has not undergone HIVMAVE lowering, the operation type has not been added to the `TypeSwitch` of `solveProblem`, the current operation does not support the `VecMemType` combination, there are conflicts among multiple consumers, or the input element bit width is unsupported, the Pass calls `signalPassFailure()` and outputs `No Solve`.

### Error Diagnosis

Failure diagnosis outputs the following information:

- The location of the failed operation and the complete operation;
- The opcode, operand types, and result type;
- The number of candidate solutions remaining before the failure;
- The input status already recorded in the failed candidates;
- Possible causes and suggested handling methods.

A typical error message is as follows:

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
  5. Unsupported element bit width (only 1/8/16/32 are supported).
======================================================
```

If a new HIVMAVE operation or a new layout combination needs to be supported, positive constraints should be added in the corresponding `solveProblem()` function to explicitly define the status relationships among operands, results, and Mask, and both success and failure tests should be supplemented. Layout problems must not be circumvented by skipping the operation, relaxing constraints by default, or rewriting an analysis failure as a pattern matching failure.
