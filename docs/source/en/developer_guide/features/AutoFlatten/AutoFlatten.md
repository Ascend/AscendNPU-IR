# Auto Flatten

## Overview

The Auto Flatten pass (`HIVMFlattenOps`) automatically collapses multi-dimensional tensor operations into lower-dimensional equivalents, reducing rank while preserving semantic correctness. This optimization simplifies memory access patterns and improves hardware utilization on the target accelerator.

---

## Hardware Background

Modern hardware accelerators often have constraints and performance characteristics that favor lower-rank tensor operations:

| Aspect                  | Impact of High Rank                                          | Benefit of Flattening                                    |
| ----------------------- | ------------------------------------------------------------ | -------------------------------------------------------- |
| **Address Calculation** | Multi-dimensional indexing requires multiple multiply-add operations | Simplified linear addressing reduces overhead            |
| **Memory Coalescing**   | Complex stride patterns may prevent efficient memory access  | Contiguous flattened dimensions enable better coalescing |
| **Hardware Loops**      | Limited number of hardware loop counters                     | Fewer dimensions = fewer loop nests required             |
| **DMA Efficiency**      | Multi-strided transfers may require multiple DMA descriptors | Collapsed dimensions enable bulk transfers               |
| **Register Pressure**   | More index variables consume registers                       | Reduced bookkeeping overhead                             |

### Example Scenario

Consider a 5D elementwise operation on shape `[1, 64, 1, 128, 256]`:

- **Before**: 5 nested loops, complex stride calculations
- **After flattening**: Shape becomes `[64, 128, 256]` or even `[64, 32768]`, enabling more efficient hardware utilization

---

## Algorithm Principle

The flattening algorithm operates as a **multi-stage pipeline** that progressively collapses dimensions while respecting operation-specific constraints.

### Core Concepts

#### Reassociation Maps

A reassociation map defines how original dimensions map to collapsed dimensions:

```
Original shape: [A, B, C, D, E] (rank 5)
Reassociation:  [[0, 1], [2], [3, 4]]
Result shape:   [A*B, C, D*E] (rank 3)
```

#### Dimension Classification (Ternary Mask)

Each dimension is classified into one of three categories:

| Category           | Symbol | Description                  | Collapsing Behavior                  |
| ------------------ | ------ | ---------------------------- | ------------------------------------ |
| **Unit**           | `U`    | Size-1 dimension             | Absorbed into adjacent groups        |
| **Collapsible**    | `C`    | Can be merged with neighbors | Forms groups, absorbs adjacent units |
| **NonCollapsible** | `N`    | Barrier dimension            | Isolated, blocks unit absorption     |

#### Barrier Dimensions

Certain dimensions cannot be collapsed together due to semantic requirements:

- **Reduce dimensions**: Must remain separate to preserve reduction semantics
- **Broadcast dimensions**: Shape mismatches prevent collapsing
- **Transpose dimensions**: Permutation requirements constrain grouping

### Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input Operation                              │
│            Shape: [1, 64, 1, 128, 1, 256]                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Unit Dimension Collapse                               │
│  ─────────────────────────────────────────────                  │
│  • Identify unit (size-1) dimensions                            │
│  • Build ternary mask considering barriers                      │
│  • Collapse units into adjacent non-barrier groups              │
│                                                                 │
│  Mask:    [U,  C, U,   C, U,   C]                               │
│  Result:  [[0, 1, 2], [3, 4], [5]]  →  Shape: [64, 128, 256]    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: Uniform Reassociation Collapse                        │
│  ─────────────────────────────────────────────                  │
│  • Check memory contiguity (stride patterns)                    │
│  • Respect target dimension boundaries                          │
│  • Apply input consistency checks (for broadcast)               │
│                                                                 │
│  Contiguous dims can be further collapsed                       │
│  Result:  [[0], [1, 2]]  →  Shape: [64, 32768]                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: Compose Results                                       │
│  ─────────────────────────────────────────────                  │
│  • Combine reassociation maps from all stages                   │
│  • Adjust target dimension indices                              │
│  • Update barrier dimension tracking                            │
│                                                                 │
│  Final: [[0, 1, 2], [3, 4, 5]]  →  Shape: [64, 32768]           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Output Operation                             │
│  • Insert memref.collapse_shape for each operand                │
│  • Clone operation with collapsed operands                      │
│  • Adjust operation attributes (reduce_dims, broadcast_dims)    │
└─────────────────────────────────────────────────────────────────┘
```

### Special Case: Transposable OTF (On-The-Fly)

For operations with inline transpose semantics, the algorithm handles input and output reassociations **separately**:

```
Input shape:   [A, B, C, D, E, F]
Permutation:   [2, 3, 0, 4, 1, 5]
Output shape:  [C, D, A, E, B, F]

Step 1: Unit collapse on input (if B, D are unit)
Step 2: Derive permutation blocks from inverse permutation
Step 3: Generate separate input/init reassociation maps
Step 4: Compose results maintaining permutation semantics
```

### Mask Building Logic

```cpp
for each dimension i:
    if (strictBarrierWithUnit && isBarrier[i]):
        mask[i] = NonCollapsible    // Strict mode: barriers isolated
    else if (isUnit[i] && !isBarrier[i]):
        mask[i] = Unit              // Unit dims absorbed
    else:
        mask[i] = Collapsible       // Can form groups
```

### Reassociation Generation from Mask

```
Input Mask: [U, C, U, N, U, C, U]

Processing:
  Segment 1: [U, C, U] → Group units with collapsible → [[0, 1, 2]]
  Segment 2: [N]       → Isolated non-collapsible    → [[3]]
  Segment 3: [U, C, U] → Group units with collapsible → [[4, 5, 6]]

Result: [[0, 1, 2], [3], [4, 5, 6]]
```

---

## API

### Pass Registration

```cpp
// Create the flatten pass
std::unique_ptr<Pass> mlir::hivm::createFlattenOpsPass();

// In pass pipeline
pm.addPass(mlir::hivm::createFlattenOpsPass());
```

### FlattenInterface

Operations must implement `FlattenInterface` to participate in auto-flattening:

```cpp
class FlattenInterface {
public:
  /// Compute flattening result for this operation
  virtual FailureOr<FlattenResult> getFlattened(FlattenOptions options) = 0;
  
  /// Adjust operation attributes after flattening
  virtual void adjustTargetDimensions(OpBuilder &builder, 
                                       const FlattenResult &result) = 0;
};
```

### FlattenOptions

```cpp
struct FlattenOptions {
  /// When true, barrier dimensions become NonCollapsible even if unit
  bool strictBarrierWithUnit = false;
  
  /// Check stride annotations for alignment requirements  
  bool checkMarkStride = false;
  
  /// Verify input shapes are consistent before collapsing (for broadcast)
  bool checkInputConsistency = false;
};
```

### FlattenResult

```cpp
struct FlattenResult {
  // Core data
  Operation *op;                               // Source operation
  SmallVector<ReassociationMap> reassociation; // Collapse mappings
  SmallVector<KindTypePair> operandTypes;      // Types after collapse
  SmallVector<Value> operandOriginalVal;       // Original operand values
  
  // Dimension tracking
  SmallVector<int64_t> originalTargetDims;     // Original target dim indices
  SmallVector<int64_t> adjustedTargetDims;     // Adjusted after collapse
  SmallVector<int64_t> barrierDims;            // Non-collapsible boundaries
  
  // Query methods
  bool isIdentityCollapse() const;
  int getRankAfterFlatten() const;
  SmallVector<Type> getOperandTypes(DpsKind kind) const;
  ReassociationMap getInputReassociation() const;
  ReassociationMap getInitReassociation() const;
  bool uniformReassociation() const;  // Same reassoc for inputs and inits
};
```

### Operation Traits

```cpp
// Indicates operation uses same reassociation for all operands
OpTrait::UniformReassociationFlattenTrait

// Indicates consecutive target dimensions can be collapsed
OpTrait::CollapsibleConsecutiveTargetDimsTrait
```

### Supported Operation Adjustments

Each operation type implements `adjustTargetDimensions`:

| Operation                  | Adjusted Attribute                            |
| -------------------------- | --------------------------------------------- |
| `VBrcOp`                   | `broadcast_dims`                              |
| `VReduceOp`                | `reduce_dims`                                 |
| `VTransposeOp`             | `permutation`                                 |
| `VCumsumOp` / `VCumprodOp` | `cum_dims`                                    |
| `VPadOp`                   | `static_low`, `static_high`                   |
| `VConcatOp`                | `dim`                                         |
| `VFlipOp`                  | `flip_axis`                                   |
| Elementwise                | `iterator_types` (broadcast/transpose arrays) |

---

## Capability & Limitation

### ✅ Capabilities

| Feature                         | Description                                                  |
| ------------------------------- | ------------------------------------------------------------ |
| **Unit Dimension Collapse**     | Automatically removes size-1 dimensions                      |
| **Contiguity-Aware**            | Respects memory layout; non-contiguous dims stay separate    |
| **Operation-Specific Handling** | Custom logic for reduce, broadcast, transpose, pad, etc.     |
| **Pipeline Composition**        | Multiple collapse stages compose correctly                   |
| **Uniform Reassociation**       | Efficient handling when all operands collapse identically    |
| **Non-Uniform Reassociation**   | Supports different input/init reassociations (transpose OTF) |
| **Barrier Preservation**        | Semantic-critical dimensions remain isolated                 |
| **Host Function Skipping**      | Automatically skips host-side functions                      |

### ⚠️ Limitations

| Limitation                   | Description                                               | Workaround                                              |
| ---------------------------- | --------------------------------------------------------- | ------------------------------------------------------- |
| **MemRef Types Only**        | Only `MemRefType` operands are collapsed                  | Tensors must be bufferized first                        |
| **Static Shapes Required**   | Dynamic dimensions may not collapse correctly             | Consider symbol dialect or shape inference passes first |
| **Strict Barrier Mode**      | `VFlipOp` requires `strictBarrierWithUnit=true`           | Handled automatically                                   |
| **Transpose Back Dimension** | Last dimension cannot be OTF transposed for certain ops   | Algorithm leaves last dim uncollapsed                   |
| **Non-HIVMStructuredOp**     | Operations not implementing the interface return identity | Implement `FlattenInterface`                            |

### Edge Cases

```cpp
// Identity collapse (no change) - pass reports match failure
if (res->isIdentityCollapse())
  return rewriter.notifyMatchFailure(op, "Identity reassociation");

// Operation cannot be handled
if (failed(res))
  return rewriter.notifyMatchFailure(op, "Operation cannot be handled");
```

### Debugging

Enable debug logging with the `LDBG` macro to trace:

- Reassociation maps at each stage
- Mask classifications
- Adjusted target dimensions
- Composition results

---

## Example Transformation

### Before

```mlir
%0 = hivm.vbrc %input broadcast_dims = [3] 
     : memref<1x64x1x128x256xf32> -> memref<1x64x16x128x256xf32>
```

### After Flatten Pass

```mlir
// Collapse input: [[0, 1, 2], [3, 4]] → rank 2
%collapsed_input = memref.collapse_shape %input [[0, 1, 2], [3, 4]] 
     : memref<1x64x1x128x256xf32> into memref<64x1x32768xf32>

// Broadcast with adjusted dims: [1, 3] → [0] (after remapping)
%0 = hivm.vbrc %collapsed_input broadcast_dims = [1]
     : memref<64x1x32768xf32> -> memref<64x16x32768xf32>

// Note: Output expansion would be handled by separate pass
```

## Additional Example Scenarios: Strided Broadcast Operations

### About Strided Memref Type

A memref with type `memref<N₀×N₁×…×Nₙ×f32, strides={S[0], S[1], …, S[n]}, offset=O>` maps a coordinate $[i_0, i_1, \dots, i_n]$ to a linear memory address:

$$\text{address} = \sum_{k=0}^{n} i_k \cdot S[k] \;+\; O$$

For example, `memref<5x6xf32>` has the default (identity) layout with strides `[6, 1]` and offset `0`. Accessing element `[2, 4]` gives:

$$\text{address} = 2 \times 6 + 4 \times 1 + 0 = 16$$

When no explicit layout is specified, MLIR uses **row-major** ordering. Strides are computed from innermost to outermost:

$S[n] = 1$

$S[i] = S[i+1] \times N_{i+1}$

This means elements along the last dimension are adjacent in memory, and each "row" of the next-outer dimension follows immediately after the previous one — no gaps.

When a dimension has size 1, its index is always 0. The stride for that dimension contributes $0 \times S[k] = 0$ to the address, making the stride value **irrelevant**. This is why the flatten pass can freely absorb unit dimensions into adjacent groups regardless of their stride values.

Without loss of generality, and under the assumption that row-major ordering is used. Two adjacent dimensions $d_i$ and $d_{i+1}$ are **contiguous** if and only if:

$$S[i] = S[i{+}1] \times N_{i+1}$$

This means that stepping through all elements of dimension $i{+}1$ and then incrementing dimension $i$ by one lands exactly at the next element — no gaps, no overlaps. The base case is that the outermost dimension (axis 0) is always considered contiguous by convention.

**Only contiguous adjacent dimensions can be collapsed.** Collapsing non-contiguous dimensions would change which memory locations are accessed.

The following scenarios demonstrate how the flatten pass interacts with **strided memory layouts**, a common situation when working with non-contiguous memory views. These examples use `hivm.hir.vbrc` — a scalar broadcast operation that fills a memref with a scalar value.

---

### Scenario 1 Example: Non-Contiguous Strides Block All Collapsing

**Function:** `@strided_brc`

```mlir
// memref<16x16xf32, strided<[16, 2]>>
//   dim 0: size=16, stride=16
//   dim 1: size=16, stride=2   ← NOT contiguous (would need stride=1)
```

**Analysis**:

Cannot merge dims 0 and 1: for contiguity, dim 1's stride must equal 1 (the element stride). Here stride = 2, indicating a non-contiguous "every-other-element" access pattern. Collapsing dimensions $[0, 1]$ into a single dimension would produce a flat index $i \cdot 16 + j$, but the actual memory access pattern is $i \cdot 16 + j \cdot 2$. These are not equivalent — flattening would silently change which memory locations are accessed.

**Output (unchanged)**:

```mlir
func.func @strided_brc(%arg0: f32, %arg1: memref<16x16xf32, strided<[16, 2]>>) {
  hivm.hir.vbrc ins(%arg0 : f32) outs(%arg1 : memref<16x16xf32, strided<[16, 2]>>)
  return
}
```

---

### Scenario 2 Example: Partially Contiguous Strides Allow Partial Collapsing

**Function:** `@strided_brc_collapse_continuous`

```mlir
// memref<8x?x4x2xf32, strided<[?, ?, 2, 1]>>
//   dim 0: size=8,  stride=?   ← dynamic, cannot verify contiguity with dim 1
//   dim 1: size=?,  stride=?   ← dynamic, cannot verify contiguity with dim 2
//   dim 2: size=4,  stride=2   ← stride = dim3.size(2) × dim3.stride(1) = 2 ✓
//   dim 3: size=2,  stride=1   ← innermost, contiguous
```

**Contiguity check for adjacent dimension pairs**:

$$\text{contiguous}(d_i, d_{i+1}) \iff \text{stride}(d_i) = \text{size}(d_{i+1}) \times \text{stride}(d_{i+1})$$

| Pair     | Calculation          | Contiguous?         |
| -------- | -------------------- | ------------------- |
| dims 0–1 | $? = ? \times ?$     | ❌ Unknown (dynamic) |
| dims 1–2 | $? = 4 \times 2 = 8$ | ❌ Unknown (dynamic) |
| dims 2–3 | $2 = 2 \times 1 = 2$ | ✅ Yes               |

**Output (dims 2 and 3 collapsed)**:

```mlir
func.func @strided_brc_collapse_continuous(
    %arg0: f32, %arg1: memref<8x?x4x2xf32, strided<[?, ?, 2, 1]>>) {
  %collapse_shape = memref.collapse_shape %arg1 [[0], [1], [2, 3]]
      : memref<8x?x4x2xf32, strided<[?, ?, 2, 1]>>
        into memref<8x?x8xf32, strided<[?, ?, 1]>>
  hivm.hir.vbrc ins(%arg0 : f32)
      outs(%collapse_shape : memref<8x?x8xf32, strided<[?, ?, 1]>>)
  return
}
```

The collapsed result has:

- Dimensions 2 and 3 merged: size $4 \times 2 = 8$, stride $= 1$ (contiguous)
- Rank reduced from 4 to 3

---

### Scenario 3 Example: Dynamic Inner Dimension Prevents Contiguity Verification

**Function:** `@scalar_brc_cannot_collapse_continuous`

```mlir
// memref<8x?x4x?xf32, strided<[?, ?, 2, 1]>>
//   dim 0: size=8,  stride=?
//   dim 1: size=?,  stride=?
//   dim 2: size=4,  stride=2
//   dim 3: size=?,  stride=1   ← dynamic size!
```

**Contiguity check for dims 2–3**:

The compiler **cannot statically prove** that $2 = ?$. If dim 3 has runtime size 2, they would be contiguous; if dim 3 has size 3, they would not. The pass conservatively refuses to collapse.

| Pair     | Calculation          | Contiguous?                           |
| -------- | -------------------- | ------------------------------------- |
| dims 0–1 | $? = ? \times ?$     | ❌ Unknown                             |
| dims 1–2 | $? = 4 \times 2 = 8$ | ❌ Unknown                             |
| dims 2–3 | $2 = ? \times 1 = ?$ | ❌ **Unknown** (dim 3 size is dynamic) |

**Output (unchanged)**:

```mlir
func.func @scalar_brc_cannot_collapse_continuous(
    %arg0: f32, %arg1: memref<8x?x4x?xf32, strided<[?, ?, 2, 1]>>) {
  hivm.hir.vbrc ins(%arg0 : f32)
      outs(%arg1 : memref<8x?x4x?xf32, strided<[?, ?, 2, 1]>>)
  return
}
```
