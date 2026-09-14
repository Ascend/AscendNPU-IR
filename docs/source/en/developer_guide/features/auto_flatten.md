# Auto Flatten

The **Auto Flatten Pass** (**HIVMFlattenOps**) automatically folds multi-dimensional tensor operations into lower-dimensional equivalent forms, reducing the rank while preserving semantic correctness. This optimization simplifies memory access patterns and improves hardware utilization on the target accelerator.

## Hardware Background

Modern hardware accelerators typically have constraints and performance characteristics that favor low-rank tensor operations:

| Item | Impact of High Rank | Benefits of Flattening |
| ----------------------- | ------------------------------------------------------------ | -------------------------------------------------------- |
| Address computation | Multi-dimensional indexing requires multiple multiply-add operations | Simplified linear addressing reduces overhead |
| Memory coalescing | Complex stride patterns may hinder efficient memory access | Contiguous flattened dimensions enable better coalescing |
| Hardware loops | The number of hardware loop counters is limited | Fewer dimensions = fewer loop nests required |
| DMA efficiency | Multi-stride transfers may require multiple DMA descriptors | Collapsed dimensions enable bulk transfers |
| Register pressure | More index variables occupy registers | Reduces bookkeeping overhead |

**Example Scenario**:

Consider a 5D element-wise operation with shape `[1, 64, 1, 128, 256]`:

- Before flattening: 5 levels of nested loops with complex stride computation.
- After flattening: the shape becomes `[64, 128, 256]` or even `[64, 32768]`, achieving more efficient hardware utilization.

## Algorithm Principles

The flattening algorithm runs as a multi-stage pipeline, progressively collapsing dimensions while adhering to operation-specific constraints.

### Core Concepts

1. **Reassociation Maps**

   A reassociation map defines how the original dimensions map to the collapsed dimensions:

   ```text
   Original shape: [A, B, C, D, E] (rank 5)
   Reassociation:  [[0, 1], [2], [3, 4]]
   Result shape:   [A*B, C, D*E] (rank 3)
   ```

2. **Dimension Classification (Ternary Mask)**

   Each dimension is classified into one of the following three types:

   | Category       | Symbol | Description          | Collapse Behavior                   |
   | -------------- | ------ | -------------------- | ---------------------------------- |
   | Unit dimension | `U`    | A dimension of size 1 | Absorbed into an adjacent group    |
   | Collapsible dimension | `C` | Can merge with neighbors | Forms a group and absorbs adjacent unit dimensions |
   | Non-collapsible dimension | `N` | Barrier dimension | Stands alone and blocks the absorption of unit dimensions |

3. **Barrier Dimensions**

   Certain dimensions cannot be collapsed together due to semantic requirements:

   - Reduction dimensions: must remain independent to preserve reduction semantics
   - Broadcast dimensions: shape mismatch prevents collapsing
   - Transpose dimension: permutation requires constrained grouping

### Pipeline Stages

```text
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

For operations with inline transpose semantics, the algorithm processes the input and output reassociation mappings separately:

```text
Input shape:   [A, B, C, D, E, F]
Permutation:   [2, 3, 0, 4, 1, 5]
Output shape:  [C, D, A, E, B, F]

Step 1: Unit collapse on input (if B, D are unit)
Step 2: Derive permutation blocks from inverse permutation
Step 3: Generate separate input/init reassociation maps
Step 4: Compose results maintaining permutation semantics
```

### Mask Construction Logic

```cpp
for each dimension i:
    if (strictBarrierWithUnit && isBarrier[i]):
        mask[i] = NonCollapsible    // Strict mode: barrier dimensions are independent.
    else if (isUnit[i] && !isBarrier[i]):
        mask[i] = Unit              // Unit dimensions are absorbed.
    else:
        mask[i] = Collapsible       // Can form a group.
```

### Generating Reassociation from the Mask

```text
Input Mask: [U, C, U, N, U, C, U]

Processing:
  Segment 1: [U, C, U] → Group units with collapsible → [[0, 1, 2]]
  Segment 2: [N]       → Isolated non-collapsible    → [[3]]
  Segment 3: [U, C, U] → Group units with collapsible → [[4, 5, 6]]

Result: [[0, 1, 2], [3], [4, 5, 6]]
```

## API

### Pass Registration

```cpp
// Create the flatten pass.
std::unique_ptr<Pass> mlir::hivm::createFlattenOpsPass();

// Use it in the pass pipeline.
pm.addPass(mlir::hivm::createFlattenOpsPass());
```

### FlattenInterface

Operations that implement auto flatten must implement `FlattenInterface`:

```cpp
class FlattenInterface {
public:
  /// Compute the flatten result of this operation.
  virtual FailureOr<FlattenResult> getFlattened(FlattenOptions options) = 0;

  /// Adjust the operation attributes after flattening.
  virtual void adjustTargetDimensions(OpBuilder &builder,
                                       const FlattenResult &result) = 0;
};
```

### FlattenOptions

```cpp
struct FlattenOptions {
  /// When true, even a barrier of a unit dimension becomes NonCollapsible.
  bool strictBarrierWithUnit = false;

  /// Check the alignment requirement of the stride annotation.
  bool checkMarkStride = false;

  /// Verify input shape consistency before collapsing (for broadcasting).
  bool checkInputConsistency = false;
};
```

### FlattenResult

```cpp
struct FlattenResult {
  // Core data
  Operation *op;                               // Source operation
  SmallVector<ReassociationMap> reassociation; // Reassociation map
  SmallVector<KindTypePair> operandTypes;      // Type after collapsing
  SmallVector<Value> operandOriginalVal;       // Original operand value

  // Dimension tracking
  SmallVector<int64_t> originalTargetDims;     // Original target dimension index
  SmallVector<int64_t> adjustedTargetDims;     // Adjusted index after collapsing
  SmallVector<int64_t> barrierDims;            // Non-collapsible boundary dimension

  // Query method
  bool isIdentityCollapse() const;
  int getRankAfterFlatten() const;
  SmallVector<Type> getOperandTypes(DpsKind kind) const;
  ReassociationMap getInputReassociation() const;
  ReassociationMap getInitReassociation() const;
  bool uniformReassociation() const;  // Whether the reassociation of all inputs and init is the same.
};
```

### Operation Traits

```cpp
// Indicate that the operation uses the same reassociation for all operands.
OpTrait::UniformReassociationFlattenTrait

// Indicate that contiguous target dimensions can be merged.
OpTrait::CollapsibleConsecutiveTargetDimsTrait
```

### Supported Operation Adjustments

Each operation type implements `adjustTargetDimensions`:

| Operation | Adjusted Attribute |
| -------------------------- | --------------------------------------------- |
| `VBrcOp` | `broadcast_dims` |
| `VReduceOp` | `reduce_dims` |
| `VTransposeOp` | `permutation` |
| `VCumsumOp` / `VCumprodOp` | `cum_dims` |
| `VPadOp` | `static_low`, `static_high` |
| `VConcatOp` | `dim` |
| `VFlipOp` | `flip_axis` |
| Element-wise operations | `iterator_types` (broadcast/transpose arrays) |

## Capabilities and Limitations

**Supported Capabilities**:

| Feature | Description |
| ------------------------------- | ------------------------------------------------------------ |
| Unit dimension collapse | Automatically removes dimensions of size 1. |
| Contiguity awareness | Respects the memory layout; non-contiguous dimensions remain independent. |
| Operation-specific handling | Custom logic for reduction, broadcast, transpose, padding, and so on. |
| Pipeline composition | Multiple collapse stages can be combined correctly. |
| Uniform reassociation | Efficient handling when all operands are collapsed in the same way. |
| Non-uniform reassociation | Supports different input/init reassociation (transpose OTF). |
| Barrier protection | Semantically critical dimensions remain independent. |
| Skip Host functions | Automatically skips Host-side functions. |

**Limitations**:

| Limitation | Description | Workaround |
| ---------------------------- | --------------------------------------------------------- | ------------------------------------------------------- |
| MemRef types only | Only `MemRefType` operands are collapsed. | Tensors must be bufferized first. |
| Static shapes required | Dynamic dimensions may not be collapsed correctly. | Run the symbolic dialect or shape inference pass first. |
| Strict barrier mode | `VFlipOp` requires `strictBarrierWithUnit=true`. | Handled automatically. |
| Transpose backward dimension | The last dimension of certain operations cannot undergo OTF transpose. | The algorithm keeps the last dimension uncollapsed. |
| Non-HIVMStructuredOp | Operations that do not implement the interface return an identity mapping. | Implement `FlattenInterface`. |

**Edge Cases**:

```cpp
// Identity collapse (no change) -- the pass reports a match failure.
if (res->isIdentityCollapse())
  return rewriter.notifyMatchFailure(op, "Identity reassociation");

// The operation cannot be handled.
if (failed(res))
  return rewriter.notifyMatchFailure(op, "Operation cannot be handled");
```

**Debugging**:

Use the `LDBG` macro to enable debug logging, which can trace:

- Reassociation mapping of each stage
- Mask classification
- Adjusted target dimension
- Combined result

## Transformation Examples

**Before transformation**:

```mlir
%0 = hivm.vbrc %input broadcast_dims = [3]
     : memref<1x64x1x128x256xf32> -> memref<1x64x16x128x256xf32>
```

**After the Flatten Pass**:

```mlir
// Collapse the input: [[0, 1, 2], [3, 4]] → rank 2
%collapsed_input = memref.collapse_shape %input [[0, 1, 2], [3, 4]]
     : memref<1x64x1x128x256xf32> into memref<64x1x32768xf32>

// Adjust the dimension after broadcasting: [1, 3] → [0] (after remapping)
%0 = hivm.vbrc %collapsed_input broadcast_dims = [1]
     : memref<64x1x32768xf32> -> memref<64x16x32768xf32>

// Note: Output expansion is handled by a separate Pass.
```

## Broadcast Examples with Strides

**About MemRef types with strides**:

A memref of type `memref<N₀×N₁×…×Nₙ×f32, strides={S[0], S[1], …, S[n]}, offset=O>` maps the coordinates $[i_0, i_1, \dots, i_n]$ to a linear memory address:

$$\text{address} = \sum_{k=0}^{n} i_k \cdot S[k] \;+\; O$$

For example, `memref<5x6xf32>` has the default (identity) layout with strides `[6, 1]` and offset 0. Accessing element `[2, 4]` yields:

$$\text{address} = 2 \times 6 + 4 \times 1 + 0 = 16$$

When no explicit layout is specified, MLIR uses row-major order. Strides are computed from the innermost dimension to the outermost dimension:

$S[n] = 1$

$S[i] = S[i+1] \times N_{i+1}$

This means that elements along the last dimension are adjacent in memory, and each "row" of the next outer dimension immediately follows the previous row—with no gaps.

When a dimension has size 1, its index is always 0. The contribution of that dimension's stride to the address is $0 \times S[k] = 0$, making the stride value irrelevant. This is why the flattening pass can freely absorb unit dimensions into adjacent groups without considering their stride values.

Without loss of generality and assuming row-major order, two adjacent dimensions $d_i$ and $d_{i+1}$ are contiguous if and only if:

$$S[i] = S[i{+}1] \times N_{i+1}$$

This means that after traversing all elements of dimension $i{+}1$, incrementing dimension $i$ by 1 lands exactly on the next element—with no gaps and no overlap. The base case is that the outermost dimension (axis 0) is always considered contiguous by convention.

Only contiguous adjacent dimensions can be collapsed. collapsing non-contiguous dimensions would change the actual memory locations accessed.

The following scenarios demonstrate how the flattening pass interacts with strided memory layouts, which is a common situation when handling non-contiguous memory views. These examples use `hivm.hir.vbrc`, a scalar broadcast operation that fills a memref with a scalar value.

1. **Scenario 1: A non-contiguous stride prevents all collapsing**

   The corresponding function is `@strided_brc`

   ```mlir
   // memref<16x16xf32, strided<[16, 2]>>
   //   dim 0: size=16, stride=16
   //   dim 1: size=16, stride=2   ← non-contiguous (contiguity requires stride=1)
   ```

   Analysis: Dimensions 0 and 1 cannot be merged. To satisfy contiguity, the stride of dimension 1 must equal 1 (the element stride). Here the stride is 2, indicating a non-contiguous access pattern of "every other element". collapsing dimensions $[0, 1]$ into a single dimension would produce a flat index of $i \cdot 16 + j$, but the actual memory access pattern is $i \cdot 16 + j \cdot 2$. The two are not equivalent—flattening would silently change the actual memory locations accessed.

   Output (unchanged):

   ```mlir
   func.func @strided_brc(%arg0: f32, %arg1: memref<16x16xf32, strided<[16, 2]>>) {
     hivm.hir.vbrc ins(%arg0 : f32) outs(%arg1 : memref<16x16xf32, strided<[16, 2]>>)
     return
   }
   ```

2. **Scenario 2: Partially contiguous strides allow partial collapsing**

   The corresponding function is `@strided_brc_collapse_continuous`

   ```mlir
   // memref<8x?x4x2xf32, strided<[?, ?, 2, 1]>>
   //   dim 0: size=8,  stride=?   ← dynamic, cannot verify contiguity with dim 1
   //   dim 1: size=?,  stride=?   ← dynamic, cannot verify contiguity with dim 2
   //   dim 2: size=4,  stride=2   ← stride = dim3.size(2) × dim3.stride(1) = 2 ✓
   //   dim 3: size=2,  stride=1   ← innermost, contiguous
   ```

   Contiguity check formula for adjacent dimension pairs:

   $$\text{contiguous}(d_i, d_{i+1}) \iff \text{stride}(d_i) = \text{size}(d_{i+1}) \times \text{stride}(d_{i+1})$$

   | Dimension pair  | Calculation                 | Is contiguous     |
   | ------- | -------------------- | ------------ |
   | Dimension 0–1 | $? = ? \times ?$     | Unknown (dynamic) |
   | Dimension 1–2 | $? = 4 \times 2 = 8$ | Unknown (dynamic) |
   | Dimension 2–3 | $2 = 2 \times 1 = 2$ | Yes           |

   Output (dimensions 2 and 3 collapsed):

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

   Result after collapsing:

   - Merge dimensions 2 and 3: size is $4 \times 2 = 8$, stride $= 1$ (contiguous)
   - The rank is reduced from 4 to 3

3. **Scenario 3: A dynamic inner dimension prevents contiguity verification**

   The corresponding function is `@scalar_brc_cannot_collapse_continuous`

   ```mlir
   // memref<8x?x4x?xf32, strided<[?, ?, 2, 1]>>
   //   dim 0: size=8,  stride=?
   //   dim 1: size=?,  stride=?
   //   dim 2: size=4,  stride=2
   //   dim 3: size=?,  stride=1   ← dynamic size
   ```

   Contiguity check for dimensions 2–3:

   The compiler cannot statically prove $2 = ?$. If the runtime size of dimension 3 is 2, they are contiguous; if the size is 3, they are not. The pass conservatively refuses to collapse.

   | Dimension pair  | Computation                 | Is contiguous                |
   | ------- | -------------------- | ----------------------- |
   | Dimension 0–1 | $? = ? \times ?$     | Unknown                    |
   | Dimension 1–2 | $? = 4 \times 2 = 8$ | Unknown                    |
   | Dimension 2–3 | $2 = ? \times 1 = ?$ | Unknown (dimension 3 size is dynamic) |

   Output (unchanged):

   ```mlir
   func.func @scalar_brc_cannot_collapse_continuous(
       %arg0: f32, %arg1: memref<8x?x4x?xf32, strided<[?, ?, 2, 1]>>) {
     hivm.hir.vbrc ins(%arg0 : f32)
         outs(%arg1 : memref<8x?x4x?xf32, strided<[?, ?, 2, 1]>>)
     return
   }
   ```
