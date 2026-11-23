# Auto Flatten

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

The **Auto Flatten Pass** (**HIVMFlattenOps**) automatically folds multi-dimensional tensor operations into lower-dimensional equivalent forms, reducing the rank while preserving semantic correctness. This optimization simplifies memory access patterns and improves hardware utilization on the target accelerator.

## Hardware Background

Modern hardware accelerators typically have constraints and performance characteristics that favor low-rank tensor operations:

| Item | Impact of High Rank | Benefit of Flattening |
| ----------------------- | ------------------------------------------------------------ | -------------------------------------------------------- |
| Address computation | Multi-dimensional indexing requires multiple multiply-add operations | Simplified linear addressing reduces overhead |
| Memory coalescing | Complex stride patterns may hinder efficient memory access | Contiguous flattened dimensions achieve better coalescing |
| Hardware loops | The number of hardware loop counters is limited | Fewer dimensions = fewer loop nests required |
| DMA efficiency | Multi-stride transfers may require multiple DMA descriptors | Folded dimensions enable bulk transfers |
| Register pressure | More index variables occupy registers | Reduced bookkeeping overhead |

**Example Scenario**:

Consider a 5D element-wise operation with shape `[1, 64, 1, 128, 256]`:

- Before flattening: 5 levels of nested loops with complex stride computation.
- After flattening: the shape becomes `[64, 128, 256]` or even `[64, 32768]`, achieving more efficient hardware utilization.

## Algorithm Principles

The flattening algorithm runs as a multi-stage pipeline, progressively collapsing dimensions while adhering to operation-specific constraints.

### Core Concepts

1. **Reassociation Maps**

   A reassociation map defines how the original dimensions map to the folded dimensions:

   ```text
   Original shape: [A, B, C, D, E] (rank 5)
   Reassociation:   [[0, 1], [2], [3, 4]]
   Result shape: [A*B, C, D*E] (rank 3)
   ```

2. **Dimension Classification**

   Each dimension is marked by a unit mask to indicate whether it is a unit dimension (size of 1); in addition, restricted axes (such as reduction and broadcast dimensions) act as barriers that prevent adjacent dimensions from folding:

   | Category     | Condition                          | Folding behavior                          |
   | ------------ | ---------------------------------- | ----------------------------------------- |
   | Unit dimension | Size of 1, and not a restricted axis | Absorbed into an adjacent group           |
   | Restricted axis | Dimensions with semantic restrictions such as broadcast/reduction | Stands alone and blocks the absorption of unit dimensions |
   | Other dimension | Non-unit, non-restricted           | Merged into the same group when adjacent and contiguous |

3. **Barrier Dimensions**

   Certain dimensions cannot be collapsed together due to semantic requirements:

   - Reduction dimensions: must remain independent to preserve reduction semantics
   - Broadcast dimensions: shape mismatch prevents collapsing
   - Transpose dimensions: permutation requirements constrain grouping

### Pipeline Stages

```text
┌─────────────────────────────────────────────────────────────────┐
│                            Input Operation                      │
│            Shape: [1, 64, 1, 128, 1, 256]                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Unit Dimension Collapse                               │
│  ─────────────────────────────────────────────                  │
│  • Identify unit (size-1) dimensions                            │
│  • Build a binary unit mask (excluding barriers)                │
│  • Collapse unit dimensions into adjacent non-barrier groups    │
│                                                                 │
│  Unit mask: [1, 0, 1, 0, 1, 0]   (1 = unit dimension)           │
│  Result:  [[0, 1, 2], [3, 4], [5]]  →  Shape: [64, 128, 256]    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: Uniform Reassociation Collapse                        │
│  ─────────────────────────────────────────────                  │
│  • Check memory contiguity (stride pattern)                     │
│  • Respect target dimension boundaries                          │
│  • Apply input consistency check (for broadcasting)             │
│                                                                 │
│  Contiguous dimensions can be further collapsed                 │
│  Result:  [[0], [1, 2]]  →  Shape: [64, 32768]                  │
│  (For example: input consistency boundary keeps dimension 0 independent)         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: Compose Results                                       │
│  ─────────────────────────────────────────────                  │
│  • Merge reassociation mappings from each stage                 │
│  • Adjust target dimension indices                              │
│  • Update barrier dimension tracking                            │
│                                                                 │
│  Final: [[0, 1, 2], [3, 4, 5]]  →  Shape: [64, 32768]           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                            Output Operation                     │
│  • Insert memref.collapse_shape for each operand                │
│  • Clone the operation and replace with the collapsed operands  │
│  • Adjust operation attributes (reduce_dims, broadcast_dims)    │
└─────────────────────────────────────────────────────────────────┘
```

### Special Case: Transposable OTF (On-The-Fly)

For operations with inline transpose semantics, the algorithm processes the input and output reassociation mappings separately:

```text
Input shape:   [A, B, C, D, E, F]
Permutation:       [2, 3, 0, 4, 1, 5]
Output shape:   [C, D, A, E, B, F]

Step 1: Collapse unit dimensions of the input (if B and D are unit dimensions)
Step 2: Derive permutation blocks from the inverse permutation
Step 3: Generate input/init reassociation mappings separately
Step 4: Compose the results while preserving permutation semantics
```

### Mask Construction Logic

The unit mask marks dimensions of size 1; restricted axes (obtained by `getLimitedAxes()`, such as broadcast and reduction dimensions) are then removed from the mask so that they form independent groups:

```cpp
// 1. Binary unit mask: true when an axis has size 1 across all operands
unitMask[i] = allOperandsDimSizeIs1(i);

// 2. Restricted axes as barriers: removed from the unit mask (bitwise AND after inversion)
limitationMask = arrayToMask(getLimitedAxes()); // Restricted axes are true
unitMask &= limitationMask.flip();              // Restricted axes are no longer unit dimensions
```

### Generate Reassociation from the Mask

```text
Input unit mask: [1, 1, 0, 0, 1, 0, 1]

Process:
  Leading unit dimensions  [0, 1]    -> merged with the first non-unit dimension
  Non-unit dimensions      [2, 3, 5] -> each starts a new group
  Trailing unit dimensions [4, 6]    -> absorbed into the previous group

Result: [[0, 1, 2], [3, 4], [5, 6]]
```

## API

### Pass Registration

```cpp
// Create a flatten pass.
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
  /// Check the alignment requirements of the stride annotation.
  bool checkMarkStride = false;

  /// Validate input shape consistency before collapsing (for broadcasting).
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

  // Query method.
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
// Indicates that the operation uses the same reassociation for all operands.
OpTrait::UniformReassociationFlattenTrait

// Indicates that contiguous target dimensions can be collapsed.
OpTrait::CollapsibleConsecutiveTargetDimsTrait
```

### Supported Operation Adjustments

Each operation type implements `adjustTargetDimensions`:

| Operation | Adjusted Attribute |
| -------------------------- | --------------------------------------------- |
| `VBrcOp` | `broadcast_dims` |
| `VReduceOp` | `reduce_dims` |
| `VTransposeOp` | `permutation` |
| `VCumsumOp` / `VCumprodOp` / `VCummaxOp` / `VCumminOp` | `cum_dims` |
| `VPadOp` | `static_low`, `static_high` |
| `VConcatOp` | `dim` |
| `VFlipOp` | `flip_axis` |
| `VSortOp` | `sort_axis` |
| Element-wise operations | `iterator_types` (broadcast, transpose arrays) |

## Capabilities and Limitations

**Supported Capabilities**:

| Feature | Description |
| ------------------------------- | ------------------------------------------------------------ |
| Unit dimension collapse | Automatically removes dimensions of size 1 |
| Contiguity awareness | Respects memory layout; non-contiguous dimensions remain independent |
| Operation-specific handling | Custom logic for reduction, broadcast, transpose, padding, etc. |
| Pipeline composition | Multiple folding stages can be composed correctly |
| Uniform reassociation | Efficient handling when all operands are collapsed in the same way. |
| Non-uniform reassociation | Supports different input/init reassociation (transpose OTF) |
| Barrier protection | Semantically critical dimensions remain independent |
| Skip Host functions | Automatically skips Host-side functions |

**Limitations**:

| Limitation | Description | Workaround |
| ---------------------------- | --------------------------------------------------------- | ------------------------------------------------------- |
| MemRef types only | Only `MemRefType` operands are collapsed. | Tensors must be bufferized first |
| Static shape required | Dynamic dimensions may not fold correctly | Run symbolic dialect or shape inference passes first |
| Strict barrier mode | Barrier dimensions (restricted axes) always remain independent | Handled automatically via `getLimitedAxes()` |
| Transpose backward dimension | The last dimension of certain operations cannot undergo OTF transpose | The algorithm keeps the last dimension unfolded |
| Non-HIVMStructuredOp | Operations that do not implement the interface return an identity mapping | Implement `FlattenInterface` |

**Edge Cases**:

```cpp
// Identity collapse  (no change) -- Pass reports match failure
if (res->isIdentityCollapse())
  return rewriter.notifyMatchFailure(op, "Identity reassociation");

// Operation cannot be handled
if (failed(res))
  return rewriter.notifyMatchFailure(op, "Operation cannot be handled");
```

**Debugging**:

Use the `LDBG` macro to enable debug logging, which can trace:

- Reassociation mapping for each stage
- Unit mask and restricted axes
- Adjusted target dimensions
- Composed results

## Transformation Example

**Before transformation**:

```mlir
%0 = hivm.vbrc %input broadcast_dims = [2]
     : memref<1x64x1x128x256xf32> -> memref<1x64x16x128x256xf32>
```

**After the Flatten Pass**:

```mlir
// Collapse input: [[0, 1], [2], [3, 4]] → rank 3
// (Broadcast dim 2 is the restricted axis and must be preserved independently; unit dim 0 is absorbed into the first group)
%collapsed_input = memref.collapse_shape %input [[0, 1], [2], [3, 4]]
     : memref<1x64x1x128x256xf32> into memref<64x1x32768xf32>

// Broadcast dimension remapping: [2] → [1]
%0 = hivm.vbrc %collapsed_input broadcast_dims = [1]
     : memref<64x1x32768xf32> -> memref<64x16x32768xf32>

// Note: Output expansion is handled by a separate pass
```

## Broadcast Operation Example with Strides

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

1. **Scenario 1: Non-contiguous stride prevents all collapsing**

   The corresponding function is `@strided_brc`

   ```mlir
   // memref<16x16xf32, strided<[16, 2]>>
   //   dim 0: size=16, stride=16
   //   dim 1: size=16, stride=2   ← non-contiguous (contiguity requires stride=1)
   ```

   Analysis: Dimensions 0 and 1 cannot be merged. To satisfy contiguity, the stride of dimension 1 must equal 1 (the element stride). Here the stride is 2, indicating a non-contiguous access pattern of "every other element". Collapsing dimensions $[0, 1]$ into a single dimension would produce a flat index of $i \cdot 16 + j$, but the actual memory access pattern is $i \cdot 16 + j \cdot 2$. The two are not equivalent—flattening would silently change the actual memory locations accessed.

   Output (unchanged):

   ```mlir
   func.func @strided_brc(%arg0: f32, %arg1: memref<16x16xf32, strided<[16, 2]>>) {
     hivm.hir.vbrc ins(%arg0 : f32) outs(%arg1 : memref<16x16xf32, strided<[16, 2]>>)
     return
   }
   ```

2. **Scenario 2: Partially continuous strides allow partial collapsing**

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

   | Dimension Pair  | Calculation                 | Is Contiguous     |
   | --------------- | --------------------------- | -------------- |
   | dimension 0–1 | $? = ? \times ?$     | Unknown (dynamic) |
   | dimension 1–2 | $? = 4 \times 2 = 8$ | Unknown (dynamic) |
   | dimension 2–3 | $2 = 2 \times 1 = 2$ | Yes           |

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

   - Merge dimensions 2 and 3: size $4 \times 2 = 8$, stride $= 1$ (contiguous)
   - Rank decreases from 4 to 3

3. **Scenario 3: A dynamic inner dimension prevents contiguity verification**

   The corresponding function is `@scalar_brc_cannot_collapse_continuous`

   ```mlir
   // memref<8x?x4x?xf32, strided<[?, ?, 2, 1]>>
   //   dim 0: size=8,  stride=?
   //   dim 1: size=?,  stride=?
   //   dim 2: size=4,  stride=2
   //   dim 3: size=?,  stride=1   ← dynamic size
   ```

   Contiguity check for dimension 2–3:

   The compiler cannot statically prove $2 = ?$. If the runtime size of dimension 3 is 2, they are contiguous; if the size is 3, they are not. The pass conservatively refuses to collapse.

   | Dimension Pair  | Computation                 | Is Contiguous                |
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
