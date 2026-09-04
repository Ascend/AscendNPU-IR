# Vectorization Generalization Tests

## Purpose

This suite compares the core vector semantics produced by two paths:

```text
HFusion/Linalg op -> hfusion-vectorize-ops
HIVM op          -> hivm-vectorize-ops
```

The paired `hfusion_*` and `hivm_*` functions are kept in the same
`// -----` group. The HFUSION RUN line adds `lower-vector-mask`; the HIVM RUN
line runs `canonicalize` and `cse` only, since hivm-vectorize-ops needs no
vector-mask lowering. Both sides normalize so the checks compare vector
semantics rather than incidental transfer-mask syntax.

The paths are considered equivalent when they preserve vector type, active
lanes, arithmetic semantics, predicate or signedness, reduction axes,
permutation, accumulator, and producer-consumer SSA flow. Different but
equivalent scaffolding is documented with `// Diff: <category>`.

## Coverage

| File | Specification |
| --- | --- |
| `test-vectorize-type.mlir` | Representative unary, binary, compare, select, cast, reduce, and transpose operations at one canonical shape. |
| `test-vectorize-shape.mlir` | Elementwise, broadcast, and reduce on 3-D tails and all-unit dimensions. |
| `test-vectorize-dtype.mlir` | i1, FP8, f16, and mixed-width operations on explicit targets. |
| `test-vectorize-predicate.mlir` | Floating ordered/unordered predicates, NaN-sensitive predicates, signed/unsigned integer predicates, and i1. |
| `test-vectorize-dynamic.mlir` | Accepted single dynamic axes and rejected multi-dynamic or over-capacity shapes. |
| `test-vectorize-broadcast.mlir` | Leading, trailing, multi-axis, shaped, and inline broadcast. |
| `test-vectorize-reduce.mlir` | Numeric and logical kinds, signedness, multiple axes, and non-zero accumulators. |
| `test-vectorize-padding.mlir` | Tail identities for sum, product, min/max, signed/unsigned extrema, XOR/OR/AND, and any/all. |
| `test-vectorize-transpose.mlir` | Identity, non-square 2-D tail, non-self-inverse 4-D cycle, unit dimensions, inverse transfer maps, and read/write masks. |
| `test-vectorize-broadcast-reduce.mlir` | Multi-axis broadcast-to-reduce and reduce-to-broadcast chains with direct SSA forwarding. |
| `test-vectorize-chain.mlir` | Elementwise and predicate producer-consumer chains after transfer elimination. |
| `test-vectorize-target.mlir` | Regbase, membase, and missing-target dtype behavior. |

Checks bind producer results to their consumers. Tail tests also bind masks or
padding constants to transfer reads, reduction accumulators to
`vector.multi_reduction`, and final vectors to transfer writes. Avoid replacing
these dependencies with unbound `{{.*}}` operands.

## Reduction kinds

`hfusion-auto-vectorize-v2` delegates vector emission to Linalg vectorization.
Linalg's `getCombinerOpKind` recognizes add, multiply, AND, OR, XOR,
signed/unsigned integer min/max, and IEEE `minimumf`/`maximumf`. The A5 build
also recognizes `minnumf`/`maxnumf`.

The HFusion-to-HIVM mapping recognizes these ordinary reductions without a
semantic loss:

- `sum`, `prod`, `max`, and `min`;
- `xori`, `ori`, and `andi` for integer elements;
- `any` and `all`, represented by OR and AND on i1 elements.

`hivm.hir.vreduce` and its downstream lowering already represent all of these
kinds. `VReduceOp::vectorize` must therefore emit the corresponding Vector
combining kinds: add, mul, signed/unsigned or floating extrema, xor, or, and.
Reductions with indices are a separate operation contract and are not covered
by this ordinary-reduction parity matrix.

The current HIVM enum has no distinct `minnumf` or `maxnumf` kind. Conversion
maps both to `min` or `max`, whose contract propagates NaNs. Preserving A5
`minnumf`/`maxnumf` semantics requires an IR-level extension and coordinated
updates to every HIVM lowering, not only `VReduceOp::vectorize`.

## Difference categories

Use only the following stable descriptions in `// Diff:` comments.

### `broadcast source rank mismatch`

Linalg inserts broadcast dimensions into the source rank. HIVM keeps a
same-rank source whose broadcast dimensions have size one. The output lanes
are equivalent.

### `broadcast scalar representation`

Linalg uses a rank-zero tensor transfer. HIVM uses a scalar followed by
`vector.broadcast`.

### `reduce shape mismatch`

Linalg removes reduced axes. HIVM retains them as unit dimensions and uses
`vector.shape_cast` around the same core reduction.

### `reduction tail representation`

Linalg masks `vector.multi_reduction` and may use zero transfer padding. HIVM
does not mask the reduction; it pads inactive source lanes with the exact
identity. These forms are equivalent only when every HIVM identity is correct.

### `transpose identity canonicalization`

An identity HIVM transpose folds before vectorization. The Linalg path may
retain an explicit transfer-read/write pair.

### `target capability mismatch`

This single category covers all target-related differences: a dtype supported
only by one target family, a dtype unsupported by every target, and inconsistent
fallback behavior when `hacc.target` is absent. The test comment states the
specific capability or fallback involved.

### `reduction NaN semantics mismatch`

In an A5 build, Linalg can vectorize `minnumf` and `maxnumf`, while the current
HFusion-to-HIVM conversion collapses them to NaN-propagating `min` and `max`.
No parity test should treat these operations as equivalent until HIVM gains
distinct reduction kinds.

## Validation

Run:

```text
build/bin/llvm-lit -sv bishengir/test/Dialect/HIVM/Vectorize/Generalization
```
