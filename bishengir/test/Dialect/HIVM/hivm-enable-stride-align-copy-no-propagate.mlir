// RUN: bishengir-opt -hivm-enable-stride-align -split-input-file %s | FileCheck %s

// Regression test for the contract: a `hivm.hir.copy ins(%src) outs(%dst)` does
// NOT propagate stride-align info between %src and %dst in either direction.
//
// Background: `processAlignPropagationAmongOperationOperands` used to treat copy
// like any other structured op — it unioned the align info of all UB operands
// (src + dst) and re-propagated the union back onto each operand. That meant a
// stride-align mark on one side of a copy "infected" the other side: a buffer
// that needed no alignment would still get padded (its alloc rewritten to a
// larger memref + subview) just because the copy peer needed it.
//
// The fix has two layers, both exercised here:
//   1. `populatePropagateAlignAmongOpOperandsPatterns` no longer registers
//      `PropagateAlignAmongOperationOperands<CopyOp>` — copy is excluded from the
//      operand-level align-propagation pattern set entirely.
//   2. `processAlignPropagationAmongOperationOperands` early-returns failure()
//      for `isa<hivm::CopyOp>(op)` as a defensive backstop.
//
// For each case below, only the side that carries a `stride_align` mark is
// rewritten (alloc grown to `2x16x1xf16` + subview `2x15xf16, strided<[16,1]>`);
// the unmarked peer stays a plain `memref<2x15xf16, #hivm.address_space<ub>>` —
// proving no align info crossed the copy in either direction.

// CHECK-LABEL: func.func @copy_src_aligned_dst_unaligned
func.func @copy_src_aligned_dst_unaligned() {
  // src carries dim0 align (32B); dst carries nothing.
  // Expect: src alloc rewritten to 2x16x1xf16 + subview; dst stays plain 2x15xf16.
  // CHECK: %[[SRC_ALLOC:.*]] = memref.alloc() : memref<2x16x1xf16, #hivm.address_space<ub>>
  // CHECK: %[[SRC_SV:.*]] = memref.subview %[[SRC_ALLOC]][0, 0, 0] [2, 15, 1] [1, 1, 1] : memref<2x16x1xf16, #hivm.address_space<ub>> to memref<2x15xf16, strided<[16, 1]>, #hivm.address_space<ub>>
  // CHECK: %[[DST_ALLOC:.*]] = memref.alloc() : memref<2x15xf16, #hivm.address_space<ub>>
  // CHECK: hivm.hir.copy ins(%[[SRC_SV]] : memref<2x15xf16, strided<[16, 1]>, #hivm.address_space<ub>>) outs(%[[DST_ALLOC]] : memref<2x15xf16, #hivm.address_space<ub>>)
  %src = memref.alloc() : memref<2x15xf16, #hivm.address_space<ub>>
  annotation.mark %src {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x15xf16, #hivm.address_space<ub>>
  %dst = memref.alloc() : memref<2x15xf16, #hivm.address_space<ub>>
  hivm.hir.copy ins(%src : memref<2x15xf16, #hivm.address_space<ub>>) outs(%dst : memref<2x15xf16, #hivm.address_space<ub>>)
  return
}

// -----

// CHECK-LABEL: func.func @copy_dst_aligned_src_unaligned
func.func @copy_dst_aligned_src_unaligned() {
  // dst carries dim0 align (32B); src carries nothing.
  // Expect: dst alloc rewritten to 2x16x1xf16 + subview; src stays plain 2x15xf16.
  // CHECK: %[[SRC_ALLOC:.*]] = memref.alloc() : memref<2x15xf16, #hivm.address_space<ub>>
  // CHECK: %[[DST_ALLOC:.*]] = memref.alloc() : memref<2x16x1xf16, #hivm.address_space<ub>>
  // CHECK: %[[DST_SV:.*]] = memref.subview %[[DST_ALLOC]][0, 0, 0] [2, 15, 1] [1, 1, 1] : memref<2x16x1xf16, #hivm.address_space<ub>> to memref<2x15xf16, strided<[16, 1]>, #hivm.address_space<ub>>
  // CHECK: hivm.hir.copy ins(%[[SRC_ALLOC]] : memref<2x15xf16, #hivm.address_space<ub>>) outs(%[[DST_SV]] : memref<2x15xf16, strided<[16, 1]>, #hivm.address_space<ub>>)
  %src = memref.alloc() : memref<2x15xf16, #hivm.address_space<ub>>
  %dst = memref.alloc() : memref<2x15xf16, #hivm.address_space<ub>>
  annotation.mark %dst {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x15xf16, #hivm.address_space<ub>>
  hivm.hir.copy ins(%src : memref<2x15xf16, #hivm.address_space<ub>>) outs(%dst : memref<2x15xf16, #hivm.address_space<ub>>)
  return
}

// -----

// CHECK-LABEL: func.func @copy_both_sides_aligned_distinct_dims
func.func @copy_both_sides_aligned_distinct_dims() {
  // Both sides carry the SAME dim0 align here — the key check is that each side
  // is rewritten independently from its OWN mark, not from a union propagated
  // across the copy. Since both marks are identical (dim0, 32B), a union-based
  // bug would produce the same shape, so this case mainly guards that having an
  // align mark on BOTH operands still yields one rewrite per side (two padded
  // allocs), never collapsing or duplicating.
  // CHECK: %[[A1:.*]] = memref.alloc() : memref<2x16x1xf16, #hivm.address_space<ub>>
  // CHECK: %[[SV1:.*]] = memref.subview %[[A1]]
  // CHECK: %[[A2:.*]] = memref.alloc() : memref<2x16x1xf16, #hivm.address_space<ub>>
  // CHECK: %[[SV2:.*]] = memref.subview %[[A2]]
  // CHECK: hivm.hir.copy ins(%[[SV1]] : memref<2x15xf16, strided<[16, 1]>, #hivm.address_space<ub>>) outs(%[[SV2]] : memref<2x15xf16, strided<[16, 1]>, #hivm.address_space<ub>>)
  %src = memref.alloc() : memref<2x15xf16, #hivm.address_space<ub>>
  annotation.mark %src {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x15xf16, #hivm.address_space<ub>>
  %dst = memref.alloc() : memref<2x15xf16, #hivm.address_space<ub>>
  annotation.mark %dst {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x15xf16, #hivm.address_space<ub>>
  hivm.hir.copy ins(%src : memref<2x15xf16, #hivm.address_space<ub>>) outs(%dst : memref<2x15xf16, #hivm.address_space<ub>>)
  return
}
