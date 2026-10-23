// RUN: bishengir-opt --hivm-canonicalization-pipeline %s | FileCheck %s
// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_957b --hivm-canonicalization-pipeline %s | FileCheck %s

// Regression tests for the HIVM canonicalization cluster
// (canonicalizationHIVMPipeline / hivm-canonicalization-pipeline): the
// CSE-created redexes must still be folded, and hivm-opt-single-point must
// still see its normalized input form.
//
// Checks assert observable results only (which ops survive and the resulting
// memory accesses), never the pass list.

// ---------------------------------------------------------------------------
// A redundant self-copy. CSE merges the two identical tensor.empty ops, which
// turns hivm.hir.copy into a self-copy (same value in and out); the cluster must
// fold it away and forward the operand.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @self_copy_folded
// CHECK-NOT: hivm.hir.copy
// CHECK: %[[E:.*]] = tensor.empty() : tensor<16x16xf32>
// CHECK: return %[[E]] : tensor<16x16xf32>
func.func @self_copy_folded() -> tensor<16x16xf32> {
  %src = tensor.empty() : tensor<16x16xf32>
  %dst = tensor.empty() : tensor<16x16xf32>
  %res = hivm.hir.copy ins(%src : tensor<16x16xf32>) outs(%dst : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %res : tensor<16x16xf32>
}

// -----

// ---------------------------------------------------------------------------
// Negative control: a copy with distinct source and destination is NOT
// redundant and must survive. Without this the first case could pass by
// deleting every copy.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @needed_copy_kept
// CHECK: hivm.hir.copy
func.func @needed_copy_kept(%src: tensor<16x16xf32>, %dst: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %res = hivm.hir.copy ins(%src : tensor<16x16xf32>) outs(%dst : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %res : tensor<16x16xf32>
}

// -----

// ---------------------------------------------------------------------------
// A chain of rank-reducing memref.collapse_shape ops must be collapsed into a
// single one.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @collapse_chain_collapsed
// CHECK: memref.collapse_shape %{{.*}} {{\[\[0, 1, 2, 3\]\]}} : memref<1x2x16x32xf8E4M3FN> into memref<1024xf8E4M3FN>
// CHECK-NOT: memref.collapse_shape %{{.*}} {{\[\[0\]\]}} : memref<1024xf8E4M3FN>
// CHECK: return
func.func @collapse_chain_collapsed(%a: memref<1x2x16x32xf8E4M3FN>) -> memref<1024xf8E4M3FN> {
  %r0 = memref.collapse_shape %a [[0, 1, 2, 3]] : memref<1x2x16x32xf8E4M3FN> into memref<1024xf8E4M3FN>
  %r1 = memref.collapse_shape %r0 [[0]] : memref<1024xf8E4M3FN> into memref<1024xf8E4M3FN>
  return %r1 : memref<1024xf8E4M3FN>
}

// -----

// ---------------------------------------------------------------------------
// hivm-opt-single-point must still see the normalized
// form: a 1x1 hivm op is fully scalarized into load/addf/store.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @single_point_after_canon
// CHECK-NOT: hivm.hir.vadd
// CHECK: memref.load
// CHECK-NEXT: memref.load
// CHECK-NEXT: arith.addf
// CHECK-NEXT: memref.store
func.func @single_point_after_canon(%a: memref<1x1xf32>, %b: memref<1x1xf32>, %c: memref<1x1xf32>) {
  hivm.hir.vadd ins(%a, %b : memref<1x1xf32>, memref<1x1xf32>) outs(%c : memref<1x1xf32>)
  return
}
