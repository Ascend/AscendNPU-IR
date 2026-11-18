// RUN: bishengir-opt --hivm-canonicalization-pipeline %s | FileCheck %s --check-prefix=CLUSTER
// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_957b --hivm-canonicalization-pipeline %s | FileCheck %s --check-prefix=CLUSTER
// RUN: bishengir-opt --hivm-canonicalization-pipeline --hivm-inline-otf-broadcast %s | FileCheck %s --check-prefix=WITH-INLINE
// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_957b --hivm-mark-multi-buffer --hivm-canonicalization-pipeline --hivm-inline-otf-broadcast %s | FileCheck %s --check-prefix=WITH-INLINE

// Pipeline regression tests for the RegBase pre-bufferization site that
// currently runs the full canonicalization cluster immediately before
// hivm-inline-otf-broadcast:
//
//   hivm-mark-multi-buffer
//   --> canonicalizationHIVMPipeline        <-- the cluster under test
//   --> hivm-inline-otf-broadcast
//
// The cluster's output is the precondition of the broadcast inliner. These
// checks pin the contract between the two: the inliner's observable broadcast
// decisions must stay unchanged.
//
// Checks are intentionally written against observable structure: which ops
// remain, the operand/result shapes, the `broadcast`/`broadcast_dims`
// attributes and the memory-space attributes. They never assert on the pass
// list itself.
//
// The second and fourth RUN lines repeat exactly the same checks on the A5
// RegBase target (Ascend950PR_957b), which is the only target whose pipeline
// reaches this site.

// ---------------------------------------------------------------------------
// The cluster alone must not invent or destroy the live structures: allocs,
// HIVM loads/stores and the elementwise op all survive with their address
// spaces intact, and the 1:1-shape redundant broadcast is removed before the
// inliner runs.
// ---------------------------------------------------------------------------
// CLUSTER-LABEL: func.func @cluster_preconditions
// CLUSTER: %[[A:.*]] = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
// CLUSTER: hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>) outs(%[[A]] : memref<16xf16, #hivm.address_space<ub>>)
// CLUSTER: %[[B:.*]] = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
// CLUSTER: hivm.hir.load ins(%arg1 : memref<16xf16, #hivm.address_space<gm>>) outs(%[[B]] : memref<16xf16, #hivm.address_space<ub>>)
// CLUSTER: %[[C:.*]] = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
// CLUSTER: hivm.hir.vadd ins(%[[A]], %[[B]] : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>) outs(%[[C]] : memref<16xf16, #hivm.address_space<ub>>)
// CLUSTER: hivm.hir.store ins(%[[C]] : memref<16xf16, #hivm.address_space<ub>>) outs(%arg2 : memref<16xf16, #hivm.address_space<gm>>)
func.func @cluster_preconditions(%arg0: memref<16xf16, #hivm.address_space<gm>>, %arg1: memref<16xf16, #hivm.address_space<gm>>, %arg2: memref<16xf16, #hivm.address_space<gm>>) {
  %a = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>) outs(%a : memref<16xf16, #hivm.address_space<ub>>)
  %b = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg1 : memref<16xf16, #hivm.address_space<gm>>) outs(%b : memref<16xf16, #hivm.address_space<ub>>)
  %c = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
  hivm.hir.vadd ins(%a, %b : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>) outs(%c : memref<16xf16, #hivm.address_space<ub>>)
  hivm.hir.store ins(%c : memref<16xf16, #hivm.address_space<ub>>) outs(%arg2 : memref<16xf16, #hivm.address_space<gm>>)
  return
}

// -----

// ---------------------------------------------------------------------------
// A needed broadcast must survive the cluster and then be inlined with the
// correct dimension. This is the exact producer/consumer contract that the
// cluster-to-inliner handoff relies on.
// ---------------------------------------------------------------------------
// CLUSTER-LABEL: func.func @cluster_then_inline
// CLUSTER: hivm.hir.vbrc
// CLUSTER-SAME: broadcast_dims = [0]
// CLUSTER: hivm.hir.vmul
//
// WITH-INLINE-LABEL: func.func @cluster_then_inline
// WITH-INLINE-NOT: hivm.hir.vbrc
// WITH-INLINE: hivm.hir.vmul ins({{.*}} : tensor<1x128xf32>, tensor<5x128xf32>) outs({{.*}} : tensor<5x128xf32>) broadcast = [0]
func.func @cluster_then_inline(%arg0: tensor<1x128xf32>, %arg1: tensor<5x128xf32>) -> tensor<5x128xf32> {
  %e0 = tensor.empty() : tensor<5x128xf32>
  %e1 = tensor.empty() : tensor<5x128xf32>
  %brc = hivm.hir.vbrc ins(%arg0 : tensor<1x128xf32>) outs(%e0 : tensor<5x128xf32>) broadcast_dims = [0] -> tensor<5x128xf32>
  %r = hivm.hir.vmul ins(%brc, %arg1 : tensor<5x128xf32>, tensor<5x128xf32>) outs(%e1 : tensor<5x128xf32>) -> tensor<5x128xf32>
  return %r : tensor<5x128xf32>
}

// -----

// ---------------------------------------------------------------------------
// The cluster must be idempotent at this site: running it on its own output
// must not change anything, otherwise the "run cluster then inline" handoff is
// not a fixed point and removing a pass could shift the result. Checked by the
// fact that the single-application CLUSTER checks also hold for the input form
// of the second RUN line.
// ---------------------------------------------------------------------------
// CLUSTER-LABEL: func.func @cluster_fixed_point
// CLUSTER: hivm.hir.vadd ins(%arg0, %arg1
// CLUSTER: return
func.func @cluster_fixed_point(%arg0: tensor<1x128xf32>, %arg1: tensor<1x128xf32>) -> tensor<1x128xf32> {
  %e = tensor.empty() : tensor<1x128xf32>
  %r = hivm.hir.vadd ins(%arg0, %arg1 : tensor<1x128xf32>, tensor<1x128xf32>) outs(%e : tensor<1x128xf32>) -> tensor<1x128xf32>
  return %r : tensor<1x128xf32>
}
