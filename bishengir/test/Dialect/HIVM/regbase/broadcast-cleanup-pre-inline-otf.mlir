// RUN: bishengir-opt --hivm-canonicalization-pipeline --hivm-inline-otf-broadcast %s | FileCheck %s
// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_957b --hivm-canonicalization-pipeline --hivm-inline-otf-broadcast %s | FileCheck %s

// Regression tests for the broadcast behavior that the cleanup cluster
// immediately preceding hivm-inline-otf-broadcast must preserve in the
// RegBase pre-bufferization pipeline
// (hivmPreBufferizationOptimizationPipeline):
//
//   * a broadcast that is redundant (1:1, no rank expansion) must be removed,
//   * a broadcast that is genuinely needed must be kept and, once inlined into
//     its consumer, must carry the correct `broadcast` dimension,
//   * a dynamic-shape unit-dimension broadcast that is inlined must keep the
//     right unit dimension,
//   * a value-forwarding case produced by the cluster must not be undone.
//
// The second RUN line repeats exactly the same checks on the A5 RegBase target
// (Ascend950PR_957b). The device spec appended by -hacc-append-device-spec only
// adds module attributes, so the same checks must hold on both targets.

// ---------------------------------------------------------------------------
// Category B1: 1:1 broadcast into the same shape is redundant and must vanish;
// the source value is forwarded directly.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @redundant_vbrc_1to1
// CHECK-NOT: hivm.hir.vbrc
func.func @redundant_vbrc_1to1(%arg0: tensor<1x128xf32>) -> tensor<1x128xf32> {
  %empty = tensor.empty() : tensor<1x128xf32>
  %brc = hivm.hir.vbrc ins(%arg0 : tensor<1x128xf32>) outs(%empty : tensor<1x128xf32>) broadcast_dims = [0] -> tensor<1x128xf32>
  return %brc : tensor<1x128xf32>
}

// -----

// ---------------------------------------------------------------------------
// Category B2: a broadcast that actually expands the leading dimension must be
// kept. It must not be dropped by redundant-broadcast cleanup.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @needed_vbrc
// CHECK: hivm.hir.vbrc
// CHECK-SAME: broadcast_dims = [0]
func.func @needed_vbrc(%arg0: tensor<1x128xf32>) -> tensor<5x128xf32> {
  %empty = tensor.empty() : tensor<5x128xf32>
  %brc = hivm.hir.vbrc ins(%arg0 : tensor<1x128xf32>) outs(%empty : tensor<5x128xf32>) broadcast_dims = [0] -> tensor<5x128xf32>
  return %brc : tensor<5x128xf32>
}

// -----

// ---------------------------------------------------------------------------
// Category B3: an inline broadcast feeding a binary vector op. The broadcast is
// folded into the consumer and the consumer must carry the matching
// `broadcast` dimension; no standalone vbrc may remain.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @inline_brc_into_vmul
// CHECK-NOT: hivm.hir.vbrc
// CHECK: hivm.hir.vmul ins({{.*}} : tensor<1x128xf32>, tensor<5x128xf32>) outs({{.*}} : tensor<5x128xf32>) broadcast = [0]
func.func @inline_brc_into_vmul(%arg0: tensor<1x128xf32>, %arg1: tensor<5x128xf32>) -> tensor<5x128xf32> {
  %e0 = tensor.empty() : tensor<5x128xf32>
  %e1 = tensor.empty() : tensor<5x128xf32>
  %brc = hivm.hir.vbrc ins(%arg0 : tensor<1x128xf32>) outs(%e0 : tensor<5x128xf32>) broadcast_dims = [0] -> tensor<5x128xf32>
  %r = hivm.hir.vmul ins(%brc, %arg1 : tensor<5x128xf32>, tensor<5x128xf32>) outs(%e1 : tensor<5x128xf32>) -> tensor<5x128xf32>
  return %r : tensor<5x128xf32>
}

// -----

// ---------------------------------------------------------------------------
// Category B4: dynamic shape with a unit dimension. The inlined broadcast must
// keep the correct (unit) dimension, and dynamic sizes must be preserved.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @dynamic_unit_dim_brc
// CHECK-NOT: hivm.hir.vbrc
// CHECK: tensor.dim
// CHECK: hivm.hir.vmul ins({{.*}} : tensor<1x?x1xf32>, tensor<1x?x16xf32>) outs({{.*}} : tensor<1x?x16xf32>) broadcast = [2]
func.func @dynamic_unit_dim_brc(%arg0: tensor<1x?x16xf32>, %arg1: tensor<1x?x1xf32>) -> tensor<1x?x16xf32> {
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c1 : tensor<1x?x16xf32>
  %0 = tensor.empty(%dim) : tensor<1x?x16xf32>
  %1 = tensor.empty(%dim) : tensor<1x?x16xf32>
  %2 = tensor.empty(%dim) : tensor<1x?x16xf32>
  %3 = hivm.hir.vbrc ins(%arg0 : tensor<1x?x16xf32>) outs(%0 : tensor<1x?x16xf32>) broadcast_dims = [0] -> tensor<1x?x16xf32>
  %4 = hivm.hir.vbrc ins(%arg1 : tensor<1x?x1xf32>) outs(%1 : tensor<1x?x16xf32>) broadcast_dims = [2] -> tensor<1x?x16xf32>
  %5 = hivm.hir.vmul ins(%4, %3 : tensor<1x?x16xf32>, tensor<1x?x16xf32>) outs(%2 : tensor<1x?x16xf32>) -> tensor<1x?x16xf32>
  return %5 : tensor<1x?x16xf32>
}
