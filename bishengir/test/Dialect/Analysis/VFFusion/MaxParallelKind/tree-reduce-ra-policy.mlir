// RUN: bishengir-opt %s --vf-fusion="fusion-mode=max-parallel enable-ra=true enable-new-tree-reduce-policy=true" --split-input-file | FileCheck %s --check-prefix=NEW
// RUN: bishengir-opt %s --vf-fusion="fusion-mode=max-parallel enable-ra=true" --split-input-file | FileCheck %s --check-prefix=LEGACY

// NEW-LABEL: func.func private @ra16_fused_0(
// NEW-SAME: %{{.*}}: tensor<16x8xf32>, %{{.*}}: tensor<8xf32>)
// NEW-NEXT: %{{.*}} = linalg.reduce
// NEW-LABEL: func.func private @ra17_fused_0(
// NEW-SAME: %{{.*}}: tensor<17x8xf32>, %{{.*}}: tensor<8xf32>)
// NEW-NEXT: %{{.*}} = linalg.reduce
// NEW-SAME: hfusion.register_tree_reduction_selected
// NEW: module attributes {hfusion.regular_tree_reduction_scope, hfusion.tree_reduction_selection_frozen}
// NEW-NOT: hfusion.legacy_tree_reduction_scope
// NEW-LABEL: func.func private @mixed_ra_ar_fused_0(
// NEW: %{{.*}} = linalg.reduce
// NEW-LABEL: func.func private @mixed_ra_ar_fused_1(
// NEW: %{{.*}} = linalg.reduce
// NEW: module attributes {hfusion.legacy_tree_reduction_scope, hfusion.tree_reduction_selection_frozen}
// NEW-LABEL: func.func private @ra64_fused_0(
// NEW-SAME: %{{.*}}: tensor<64x8xf32>, %{{.*}}: tensor<8xf32>)
// NEW-NEXT: %{{.*}} = linalg.reduce
// NEW-SAME: hfusion.regular_tree_reduction_selected
// NEW-LABEL: func.func private @ra65_fused_0(
// NEW: linalg.elemwise_binary
// NEW: linalg.fill
// NEW: linalg.reduce
// NEW-LABEL: func.func private @ra128_fused_0(
// NEW: linalg.elemwise_binary
// NEW: linalg.fill
// NEW: linalg.reduce
// NEW-LABEL: func.func private @ra128_noncanonical_fused_0(
// NEW-NEXT: %{{.*}} = linalg.generic

// LEGACY-LABEL: func.func private @ra16_fused_0(
// LEGACY-NEXT: %{{.*}} = linalg.reduce
// LEGACY-LABEL: func.func private @ra17_fused_0(
// LEGACY-NEXT: %{{.*}} = linalg.reduce
// LEGACY-LABEL: func.func private @ra64_fused_0(
// LEGACY-NEXT: %{{.*}} = linalg.reduce
// LEGACY-LABEL: func.func private @ra65_fused_0(
// LEGACY-NEXT: %{{.*}} = linalg.reduce
// LEGACY-LABEL: func.func private @ra128_fused_0(
// LEGACY-NEXT: %{{.*}} = linalg.reduce
// LEGACY-LABEL: func.func private @ra128_noncanonical_fused_0(
// LEGACY-NEXT: %{{.*}} = linalg.generic

module {
  func.func @ra16(%arg0: tensor<16x8xf32>) -> tensor<8xf32>
      attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0 = arith.constant 0.0 : f32
    %empty0 = tensor.empty() : tensor<16x8xf32>
    %mapped = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
        ins(%arg0, %arg0 : tensor<16x8xf32>, tensor<16x8xf32>)
        outs(%empty0 : tensor<16x8xf32>) -> tensor<16x8xf32>
    %empty1 = tensor.empty() : tensor<8xf32>
    %init = linalg.fill ins(%c0 : f32) outs(%empty1 : tensor<8xf32>)
        -> tensor<8xf32>
    %result = linalg.reduce ins(%mapped : tensor<16x8xf32>)
        outs(%init : tensor<8xf32>) dimensions = [0]
      (%in: f32, %out: f32) {
        %sum = arith.addf %in, %out : f32
        linalg.yield %sum : f32
      }
    return %result : tensor<8xf32>
  }
}

// -----

module {
  func.func @ra17(%arg0: tensor<17x8xf32>) -> tensor<8xf32>
      attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0 = arith.constant 0.0 : f32
    %empty0 = tensor.empty() : tensor<17x8xf32>
    %mapped = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
        ins(%arg0, %arg0 : tensor<17x8xf32>, tensor<17x8xf32>)
        outs(%empty0 : tensor<17x8xf32>) -> tensor<17x8xf32>
    %empty1 = tensor.empty() : tensor<8xf32>
    %init = linalg.fill ins(%c0 : f32) outs(%empty1 : tensor<8xf32>)
        -> tensor<8xf32>
    %result = linalg.reduce ins(%mapped : tensor<17x8xf32>)
        outs(%init : tensor<8xf32>) dimensions = [0]
      (%in: f32, %out: f32) {
        %sum = arith.addf %in, %out : f32
        linalg.yield %sum : f32
      }
    return %result : tensor<8xf32>
  }
}

// -----

// A scope containing several canonical RA candidates plus another reduction
// is not a safe TreeReduceV2 scope: that pass replaces a surrounding loop.
// Keep it on the regular path while preserving reduction isolation.
module {
  func.func @mixed_ra_ar(%arg0: tensor<16x32xf32>)
      -> (tensor<32xf32>, tensor<32xf32>, tensor<16xf32>)
      attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0 = arith.constant 0.0 : f32
    %empty0 = tensor.empty() : tensor<32xf32>
    %init0 = linalg.fill ins(%c0 : f32) outs(%empty0 : tensor<32xf32>)
        -> tensor<32xf32>
    %result0 = linalg.reduce ins(%arg0 : tensor<16x32xf32>)
        outs(%init0 : tensor<32xf32>) dimensions = [0]
      (%in: f32, %out: f32) {
        %sum = arith.addf %in, %out : f32
        linalg.yield %sum : f32
      }
    %empty1 = tensor.empty() : tensor<32xf32>
    %init1 = linalg.fill ins(%c0 : f32) outs(%empty1 : tensor<32xf32>)
        -> tensor<32xf32>
    %result1 = linalg.reduce ins(%arg0 : tensor<16x32xf32>)
        outs(%init1 : tensor<32xf32>) dimensions = [0]
      (%in: f32, %out: f32) {
        %sum = arith.addf %in, %out : f32
        linalg.yield %sum : f32
      }
    %empty2 = tensor.empty() : tensor<16xf32>
    %init2 = linalg.fill ins(%c0 : f32) outs(%empty2 : tensor<16xf32>)
        -> tensor<16xf32>
    %result2 = linalg.reduce ins(%arg0 : tensor<16x32xf32>)
        outs(%init2 : tensor<16xf32>) dimensions = [1]
      (%in: f32, %out: f32) {
        %sum = arith.addf %in, %out : f32
        linalg.yield %sum : f32
      }
    return %result0, %result1, %result2
        : tensor<32xf32>, tensor<32xf32>, tensor<16xf32>
  }
}

// -----

module {
  func.func @ra64(%arg0: tensor<64x8xf32>) -> tensor<8xf32>
      attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0 = arith.constant 0.0 : f32
    %empty0 = tensor.empty() : tensor<64x8xf32>
    %mapped = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
        ins(%arg0, %arg0 : tensor<64x8xf32>, tensor<64x8xf32>)
        outs(%empty0 : tensor<64x8xf32>) -> tensor<64x8xf32>
    %empty1 = tensor.empty() : tensor<8xf32>
    %init = linalg.fill ins(%c0 : f32) outs(%empty1 : tensor<8xf32>)
        -> tensor<8xf32>
    %result = linalg.reduce ins(%mapped : tensor<64x8xf32>)
        outs(%init : tensor<8xf32>) dimensions = [0]
      (%in: f32, %out: f32) {
        %sum = arith.addf %in, %out : f32
        linalg.yield %sum : f32
      }
    return %result : tensor<8xf32>
  }
}

// -----

module {
  func.func @ra65(%arg0: tensor<65x8xf32>) -> tensor<8xf32>
      attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0 = arith.constant 0.0 : f32
    %empty0 = tensor.empty() : tensor<65x8xf32>
    %mapped = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
        ins(%arg0, %arg0 : tensor<65x8xf32>, tensor<65x8xf32>)
        outs(%empty0 : tensor<65x8xf32>) -> tensor<65x8xf32>
    %empty1 = tensor.empty() : tensor<8xf32>
    %init = linalg.fill ins(%c0 : f32) outs(%empty1 : tensor<8xf32>)
        -> tensor<8xf32>
    %result = linalg.reduce ins(%mapped : tensor<65x8xf32>)
        outs(%init : tensor<8xf32>) dimensions = [0]
      (%in: f32, %out: f32) {
        %sum = arith.addf %in, %out : f32
        linalg.yield %sum : f32
      }
    return %result : tensor<8xf32>
  }
}

// -----

module {
  func.func @ra128(%arg0: tensor<128x8xf32>) -> tensor<8xf32>
      attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0 = arith.constant 0.0 : f32
    %empty0 = tensor.empty() : tensor<128x8xf32>
    %mapped = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
        ins(%arg0, %arg0 : tensor<128x8xf32>, tensor<128x8xf32>)
        outs(%empty0 : tensor<128x8xf32>) -> tensor<128x8xf32>
    %empty1 = tensor.empty() : tensor<8xf32>
    %init = linalg.fill ins(%c0 : f32) outs(%empty1 : tensor<8xf32>)
        -> tensor<8xf32>
    %result = linalg.reduce ins(%mapped : tensor<128x8xf32>)
        outs(%init : tensor<8xf32>) dimensions = [0]
      (%in: f32, %out: f32) {
        %sum = arith.addf %in, %out : f32
        linalg.yield %sum : f32
      }
    return %result : tensor<8xf32>
  }
}

// -----

// A structurally unsupported large RA reduction must keep the established
// isolated policy.  The selective fallback is only for the canonical AddF
// reduction that AutoVectorizeV2 can otherwise lower as a tree.

module {
  func.func @ra128_noncanonical(%arg0: tensor<128x8xf32>) -> tensor<8xf32>
      attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0 = arith.constant 0.0 : f32
    %empty0 = tensor.empty() : tensor<128x8xf32>
    %mapped = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
        ins(%arg0, %arg0 : tensor<128x8xf32>, tensor<128x8xf32>)
        outs(%empty0 : tensor<128x8xf32>) -> tensor<128x8xf32>
    %empty1 = tensor.empty() : tensor<8xf32>
    %init = linalg.fill ins(%c0 : f32) outs(%empty1 : tensor<8xf32>)
        -> tensor<8xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d1)>],
        iterator_types = ["reduction", "parallel"]
      } ins(%mapped : tensor<128x8xf32>) outs(%init : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      %scaled = arith.mulf %sum, %in : f32
      linalg.yield %scaled : f32
    } -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}

// -----

// Two bounded register candidates cannot be rewritten by the single-loop
// direct strategy. Keep the whole scope on the established TreeReduceV2 path;
// regular fusion changes the summation order of this accuracy-sensitive case.
// NEW: module attributes {hfusion.legacy_tree_reduction_scope, hfusion.tree_reduction_selection_frozen}
// NEW-LABEL: func.func private @two_ra32_fused_0(
// NEW-NEXT: %{{.*}} = linalg.reduce
// NEW-LABEL: func.func private @two_ra32_fused_1(
// NEW-NEXT: %{{.*}} = linalg.reduce
// LEGACY-LABEL: func.func private @two_ra32_fused_0(
// LEGACY-NEXT: %{{.*}} = linalg.reduce
// LEGACY-LABEL: func.func private @two_ra32_fused_1(
// LEGACY-NEXT: %{{.*}} = linalg.reduce

module {
  func.func @two_ra32(%arg0: tensor<32x32xf32>)
      -> (tensor<32xf32>, tensor<32xf32>)
      attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0 = arith.constant 0.0 : f32
    %empty0 = tensor.empty() : tensor<32xf32>
    %init0 = linalg.fill ins(%c0 : f32) outs(%empty0 : tensor<32xf32>)
        -> tensor<32xf32>
    %result0 = linalg.reduce ins(%arg0 : tensor<32x32xf32>)
        outs(%init0 : tensor<32xf32>) dimensions = [0]
      (%in: f32, %out: f32) {
        %sum = arith.addf %in, %out : f32
        linalg.yield %sum : f32
      }
    %empty1 = tensor.empty() : tensor<32xf32>
    %init1 = linalg.fill ins(%c0 : f32) outs(%empty1 : tensor<32xf32>)
        -> tensor<32xf32>
    %result1 = linalg.reduce ins(%arg0 : tensor<32x32xf32>)
        outs(%init1 : tensor<32xf32>) dimensions = [0]
      (%in: f32, %out: f32) {
        %sum = arith.addf %in, %out : f32
        linalg.yield %sum : f32
      }
    return %result0, %result1 : tensor<32xf32>, tensor<32xf32>
  }
}

// -----

// A cumulative scan and its block-total reduction must retain the established
// TreeReduceV2 association. This compatibility route is deliberately bounded
// to one scan, one canonical reduction, and the largest supported tile.
// NEW: module attributes {hfusion.legacy_tree_reduction_scope, hfusion.tree_reduction_selection_frozen}
// NEW-LABEL: func.func private @cumsum_ra128_fused_0(
// NEW: linalg.reduce

module {
  func.func @cumsum_ra128(%arg0: tensor<128x8xf32>)
      -> (tensor<128x8xf32>, tensor<8xf32>)
      attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %scan = hfusion.cumsum %arg0 : tensor<128x8xf32> cum_dims = [0]
        reverse = false -> tensor<128x8xf32>
    %c0 = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<8xf32>
    %init = linalg.fill ins(%c0 : f32) outs(%empty : tensor<8xf32>)
        -> tensor<8xf32>
    %result = linalg.reduce ins(%arg0 : tensor<128x8xf32>)
        outs(%init : tensor<8xf32>) dimensions = [0]
      (%in: f32, %out: f32) {
        %sum = arith.addf %in, %out : f32
        linalg.yield %sum : f32
      }
    return %scan, %result : tensor<128x8xf32>, tensor<8xf32>
  }
}
