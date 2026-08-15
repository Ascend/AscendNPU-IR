// RUN: bishengir-opt %s --vf-fusion="fusion-mode=max-parallel enable-ra=true enable-new-tree-reduce-policy=true" --split-input-file | FileCheck %s --check-prefix=NEW
// RUN: bishengir-opt %s --vf-fusion="fusion-mode=max-parallel enable-ra=true" --split-input-file | FileCheck %s --check-prefix=LEGACY

// NEW-LABEL: func.func private @ra16_fused_0(
// NEW-SAME: %{{.*}}: tensor<16x8xf32>, %{{.*}}: tensor<8xf32>)
// NEW-NEXT: %{{.*}} = linalg.reduce
// NEW-LABEL: func.func private @ra17_fused_0(
// NEW: linalg.elemwise_binary
// NEW: linalg.fill
// NEW: linalg.reduce
// NEW-LABEL: func.func private @ra64_fused_0(
// NEW: linalg.elemwise_binary
// NEW: linalg.fill
// NEW: linalg.reduce
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
