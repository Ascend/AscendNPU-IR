// RUN: bishengir-opt %s -hfusion-auto-vectorize-v2 --split-input-file | FileCheck %s

// Background:
//   A vsstb-pattern transpose and a rank-reducing op (whose inputs carry
//   broadcast semantics) must NOT fuse into the same FusedNode.
//
// Why:
//   Fusing them incorrectly triggers multi-axis vectorization, which
//   subsequently prevents the AVE layer from correctly resolving the
//   permutation map of transfer_read, leading to precision
//   issues.

// Test 1: vsstb -> rank-reducing via FallbackLeaf path
// Block layout:
// vsstb
//   |
//   +- // -> rank-reducing (sibling, leaf)
//   |
//   =        sync barrier
//   |
//   \------> user (consumer, leaf)


// CHECK-LABEL: func.func @producer_vsstb_and_rank_reducing
// CHECK:         %[[t:.*]] = scf.for
// CHECK:           scf.for
// CHECK:           } {unroll_for_vsstb}
// CHECK:         } {"outlined-loop-target-
// CHECK:         %[[r:.*]] = scf.for
// CHECK:           arith.addf
// CHECK:         } {"outlined-loop-target-
// CHECK:         hivm.hir.sync_block_set
// CHECK:         %[[u:.*]] = scf.for
// CHECK:           %[[t]]
// CHECK:           arith.mulf
// CHECK:         } {"outlined-loop-target-
// CHECK:         return %[[r]], %[[u]]

#map = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @producer_vsstb_and_rank_reducing(%arg0: tensor<8x8x16xf16>, %arg1: tensor<8xf16>) -> (tensor<8x8x16xf16>, tensor<8x8x16xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %empty = tensor.empty() : tensor<8x8x16xf16>

    %transposed = linalg.transpose ins(%arg0 : tensor<8x8x16xf16>) outs(%empty : tensor<8x8x16xf16>) permutation = [1, 0, 2]

    %rank_reducing = linalg.generic {indexing_maps = [#map1, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<8x8x16xf16>, tensor<8xf16>) outs(%empty : tensor<8x8x16xf16>) {
    ^bb0(%in: f16, %in_2: f16, %out: f16):
      %0 = arith.addf %in, %in_2 : f16
      linalg.yield %0 : f16
    } -> tensor<8x8x16xf16>

    hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15

    %user = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed : tensor<8x8x16xf16>) outs(%empty : tensor<8x8x16xf16>) {
    ^bb0(%in: f16, %out: f16):
      %0 = arith.mulf %in, %in : f16
      linalg.yield %0 : f16
    } -> tensor<8x8x16xf16>

    return %rank_reducing, %user : tensor<8x8x16xf16>, tensor<8x8x16xf16>
  }
}

// -----

// Test 2: vsstb -> rank-reducing via Sibling path
// Block layout:
// vsstb (leaf)
//   |
//   +- // -> rank-reducing (sibling, leaf)

// CHECK-LABEL: func.func @vsstb_and_rank_reducing
// CHECK:         %[[t:.*]] = scf.for
// CHECK:           scf.for
// CHECK:           } {unroll_for_vsstb}
// CHECK:         } {"outlined-loop-target-
// CHECK:         %[[r:.*]] = scf.for
// CHECK:           arith.addf
// CHECK:         } {"outlined-loop-target-
// CHECK:         return %[[t]], %[[r]]

#map = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @vsstb_and_rank_reducing(%arg0: tensor<8x8x16xf16>, %arg1: tensor<8xf16>) -> (tensor<8x8x16xf16>, tensor<8x8x16xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %empty = tensor.empty() : tensor<8x8x16xf16>

    %transposed = linalg.transpose ins(%arg0 : tensor<8x8x16xf16>) outs(%empty : tensor<8x8x16xf16>) permutation = [1, 0, 2]

    %rank_reducing = linalg.generic {indexing_maps = [#map1, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<8x8x16xf16>, tensor<8xf16>) outs(%empty : tensor<8x8x16xf16>) {
    ^bb0(%in: f16, %in_2: f16, %out: f16):
      %0 = arith.addf %in, %in_2 : f16
      linalg.yield %0 : f16
    } -> tensor<8x8x16xf16>

    return %transposed, %rank_reducing : tensor<8x8x16xf16>, tensor<8x8x16xf16>
  }
}

// -----

// Test 3: vsstb -> rank-reducing via Producer path
// Block layout:
// vsstb
//   |
//   +- // -> rank-reducing (consumer, leaf)

// CHECK-LABEL: func.func @vsstb_and_rank_reducing_consumer
// CHECK:         %[[t:.*]] = scf.for
// CHECK:           scf.for
// CHECK:           } {unroll_for_vsstb}
// CHECK:         } {"outlined-loop-target-
// CHECK:         %[[r:.*]] = scf.for
// CHECK:           %[[t]]
// CHECK:           arith.addf
// CHECK:         } {"outlined-loop-target-
// CHECK:         return %[[r]]

#map = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @vsstb_and_rank_reducing_consumer(%arg0: tensor<8x8x16xf16>, %arg1: tensor<8xf16>) -> (tensor<8x8x16xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %empty = tensor.empty() : tensor<8x8x16xf16>

    %transposed = linalg.transpose ins(%arg0 : tensor<8x8x16xf16>) outs(%empty : tensor<8x8x16xf16>) permutation = [1, 0, 2]

    %rank_reducing = linalg.generic {indexing_maps = [#map1, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed, %arg1 : tensor<8x8x16xf16>, tensor<8xf16>) outs(%empty : tensor<8x8x16xf16>) {
    ^bb0(%in: f16, %in_2: f16, %out: f16):
      %0 = arith.addf %in, %in_2 : f16
      linalg.yield %0 : f16
    } -> tensor<8x8x16xf16>

    return %rank_reducing : tensor<8x8x16xf16>
  }
}

// -----

// Test 4: rank-reducing -> vsstb via FallbackLeaf path
// Block layout:
// rank-reducing
//   |
//   +- // -> vsstb (sibling, leaf)
//   |
//   =        sync barrier
//   |
//   \------> user (consumer, leaf)


// CHECK-LABEL: func.func @producer_rank_reducing_and_vsstb
// CHECK:         %[[r:.*]] = scf.for
// CHECK:           arith.addf
// CHECK:         } {"outlined-loop-target-
// CHECK:         %[[t:.*]] = scf.for
// CHECK:           scf.for
// CHECK:           } {unroll_for_vsstb}
// CHECK:         } {"outlined-loop-target-
// CHECK:         hivm.hir.sync_block_set
// CHECK:         %[[u:.*]] = scf.for
// CHECK:           %[[t]]
// CHECK:           arith.mulf
// CHECK:         } {"outlined-loop-target-
// CHECK:         return %[[r]], %[[u]]

#map = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @producer_rank_reducing_and_vsstb(%arg0: tensor<8x8x16xf16>, %arg1: tensor<8xf16>) -> (tensor<8x8x16xf16>, tensor<8x8x16xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %empty = tensor.empty() : tensor<8x8x16xf16>

    %rank_reducing = linalg.generic {indexing_maps = [#map1, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<8x8x16xf16>, tensor<8xf16>) outs(%empty : tensor<8x8x16xf16>) {
    ^bb0(%in: f16, %in_2: f16, %out: f16):
      %0 = arith.addf %in, %in_2 : f16
      linalg.yield %0 : f16
    } -> tensor<8x8x16xf16>

    %transposed = linalg.transpose ins(%arg0 : tensor<8x8x16xf16>) outs(%empty : tensor<8x8x16xf16>) permutation = [1, 0, 2]

    hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15

    %user = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed : tensor<8x8x16xf16>) outs(%empty : tensor<8x8x16xf16>) {
    ^bb0(%in: f16, %out: f16):
      %0 = arith.mulf %in, %in : f16
      linalg.yield %0 : f16
    } -> tensor<8x8x16xf16>

    return %rank_reducing, %user : tensor<8x8x16xf16>, tensor<8x8x16xf16>
  }
}

// -----

// Test 5: rank-reducing -> vsstb via Sibling path
// Block layout:
// rank-reducing (leaf)
//   |
//   +- // -> vsstb (sibling, leaf)

// CHECK-LABEL: func.func @rank_reducing_and_vsstb
// CHECK:         %[[r:.*]] = scf.for
// CHECK:           arith.addf
// CHECK:         } {"outlined-loop-target-
// CHECK:         %[[t:.*]] = scf.for
// CHECK:           scf.for
// CHECK:           } {unroll_for_vsstb}
// CHECK:         } {"outlined-loop-target-
// CHECK:         return %[[r]], %[[t]]

#map = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @rank_reducing_and_vsstb(%arg0: tensor<8x8x16xf16>, %arg1: tensor<8xf16>) -> (tensor<8x8x16xf16>, tensor<8x8x16xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %empty = tensor.empty() : tensor<8x8x16xf16>

    %rank_reducing = linalg.generic {indexing_maps = [#map1, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<8x8x16xf16>, tensor<8xf16>) outs(%empty : tensor<8x8x16xf16>) {
    ^bb0(%in: f16, %in_2: f16, %out: f16):
      %0 = arith.addf %in, %in_2 : f16
      linalg.yield %0 : f16
    } -> tensor<8x8x16xf16>

    %transposed = linalg.transpose ins(%arg0 : tensor<8x8x16xf16>) outs(%empty : tensor<8x8x16xf16>) permutation = [1, 0, 2]

    return %rank_reducing, %transposed : tensor<8x8x16xf16>, tensor<8x8x16xf16>
  }
}

// -----

// Test 6: rank-reducing -> vsstb via Producer path
// Block layout:
// rank-reducing
//   |
//   +- // -> vsstb (consumer, leaf)

// CHECK-LABEL: func.func @vsstb_and_rank_reducing_consumer
// CHECK:         %[[r:.*]] = scf.for
// CHECK:           arith.addf
// CHECK:         } {"outlined-loop-target-
// CHECK:         %[[t:.*]] = scf.for
// CHECK:           scf.for
// CHECK:           %[[r]]
// CHECK:           } {unroll_for_vsstb}
// CHECK:         } {"outlined-loop-target-
// CHECK:         return %[[t]]

#map = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @vsstb_and_rank_reducing_consumer(%arg0: tensor<8x8x16xf16>, %arg1: tensor<8xf16>) -> (tensor<8x8x16xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %empty = tensor.empty() : tensor<8x8x16xf16>

    %rank_reducing = linalg.generic {indexing_maps = [#map1, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<8x8x16xf16>, tensor<8xf16>) outs(%empty : tensor<8x8x16xf16>) {
    ^bb0(%in: f16, %in_2: f16, %out: f16):
      %0 = arith.addf %in, %in_2 : f16
      linalg.yield %0 : f16
    } -> tensor<8x8x16xf16>

    %transposed = linalg.transpose ins(%rank_reducing : tensor<8x8x16xf16>) outs(%empty : tensor<8x8x16xf16>) permutation = [1, 0, 2]

    return %transposed : tensor<8x8x16xf16>
  }
}

// -----

// Test 7: vsstb -> [ rank-reducing leaf-op ] via FallbackLeaf path
// Block layout:
// vsstb
//   |
//   +- // --> rank-reducing (sibling)
//   |     |
//   |     +-> elemwise (sibling, leaf)
//   |
//   =        sync barrier
//   |
//   \------> user (consumer, leaf)
// Test 7: rank-reducing -> vsstb via Producer path
// Block layout:
// rank-reducing
//   |
//   +- // -> vsstb (consumer, leaf)

// CHECK-LABEL: func.func @rank_reducing_not_leaf
// CHECK:         %[[t:.*]] = scf.for
// CHECK:           scf.for
// CHECK:           } {unroll_for_vsstb}
// CHECK:         } {"outlined-loop-target-
// CHECK:         %[[re:.*]] = scf.for
// CHECK:           arith.addf
// CHECK:           arith.mulf
// CHECK:         } {"outlined-loop-target-
// CHECK:         hivm.hir.sync_block_set
// CHECK:         %[[u:.*]] = scf.for
// CHECK:           %[[t]]
// CHECK:           arith.mulf
// CHECK:         } {"outlined-loop-target-
// CHECK:         return %[[u]], %[[re]]

#map = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @rank_reducing_not_leaf(%arg0: tensor<8x8x16xf16>, %arg1: tensor<8xf16>) -> (tensor<8x8x16xf16>, tensor<8x8x16xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %empty = tensor.empty() : tensor<8x8x16xf16>

    %transposed = linalg.transpose ins(%arg0 : tensor<8x8x16xf16>) outs(%empty : tensor<8x8x16xf16>) permutation = [1, 0, 2]

    %rank_reducing = linalg.generic {indexing_maps = [#map1, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<8x8x16xf16>, tensor<8xf16>) outs(%empty : tensor<8x8x16xf16>) {
    ^bb0(%in: f16, %in_2: f16, %out: f16):
      %0 = arith.addf %in, %in_2 : f16
      linalg.yield %0 : f16
    } -> tensor<8x8x16xf16>

    %elemwise = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%rank_reducing, %rank_reducing : tensor<8x8x16xf16>, tensor<8x8x16xf16>) outs(%empty : tensor<8x8x16xf16>) {
    ^bb0(%in: f16, %in_2: f16, %out: f16):
      %0 = arith.mulf %in, %in_2 : f16
      linalg.yield %0 : f16
    } -> tensor<8x8x16xf16>

    hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15

    %user = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed : tensor<8x8x16xf16>) outs(%empty : tensor<8x8x16xf16>) {
    ^bb0(%in: f16, %out: f16):
      %0 = arith.mulf %in, %in : f16
      linalg.yield %0 : f16
    } -> tensor<8x8x16xf16>

    return %user, %elemwise : tensor<8x8x16xf16>, tensor<8x8x16xf16>
  }
}
