// RUN: bishengir-opt %s -hfusion-auto-vectorize-v2="enable-multiple-consumer-fusion=true" --remove-redundant-write-and-read-pair -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -hfusion-auto-vectorize-v2="enable-multiple-consumer-fusion=true emit-transform-sequence" -split-input-file | FileCheck %s --check-prefix=CHECK-ORDER

// Test: producer consolidation with forward-slice users.
//
// Block layout:
// P0 -> L0(P0) -> expand0(L0) -> P1 -> L1(P1) -> expand1(L1) -> P2 -> collapse_shape -> L2(P2, collapsed) -> expand2(L2)
//
// After consolidation all
// collapse_shape -> {P0,P1,P2,L0,L1,L2} -> expand0 -> expand1 -> expand2

// CHECK-ORDER: func.func @test_producer_consolidate_to_tail
// FusedNode 1 [ P0 P1 P2 L0 L1 L2 ]
// CHECK-ORDER-NOT: {"hfusion-auto-vectorize-target-
// CHECK-ORDER: {"hfusion-auto-vectorize-target-5"}
// CHECK-ORDER-NOT: {"hfusion-auto-vectorize-target-
// CHECK-ORDER: {"hfusion-auto-vectorize-target-3"}
// CHECK-ORDER-NOT: {"hfusion-auto-vectorize-target-
// CHECK-ORDER: {"hfusion-auto-vectorize-target-1"}
// CHECK-ORDER-NOT: {"hfusion-auto-vectorize-target-
// CHECK-ORDER: {"hfusion-auto-vectorize-target-2"}
// CHECK-ORDER-NOT: {"hfusion-auto-vectorize-target-
// CHECK-ORDER: {"hfusion-auto-vectorize-target-4"}
// CHECK-ORDER-NOT: {"hfusion-auto-vectorize-target-
// CHECK-ORDER: {"hfusion-auto-vectorize-target-6"}
// CHECK-ORDER: transform.sequence
// CHECK-ORDER: {"hfusion-auto-vectorize-target-2"}
// CHECK-ORDER: {"hfusion-auto-vectorize-target-4"}
// CHECK-ORDER: {"hfusion-auto-vectorize-target-6"}
// CHECK-ORDER: annotate {{.*}} "outlined-loop-target-1"
// CHECK-ORDER: {"hfusion-auto-vectorize-target-1"}
// CHECK-ORDER: {"outlined-loop-target-1"}
// CHECK-ORDER: {"hfusion-auto-vectorize-target-3"}
// CHECK-ORDER: {"outlined-loop-target-1"}
// CHECK-ORDER: {"hfusion-auto-vectorize-target-5"}
// CHECK-ORDER: {"outlined-loop-target-1"}

// CHECK:         func.func @test_producer_consolidate_to_tail
// CHECK-COUNT-1:   %[[collapsed:.*]] = tensor.collapse_shape
// CHECK:           %[[res:.*]]:3 = scf.for
// CHECK-DAG:         %[[p0:.*]] = arith.addf
// CHECK-DAG:         %[[l0:.*]] = arith.mulf %[[p0]]
// CHECK-DAG:         %[[p1:.*]] = arith.subf
// CHECK-DAG:         %[[l1:.*]] = arith.mulf %[[p1]]
// CHECK-DAG:         %[[p2:.*]] = arith.mulf
// CHECK-DAG:         %[[slice_collapsed:.*]] = tensor.extract_slice %[[collapsed]]
// CHECK-DAG:         %[[read_collapsed:.*]] = vector.transfer_read %[[slice_collapsed]]
// CHECK-DAG:         %[[l2:.*]] = arith.mulf %[[p2]], %[[read_collapsed]]
// CHECK-DAG:         %[[write_l0:.*]] = vector.transfer_write %[[l0]]
// CHECK-DAG:         %[[inserted_l0:.*]] = tensor.insert_slice %[[write_l0]]
// CHECK-DAG:         %[[write_l1:.*]] = vector.transfer_write %[[l1]]
// CHECK-DAG:         %[[inserted_l1:.*]] = tensor.insert_slice %[[write_l1]]
// CHECK-DAG:         %[[write_l2:.*]] = vector.transfer_write %[[l2]]
// CHECK-DAG:         %[[inserted_l2:.*]] = tensor.insert_slice %[[write_l2]]
// CHECK-DAG:         scf.yield %[[inserted_l0]], %[[inserted_l1]], %[[inserted_l2]]
// CHECK:           {"outlined-loop-target-1"}
// CHECK-DAG:       %[[e0:.*]] = tensor.expand_shape %[[res]]#0
// CHECK-DAG:       %[[e1:.*]] = tensor.expand_shape %[[res]]#1
// CHECK-DAG:       %[[e2:.*]] = tensor.expand_shape %[[res]]#2
// CHECK:           return %[[e0]], %[[e1]], %[[e2]]

#map = affine_map<(d0) -> (d0)>
module {
  func.func @test_producer_consolidate_to_tail(%arg0: tensor<128xf16>, %arg1: tensor<128xf16>, %arg2: tensor<32x4xf16>) -> (tensor<32x4xf16>, tensor<32x4xf16>, tensor<32x4xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = tensor.empty() : tensor<128xf16>
    %producer_0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<128xf16>, tensor<128xf16>) outs(%0 : tensor<128xf16>) {
    ^bb0(%in: f16, %in_2: f16, %out: f16):
      %res = arith.addf %in, %in_2 : f16
      linalg.yield %res : f16
    } -> tensor<128xf16>
    %leaf_0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%producer_0, %arg1 : tensor<128xf16>, tensor<128xf16>) outs(%0 : tensor<128xf16>) {
    ^bb0(%in: f16, %in_2: f16, %out: f16):
      %res = arith.mulf %in, %in_2 : f16
      linalg.yield %res : f16
    } -> tensor<128xf16>
    %expanded_0 = tensor.expand_shape %leaf_0 [[0, 1]] output_shape [32, 4] : tensor<128xf16> into tensor<32x4xf16>

    %1 = tensor.empty() : tensor<128xf16>
    %producer_1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<128xf16>, tensor<128xf16>) outs(%1 : tensor<128xf16>) {
    ^bb0(%in: f16, %in_2: f16, %out: f16):
      %res = arith.subf %in, %in_2 : f16
      linalg.yield %res : f16
    } -> tensor<128xf16>
    %leaf_1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%producer_1, %arg1 : tensor<128xf16>, tensor<128xf16>) outs(%1 : tensor<128xf16>) {
    ^bb0(%in: f16, %in_2: f16, %out: f16):
      %res = arith.mulf %in, %in_2 : f16
      linalg.yield %res : f16
    } -> tensor<128xf16>
    %expanded_1 = tensor.expand_shape %leaf_1 [[0, 1]] output_shape [32, 4] : tensor<128xf16> into tensor<32x4xf16>

    %2 = tensor.empty() : tensor<128xf16>
    %producer_2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<128xf16>, tensor<128xf16>) outs(%2 : tensor<128xf16>) {
    ^bb0(%in: f16, %in_2: f16, %out: f16):
      %res = arith.mulf %in, %in_2 : f16
      linalg.yield %res : f16
    } -> tensor<128xf16>
    %collapsed = tensor.collapse_shape %arg2 [[0, 1]] : tensor<32x4xf16> into tensor<128xf16>
    %leaf_2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%producer_2, %collapsed : tensor<128xf16>, tensor<128xf16>) outs(%2 : tensor<128xf16>) {
    ^bb0(%in: f16, %in_2: f16, %out: f16):
      %res = arith.mulf %in, %in_2 : f16
      linalg.yield %res : f16
    } -> tensor<128xf16>
    %expanded_2 = tensor.expand_shape %leaf_2 [[0, 1]] output_shape [32, 4] : tensor<128xf16> into tensor<32x4xf16>

    return %expanded_0, %expanded_1, %expanded_2 : tensor<32x4xf16>, tensor<32x4xf16>, tensor<32x4xf16>
  }
}

// -----

// CHECK-ORDER: func.func @mcf_producer_fuse_into_later_node
// FusedNode 1 [ producer leaf_2 ] add -> sub
// CHECK-ORDER-NOT: {"hfusion-auto-vectorize-target-
// CHECK-ORDER: {"hfusion-auto-vectorize-target-1"}
// CHECK-ORDER-NOT: {"hfusion-auto-vectorize-target-
// CHECK-ORDER: {"hfusion-auto-vectorize-target-4"}
// FusedNode 2 [ other    leaf_1 ] mul -> add
// CHECK-ORDER-NOT: {"hfusion-auto-vectorize-target-
// CHECK-ORDER: {"hfusion-auto-vectorize-target-2"}
// CHECK-ORDER-NOT: {"hfusion-auto-vectorize-target-
// CHECK-ORDER: {"hfusion-auto-vectorize-target-3"}
// CHECK-ORDER: hivm.hir.sync_block_set
// CHECK-ORDER-NOT: {"hfusion-auto-vectorize-target-
// CHECK-ORDER: {"hfusion-auto-vectorize-target-5"}
// CHECK-ORDER-NOT: {"hfusion-auto-vectorize-target-
// CHECK-ORDER: {"hfusion-auto-vectorize-target-6"}
// CHECK-ORDER: transform.sequence

// CHECK-LABEL: func.func @mcf_producer_fuse_into_later_node
// CHECK:         %[[producer:.*]]:2 = scf.for
// CHECK:           arith.addf
// CHECK:           arith.subf
// CHECK:         } {"outlined-loop-target-
// CHECK:         %[[other:.*]]:2 = scf.for
// CHECK:           %[[producer]]#1
// CHECK:           arith.mulf
// CHECK:           arith.addf
// CHECK:         } {"outlined-loop-target-
// CHECK:         hivm.hir.sync_block_set

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @mcf_producer_fuse_into_later_node(%arg0: tensor<8xf32>, %arg1: tensor<8x128xf32>) -> (tensor<8x128xf32>, tensor<8xf32>, tensor<8x128xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %empty_1d = tensor.empty() : tensor<8xf32>
    %empty_2d = tensor.empty() : tensor<8x128xf32>

    // producer -> 8xf32 {"hfusion-auto-vectorize-target-1"}
    %producer = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<8xf32>) outs(%empty_1d : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.addf %in, %in : f32
      linalg.yield %0 : f32
    } -> tensor<8xf32>

    // other -> 8x128xf32 {"hfusion-auto-vectorize-target-2"}
    %other = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<8x128xf32>) outs(%empty_2d : tensor<8x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.mulf %in, %in : f32
      linalg.yield %0 : f32
    } -> tensor<8x128xf32>

    // leaf_1: (producer, other) -> 8x128xf32 {"hfusion-auto-vectorize-target-3"}
    %leaf_1 = linalg.generic {indexing_maps = [#map2, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%producer, %other : tensor<8xf32>, tensor<8x128xf32>) outs(%empty_2d : tensor<8x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.addf %in, %in_0 : f32
      linalg.yield %0 : f32
    } -> tensor<8x128xf32>

    // leaf_2: (producer) -> 8xf32 {"hfusion-auto-vectorize-target-4"}
    %leaf_2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%producer : tensor<8xf32>) outs(%empty_1d : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.subf %in, %in : f32
      linalg.yield %0 : f32
    } -> tensor<8xf32>

    hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15

    // user3: (producer) -> 8xf32
    %user3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%producer : tensor<8xf32>) outs(%empty_1d : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.addf %in, %in : f32
      linalg.yield %0 : f32
    } -> tensor<8xf32>

    // leaf_3: (other, user3) -> 8x128xf32
    %leaf_3 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%other, %user3 : tensor<8x128xf32>, tensor<8xf32>) outs(%empty_2d : tensor<8x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.subf %in, %in_0 : f32
      linalg.yield %0 : f32
    } -> tensor<8x128xf32>
    return %leaf_1, %leaf_2, %leaf_3 : tensor<8x128xf32>, tensor<8xf32>, tensor<8x128xf32>
  }
}
