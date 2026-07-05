// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2 -split-input-file | FileCheck %s

// -----
// many_users_no_fusion_opportunity: producer's two consumers fuse into different
// loops (scf.for barrier). 
// producer stays standalone loop(no fusion opportunity).
// CHECK-LABEL: func.func @many_users_no_fusion_opportunity
// CHECK: "outlined-loop-target-3"
// CHECK: "outlined-loop-target-1"
// CHECK: "outlined-loop-target-2"
// CHECK-NOT: "outlined-loop-target-4"

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
#map5 = affine_map<(d0) -> ()>
module {
  func.func @many_users_no_fusion_opportunity(%arg0: tensor<1x16xi32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> (tensor<1xf32>, tensor<1xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    
    // producer, result used by two consumers
    %producer = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1x16xi32>, tensor<1xf32>) outs(%1 : tensor<1xf32>) {
    ^bb0(%in: i32, %in_0: f32, %out: f32):
      %7 = arith.sitofp %in : i32 to f32
      %8 = arith.addf %7, %in_0 : f32
      %9 = arith.addf %8, %out : f32
      linalg.yield %9 : f32
    } -> tensor<1xf32>
    %3 = tensor.empty() : tensor<1xf32>
    %user0 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%producer, %arg1 : tensor<1xf32>, tensor<1xf32>) outs(%3 : tensor<1xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.addf %in, %in_0 : f32
      linalg.yield %7 : f32
    } -> tensor<1xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c1 step %c1 {
      scf.yield
    }
    %5 = tensor.empty() : tensor<1xf32>
    %user1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%producer, %arg2 : tensor<1xf32>, tensor<1xf32>) outs(%5 : tensor<1xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.addf %in, %in_0 : f32
      linalg.yield %7 : f32
    } -> tensor<1xf32>
    return %user0, %user1 : tensor<1xf32>, tensor<1xf32>
  }
}

// -----
// many_users_has_fusion_opportunity: both consumers fuse into same loop (no
// barrier). 
// producer fuses into that fused loop(has fusion opportunity).
// CHECK-LABEL: func.func @many_users_has_fusion_opportunity
// CHECK: "outlined-loop-target-2"
// CHECK: "outlined-loop-target-1"
// CHECK-NOT: "outlined-loop-target-3"

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
module {
  func.func @many_users_has_fusion_opportunity(%arg0: tensor<1x16xi32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> (tensor<1xf32>, tensor<1xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %producer = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1x16xi32>, tensor<1xf32>) outs(%1 : tensor<1xf32>) {
    ^bb0(%in: i32, %in_0: f32, %out: f32):
      %7 = arith.sitofp %in : i32 to f32
      %8 = arith.addf %7, %in_0 : f32
      %9 = arith.addf %8, %out : f32
      linalg.yield %9 : f32
    } -> tensor<1xf32>
    %3 = tensor.empty() : tensor<1xf32>
    %user0 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%producer, %arg1 : tensor<1xf32>, tensor<1xf32>) outs(%3 : tensor<1xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.addf %in, %in_0 : f32
      linalg.yield %7 : f32
    } -> tensor<1xf32>
    %5 = tensor.empty() : tensor<1xf32>
    %user1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%producer, %arg2 : tensor<1xf32>, tensor<1xf32>) outs(%5 : tensor<1xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.addf %in, %in_0 : f32
      linalg.yield %7 : f32
    } -> tensor<1xf32>
    return %user0, %user1 : tensor<1xf32>, tensor<1xf32>
  }
}

// -----
// mixed_reduction_parallel_users_single_tile_fusion: producer's consumers are
// mixed reduction/parallel linalg users, but the producer fits in a single
// vector tile, so fusion is still safe.
// CHECK-LABEL: func.func @mixed_reduction_parallel_users_single_tile_fusion
// CHECK: "outlined-loop-target-1"
// CHECK-NOT: "outlined-loop-target-2"

#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @mixed_reduction_parallel_users_single_tile_fusion(%arg0: tensor<1x64xf32>, %arg1: tensor<1xf32>) -> tensor<1x64xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %tmp = tensor.empty() : tensor<1x64xf32>
    %producer = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x64xf32>) outs(%tmp : tensor<1x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = math.exp %in : f32
      linalg.yield %v : f32
    } -> tensor<1x64xf32>

    %sum = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%producer : tensor<1x64xf32>) outs(%arg1 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.addf %in, %out : f32
      linalg.yield %v : f32
    } -> tensor<1xf32>

    %out_init = tensor.empty() : tensor<1x64xf32>
    %user = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%producer, %sum : tensor<1x64xf32>, tensor<1xf32>) outs(%out_init : tensor<1x64xf32>) {
    ^bb0(%in: f32, %sum_in: f32, %out: f32):
      %v = arith.divf %in, %sum_in : f32
      linalg.yield %v : f32
    } -> tensor<1x64xf32>

    return %user : tensor<1x64xf32>
  }
}

// -----
// mixed_reduction_parallel_users_row_reduction_fusion: producer's consumers
// are mixed reduction/parallel linalg users and the reduction is a row
// reduction (last-axis reduction with remaining parallel axes), so fusion is
// allowed.
// CHECK-LABEL: func.func @mixed_reduction_parallel_users_row_reduction_fusion
// CHECK: "outlined-loop-target-1"
// CHECK-NOT: "outlined-loop-target-2"

#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @mixed_reduction_parallel_users_row_reduction_fusion(%arg0: tensor<64x128xf32>, %arg1: tensor<64xf32>) -> tensor<64x128xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %tmp = tensor.empty() : tensor<64x128xf32>
    %producer = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<64x128xf32>) outs(%tmp : tensor<64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = math.exp %in : f32
      linalg.yield %v : f32
    } -> tensor<64x128xf32>

    %sum = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%producer : tensor<64x128xf32>) outs(%arg1 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.addf %in, %out : f32
      linalg.yield %v : f32
    } -> tensor<64xf32>

    %out_init = tensor.empty() : tensor<64x128xf32>
    %user = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%producer, %sum : tensor<64x128xf32>, tensor<64xf32>) outs(%out_init : tensor<64x128xf32>) {
    ^bb0(%in: f32, %sum_in: f32, %out: f32):
      %v = arith.divf %in, %sum_in : f32
      linalg.yield %v : f32
    } -> tensor<64x128xf32>

    return %user : tensor<64x128xf32>
  }
}

// -----
// mixed_reduction_parallel_users_no_fusion: producer's consumers are in the
// same fusion region, but the reduction is over the entire 1D producer domain.
// The producer must stay standalone to preserve full-tensor semantics.
// CHECK-LABEL: func.func @mixed_reduction_parallel_users_no_fusion
// CHECK: "outlined-loop-target-2"
// CHECK: "outlined-loop-target-1"
// CHECK-NOT: "outlined-loop-target-3"

#map2 = affine_map<(d0) -> (d0)>
#map5 = affine_map<(d0) -> ()>
module {
  func.func @mixed_reduction_parallel_users_no_fusion(%arg0: tensor<128xf32>, %arg1: tensor<f32>) -> tensor<128xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %tmp = tensor.empty() : tensor<128xf32>
    %producer = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%arg0 : tensor<128xf32>) outs(%tmp : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = math.exp %in : f32
      linalg.yield %v : f32
    } -> tensor<128xf32>

    %sum = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["reduction"]} ins(%producer : tensor<128xf32>) outs(%arg1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.addf %in, %out : f32
      linalg.yield %v : f32
    } -> tensor<f32>

    %out_init = tensor.empty() : tensor<128xf32>
    %user = linalg.generic {indexing_maps = [#map2, #map5, #map2], iterator_types = ["parallel"]} ins(%producer, %sum : tensor<128xf32>, tensor<f32>) outs(%out_init : tensor<128xf32>) {
    ^bb0(%in: f32, %sum_in: f32, %out: f32):
      %v = arith.divf %in, %sum_in : f32
      linalg.yield %v : f32
    } -> tensor<128xf32>

    return %user : tensor<128xf32>
  }
}
