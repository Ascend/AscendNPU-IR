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
