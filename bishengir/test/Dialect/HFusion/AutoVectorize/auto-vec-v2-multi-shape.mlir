// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2 --outline-vector-function 2>&1 | FileCheck %s

// CHECK-NOT: operations cannot be fused
// CHECK-NOT: AutoVectorizeV2 failed;

#map = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> ()>
#map6 = affine_map<(d0, d1) -> (0, d1)>

module {
  func.func @test_broadcast_add_cmp(%arg0: tensor<1x32xi64>, %arg1: tensor<16x32xi32>, %arg2: tensor<1x32xi32>) -> (tensor<16x32xi64>, tensor<1x32xi1>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>, mix_mode = "aiv", parallel_mode = "simd"} {
    %c0 = arith.constant 0 : i32
    %empty_0 = tensor.empty() : tensor<16x32xi64>
    // broadcast 1x32 -> 16x32, then add (extsi i32->i64)
    %add = linalg.generic {indexing_maps = [#map6, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x32xi64>, tensor<16x32xi32>) outs(%empty_0 : tensor<16x32xi64>) {
    ^bb0(%in: i64, %in_10: i32, %out: i64):
      %0 = arith.extsi %in_10 {round_mode = #hfusion.round_mode<rint>} : i32 to i64
      %1 = arith.addi %in, %0 : i64
      linalg.yield %1 : i64
    } -> tensor<16x32xi64>
    %empty_1 = tensor.empty() : tensor<1x32xi1>
    // scalar compare -> i1 mask
    %cmp = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%arg2, %c0 : tensor<1x32xi32>, i32) outs(%empty_1 : tensor<1x32xi1>) {
    ^bb0(%in: i32, %in_10: i32, %out: i1):
      %0 = arith.cmpi slt, %in, %in_10 : i32
      linalg.yield %0 : i1
    } -> tensor<1x32xi1>
    return %add, %cmp : tensor<16x32xi64>, tensor<1x32xi1>
  }
}
