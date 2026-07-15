// RUN: bishengir-opt %s -hfusion-pre-vectorization-fusion -split-input-file | FileCheck %s

// Existing test: rank-0 fill-like generic → rank-1.
func.func @test_extract_inline() -> tensor<i32> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %69 = bufferization.alloc_tensor() : tensor<i32>
  // CHECK:  %2 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%1 : tensor<1xi32>)
  %res0 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%c0_i32 : i32) outs(%69 : tensor<i32>) {
      ^bb0(%in: i32, %out: i32):
        linalg.yield %in : i32
      } -> tensor<i32>
  return %res0 : tensor<i32>
}

// -----
// Rank-0 non-fill generic (e.g. scalar cmpf from hfusion.compare generalization)
// should also be lifted to rank-1. The rank-0 tensor input gets expand_shape'd
// to tensor<1x...> so both operands are rank-1 with identity maps.
// CHECK-LABEL: func @test_rank0_cmpf_generic_to_1d
// CHECK-NOT: iterator_types = []
// CHECK: tensor.expand_shape {{.*}} : tensor<f16> into tensor<1xf16>
// CHECK: linalg.generic {indexing_maps = [#map{{.*}}, #map{{.*}}], iterator_types = ["parallel"]}
// CHECK-SAME: ins({{.*}} : tensor<1xf16>)
// CHECK-SAME: outs({{.*}} : tensor<1xi1>)
#map_rank0 = affine_map<() -> ()>
func.func @test_rank0_cmpf_generic_to_1d(%arg0: tensor<f16>) -> tensor<i1> {
  %cst = arith.constant 0.000000e+00 : f16
  %init = tensor.empty() : tensor<i1>
  %res = linalg.generic {indexing_maps = [#map_rank0, #map_rank0], iterator_types = []}
                        ins(%arg0 : tensor<f16>) outs(%init : tensor<i1>) {
  ^bb0(%in: f16, %out: i1):
    %cmp = arith.cmpf une, %in, %cst : f16
    linalg.yield %cmp : i1
  } -> tensor<i1>
  return %res : tensor<i1>
}

// -----
// Rank-0 generic with mixed inputs: rank-0 tensor + scalar.
// The tensor input gets expand_shape'd; the scalar stays broadcast.
// CHECK-LABEL: func @test_rank0_mixed_tensor_and_scalar
// CHECK-NOT: iterator_types = []
// CHECK: tensor.expand_shape {{.*}} : tensor<f16> into tensor<1xf16>
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel"]
// CHECK-SAME: tensor<1xf16>
#map_rank0_2 = affine_map<() -> ()>
func.func @test_rank0_mixed_tensor_and_scalar(%arg0: tensor<f16>) -> tensor<f16> {
  %cst = arith.constant 1.000000e+00 : f16
  %init = tensor.empty() : tensor<f16>
  %res = linalg.generic {indexing_maps = [#map_rank0_2, #map_rank0_2, #map_rank0_2],
                          iterator_types = []}
                        ins(%arg0, %cst : tensor<f16>, f16) outs(%init : tensor<f16>) {
  ^bb0(%in1: f16, %in2: f16, %out: f16):
    %add = arith.addf %in1, %in2 : f16
    linalg.yield %add : f16
  } -> tensor<f16>
  return %res : tensor<f16>
}

// -----
// Rank-0 generic with all-scalar inputs (non-fill body with multiple ops).
// No expand_shape needed — scalars use broadcast maps as in the fill case.
// CHECK-LABEL: func @test_rank0_all_scalar_inputs
// CHECK-NOT: iterator_types = []
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel"]
#map_rank0_3 = affine_map<() -> ()>
func.func @test_rank0_all_scalar_inputs() -> tensor<f16> {
  %a = arith.constant 1.000000e+00 : f16
  %b = arith.constant 2.000000e+00 : f16
  %init = tensor.empty() : tensor<f16>
  %res = linalg.generic {indexing_maps = [#map_rank0_3, #map_rank0_3, #map_rank0_3],
                          iterator_types = []}
                        ins(%a, %b : f16, f16) outs(%init : tensor<f16>) {
  ^bb0(%in1: f16, %in2: f16, %out: f16):
    %add = arith.addf %in1, %in2 : f16
    linalg.yield %add : f16
  } -> tensor<f16>
  return %res : tensor<f16>
}