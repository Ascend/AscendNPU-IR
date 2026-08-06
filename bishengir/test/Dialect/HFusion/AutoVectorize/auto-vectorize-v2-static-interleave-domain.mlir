// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2 --outline-vector-function -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @deinterleave_uses_result_iteration_domain(
// CHECK: func.call @deinterleave_uses_result_iteration_domain_outlined_vf_0
// CHECK: func.call @deinterleave_uses_result_iteration_domain_outlined_vf_1
func.func @deinterleave_uses_result_iteration_domain(
    %arg0: tensor<2xi32>) -> (tensor<2xi32>, tensor<1xi32>)
    attributes {
      hacc.function_kind = #hacc.function_kind<DEVICE>,
      parallel_mode = "simd"
    } {
  %c0_i32 = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<2xi32>
  %zero = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } outs(%empty : tensor<2xi32>) {
  ^bb0(%out: i32):
    linalg.yield %c0_i32 : i32
  } -> tensor<2xi32>
  %deinterleave = hfusion.deinterleave %arg0 channel<0>
      : tensor<2xi32> -> tensor<1xi32>
  return %zero, %deinterleave : tensor<2xi32>, tensor<1xi32>
}

// -----

// CHECK-LABEL: func.func @interleave_uses_result_iteration_domain(
// CHECK: func.call @interleave_uses_result_iteration_domain_outlined_vf_0
// CHECK: func.call @interleave_uses_result_iteration_domain_outlined_vf_1
func.func @interleave_uses_result_iteration_domain(
    %arg0: tensor<2xi32>, %arg1: tensor<2xi32>)
    -> (tensor<2xi32>, tensor<4xi32>)
    attributes {
      hacc.function_kind = #hacc.function_kind<DEVICE>,
      parallel_mode = "simd"
    } {
  %c0_i32 = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<2xi32>
  %zero = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } outs(%empty : tensor<2xi32>) {
  ^bb0(%out: i32):
    linalg.yield %c0_i32 : i32
  } -> tensor<2xi32>
  %interleave = hfusion.interleave %arg0, %arg1
      : tensor<2xi32>, tensor<2xi32> -> tensor<4xi32>
  return %zero, %interleave : tensor<2xi32>, tensor<4xi32>
}
