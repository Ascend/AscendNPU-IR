// RUN: bishengir-opt -hfusion-convert-generic-to-named %s -split-input-file -verify-diagnostics | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @test_sqrt_op
func.func @test_sqrt_op(%arg0: tensor<6x4xf32>) -> tensor<6x4xf32> {
  %0 = tensor.empty() : tensor<6x4xf32>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>}
  %1 = linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<6x4xf32>) 
    outs(%0 : tensor<6x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = math.sqrt %in : f32
    linalg.yield %2 : f32
  } -> tensor<6x4xf32>
  return %1 : tensor<6x4xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @test_rsqrt_op
func.func @test_rsqrt_op(%arg0: tensor<6x4xf32>) -> tensor<6x4xf32> {
  %0 = tensor.empty() : tensor<6x4xf32>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>}
  %1 = linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<6x4xf32>) 
    outs(%0 : tensor<6x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = math.rsqrt %in : f32
    linalg.yield %2 : f32
  } -> tensor<6x4xf32>
  return %1 : tensor<6x4xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @test_reciprocal_op
func.func @test_reciprocal_op(%arg0: tensor<6x4xf32>) -> tensor<6x4xf32> {
  %0 = tensor.empty() : tensor<6x4xf32>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>}
  %1 = linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<6x4xf32>) 
    outs(%0 : tensor<6x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %cst = arith.constant 1.000000e+00 : f32
    %2 = arith.divf %cst, %in : f32
    linalg.yield %2 : f32
  } -> tensor<6x4xf32>
  return %1 : tensor<6x4xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @test_relu_op
func.func @test_relu_op(%arg0: tensor<6x4xf32>) -> tensor<6x4xf32> {
  %0 = tensor.empty() : tensor<6x4xf32>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>}
  %1 = linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<6x4xf32>) 
    outs(%0 : tensor<6x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %cst = arith.constant 0.000000e+00 : f32
    %2 = arith.maximumf %cst, %in : f32
    linalg.yield %2 : f32
  } -> tensor<6x4xf32>
  return %1 : tensor<6x4xf32>
}

// -----

// CHECK-LABEL: func.func @test_maximumf_op
func.func @test_maximumf_op(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> tensor<6xf32> {
  %0 = tensor.empty() : tensor<6xf32>
  // CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>}
  %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<6xf32>, tensor<6xf32>) outs(%0 : tensor<6xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %10 = arith.maximumf %in_2, %in : f32
      linalg.yield %10 : f32
    } -> tensor<6xf32>
  return %1 : tensor<6xf32>
}

// -----

// CHECK-LABEL: func.func @test_minimumf_op
func.func @test_minimumf_op(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> tensor<6xf32> {
  %0 = tensor.empty() : tensor<6xf32>
  // CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>}
  %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<6xf32>, tensor<6xf32>) outs(%0 : tensor<6xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %10 = arith.minimumf %in_2, %in : f32
      linalg.yield %10 : f32
    } -> tensor<6xf32>
  return %1 : tensor<6xf32>
}

// -----

// CHECK-LABEL: func.func @test_vxor_op_unary
// CHECK-SAME: (%[[arg0:.*]]: tensor<6x4xi1>)
// CHECK: %[[true:.*]] = arith.constant true
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<vxor>} ins(%[[true]], %[[arg0]]
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @test_vxor_op_unary(%arg0: tensor<6x4xi1>) -> tensor<6x4xi1> {
  %0 = tensor.empty() : tensor<6x4xi1>
  %1 = linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<6x4xi1>) 
    outs(%0 : tensor<6x4xi1>) {
  ^bb0(%in: i1, %out: i1):
    %cst = arith.constant 1 : i1
    %2 = arith.xori %cst, %in : i1
    linalg.yield %2 : i1
  } -> tensor<6x4xi1>
  return %1 : tensor<6x4xi1>
}

// -----

// CHECK-LABEL: func.func @test_vxor_op_binary
// CHECK-SAME: (%[[arg0:.*]]: tensor<6x4xi1>, %[[arg1:.*]]: tensor<6x4xi1>)
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<vxor>} ins(%[[arg0]], %[[arg1]]
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @test_vxor_op_binary(%arg0: tensor<6x4xi1>, %arg1: tensor<6x4xi1>) -> tensor<6x4xi1> {
  %0 = tensor.empty() : tensor<6x4xi1>
  %1 = linalg.generic {
    indexing_maps = [#map, #map, #map], 
    iterator_types = ["parallel", "parallel"]
  } 
  ins(%arg0, %arg1 : tensor<6x4xi1>, tensor<6x4xi1>) 
    outs(%0 : tensor<6x4xi1>) {
  ^bb0(%in: i1, %in_0: i1, %out: i1):
    %2 = arith.xori %in, %in_0 : i1
    linalg.yield %2 : i1
  } -> tensor<6x4xi1>
  return %1 : tensor<6x4xi1>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @test_and_op
func.func @test_and_op(%arg0: tensor<6x4xi32>, %arg1: tensor<6x4xi32>) -> tensor<6x4xi32> {
  %0 = tensor.empty() : tensor<6x4xi32>
  // CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
  %1 = linalg.generic {
    indexing_maps = [#map, #map, #map], 
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<6x4xi32>, tensor<6x4xi32>) 
    outs(%0 : tensor<6x4xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %2 = arith.andi %in, %in_0 : i32
    linalg.yield %2 : i32
  } -> tensor<6x4xi32>
  return %1 : tensor<6x4xi32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @test_or_op
func.func @test_or_op(%arg0: tensor<6x4xi32>, %arg1: tensor<6x4xi32>) -> tensor<6x4xi32> {
  %0 = tensor.empty() : tensor<6x4xi32>
  // CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>}
  %1 = linalg.generic {
    indexing_maps = [#map, #map, #map], 
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<6x4xi32>, tensor<6x4xi32>) 
    outs(%0 : tensor<6x4xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %2 = arith.ori %in, %in_0 : i32
    linalg.yield %2 : i32
  } -> tensor<6x4xi32>
  return %1 : tensor<6x4xi32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @test_reciprocal_outer_vector_scalar_op
func.func @test_reciprocal_outer_vector_scalar_op(%arg0: tensor<6x4xf32>) -> tensor<6x4xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %0 = tensor.empty() : tensor<6x4xf32>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>}
  %1 = linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<6x4xf32>) 
    outs(%0 : tensor<6x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.divf %cst, %in : f32
    linalg.yield %2 : f32
  } -> tensor<6x4xf32>
  return %1 : tensor<6x4xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
// CHECK-LABEL: func.func @test_reciprocal_inline_vector_scalar_op
func.func @test_reciprocal_inline_vector_scalar_op(%arg0: tensor<6x4xf32>) -> tensor<6x4xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %0 = tensor.empty() : tensor<6x4xf32>
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>}
  %1 = linalg.generic {
    indexing_maps = [#map1, #map, #map], 
    iterator_types = ["parallel", "parallel"]
  } ins(%cst, %arg0 : f32, tensor<6x4xf32>) 
    outs(%0 : tensor<6x4xf32>) {
  ^bb0(%in_0: f32, %in: f32, %out: f32):
    %2 = arith.divf %in_0, %in : f32
    linalg.yield %2 : f32
  } -> tensor<6x4xf32>
  return %1 : tensor<6x4xf32>
}
