// RUN: bishengir-opt -hfusion-convert-generic-to-named %s -split-input-file -verify-diagnostics | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @test_exp_op
func.func @test_exp_op(%arg0: tensor<6x4xf32>) -> tensor<6x4xf32> {
  %0 = tensor.empty() : tensor<6x4xf32>
  // CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
  %1 = linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<6x4xf32>) 
    outs(%0 : tensor<6x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = math.exp %in : f32
    linalg.yield %2 : f32
  } -> tensor<6x4xf32>
  return %1 : tensor<6x4xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @test_abs_op
func.func @test_abs_op(%arg0: tensor<6x4xf32>) -> tensor<6x4xf32> {
  %0 = tensor.empty() : tensor<6x4xf32>
  // CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
  %1 = linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<6x4xf32>) 
    outs(%0 : tensor<6x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = math.absf %in : f32
    linalg.yield %2 : f32
  } -> tensor<6x4xf32>
  return %1 : tensor<6x4xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @test_log_op
func.func @test_log_op(%arg0: tensor<6x4xf32>) -> tensor<6x4xf32> {
  %0 = tensor.empty() : tensor<6x4xf32>
  // CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<log>}
  %1 = linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<6x4xf32>) 
    outs(%0 : tensor<6x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = math.log %in : f32
    linalg.yield %2 : f32
  } -> tensor<6x4xf32>
  return %1 : tensor<6x4xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @test_add_op
func.func @test_add_op(%arg0: tensor<6x6xf32>, %arg1: tensor<6x6xf32>) -> tensor<6x6xf32> {
  %0 = tensor.empty() : tensor<6x6xf32>
  // CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
  %1 = linalg.generic {
    indexing_maps = [#map, #map, #map], 
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<6x6xf32>, tensor<6x6xf32>)
    outs(%0 : tensor<6x6xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %2 = arith.addf %in, %in_0 : f32
    linalg.yield %2 : f32
  } -> tensor<6x6xf32>
  return %1 : tensor<6x6xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @test_mul_op
func.func @test_mul_op(%arg0: tensor<6x6xf32>, %arg1: tensor<6x6xf32>) -> tensor<6x6xf32> {
  %0 = tensor.empty() : tensor<6x6xf32>
  // CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
  %1 = linalg.generic {
    indexing_maps = [#map, #map, #map], 
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<6x6xf32>, tensor<6x6xf32>)
    outs(%0 : tensor<6x6xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %2 = arith.mulf %in, %in_0 : f32
    linalg.yield %2 : f32
  } -> tensor<6x6xf32>
  return %1 : tensor<6x6xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @test_sub_op
func.func @test_sub_op(%arg0: tensor<6x6xf32>, %arg1: tensor<6x6xf32>) -> tensor<6x6xf32> {
  %0 = tensor.empty() : tensor<6x6xf32>
  // CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
  %1 = linalg.generic {
    indexing_maps = [#map, #map, #map], 
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<6x6xf32>, tensor<6x6xf32>)
    outs(%0 : tensor<6x6xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %2 = arith.subf %in, %in_0 : f32
    linalg.yield %2 : f32
  } -> tensor<6x6xf32>
  return %1 : tensor<6x6xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @test_div_op
func.func @test_div_op(%arg0: tensor<6x6xf32>, %arg1: tensor<6x6xf32>) -> tensor<6x6xf32> {
  %0 = tensor.empty() : tensor<6x6xf32>
  // CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
  %1 = linalg.generic {
    indexing_maps = [#map, #map, #map], 
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<6x6xf32>, tensor<6x6xf32>)
    outs(%0 : tensor<6x6xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %2 = arith.divf %in, %in_0 : f32
    linalg.yield %2 : f32
  } -> tensor<6x6xf32>
  return %1 : tensor<6x6xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @test_max_unsigned_op
func.func @test_max_unsigned_op(%arg0: tensor<6x4xi32>, %arg1: tensor<6x4xi32>) -> tensor<6x4xi32> {
  %0 = tensor.empty() : tensor<6x4xi32>
  // CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<max_unsigned>}
  %1 = linalg.generic {
    indexing_maps = [#map, #map, #map], 
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<6x4xi32>, tensor<6x4xi32>) 
    outs(%0 : tensor<6x4xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %2 = arith.maxui %in, %in_0 : i32
    linalg.yield %2 : i32
  } -> tensor<6x4xi32>
  return %1 : tensor<6x4xi32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @test_max_signed_op
func.func @test_max_signed_op(%arg0: tensor<6x4xi32>, %arg1: tensor<6x4xi32>) -> tensor<6x4xi32> {
  %0 = tensor.empty() : tensor<6x4xi32>
  // CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>}
  %1 = linalg.generic {
    indexing_maps = [#map, #map, #map], 
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<6x4xi32>, tensor<6x4xi32>) 
    outs(%0 : tensor<6x4xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %2 = arith.maxsi %in, %in_0 : i32
    linalg.yield %2 : i32
  } -> tensor<6x4xi32>
  return %1 : tensor<6x4xi32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @test_min_unsigned_op
func.func @test_min_unsigned_op(%arg0: tensor<6x4xi32>, %arg1: tensor<6x4xi32>) -> tensor<6x4xi32> {
  %0 = tensor.empty() : tensor<6x4xi32>
  // CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<min_unsigned>}
  %1 = linalg.generic {
    indexing_maps = [#map, #map, #map], 
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<6x4xi32>, tensor<6x4xi32>) 
    outs(%0 : tensor<6x4xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %2 = arith.minui %in, %in_0 : i32
    linalg.yield %2 : i32
  } -> tensor<6x4xi32>
  return %1 : tensor<6x4xi32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @test_min_signed_op
func.func @test_min_signed_op(%arg0: tensor<6x4xi32>, %arg1: tensor<6x4xi32>) -> tensor<6x4xi32> {
  %0 = tensor.empty() : tensor<6x4xi32>
  // CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>}
  %1 = linalg.generic {
    indexing_maps = [#map, #map, #map], 
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<6x4xi32>, tensor<6x4xi32>) 
    outs(%0 : tensor<6x4xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %2 = arith.minsi %in, %in_0 : i32
    linalg.yield %2 : i32
  } -> tensor<6x4xi32>
  return %1 : tensor<6x4xi32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @test_div_outer_vector_scalar_op
func.func @test_div_outer_vector_scalar_op(%arg0: tensor<6x4xf32>) -> tensor<6x4xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<6x4xf32>
  // CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
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
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<6x4xf32>
  // CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
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

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
// CHECK-LABEL: func.func @test_add_outer_vector_scalar_op
func.func @test_add_outer_vector_scalar_ops(%arg0: memref<6x6xf32>, %arg1: memref<6x6xf32>) {
  %cst = arith.constant 1.000000e+00 : f32
  // CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
  linalg.generic {
    indexing_maps = [#map, #map1, #map], 
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %cst : memref<6x6xf32>, f32) 
    outs(%arg1 : memref<6x6xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %0 = arith.addf %in, %in_0 : f32
    linalg.yield %0 : f32
  }
  return
}

// -----

// CHECK-LABEL: func.func @test_i16_generic_to_named
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @test_i16_generic_to_named(%arg0: tensor<16x16xi16>, %arg1: tensor<16x16xi16>) -> tensor<16x16xi16> {
    %0 = tensor.empty() : tensor<16x16xi16>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<16x16xi16>, tensor<16x16xi16>) outs(%0 : tensor<16x16xi16>) {
    ^bb0(%in: i16, %in_0: i16, %out: i16):
      %2 = arith.addi %in_0, %in : i16
      linalg.yield %2 : i16
    } -> tensor<16x16xi16>
    return %1 : tensor<16x16xi16>
  }
}

// -----

// CHECK-LABEL: func.func @test_i64_generic_to_named
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @test_i64_generic_to_named(%arg0: tensor<16x16xi64>, %arg1: tensor<16x16xi64>) -> tensor<16x16xi64> {
    %0 = tensor.empty() : tensor<16x16xi64>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<16x16xi64>, tensor<16x16xi64>) outs(%0 : tensor<16x16xi64>) {
    ^bb0(%in: i64, %in_0: i64, %out: i64):
      %2 = arith.addi %in_0, %in : i64
      linalg.yield %2 : i64
    } -> tensor<16x16xi64>
    return %1 : tensor<16x16xi64>
  }
}

// -----

// CHECK-LABEL: func.func @test_i64_generic_to_named_xori
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<vxor>} 
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @test_i64_generic_to_named_xori(%arg0: tensor<16x16xi64>, %arg1: tensor<16x16xi64>) -> tensor<16x16xi64> {
    %0 = tensor.empty() : tensor<16x16xi64>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<16x16xi64>, tensor<16x16xi64>) outs(%0 : tensor<16x16xi64>) {
    ^bb0(%in: i64, %in_0: i64, %out: i64):
      %2 = arith.xori %in_0, %in : i64
      linalg.yield %2 : i64
    } -> tensor<16x16xi64>
    return %1 : tensor<16x16xi64>
  }
}
