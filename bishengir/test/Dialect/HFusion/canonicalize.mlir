// RUN: bishengir-opt %s -canonicalize-ext -split-input-file | FileCheck %s

// CHECK-LABEL: @fuse_consecutive_reduce_ops_success
// linalg.reduce ins({{.*}} : tensor<1x4x2047x2047xf32>) outs({{.*}} : tensor<2047xf32>) dimensions = {{\[}}0, 1, 2]
func.func @fuse_consecutive_reduce_ops_success(%arg0: tensor<4x2047x2047xf32>, %arg1: tensor<1x4x2047x2047xf32>) -> tensor<1x1x2047xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 4.8851978505129456E-4 : f64
  %expanded = tensor.expand_shape %arg0 [[0, 1], [2], [3]] output_shape [1, 4, 2047, 2047] : tensor<4x2047x2047xf32> into tensor<1x4x2047x2047xf32>
  %0 = tensor.empty() : tensor<1x4x2047x2047xf32>
  %1 = arith.truncf %cst_0 : f64 to f32
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expanded, %1 : tensor<1x4x2047x2047xf32>, f32) outs(%0 : tensor<1x4x2047x2047xf32>) -> tensor<1x4x2047x2047xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %arg1 : tensor<1x4x2047x2047xf32>, tensor<1x4x2047x2047xf32>) outs(%0 : tensor<1x4x2047x2047xf32>) -> tensor<1x4x2047x2047xf32>
  %4 = tensor.empty() : tensor<2047x2047xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<2047x2047xf32>) -> tensor<2047x2047xf32>
  %reduced = linalg.reduce ins(%3 : tensor<1x4x2047x2047xf32>) outs(%5 : tensor<2047x2047xf32>) dimensions = [0, 1] 
    (%in: f32, %init: f32) {
      %8 = arith.addf %in, %init : f32
      linalg.yield %8 : f32
    }
  %6 = tensor.empty() : tensor<2047xf32>
  %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<2047xf32>) -> tensor<2047xf32>
  %reduced_1 = linalg.reduce ins(%reduced : tensor<2047x2047xf32>) outs(%7 : tensor<2047xf32>) dimensions = [0] 
    (%in: f32, %init: f32) {
      %8 = arith.addf %in, %init : f32
      linalg.yield %8 : f32
    }
  %expanded_2 = tensor.expand_shape %reduced_1 [[0, 1, 2]] output_shape [1, 1, 2047] : tensor<2047xf32> into tensor<1x1x2047xf32>
  return %expanded_2 : tensor<1x1x2047xf32>
}

// -----

// CHECK-LABEL: @fuse_consecutive_reduce_ops_with_diff_regions_fail
// CHECK: linalg.reduce ins({{.*}} : tensor<1x4x2047x2047xf32>) outs({{.*}} : tensor<2047x2047xf32>) dimensions = {{\[}}0, 1]
// CHECK: linalg.reduce ins({{.*}} : tensor<2047x2047xf32>) outs({{.*}} : tensor<2047xf32>) dimensions = {{\[}}0]
func.func @fuse_consecutive_reduce_ops_with_diff_regions_fail(%arg0: tensor<4x2047x2047xf32>, %arg1: tensor<1x4x2047x2047xf32>) -> tensor<1x1x2047xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 4.8851978505129456E-4 : f64
  %expanded = tensor.expand_shape %arg0 [[0, 1], [2], [3]] output_shape [1, 4, 2047, 2047] : tensor<4x2047x2047xf32> into tensor<1x4x2047x2047xf32>
  %0 = tensor.empty() : tensor<1x4x2047x2047xf32>
  %1 = arith.truncf %cst_0 : f64 to f32
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expanded, %1 : tensor<1x4x2047x2047xf32>, f32) outs(%0 : tensor<1x4x2047x2047xf32>) -> tensor<1x4x2047x2047xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %arg1 : tensor<1x4x2047x2047xf32>, tensor<1x4x2047x2047xf32>) outs(%0 : tensor<1x4x2047x2047xf32>) -> tensor<1x4x2047x2047xf32>
  %4 = tensor.empty() : tensor<2047x2047xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<2047x2047xf32>) -> tensor<2047x2047xf32>
  %reduced = linalg.reduce ins(%3 : tensor<1x4x2047x2047xf32>) outs(%5 : tensor<2047x2047xf32>) dimensions = [0, 1] 
    (%in: f32, %init: f32) {
      %8 = arith.addf %in, %init : f32
      linalg.yield %8 : f32
    }
  %6 = tensor.empty() : tensor<2047xf32>
  %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<2047xf32>) -> tensor<2047xf32>
  %reduced_1 = linalg.reduce ins(%reduced : tensor<2047x2047xf32>) outs(%7 : tensor<2047xf32>) dimensions = [0] 
    (%in: f32, %init: f32) {
      %8 = arith.mulf %in, %init : f32
      linalg.yield %8 : f32
    }
  %expanded_2 = tensor.expand_shape %reduced_1 [[0, 1, 2]] output_shape [1, 1, 2047] : tensor<2047xf32> into tensor<1x1x2047xf32>
  return %expanded_2 : tensor<1x1x2047xf32>
}

// -----

// CHECK-LABEL: @empty_cast_fold(
// CHECK: tensor.empty() : tensor<24x128x128x96xf32>
// CHECK-NOT: hfusion.cast
func.func @empty_cast_fold() -> tensor<24x128x128x96xf32> {
  %0 = tensor.empty() : tensor<24x128x128x96xbf16>
  %1 = tensor.empty() : tensor<24x128x128x96xf32>
  %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%0 : tensor<24x128x128x96xbf16>) outs(%1 : tensor<24x128x128x96xf32>) -> tensor<24x128x128x96xf32>
  return %2 : tensor<24x128x128x96xf32>
}

// CHECK-LABEL: func.func @test_cast_rank0
func.func @test_cast_rank0() -> tensor<bf16> {
  //CHECK: %[[CST:.*]] = arith.constant dense<1.000000e+00> : tensor<bf16>
  //CHECK: return %[[CST]] : tensor<bf16>
  %cst = arith.constant dense<1> : tensor<i64>
  %5 = tensor.empty() : tensor<bf16>
  %6 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%cst : tensor<i64>) outs(%5 : tensor<bf16>) -> tensor<bf16>
  return %6 : tensor<bf16>
}

// CHECK-LABEL: func.func @test_minimumf_rank0
func.func @test_minimumf_rank0() -> tensor<f16> {
  //CHECK: %[[CST:.*]] = arith.constant dense<1.000000e+00> : tensor<f16>
  //CHECK: return %[[CST]] : tensor<f16>
  %cst0 = arith.constant dense<1.0> : tensor<f16>
  %cst1 = arith.constant dense<2.0> : tensor<f16>
  %0 = tensor.empty() : tensor<f16>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%cst0, %cst1 : tensor<f16>, tensor<f16>) outs(%0 : tensor<f16>) -> tensor<f16>
  return %1 : tensor<f16>
}

// CHECK-LABEL: func.func @test_sqrt_rank0
func.func @test_sqrt_rank0() -> tensor<f32> {
  //CHECK: %[[CST:.*]] = arith.constant dense<4.000000e+00> : tensor<f32>
  //CHECK: return %[[CST]] : tensor<f32>
  %cst0 = arith.constant dense<16.0> : tensor<f32>
  %0 = tensor.empty() : tensor<f32>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%cst0 : tensor<f32>) outs(%0 : tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// CHECK-LABEL: func.func @test_log_rank0
func.func @test_log_rank0() -> tensor<f16> {
  //CHECK: %[[CST:.*]] = arith.constant dense<4.000000e+00> : tensor<f16>
  //CHECK: %[[RES:.*]] = math.log %[[CST]] : tensor<f16>
  //CHECK: return %[[RES]] : tensor<f16>
  %cst0 = arith.constant dense<4.0> : tensor<f16>
  %0 = tensor.empty() : tensor<f16>
  %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%cst0 : tensor<f16>) outs(%0: tensor<f16>) -> tensor<f16>
  return %1 : tensor<f16>
}

// CHECK-LABEL: func.func @test_extf_dense_float32_with_rank0
// CHECK: %[[CST:.*]] = arith.constant 5.000000e-01 : f32
// CHECK: linalg.fill ins(%[[CST]] : f32)
func.func @test_extf_dense_float32_with_rank0() -> tensor<16xf32> {
  %cst = arith.constant dense<0.5> : tensor<bf16>
  %0 = arith.extf %cst : tensor<bf16> to tensor<f32>
  %dst = tensor.empty() : tensor<16xf32>
  %broadcasted = linalg.broadcast ins(%0 : tensor<f32>) outs(%dst : tensor<16xf32>) dimensions = [0]
  return %broadcasted : tensor<16xf32>
}

// CHECK-LABEL: func.func @test_extf_dense_float64_with_rank0
// CHECK: %[[CST:.*]] = arith.constant 5.000000e-01 : f64
// CHECK: linalg.fill ins(%[[CST]] : f64)
func.func @test_extf_dense_float64_with_rank0() -> tensor<16xf64> {
  %cst = arith.constant dense<0.5> : tensor<bf16>
  %0 = arith.extf %cst : tensor<bf16> to tensor<f64>
  %dst = tensor.empty() : tensor<16xf64>
  %broadcasted = linalg.broadcast ins(%0 : tensor<f64>) outs(%dst : tensor<16xf64>) dimensions = [0]
  return %broadcasted : tensor<16xf64>
}

// CHECK-LABEL: func.func @test_extf_dense_float128
// CHECK: %[[CST:.*]] = arith.constant 5.000000e-01 : f128
// CHECK: linalg.fill ins(%[[CST]] : f128) outs(%{{.*}} : tensor<16xf128>) -> tensor<16xf128>
func.func @test_extf_dense_float128() -> tensor<16xf128> {
  %cst = arith.constant dense<0.5> : tensor<bf16>
  %0 = arith.extf %cst : tensor<bf16> to tensor<f128>
  %dst = tensor.empty() : tensor<16xf128>
  %broadcasted = linalg.broadcast ins(%0 : tensor<f128>) outs(%dst : tensor<16xf128>) dimensions = [0]
  return %broadcasted : tensor<16xf128>
}

// CHECK-LABEL: func.func @test_cst_folding
// CHECK: arith.constant dense<2.000000e+00> : tensor<16xf32>
func.func @test_cst_folding() -> tensor<16xf32> {
  %cst = arith.constant dense<2> : tensor<16xi64>
  %0 = tensor.empty() : tensor<16xf32>
  %1 = hfusion.cast ins(%cst : tensor<16xi64>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>    
  return %1 : tensor<16xf32>
}

// CHECK-LABEL: func.func @test_cst_folding_same_type
// CHECK: arith.constant dense<3.000000e+00> : tensor<16xf32>
func.func @test_cst_folding_same_type() -> tensor<16xf32> {
  %cst = arith.constant dense<2.8> : tensor<16xf32>
  %0 = tensor.empty() : tensor<16xf32>
  %1 = hfusion.cast ins(%cst : tensor<16xf32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>    
  return %1 : tensor<16xf32>
}


// -----

// CHECK-LABEL: func.func @inline_splat_dense
// CHECK-DAG: %[[cst_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[cst_1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins({{.*}}, %[[cst_0]]
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[cst_1]]
func.func @inline_splat_dense(%arg0: tensor<1x4x2047x2047xf32>, %arg1: tensor<f32>) -> (tensor<1x4x2047x2047xf32>, tensor<f32>) 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %cst = arith.constant dense<0.000000e+00> : tensor<1x4x2047x2047xf32>
  %cst_0 = arith.constant dense<1.000000e+00> : tensor<f32>
  %0 = tensor.empty() : tensor<1x4x2047x2047xf32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %cst : tensor<1x4x2047x2047xf32>, tensor<1x4x2047x2047xf32>) outs(%0 : tensor<1x4x2047x2047xf32>) -> tensor<1x4x2047x2047xf32>
  %2 = tensor.empty() : tensor<f32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%cst_0, %arg1 : tensor<f32>, tensor<f32>) 
                                                             outs(%2 : tensor<f32>) -> tensor<f32>
  return %1, %3 : tensor<1x4x2047x2047xf32>, tensor<f32>
}

// -----

// CHECK-LABEL: func.func @inline_splat_dense_to_hfusion_compare
// CHECK: %[[cst:.*]] = arith.constant 9.99999997E-7 : f32
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>} ins({{.*}}, %[[cst]] : tensor<1x2047x1xf32>, f32) outs({{.*}} : tensor<1x2047x1xi1>)
func.func @inline_splat_dense_to_hfusion_compare(%arg0: tensor<1x2047x1xf32>, %arg1: tensor<1x2047x1xf32>, 
                                                 %arg2: tensor<1x2047x1xf32>, %arg3: tensor<1x2047x1xf32>) -> tensor<1x2047x1xf32> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %cst = arith.constant dense<9.99999997E-7> : tensor<1x2047x1xf32>
  %0 = tensor.empty() : tensor<1x2047x1xi1>
  %1 = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>} ins(%arg0, %cst : tensor<1x2047x1xf32>, tensor<1x2047x1xf32>) outs(%0 : tensor<1x2047x1xi1>) -> tensor<1x2047x1xi1>
  %2 = tensor.empty() : tensor<1x2047x1xf32>
  %3 = hfusion.select ins(%1, %arg1, %arg2 : tensor<1x2047x1xi1>, tensor<1x2047x1xf32>, tensor<1x2047x1xf32>) outs(%2 : tensor<1x2047x1xf32>) -> tensor<1x2047x1xf32>
  %4 = tensor.empty() : tensor<1x2047x1xf32>
  %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg3, %3 : tensor<1x2047x1xf32>, tensor<1x2047x1xf32>) outs(%4 : tensor<1x2047x1xf32>) -> tensor<1x2047x1xf32>
  return %5 : tensor<1x2047x1xf32>
}


// -----

// CHECK-LABEL: func.func @inline_splat_constant_to_generic_region
// CHECK-DAG: %[[cst_0:.*]] = arith.constant dense<0.000000e+00> : tensor<1x4x2047x2047xf32>
// CHECK-DAG: %[[cst_1:.*]] = arith.constant dense<8.000000e+00> : tensor<f32>
// CHECK: return %[[cst_0]], %[[cst_1]] : tensor<1x4x2047x2047xf32>, tensor<f32>
func.func @inline_splat_constant_to_generic_region(%arg0: tensor<1x4x2047x2047xf32>, %arg1: tensor<f32>) -> (tensor<1x4x2047x2047xf32>, tensor<f32>) 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %cst = arith.constant dense<0.000000e+00> : tensor<1x4x2047x2047xf32>
  %cst_0 = arith.constant dense<2.000000e+00> : tensor<f32>
  %cst_1 = arith.constant dense<4.000000e+00> : tensor<1x4x2047x2047xf32>
  %cst_2 = arith.constant dense<3.000000e+00> : tensor<f32>
  %0 = tensor.empty() : tensor<1x4x2047x2047xf32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%cst_1, %cst : tensor<1x4x2047x2047xf32>, tensor<1x4x2047x2047xf32>) outs(%0 : tensor<1x4x2047x2047xf32>) -> tensor<1x4x2047x2047xf32>
  %2 = tensor.empty() : tensor<f32>
  %3 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%cst_0, %cst_2 : tensor<f32>, tensor<f32>) 
                                                             outs(%2 : tensor<f32>) -> tensor<f32>
  return %1, %3 : tensor<1x4x2047x2047xf32>, tensor<f32>
}

// -----

// CHECK-LABEL: func.func @test_reduce_with_index_canonicalized_to_linalg
// CHECK: linalg.reduce
func.func @test_reduce_with_index_canonicalized_to_linalg(%arg0: tensor<24x768x768xf32>) -> tensor<24x768x1xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<24x768xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<24x768xi64>) -> tensor<24x768xi64>
  %2 = tensor.empty() : tensor<24x768xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<24x768xf32>) -> tensor<24x768xf32>
  %4:2 = hfusion.reduce_with_index {tie_break_left = true} <max> ins(%arg0 : tensor<24x768x768xf32>) outs(%3, %1 : tensor<24x768xf32>, tensor<24x768xi64>) dimensions = [2] -> tensor<24x768xf32>, tensor<24x768xi64>
  %expanded = tensor.expand_shape %4#0 [[0], [1, 2]] output_shape [24, 768, 1] : tensor<24x768xf32> into tensor<24x768x1xf32>
  return %expanded : tensor<24x768x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_expand_shape_canonicalize
// CHECK-NEXT: tensor.expand_shape
// CHECK-NEXT: return
func.func @test_expand_shape_canonicalize(%arg0: tensor<f32>) -> tensor<1xf32> {
    %expanded = tensor.expand_shape %arg0 [] output_shape [1, 1] : tensor<f32> into tensor<1x1xf32>
    %collapsed = tensor.collapse_shape %expanded [[0, 1]] : tensor<1x1xf32> into tensor<1xf32>
    return %collapsed : tensor<1xf32>
}

// -----
// CHECK-LABEL: func.func @empty_reduce_consecutive(
// CHECK: linalg.reduce
// CHECK-SAME: dimensions = [1]
// CHECK-NOT: linalg.reduce
// CHECK: return
module {
  func.func @empty_reduce_consecutive(%arg0: tensor<27x22xi32>) -> tensor<27xi32> {
    %0 = tensor.empty() : tensor<27xi32>
    %reduced = linalg.reduce ins(%arg0 : tensor<27x22xi32>) outs(%0 : tensor<27xi32>) dimensions = [1]
      (%in: i32, %init: i32) {
        %1 = arith.xori %in, %init : i32
        linalg.yield %1 : i32
      }
    %reduced_0 = linalg.reduce ins(%reduced : tensor<27xi32>) outs(%0 : tensor<27xi32>) dimensions = []
      (%in: i32, %init: i32) {
        %1 = arith.xori %in, %init : i32
        linalg.yield %1 : i32
      }
    return %reduced_0 : tensor<27xi32>
  }
}
