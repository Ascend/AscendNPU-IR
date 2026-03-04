// RUN: bishengir-opt -hfusion-legalize-fp8 %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_ceil_fp8e4m3
func.func @test_ceil_fp8e4m3(%src : tensor<6x4xf8E4M3FN>, %dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%[[VAL_1]] : tensor<6x4xf32>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<6x4xf8E4M3FN>
  // CHECK: %[[VAL_6:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_4]] : tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  // CHECK: return %[[VAL_6]] : tensor<6x4xf8E4M3FN>
  %res = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%src : tensor<6x4xf8E4M3FN>) outs(%dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  return %res : tensor<6x4xf8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @test_ceil_fp8e5m2
func.func @test_ceil_fp8e5m2(%src : tensor<6x4xf8E5M2>, %dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E5M2>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E5M2>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%[[VAL_1]] : tensor<6x4xf32>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<6x4xf8E5M2>
  // CHECK: %[[VAL_6:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_4]] : tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  // CHECK: return %[[VAL_6]] : tensor<6x4xf8E5M2>
  %res = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%src : tensor<6x4xf8E5M2>) outs(%dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  return %res : tensor<6x4xf8E5M2>
}

// -----

// CHECK-LABEL: func.func @test_cos_fp8e4m3
func.func @test_cos_fp8e4m3(%src : tensor<6x4xf8E4M3FN>, %dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>} ins(%[[VAL_1]] : tensor<6x4xf32>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<6x4xf8E4M3FN>
  // CHECK: %[[VAL_6:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_4]] : tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  // CHECK: return %[[VAL_6]] : tensor<6x4xf8E4M3FN>
  %res = hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>} ins(%src : tensor<6x4xf8E4M3FN>) outs(%dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  return %res : tensor<6x4xf8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @test_cos_fp8e5m2
func.func @test_cos_fp8e5m2(%src : tensor<6x4xf8E5M2>, %dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E5M2>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E5M2>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>} ins(%[[VAL_1]] : tensor<6x4xf32>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<6x4xf8E5M2>
  // CHECK: %[[VAL_6:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_4]] : tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  // CHECK: return %[[VAL_6]] : tensor<6x4xf8E5M2>
  %res = hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>} ins(%src : tensor<6x4xf8E5M2>) outs(%dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  return %res : tensor<6x4xf8E5M2>
}

// -----

// CHECK-LABEL: func.func @test_sin_fp8e4m3
func.func @test_sin_fp8e4m3(%src : tensor<6x4xf8E4M3FN>, %dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sin>} ins(%[[VAL_1]] : tensor<6x4xf32>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<6x4xf8E4M3FN>
  // CHECK: %[[VAL_6:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_4]] : tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  // CHECK: return %[[VAL_6]] : tensor<6x4xf8E4M3FN>
  %res = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sin>} ins(%src : tensor<6x4xf8E4M3FN>) outs(%dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  return %res : tensor<6x4xf8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @test_sin_fp8e5m2
func.func @test_sin_fp8e5m2(%src : tensor<6x4xf8E5M2>, %dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E5M2>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E5M2>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sin>} ins(%[[VAL_1]] : tensor<6x4xf32>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<6x4xf8E5M2>
  // CHECK: %[[VAL_6:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_4]] : tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  // CHECK: return %[[VAL_6]] : tensor<6x4xf8E5M2>
  %res = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sin>} ins(%src : tensor<6x4xf8E5M2>) outs(%dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  return %res : tensor<6x4xf8E5M2>
}

// -----

// CHECK-LABEL: func.func @test_div_rn_fp8e4m3
func.func @test_div_rn_fp8e4m3(%src0 : tensor<6x4xf8E4M3FN>, %src1 : tensor<6x4xf8E4M3FN>, %dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] =  hfusion.cast {{.*}} ins(%arg2 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_4]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_6:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[VAL_1]], %[[VAL_3]] : tensor<6x4xf32>, tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_7:.*]] = tensor.empty() : tensor<6x4xf8E4M3FN>
  // CHECK: %[[VAL_8:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_6]] : tensor<6x4xf32>) outs(%[[VAL_7]] : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  // CHECK: return %[[VAL_8]] : tensor<6x4xf8E4M3FN>
  %res = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%src0, %src1 : tensor<6x4xf8E4M3FN>, tensor<6x4xf8E4M3FN>) outs(%dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  return %res : tensor<6x4xf8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @test_div_rn_fp8e5m2
func.func @test_div_rn_fp8e5m2(%src0 : tensor<6x4xf8E5M2>, %src1 : tensor<6x4xf8E5M2>, %dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E5M2>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E5M2>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] =  hfusion.cast {{.*}} ins(%arg2 : tensor<6x4xf8E5M2>) outs(%[[VAL_4]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_6:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[VAL_1]], %[[VAL_3]] : tensor<6x4xf32>, tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_7:.*]] = tensor.empty() : tensor<6x4xf8E5M2>
  // CHECK: %[[VAL_8:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_6]] : tensor<6x4xf32>) outs(%[[VAL_7]] : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  // CHECK: return %[[VAL_8]] : tensor<6x4xf8E5M2>
  %res = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%src0, %src1 : tensor<6x4xf8E5M2>, tensor<6x4xf8E5M2>) outs(%dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  return %res : tensor<6x4xf8E5M2>
}

// -----

// CHECK-LABEL: func.func @test_floor_fp8e4m3
func.func @test_floor_fp8e4m3(%src : tensor<6x4xf8E4M3FN>, %dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>} ins(%[[VAL_1]] : tensor<6x4xf32>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<6x4xf8E4M3FN>
  // CHECK: %[[VAL_6:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_4]] : tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  // CHECK: return %[[VAL_6]] : tensor<6x4xf8E4M3FN>
  %res = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>} ins(%src : tensor<6x4xf8E4M3FN>) outs(%dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  return %res : tensor<6x4xf8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @test_floor_fp8e5m2
func.func @test_floor_fp8e5m2(%src : tensor<6x4xf8E5M2>, %dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E5M2>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E5M2>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>} ins(%[[VAL_1]] : tensor<6x4xf32>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<6x4xf8E5M2>
  // CHECK: %[[VAL_6:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_4]] : tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  // CHECK: return %[[VAL_6]] : tensor<6x4xf8E5M2>
  %res = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>} ins(%src : tensor<6x4xf8E5M2>) outs(%dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  return %res : tensor<6x4xf8E5M2>
}

// -----

// CHECK-LABEL: func.func @test_cumsum_fp8e4m3
func.func @test_cumsum_fp8e4m3(%src : tensor<22x1x1xf8E4M3FN>) -> tensor<22x1x1xf8E4M3FN> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<22x1x1xf32> 
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<22x1x1xf8E4M3FN>) outs(%[[VAL_0]] : tensor<22x1x1xf32>) -> tensor<22x1x1xf32>
  // CHECK: %[[VAL_2:.*]] = hfusion.cumsum %[[VAL_1]] : tensor<22x1x1xf32> cum_dims = [2] reverse = false -> tensor<22x1x1xf32>
  // CHECK: %[[VAL_3:.*]] = tensor.empty() : tensor<22x1x1xf8E4M3FN> 
  // CHECK: %[[VAL_4:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_2]] : tensor<22x1x1xf32>) outs(%[[VAL_3]] : tensor<22x1x1xf8E4M3FN>) -> tensor<22x1x1xf8E4M3FN> 
  // CHECK: return %[[VAL_4]] : tensor<22x1x1xf8E4M3FN> 
  %res = hfusion.cumsum %src : tensor<22x1x1xf8E4M3FN> cum_dims = [2] reverse = false -> tensor<22x1x1xf8E4M3FN>
  return %res : tensor<22x1x1xf8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @test_cumsum_fp8e5m2
func.func @test_cumsum_fp8e5m2(%src : tensor<22x1x1xf8E5M2>) -> tensor<22x1x1xf8E5M2> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<22x1x1xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<22x1x1xf8E5M2>) outs(%[[VAL_0]] : tensor<22x1x1xf32>) -> tensor<22x1x1xf32>
  // CHECK: %[[VAL_2:.*]] = hfusion.cumsum %[[VAL_1]] : tensor<22x1x1xf32> cum_dims = [2] reverse = false -> tensor<22x1x1xf32>
  // CHECK: %[[VAL_3:.*]] = tensor.empty() : tensor<22x1x1xf8E5M2> 
  // CHECK: %[[VAL_4:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_2]] : tensor<22x1x1xf32>) outs(%[[VAL_3]] : tensor<22x1x1xf8E5M2>) -> tensor<22x1x1xf8E5M2> 
  // CHECK: return %[[VAL_4]] : tensor<22x1x1xf8E5M2> 
  %res = hfusion.cumsum %src : tensor<22x1x1xf8E5M2> cum_dims = [2] reverse = false -> tensor<22x1x1xf8E5M2>
  return %res : tensor<22x1x1xf8E5M2>
}

// -----

// CHECK-LABEL: func.func @test_reduce_with_index_fp8e5m2
func.func @test_reduce_with_index_fp8e5m2(%src : tensor<?x?xf8E5M2>, %src_index : tensor<?x?xi32>, %dst : tensor<?xf8E5M2>, %dst_index : tensor<?xi32>) -> tensor<?xf8E5M2> {
  // CHECK:  hfusion.cast
  // CHECK-SAME: tensor<?x?xf8E5M2>) {{.*}} -> tensor<?x?xf32>
  // CHECK: hfusion.cast
  // CHECK-SAME:  tensor<?xf8E5M2>) {{.*}} -> tensor<?xf32>
  // CHECK: hfusion.reduce_with_index
  // CHECK-SAME: tensor<?x?xf32>, tensor<?x?xi32>
  // CHECK-SAME:  tensor<?xf32>, tensor<?xi32>
  // CHECK:  hfusion.cast
  // CHECK-SAME: tensor<?xf32>) {{.*}} -> tensor<?xf8E5M2>
  %res:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <max> ins(%src, %src_index : tensor<?x?xf8E5M2>, tensor<?x?xi32>) 
                                      outs(%dst, %dst_index : tensor<?xf8E5M2>, tensor<?xi32>) dimensions = [0]  -> tensor<?xf8E5M2>, tensor<?xi32>
  return %res#0 : tensor<?xf8E5M2>
}

// -----

// CHECK-LABEL: func.func @test_reduce_with_index_fp8e4m3
func.func @test_reduce_with_index_fp8e4m3(%src : tensor<?x?xf8E4M3FN>, %src_index : tensor<?x?xi32>, %dst : tensor<?xf8E4M3FN>, %dst_index : tensor<?xi32>) -> tensor<?xf8E4M3FN> {
  // CHECK:  hfusion.cast
  // CHECK-SAME: tensor<?x?xf8E4M3FN>) {{.*}} -> tensor<?x?xf32>
  // CHECK: hfusion.cast
  // CHECK-SAME:  tensor<?xf8E4M3FN>) {{.*}} -> tensor<?xf32>
  // CHECK: hfusion.reduce_with_index
  // CHECK-SAME: tensor<?x?xf32>, tensor<?x?xi32>
  // CHECK-SAME:  tensor<?xf32>, tensor<?xi32>
  // CHECK:  hfusion.cast
  // CHECK-SAME: tensor<?xf32>) {{.*}} -> tensor<?xf8E4M3FN>
  %res:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <max> ins(%src, %src_index : tensor<?x?xf8E4M3FN>, tensor<?x?xi32>) 
                                      outs(%dst, %dst_index : tensor<?xf8E4M3FN>, tensor<?xi32>) dimensions = [0]  -> tensor<?xf8E4M3FN>, tensor<?xi32>
  return %res#0 : tensor<?xf8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @legalize_fp8_reduction$sum$f8E4M3FN
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<8xf8E4M3FN>, %[[ARG1:.*]]: tensor<f8E4M3FN>, %[[ARG2:.*]]: f8E4M3FN) -> tensor<f8E4M3FN> {
// CHECK:       %[[EMPTY_F32_8:.*]] = tensor.empty() : tensor<8xf32>
// CHECK:       %[[CAST_IN:.*]] = hfusion.cast 
// CHECK-SAME:    ins(%[[ARG0]] : tensor<8xf8E4M3FN>) outs(%[[EMPTY_F32_8]] : tensor<8xf32>) -> tensor<8xf32>
// CHECK:       %[[EMPTY_F32_SCALAR:.*]] = tensor.empty() : tensor<f32>
// CHECK:       %[[CAST_INIT:.*]] = hfusion.cast 
// CHECK-SAME:    ins(%[[ARG1]] : tensor<f8E4M3FN>) outs(%[[EMPTY_F32_SCALAR]] : tensor<f32>) -> tensor<f32>
// CHECK:       %[[REDUCED:.*]] = linalg.reduce ins(%[[CAST_IN]] : tensor<8xf32>) outs(%[[CAST_INIT]] : tensor<f32>) dimensions = [0] 
// CHECK:       (%[[IN:.*]]: f32, %[[INIT:.*]]: f32) {
// CHECK:         %[[ADD:.*]] = arith.addf %[[IN]], %[[INIT]] : f32
// CHECK:         linalg.yield %[[ADD]] : f32
// CHECK:       }
// CHECK:       %[[EMPTY_OUT:.*]] = tensor.empty() : tensor<f8E4M3FN>
// CHECK:       %[[CAST_OUT:.*]] = hfusion.cast 
// CHECK-SAME:    ins(%[[REDUCED]] : tensor<f32>) outs(%[[EMPTY_OUT]] : tensor<f8E4M3FN>) -> tensor<f8E4M3FN>
// CHECK:       return %[[CAST_OUT]] : tensor<f8E4M3FN>
func.func @legalize_fp8_reduction$sum$f8E4M3FN(%arg0: tensor<8xf8E4M3FN>, %arg1: tensor<f8E4M3FN>, %arg2: f8E4M3FN) -> tensor<f8E4M3FN> {
  %reduced = linalg.reduce ins(%arg0 : tensor<8xf8E4M3FN>) outs(%arg1 : tensor<f8E4M3FN>) dimensions = [0] 
    (%in: f8E4M3FN, %init: f8E4M3FN) {
      %1 = arith.addf %in, %init : f8E4M3FN
      linalg.yield %1 : f8E4M3FN
    }
  return %reduced : tensor<f8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @legalize_fp8_reduction$sum$f8E5M2
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<8xf8E5M2>, %[[ARG1:.*]]: tensor<f8E5M2>, %[[ARG2:.*]]: f8E5M2) -> tensor<f8E5M2> {
// CHECK:       %[[EMPTY_F32_8:.*]] = tensor.empty() : tensor<8xf32>
// CHECK:       %[[CAST_IN:.*]] = hfusion.cast 
// CHECK-SAME:    ins(%[[ARG0]] : tensor<8xf8E5M2>) outs(%[[EMPTY_F32_8]] : tensor<8xf32>) -> tensor<8xf32>
// CHECK:       %[[EMPTY_F32_SCALAR:.*]] = tensor.empty() : tensor<f32>
// CHECK:       %[[CAST_INIT:.*]] = hfusion.cast 
// CHECK-SAME:    ins(%[[ARG1]] : tensor<f8E5M2>) outs(%[[EMPTY_F32_SCALAR]] : tensor<f32>) -> tensor<f32>
// CHECK:       %[[REDUCED:.*]] = linalg.reduce ins(%[[CAST_IN]] : tensor<8xf32>) outs(%[[CAST_INIT]] : tensor<f32>) dimensions = [0] 
// CHECK:       (%[[IN:.*]]: f32, %[[INIT:.*]]: f32) {
// CHECK:         %[[ADD:.*]] = arith.addf %[[IN]], %[[INIT]] : f32
// CHECK:         linalg.yield %[[ADD]] : f32
// CHECK:       }
// CHECK:       %[[EMPTY_OUT:.*]] = tensor.empty() : tensor<f8E5M2>
// CHECK:       %[[CAST_OUT:.*]] = hfusion.cast 
// CHECK-SAME:    ins(%[[REDUCED]] : tensor<f32>) outs(%[[EMPTY_OUT]] : tensor<f8E5M2>) -> tensor<f8E5M2>
// CHECK:       return %[[CAST_OUT]] : tensor<f8E5M2>
func.func @legalize_fp8_reduction$sum$f8E5M2(%arg0: tensor<8xf8E5M2>, %arg1: tensor<f8E5M2>, %arg2: f8E5M2) -> tensor<f8E5M2> {
  %reduced = linalg.reduce ins(%arg0 : tensor<8xf8E5M2>) outs(%arg1 : tensor<f8E5M2>) dimensions = [0] 
    (%in: f8E5M2, %init: f8E5M2) {
      %1 = arith.addf %in, %init : f8E5M2
      linalg.yield %1 : f8E5M2
    }
  return %reduced : tensor<f8E5M2>
}

// -----

// CHECK-LABEL: func.func @test_abs_fp8e4m3
func.func @test_abs_fp8e4m3(%src : tensor<6x4xf8E4M3FN>, %dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[VAL_1]] : tensor<6x4xf32>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<6x4xf8E4M3FN>
  // CHECK: %[[VAL_6:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_4]] : tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  // CHECK: return %[[VAL_6]] : tensor<6x4xf8E4M3FN>
  %res = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%src : tensor<6x4xf8E4M3FN>) outs(%dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  return %res : tensor<6x4xf8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @test_abs_fp8e5m2
func.func @test_abs_fp8e5m2(%src : tensor<6x4xf8E5M2>, %dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E5M2>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E5M2>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[VAL_1]] : tensor<6x4xf32>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<6x4xf8E5M2>
  // CHECK: %[[VAL_6:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_4]] : tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  // CHECK: return %[[VAL_6]] : tensor<6x4xf8E5M2>
  %res = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%src : tensor<6x4xf8E5M2>) outs(%dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  return %res : tensor<6x4xf8E5M2>
}

// -----

// CHECK-LABEL: func.func @test_exp_fp8e4m3
func.func @test_exp_fp8e4m3(%src : tensor<6x4xf8E4M3FN>, %dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[VAL_1]] : tensor<6x4xf32>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<6x4xf8E4M3FN>
  // CHECK: %[[VAL_6:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_4]] : tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  // CHECK: return %[[VAL_6]] : tensor<6x4xf8E4M3FN>
  %res = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%src : tensor<6x4xf8E4M3FN>) outs(%dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  return %res : tensor<6x4xf8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @test_exp_fp8e5m2
func.func @test_exp_fp8e5m2(%src : tensor<6x4xf8E5M2>, %dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E5M2>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E5M2>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[VAL_1]] : tensor<6x4xf32>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<6x4xf8E5M2>
  // CHECK: %[[VAL_6:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_4]] : tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  // CHECK: return %[[VAL_6]] : tensor<6x4xf8E5M2>
  %res = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%src : tensor<6x4xf8E5M2>) outs(%dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  return %res : tensor<6x4xf8E5M2>
}

// -----

// CHECK-LABEL: func.func @test_log_fp8e4m3
func.func @test_log_fp8e4m3(%src : tensor<6x4xf8E4M3FN>, %dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[VAL_1]] : tensor<6x4xf32>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<6x4xf8E4M3FN>
  // CHECK: %[[VAL_6:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_4]] : tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  // CHECK: return %[[VAL_6]] : tensor<6x4xf8E4M3FN>
  %res = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%src : tensor<6x4xf8E4M3FN>) outs(%dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  return %res : tensor<6x4xf8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @test_log_fp8e5m2
func.func @test_log_fp8e5m2(%src : tensor<6x4xf8E5M2>, %dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E5M2>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E5M2>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[VAL_1]] : tensor<6x4xf32>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<6x4xf8E5M2>
  // CHECK: %[[VAL_6:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_4]] : tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  // CHECK: return %[[VAL_6]] : tensor<6x4xf8E5M2>
  %res = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%src : tensor<6x4xf8E5M2>) outs(%dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  return %res : tensor<6x4xf8E5M2>
}

// -----

// CHECK-LABEL: func.func @test_sqrt_fp8e4m3
func.func @test_sqrt_fp8e4m3(%src : tensor<6x4xf8E4M3FN>, %dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<sqrt>} ins(%[[VAL_1]] : tensor<6x4xf32>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<6x4xf8E4M3FN>
  // CHECK: %[[VAL_6:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_4]] : tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  // CHECK: return %[[VAL_6]] : tensor<6x4xf8E4M3FN>
  %res = linalg.elemwise_unary {fun = #linalg.unary_fn<sqrt>} ins(%src : tensor<6x4xf8E4M3FN>) outs(%dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  return %res : tensor<6x4xf8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @test_sqrt_fp8e5m2
func.func @test_sqrt_fp8e5m2(%src : tensor<6x4xf8E5M2>, %dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E5M2>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E5M2>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<sqrt>} ins(%[[VAL_1]] : tensor<6x4xf32>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<6x4xf8E5M2>
  // CHECK: %[[VAL_6:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_4]] : tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  // CHECK: return %[[VAL_6]] : tensor<6x4xf8E5M2>
  %res = linalg.elemwise_unary {fun = #linalg.unary_fn<sqrt>} ins(%src : tensor<6x4xf8E5M2>) outs(%dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  return %res : tensor<6x4xf8E5M2>
}

// -----

// CHECK-LABEL: func.func @test_exp2_fp8e4m3
func.func @test_exp2_fp8e4m3(%src : tensor<6x4xf8E4M3FN>, %dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<exp2>} ins(%[[VAL_1]] : tensor<6x4xf32>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<6x4xf8E4M3FN>
  // CHECK: %[[VAL_6:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_4]] : tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  // CHECK: return %[[VAL_6]] : tensor<6x4xf8E4M3FN>
  %res = hfusion.elemwise_unary {fun = #hfusion.unary_fn<exp2>} ins(%src : tensor<6x4xf8E4M3FN>) outs(%dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  return %res : tensor<6x4xf8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @test_exp2_fp8e5m2
func.func @test_exp2_fp8e5m2(%src : tensor<6x4xf8E5M2>, %dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E5M2>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E5M2>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<exp2>} ins(%[[VAL_1]] : tensor<6x4xf32>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<6x4xf8E5M2>
  // CHECK: %[[VAL_6:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_4]] : tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  // CHECK: return %[[VAL_6]] : tensor<6x4xf8E5M2>
  %res = hfusion.elemwise_unary {fun = #hfusion.unary_fn<exp2>} ins(%src : tensor<6x4xf8E5M2>) outs(%dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  return %res : tensor<6x4xf8E5M2>
}

// -----

// CHECK-LABEL: func.func @test_log2_fp8e4m3
func.func @test_log2_fp8e4m3(%src : tensor<6x4xf8E4M3FN>, %dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<log2>} ins(%[[VAL_1]] : tensor<6x4xf32>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<6x4xf8E4M3FN>
  // CHECK: %[[VAL_6:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_4]] : tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  // CHECK: return %[[VAL_6]] : tensor<6x4xf8E4M3FN>
  %res = hfusion.elemwise_unary {fun = #hfusion.unary_fn<log2>} ins(%src : tensor<6x4xf8E4M3FN>) outs(%dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  return %res : tensor<6x4xf8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @test_log2_fp8e5m2
func.func @test_log2_fp8e5m2(%src : tensor<6x4xf8E5M2>, %dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E5M2>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E5M2>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<log2>} ins(%[[VAL_1]] : tensor<6x4xf32>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<6x4xf8E5M2>
  // CHECK: %[[VAL_6:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_4]] : tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  // CHECK: return %[[VAL_6]] : tensor<6x4xf8E5M2>
  %res = hfusion.elemwise_unary {fun = #hfusion.unary_fn<log2>} ins(%src : tensor<6x4xf8E5M2>) outs(%dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  return %res : tensor<6x4xf8E5M2>
}

// -----

// CHECK-LABEL: func.func @test_rsqrt_fp8e4m3
func.func @test_rsqrt_fp8e4m3(%src : tensor<6x4xf8E4M3FN>, %dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%[[VAL_1]] : tensor<6x4xf32>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<6x4xf8E4M3FN>
  // CHECK: %[[VAL_6:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_4]] : tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  // CHECK: return %[[VAL_6]] : tensor<6x4xf8E4M3FN>
  %res = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%src : tensor<6x4xf8E4M3FN>) outs(%dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  return %res : tensor<6x4xf8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @test_rsqrt_fp8e5m2
func.func @test_rsqrt_fp8e5m2(%src : tensor<6x4xf8E5M2>, %dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E5M2>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E5M2>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%[[VAL_1]] : tensor<6x4xf32>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<6x4xf8E5M2>
  // CHECK: %[[VAL_6:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_4]] : tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  // CHECK: return %[[VAL_6]] : tensor<6x4xf8E5M2>
  %res = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%src : tensor<6x4xf8E5M2>) outs(%dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  return %res : tensor<6x4xf8E5M2>
}

// -----

// CHECK-LABEL: func.func @test_sqrt_fp8e4m3
func.func @test_sqrt_fp8e4m3(%src : tensor<6x4xf8E4M3FN>, %dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[VAL_1]] : tensor<6x4xf32>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<6x4xf8E4M3FN>
  // CHECK: %[[VAL_6:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_4]] : tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  // CHECK: return %[[VAL_6]] : tensor<6x4xf8E4M3FN>
  %res = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%src : tensor<6x4xf8E4M3FN>) outs(%dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  return %res : tensor<6x4xf8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @test_sqrt_fp8e5m2
func.func @test_sqrt_fp8e5m2(%src : tensor<6x4xf8E5M2>, %dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E5M2>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E5M2>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[VAL_1]] : tensor<6x4xf32>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<6x4xf8E5M2>
  // CHECK: %[[VAL_6:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_4]] : tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  // CHECK: return %[[VAL_6]] : tensor<6x4xf8E5M2>
  %res = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%src : tensor<6x4xf8E5M2>) outs(%dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  return %res : tensor<6x4xf8E5M2>
}

// -----

// CHECK-LABEL: func.func @test_sub_fp8e4m3
func.func @test_sub_fp8e4m3(%src0 : tensor<6x4xf8E4M3FN>, %src1 : tensor<6x4xf8E4M3FN>, %dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] =  hfusion.cast {{.*}} ins(%arg2 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_4]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_6:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_1]], %[[VAL_3]] : tensor<6x4xf32>, tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_7:.*]] = tensor.empty() : tensor<6x4xf8E4M3FN>
  // CHECK: %[[VAL_8:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_6]] : tensor<6x4xf32>) outs(%[[VAL_7]] : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  // CHECK: return %[[VAL_8]] : tensor<6x4xf8E4M3FN>
  %res = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%src0, %src1 : tensor<6x4xf8E4M3FN>, tensor<6x4xf8E4M3FN>) outs(%dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  return %res : tensor<6x4xf8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @test_sub_fp8e5m2
func.func @test_sub_fp8e5m2(%src0 : tensor<6x4xf8E5M2>, %src1 : tensor<6x4xf8E5M2>, %dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E5M2>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E5M2>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] =  hfusion.cast {{.*}} ins(%arg2 : tensor<6x4xf8E5M2>) outs(%[[VAL_4]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_6:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_1]], %[[VAL_3]] : tensor<6x4xf32>, tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_7:.*]] = tensor.empty() : tensor<6x4xf8E5M2>
  // CHECK: %[[VAL_8:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_6]] : tensor<6x4xf32>) outs(%[[VAL_7]] : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  // CHECK: return %[[VAL_8]] : tensor<6x4xf8E5M2>
  %res = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%src0, %src1 : tensor<6x4xf8E5M2>, tensor<6x4xf8E5M2>) outs(%dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  return %res : tensor<6x4xf8E5M2>
}

// -----

// CHECK-LABEL: func.func @test_maxf_fp8e5m2
func.func @test_maxf_fp8e5m2(%src0 : tensor<6x4xf8E5M2>, %src1 : tensor<6x4xf8E5M2>, %dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E5M2>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E5M2>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] =  hfusion.cast {{.*}} ins(%arg2 : tensor<6x4xf8E5M2>) outs(%[[VAL_4]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_6:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%[[VAL_1]], %[[VAL_3]] : tensor<6x4xf32>, tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_7:.*]] = tensor.empty() : tensor<6x4xf8E5M2>
  // CHECK: %[[VAL_8:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_6]] : tensor<6x4xf32>) outs(%[[VAL_7]] : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  // CHECK: return %[[VAL_8]] : tensor<6x4xf8E5M2>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%src0, %src1 : tensor<6x4xf8E5M2>, tensor<6x4xf8E5M2>) outs(%dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  return %res : tensor<6x4xf8E5M2>
}

// -----

// CHECK-LABEL: func.func @test_minf_fp8e4m3
func.func @test_minf_fp8e4m3(%src0 : tensor<6x4xf8E4M3FN>, %src1 : tensor<6x4xf8E4M3FN>, %dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] =  hfusion.cast {{.*}} ins(%arg2 : tensor<6x4xf8E4M3FN>) outs(%[[VAL_4]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_6:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[VAL_1]], %[[VAL_3]] : tensor<6x4xf32>, tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_7:.*]] = tensor.empty() : tensor<6x4xf8E4M3FN>
  // CHECK: %[[VAL_8:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_6]] : tensor<6x4xf32>) outs(%[[VAL_7]] : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  // CHECK: return %[[VAL_8]] : tensor<6x4xf8E4M3FN>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%src0, %src1 : tensor<6x4xf8E4M3FN>, tensor<6x4xf8E4M3FN>) outs(%dst : tensor<6x4xf8E4M3FN>) -> tensor<6x4xf8E4M3FN>
  return %res : tensor<6x4xf8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @test_minf_fp8e5m2
func.func @test_minf_fp8e5m2(%src0 : tensor<6x4xf8E5M2>, %src1 : tensor<6x4xf8E5M2>, %dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {{.*}} ins(%arg0 : tensor<6x4xf8E5M2>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {{.*}} ins(%arg1 : tensor<6x4xf8E5M2>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] =  hfusion.cast {{.*}} ins(%arg2 : tensor<6x4xf8E5M2>) outs(%[[VAL_4]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_6:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[VAL_1]], %[[VAL_3]] : tensor<6x4xf32>, tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_7:.*]] = tensor.empty() : tensor<6x4xf8E5M2>
  // CHECK: %[[VAL_8:.*]] =  hfusion.cast {{.*}} ins(%[[VAL_6]] : tensor<6x4xf32>) outs(%[[VAL_7]] : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  // CHECK: return %[[VAL_8]] : tensor<6x4xf8E5M2>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%src0, %src1 : tensor<6x4xf8E5M2>, tensor<6x4xf8E5M2>) outs(%dst : tensor<6x4xf8E5M2>) -> tensor<6x4xf8E5M2>
  return %res : tensor<6x4xf8E5M2>
}
