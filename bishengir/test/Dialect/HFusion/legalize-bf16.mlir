// RUN: bishengir-opt -hfusion-legalize-bf16 %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_elemwise_unary_ops
func.func @test_elemwise_unary_ops(
  %src : tensor<6x4xbf16>, %dst : tensor<6x4xbf16>) -> tensor<6x4xbf16> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<6x4xbf16>) outs(%[[VAL_0]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<6x4xbf16>) outs(%[[VAL_2]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[VAL_1]] : tensor<6x4xf32>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<6x4xbf16>
  // CHECK: %[[VAL_6:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_4]] : tensor<6x4xf32>) outs(%[[VAL_5]] : tensor<6x4xbf16>) -> tensor<6x4xbf16>
  // CHECK: return %[[VAL_6]] : tensor<6x4xbf16>
  %res = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>}
    ins(%src : tensor<6x4xbf16>)
    outs(%dst : tensor<6x4xbf16>)
    -> tensor<6x4xbf16>
  return %res : tensor<6x4xbf16>
}

// -----

// CHECK-LABEL: func.func @test_two_elemwise_unary_ops
func.func @test_two_elemwise_unary_ops(%arg0 : tensor<6x4xbf16>, %dst : tensor<6x4xbf16>) -> tensor<6x4xbf16> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xbf16>
  // CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<6x4xbf16>) outs(%[[VAL_1]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_0]] : tensor<6x4xbf16>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[VAL_2]] : tensor<6x4xf32>) outs(%[[VAL_4]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_6:.*]] = tensor.empty() : tensor<6x4xbf16>
  // CHECK: %[[VAL_7:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_5]] : tensor<6x4xf32>) outs(%[[VAL_6]] : tensor<6x4xbf16>) -> tensor<6x4xbf16>
  // CHECK: %[[VAL_8:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_9:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_7]] : tensor<6x4xbf16>) outs(%[[VAL_8]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_10:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_11:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<6x4xbf16>) outs(%[[VAL_10]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_12:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%[[VAL_9]] : tensor<6x4xf32>) outs(%[[VAL_11]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_13:.*]] = tensor.empty() : tensor<6x4xbf16>
  // CHECK: %[[VAL_14:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_12]] : tensor<6x4xf32>) outs(%[[VAL_13]] : tensor<6x4xbf16>) -> tensor<6x4xbf16>
  // CHECK: return %[[VAL_14]] : tensor<6x4xbf16>
  %0 = tensor.empty() : tensor<6x4xbf16>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%arg0 : tensor<6x4xbf16>) outs(%0 : tensor<6x4xbf16>) -> tensor<6x4xbf16>
	%2 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%1 : tensor<6x4xbf16>) outs(%dst : tensor<6x4xbf16>) -> tensor<6x4xbf16>
  return %2 : tensor<6x4xbf16>
}

// -----

// CHECK-LABEL: func.func @test_three_elemwise_ops
func.func @test_three_elemwise_ops(%arg0 : tensor<6x4xbf16>, %arg1 : tensor<6x4xbf16>, %dst : tensor<6x4xbf16>) -> tensor<6x4xbf16> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xbf16>
  // CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<6x4xbf16>) outs(%[[VAL_1]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_0]] : tensor<6x4xbf16>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[VAL_2]] : tensor<6x4xf32>) outs(%[[VAL_4]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_6:.*]] = tensor.empty() : tensor<6x4xbf16>
  // CHECK: %[[VAL_7:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_5]] : tensor<6x4xf32>) outs(%[[VAL_6]] : tensor<6x4xbf16>) -> tensor<6x4xbf16>
  // CHECK: %[[VAL_8:.*]] = tensor.empty() : tensor<6x4xbf16>
  // CHECK: %[[VAL_9:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_10:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<6x4xbf16>) outs(%[[VAL_9]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_11:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_12:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_8]] : tensor<6x4xbf16>) outs(%[[VAL_11]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_13:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%[[VAL_10]] : tensor<6x4xf32>) outs(%[[VAL_12]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_14:.*]] = tensor.empty() : tensor<6x4xbf16>
  // CHECK: %[[VAL_15:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_13]] : tensor<6x4xf32>) outs(%[[VAL_14]] : tensor<6x4xbf16>) -> tensor<6x4xbf16>
  // CHECK: %[[VAL_16:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_17:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_7]] : tensor<6x4xbf16>) outs(%[[VAL_16]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_18:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_19:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_15]] : tensor<6x4xbf16>) outs(%[[VAL_18]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_20:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_21:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg2 : tensor<6x4xbf16>) outs(%[[VAL_20]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_22:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_17]], %[[VAL_19]] : tensor<6x4xf32>, tensor<6x4xf32>) outs(%[[VAL_21]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_23:.*]] = tensor.empty() : tensor<6x4xbf16>
  // CHECK: %[[VAL_24:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_22]] : tensor<6x4xf32>) outs(%[[VAL_23]] : tensor<6x4xbf16>) -> tensor<6x4xbf16>
  // CHECK: return %[[VAL_24]] : tensor<6x4xbf16>
  %0 = tensor.empty() : tensor<6x4xbf16>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%arg0 : tensor<6x4xbf16>) outs(%0 : tensor<6x4xbf16>) -> tensor<6x4xbf16>
	%2 = tensor.empty() : tensor<6x4xbf16>
	%3 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%arg1 : tensor<6x4xbf16>) outs(%2 : tensor<6x4xbf16>) -> tensor<6x4xbf16>
	%4 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %3 : tensor<6x4xbf16>, tensor<6x4xbf16>) outs(%dst : tensor<6x4xbf16>) -> tensor<6x4xbf16>
  return %4 : tensor<6x4xbf16>
}

// -----

// CHECK-LABEL: func.func @test_elemwise_broadcast_ops
func.func @test_elemwise_broadcast_ops(%src : tensor<6x4xbf16>, %dst : tensor<6x4x3xbf16>) -> tensor<6x4x3xbf16> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xbf16>
  // CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_2:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<6x4xbf16>) outs(%[[VAL_1]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_3:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_4:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_0]] : tensor<6x4xbf16>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_5:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[VAL_2]] : tensor<6x4xf32>) outs(%[[VAL_4]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_6:.*]] = tensor.empty() : tensor<6x4xbf16>
  // CHECK: %[[VAL_7:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_5]] : tensor<6x4xf32>) outs(%[[VAL_6]] : tensor<6x4xbf16>) -> tensor<6x4xbf16>
  // CHECK: %[[VAL_8:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK: %[[VAL_9:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_7]] : tensor<6x4xbf16>) outs(%[[VAL_8]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[VAL_10:.*]] = tensor.empty() : tensor<6x4x3xf32>
  // CHECK: %[[VAL_11:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<6x4x3xbf16>) outs(%[[VAL_10]] : tensor<6x4x3xf32>) -> tensor<6x4x3xf32>
  // CHECK: %[[VAL_BRC:.*]] = linalg.broadcast ins(%[[VAL_9]] : tensor<6x4xf32>) outs(%[[VAL_11]] : tensor<6x4x3xf32>) dimensions = [2]
  // CHECK: %[[VAL_12:.*]] = tensor.empty() : tensor<6x4x3xbf16>
  // CHECK: %[[VAL_13:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_BRC]] : tensor<6x4x3xf32>) outs(%[[VAL_12]] : tensor<6x4x3xbf16>) -> tensor<6x4x3xbf16>
  // CHECK: return %[[VAL_13]] : tensor<6x4x3xbf16>
	%0 = tensor.empty() : tensor<6x4xbf16>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%src : tensor<6x4xbf16>) outs(%0 : tensor<6x4xbf16>) -> tensor<6x4xbf16>
	%res = linalg.broadcast ins(%1 : tensor<6x4xbf16>) outs(%dst : tensor<6x4x3xbf16>) dimensions = [2]
  return %res : tensor<6x4x3xbf16>
}

// -----

// CHECK-LABEL: func.func @test_elemwise_reduce_ops
func.func @test_elemwise_reduce_ops(%arg0 : tensor<6x4xbf16>, %arg1 : tensor<6x4xbf16>, %dst : tensor<6xbf16>) -> tensor<6xbf16> {
  // CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xbf16>
  // CHECK:  %[[VAL_1:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK:  %[[VAL_2:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<6x4xbf16>) outs(%[[VAL_1]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK:  %[[VAL_3:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK:  %[[VAL_4:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<6x4xbf16>) outs(%[[VAL_3]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK:  %[[VAL_5:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK:  %[[VAL_6:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_0]] : tensor<6x4xbf16>) outs(%[[VAL_5]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK:  %[[VAL_7:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_2]], %[[VAL_4]] : tensor<6x4xf32>, tensor<6x4xf32>) outs(%6 : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK:  %[[VAL_8:.*]] = tensor.empty() : tensor<6x4xbf16>
  // CHECK:  %[[VAL_9:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_7]] : tensor<6x4xf32>) outs(%[[VAL_8]] : tensor<6x4xbf16>) -> tensor<6x4xbf16>
  // CHECK:  %[[VAL_10:.*]] = tensor.empty() : tensor<6x4xf32>
  // CHECK:  %[[VAL_11:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_9]] : tensor<6x4xbf16>) outs(%[[VAL_10]] : tensor<6x4xf32>) -> tensor<6x4xf32>
  // CHECK:  %[[VAL_12:.*]] = tensor.empty() : tensor<6xf32>
  // CHECK:  %[[VAL_13:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg2 : tensor<6xbf16>) outs(%[[VAL_12]] : tensor<6xf32>) -> tensor<6xf32>
  // CHECK:  %[[VAL_REDUCE:.*]] = linalg.reduce { arith.addf } ins(%[[VAL_11]] : tensor<6x4xf32>) outs(%[[VAL_13]] : tensor<6xf32>) dimensions = [1]
  // CHECK:  %[[VAL_14:.*]] = tensor.empty() : tensor<6xbf16>
  // CHECK:  %[[VAL_15:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_REDUCE]] : tensor<6xf32>) outs(%[[VAL_14]] : tensor<6xbf16>) -> tensor<6xbf16>
  // CHECK:  return %[[VAL_15]] : tensor<6xbf16>
  %0 = tensor.empty() : tensor<6x4xbf16>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<6x4xbf16>, tensor<6x4xbf16>) outs(%0 : tensor<6x4xbf16>) -> tensor<6x4xbf16>
  %2 = linalg.reduce { arith.addf } ins(%1: tensor<6x4xbf16>) outs(%dst: tensor<6xbf16>) dimensions = [1]
  return %2 : tensor<6xbf16>
}

// -----

// CHECK-LABEL: func.func @test_elemwise_unary_dynamic_shape_ops
func.func @test_elemwise_unary_dynamic_shape_ops(
  %src : tensor<?x?xbf16>, %dst : tensor<?x?xbf16>) -> tensor<?x?xbf16> {
  // CHECK: %c1 = arith.constant 1 : index
  // CHECK: %c0 = arith.constant 0 : index
  // CHECK: %dim = tensor.dim %arg0, %c0 : tensor<?x?xbf16>
  // CHECK: %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xbf16>
  // CHECK: %[[VAL_0:.*]] = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  // CHECK: %[[VAL_1:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<?x?xbf16>) outs(%[[VAL_0]] : tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %dim_1 = tensor.dim %arg1, %c0 : tensor<?x?xbf16>
  // CHECK: %dim_2 = tensor.dim %arg1, %c1 : tensor<?x?xbf16>
  // CHECK: %[[VAL_2:.*]] = tensor.empty(%dim_1, %dim_2) : tensor<?x?xf32>
  // CHECK: %[[VAL_3:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<?x?xbf16>) outs(%[[VAL_2]] : tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %[[VAL_4:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[VAL_1]] : tensor<?x?xf32>) outs(%[[VAL_3]] : tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %dim_3 = tensor.dim %4, %c0 : tensor<?x?xf32>
  // CHECK: %dim_4 = tensor.dim %4, %c1 : tensor<?x?xf32>
  // CHECK: %[[VAL_5:.*]] = tensor.empty(%dim_3, %dim_4) : tensor<?x?xbf16>
  // CHECK: %[[VAL_6:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_4]] : tensor<?x?xf32>) outs(%[[VAL_5]] : tensor<?x?xbf16>) -> tensor<?x?xbf16>
  // CHECK: return %[[VAL_6]] : tensor<?x?xbf16>
  %res = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>}
    ins(%src : tensor<?x?xbf16>)
    outs(%dst : tensor<?x?xbf16>)
    -> tensor<?x?xbf16>
  return %res : tensor<?x?xbf16>
}

// -----

// CHECK-LABEL: func.func @test_fill_copy_ops
func.func @test_fill_copy_ops(%dst : tensor<6x4xbf16>) -> tensor<6x4xbf16> {
  // CHECK: %cst = arith.constant 0.000000e+00 : bf16
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x4xbf16>
  // CHECK: %[[VAL_1:.*]] = linalg.fill ins(%cst : bf16) outs(%[[VAL_0]] : tensor<6x4xbf16>) -> tensor<6x4xbf16>
  // CHECK: %[[VAL_2:.*]] = linalg.copy ins(%[[VAL_1]] : tensor<6x4xbf16>) outs(%arg0 : tensor<6x4xbf16>) -> tensor<6x4xbf16>
  // CHECK: return %[[VAL_2]] : tensor<6x4xbf16>
  %cst_0 = arith.constant 0.000000e+00 : bf16
  %0 = tensor.empty() : tensor<6x4xbf16>
  %1 = linalg.fill ins(%cst_0 : bf16) outs(%0 : tensor<6x4xbf16>) -> tensor<6x4xbf16>
  %2 = linalg.copy ins(%1: tensor<6x4xbf16>) outs(%dst: tensor<6x4xbf16>) -> tensor<6x4xbf16>
  return %2 : tensor<6x4xbf16>
}


// -----

// CHECK-LABEL: func.func @test_binary_vs
func.func @test_binary_vs(%scalar : bf16, %src : tensor<6x4xbf16>, %dst : tensor<6x4xbf16>) -> (tensor<6x4xbf16>, tensor<6x4xbf16>) {
  // CHECK: arith.constant 1.000000e+00 : f32
  %cst = arith.constant 1.000000e+00 : bf16
  %1 = linalg.elemwise_binary ins(%src, %cst : tensor<6x4xbf16>, bf16) outs(%dst : tensor<6x4xbf16>) -> tensor<6x4xbf16>
  // CHECK: arith.extf
  %2 = linalg.elemwise_binary ins(%scalar, %src : bf16, tensor<6x4xbf16>) outs(%dst : tensor<6x4xbf16>) -> tensor<6x4xbf16>
  return %1, %2 : tensor<6x4xbf16>, tensor<6x4xbf16>
}

// -----

// CHECK-LABEL: func.func @test_compare_eq_ops
// CHECK: %[[res:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%[[arg0_f32:.*]], %[[arg1_f32:.*]] : tensor<16xf32>, tensor<16xf32>) outs(%[[empty0:.*]] : tensor<16xi1>) -> tensor<16xi1>
func.func @test_compare_eq_ops(%arg0 : tensor<16xbf16>, %arg1 : tensor<16xbf16>) ->  tensor<16xi1> {
  %0 = tensor.empty() : tensor<16xi1>
  %res = hfusion.compare {compare_fn  = #hfusion.compare_fn<vne>}
    ins(%arg0, %arg1 : tensor<16xbf16>, tensor<16xbf16>)
    outs(%0 : tensor<16xi1>)
    -> tensor<16xi1>
  return %res : tensor<16xi1>
}

// -----

// CHECK-LABEL: func.func @test_isfinite
func.func @test_isfinite() -> tensor<8192xi1> {
  %0 = tensor.empty() : tensor<8192xbf16>
  // CHECK: %[[res:.*]] = hfusion.isfinite %[[ins:.*]] : tensor<8192xf32> -> tensor<8192xi1>
  %2 = hfusion.isfinite %0 : tensor<8192xbf16> -> tensor<8192xi1>
  return %2 : tensor<8192xi1>
}

// -----

// CHECK-LABEL: func.func @test_isnan
func.func @test_isnan() -> tensor<8192xi1> {
  %0 = tensor.empty() : tensor<8192xbf16>
  // CHECK: %[[res:.*]] = hfusion.isnan %[[ins:.*]] : tensor<8192xf32> -> tensor<8192xi1>
  %2 = hfusion.isnan %0 : tensor<8192xbf16> -> tensor<8192xi1>
  return %2 : tensor<8192xi1>
}

// -----

// CHECK-LABEL: func.func @test_isinf
func.func @test_isinf() -> tensor<8192xi1> {
  %0 = tensor.empty() : tensor<8192xbf16>
  // CHECK: %[[res:.*]] = hfusion.isinf %[[ins:.*]] : tensor<8192xf32> -> tensor<8192xi1>
  %2 = hfusion.isinf %0 : tensor<8192xbf16> -> tensor<8192xi1>
  return %2 : tensor<8192xi1>
}

// -----

// CHECK-LABEL: func.func @test_const_inop
// CHECK: arith.constant 0.000000e+00 : f32
func.func @test_const_inop(%arg0: tensor<3x31x256xbf16>) -> tensor<3x31x256xbf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = tensor.empty() : tensor<3x31x256xbf16>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>} ins(%arg0 : tensor<3x31x256xbf16>) outs(%0 : tensor<3x31x256xbf16>) -> tensor<3x31x256xbf16>
  return %1 : tensor<3x31x256xbf16>
}

// -----

// CHECK-LABEL: func.func @test_no_load_or_store
func.func @test_no_load_or_store(%arg: tensor<bf16>) -> tensor<bf16> {
  %0 = tensor.empty() : tensor<bf16>
  %1 = tensor.empty() : tensor<bf16>
  // CHECK: hfusion.load{{.*}}tensor<bf16>{{.*}}tensor<bf16>{{.*}}tensor<bf16>
  %2 = hfusion.load ins(%arg: tensor<bf16>) outs(%0: tensor<bf16>) -> tensor<bf16>
  // CHECK: hfusion.store{{.*}}tensor<bf16>{{.*}}tensor<bf16>{{.*}}tensor<bf16>
  %3 = hfusion.store ins(%2: tensor<bf16>) outs(%1: tensor<bf16>) -> tensor<bf16>
  return %3 : tensor<bf16>
}

// -----

// CHECK-LABEL: func.func @test_tensor_concat
// CHECK: %[[cast0:.*]] = hfusion.cast {{.*}} ins({{.*}} : tensor<6x4xbf16>) outs({{.*}} : tensor<6x4xf32>)
// CHECK: %[[cast1:.*]] = hfusion.cast {{.*}} ins({{.*}} : tensor<6x4xbf16>) outs({{.*}} : tensor<6x4xf32>)
// CHECK: %[[concat:.*]] = tensor.concat dim(1) %[[cast0]], %[[cast1]]
// CHECK: hfusion.cast {{.*}} ins(%[[concat]] : tensor<6x8xf32>) outs({{.*}} : tensor<6x8xbf16>)
func.func @test_tensor_concat(%arg0 : tensor<6x4xbf16>, %arg1 : tensor<6x4xbf16>) -> tensor<6x8xbf16> {
  %concat = tensor.concat dim(1) %arg0, %arg1 : (tensor<6x4xbf16>, tensor<6x4xbf16>) -> tensor<6x8xbf16>
  return %concat : tensor<6x8xbf16>
}

// -----

// CHECK-LABEL: func.func @test_tensor_pad
// CHECK: %[[cast0:.*]] = hfusion.cast {{.*}} ins({{.*}} : tensor<2x3xbf16>) outs({{.*}} : tensor<2x3xf32>)
// CHECK: %[[padded:.*]] = tensor.pad %[[cast0]]
// CHECK: hfusion.cast {{.*}} ins(%[[padded]] : tensor<7x35xf32>) outs({{.*}} : tensor<7x35xbf16>)
func.func @test_tensor_pad(%arg0: tensor<2x3xbf16>) -> tensor<7x35xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = tensor.empty() : tensor<2x3xbf16>
  %padded = tensor.pad %arg0 low[5, 32] high[0, 0] {
    ^bb0(%arg1: index, %arg2: index):
    tensor.yield %cst : bf16
  } : tensor<2x3xbf16> to tensor<7x35xbf16>
  return %padded : tensor<7x35xbf16>
}

// -----

// CHECK-LABEL: func.func @test_gather
// CHECK: %[[VAL_21:.*]] = hfusion.gather {operandSegmentSizes = array<i32: 2, 1>} ins(%[[VAL_18:.*]], %[[VAL_15:.*]] : tensor<16x4x64xf32>, tensor<16x4x32xi32>) outs(%[[VAL_20:.*]] : tensor<16x4x32xf32>) axis = 2 -> tensor<16x4x32xf32>
func.func @test_gather(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg1: memref<?xbf16> {tt.divisibility = 16 : i32}, %arg2: memref<?xi32> {tt.divisibility = 16 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {WorkspaceArgIdx = 0 : i64, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv"} {
  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [16, 4, 64], strides: [256, 64, 1] : memref<?xbf16> to memref<16x4x64xbf16, strided<[256, 64, 1]>>
  %alloc = memref.alloc() : memref<16x4x64xbf16>
  memref.copy %reinterpret_cast, %alloc : memref<16x4x64xbf16, strided<[256, 64, 1]>> to memref<16x4x64xbf16>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<16x4x64xbf16>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16, 4, 32], strides: [128, 32, 1] : memref<?xi32> to memref<16x4x32xi32, strided<[128, 32, 1]>>
  %alloc_1 = memref.alloc() : memref<16x4x32xi32>
  memref.copy %reinterpret_cast_0, %alloc_1 : memref<16x4x32xi32, strided<[128, 32, 1]>> to memref<16x4x32xi32>
  %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<16x4x32xi32>
  %2 = tensor.empty() : tensor<16x4x32xbf16>
  %3 = hfusion.gather {operandSegmentSizes = array<i32: 2, 1>} ins(%0, %1 : tensor<16x4x64xbf16>, tensor<16x4x32xi32>) outs(%2 : tensor<16x4x32xbf16>) axis = 2 -> tensor<16x4x32xbf16>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [16, 4, 32], strides: [128, 32, 1] : memref<?xbf16> to memref<16x4x32xbf16, strided<[128, 32, 1]>>
  bufferization.materialize_in_destination %3 in writable %reinterpret_cast_2 : (tensor<16x4x32xbf16>, memref<16x4x32xbf16, strided<[128, 32, 1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @test_max_with_index
// CHECK-SAME: %[[arg0:.*]]: tensor<1x22x39xbf16>, %[[arg1:.*]]: tensor<1x22x39xi32>
// CHECK: %[[cast:.*]] = hfusion.cast {{.*}} ins({{.*}} : tensor<1x22x39xbf16>) outs({{.*}} : tensor<1x22x39xf32>)
// CHECK: hfusion.reduce_with_index {tie_break_left = true} <max> ins(%[[cast]], %[[arg1]] : tensor<1x22x39xf32>, tensor<1x22x39xi32>)
func.func @test_max_with_index(%arg0: tensor<1x22x39xbf16>, %arg1: tensor<1x22x39xi32>) -> (tensor<1x22xbf16>, tensor<1x22xi32>) {
  %0 = tensor.empty() : tensor<1x22xbf16>
  %1 = tensor.empty() : tensor<1x22xi32>
  %2:2 = hfusion.reduce_with_index {tie_break_left = true} <max> ins(%arg0, %arg1 : tensor<1x22x39xbf16>, tensor<1x22x39xi32>) 
                                         outs(%0, %1 : tensor<1x22xbf16>, tensor<1x22xi32>) dimensions = [2]  -> tensor<1x22xbf16>, tensor<1x22xi32>
  return %2#0, %2#1 : tensor<1x22xbf16>, tensor<1x22xi32>
}

// -----
// CHECK-LABEL: func.func @test_select_bf16
// CHECK: hfusion.select ins({{.*}} : tensor<16x32x4xi1>, tensor<16x32x4xbf16>, tensor<16x32x4xbf16>)
func.func @test_select_bf16(%arg0 : tensor<16x32x4xbf16>, %arg1 : tensor<16x32x4xbf16>, %arg2 : tensor<16x32x4xi1>) -> tensor<16x32x4xbf16> {
  %0 = tensor.empty() : tensor<16x32x4xbf16>
  %selected = hfusion.select ins(%arg2, %arg0, %arg1 : tensor<16x32x4xi1>, tensor<16x32x4xbf16>, tensor<16x32x4xbf16>) outs(%0 : tensor<16x32x4xbf16>) -> tensor<16x32x4xbf16>
  return %selected : tensor<16x32x4xbf16>
}
