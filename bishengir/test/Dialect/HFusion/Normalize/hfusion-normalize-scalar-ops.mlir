// RUN: bishengir-opt --hfusion-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_elemwise_unary_absi
// CHECK: %c5_i32 = arith.constant 5 : i32
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<absi>} ins(%c5_i32 : i32) outs(%arg0 : tensor<1xi32>) -> tensor<1xi32>
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_elemwise_unary_absi(%dst: tensor<1xi32>) -> tensor<1xi32> {
  %src = arith.constant dense<5> : tensor<1xi32>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<absi>} ins(%src : tensor<1xi32>) outs(%dst : tensor<1xi32>) -> tensor<1xi32>

  return %1 : tensor<1xi32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_elemwise_unary_absi_rank0
// CHECK: %c5_i32 = arith.constant 5 : i32
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<absi>} ins(%c5_i32 : i32) outs(%arg0 : tensor<i32>) -> tensor<i32>
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_elemwise_unary_absi_rank0(%dst: tensor<i32>) -> tensor<i32> {
  %src = arith.constant dense<5> : tensor<i32>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<absi>} ins(%src : tensor<i32>) outs(%dst : tensor<i32>) -> tensor<i32>

  return %1 : tensor<i32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_elemwise_binary
// CHECK: %cst = arith.constant 5.000000e-01 : f32
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%cst, %arg0 : f32, tensor<1xf32>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_elemwise_binary(%src: tensor<1xf32>) -> tensor<1xf32> {
  %cst = arith.constant dense<0.5> : tensor<1xf32>
  %dst = tensor.empty() : tensor<1xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%cst, %src : tensor<1xf32>, tensor<1xf32>) outs(%dst : tensor<1xf32>) -> tensor<1xf32>
  return %res : tensor<1xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_elemwise_binary_both_inputs
// CHECK-DAG: %[[HALF:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG: %[[QUARTER:.*]] = arith.constant 2.500000e-01 : f32
// CHECK: %0 = tensor.empty() : tensor<1xf32>
// CHECK: %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[HALF]], %[[QUARTER]] : f32, f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_elemwise_binary_both_inputs() -> tensor<1xf32> {
  %lhs = arith.constant dense<0.5> : tensor<1xf32>
  %rhs = arith.constant dense<0.25> : tensor<1xf32>
  %dst = tensor.empty() : tensor<1xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%lhs, %rhs : tensor<1xf32>, tensor<1xf32>) outs(%dst : tensor<1xf32>) -> tensor<1xf32>
  return %res : tensor<1xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_elemwise_binary_rank2_constant_no_rewrite
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%{{.*}}, %arg0 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%{{.*}} : tensor<1x1xf32>) -> tensor<1x1xf32>
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_elemwise_binary_rank2_constant_no_rewrite(%src: tensor<1x1xf32>) -> tensor<1x1xf32> {
  %cst = arith.constant dense<0.5> : tensor<1x1xf32>
  %dst = tensor.empty() : tensor<1x1xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%cst, %src : tensor<1x1xf32>, tensor<1x1xf32>) outs(%dst : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %res : tensor<1x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_compare
// CHECK: %c5_i64 = arith.constant 5 : i64
// CHECK: %0 = tensor.empty() : tensor<1xi1>
// CHECK: %1 = tensor.empty() : tensor<1xi1>
// CHECK: %2 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%arg0, %c5_i64 : tensor<1xi64>, i64) outs(%1 : tensor<1xi1>) -> tensor<1xi1>
// CHECK: %3 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%2 : tensor<1xi1>) outs(%0 : tensor<1xi1>) -> tensor<1xi1>
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_compare(%src: tensor<1xi64>) -> tensor<1xi1> {
  %cst = arith.constant dense<5> : tensor<1xi64>
  %dst = tensor.empty() : tensor<1xi1>
  %res = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
    ins(%src, %cst : tensor<1xi64>, tensor<1xi64>)
    outs(%dst : tensor<1xi1>) -> tensor<1xi1>
  return %res : tensor<1xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_compare_veq
// CHECK: %c7_i64 = arith.constant 7 : i64
// CHECK: %0 = tensor.empty() : tensor<1xi1>
// CHECK: %1 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%arg0, %c7_i64 : tensor<1xi64>, i64) outs(%0 : tensor<1xi1>) -> tensor<1xi1>
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_compare_veq(%src: tensor<1xi64>) -> tensor<1xi1> {
  %cst = arith.constant dense<7> : tensor<1xi64>
  %dst = tensor.empty() : tensor<1xi1>
  %res = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
    ins(%src, %cst : tensor<1xi64>, tensor<1xi64>)
    outs(%dst : tensor<1xi1>) -> tensor<1xi1>
  return %res : tensor<1xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_select
// CHECK: %c5_i32 = arith.constant 5 : i32
// CHECK: hfusion.select ins(%arg0, %c5_i32, %arg1 : memref<1xi1>, i32, memref<1xi32>) outs(%arg2 : memref<1xi32>)
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_select(
  %src1 : memref<1xi1>, %src3 : memref<1xi32>, %dst : memref<1xi32>) {
  %src2 = arith.constant dense<5> : memref<1xi32>
  hfusion.select
    ins(%src1, %src2, %src3 : memref<1xi1>, memref<1xi32>, memref<1xi32>)
    outs(%dst : memref<1xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_select_both_values
// CHECK: %c5_i32 = arith.constant 5 : i32
// CHECK: %c7_i32 = arith.constant 7 : i32
// CHECK: hfusion.select ins(%arg0, %c5_i32, %c7_i32 : memref<1xi1>, i32, i32) outs(%arg1 : memref<1xi32>)
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_select_both_values(
  %cond : memref<1xi1>, %dst : memref<1xi32>) {
  %trueValue = arith.constant dense<5> : memref<1xi32>
  %falseValue = arith.constant dense<7> : memref<1xi32>
  hfusion.select
    ins(%cond, %trueValue, %falseValue : memref<1xi1>, memref<1xi32>, memref<1xi32>)
    outs(%dst : memref<1xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_cast
// CHECK: %cst = arith.constant 0.869565188 : f32
// CHECK: %0 = tensor.empty() : tensor<f16>
// CHECK: %1 = hfusion.cast {{.*}} ins(%cst : f32) outs(%0 : tensor<f16>) -> tensor<f16>
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_cast() -> tensor<f16> {
    %src = arith.constant dense<0.86956521739130432> : tensor<f32>
    %dst = tensor.empty() : tensor<f16>
    %res = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%src : tensor<f32>) outs(%dst : tensor<f16>) -> tensor<f16>
    return %res : tensor<f16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_cast_rank1
// CHECK: %cst = arith.constant 1.250000e+00 : f32
// CHECK: %0 = tensor.empty() : tensor<1xf16>
// CHECK: %1 = hfusion.cast {{.*}} ins(%cst : f32) outs(%0 : tensor<1xf16>) -> tensor<1xf16>
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_cast_rank1() -> tensor<1xf16> {
    %src = arith.constant dense<1.25> : tensor<1xf32>
    %dst = tensor.empty() : tensor<1xf16>
    %res = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%src : tensor<1xf32>) outs(%dst : tensor<1xf16>) -> tensor<1xf16>
    return %res : tensor<1xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_cast_nonconstant_rank0_no_rewrite
// CHECK: hfusion.cast {{.*}} ins(%arg0 : tensor<f32>) outs(%0 : tensor<f16>) -> tensor<f16>
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_hfusion_cast_nonconstant_rank0_no_rewrite(%src : tensor<f32>) -> tensor<f16> {
    %dst = tensor.empty() : tensor<f16>
    %res = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%src : tensor<f32>) outs(%dst : tensor<f16>) -> tensor<f16>
    return %res : tensor<f16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_linalg_elemwise_unary_abs
// CHECK: %cst = arith.constant 5.100000e+00 : f32
// CHECK: %0 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%cst : f32) outs(%arg0 : tensor<f32>) -> tensor<f32>
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_linalg_elemwise_unary_abs(%output : tensor<f32>) -> tensor<f32> {
  %src = arith.constant dense<5.1> : tensor<f32>
  %0 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
          ins(%src: tensor<f32>) outs(%output: tensor<f32>) -> tensor<f32>
  return %0: tensor<f32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_linalg_elemwise_binary
// CHECK: %cst = arith.constant 5.000000e-01 : f32
// CHECK: %0 = tensor.empty() : tensor<1xf32>
// CHECK: %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%cst, %arg0 : f32, tensor<1xf32>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_linalg_elemwise_binary(%src: tensor<1xf32>) -> tensor<1xf32> {
  %cst = arith.constant dense<0.5> : tensor<1xf32>
  %dst = tensor.empty() : tensor<1xf32>
  %res = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%cst, %src : tensor<1xf32>, tensor<1xf32>) outs(%dst : tensor<1xf32>) -> tensor<1xf32>
  return %res : tensor<1xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_linalg_elemwise_binary_both_inputs
// CHECK-DAG: %[[HALF:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG: %[[QUARTER:.*]] = arith.constant 2.500000e-01 : f32
// CHECK: %0 = tensor.empty() : tensor<1xf32>
// CHECK: %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[HALF]], %[[QUARTER]] : f32, f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_linalg_elemwise_binary_both_inputs() -> tensor<1xf32> {
  %lhs = arith.constant dense<0.5> : tensor<1xf32>
  %rhs = arith.constant dense<0.25> : tensor<1xf32>
  %dst = tensor.empty() : tensor<1xf32>
  %res = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%lhs, %rhs : tensor<1xf32>, tensor<1xf32>) outs(%dst : tensor<1xf32>) -> tensor<1xf32>
  return %res : tensor<1xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_xori(
// CHECK-SAME: %[[VAL_0:.*]]: tensor<i8>) -> tensor<i8> {
// CHECK: %[[VAL_1:.*]] = arith.constant 127 : i8
// CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<i8>
// CHECK: %[[VAL_3:.*]] = hfusion.elemwise_binary {fun = {{.*}}<vor>} ins(%[[VAL_1]], %[[VAL_0]] : i8, tensor<i8>) outs(%[[VAL_2]] : tensor<i8>) -> tensor<i8>
// CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<i8>
// CHECK: %[[VAL_5:.*]] = hfusion.elemwise_binary {fun = {{.*}}<vand>} ins(%[[VAL_1]], %[[VAL_0]] : i8, tensor<i8>) outs(%[[VAL_4]] : tensor<i8>) -> tensor<i8>
// CHECK: %[[VAL_6:.*]] = hfusion.elemwise_unary {fun = {{.*}}<vnot>} ins(%[[VAL_5]] : tensor<i8>) outs(%[[VAL_5]] : tensor<i8>) -> tensor<i8>
// CHECK: %[[VAL_7:.*]] = tensor.empty() : tensor<i8>
// CHECK: %[[VAL_8:.*]] = hfusion.elemwise_binary {fun = {{.*}}<vand>} ins(%[[VAL_6]], %[[VAL_3]] : tensor<i8>, tensor<i8>) outs(%[[VAL_7]] : tensor<i8>) -> tensor<i8>
// CHECK: return %[[VAL_8]] : tensor<i8>
func.func @test_NormalizeScalarLikeTensor_scalar_like_tensor_conversion_xori(%arg0: tensor<i8>) -> tensor<i8> {
  %c127_i8 = arith.constant 127 : i8
  %0 = tensor.empty() : tensor<i8>
  %2 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vxor>} ins(%c127_i8, %arg0 : i8, tensor<i8>) outs(%0 : tensor<i8>) -> tensor<i8>
  return %2 : tensor<i8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeScalarLikeTensorLinalgBrc_scalar_like_tensor_conversion_linalg_brc
// CHECK: %cst = arith.constant 5.000000e-01 : f32
// CHECK: %0 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<4x1xf32>) -> tensor<4x1xf32>
func.func @test_NormalizeScalarLikeTensorLinalgBrc_scalar_like_tensor_conversion_linalg_brc(%init: tensor<4x1xf32>) -> tensor<4x1xf32> {
  %input = arith.constant dense<0.5> : tensor<1xf32>
  %res = linalg.broadcast
      ins(%input: tensor<1xf32>)
      outs(%init: tensor<4x1xf32>)
      dimensions = [0]
  func.return %res : tensor<4x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeScalarLikeTensorLinalgBrc_scalar_like_tensor_conversion_linalg_brc_rank0
// CHECK: %cst = arith.constant 1.250000e+00 : f32
// CHECK: %0 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<4x1xf32>) -> tensor<4x1xf32>
func.func @test_NormalizeScalarLikeTensorLinalgBrc_scalar_like_tensor_conversion_linalg_brc_rank0(%init: tensor<4x1xf32>) -> tensor<4x1xf32> {
  %input = arith.constant dense<1.25> : tensor<f32>
  %res = linalg.broadcast
      ins(%input: tensor<f32>)
      outs(%init: tensor<4x1xf32>)
      dimensions = [0, 1]
  func.return %res : tensor<4x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeScalarLikeTensorLinalgBrc_scalar_like_tensor_conversion_linalg_brc_rank2_constant_no_rewrite
// CHECK: linalg.broadcast
func.func @test_NormalizeScalarLikeTensorLinalgBrc_scalar_like_tensor_conversion_linalg_brc_rank2_constant_no_rewrite(%init: tensor<1x1x4x8xf32>) -> tensor<1x1x4x8xf32> {
  %input = arith.constant dense<0.5> : tensor<1x1xf32>
  %res = linalg.broadcast
      ins(%input: tensor<1x1xf32>)
      outs(%init: tensor<1x1x4x8xf32>)
      dimensions = [2, 3]
  func.return %res : tensor<1x1x4x8xf32>
}

// -----

// CHECK-LABEL: @test_NormalizeScalarLikeTensorLinalgBrc_scalar_like_tensor_conversion_linalg_brc_nondense_regbase(
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[EXTRACT:.*]] = tensor.extract %arg0[%[[C0]], %[[C0]]] : tensor<1x1xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[EXTRACT]] : f32) outs(%arg1 : tensor<1x1x4x8xf32>) -> tensor<1x1x4x8xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeScalarLikeTensorLinalgBrc_scalar_like_tensor_conversion_linalg_brc_nondense_regbase(%input: tensor<1x1xf32>, %init: tensor<1x1x4x8xf32>) -> tensor<1x1x4x8xf32> {
    %res = linalg.broadcast
        ins(%input: tensor<1x1xf32>)
        outs(%init: tensor<1x1x4x8xf32>)
        dimensions = [2, 3]
    return %res : tensor<1x1x4x8xf32>
  }
}

// -----

// CHECK-LABEL: @test_NormalizeScalarCast_normalize_cast_sacalar(
// CHECK: %{{.*}} = hfusion.cast {{.*}} ins(%{{.*}} : tensor<1xi16>) outs(%{{.*}} : tensor<1xi8>) -> tensor<1xi8>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2097152 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:89", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  func.func @test_NormalizeScalarCast_normalize_cast_sacalar(%arg0: tensor<i16>) -> tensor<i8> {
    %0 = tensor.empty() : tensor<i8>
    %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<i16>) outs(%0 : tensor<i8>) -> tensor<i8>
    return %1 : tensor<i8>
  }
}
