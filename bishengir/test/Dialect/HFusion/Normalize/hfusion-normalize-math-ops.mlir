// RUN: bishengir-opt --hfusion-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_NormalizeAnyToF32UnaryRec_normalize_rec_i16_to_f32(
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<1x2xi16>) outs({{.*}} : tensor<1x2xf32>)
// CHECK: hfusion.elemwise_unary
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<1x2xf32>) outs({{.*}} : tensor<1x2xi32>)
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<1x2xi32>) outs({{.*}} : tensor<1x2xi16>)
func.func @test_NormalizeAnyToF32UnaryRec_normalize_rec_i16_to_f32(%arg0 : tensor<1x2xi16>) -> tensor<1x2xi16> {
    %0 = tensor.empty() : tensor<1x2xi16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>, rec} ins(%arg0 : tensor<1x2xi16>) outs(%0 : tensor<1x2xi16>) -> tensor<1x2xi16>
    return %1 : tensor<1x2xi16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAnyToF32UnaryRec_normalize_rec_i32_to_f32(
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<1x2xi32>) outs({{.*}} : tensor<1x2xf32>)
// CHECK: hfusion.elemwise_unary
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<1x2xf32>) outs({{.*}} : tensor<1x2xi32>)
func.func @test_NormalizeAnyToF32UnaryRec_normalize_rec_i32_to_f32(%arg0 : tensor<1x2xi32>) -> tensor<1x2xi32> {
    %0 = tensor.empty() : tensor<1x2xi32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>, rec} ins(%arg0 : tensor<1x2xi32>) outs(%0 : tensor<1x2xi32>) -> tensor<1x2xi32>
    return %1 : tensor<1x2xi32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAnyToF32UnaryRec_normalize_rec_i64_to_f32(
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<1x2xi64>) outs({{.*}} : tensor<1x2xf32>)
// CHECK: hfusion.elemwise_unary
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<1x2xf32>) outs({{.*}} : tensor<1x2xi64>)
func.func @test_NormalizeAnyToF32UnaryRec_normalize_rec_i64_to_f32(%arg0 : tensor<1x2xi64>) -> tensor<1x2xi64> {
    %0 = tensor.empty() : tensor<1x2xi64>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>, rec} ins(%arg0 : tensor<1x2xi64>) outs(%0 : tensor<1x2xi64>) -> tensor<1x2xi64>
    return %1 : tensor<1x2xi64>
}

// -----

// CHECK-LABEL: @test_NormalizeAnyToF32UnaryRec_i8_elemwise_unary_rec
// CHECK: %[[cast0:.*]] = hfusion.cast
// CHECK: %[[cast1:.*]] = hfusion.cast
// CHECK: %[[unary_rec:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%[[cast1]] : tensor<16xf32>) outs({{.*}} : tensor<16xf32>)
// CHECK: hfusion.cast
// CHECK: hfusion.cast
func.func @test_NormalizeAnyToF32UnaryRec_i8_elemwise_unary_rec(%arg0: tensor<16xi8>) -> tensor<16xi8> {
  %dst = tensor.empty() : tensor<16xi8>
  %res = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>}
          ins(%arg0 : tensor<16xi8>)
          outs(%dst : tensor<16xi8>) -> tensor<16xi8>
  return %res : tensor<16xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeLogLike_hfusion_elemwise_unary_log2
// CHECK: %[[CSTTWO:.*]] : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[LOG_RES1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[ARG0:.*]] : tensor<1024xf32>) outs(%[[EMPTY0:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[CSTTWO:.*]] : f32) outs(%[[EMPTY0:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[LOG_RES2:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[FILL:.*]] : tensor<1024xf32>) outs(%[[EMPTY0:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[RES:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LOG_RES1:.*]], %[[LOG_RES2:.*]]: tensor<1024xf32>, tensor<1024xf32>) outs(%[[EMPTY1:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
func.func @test_NormalizeLogLike_hfusion_elemwise_unary_log2(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<log2>} ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeLogLike_hfusion_elemwise_unary_log10
// CHECK: %[[CSTTEN:.*]] : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[LOG_RES1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[ARG0:.*]] : tensor<1024xf32>) outs(%[[EMPTY0:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[CSTTEN:.*]] : f32) outs(%[[EMPTY0:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[LOG_RES2:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[FILL:.*]] : tensor<1024xf32>) outs(%[[EMPTY0:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[RES:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LOG_RES1:.*]], %[[LOG_RES2:.*]]: tensor<1024xf32>, tensor<1024xf32>) outs(%[[EMPTY1:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
func.func @test_NormalizeLogLike_hfusion_elemwise_unary_log10(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<log10>} ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeLogLike_hfusion_elemwise_unary_log2_f16
// CHECK: %[[CSTTWO:.*]] : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[CAST_RES1:.*]] = hfusion.cast {{.*}} ins(%[[ARG0:.*]] : tensor<1024xf16>) outs(%[[EMPTY1:.*]]  : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[LOG_RES1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[CAST_RES1:.*]] : tensor<1024xf32>) outs(%[[EMPTY2:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[CSTTWO:.*]] : f32) outs(%[[EMPTY2:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[LOG_RES2:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[FILL:.*]] : tensor<1024xf32>) outs(%[[EMPTY2:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[DIV_RES:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LOG_RES1:.*]], %[[LOG_RES2:.*]]: tensor<1024xf32>, tensor<1024xf32>) outs(%[[EMPTY3:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[CAST_RES2:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%[[DIV_RES:.*]] : tensor<1024xf32>) outs(%[[EMPTY0:.*]]  : tensor<1024xf16>) -> tensor<1024xf16>
func.func @test_NormalizeLogLike_hfusion_elemwise_unary_log2_f16(%arg0: tensor<1024xf16>) -> tensor<1024xf16> {
    %0 = tensor.empty() : tensor<1024xf16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<log2>} ins(%arg0 : tensor<1024xf16>) outs(%0 : tensor<1024xf16>) -> tensor<1024xf16>
    return %1 : tensor<1024xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeLogLike_hfusion_elemwise_unary_log10_f16
// CHECK: %[[CSTTEN:.*]] : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[CAST_RES1:.*]] = hfusion.cast {{.*}} ins(%[[ARG0:.*]] : tensor<1024xf16>) outs(%[[EMPTY1:.*]]  : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[LOG_RES1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[CAST_RES1:.*]] : tensor<1024xf32>) outs(%[[EMPTY2:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[CSTTEN:.*]] : f32) outs(%[[EMPTY2:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[LOG_RES2:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[FILL:.*]] : tensor<1024xf32>) outs(%[[EMPTY2:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[DIV_RES:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LOG_RES1:.*]], %[[LOG_RES2:.*]]: tensor<1024xf32>, tensor<1024xf32>) outs(%[[EMPTY3:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[CAST_RES2:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%[[DIV_RES:.*]] : tensor<1024xf32>) outs(%[[EMPTY0:.*]]  : tensor<1024xf16>) -> tensor<1024xf16>
func.func @test_NormalizeLogLike_hfusion_elemwise_unary_log10_f16(%arg0: tensor<1024xf16>) -> tensor<1024xf16> {
    %0 = tensor.empty() : tensor<1024xf16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<log10>} ins(%arg0 : tensor<1024xf16>) outs(%0 : tensor<1024xf16>) -> tensor<1024xf16>
    return %1 : tensor<1024xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeLog1p_hfusion_elemwise_unary_log1p
// CHECK: %[[CSTONE:.*]] : f32
// CHECK: %[[EMPTYO:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[ADD_RES:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[ARG0:.*]],  %[[CSTONE:.*]] : tensor<1024xf32>, f32) outs(%[[EMPTY0:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[RES:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[ADD_RES:.*]] : tensor<1024xf32>) outs(%[[EMPTY1:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
func.func @test_NormalizeLog1p_hfusion_elemwise_unary_log1p(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<log1p>} ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeLog1p_hfusion_elemwise_unary_log1p_f16
// CHECK: %[[CSTONE:.*]] : f16
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[ADD_RES:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[ARG0:.*]], %[[CSTONE:.*]] : tensor<1024xf16>, f16) outs(%[[EMPTY0:.*]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[CAST_IN:.*]] = hfusion.cast {{.*}} ins(%[[ADD_RES:.*]] : tensor<1024xf16>) outs(%[[EMPTY2:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[CAST_OUT:.*]] = hfusion.cast {{.*}} ins(%[[EMPTY1:.*]] : tensor<1024xf16>) outs(%[[EMPTY3:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[LOG:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[CAST_IN:.*]] : tensor<1024xf32>) outs(%[[CAST_OUT:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY4:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[RES:.*]] = hfusion.cast {{.*}} ins(%[[LOG:.*]] : tensor<1024xf32>) outs(%[[EMPTY4:.*]] : tensor<1024xf16>) -> tensor<1024xf16>
func.func @test_NormalizeLog1p_hfusion_elemwise_unary_log1p_f16(%arg0: tensor<1024xf16>) -> tensor<1024xf16> {
    %0 = tensor.empty() : tensor<1024xf16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<log1p>} ins(%arg0 : tensor<1024xf16>) outs(%0 : tensor<1024xf16>) -> tensor<1024xf16>
    return %1 : tensor<1024xf16>
}

// -----

// CHECK-LABEL: @test_NormalizeF16ToF32Type_normalize_vlog_f16_to_f32
// CHECK: %[[a0:.*]] = tensor.empty() : tensor<17x256xf16>
// CHECK: %[[a1:.*]] = tensor.empty() : tensor<17x256xf32>
// CHECK: %[[a2:.*]] = hfusion.cast {{.*}} ins(%[[arg0:.*]] : tensor<17x256xf16>) outs(%[[a1]] : tensor<17x256xf32>) -> tensor<17x256xf32>
// CHECK: %[[a3:.*]] = tensor.empty() : tensor<17x256xf32>
// CHECK: %[[a4:.*]] = hfusion.cast {{.*}} ins(%[[a0]] : tensor<17x256xf16>) outs(%[[a3]] : tensor<17x256xf32>) -> tensor<17x256xf32>
// CHECK: %[[a5:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[a2]] : tensor<17x256xf32>) outs(%[[a4]] : tensor<17x256xf32>) -> tensor<17x256xf32>
// CHECK: %[[a6:.*]] = tensor.empty() : tensor<17x256xf16>
// CHECK: %[[a7:.*]] = hfusion.cast {{.*}} ins(%[[a5]] : tensor<17x256xf32>) outs(%[[a6]] : tensor<17x256xf16>) -> tensor<17x256xf16>
func.func @test_NormalizeF16ToF32Type_normalize_vlog_f16_to_f32(%arg0: tensor<17x256xf16>) -> tensor<17x256xf16> {
  %0 = tensor.empty() : tensor<17x256xf16>
  %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg0 : tensor<17x256xf16>) outs(%0 : tensor<17x256xf16>) -> tensor<17x256xf16>
  return %1 : tensor<17x256xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeExp2_hfusion_elemwise_unary_exp2
// CHECK: %[[CSTLNTWO:.*]] : f32
// CHECK: %[[EMPTYO:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[MUL_RES1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ARG0:.*]],  %[[CSTLNTWO:.*]]: tensor<1024xf32>, f32) outs(%[[EMPTYO:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[RES:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[MUL_RES1:.*]] : tensor<1024xf32>) outs(%[[EMPTY1:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
func.func @test_NormalizeExp2_hfusion_elemwise_unary_exp2(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<exp2>} ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeExp2_hfusion_elemwise_unary_exp2_f16
// CHECK: %[[cast0:.*]] = hfusion.cast {{.*}} ins({{.*}} : tensor<1024xf16>) outs({{.*}} : tensor<1024xf32>)
// CHECK: %[[mul:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[cast0]]
// CHECK: %[[exp:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[mul]] : tensor<1024xf32>)
// CHECK: %[[cast1:.*]] = hfusion.cast {{.*}} ins(%[[exp]] : tensor<1024xf32>) outs({{.*}} : tensor<1024xf16>)
func.func @test_NormalizeExp2_hfusion_elemwise_unary_exp2_f16(%arg0: tensor<1024xf16>) -> tensor<1024xf16> {
    %0 = tensor.empty() : tensor<1024xf16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<exp2>} ins(%arg0 : tensor<1024xf16>) outs(%0 : tensor<1024xf16>) -> tensor<1024xf16>
    return %1 : tensor<1024xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeExpM1_hfusion_elemwise_unary_expm1
// CHECK: %[[CSTLNTWO:.*]] : f32
// CHECK: %[[EMPTYO:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[EXP_RES1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[ARG0:.*]] : tensor<1024xf32>) outs(%[[EMPTYO:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[RES:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[EXP_RES1:.*]], %[[CSTLNTWO:.*]] : tensor<1024xf32>, f32) outs(%[[EMPTY1:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
func.func @test_NormalizeExpM1_hfusion_elemwise_unary_expm1(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<expm1>} ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeExpM1_hfusion_elemwise_unary_expm1_f16
// CHECK: %[[cast0:.*]] = hfusion.cast {{.*}} ins({{.*}} : tensor<1024xf16>) outs({{.*}} : tensor<1024xf32>)
// CHECK: %[[exp:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[cast0:.*]] : tensor<1024xf32>)
// CHECK: %[[sub:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[exp]]
// CHECK: %[[cast1:.*]] = hfusion.cast {{.*}} ins(%[[sub]] : tensor<1024xf32>) outs({{.*}} : tensor<1024xf16>)
func.func @test_NormalizeExpM1_hfusion_elemwise_unary_expm1_f16(%arg0: tensor<1024xf16>) -> tensor<1024xf16> {
    %0 = tensor.empty() : tensor<1024xf16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<expm1>} ins(%arg0 : tensor<1024xf16>) outs(%0 : tensor<1024xf16>) -> tensor<1024xf16>
    return %1 : tensor<1024xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizePowf_hfusion_powf_f32
// CHECK: %cst = arith.constant 0x7F800000 : f32
// CHECK: %[[cst:.*]] = arith.constant 2.13909504E+9 : f32
// CHECK: %[[cst_nan:.*]] = arith.constant 0x7FC00000 : f32
// CHECK: %[[cst_1:.*]] = arith.constant -2.000000e+00 : f32
// CHECK: %[[cst_2:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[cst_3:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[c_1_i32:.*]] = arith.constant -1 : i32
// CHECK: %[[empty0:.*]] = tensor.empty() : tensor<16xi32>
// CHECK: %[[bitcast:.*]] = hfusion.bitcast ins(%arg0 : tensor<16xf32>) outs(%[[empty0:.*]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[empty1:.*]] = tensor.empty() : tensor<16xi32>
// CHECK: %[[shift:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%[[bitcast:.*]],  %[[c31_i32:.*]] : tensor<16xi32>, i32) outs(%[[empty1]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[empty2:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq0:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[shift:.*]], %[[c1_i32:.*]] : tensor<16xi32>, i32) outs(%[[empty2]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty4:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[cast1:.*]] = hfusion.cast {{.*}} ins(%arg1 : tensor<16xf32>) outs(%[[empty4]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty5:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq1:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[cast1]], %arg1 : tensor<16xf32>, tensor<16xf32>) outs(%[[empty5]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty6:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vand0:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[cmp_eq0]], %[[cmp_eq1]] : tensor<16xi1>, tensor<16xi1>) outs(%[[empty6]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty7:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty8:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty9:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs0:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%arg1 : tensor<16xf32>) outs(%[[empty9]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty10:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[mul0:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<divfhp>} ins(%[[abs0]], %[[cst_2]] : tensor<16xf32>, f32) outs(%[[empty10]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty11:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[cast2:.*]] = hfusion.cast {{.*}} ins(%[[mul0]] : tensor<16xf32>) outs(%[[empty11]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty12:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[mul1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[cast2]], %[[cst_2]] : tensor<16xf32>, f32) outs(%[[empty12]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty13:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[sub:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[abs0]], %[[mul1]] : tensor<16xf32>, tensor<16xf32>) outs(%[[empty13]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[mod_output:.*]] = hfusion.select ins({{.*}}, {{.*}}, %[[sub]] : tensor<16xi1>, f32, tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty19:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[mul2:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[mod_output]], %[[cst_1]] : tensor<16xf32>, f32) outs(%[[empty19]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty20:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[add1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[mul2]], %[[cst_3]] : tensor<16xf32>, f32) outs(%[[empty20]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty21:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%arg0 : tensor<16xf32>) outs(%[[empty21]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[log0:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[abs1]] : tensor<16xf32>) outs(%[[empty7]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[mul3:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[log0]], %arg1 : tensor<16xf32>, tensor<16xf32>) outs(%[[empty8]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty22:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[exp0:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[mul3]] : tensor<16xf32>) outs(%[[empty22]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty23:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[mul4:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[exp0]], %[[add1]] : tensor<16xf32>, tensor<16xf32>) outs(%[[empty23]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty24:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty25:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty26:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty27:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs2:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%arg0 : tensor<16xf32>) outs(%[[empty27]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[log1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[abs2]] : tensor<16xf32>) outs(%[[empty24]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[mul5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[log1]], %arg1 : tensor<16xf32>, tensor<16xf32>) outs(%[[empty25]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[exp1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[mul5]] : tensor<16xf32>) outs(%[[empty26]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty28:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[select0:.*]] = hfusion.select ins(%[[vand0]], %[[mul4]], %[[exp1]] : tensor<16xi1>, tensor<16xf32>, tensor<16xf32>) outs(%[[empty28]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty29:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs3:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%arg0 : tensor<16xf32>) outs(%[[empty29]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty30:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq3:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[abs3]], %[[cst_3]] : tensor<16xf32>, f32) outs(%[[empty30]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty31:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs4:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%arg1 : tensor<16xf32>) outs(%[[empty31]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty32:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq4:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[abs4]], %[[cst]] : tensor<16xf32>, f32) outs(%[[empty32]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty33:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vand1:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[cmp_eq3]], %[[cmp_eq4]] : tensor<16xi1>, tensor<16xi1>) outs(%[[empty33]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty34:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[select1:.*]] = hfusion.select ins(%[[vand1]], %[[cst_3]], %[[select0]] : tensor<16xi1>, f32, tensor<16xf32>) outs(%[[empty34]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[cmp_lt0:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}
// CHECK: %[[abs5:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
// CHECK: %[[cmp_eq6:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
// CHECK: %[[vnot0:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
// CHECK: %[[vand2:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
// CHECK: %[[abs6:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
// CHECK: %[[cmp_eq7:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
// CHECK: %[[vnot1:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
// CHECK: %[[cast3:.*]] = hfusion.cast {{.*}}
// CHECK: %[[cmp_eq8:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
// CHECK: %[[vnot2:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
// CHECK: %[[vand3:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
// CHECK: %[[vand4:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
// CHECK: %[[select2:.*]] = hfusion.select
// CHECK: %[[cmp_eq9:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
// CHECK: %[[select3:.*]] = hfusion.select
// CHECK: return %[[select3]] : tensor<16xf32>
func.func @test_NormalizePowf_hfusion_powf_f32(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32>{
  %0 = tensor.empty(): tensor<16xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0, %arg1: tensor<16xf32>, tensor<16xf32>) outs(%0: tensor<16xf32>) -> tensor<16xf32>
  return %res : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizePowf_hfusion_powf_f16
// CHECK: %[[cst:.*]] = arith.constant 2.13909504E+9 : f32
// CHECK: %[[c1_i32:.*]] = arith.constant 1 : i32
// CHECK: %[[c1_neg_i32:.*]] = arith.constant -1 : i32
// CHECK: %[[c31_i32:.*]] = arith.constant 31 : i32
// CHECK: %[[tmp0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[cast0_f32:.*]] = hfusion.cast {{.*}} ins(%arg0 : tensor<16xf16>) outs(%[[tmp0:.*]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[tmp1:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[cast1_f32:.*]] = hfusion.cast {{.*}} ins(%arg1 : tensor<16xf16>) outs(%[[tmp1:.*]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty0:.*]] = tensor.empty() : tensor<16xi32>
// CHECK: %[[bitcast:.*]] = hfusion.bitcast ins(%[[cast0_f32:.*]] : tensor<16xf32>) outs(%[[empty0:.*]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[empty1:.*]] = tensor.empty() : tensor<16xi32>
// CHECK: %[[shift:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%[[bitcast:.*]],  %[[c31_i32:.*]] : tensor<16xi32>, i32) outs(%[[empty1]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[empty2:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq0:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[shift:.*]], %[[c1_neg_i32:.*]] : tensor<16xi32>, i32) outs(%[[empty2]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty4:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[cast1:.*]] = hfusion.cast {{.*}} ins(%[[cast1_f32:.*]] : tensor<16xf32>) outs(%[[empty4]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty5:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq1:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[cast1]], %[[cast1_f32:.*]] : tensor<16xf32>, tensor<16xf32>) outs(%[[empty5]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty6:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vand0:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[cmp_eq0]], %[[cmp_eq1]] : tensor<16xi1>, tensor<16xi1>) outs(%[[empty6]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty7:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty8:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty9:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs0:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[cast1_f32:.*]] : tensor<16xf32>) outs(%[[empty9]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty10:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[mul0:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<divfhp>} ins(%[[abs0]], {{.*}} : tensor<16xf32>, f32) outs(%[[empty10]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty11:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[cast2:.*]] = hfusion.cast {{.*}} ins(%[[mul0]] : tensor<16xf32>) outs(%[[empty11]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty12:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[mul1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[cast2]], {{.*}} : tensor<16xf32>, f32) outs(%[[empty12]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty13:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[sub:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[abs0]], %[[mul1]] : tensor<16xf32>, tensor<16xf32>) outs(%[[empty13]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[mod_output:.*]] = hfusion.select ins({{.*}}, {{.*}}, %[[sub]] : tensor<16xi1>, f32, tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty19:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[mul2:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[mod_output]], {{.*}} : tensor<16xf32>, f32) outs(%[[empty19]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty20:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[add1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[mul2]], {{.*}} : tensor<16xf32>, f32) outs(%[[empty20]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty21:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[cast0_f32:.*]] : tensor<16xf32>) outs(%[[empty21]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[log0:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[abs1]] : tensor<16xf32>) outs(%[[empty7]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[mul3:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[log0]], %[[cast1_f32:.*]] : tensor<16xf32>, tensor<16xf32>) outs(%[[empty8]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty22:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[exp0:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[mul3]] : tensor<16xf32>) outs(%[[empty22]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty23:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[mul4:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[exp0]], %[[add1]] : tensor<16xf32>, tensor<16xf32>) outs(%[[empty23]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty24:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty25:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty26:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty27:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs2:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[cast0_f32:.*]] : tensor<16xf32>) outs(%[[empty27]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[log1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[abs2]] : tensor<16xf32>) outs(%[[empty24]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[mul5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[log1]], %[[cast1_f32:.*]] : tensor<16xf32>, tensor<16xf32>) outs(%[[empty25]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[exp1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[mul5]] : tensor<16xf32>) outs(%[[empty26]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty28:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[select0:.*]] = hfusion.select ins(%[[vand0]], %[[mul4]], %[[exp1]] : tensor<16xi1>, tensor<16xf32>, tensor<16xf32>) outs(%[[empty28]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty29:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs3:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[cast0_f32:.*]] : tensor<16xf32>) outs(%[[empty29]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty30:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq3:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[abs3]], {{.*}} : tensor<16xf32>, f32) outs(%[[empty30]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty31:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs4:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[cast1_f32:.*]] : tensor<16xf32>) outs(%[[empty31]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty32:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq4:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[abs4]], %[[cst]] : tensor<16xf32>, f32) outs(%[[empty32]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty33:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vand1:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[cmp_eq3]], %[[cmp_eq4]] : tensor<16xi1>, tensor<16xi1>) outs(%[[empty33]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty34:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[select1:.*]] = hfusion.select ins(%[[vand1]], {{.*}}, %[[select0]] : tensor<16xi1>, f32, tensor<16xf32>) outs(%[[empty34]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty37:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_lt0:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}
// CHECK: %[[empty38:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs5:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
// CHECK: %[[empty39:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq6:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
// CHECK: %[[empty40:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vnot0:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
// CHECK: %[[empty41:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vand2:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
// CHECK: %[[empty42:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs6:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
// CHECK: %[[empty43:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq7:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
// CHECK: %[[empty44:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vnot1:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
// CHECK: %[[empty45:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[cast3:.*]] = hfusion.cast {{.*}}
// CHECK: %[[empty46:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq8:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
// CHECK: %[[empty47:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vnot2:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
// CHECK: %[[empty48:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vand3:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
// CHECK: %[[empty49:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vand4:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
// CHECK: %[[select2:.*]] = hfusion.select
// CHECK: %[[empty50:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq9:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
// CHECK: %[[select3:.*]] = hfusion.select
// CHECK: %[[res:.*]] = hfusion.cast {{.*}} ins(%[[select3:.*]] : tensor<16xf32>) outs(%[[empty30:.*]] : tensor<16xf16>) -> tensor<16xf16>
// CHECK: return %[[res:.*]] : tensor<16xf16>
func.func @test_NormalizePowf_hfusion_powf_f16(%arg0: tensor<16xf16>, %arg1: tensor<16xf16>) -> tensor<16xf16>{
  %0 = tensor.empty(): tensor<16xf16>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0, %arg1: tensor<16xf16>, tensor<16xf16>) outs(%0: tensor<16xf16>) -> tensor<16xf16>
  return %res : tensor<16xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizePowf_hfusion_powf_half
// CHECK: %[[empty0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[res:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[arg0:.*]]: tensor<16xf32>) outs(%[[empty0:.*]] : tensor<16xf32>) -> tensor<16xf32>
func.func @test_NormalizePowf_hfusion_powf_half(%arg0: tensor<16xf32>) -> tensor<16xf32>{
  %0 = tensor.empty(): tensor<16xf32>
  %cst_1 = arith.constant 0.500000e+00 : f32
  %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0,  %1: tensor<16xf32>, tensor<16xf32>) outs(%0: tensor<16xf32>) -> tensor<16xf32>
  return %res : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizePowf_hfusion_powf_const_dense
// CHECK: %[[empty0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[res:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[arg0:.*]]: tensor<16xf32>) outs(%[[empty0:.*]] : tensor<16xf32>) -> tensor<16xf32>
func.func @test_NormalizePowf_hfusion_powf_const_dense(%arg0: tensor<16xf32>) -> tensor<16xf32>{
  %0 = tensor.empty(): tensor<16xf32>
  %cst_dense = arith.constant dense<0.500000e+00> : tensor<16xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0, %cst_dense: tensor<16xf32>, tensor<16xf32>) outs(%0: tensor<16xf32>) -> tensor<16xf32>
  return %res : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizePowf_hfusion_powf_cast_fill
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>}
func.func @test_NormalizePowf_hfusion_powf_cast_fill(%arg0: tensor<16xf32>) -> tensor<16xf32>{
  %0 = tensor.empty(): tensor<16xf32>
  %cst_1 = arith.constant 0.5 : f16
  %1 = tensor.empty(): tensor<16xf16>
  %2 = linalg.fill ins(%cst_1 : f16) outs(%1 : tensor<16xf16>) -> tensor<16xf16>
  %3 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%2 : tensor<16xf16>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0, %3: tensor<16xf32>, tensor<16xf32>) outs(%0: tensor<16xf32>) -> tensor<16xf32>
  return %res : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizePowf_hfusion_powf_5
// CHECK: %[[cst_5:.*]] = arith.constant 5.000000e+00 : f32
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[fill:.*]] = linalg.fill ins(%[[cst_5:.*]] : f32) outs(%[[empty:.*]] : tensor<1xf32>) -> tensor<1xf32>
func.func @test_NormalizePowf_hfusion_powf_5(%arg0: tensor<1xf32>) -> tensor<1xf32>{
  %0 = tensor.empty(): tensor<1xf32>
  %cst_5 = arith.constant dense<5.000000e+00> : tensor<1xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0, %cst_5: tensor<1xf32>, tensor<1xf32>) outs(%0: tensor<1xf32>) -> tensor<1xf32>
  return %res : tensor<1xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizePowf_hfusion_powf_0
// CHECK: %[[cst_0:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[res:.*]] = linalg.fill ins(%[[cst_0:.*]] : f32) outs(%[[empty:.*]] : tensor<1xf32>) -> tensor<1xf32>
func.func @test_NormalizePowf_hfusion_powf_0(%arg0: tensor<1xf32>) -> tensor<1xf32>{
  %0 = tensor.empty(): tensor<1xf32>
  %cst_0 = arith.constant dense<0.0> : tensor<1xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0, %cst_0: tensor<1xf32>, tensor<1xf32>) outs(%0: tensor<1xf32>) -> tensor<1xf32>
  return %res : tensor<1xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizePowf_hfusion_powf_2
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[res:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[arg0:.*]], %[[arg0]] : tensor<1xf32>, tensor<1xf32>) outs(%[[empty:.*]] : tensor<1xf32>) -> tensor<1xf32>
func.func @test_NormalizePowf_hfusion_powf_2(%arg0: tensor<1xf32>) -> tensor<1xf32>{
  %0 = tensor.empty(): tensor<1xf32>
  %cst_2 = arith.constant dense<2.0> : tensor<1xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0, %cst_2: tensor<1xf32>, tensor<1xf32>) outs(%0: tensor<1xf32>) -> tensor<1xf32>
  return %res : tensor<1xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizePowf_hfusion_powf_cast_fill
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>}
module attributes {hacc.target = #hacc.target<"Ascend310B4">} {
  func.func @test_NormalizePowf_hfusion_powf_cast_fill(%arg0: tensor<16xf32>) -> tensor<16xf32>{
    %0 = tensor.empty(): tensor<16xf32>
    %cst_1 = arith.constant 0.5 : f16
    %1 = tensor.empty(): tensor<16xf16>
    %2 = linalg.fill ins(%cst_1 : f16) outs(%1 : tensor<16xf16>) -> tensor<16xf16>
    %3 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%2 : tensor<16xf16>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0, %3: tensor<16xf32>, tensor<16xf32>) outs(%0: tensor<16xf32>) -> tensor<16xf32>
    return %res : tensor<16xf32>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeVPowi_normalize_hfusion_powi_i64
func.func @test_NormalizeVPowi_normalize_hfusion_powi_i64(%arg0 : tensor<4x2x32xi64>, %arg1 : tensor<4x2x32xi64>) -> tensor<4x2x32xi64> {
  %0 = tensor.empty() : tensor<4x2x32xi64>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powi>} ins(%arg0,  %arg1: tensor<4x2x32xi64>, tensor<4x2x32xi64>) outs(%0: tensor<4x2x32xi64>) -> tensor<4x2x32xi64>
  return %1 : tensor<4x2x32xi64>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeVPowi_normalize_hfusion_powi_i16
// CHECK: %[[VAL_0:.*]] = hfusion.cast {{.*}} ins(%arg0 : tensor<4x2x32xi16>) outs(%[[empty0:.*]] : tensor<4x2x32xf32>) -> tensor<4x2x32xf32>
// CHECK: %[[VAL_1:.*]] = hfusion.cast {{.*}} ins(%arg1 : tensor<4x2x32xi16>) outs(%[[empty1:.*]] : tensor<4x2x32xf32>) -> tensor<4x2x32xf32>
// CHECK: %[[Result:.*]] = hfusion.cast {{.*}} ins(%[[empty2:.*]] : tensor<4x2x32xf32>) outs(%[[empty4:.*]] : tensor<4x2x32xi32>) -> tensor<4x2x32xi32>
// CHECK: %[[Result:.*]] = hfusion.cast {{.*}} ins(%[[empty4:.*]] : tensor<4x2x32xi32>) outs(%[[empty3:.*]] : tensor<4x2x32xi16>) -> tensor<4x2x32xi16>
// CHECK: return %[[Result]] : tensor<4x2x32xi16>
func.func @test_NormalizeVPowi_normalize_hfusion_powi_i16(%arg0 : tensor<4x2x32xi16>, %arg1 : tensor<4x2x32xi16>) -> tensor<4x2x32xi16> {
  %0 = tensor.empty() : tensor<4x2x32xi16>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powi>} ins(%arg0,  %arg1: tensor<4x2x32xi16>, tensor<4x2x32xi16>) outs(%0: tensor<4x2x32xi16>) -> tensor<4x2x32xi16>
  return %1 : tensor<4x2x32xi16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeVPowi_normalize_hfusion_powi_i8
// CHECK: %[[RESULT:.*]] = hfusion.cast {{.*}} -> tensor<4x2x32xi8>
// CHECK: return %[[RESULT]] : tensor<4x2x32xi8>
func.func @test_NormalizeVPowi_normalize_hfusion_powi_i8(%arg0 : tensor<4x2x32xi8>, %arg1 : tensor<4x2x32xi8>) -> tensor<4x2x32xi8> {
  %0 = tensor.empty() : tensor<4x2x32xi8>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powi>} ins(%arg0,  %arg1: tensor<4x2x32xi8>, tensor<4x2x32xi8>) outs(%0: tensor<4x2x32xi8>) -> tensor<4x2x32xi8>
  return %1 : tensor<4x2x32xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeNegToMul_linalg_negf_mul
// CHECK-SAME: (%[[arg0:.*]]: tensor<5x1xf32>)
// CHECK: %[[CST:.*]]: f32
// CHECK: %[[ZERO:.*]] : tensor<5x1xf32
// CHECK: %[[ONE:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[arg0:.*]], %[[CST:.*]]: tensor<5x1xf32>, f32) outs(%[[ZERO:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: return %[[ONE:.*]]
func.func @test_NormalizeNegToMul_linalg_negf_mul(%src: tensor<5x1xf32>) -> tensor<5x1xf32> {
  %x = tensor.empty() : tensor<5x1xf32>
  %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%src : tensor<5x1xf32>) outs(%x : tensor<5x1xf32>) -> tensor<5x1xf32>
  return %1 : tensor<5x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCeilandFloor_linalg_floor_to_hfusion_cast
// CHECK-SAME: (%[[arg0:.*]]: tensor<1024xf16>)
// CHECK: %[[DST0:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[DST1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[RES0:.*]] = hfusion.cast {{.*}} ins(%[[arg0]] : tensor<1024xf16>) outs(%[[DST1]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[DST2:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[RES1:.*]] = hfusion.cast {{.*}} ins(%[[RES0]] : tensor<1024xf32>) outs(%[[DST2]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[RES2:.*]] = hfusion.cast {{.*}} ins(%[[RES1]] : tensor<1024xf32>) outs(%[[DST0]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: return %[[RES2]]
func.func @test_NormalizeCeilandFloor_linalg_floor_to_hfusion_cast(%src: tensor<1024xf16>) -> tensor<1024xf16> {
    %dst = tensor.empty() : tensor<1024xf16>
    %res = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>} ins(%src : tensor<1024xf16>) outs(%dst : tensor<1024xf16>) -> tensor<1024xf16>
   return %res : tensor<1024xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCeilandFloor_linalg_ceil_to_hfusion_cast_f32
// CHECK-SAME: (%[[arg0:.*]]: tensor<1024xf32>)
// CHECK: %[[DST0:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[RES0:.*]] = hfusion.cast {{.*}}round_mode = #hfusion.round_mode<ceil>{{.*}} ins(%[[arg0]] : tensor<1024xf32>) outs(%[[DST0]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: return %[[RES0]]
func.func @test_NormalizeCeilandFloor_linalg_ceil_to_hfusion_cast_f32(%src: tensor<1024xf32>) -> tensor<1024xf32> {
    %dst = tensor.empty() : tensor<1024xf32>
    %res = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%src : tensor<1024xf32>) outs(%dst : tensor<1024xf32>) -> tensor<1024xf32>
   return %res : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCeilandFloor_linalg_floor_to_hfusion_cast
// CHECK-SAME: (%[[arg0:.*]]: tensor<1024xf16>)
// CHECK: %[[DST0:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[RES1:.*]] = hfusion.cast {{.*}} ins(%[[arg0]] : tensor<1024xf16>) outs(%[[DST0]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: return %[[RES1]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeCeilandFloor_linalg_floor_to_hfusion_cast(%src: tensor<1024xf16>) -> tensor<1024xf16> {
      %dst = tensor.empty() : tensor<1024xf16>
      %res = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>} ins(%src : tensor<1024xf16>) outs(%dst : tensor<1024xf16>) -> tensor<1024xf16>
    return %res : tensor<1024xf16>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeMod_hfusion_mod_f32
// CHECK-SAME: (%[[x:.*]]: tensor<2048xf32>, %[[y:.*]]: tensor<2048xf32>)
// CHECK: %[[div:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<divfhp>} ins(%[[x]], %[[y]] : tensor<2048xf32>, tensor<2048xf32>) outs({{.*}} : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK: %[[div_trunc:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<trunc>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%[[div]] : tensor<2048xf32>) outs({{.*}} : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK: %[[mul:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[div_trunc]], %[[y]] : tensor<2048xf32>, tensor<2048xf32>) outs({{.*}} : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK: %[[rem:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[x]], %[[mul]] : tensor<2048xf32>, tensor<2048xf32>) outs({{.*}} : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK: %[[output:.*]] = hfusion.select ins({{.*}}, {{.*}}, %[[rem]] : tensor<2048xi1>, f32, tensor<2048xf32>) outs({{.*}} : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK: return %[[output]] : tensor<2048xf32>
func.func @test_NormalizeMod_hfusion_mod_f32(%src0: tensor<2048xf32>, %src1: tensor<2048xf32>) -> tensor<2048xf32> {
    %3 = tensor.empty() : tensor<2048xf32>
    %4 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%src0, %src1 : tensor<2048xf32>, tensor<2048xf32>) outs(%3 : tensor<2048xf32>) -> tensor<2048xf32>
    return %4 : tensor<2048xf32>
}

// CHECK-LABEL: func.func @test_NormalizeMod_hfusion_mod_bool
// CHECK: %[[false_const:.*]] = arith.constant false
// CHECK: %[[fill:.*]] = linalg.fill
// CHECK: return %[[fill]]
func.func @test_NormalizeMod_hfusion_mod_bool(%src0: tensor<2048xi1 >, %src1: tensor<2048xi1 >) -> tensor<2048xi1 > {
    %3 = tensor.empty() : tensor<2048xi1 >
    %4 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%src0, %src1 : tensor<2048xi1 >, tensor<2048xi1 >) outs(%3 : tensor<2048xi1 >) -> tensor<2048xi1 >
    return %4 : tensor<2048xi1 >
}

// -----

// CHECK-LABEL: func.func @test_NormalizeMod_hfusion_mod_f16
// CHECK-SAME: (%[[x:.*]]: tensor<2048xf16>, %[[y:.*]]: tensor<2048xf16>)
// CHECK: %[[x_cast:.*]] = hfusion.cast {{.*}} ins(%[[x]] : tensor<2048xf16>) outs({{.*}} : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK: %[[y_cast:.*]] = hfusion.cast {{.*}} ins(%[[y]] : tensor<2048xf16>) outs({{.*}} : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK: %[[div:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<divfhp>} ins(%[[x_cast]], %[[y_cast]] : tensor<2048xf32>, tensor<2048xf32>) outs({{.*}} : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK: %[[div_trunc:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<trunc>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%[[div]] : tensor<2048xf32>) outs({{.*}} : tensor<2048xf16>) -> tensor<2048xf16>
// CHECK: %[[mul:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[div_trunc]], %[[y]] : tensor<2048xf16>, tensor<2048xf16>) outs({{.*}} : tensor<2048xf16>) -> tensor<2048xf16>
// CHECK: %[[rem:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[x]], %[[mul]] : tensor<2048xf16>, tensor<2048xf16>) outs({{.*}} : tensor<2048xf16>) -> tensor<2048xf16>
// CHECK: %[[output:.*]] = hfusion.select ins({{.*}}, {{.*}}, %[[rem]] : tensor<2048xi1>, f16, tensor<2048xf16>) outs({{.*}} : tensor<2048xf16>) -> tensor<2048xf16>
// CHECK: return %[[output]] : tensor<2048xf16>
func.func @test_NormalizeMod_hfusion_mod_f16(%src0: tensor<2048xf16>, %src1: tensor<2048xf16>) -> tensor<2048xf16> {
    %3 = tensor.empty() : tensor<2048xf16>
    %4 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%src0, %src1 : tensor<2048xf16>, tensor<2048xf16>) outs(%3 : tensor<2048xf16>) -> tensor<2048xf16>
    return %4 : tensor<2048xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeMod_hfusion_mod_i8
// CHECK-SAME: (%[[x:.*]]: tensor<2048xi8>, %[[y:.*]]: tensor<2048xi8>)
// CHECK: %[[x_f16:.*]] = hfusion.cast {{.*}} ins(%[[x]] : tensor<2048xi8>) outs({{.*}} : tensor<2048xf16>) -> tensor<2048xf16>
// CHECK: %[[x_i16:.*]] = hfusion.cast {{.*}}round_mode = #hfusion.round_mode<trunc>{{.*}} ins(%[[x_f16]] : tensor<2048xf16>) outs({{.*}} : tensor<2048xi16>) -> tensor<2048xi16>
// CHECK: %[[y_f16:.*]] = hfusion.cast {{.*}} ins(%[[y]] : tensor<2048xi8>) outs({{.*}} : tensor<2048xf16>) -> tensor<2048xf16>
// CHECK: %[[y_i16:.*]] = hfusion.cast {{.*}}round_mode = #hfusion.round_mode<trunc>{{.*}} ins(%[[y_f16]] : tensor<2048xf16>) outs({{.*}} : tensor<2048xi16>) -> tensor<2048xi16>
// CHECK: %[[mod_i16:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%[[x_i16]], %[[y_i16]] : tensor<2048xi16>, tensor<2048xi16>) outs({{.*}} : tensor<2048xi16>) -> tensor<2048xi16>
// CHECK: %[[output:.*]] = hfusion.cast {{.*}} ins(%[[mod_i16]] : tensor<2048xi16>) outs({{.*}} : tensor<2048xi8>) -> tensor<2048xi8>
// CHECK: return %[[output]] : tensor<2048xi8>
func.func @test_NormalizeMod_hfusion_mod_i8(%src0: tensor<2048xi8>, %src1: tensor<2048xi8>) -> tensor<2048xi8> {
    %3 = tensor.empty() : tensor<2048xi8>
    %4 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%src0, %src1 : tensor<2048xi8>, tensor<2048xi8>) outs(%3 : tensor<2048xi8>) -> tensor<2048xi8>
    return %4 : tensor<2048xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeMod_hfusion_modui_i8
// CHECK-SAME: (%[[x:.*]]: tensor<2048xi8>, %[[y:.*]]: tensor<2048xi8>)
// CHECK: %[[x_f16:.*]] = hfusion.cast {{.*}}cast = #hfusion.type_fn<cast_unsigned>{{.*}} ins(%[[x]] : tensor<2048xi8>) outs({{.*}} : tensor<2048xf16>) -> tensor<2048xf16>
// CHECK: %[[x_i16:.*]] = hfusion.cast {{.*}}round_mode = #hfusion.round_mode<trunc>{{.*}} ins(%[[x_f16]] : tensor<2048xf16>) outs({{.*}} : tensor<2048xi16>) -> tensor<2048xi16>
// CHECK: %[[y_f16:.*]] = hfusion.cast {{.*}}cast = #hfusion.type_fn<cast_unsigned>{{.*}} ins(%[[y]] : tensor<2048xi8>) outs({{.*}} : tensor<2048xf16>) -> tensor<2048xf16>
// CHECK: %[[y_i16:.*]] = hfusion.cast {{.*}}round_mode = #hfusion.round_mode<trunc>{{.*}} ins(%[[y_f16]] : tensor<2048xf16>) outs({{.*}} : tensor<2048xi16>) -> tensor<2048xi16>
// CHECK: %[[mod_i16:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>} ins(%[[x_i16]], %[[y_i16]] : tensor<2048xi16>, tensor<2048xi16>) outs({{.*}} : tensor<2048xi16>) -> tensor<2048xi16>
// CHECK: %[[output:.*]] = hfusion.cast {{.*}} ins(%[[mod_i16]] : tensor<2048xi16>) outs({{.*}} : tensor<2048xi8>) -> tensor<2048xi8>
// CHECK: return %[[output]] : tensor<2048xi8>
func.func @test_NormalizeMod_hfusion_modui_i8(%src0: tensor<2048xi8>, %src1: tensor<2048xi8>) -> tensor<2048xi8> {
    %3 = tensor.empty() : tensor<2048xi8>
    %4 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>} ins(%src0, %src1 : tensor<2048xi8>, tensor<2048xi8>) outs(%3 : tensor<2048xi8>) -> tensor<2048xi8>
    return %4 : tensor<2048xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeMod_hfusion_mod_i64
// CHECK-SAME: (%[[SRC0:.*]]: tensor<2048xi64>, %[[SRC1:.*]]: tensor<2048xi64>)
// CHECK: %[[OUTPUT:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%[[SRC0]], %[[SRC1]] : tensor<2048xi64>, tensor<2048xi64>) outs({{%.*}} : tensor<2048xi64>) -> tensor<2048xi64>
func.func @test_NormalizeMod_hfusion_mod_i64(%src0: tensor<2048xi64>, %src1: tensor<2048xi64>) -> tensor<2048xi64> {
    %3 = tensor.empty() : tensor<2048xi64>
    %4 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%src0, %src1 : tensor<2048xi64>, tensor<2048xi64>) outs(%3 : tensor<2048xi64>) -> tensor<2048xi64>
    return %4 : tensor<2048xi64>
}

// -----

// CHECK-LABEL: @test_NormalizeMod_i8_elemwise_mod
// CHECK: hfusion.cast {{.*}}
// CHECK: hfusion.cast {{.*}}
// CHECK: hfusion.cast {{.*}}
// CHECK: hfusion.cast {{.*}}
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} {{.*}}
// CHECK: hfusion.cast {{.*}}
func.func @test_NormalizeMod_i8_elemwise_mod(%arg0: tensor<64xi8>, %arg1: tensor<64xi8>) -> tensor<64xi8> {
  %0 = tensor.empty() : tensor<64xi8>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
                                ins(%arg0, %arg1 : tensor<64xi8>, tensor<64xi8>)
                                outs(%0 : tensor<64xi8>) -> tensor<64xi8>
  return %1 : tensor<64xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeMulRec_linalg_mul_div_by_one
// CHECK-SAME: (%[[arg0:.*]]: tensor<5x1xf16>, %[[arg1:.*]]: tensor<5x1xf16>)
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<5x1xf16>
// CHECK: %[[res:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[arg1]], %[[arg0]] : tensor<5x1xf16>, tensor<5x1xf16>) outs(%[[empty]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[res]], %[[arg0]] : tensor<5x1xf16>, tensor<5x1xf16>)
func.func @test_NormalizeMulRec_linalg_mul_div_by_one(%arg0: tensor<5x1xf16>, %arg1: tensor<5x1xf16>) -> tensor<5x1xf16> {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = tensor.empty() : tensor<5x1xf16>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst, %arg0 : f16, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %arg1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    return %3 : tensor<5x1xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeMulRec_linalg_mul_div_by_one_rec
// CHECK-SAME: (%[[arg0:.*]]: tensor<5x1xf16>, %[[arg1:.*]]: tensor<5x1xf16>)
// CHECK: %[[res:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[arg1]], %[[arg0]] : tensor<5x1xf16>, tensor<5x1xf16>)
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[res]], %[[arg0]] : tensor<5x1xf16>, tensor<5x1xf16>)
func.func @test_NormalizeMulRec_linalg_mul_div_by_one_rec(%arg0: tensor<5x1xf16>, %arg1: tensor<5x1xf16>) -> tensor<5x1xf16> {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = tensor.empty() : tensor<5x1xf16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%arg0 : tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %arg1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    return %3 : tensor<5x1xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeSubVSToVMulAndVAdd_linalg_sub_sv_to_muls_and_adds
// CHECK-SAME: (%[[arg0:.*]]: tensor<16xf32>)
// CHECK: %[[N1:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[F5:.*]] = arith.constant 5.000000e+00 : f32
// CHECK: %[[INIT0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[INIT1:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[MUL:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[arg0]], %[[N1]] : tensor<16xf32>, f32) outs(%[[INIT1]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[ADD:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[F5]], %[[MUL]] : f32, tensor<16xf32>) outs(%[[INIT0]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: return %[[ADD]]
func.func @test_NormalizeSubVSToVMulAndVAdd_linalg_sub_sv_to_muls_and_adds(%arg0: tensor<16xf32>) -> tensor<16xf32>{
  %0 = tensor.empty(): tensor<16xf32>
  %cst = arith.constant 5.0 : f32
  %d = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%cst ,%arg0: f32, tensor<16xf32>) outs(%0: tensor<16xf32>) -> tensor<16xf32>
  return %d : tensor<16xf32>
}

// -----

// CHECK-LABEL: @test_NormalizeSubVSToVMulAndVAdd_simplify_vsub
// CHECK-NEXT: %cst = arith.constant -1.000000e+00 : f32
// CHECK-NEXT: %0 = tensor.empty() : tensor<1024xf32>
// CHECK-NEXT: %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %cst : tensor<1024xf32>, f32) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT: return %1 : tensor<1024xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeSubVSToVMulAndVAdd_simplify_vsub(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%cst, %arg0 : f32, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeLdexp_hfusion_frexp
// CHECK: %[[CSTTWO:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[CSTZERO:.*]] = arith.constant 0.000000e+00 : f16
// CHECK: %[[CSTONE:.*]] = arith.constant 1.000000e+00 : f16
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<10xf16>
// CHECK: %[[FILLONE:.*]] = linalg.fill ins(%[[CSTONE:.*]] : f16) outs(%[[EMPTY0:.*]] : tensor<10xf16>) -> tensor<10xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<10xf16>
// CHECK: %[[ABS:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[ARG0:.*]] : tensor<10xf16>) outs(%[[EMPTY1:.*]] : tensor<10xf16>) -> tensor<10xf16>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<10xf16>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<10xf32>
// CHECK: %[[CAST1:.*]] = hfusion.cast {{.*}} ins(%[[ABS:.*]] : tensor<10xf16>) outs(%[[EMPTY3:.*]] : tensor<10xf32>) -> tensor<10xf32>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<10xf32>
// CHECK: %[[EMPTY4:.*]] = tensor.empty() : tensor<10xf32>
// CHECK: %[[LOG1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[CAST1:.*]] : tensor<10xf32>) outs(%[[EMPTY3:.*]] : tensor<10xf32>) -> tensor<10xf32>
// CHECK: %[[FILLTWO:.*]] = linalg.fill ins(%[[CSTTWO:.*]] : f32) outs(%[[EMPTY3:.*]] : tensor<10xf32>) -> tensor<10xf32>
// CHECK: %[[LOG2:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[FILLTWO:.*]] : tensor<10xf32>) outs(%[[EMPTY3:.*]] : tensor<10xf32>) -> tensor<10xf32>
// CHECK: %[[DIV1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LOG1:.*]], %[[LOG2:.*]] : tensor<10xf32>, tensor<10xf32>) outs(%[[EMPTY4:.*]] : tensor<10xf32>) -> tensor<10xf32>
// CHECK: %[[CAST2:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%[[DIV1:.*]] : tensor<10xf32>) outs(%[[EMPTY2:.*]] : tensor<10xf16>) -> tensor<10xf16>
// CHECK: %[[EMPTY5:.*]] = tensor.empty() : tensor<10xf16>
// CHECK: %[[EMPTY6:.*]] = tensor.empty() : tensor<10xf32>
// CHECK: %[[CAST3:.*]] = hfusion.cast {{.*}} ins(%[[CAST2:.*]] : tensor<10xf16>) outs(%[[EMPTY6:.*]] : tensor<10xf32>) -> tensor<10xf32>
// CHECK: %[[EMPTY7:.*]] = tensor.empty() : tensor<10xf32>
// CHECK: %[[CAST4:.*]] = hfusion.cast {{.*}} ins(%[[CAST3:.*]] : tensor<10xf32>) outs(%[[EMPTY7:.*]] : tensor<10xf32>) -> tensor<10xf32>
// CHECK: %[[CAST5:.*]] = hfusion.cast {{.*}} ins(%[[CAST4:.*]] : tensor<10xf32>) outs(%[[EMPTY5:.*]] : tensor<10xf16>) -> tensor<10xf16>
// CHECK: %[[EMPTY8:.*]] = tensor.empty() : tensor<10xf16>
// CHECK: %[[ADD:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[CAST5:.*]], %[[FILLONE:.*]] : tensor<10xf16>, tensor<10xf16>) outs(%[[EMPTY8:.*]] : tensor<10xf16>) -> tensor<10xf16>
// CHECK: %[[EMPTY9:.*]] = tensor.empty() : tensor<10xf16>
// CHECK: %[[SUB:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%3, %23 : tensor<10xf16>, tensor<10xf16>) outs(%[[EMPTY9:.*]] : tensor<10xf16>) -> tensor<10xf16>
// CHECK: %[[EMPTY10:.*]] = tensor.empty() : tensor<10xf16>
// CHECK: %[[MUL:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ARG0:.*]], %[[SUB:.*]] : tensor<10xf16>, tensor<10xf16>) outs(%[[EMPTY10:.*]] : tensor<10xf16>) -> tensor<10xf16>
func.func @test_NormalizeLdexp_hfusion_frexp(%arg0: tensor<10xf16>) -> tensor<10xf16>{
  %cst_0 = arith.constant 0.000000e+00 : f16
  %cst_1 = arith.constant 1.000000e+00 : f16
  %0 = tensor.empty() : tensor<10xf16>
  %1 = linalg.fill ins(%cst_1 : f16) outs(%0 : tensor<10xf16>) -> tensor<10xf16>
  %2 = tensor.empty() : tensor<10xf16>
  %3 = linalg.fill ins(%cst_0 : f16) outs(%2 : tensor<10xf16>) -> tensor<10xf16>
  %4 = tensor.empty() : tensor<10xf16>
  %5 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<ilogb>} ins(%arg0 : tensor<10xf16>) outs(%4 : tensor<10xf16>) -> tensor<10xf16>
  %6 = tensor.empty() : tensor<10xf16>
  %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%5, %1 : tensor<10xf16>, tensor<10xf16>) outs(%6 : tensor<10xf16>) -> tensor<10xf16>
  %8 = tensor.empty() : tensor<10xi1>
  %9 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%arg0, %3 : tensor<10xf16>, tensor<10xf16>) outs(%8 : tensor<10xi1>) -> tensor<10xi1>
  %10 = tensor.empty() : tensor<10xf16>
  %11 = hfusion.select ins(%9, %3, %7 : tensor<10xi1>, tensor<10xf16>, tensor<10xf16>) outs(%10 : tensor<10xf16>) -> tensor<10xf16>
  %12 = tensor.empty() : tensor<10xf16>
  %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%3, %7 : tensor<10xf16>, tensor<10xf16>) outs(%12 : tensor<10xf16>) -> tensor<10xf16>
  %14 = tensor.empty() : tensor<10xf16>
  %15 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ldexp>} ins(%arg0, %13 : tensor<10xf16>, tensor<10xf16>) outs(%14 : tensor<10xf16>) -> tensor<10xf16>
  return %15 : tensor<10xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeIlogb_hfusion_elemwise_unary_ilogb_f32
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf32>)
// CHECK: %[[CSTTWO:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[ABS_EMPTY:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[ABS:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[ARG0]] : tensor<16xf32>) outs(%[[ABS_EMPTY]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[LOG_EMPTY0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[LOG_EMPTY1:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[LOG1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[ABS]] : tensor<16xf32>) outs(%[[LOG_EMPTY0]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[CSTTWO]] : f32) outs(%[[LOG_EMPTY0]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[LOG2:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[FILL]] : tensor<16xf32>) outs(%[[LOG_EMPTY0]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[DIV:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LOG1]], %[[LOG2]] : tensor<16xf32>, tensor<16xf32>) outs(%[[LOG_EMPTY1]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[FLOOR_EMPTY:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[RES:.*]] = hfusion.cast {{.*}}round_mode = #hfusion.round_mode<floor>{{.*}} ins(%[[DIV]] : tensor<16xf32>) outs(%[[FLOOR_EMPTY]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: return %[[RES]] : tensor<16xf32>
func.func @test_NormalizeIlogb_hfusion_elemwise_unary_ilogb_f32(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  %0 = tensor.empty() : tensor<16xf32>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<ilogb>} ins(%arg0 : tensor<16xf32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
  return %1 : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeLdexp_hfusion_elemwise_binary_ldexp_f32
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf32>, %[[ARG1:.*]]: tensor<16xf32>)
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[RES:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ARG0]], %[[ARG1]] : tensor<16xf32>, tensor<16xf32>) outs(%[[EMPTY0]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: return %[[RES]] : tensor<16xf32>
func.func @test_NormalizeLdexp_hfusion_elemwise_binary_ldexp_f32(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
  %0 = tensor.empty() : tensor<16xf32>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ldexp>} ins(%arg0, %arg1 : tensor<16xf32>, tensor<16xf32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
  return %1 : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeErf_hfusion_elemwise_erf
// CHECK-SAME: (%[[arg0:.*]]: tensor<1024xf32>)
// CHECK: %[[P5:.*]] = arith.constant 26267.2246 : f32
// CHECK: %[[P4:.*]] = arith.constant 13243.3662 : f32
// CHECK: %[[P3:.*]] = arith.constant 3023.12476 : f32
// CHECK: %[[P2:.*]] = arith.constant 398.569641 : f32
// CHECK: %[[P1:.*]] = arith.constant 31.2128582 : f32
// CHECK: %[[T5:.*]] = arith.constant 29639.3848 : f32
// CHECK: %[[T4:.*]] = arith.constant 5063.7915 : f32
// CHECK: %[[T3:.*]] = arith.constant 1393.80615 : f32
// CHECK: %[[T2:.*]] = arith.constant 101.62809 : f32
// CHECK: %[[T1:.*]] = arith.constant 7.55170154 : f32
// CHECK: %[[CST0:.*]] = arith.constant 0.0534437485 : f32
// CHECK: %[[LOWER_BOUND:.*]] = arith.constant -3.920000e+00 : f32
// CHECK: %[[UPPER_BOUND:.*]] = arith.constant 3.920000e+00 : f32
// CHECK: %[[NORM_SRC:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[MINOP:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[arg0]], %[[UPPER_BOUND]] : tensor<1024xf32>, f32) outs(%[[NORM_SRC]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[MAXOP:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%[[MINOP]], %[[LOWER_BOUND]] : tensor<1024xf32>, f32) outs(%[[NORM_SRC]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[SQURE_X_RES:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[SQURE_X:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[MAXOP]], %[[MAXOP]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[SQURE_X_RES]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[NUMER_RES:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[NUMER_INPUT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[SQURE_X]], %[[CST0]] : tensor<1024xf32>, f32) outs(%[[NUMER_RES]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[NUMER_TMP:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[TMP0:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[NUMER_INPUT]], %[[T1]] : tensor<1024xf32>, f32) outs(%[[NUMER_TMP]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[TMP0]], %[[SQURE_X]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[NUMER_TMP]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP2:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[TMP1]], %[[T2]] : tensor<1024xf32>, f32) outs(%[[NUMER_TMP]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP3:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[TMP2]], %[[SQURE_X]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[NUMER_TMP]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP4:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[TMP3]], %[[T3]] : tensor<1024xf32>, f32) outs(%[[NUMER_TMP]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[TMP4]], %[[SQURE_X]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[NUMER_TMP]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP6:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[TMP5]], %[[T4]] : tensor<1024xf32>, f32) outs(%[[NUMER_TMP]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP7:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[TMP6]], %[[SQURE_X]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[NUMER_TMP]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP8:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[TMP7]], %[[T5]] : tensor<1024xf32>, f32) outs(%[[NUMER_TMP]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[NUMER_RES_OP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[MAXOP]], %[[TMP8]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[NUMER_RES]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[DEMON_RES:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[TMP11:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[SQURE_X]], %[[P1]] : tensor<1024xf32>, f32) outs(%[[DEMON_RES]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP12:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[TMP11]], %[[SQURE_X]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[DEMON_RES]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP13:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[TMP12]], %[[P2]] : tensor<1024xf32>, f32) outs(%[[DEMON_RES]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP14:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[TMP13]], %[[SQURE_X]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[DEMON_RES]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP15:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[TMP14]], %[[P3]] : tensor<1024xf32>, f32) outs(%[[DEMON_RES]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP16:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[TMP15]], %[[SQURE_X]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[DEMON_RES]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP17:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[TMP16]], %[[P4]] : tensor<1024xf32>, f32) outs(%[[DEMON_RES]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP18:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[TMP17]], %[[SQURE_X]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[DEMON_RES]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP19:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[TMP18]], %[[P5]] : tensor<1024xf32>, f32) outs(%[[DEMON_RES]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[ERF_RES:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[ERF_RES_OP:.*]]= linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[NUMER_RES_OP]], %[[TMP19]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[ERF_RES]] : tensor<1024xf32>) -> tensor<1024xf32>
func.func @test_NormalizeErf_hfusion_elemwise_erf(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<erf>} ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeErf_hfusion_elemwise_erf_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1024xf16>)
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[CAST0:.*]] = hfusion.cast {{.*}} ins(%[[ARG0]] : tensor<1024xf16>) outs(%[[EMPTY0]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[NORM_SRC:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[MINOP:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[CAST0]], {{.*}} : tensor<1024xf32>, f32) outs(%[[NORM_SRC]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[MAXOP:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%[[MINOP]], {{.*}} : tensor<1024xf32>, f32) outs(%[[NORM_SRC]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[DIV:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins({{.*}}, {{.*}} : tensor<1024xf32>, tensor<1024xf32>) outs({{.*}} : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[CAST1:.*]] = hfusion.cast {{.*}} ins(%[[DIV]] : tensor<1024xf32>) outs({{.*}} : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: return %[[CAST1]] : tensor<1024xf16>
func.func @test_NormalizeErf_hfusion_elemwise_erf_f16(%arg0: tensor<1024xf16>) -> tensor<1024xf16> {
    %0 = tensor.empty() : tensor<1024xf16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<erf>} ins(%arg0 : tensor<1024xf16>) outs(%0 : tensor<1024xf16>) -> tensor<1024xf16>
    return %1 : tensor<1024xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeElemwiseMinMaxNumF_minf
// CHECK: %[[INFINITY:.*]] : f32
// CHECK: %[[SRC_0_NAN_MASK:.*]] -> tensor<512xi1>
// CHECK: %[[EMPTY_0:.*]] = tensor.empty() : tensor<512xf32>
// CHECK: %[[NEW_SRC_0:.*]] = hfusion.select ins(%[[SRC_0_NAN_MASK:.*]], %[[INFINITY:.*]], %arg0 : tensor<512xi1>, f32, tensor<512xf32>) outs(%[[EMPTY_0:.*]] : tensor<512xf32>) -> tensor<512xf32>
// CHECK: %[[SRC_1_NAN_MASK:.*]] -> tensor<512xi1>
// CHECK: %[[EMPTY_1:.*]] = tensor.empty() : tensor<512xf32>
// CHECK: %[[NEW_SRC_1:.*]] = hfusion.select ins(%[[SRC_1_NAN_MASK:.*]], %[[INFINITY:.*]], %arg1 : tensor<512xi1>, f32, tensor<512xf32>) outs(%[[EMPTY_1:.*]] : tensor<512xf32>) -> tensor<512xf32>
// CHECK: %[[EMPTY_RES:.*]] = tensor.empty() : tensor<512xf32>
// CHECK: %[[RET:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[NEW_SRC_0:.*]], %[[NEW_SRC_1:.*]] : tensor<512xf32>, tensor<512xf32>) outs(%[[EMPTY_RES:.*]] : tensor<512xf32>) -> tensor<512xf32>
func.func @test_NormalizeElemwiseMinMaxNumF_minf(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
  %0 = tensor.empty() : tensor<512xf32>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minnumf>} ins(%arg0, %arg1 : tensor<512xf32>, tensor<512xf32>) outs(%0 : tensor<512xf32>) -> tensor<512xf32>
  return %1 : tensor<512xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeElemwiseMinMaxNumF_maxf
// CHECK: %[[INFINITY:.*]] : f32
// CHECK: %[[SRC_0_NAN_MASK:.*]] -> tensor<512xi1>
// CHECK: %[[EMPTY_0:.*]] = tensor.empty() : tensor<512xf32>
// CHECK: %[[NEW_SRC_0:.*]] = hfusion.select ins(%[[SRC_0_NAN_MASK:.*]], %[[INFINITY:.*]], %arg0 : tensor<512xi1>, f32, tensor<512xf32>) outs(%[[EMPTY_0:.*]] : tensor<512xf32>) -> tensor<512xf32>
// CHECK: %[[SRC_1_NAN_MASK:.*]] -> tensor<512xi1>
// CHECK: %[[EMPTY_1:.*]] = tensor.empty() : tensor<512xf32>
// CHECK: %[[NEW_SRC_1:.*]] = hfusion.select ins(%[[SRC_1_NAN_MASK:.*]], %[[INFINITY:.*]], %arg1 : tensor<512xi1>, f32, tensor<512xf32>) outs(%[[EMPTY_1:.*]] : tensor<512xf32>) -> tensor<512xf32>
// CHECK: %[[EMPTY_RES:.*]] = tensor.empty() : tensor<512xf32>
// CHECK: %[[RET:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%[[NEW_SRC_0:.*]], %[[NEW_SRC_1:.*]] : tensor<512xf32>, tensor<512xf32>) outs(%[[EMPTY_RES:.*]] : tensor<512xf32>) -> tensor<512xf32>
func.func @test_NormalizeElemwiseMinMaxNumF_maxf(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> {
  %0 = tensor.empty() : tensor<512xf32>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxnumf>} ins(%arg0, %arg1 : tensor<512xf32>, tensor<512xf32>) outs(%0 : tensor<512xf32>) -> tensor<512xf32>
  return %1 : tensor<512xf32>
}
