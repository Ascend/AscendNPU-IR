// RUN: bishengir-opt --hfusion-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_hfusion_compare_neq_ops
// CHECK-SAME: (%[[arg0:.*]]: tensor<1024xi64>, %[[arg1:.*]]: tensor<1024xi1>)
// CHECK: %[[arg2:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[arg3:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[arg0]] : tensor<1024xi64>) outs(%[[arg2]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[arg4:.*]] = tensor.empty() : tensor<1024xi1>
// CHECK: %[[arg6:.*]] = tensor.empty() : tensor<1024xi1>
// CHECK: %[[arg5:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[arg3]], %[[cst:.*]] : tensor<1024xf32>, f32) outs(%[[arg6]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: %[[arg7:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[arg5]] : tensor<1024xi1>) outs(%[[arg4]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: return %[[arg7]]
func.func @test_hfusion_compare_neq_ops(
  %src1 : tensor<1024xi64>,  %dst : tensor<1024xi1>) ->  tensor<1024xi1> {
  %c0_i64 = arith.constant 0 : i64
  %0 = tensor.empty() : tensor<1024xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1024xi64>) -> tensor<1024xi64>
  %ret = hfusion.compare {compare_fn  = #hfusion.compare_fn<vne>}
    ins(%src1, %1 : tensor<1024xi64>, tensor<1024xi64>)
    outs(%dst : tensor<1024xi1>)
    -> tensor<1024xi1>
  return %ret : tensor<1024xi1>
}

// -----

// CHECK-LABEL: func.func @test_linalg_negf_mul
// CHECK-SAME: (%[[arg0:.*]]: tensor<5x1xf32>)
// CHECK: %[[CST:.*]]: f32
// CHECK: %[[ZERO:.*]] : tensor<5x1xf32
// CHECK: %[[ONE:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[arg0:.*]], %[[CST:.*]]: tensor<5x1xf32>, f32) outs(%[[ZERO:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: return %[[ONE:.*]]
func.func @test_linalg_negf_mul(%src: tensor<5x1xf32>) -> tensor<5x1xf32> {
  %x = tensor.empty() : tensor<5x1xf32>
  %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%src : tensor<5x1xf32>) outs(%x : tensor<5x1xf32>) -> tensor<5x1xf32>
  return %1 : tensor<5x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_linalg_div_to_hfusion_rec
// CHECK-SAME: (%[[arg0:.*]]: tensor<5x1xf16>)
// CHECK: %[[cast0:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[arg0]] : tensor<5x1xf16>)
// CHECK: %[[rec:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%[[cast0:.*]]: tensor<5x1xf32>) outs({{.*}} : tensor<5x1xf32>)
// CHECK: %[[cast1:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[rec]] : tensor<5x1xf32>) outs({{.*}} : tensor<5x1xf16>)
// CHECK: return %[[cast1]]
func.func @test_linalg_div_to_hfusion_rec(%src: tensor<5x1xf16>) -> tensor<5x1xf16> {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = tensor.empty() : tensor<5x1xf16>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst, %src : f16, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    return %1 : tensor<5x1xf16>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_rsqrt_to_hfusion_sqrt
// CHECK-SAME: (%[[arg0:.*]]: tensor<5x1xf32>)
// CHECK: %[[ONE:.*]]: tensor<5x1xf32>
// CHECK: %[[TWO:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[arg0:.*]]: tensor<5x1xf32>) outs(%[[ONE:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[THREE:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%[[TWO:.*]]: tensor<5x1xf32>) outs(%[[ZERO:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: return %[[THREE:.*]]
func.func @test_hfusion_rsqrt_to_hfusion_sqrt(%arg0: tensor<5x1xf32>) -> tensor<5x1xf32> {
    %0 = tensor.empty() : tensor<5x1xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%arg0 : tensor<5x1xf32>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
    return %1 : tensor<5x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_rsqrt_f16
// CHECK-SAME: (%[[arg0:.*]]: tensor<16xf16>)
// CHECK: %[[EMPTY0:.*]]: tensor<16xf16>
// CHECK: %[[EMPTY1:.*]]: tensor<16xf32>
// CHECK: %[[CAST0:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[arg0:.*]] : tensor<16xf16>) outs(%[[EMPTY1:.*]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EMPTY2:.*]]: tensor<16xf32>
// CHECK: %[[CAST1:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[EMPTY0:.*]] : tensor<16xf16>) outs(%[[EMPTY2:.*]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EMPTY3:.*]]: tensor<16xf32>
// CHECK: %[[SQRT0:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[CAST0:.*]]: tensor<16xf32>) outs(%[[EMPTY3:.*]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[REC0:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%[[SQRT0:.*]]: tensor<16xf32>) outs(%[[CAST1:.*]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EMPTY4:.*]]: tensor<16xf16>
// CHECK: %[[CAST2:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[REC0:.*]] : tensor<16xf32>) outs(%[[EMPTY4:.*]] : tensor<16xf16>) -> tensor<16xf16>
// CHECK: return %[[REC0:.*]] : tensor<16xf16>
func.func @test_hfusion_rsqrt_f16(%arg0: tensor<16xf16>) -> tensor<16xf16> {
    %0 = tensor.empty() : tensor<16xf16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%arg0 : tensor<16xf16>) outs(%0 : tensor<16xf16>) -> tensor<16xf16>
    return %1 : tensor<16xf16>
}

// -----

// CHECK: func.func @test_hfusion_rsqrt_to_hfusion_sqrt_dynshape(%[[ARG0:.*]]: tensor<5x?xf32>, %[[ARG1:.*]]: index)
// CHECK: %[[ONE:.*]]: tensor<5x?xf32>
// CHECK: %[[TWO:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[arg0:.*]]: tensor<5x?xf32>) outs(%[[ONE:.*]] : tensor<5x?xf32>) -> tensor<5x?xf32>
// CHECK: %[[THREE:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%[[TWO:.*]]: tensor<5x?xf32>) outs(%[[ZERO:.*]] : tensor<5x?xf32>) -> tensor<5x?xf32>
// CHECK: return %[[THREE:.*]]
func.func @test_hfusion_rsqrt_to_hfusion_sqrt_dynshape(%s: tensor<5x?xf32>, %d : index) -> tensor<5x?xf32> {
    %0 = tensor.empty(%d) : tensor<5x?xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%s : tensor<5x?xf32>) outs(%0 : tensor<5x?xf32>) -> tensor<5x?xf32>
    return %1 : tensor<5x?xf32>
}

// -----

// CHECK-LABEL: func.func @test_normalize_rec_i16_to_f32(
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<1x2xi16>) outs({{.*}} : tensor<1x2xf32>)
// CHECK: hfusion.elemwise_unary
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<1x2xf32>) outs({{.*}} : tensor<1x2xi32>)
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} ins({{.*}} : tensor<1x2xi32>) outs({{.*}} : tensor<1x2xi16>)
func.func @test_normalize_rec_i16_to_f32(%arg0 : tensor<1x2xi16>) -> tensor<1x2xi16> {
    %0 = tensor.empty() : tensor<1x2xi16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>, rec} ins(%arg0 : tensor<1x2xi16>) outs(%0 : tensor<1x2xi16>) -> tensor<1x2xi16>
    return %1 : tensor<1x2xi16>
}

// -----

// CHECK-LABEL: func.func @test_normalize_rec_i32_to_f32(
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<1x2xi32>) outs({{.*}} : tensor<1x2xf32>)
// CHECK: hfusion.elemwise_unary
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<1x2xf32>) outs({{.*}} : tensor<1x2xi32>)
func.func @test_normalize_rec_i32_to_f32(%arg0 : tensor<1x2xi32>) -> tensor<1x2xi32> {
    %0 = tensor.empty() : tensor<1x2xi32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>, rec} ins(%arg0 : tensor<1x2xi32>) outs(%0 : tensor<1x2xi32>) -> tensor<1x2xi32>
    return %1 : tensor<1x2xi32>
}

// -----

// CHECK-LABEL: func.func @test_normalize_rec_i64_to_f32(
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<1x2xi64>) outs({{.*}} : tensor<1x2xf32>)
// CHECK: hfusion.elemwise_unary
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<1x2xf32>) outs({{.*}} : tensor<1x2xi64>)
func.func @test_normalize_rec_i64_to_f32(%arg0 : tensor<1x2xi64>) -> tensor<1x2xi64> {
    %0 = tensor.empty() : tensor<1x2xi64>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>, rec} ins(%arg0 : tensor<1x2xi64>) outs(%0 : tensor<1x2xi64>) -> tensor<1x2xi64>
    return %1 : tensor<1x2xi64>
}

// -----

// CHECK-LABEL: func.func @test_linalg_floor_to_hfusion_cast
// CHECK-SAME: (%[[arg0:.*]]: tensor<1024xf16>)
// CHECK: %[[DST0:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[DST1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[RES0:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[arg0]] : tensor<1024xf16>) outs(%[[DST1]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[DST2:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[RES1:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<floor>} ins(%[[RES0]] : tensor<1024xf32>) outs(%[[DST2]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[RES2:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<floor>} ins(%[[RES1]] : tensor<1024xf32>) outs(%[[DST0]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: return %[[RES2]]
func.func @test_linalg_floor_to_hfusion_cast(%src: tensor<1024xf16>) -> tensor<1024xf16> {
    %dst = tensor.empty() : tensor<1024xf16>
    %res = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>} ins(%src : tensor<1024xf16>) outs(%dst : tensor<1024xf16>) -> tensor<1024xf16>
   return %res : tensor<1024xf16>
}

// CHECK-LABEL:   func.func @test_hfusion_mod_uint8(
// CHECK-SAME:                                %[[VAL_0:.*]]: tensor<2048xi8>,
// CHECK-SAME:                                %[[VAL_1:.*]]: tensor<2048xi8>) -> tensor<2048xi8> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 2.550000e+02 : f32
// CHECK:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_4:.*]] = tensor.empty() : tensor<2048xf16>
// CHECK:           %[[VAL_5:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_0]] : tensor<2048xi8>) outs(%[[VAL_4]] : tensor<2048xf16>) -> tensor<2048xf16>
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<2048xf32>
// CHECK:           %[[VAL_7:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_5]] : tensor<2048xf16>) outs(%[[VAL_6]] : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK:           %[[VAL_8:.*]] = tensor.empty() : tensor<2048xf16>
// CHECK:           %[[VAL_9:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_1]] : tensor<2048xi8>) outs(%[[VAL_8]] : tensor<2048xf16>) -> tensor<2048xf16>
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<2048xf32>
// CHECK:           %[[VAL_11:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_9]] : tensor<2048xf16>) outs(%[[VAL_10]] : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK:           %[[VAL_12:.*]] = tensor.empty() : tensor<2048xf32>
// CHECK:           %[[VAL_13:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[VAL_7]], %[[VAL_11]] : tensor<2048xf32>, tensor<2048xf32>) outs(%[[VAL_12]] : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK:           %[[VAL_14:.*]] = tensor.empty() : tensor<2048xf32>
// CHECK:           %[[VAL_15:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[VAL_13]] : tensor<2048xf32>) outs(%[[VAL_14]] : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK:           %[[VAL_16:.*]] = tensor.empty() : tensor<2048xf32>
// CHECK:           %[[VAL_17:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_11]], %[[VAL_15]] : tensor<2048xf32>, tensor<2048xf32>) outs(%[[VAL_16]] : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK:           %[[VAL_18:.*]] = tensor.empty() : tensor<2048xf32>
// CHECK:           %[[VAL_19:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_7]], %[[VAL_17]] : tensor<2048xf32>, tensor<2048xf32>) outs(%[[VAL_18]] : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK:           %[[VAL_20:.*]] = tensor.empty() : tensor<2048xi1>
// CHECK:           %[[VAL_21:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[VAL_11]], %[[VAL_3]] : tensor<2048xf32>, f32) outs(%[[VAL_20]] : tensor<2048xi1>) -> tensor<2048xi1>
// CHECK:           %[[VAL_22:.*]] = tensor.empty() : tensor<2048xf32>
// CHECK:           %[[VAL_23:.*]] = hfusion.select ins(%[[VAL_21]], %[[VAL_2]], %[[VAL_19]] : tensor<2048xi1>, f32, tensor<2048xf32>) outs(%[[VAL_22]] : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK:           %[[VAL_24:.*]] = tensor.empty() : tensor<2048xf16>
// CHECK:           %[[VAL_25:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_23]] : tensor<2048xf32>) outs(%[[VAL_24]] : tensor<2048xf16>) -> tensor<2048xf16>
// CHECK:           %[[VAL_26:.*]] = tensor.empty() : tensor<2048xi8>
// CHECK:           %[[VAL_27:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins(%[[VAL_25]] : tensor<2048xf16>) outs(%[[VAL_26]] : tensor<2048xi8>) -> tensor<2048xi8>
// CHECK:           return %[[VAL_27]] : tensor<2048xi8>
// CHECK:         }
func.func @test_hfusion_mod_uint8(%src0: tensor<2048xi8>, %src1: tensor<2048xi8>) -> tensor<2048xi8> {
  %3 = tensor.empty() : tensor<2048xi8>
  %4 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>} ins(%src0, %src1 : tensor<2048xi8>, tensor<2048xi8>) outs(%3 : tensor<2048xi8>) -> tensor<2048xi8>
  return %4 : tensor<2048xi8>
}

// -----


// CHECK-LABEL:   func.func @test_hfusion_mod(
// CHECK-SAME:                                %[[VAL_0:.*]]: tensor<2048xi32>,
// CHECK-SAME:                                %[[VAL_1:.*]]: tensor<2048xi32>) -> tensor<2048xi32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant -1 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:.*]] = tensor.empty() : tensor<2048xi32>
// CHECK:           %[[VAL_5:.*]] = hfusion.elemwise_binary {already_handle_mod_zero = true, fun = #hfusion.binary_fn<mod>} ins(%[[VAL_0]], %[[VAL_1]] : tensor<2048xi32>, tensor<2048xi32>) outs(%[[VAL_4]] : tensor<2048xi32>) -> tensor<2048xi32>
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<2048xi1>
// CHECK:           %[[VAL_7:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[VAL_1]], %[[VAL_3]] : tensor<2048xi32>, i32) outs(%[[VAL_6]] : tensor<2048xi1>) -> tensor<2048xi1>
// CHECK:           %[[VAL_8:.*]] = tensor.empty() : tensor<2048xi32>
// CHECK:           %[[VAL_9:.*]] = hfusion.select ins(%[[VAL_7]], %[[VAL_2]], %[[VAL_5]] : tensor<2048xi1>, i32, tensor<2048xi32>) outs(%[[VAL_8]] : tensor<2048xi32>) -> tensor<2048xi32>
// CHECK:           return %[[VAL_9]] : tensor<2048xi32>
// CHECK:         }


func.func @test_hfusion_mod(%src0: tensor<2048xi32>, %src1: tensor<2048xi32>) -> tensor<2048xi32> {
  %3 = tensor.empty() : tensor<2048xi32>
  %4 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%src0, %src1 : tensor<2048xi32>, tensor<2048xi32>) outs(%3 : tensor<2048xi32>) -> tensor<2048xi32>
  return %4 : tensor<2048xi32>
}

// -----

// CHECK-LABEL:   func.func @test_normalize_i16_elemwise_mod(
// CHECK-SAME:                                               %[[VAL_0:.*]]: tensor<64xi16>,
// CHECK-SAME:                                               %[[VAL_1:.*]]: tensor<64xi16>) -> tensor<64xi16> {
// CHECK:           %[[VAL_2:.*]] = arith.constant -1.000000e+00 : f32
// CHECK:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_4:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:           %[[VAL_5:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[VAL_0]] : tensor<64xi16>) outs(%[[VAL_4]] : tensor<64xf32>) -> tensor<64xf32>
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:           %[[VAL_7:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[VAL_1]] : tensor<64xi16>) outs(%[[VAL_6]] : tensor<64xf32>) -> tensor<64xf32>
// CHECK:           %[[VAL_8:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:           %[[VAL_9:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[VAL_5]], %[[VAL_7]] : tensor<64xf32>, tensor<64xf32>) outs(%[[VAL_8]] : tensor<64xf32>) -> tensor<64xf32>
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:           %[[VAL_11:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[VAL_9]] : tensor<64xf32>) outs(%[[VAL_10]] : tensor<64xf32>) -> tensor<64xf32>
// CHECK:           %[[VAL_12:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:           %[[VAL_13:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_7]], %[[VAL_11]] : tensor<64xf32>, tensor<64xf32>) outs(%[[VAL_12]] : tensor<64xf32>) -> tensor<64xf32>
// CHECK:           %[[VAL_14:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:           %[[VAL_15:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_5]], %[[VAL_13]] : tensor<64xf32>, tensor<64xf32>) outs(%[[VAL_14]] : tensor<64xf32>) -> tensor<64xf32>
// CHECK:           %[[VAL_16:.*]] = tensor.empty() : tensor<64xi1>
// CHECK:           %[[VAL_17:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[VAL_7]], %[[VAL_3]] : tensor<64xf32>, f32) outs(%[[VAL_16]] : tensor<64xi1>) -> tensor<64xi1>
// CHECK:           %[[VAL_18:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:           %[[VAL_19:.*]] = hfusion.select ins(%[[VAL_17]], %[[VAL_2]], %[[VAL_15]] : tensor<64xi1>, f32, tensor<64xf32>) outs(%[[VAL_18]] : tensor<64xf32>) -> tensor<64xf32>
// CHECK:           %[[VAL_20:.*]] = tensor.empty() : tensor<64xi16>
// CHECK:           %[[VAL_21:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins(%[[VAL_19]] : tensor<64xf32>) outs(%[[VAL_20]] : tensor<64xi16>) -> tensor<64xi16>
// CHECK:           return %[[VAL_21]] : tensor<64xi16>
// CHECK:         }


func.func @test_normalize_i16_elemwise_mod(%arg0: tensor<64xi16>, %arg1: tensor<64xi16>) -> tensor<64xi16> {
  %0 = tensor.empty() : tensor<64xi16>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
                                ins(%arg0, %arg1 : tensor<64xi16>, tensor<64xi16>)
                                outs(%0 : tensor<64xi16>) -> tensor<64xi16>
  return %1 : tensor<64xi16>
}


// -----

// CHECK-LABEL: func.func @test_linalg_divi_to_divf
func.func @test_linalg_divi_to_divf(%arg0: tensor<48xi32>, %arg1: tensor<48xi32>) -> tensor<48xi32> {
    %res = tensor.empty() : tensor<48xi32>
    // CHECK: %[[LHS:.*]] = tensor.empty() : tensor<48xf32>
    // CHECK: %[[LHSFP:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<48xi32>) outs(%[[LHS]] : tensor<48xf32>) -> tensor<48xf32>
    // CHECK: %[[RHS:.*]] = tensor.empty() : tensor<48xf32>
    // CHECK: %[[RHSFP:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<48xi32>) outs(%[[RHS]] : tensor<48xf32>) -> tensor<48xf32>
    // CHECK: %[[RES:.*]] = tensor.empty() : tensor<48xf32>
    // CHECK: %[[RESFP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LHSFP]], %[[RHSFP]] : tensor<48xf32>, tensor<48xf32>) outs(%[[RES]] : tensor<48xf32>) -> tensor<48xf32>
    // CHECK: %[[RESINT:.*]] = tensor.empty() : tensor<48xi32>
    // CHECK: %[[RET:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[RESFP]] : tensor<48xf32>) outs(%[[RESINT]] : tensor<48xi32>) -> tensor<48xi32>
    %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %arg1 : tensor<48xi32>, tensor<48xi32>) outs(%res : tensor<48xi32>) -> tensor<48xi32>
    return %0 : tensor<48xi32>
}

// -----


// CHECK-LABEL: func.func @test_linalg_divi_to_divf_vs
func.func @test_linalg_divi_to_divf_vs(%arg0: tensor<48xi32>, %arg1: i32) -> tensor<48xi32> {
    %res = tensor.empty() : tensor<48xi32>
    // CHECK: %[[LHS:.*]] = tensor.empty() : tensor<48xf32>
    // CHECK: %[[LHSFP:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<48xi32>) outs(%[[LHS]] : tensor<48xf32>) -> tensor<48xf32>
    // CHECK: %[[RHSCASTED:.*]] = arith.sitofp %arg1 : i32 to f32
    // CHECK: %[[RES:.*]] = tensor.empty() : tensor<48xf32>
    // CHECK: %[[RESFP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LHSFP]], %[[RHSCASTED]] : tensor<48xf32>, f32) outs(%[[RES]] : tensor<48xf32>) -> tensor<48xf32>
    // CHECK: %[[RESINT:.*]] = tensor.empty() : tensor<48xi32>
    // CHECK: %[[RET:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[RESFP]] : tensor<48xf32>) outs(%[[RESINT]] : tensor<48xi32>) -> tensor<48xi32>
    %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %arg1 : tensor<48xi32>, i32) outs(%res : tensor<48xi32>) -> tensor<48xi32>
    return %0 : tensor<48xi32>
}

// -----

// CHECK-LABEL: func.func @test_linalg_divi_to_divf_sv
func.func @test_linalg_divi_to_divf_sv(%arg0: i32, %arg1: tensor<48xi32>) -> tensor<48xi32> {
    %res = tensor.empty() : tensor<48xi32>
    // CHECK: %[[LHSCASTED:.*]] = arith.sitofp %arg0 : i32 to f32
    // CHECK: %[[RHS:.*]] = tensor.empty() : tensor<48xf32>
    // CHECK: %[[RHSFP:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<48xi32>) outs(%[[RHS]] : tensor<48xf32>) -> tensor<48xf32>
    // CHECK: %[[RES:.*]] = tensor.empty() : tensor<48xf32>
    // CHECK: %[[RESFP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LHSCASTED]], %[[RHSFP]] : f32, tensor<48xf32>) outs(%[[RES]] : tensor<48xf32>) -> tensor<48xf32>
    // CHECK: %[[RESINT:.*]] = tensor.empty() : tensor<48xi32>
    // CHECK: %[[RET:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[RESFP]] : tensor<48xf32>) outs(%[[RESINT]] : tensor<48xi32>) -> tensor<48xi32>
    %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %arg1 : i32, tensor<48xi32>) outs(%res : tensor<48xi32>) -> tensor<48xi32>
    return %0 : tensor<48xi32>
}

// -----

// CHECK-LABEL: func.func @test_cast_op_tensor_i64_to_f16
// CHECK: %[[ZERO:.*]] = tensor.empty() : tensor<23xf16>
// CHECK: %[[ONE:.*]] = tensor.empty() : tensor<23xf32>
// CHECK: %[[TWO:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[arg0:.*]] : tensor<23xi64>) outs(%[[ONE:.*]]: tensor<23xf32>) -> tensor<23xf32>
// CHECK: %[[THREE:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[TWO:.*]] : tensor<23xf32>) outs(%[[ZERO:.*]] : tensor<23xf16>) -> tensor<23xf16>
func.func @test_cast_op_tensor_i64_to_f16(%arg0: tensor<23xi64>, %arg1: tensor<f16>) -> tensor<23xf16> attributes {hacc.entry} {
    %cst = arith.constant 0.86956521739130432 : f64
    %0 = tensor.empty() : tensor<23xf16>
    %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<23xi64>) outs(%0 : tensor<23xf16>) -> tensor<23xf16>
    %broadcasted = linalg.broadcast ins(%arg1 : tensor<f16>) outs(%0 : tensor<23xf16>) dimensions = [0]
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%1, %broadcasted : tensor<23xf16>, tensor<23xf16>) outs(%0 : tensor<23xf16>) -> tensor<23xf16>
    %3 = arith.truncf %cst : f64 to f16
    %4 = linalg.fill ins(%3 : f16) outs(%0 : tensor<23xf16>) -> tensor<23xf16>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %4 : tensor<23xf16>, tensor<23xf16>) outs(%0 : tensor<23xf16>) -> tensor<23xf16>
    return %5 : tensor<23xf16>
  }


// -----
// CHECK-LABEL: func.func @test_isinf
// CHECK: %[[CST0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[NEGONE:.*]] = arith.constant -1 : i32
// CHECK: %[[POSONE:.*]] = arith.constant 1 : i32
// CHECK: %[[NEGINF:.*]] = arith.constant -2139095040 : i32
// CHECK: %[[MASKVAL:.*]] = arith.constant 2147483647 : i32
// CHECK: %[[INPUT:.*]] = tensor.empty() : tensor<8192xf32>
// CHECK: %[[MASKRES:.*]] = tensor.empty() : tensor<8192xi32>
// CHECK: %[[VDUPOP:.*]] = linalg.fill ins(%[[MASKVAL]] : i32) outs(%[[MASKRES]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[BITCASTEMPTY:.*]] = tensor.empty() : tensor<8192xi32>
// CHECK: %[[BITCASTINPUT:.*]] = hfusion.bitcast ins(%[[INPUT]] : tensor<8192xf32>) outs(%[[BITCASTOUT:.*]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[VANDOP:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[BITCASTINPUT]], %[[VDUPOP]] : tensor<8192xi32>, tensor<8192xi32>) outs(%[[VANDOUTPUT:.*]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[VADDRES:.*]] = tensor.empty() : tensor<8192xi32>
// CHECK: %[[VADDOP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VANDOP]], %[[NEGINF]] : tensor<8192xi32>, i32) outs(%[[VADDRES]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[BITCASTADD:.*]] = hfusion.bitcast ins(%[[VADDOP]] : tensor<8192xi32>) outs(%[[BITCASTOUT:.*]] : tensor<8192xf32>) -> tensor<8192xf32>
// CHECK: %[[VABSOP:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[BITCASTADD]] : tensor<8192xf32>) outs(%[[BITCASTOUTPUT:.*]] : tensor<8192xf32>) -> tensor<8192xf32>
// CHECK: %[[BITCASTABS:.*]] = hfusion.bitcast ins(%[[VABSOP]] : tensor<8192xf32>) outs(%[[BITCASTOUT:.*]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[VMINOP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>} ins(%[[BITCASTABS]], %[[POSONE]] : tensor<8192xi32>, i32) outs(%[[BITCASTOUTPUT:.*]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[VMULOP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VMINOP]], %[[NEGONE]] : tensor<8192xi32>, i32) outs(%[[VMINOP]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[VADDOP1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VMULOP]], %[[POSONE]] : tensor<8192xi32>, i32) outs(%[[VMULOP]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[TMPF:.*]] = tensor.empty() : tensor<8192xf32>
// CHECK: %[[CASTF:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VADDOP1]] : tensor<8192xi32>) outs(%[[TMPF]] : tensor<8192xf32>) -> tensor<8192xf32>
// CHECK: %[[OUT1:.*]] = tensor.empty() : tensor<8192xi1>
// CHECK: %[[OUT2:.*]] = tensor.empty() : tensor<8192xi1>
// CHECK: %[[RES:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[CASTF]], %[[CST0]] : tensor<8192xf32>, f32) outs(%[[OUT2]] : tensor<8192xi1>) -> tensor<8192xi1>
// CHECK: %[[RES2:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[RES]] : tensor<8192xi1>) outs(%[[OUT1]] : tensor<8192xi1>) -> tensor<8192xi1>
func.func @test_isinf() -> tensor<8192xi1> {
  %0 = tensor.empty() : tensor<8192xf32>
  %2 = hfusion.isinf %0 : tensor<8192xf32> -> tensor<8192xi1>
  return %2 : tensor<8192xi1>
}
// -----

// CHECK-LABEL: @lowering_cast_i64_to_bf16(
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<4x4xi64>) outs({{.*}} : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<4x4xf32>) outs({{.*}} : tensor<4x4xbf16>) -> tensor<4x4xbf16>
func.func @lowering_cast_i64_to_bf16(%arg0: tensor<4x4xi64>) -> tensor<4x4xbf16> {
  %0 = tensor.empty() : tensor<4x4xbf16>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<4x4xi64>) outs(%0 : tensor<4x4xbf16>) -> tensor<4x4xbf16>
  return %1 : tensor<4x4xbf16>
}
// -----

// CHECK-LABEL: @test_cast_i8_to_bf16
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<4x4xi8>) outs({{.*}} : tensor<4x4xf16>) -> tensor<4x4xf16>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<4x4xf16>) outs({{.*}} : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<4x4xf32>) outs({{.*}} : tensor<4x4xbf16>) -> tensor<4x4xbf16>
func.func @test_cast_i8_to_bf16(%arg0: tensor<4x4xi8>) -> tensor<4x4xbf16> {
  %0 = tensor.empty() : tensor<4x4xbf16>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<4x4xi8>) outs(%0 : tensor<4x4xbf16>) -> tensor<4x4xbf16>
  return %1 : tensor<4x4xbf16>
}
// -----

// CHECK-LABEL: func.func @test_cast_bf16_to_i1
// CHECK-SAME: (%[[arg0:.*]]: tensor<16xbf16>)
// CHECK: %[[arg1:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[arg2:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[arg0]] : tensor<16xbf16>) outs(%[[arg1]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[arg3:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[arg5:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[arg4:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[arg2]], %[[cst:.*]] : tensor<16xf32>, f32) outs(%[[arg5]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[arg6:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[arg4]] : tensor<16xi1>) outs(%[[arg3]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: return %[[arg6]]
func.func @test_cast_bf16_to_i1(%arg0: tensor<16xbf16>) -> tensor<16xi1> {
  %0 = tensor.empty() : tensor<16xi1>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<16xbf16>) outs(%0 : tensor<16xi1>) -> tensor<16xi1>
  return %1 : tensor<16xi1>
}
// -----

// CHECK-LABEL: func.func @test_cast_f32_to_i1
// CHECK: %[[arg4:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[arg2:.*]], %[[cst:.*]] : tensor<2x256x12x257xf32>, f32) outs(%[[arg5:.*]] : tensor<2x256x12x257xi1>) -> tensor<2x256x12x257xi1>
// ChECK: %[[arg6:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[arg4:.*]] : tensor<2x256x12x257xi1>) outs(%[[arg3:.*]] : tensor<2x256x12x257xi1>) -> tensor<2x256x12x257xi1>
func.func @test_cast_f32_to_i1(%arg0: tensor<2x256x12x257xf32>) -> tensor<2x256x12x257xi1> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %1 = tensor.empty() : tensor<2x256x12x257xi1>
  %2 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<2x256x12x257xf32>) outs(%1 : tensor<2x256x12x257xi1>) -> tensor<2x256x12x257xi1>
  return %2 : tensor<2x256x12x257xi1>
}
// -----

// CHECK-LABEL: func.func @test_cast_f16_to_i1
// CHECK: %[[arg4:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[arg2:.*]], %[[cst:.*]] : tensor<2x256x12x257xf16>, f16) outs(%[[arg5:.*]] : tensor<2x256x12x257xi1>) -> tensor<2x256x12x257xi1>
// ChECK: %[[arg6:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[arg4:.*]] : tensor<2x256x12x257xi1>) outs(%[[arg3:.*]] : tensor<2x256x12x257xi1>) -> tensor<2x256x12x257xi1>
func.func @test_cast_f16_to_i1(%arg0: tensor<2x256x12x257xf16>) -> tensor<2x256x12x257xi1> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %1 = tensor.empty() : tensor<2x256x12x257xi1>
  %2 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<2x256x12x257xf16>) outs(%1 : tensor<2x256x12x257xi1>) -> tensor<2x256x12x257xi1>
  return %2 : tensor<2x256x12x257xi1>
}

// -----
func.func @scalar_like_tensor_conversion_hfusion_elemwise_unary_absi(%dst: tensor<1xi32>) -> tensor<1xi32> {
  %src = arith.constant dense<5> : tensor<1xi32>
  // CHECK: %c5_i32 = arith.constant 5 : i32
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<absi>} ins(%c5_i32 : i32) outs(%arg0 : tensor<1xi32>) -> tensor<1xi32>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<absi>} ins(%src : tensor<1xi32>) outs(%dst : tensor<1xi32>) -> tensor<1xi32>

  return %1 : tensor<1xi32>
}

// -----
func.func @scalar_like_tensor_conversion_hfusion_elemwise_unary_absi_rank0(%dst: tensor<i32>) -> tensor<i32> {
  %src = arith.constant dense<5> : tensor<i32>
  // CHECK: %c5_i32 = arith.constant 5 : i32
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<absi>} ins(%c5_i32 : i32) outs(%arg0 : tensor<i32>) -> tensor<i32>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<absi>} ins(%src : tensor<i32>) outs(%dst : tensor<i32>) -> tensor<i32>

  return %1 : tensor<i32>
}

// -----
func.func @scalar_like_tensor_conversion_hfusion_elemwise_binary(%src: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK: %cst = arith.constant 5.000000e-01 : f32
  %cst = arith.constant dense<0.5> : tensor<1xf32>
  %dst = tensor.empty() : tensor<1xf32>
  // CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%cst, %arg0 : f32, tensor<1xf32>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%cst, %src : tensor<1xf32>, tensor<1xf32>) outs(%dst : tensor<1xf32>) -> tensor<1xf32>
  return %res : tensor<1xf32>
}

// -----
func.func @scalar_like_tensor_conversion_hfusion_compare(%src: tensor<1xi64>) -> tensor<1xi1> {
  // CHECK: %c5_i64 = arith.constant 5 : i64
  // CHECK: %0 = tensor.empty() : tensor<1xi1>
  // CHECK: %1 = tensor.empty() : tensor<1xi1>
  // CHECK: %2 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%arg0, %c5_i64 : tensor<1xi64>, i64) outs(%1 : tensor<1xi1>) -> tensor<1xi1>
  // CHECK: %3 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%2 : tensor<1xi1>) outs(%0 : tensor<1xi1>) -> tensor<1xi1>

  %cst = arith.constant dense<5> : tensor<1xi64>
  %dst = tensor.empty() : tensor<1xi1>

  %res = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
    ins(%src, %cst : tensor<1xi64>, tensor<1xi64>)
    outs(%dst : tensor<1xi1>) -> tensor<1xi1>

  return %res : tensor<1xi1>
}

// -----
func.func @scalar_like_tensor_conversion_hfusion_select(
  %src1 : memref<1xi1>, %src3 : memref<1xi32>, %dst : memref<1xi32>) {
  // CHECK: %c5_i32 = arith.constant 5 : i32
  // CHECK: hfusion.select ins(%arg0, %c5_i32, %arg1 : memref<1xi1>, i32, memref<1xi32>) outs(%arg2 : memref<1xi32>)

  %src2 = arith.constant dense<5> : memref<1xi32>

  hfusion.select
    ins(%src1, %src2, %src3 : memref<1xi1>, memref<1xi32>, memref<1xi32>)
    outs(%dst : memref<1xi32>)
  return
}

// -----
func.func @scalar_like_tensor_conversion_hfusion_cast() -> tensor<f16> {
    // CHECK: %cst = arith.constant 0.869565188 : f32
    // CHECK: %0 = tensor.empty() : tensor<f16>
    // CHECK: %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%cst : f32) outs(%0 : tensor<f16>) -> tensor<f16>

    %src = arith.constant dense<0.86956521739130432> : tensor<f32>
    %dst = tensor.empty() : tensor<f16>

    %res = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%src : tensor<f32>) outs(%dst : tensor<f16>) -> tensor<f16>
    return %res : tensor<f16>
}

// -----
func.func @scalar_like_tensor_conversion_linalg_elemwise_unary_abs(%output : tensor<f32>) -> tensor<f32> {
  // CHECK: %cst = arith.constant 5.100000e+00 : f32
  // CHECK: %0 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%cst : f32) outs(%arg0 : tensor<f32>) -> tensor<f32>
  %src = arith.constant dense<5.1> : tensor<f32>
  %0 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
                              ins(%src: tensor<f32>) outs(%output: tensor<f32>) -> tensor<f32>
  return %0: tensor<f32>
}

// -----
func.func @scalar_like_tensor_conversion_linalg_elemwise_binary(%src: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK: %cst = arith.constant 5.000000e-01 : f32
  // CHECK: %0 = tensor.empty() : tensor<1xf32>
  // CHECK: %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%cst, %arg0 : f32, tensor<1xf32>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>

  %cst = arith.constant dense<0.5> : tensor<1xf32>
  %dst = tensor.empty() : tensor<1xf32>
  %res = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%cst, %src : tensor<1xf32>, tensor<1xf32>) outs(%dst : tensor<1xf32>) -> tensor<1xf32>
  return %res : tensor<1xf32>
}

// -----
func.func @scalar_like_tensor_conversion_xori(%arg0: tensor<i8>) -> tensor<i8> {
  %c127_i8 = arith.constant 127 : i8
  %0 = tensor.empty() : tensor<i8>
  %2 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vxor>} ins(%c127_i8, %arg0 : i8, tensor<i8>) outs(%0 : tensor<i8>) -> tensor<i8>
  return %2 : tensor<i8>
}

// CHECK-LABEL:   func.func @scalar_like_tensor_conversion_xori(
// CHECK-SAME:                                                  %[[VAL_0:.*]]: tensor<i8>) -> tensor<i8> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 127 : i8
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<i8>
// CHECK:           %[[VAL_3:.*]] = hfusion.elemwise_binary {fun = {{.*}}<vor>} ins(%[[VAL_1]], %[[VAL_0]] : i8, tensor<i8>) outs(%[[VAL_2]] : tensor<i8>) -> tensor<i8>
// CHECK:           %[[VAL_4:.*]] = tensor.empty() : tensor<i8>
// CHECK:           %[[VAL_5:.*]] = hfusion.elemwise_binary {fun = {{.*}}<vand>} ins(%[[VAL_1]], %[[VAL_0]] : i8, tensor<i8>) outs(%[[VAL_4]] : tensor<i8>) -> tensor<i8>
// CHECK:           %[[VAL_6:.*]] = hfusion.elemwise_unary {fun = {{.*}}<vnot>} ins(%[[VAL_5]] : tensor<i8>) outs(%[[VAL_5]] : tensor<i8>) -> tensor<i8>
// CHECK:           %[[VAL_7:.*]] = tensor.empty() : tensor<i8>
// CHECK:           %[[VAL_8:.*]] = hfusion.elemwise_binary {fun = {{.*}}<vand>} ins(%[[VAL_6]], %[[VAL_3]] : tensor<i8>, tensor<i8>) outs(%[[VAL_7]] : tensor<i8>) -> tensor<i8>
// CHECK:           return %[[VAL_8]] : tensor<i8>
// CHECK:         }

// -----
func.func @scalar_like_tensor_conversion_linalg_brc(%init: tensor<4x1xf32>) -> tensor<4x1xf32> {
  // CHECK: %cst = arith.constant 5.000000e-01 : f32
  // CHECK: %0 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<4x1xf32>) -> tensor<4x1xf32>
  %input = arith.constant dense<0.5> : tensor<1xf32>
  %res = linalg.broadcast
      ins(%input: tensor<1xf32>)
      outs(%init: tensor<4x1xf32>)
      dimensions = [0]
  func.return %res : tensor<4x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_elemwise_unary_log2
// CHECK: %[[CSTTWO:.*]] : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[LOG_RES1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[ARG0:.*]] : tensor<1024xf32>) outs(%[[EMPTY0:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[CSTTWO:.*]] : f32) outs(%[[EMPTY0:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[LOG_RES2:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[FILL:.*]] : tensor<1024xf32>) outs(%[[EMPTY0:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[RES:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LOG_RES1:.*]], %[[LOG_RES2:.*]]: tensor<1024xf32>, tensor<1024xf32>) outs(%[[EMPTY1:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
func.func @test_hfusion_elemwise_unary_log2(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<log2>} ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
}

// -----
// CHECK-LABEL: func.func @test_hfusion_elemwise_unary_sinh
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1024xf32>)
// CHECK-DAG: %[[CST_SMALL_POS:.*]] = arith.constant 1.000000e-01 : f32
// CHECK-DAG: %[[CST_SMALL_NEG:.*]] = arith.constant -1.000000e-01 : f32
// CHECK-DAG: %[[CST_NEG5:.*]] = arith.constant -5.000000e+00 : f32
// CHECK-DAG: %[[CST_POS5:.*]] = arith.constant 5.000000e+00 : f32
// CHECK-DAG: %[[CST_NEG1:.*]] = arith.constant -1.000000e+00 : f32
// CHECK-DAG: %[[CST_HALF:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG: %[[CST_NEG_HALF:.*]] = arith.constant -5.000000e-01 : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[EXP0:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[ARG0]] : tensor<1024xf32>) outs(%[[EMPTY0]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[NEGX:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ARG0]], %[[CST_NEG1]] : tensor<1024xf32>, f32) outs(%[[EMPTY1]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[EXP1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[NEGX]] : tensor<1024xf32>) outs(%[[EMPTY2]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[SUB:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[EXP0]], %[[EXP1]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[EMPTY3]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY4:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[MID:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[SUB]], %[[CST_HALF]] : tensor<1024xf32>, f32) outs(%[[EMPTY4]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY5:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[LPOS:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[EXP0]], %[[CST_HALF]] : tensor<1024xf32>, f32) outs(%[[EMPTY5]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY6:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[LNEG:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[EXP1]], %[[CST_NEG_HALF]] : tensor<1024xf32>, f32) outs(%[[EMPTY6]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY7:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[POS5_T:.*]] = linalg.fill ins(%[[CST_POS5]] : f32) outs(%[[EMPTY7]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[GT_BIG:.*]] = arith.cmpf ogt, %[[ARG0]], %[[POS5_T]] : tensor<1024xf32>
// CHECK: %[[EMPTY8:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[NEG5_T:.*]] = linalg.fill ins(%[[CST_NEG5]] : f32) outs(%[[EMPTY8]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[LT_BIG:.*]] = arith.cmpf olt, %[[ARG0]], %[[NEG5_T]] : tensor<1024xf32>
// CHECK: %[[EMPTY9:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[SMALL_NEG_T:.*]] = linalg.fill ins(%[[CST_SMALL_NEG]] : f32) outs(%[[EMPTY9]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[GT_SMALL_NEG:.*]] = arith.cmpf ogt, %[[ARG0]], %[[SMALL_NEG_T]] : tensor<1024xf32>
// CHECK: %[[EMPTY10:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[SMALL_POS_T:.*]] = linalg.fill ins(%[[CST_SMALL_POS]] : f32) outs(%[[EMPTY10]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[LT_SMALL_POS:.*]] = arith.cmpf olt, %[[ARG0]], %[[SMALL_POS_T]] : tensor<1024xf32>
// CHECK: %[[SMALL_MASK:.*]] = arith.andi %[[GT_SMALL_NEG]], %[[LT_SMALL_POS]] : tensor<1024xi1>
// CHECK: %[[BASE:.*]] = arith.select %[[SMALL_MASK]], %[[ARG0]], %[[MID]] : tensor<1024xi1>, tensor<1024xf32>
// CHECK: %[[TMP:.*]] = arith.select %[[GT_BIG]], %[[LPOS]], %[[BASE]] : tensor<1024xi1>, tensor<1024xf32>
// CHECK: %[[RES:.*]] = arith.select %[[LT_BIG]], %[[LNEG]], %[[TMP]] : tensor<1024xi1>, tensor<1024xf32>
// CHECK: return %[[RES]] : tensor<1024xf32>
func.func @test_hfusion_elemwise_unary_sinh(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %0 = tensor.empty() : tensor<1024xf32>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sinh>} ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
  return %1 : tensor<1024xf32>
}

// -----
// CHECK-LABEL: func.func @test_hfusion_elemwise_unary_sinh_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1024xf16>)
// CHECK-DAG: %[[CST_SMALL_POS:.*]] = arith.constant 1.000000e-01 : f32
// CHECK-DAG: %[[CST_SMALL_NEG:.*]] = arith.constant -1.000000e-01 : f32
// CHECK-DAG: %[[CST_NEG5:.*]] = arith.constant -5.000000e+00 : f32
// CHECK-DAG: %[[CST_POS5:.*]] = arith.constant 5.000000e+00 : f32
// CHECK-DAG: %[[CST_NEG_HALF:.*]] = arith.constant -5.000000e-01 : f32
// CHECK-DAG: %[[CST_HALF:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG: %[[CST_NEG1:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[CAST0:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%[[ARG0]] : tensor<1024xf16>) outs(%[[EMPTY0]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[EXP0:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[CAST0]] : tensor<1024xf32>) outs(%[[EMPTY1]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[NEGX:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[CAST0]], %[[CST_NEG1]] : tensor<1024xf32>, f32) outs(%[[EMPTY2]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[EXP1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[NEGX]] : tensor<1024xf32>) outs(%[[EMPTY3]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY4:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[SUB:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[EXP0]], %[[EXP1]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[EMPTY4]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY5:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[MID:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[SUB]], %[[CST_HALF]] : tensor<1024xf32>, f32) outs(%[[EMPTY5]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY6:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[LPOS:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[EXP0]], %[[CST_HALF]] : tensor<1024xf32>, f32) outs(%[[EMPTY6]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY7:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[LNEG:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[EXP1]], %[[CST_NEG_HALF]] : tensor<1024xf32>, f32) outs(%[[EMPTY7]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY8:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[POS5_T:.*]] = linalg.fill ins(%[[CST_POS5]] : f32) outs(%[[EMPTY8]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[GT_BIG:.*]] = arith.cmpf ogt, %[[CAST0]], %[[POS5_T]] : tensor<1024xf32>
// CHECK: %[[EMPTY9:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[NEG5_T:.*]] = linalg.fill ins(%[[CST_NEG5]] : f32) outs(%[[EMPTY9]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[LT_BIG:.*]] = arith.cmpf olt, %[[CAST0]], %[[NEG5_T]] : tensor<1024xf32>
// CHECK: %[[EMPTY10:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[SMALL_NEG_T:.*]] = linalg.fill ins(%[[CST_SMALL_NEG]] : f32) outs(%[[EMPTY10]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[GT_SMALL_NEG:.*]] = arith.cmpf ogt, %[[CAST0]], %[[SMALL_NEG_T]] : tensor<1024xf32>
// CHECK: %[[EMPTY11:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[SMALL_POS_T:.*]] = linalg.fill ins(%[[CST_SMALL_POS]] : f32) outs(%[[EMPTY11]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[LT_SMALL_POS:.*]] = arith.cmpf olt, %[[CAST0]], %[[SMALL_POS_T]] : tensor<1024xf32>
// CHECK: %[[SMALL_MASK:.*]] = arith.andi %[[GT_SMALL_NEG]], %[[LT_SMALL_POS]] : tensor<1024xi1>
// CHECK: %[[BASE:.*]] = arith.select %[[SMALL_MASK]], %[[CAST0]], %[[MID]] : tensor<1024xi1>, tensor<1024xf32>
// CHECK: %[[TMP:.*]] = arith.select %[[GT_BIG]], %[[LPOS]], %[[BASE]] : tensor<1024xi1>, tensor<1024xf32>
// CHECK: %[[RES32:.*]] = arith.select %[[LT_BIG]], %[[LNEG]], %[[TMP]] : tensor<1024xi1>, tensor<1024xf32>
// CHECK: %[[EMPTY12:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[CAST1:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%[[RES32]] : tensor<1024xf32>) outs(%[[EMPTY12]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: return %[[CAST1]] : tensor<1024xf16>
func.func @test_hfusion_elemwise_unary_sinh_f16(%arg0: tensor<1024xf16>) -> tensor<1024xf16> {
  %0 = tensor.empty() : tensor<1024xf16>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sinh>} ins(%arg0 : tensor<1024xf16>) outs(%0 : tensor<1024xf16>) -> tensor<1024xf16>
  return %1 : tensor<1024xf16>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_elemwise_unary_log10
// CHECK: %[[CSTTEN:.*]] : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[LOG_RES1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[ARG0:.*]] : tensor<1024xf32>) outs(%[[EMPTY0:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[CSTTEN:.*]] : f32) outs(%[[EMPTY0:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[LOG_RES2:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[FILL:.*]] : tensor<1024xf32>) outs(%[[EMPTY0:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[RES:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LOG_RES1:.*]], %[[LOG_RES2:.*]]: tensor<1024xf32>, tensor<1024xf32>) outs(%[[EMPTY1:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
func.func @test_hfusion_elemwise_unary_log10(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<log10>} ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_elemwise_unary_log2_f16
// CHECK: %[[CSTTWO:.*]] : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[CAST_RES1:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[ARG0:.*]] : tensor<1024xf16>) outs(%[[EMPTY1:.*]]  : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[LOG_RES1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[CAST_RES1:.*]] : tensor<1024xf32>) outs(%[[EMPTY2:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[CSTTWO:.*]] : f32) outs(%[[EMPTY2:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[LOG_RES2:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[FILL:.*]] : tensor<1024xf32>) outs(%[[EMPTY2:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[DIV_RES:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LOG_RES1:.*]], %[[LOG_RES2:.*]]: tensor<1024xf32>, tensor<1024xf32>) outs(%[[EMPTY3:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[CAST_RES2:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%[[DIV_RES:.*]] : tensor<1024xf32>) outs(%[[EMPTY0:.*]]  : tensor<1024xf16>) -> tensor<1024xf16>
func.func @test_hfusion_elemwise_unary_log2_f16(%arg0: tensor<1024xf16>) -> tensor<1024xf16> {
    %0 = tensor.empty() : tensor<1024xf16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<log2>} ins(%arg0 : tensor<1024xf16>) outs(%0 : tensor<1024xf16>) -> tensor<1024xf16>
    return %1 : tensor<1024xf16>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_elemwise_unary_log1p
// CHECK: %[[CSTONE:.*]] : f32
// CHECK: %[[EMPTYO:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[ADD_RES:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[ARG0:.*]],  %[[CSTONE:.*]] : tensor<1024xf32>, f32) outs(%[[EMPTY0:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[RES:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[ADD_RES:.*]] : tensor<1024xf32>) outs(%[[EMPTY1:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
func.func @test_hfusion_elemwise_unary_log1p(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<log1p>} ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_elemwise_unary_exp2
// CHECK: %[[CSTLNTWO:.*]] : f32
// CHECK: %[[EMPTYO:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[MUL_RES1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ARG0:.*]],  %[[CSTLNTWO:.*]]: tensor<1024xf32>, f32) outs(%[[EMPTYO:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[RES:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[MUL_RES1:.*]] : tensor<1024xf32>) outs(%[[EMPTY1:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
func.func @test_hfusion_elemwise_unary_exp2(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<exp2>} ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_elemwise_unary_exp2_f16
// CHECK: %[[cast0:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins({{.*}} : tensor<1024xf16>) outs({{.*}} : tensor<1024xf32>)
// CHECK: %[[mul:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[cast0]]
// CHECK: %[[exp:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[mul]] : tensor<1024xf32>)
// CHECK: %[[cast1:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%[[exp]] : tensor<1024xf32>) outs({{.*}} : tensor<1024xf16>)
func.func @test_hfusion_elemwise_unary_exp2_f16(%arg0: tensor<1024xf16>) -> tensor<1024xf16> {
    %0 = tensor.empty() : tensor<1024xf16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<exp2>} ins(%arg0 : tensor<1024xf16>) outs(%0 : tensor<1024xf16>) -> tensor<1024xf16>
    return %1 : tensor<1024xf16>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_elemwise_unary_expm1
// CHECK: %[[CSTLNTWO:.*]] : f32
// CHECK: %[[EMPTYO:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[EXP_RES1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[ARG0:.*]] : tensor<1024xf32>) outs(%[[EMPTYO:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[RES:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[EXP_RES1:.*]], %[[CSTLNTWO:.*]] : tensor<1024xf32>, f32) outs(%[[EMPTY1:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
func.func @test_hfusion_elemwise_unary_expm1(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<expm1>} ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_elemwise_unary_expm1_f16
// CHECK: %[[cast0:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins({{.*}} : tensor<1024xf16>) outs({{.*}} : tensor<1024xf32>)
// CHECK: %[[exp:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[cast0:.*]] : tensor<1024xf32>)
// CHECK: %[[sub:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[exp]]
// CHECK: %[[cast1:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%[[sub]] : tensor<1024xf32>) outs({{.*}} : tensor<1024xf16>)
func.func @test_hfusion_elemwise_unary_expm1_f16(%arg0: tensor<1024xf16>) -> tensor<1024xf16> {
    %0 = tensor.empty() : tensor<1024xf16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<expm1>} ins(%arg0 : tensor<1024xf16>) outs(%0 : tensor<1024xf16>) -> tensor<1024xf16>
    return %1 : tensor<1024xf16>
}

// -----

// CHECK-LABEL: func.func @test_linalg_mul_div_by_one
// CHECK-SAME: (%[[arg0:.*]]: tensor<5x1xf16>, %[[arg1:.*]]: tensor<5x1xf16>)
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<5x1xf16>
// CHECK: %[[res:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[arg1]], %[[arg0]] : tensor<5x1xf16>, tensor<5x1xf16>) outs(%[[empty]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[res]], %[[arg0]] : tensor<5x1xf16>, tensor<5x1xf16>)
func.func @test_linalg_mul_div_by_one(%arg0: tensor<5x1xf16>, %arg1: tensor<5x1xf16>) -> tensor<5x1xf16> {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = tensor.empty() : tensor<5x1xf16>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst, %arg0 : f16, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %arg1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    return %3 : tensor<5x1xf16>
}

// -----
// CHECK-LABEL: func.func @test_linalg_mul_div_by_one_rec
// CHECK-SAME: (%[[arg0:.*]]: tensor<5x1xf16>, %[[arg1:.*]]: tensor<5x1xf16>)
// CHECK: %[[res:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[arg1]], %[[arg0]] : tensor<5x1xf16>, tensor<5x1xf16>)
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[res]], %[[arg0]] : tensor<5x1xf16>, tensor<5x1xf16>)
func.func @test_linalg_mul_div_by_one_rec(%arg0: tensor<5x1xf16>, %arg1: tensor<5x1xf16>) -> tensor<5x1xf16> {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = tensor.empty() : tensor<5x1xf16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%arg0 : tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %arg1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    return %3 : tensor<5x1xf16>
}

// -----
// CHECK-LABEL: func.func @test_normalize_i1_hfusion_compare
// CHECK-NOT: hfusion.compare
// CHECK: return %arg0
func.func @test_normalize_i1_hfusion_compare(%arg0: tensor<16xi1>) -> tensor<16xi1> {
  %false = arith.constant false
  %0 = tensor.empty() : tensor<16xi1>
  %1 = linalg.fill ins(%false : i1) outs (%0 : tensor<16xi1>) -> tensor<16xi1>
  %dst0 = tensor.empty() : tensor<16xi1>
  %res1 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
    ins(%arg0, %1 : tensor<16xi1>, tensor<16xi1>) outs(%dst0 : tensor<16xi1>) -> tensor<16xi1>
  return %res1 : tensor<16xi1>
}

// -----
// CHECK-LABEL: func.func @test_normalize_i8_hfusion_compare
// CHECK: %[[CAST0:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<16xi8>)
// CHECK: %[[CAST1:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<16xi8>)
// CHECK: %[[Veq:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[CAST0]], %[[CAST1]] : tensor<16xf16>, tensor<16xf16>)
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[Veq]] : tensor<16xi1>)
func.func @test_normalize_i8_hfusion_compare(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi1> {
  %dst1 = tensor.empty() : tensor<16xi1>
  %dst2 = tensor.empty() : tensor<16xi1>
  %res1 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
    ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>)
    outs(%dst1 : tensor<16xi1>) -> tensor<16xi1>
  return %res1 : tensor<16xi1>
}

// -----
// CHECK-LABEL: func.func @test_normalize_i32_hfusion_compare
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[ARG0:.*]], %[[ARG0:.*]] : tensor<16xi32>, tensor<16xi32>)
// CHECK: %[[Veq:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[ARG1:.*]], %[[ARG1:.*]] : tensor<16xi32>, tensor<16xi32>)
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[Veq]] : tensor<16xi1>)
func.func @test_normalize_i32_hfusion_compare(%arg0: tensor<16xi32>, %arg1: tensor<16xi32>) -> (tensor<16xi1>, tensor<16xi1>) {
  %dst1 = tensor.empty() : tensor<16xi1>
  %dst2 = tensor.empty() : tensor<16xi1>
  %res1 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
    ins(%arg0, %arg1 : tensor<16xi32>, tensor<16xi32>)
    outs(%dst1 : tensor<16xi1>) -> tensor<16xi1>
  %res2 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
    ins(%arg0, %arg0 : tensor<16xi32>, tensor<16xi32>)
    outs(%dst2 : tensor<16xi1>) -> tensor<16xi1>
  return %res1, %res2 : tensor<16xi1>, tensor<16xi1>
}

// -----
// CHECK-LABEL: func.func @test_normalize_i32_hfusion_compare_dynamic(
// CHECK: hfusion.compare {fun = #hfusion.compare_fn<veq>} ins(%[[ARG0:.*]], %[[ARG1:.*]] : tensor<?x?xi32>, tensor<?x?xi32>)
func.func @test_normalize_i32_hfusion_compare_dynamic(%arg0: tensor<?x?xi32>) -> (tensor<?x?xi32>, tensor<?x?xi1>) attributes {OperatorType = "Default", compute_capability = "", frontend_symbol = {input_0 = ["s93", "s94"], output_0 = ["s93", "s94"], output_1 = ["s93", "s94"]}, hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32319_i32 = arith.constant 32319 : i32
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xi32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xi32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xi32>
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?x?xi32>) -> tensor<?x?xi32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%arg0, %1 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%0 : tensor<?x?xi32>) -> tensor<?x?xi32>
  %3 = tensor.empty(%dim, %dim_0) : tensor<?x?xi32>
  %4 = linalg.fill ins(%c32319_i32 : i32) outs(%3 : tensor<?x?xi32>) -> tensor<?x?xi32>
  %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>} ins(%2, %4 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%3 : tensor<?x?xi32>) -> tensor<?x?xi32>
  %6 = tensor.empty(%dim, %dim_0) : tensor<?x?xi1>
  %7 = hfusion.compare {fun = #hfusion.compare_fn<veq>} ins(%arg0, %5 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%6 : tensor<?x?xi1>) -> tensor<?x?xi1>
  return %5, %7 : tensor<?x?xi32>, tensor<?x?xi1>
}

// -----
// CHECK-LABEL: func.func @test_normalize_i8_to_f32
// CHECK: %[[cast0:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<1xi8>) outs({{.*}} : tensor<1xf16>)
// CHECK: %[[cast1:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[cast0]] : tensor<1xf16>) outs({{.*}} : tensor<1xf32>)
// CHECK: %[[cast2:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<1xi8>) outs({{.*}} : tensor<1xf16>)
// CHECK: %[[cast3:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[cast2]] : tensor<1xf16>) outs({{.*}} : tensor<1xf32>)
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[cast1]], %[[cast3]] : tensor<1xf32>, tensor<1xf32>)
func.func @test_normalize_i8_to_f32(%arg0: memref<?xi8>, %arg1: tensor<1xi8>, %arg2: tensor<1xi8>) {
  %0 = tensor.empty() : tensor<1xf32>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg1 : tensor<1xi8>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
  %2 = tensor.empty() : tensor<1xf32>
  %3 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg2 : tensor<1xi8>) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
  %4 = tensor.empty() : tensor<1xf32>
  %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%1, %3 : tensor<1xf32>, tensor<1xf32>) outs(%4 : tensor<1xf32>) -> tensor<1xf32>
  %6 = tensor.empty() : tensor<1xi8>
  %7 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%5 : tensor<1xf32>) outs(%6 : tensor<1xi8>) -> tensor<1xi8>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1], offset: ?>>
  bufferization.materialize_in_destination %7 in writable %reinterpret_cast : (tensor<1xi8>, memref<1xi8, strided<[1], offset: ?>>) -> ()
  return
}

// -----
// CHECK-LABEL: func.func @test_xori
// CHECK-SAME: (%[[arg0:.*]]: tensor<512xi16>, %[[arg1:.*]]: tensor<512xi16>)
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<512xi16>
// CHECK: %[[vor:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%[[arg0]], %[[arg1]] : tensor<512xi16>, tensor<512xi16>) outs(%[[empty]] : tensor<512xi16>) -> tensor<512xi16>
// CHECK: %[[empty1:.*]] = tensor.empty() : tensor<512xi16>
// CHECK: %[[vand:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[arg0]], %[[arg1]] : tensor<512xi16>, tensor<512xi16>) outs(%[[empty1]] : tensor<512xi16>) -> tensor<512xi16>
// CHECK: %[[vnot:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[vand]] : tensor<512xi16>) outs(%[[vand]] : tensor<512xi16>) -> tensor<512xi16>
// CHECK: %[[empty2:.*]] = tensor.empty() : tensor<512xi16>
// CHECK: %[[xor:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[vnot]], %[[vor]] : tensor<512xi16>, tensor<512xi16>) outs(%[[empty2]] : tensor<512xi16>) -> tensor<512xi16>
// CHECK: return %[[xor]] : tensor<512xi16>
func.func @test_xori(%arg0: tensor<512xi16>, %arg1: tensor<512xi16>) -> tensor<512xi16> {
    %0 = tensor.empty() : tensor<512xi16>
    %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vxor>} ins(%arg0, %arg1 : tensor<512xi16>, tensor<512xi16>) outs(%0 : tensor<512xi16>) -> tensor<512xi16>
    return %1 : tensor<512xi16>
}

// -----
// CHECK-LABEL: func.func @test_hfusion_sin_ops(
// CHECK-SAME: %[[ARG:.*]]: tensor<5x1xf32>) -> tensor<5x1xf32> {
// CHECK: %[[VAL_0:.*]] = arith.constant -0.166666582 : f32
// CHECK: %[[VAL_1:.*]] = arith.constant 8.333050e-03 : f32
// CHECK: %[[VAL_2:.*]] = arith.constant -1.98089445E-4 : f32
// CHECK: %[[VAL_3:.*]] = arith.constant 2.60492652E-6 : f32
// CHECK: %[[VAL_4:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[VAL_5:.*]]  = arith.constant -2.000000e+00 : f32
// CHECK: %[[VAL_6:.*]] = arith.constant 4.000000e+00 : f32
// CHECK: %[[VAL_7:.*]] = arith.constant 5.000000e-01 : f32
// CHECK: %[[VAL_8:.*]] = arith.constant 1.24467439E-13 : f32
// CHECK: %[[VAL_9:.*]] = arith.constant -1.74122761E-9 : f32
// CHECK: %[[VAL_10:.*]] = arith.constant -8.9071691E-6 : f32
// CHECK: %[[VAL_11:.*]] = arith.constant 3.14160156 : f32
// CHECK: %[[VAL_12:.*]] = arith.constant 2.048000e+03 : f32
// CHECK: %[[VAL_13:.*]] = arith.constant 4.8828125E-4 : f32
// CHECK: %[[VAL_14:.*]] = arith.constant 0.318309873 : f32
// CHECK: %[[VAL_15:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_16:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ARG:.*]], %[[VAL_14:.*]] : tensor<5x1xf32>, f32) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_17:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %[[VAL_13:.*]] : tensor<5x1xf32>, f32) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_18:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_19:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%[[VAL_16:.*]] : tensor<5x1xf32>) outs(%[[VAL_18:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_20:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_21:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%[[VAL_17:.*]] : tensor<5x1xf32>) outs(%[[VAL_20:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_22:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %[[VAL_12:.*]] : tensor<5x1xf32>, f32) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_23:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%4, %[[VAL_22:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_24:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_11:.*]], %[[VAL_22:.*]] : f32, tensor<5x1xf32>) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_25:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[ARG:.*]], %[[VAL_24:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_26:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_11:.*]], %[[VAL_23:.*]] : f32, tensor<5x1xf32>) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_27:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%10, %[[VAL_26:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_28:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_10:.*]], %[[VAL_22:.*]] : f32, tensor<5x1xf32>) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_29:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%12, %[[VAL_28:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_30:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_10:.*]], %[[VAL_23:.*]] : f32, tensor<5x1xf32>) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_31:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%14, %[[VAL_30:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_32:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_9:.*]], %[[VAL_22:.*]] : f32, tensor<5x1xf32>) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_33:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%16, %[[VAL_32:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_34:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_9:.*]], %[[VAL_23:.*]] : f32, tensor<5x1xf32>) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_35:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%18, %[[VAL_34:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_36:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%cst_7, %[[VAL_22:.*]] : f32, tensor<5x1xf32>) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_37:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%20, %[[VAL_36:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_38:.*]]  = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%cst_7, %[[VAL_23:.*]] : f32, tensor<5x1xf32>) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_39:.*]]  = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%22, %[[VAL_38:.*]]  : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_40:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %[[VAL_7:.*]] : tensor<5x1xf32>, f32) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_41:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_42:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<floor>} ins(%[[VAL_40:.*]] : tensor<5x1xf32>) outs(%[[VAL_41:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_43:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_42:.*]], %[[VAL_6:.*]] : tensor<5x1xf32>, f32) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_44:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %[[VAL_5:.*]]  : tensor<5x1xf32>, f32) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_45:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_43:.*]], %[[VAL_44:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_46:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_45:.*]], %[[VAL_4:.*]] : tensor<5x1xf32>, f32) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_47:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%24, %[[VAL_39:.*]]  : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_48:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_47:.*]], %[[VAL_3:.*]] : tensor<5x1xf32>, f32) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_49:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_48:.*]], %[[VAL_2:.*]] : tensor<5x1xf32>, f32) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_50:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_49:.*]], %[[VAL_47:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_51:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_50:.*]], %[[VAL_1:.*]] : tensor<5x1xf32>, f32) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_52:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_51:.*]], %[[VAL_47:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_53:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_52:.*]], %[[VAL_0:.*]] : tensor<5x1xf32>, f32) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_54:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_53:.*]], %[[VAL_47:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_55:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_54:.*]], %[[VAL_4:.*]] : tensor<5x1xf32>, f32) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_56:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_55:.*]], %[[VAL_39:.*]]  : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_57:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_56:.*]], %[[VAL_46:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: return %[[VAL_57:.*]] : tensor<5x1xf32>
// CHECK: }
func.func @test_hfusion_sin_ops(%arg0 : tensor<5x1xf32>) ->  tensor<5x1xf32> {
  %0 = tensor.empty() : tensor<5x1xf32>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sin>} ins(%arg0 : tensor<5x1xf32>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  return %ret : tensor<5x1xf32>
}



// -----
// CHECK-LABEL: func.func @test_hfusion_asin_ops(
// CHECK-SAME: %[[ARG0:.*]]: tensor<32xf32>) -> tensor<32xf32> {
// CHECK-DAG: %[[PI_O_2:.*]] = arith.constant 1.57079637 : f32
// CHECK-DAG: %[[TWO:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG: %[[C_1:.*]] = arith.constant 1.666890e-01 : f32
// CHECK-DAG: %[[C_2:.*]] = arith.constant 7.349690e-02 : f32
// CHECK-DAG: %[[C_3:.*]] = arith.constant 0.0592745841 : f32
// CHECK-DAG: %[[HALF:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG: %[[NEG_ONE:.*]] = arith.constant -1.000000e+00 : f32
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32

// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32xf32>

// abs_x
// CHECK: %[[ABS_X:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[ARG0]] : tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// 1 - x
// CHECK: %[[X_NEG:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ABS_X]], %[[NEG_ONE]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[X_MINUS_1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[ONE]], %[[X_NEG]] : f32, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// (1-x)*0.5 -> half
// CHECK: %[[HALF_VAL:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[X_MINUS_1]], %[[HALF]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[SQRT_HALF:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[HALF_VAL]] : tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)

// cond select x
// CHECK: %[[COND_LT_HALF:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>} ins(%[[ABS_X]], %[[HALF]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xi1>)
// CHECK: %[[X_VAL:.*]] = hfusion.select ins(%[[COND_LT_HALF]], %[[ABS_X]], %[[SQRT_HALF]] : tensor<32xi1>, tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)

// Calculate polynomial for asin
// x^2
// CHECK: %[[X_SQ:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[X_VAL]], %[[X_VAL]] : tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// ...
// CHECK: %[[T1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[C_3]], %[[X_SQ]] : f32, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[T2:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[T1]], %[[C_2]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[T3:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[T2]], %[[X_SQ]] : tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[T4:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[T3]], %[[C_1]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[T5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[X_VAL]], %[[X_SQ]] : tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[POLY:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[T5]], %[[T4]] : tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// asin_x
// CHECK: %[[ASIN_X:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[X_VAL]], %[[POLY]] : tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)

// Process large path
// CHECK: %[[ASIN_X_2:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ASIN_X]], %[[TWO]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[ASIN_X_NEG_LARGE:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ASIN_X_2]], %[[NEG_ONE]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[RES_LARGE:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[PI_O_2]], %[[ASIN_X_NEG_LARGE]] : f32, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)

// Select Final Phase
// CHECK: %[[RES_ABS:.*]] = hfusion.select ins(%[[COND_LT_HALF]], %[[ASIN_X]], %[[RES_LARGE]] : tensor<32xi1>, tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)

// Assign Sign
// CHECK: %[[RES_NEG:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[RES_ABS]], %[[NEG_ONE]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[COND_LT_ZERO:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>} ins(%[[ARG0]], %[[ZERO]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xi1>)
// CHECK: %[[FINAL_RES:.*]] = hfusion.select ins(%[[COND_LT_ZERO]], %[[RES_NEG]], %[[RES_ABS]] : tensor<32xi1>, tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// CHECK: return %[[FINAL_RES:.*]] : tensor<32xf32>
// CHECK: }
func.func @test_hfusion_asin_ops(%arg0 : tensor<32xf32>) -> tensor<32xf32> {
  %0 = tensor.empty() : tensor<32xf32>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<asin>} ins(%arg0 : tensor<32xf32>) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  return %1 : tensor<32xf32>
}


// -----
// CHECK-LABEL: func.func @test_hfusion_acos_ops(
// CHECK-SAME: %[[ARG0:.*]]: tensor<32xf32>) -> tensor<32xf32> {
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[PI:.*]] = arith.constant 3.14159274 : f32
// CHECK-DAG: %[[TWO:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG: %[[PI_O_2:.*]] = arith.constant 1.57079637 : f32
// CHECK-DAG: %[[C_1:.*]] = arith.constant 1.666890e-01 : f32
// CHECK-DAG: %[[C_2:.*]] = arith.constant 7.349690e-02 : f32
// CHECK-DAG: %[[C_3:.*]] = arith.constant 0.0592745841 : f32
// CHECK-DAG: %[[HALF:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG: %[[NEG_ONE:.*]] = arith.constant -1.000000e+00 : f32
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32

// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32xf32>

// abs_x
// CHECK: %[[ABS_X:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[ARG0]] : tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// 1 - x (for large branch calculation)
// CHECK: %[[X_NEG:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ABS_X]], %[[NEG_ONE]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[X_MINUS_1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[ONE]], %[[X_NEG]] : f32, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// (1-x)*0.5 -> half
// CHECK: %[[HALF_VAL:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[X_MINUS_1]], %[[HALF]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[SQRT_HALF:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[HALF_VAL]] : tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)

// x_val select
// CHECK: %[[COND_LT_HALF:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>} ins(%[[ABS_X]], %[[HALF]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xi1>)
// CHECK: %[[X_VAL:.*]] = hfusion.select ins(%[[COND_LT_HALF]], %[[ABS_X]], %[[SQRT_HALF]] : tensor<32xi1>, tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)

// polynomial
// x^2
// CHECK: %[[X_SQ:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[X_VAL]], %[[X_VAL]] : tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// (...)
// CHECK: %[[T1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[C_3]], %[[X_SQ]] : f32, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[T2:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[T1]], %[[C_2]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[T3:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[T2]], %[[X_SQ]] : tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[T4:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[T3]], %[[C_1]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[T5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[X_VAL]], %[[X_SQ]] : tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[POLY:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[T5]], %[[T4]] : tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// asin_x
// CHECK: %[[ASIN_X:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[X_VAL]], %[[POLY]] : tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)

// Calculate small paths
// CHECK: %[[ASIN_X_NEG:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ASIN_X]], %[[NEG_ONE]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[SMALL_POS:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[PI_O_2]], %[[ASIN_X_NEG]] : f32, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[SMALL_NEG:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[PI_O_2]], %[[ASIN_X]] : f32, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)

// Calculate large paths
// CHECK: %[[ASIN_X_2:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ASIN_X]], %[[TWO]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[ASIN_X_2_NEG:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ASIN_X_2]], %[[NEG_ONE]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[LARGE_NEG:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[PI]], %[[ASIN_X_2_NEG]] : f32, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)

// Final selects
// CHECK: %[[COND_LT_ZERO:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>} ins(%[[ARG0]], %[[ZERO]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xi1>)
// CHECK: %[[RES_SMALL:.*]] = hfusion.select ins(%[[COND_LT_ZERO]], %[[SMALL_NEG]], %[[SMALL_POS]] : tensor<32xi1>, tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[RES_LARGE:.*]] = hfusion.select ins(%[[COND_LT_ZERO]], %[[LARGE_NEG]], %[[ASIN_X_2]] : tensor<32xi1>, tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)

// Compare phase 4
// CHECK: %[[FINAL_RES:.*]] = hfusion.select ins(%[[COND_LT_HALF]], %[[RES_SMALL]], %[[RES_LARGE]] : tensor<32xi1>, tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// CHECK: return %[[FINAL_RES]] : tensor<32xf32>
// CHECK: }
func.func @test_hfusion_acos_ops(%arg0 : tensor<32xf32>) -> tensor<32xf32> {
  %0 = tensor.empty() : tensor<32xf32>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<acos>} ins(%arg0 : tensor<32xf32>) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  return %1 : tensor<32xf32>
}

// -----
// CHECK-LABEL: func.func @test_hfusion_acosh_ops(
// CHECK-SAME: %[[ARG0:.*]]: tensor<32xf32>) -> tensor<32xf32> {
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[NAN:.*]] = arith.constant 0x7FC00000 : f32
// CHECK-DAG: %[[LOG2:.*]] = arith.constant 0.693147182 : f32
// CHECK-DAG: %[[LARGE_THRESH:.*]] = arith.constant 1.000000e+08 : f32
// CHECK-DAG: %[[TWO:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32

// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32xf32>

// Path 1: x >= 1e8
// log(x) + log(2)
// CHECK: %[[LOG_X:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[ARG0]] : tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[RES_LARGE:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[LOG_X]], %[[LOG2]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xf32>)

// Path 2: 2.0 <= x < 1e8
// x - 1
// CHECK: %[[X_MINUS_1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[ARG0]], %[[ONE]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xf32>)
// x + 1
// CHECK: %[[X_PLUS_1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[ARG0]], %[[ONE]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xf32>)
// term = (x-1)*(x+1)
// CHECK: %[[TERM_MED:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[X_MINUS_1]], %[[X_PLUS_1]] : tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// sqrt(term)
// CHECK: %[[SQRT_MED:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[TERM_MED]] : tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// x + sqrt
// CHECK: %[[ARG_MED:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[ARG0]], %[[SQRT_MED]] : tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// log(...)
// CHECK: %[[RES_MED:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[ARG_MED]] : tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)

// Path 3: 1 <= x < 2.0
// z = x - 1
// CHECK: %[[Z:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[ARG0]], %[[ONE]] : tensor<32xf32>, f32) outs(%[[EMPTY]] : tensor<32xf32>)
// z*z
// CHECK: %[[Z_SQ:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[Z]], %[[Z]] : tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// 2*z
// CHECK: %[[TWO_Z:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[TWO]], %[[Z]] : f32, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// z^2 + 2z
// CHECK: %[[INNER_SMALL:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[Z_SQ]], %[[TWO_Z]] : tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// sqrt(...)
// CHECK: %[[SQRT_SMALL:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[INNER_SMALL]] : tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// z + sqrt(...)
// CHECK: %[[ARG_SMALL:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[Z]], %[[SQRT_SMALL]] : tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// log1p(argSmall) -> log(argSmall + 1)
// CHECK: %[[ARG_SMALL_PLUS_1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[ARG_SMALL]], %[[ONE]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[RES_SMALL:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[ARG_SMALL_PLUS_1]] : tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)

// Combine Results
// x >= 2.0 ?
// CHECK: %[[COND_MED:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vge>} ins(%[[ARG0]], %[[TWO]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xi1>)
// CHECK: %[[RES1:.*]] = hfusion.select ins(%[[COND_MED]], %[[RES_MED]], %[[RES_SMALL]] : tensor<32xi1>, tensor<32xf32>, tensor<32xf32>) outs(%[[EMPTY]] : tensor<32xf32>)

// x >= 1e8 ?
// CHECK: %[[COND_LARGE:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vge>} ins(%[[ARG0]], %[[LARGE_THRESH]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xi1>)
// CHECK: %[[RES_ABS:.*]] = hfusion.select ins(%[[COND_LARGE]], %[[RES_LARGE]], %[[RES1]] : tensor<32xi1>, tensor<32xf32>, tensor<32xf32>) outs(%[[EMPTY]] : tensor<32xf32>)

// Domain Error: x < 1.0
// CHECK: %[[COND_DOMAIN:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>} ins(%[[ARG0]], %[[ONE]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xi1>)
// CHECK: %[[RES_NAN:.*]] = hfusion.select ins(%[[COND_DOMAIN]], %[[NAN]], %[[RES_ABS]] : tensor<32xi1>, f32, tensor<32xf32>) outs(%[[EMPTY]] : tensor<32xf32>)

// Exact 1.0: x == 1.0
// CHECK: %[[COND_ONE:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[ARG0]], %[[ONE]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xi1>)
// CHECK: %[[FINAL:.*]] = hfusion.select ins(%[[COND_ONE]], %[[ZERO]], %[[RES_NAN]] : tensor<32xi1>, f32, tensor<32xf32>) outs(%[[EMPTY]] : tensor<32xf32>)
// CHECK: return %[[FINAL]] : tensor<32xf32>
// CHECK: }
func.func @test_hfusion_acosh_ops(%arg0 : tensor<32xf32>) -> tensor<32xf32> {
  %0 = tensor.empty() : tensor<32xf32>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<acosh>} ins(%arg0 : tensor<32xf32>) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  return %1 : tensor<32xf32>
}

// -----
// CHECK-LABEL: func.func @test_hfusion_asinh_ops(
// CHECK-SAME: %[[ARG0:.*]]: tensor<32xf32>) -> tensor<32xf32> {
// CHECK-DAG: %[[NEG_ONE:.*]] = arith.constant -1.000000e+00 : f32
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG: %[[LOG2:.*]] = arith.constant 0.693147182 : f32
// CHECK-DAG: %[[SMALL_THRESH:.*]] = arith.constant 2.44140625E-4 : f32
// CHECK-DAG: %[[LARGE_THRESH:.*]] = arith.constant 1.000000e+08 : f32

// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32xf32>

// 1. Preprocess: z = abs(x)
// CHECK: %[[ABS_X:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[ARG0]] : tensor<32xf32>) outs(%[[EMPTY]] : tensor<32xf32>)

// 2. Path 1: Large (|x| >= 1e8) -> log(z) + log(2)
// CHECK: %[[LOG_Z:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[ABS_X]] : tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[RES_LARGE:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[LOG_Z]], %[[LOG2]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xf32>)

// 3. Path 2: Normal (|x| < 1e8) -> log(z + sqrt(z^2 + 1))
// CHECK: %[[Z_SQ:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ABS_X]], %[[ABS_X]] : tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[Z_SQ_P1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[Z_SQ]], %[[ONE]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[SQRT_Z:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[Z_SQ_P1]] : tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[SUM_NORMAL:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[ABS_X]], %[[SQRT_Z]] : tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)
// CHECK: %[[RES_NORMAL:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[SUM_NORMAL]] : tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>)

// 4. Piecewise Selection for Magnitude
// Select between Small (z < 2.44e-4) and Normal
// CHECK: %[[COND_SMALL:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>} ins(%[[ABS_X]], %[[SMALL_THRESH]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xi1>)
// CHECK: %[[RES_MID_SMALL:.*]] = hfusion.select ins(%[[COND_SMALL]], %[[ABS_X]], %[[RES_NORMAL]] : tensor<32xi1>, tensor<32xf32>, tensor<32xf32>) outs(%[[EMPTY]] : tensor<32xf32>)
// Select between Large (z >= 1e8) and ResultOfMidSmall
// CHECK: %[[COND_LARGE:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vge>} ins(%[[ABS_X]], %[[LARGE_THRESH]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xi1>)
// CHECK: %[[MAG:.*]] = hfusion.select ins(%[[COND_LARGE]], %[[RES_LARGE]], %[[RES_MID_SMALL]] : tensor<32xi1>, tensor<32xf32>, tensor<32xf32>) outs(%[[EMPTY]] : tensor<32xf32>)

// 5. Restore Sign: x < 0 ? -mag : mag
// CHECK: %[[IS_NEG:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>} ins(%[[ARG0]], %[[ZERO]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xi1>)
// CHECK: %[[NEG_MAG:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[MAG]], %[[NEG_ONE]] : tensor<32xf32>, f32) outs(%[[EMPTY]] : tensor<32xf32>)
// CHECK: %[[FINAL:.*]] = hfusion.select ins(%[[IS_NEG]], %[[NEG_MAG]], %[[MAG]] : tensor<32xi1>, tensor<32xf32>, tensor<32xf32>) outs(%[[EMPTY]] : tensor<32xf32>)

// CHECK: return %[[FINAL]] : tensor<32xf32>
// CHECK: }
func.func @test_hfusion_asinh_ops(%arg0 : tensor<32xf32>) -> tensor<32xf32> {
  %0 = tensor.empty() : tensor<32xf32>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<asinh>} ins(%arg0 : tensor<32xf32>) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  return %1 : tensor<32xf32>
}

// -----
// CHECK-LABEL: func.func @test_hfusion_cos_ops(
// CHECK-SAME: %[[ARG:.*]]: tensor<5x1xf16>) -> tensor<5x1xf16> {
// CHECK: %[[VAL_0:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[VAL_1:.*]] = arith.constant -0.166666582 : f32
// CHECK: %[[VAL_2:.*]] = arith.constant 8.333050e-03 : f32
// CHECK: %[[VAL_3:.*]] = arith.constant -1.98089445E-4 : f32
// CHECK: %[[VAL_4:.*]] = arith.constant 2.60492652E-6 : f32
// CHECK: %[[VAL_5:.*]]  = arith.constant 1.000000e+00 : f32
// CHECK: %[[VAL_6:.*]] = arith.constant -2.000000e+00 : f32
// CHECK: %[[VAL_7:.*]] = arith.constant 4.000000e+00 : f32
// CHECK: %[[VAL_8:.*]] = arith.constant -4.37113883E-8 : f32
// CHECK: %[[VAL_9:.*]] = arith.constant 1.24467439E-13 : f32
// CHECK: %[[VAL_10:.*]] = arith.constant -1.74122761E-9 : f32
// CHECK: %[[VAL_11:.*]] = arith.constant 1.57079637 : f32
// CHECK: %[[VAL_12:.*]] = arith.constant -8.9071691E-6 : f32
// CHECK: %[[VAL_13:.*]] = arith.constant 3.14160156 : f32
// CHECK: %[[VAL_14:.*]] = arith.constant 2.048000e+03 : f32
// CHECK: %[[VAL_68:.*]] = arith.constant 4.8828125E-4 : f32
// CHECK: %[[VAL_69:.*]] = arith.constant 5.000000e-01 : f32
// CHECK: %[[VAL_70:.*]] = arith.constant 0.318309873 : f32
// CHECK: %[[VAL_15:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_16:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%[[ARG:.*]] : tensor<5x1xf16>) outs(%[[VAL_15:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_17:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_18:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %[[VAL_70:.*]] : tensor<5x1xf32>, f32) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_19:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %[[VAL_69:.*]] : tensor<5x1xf32>, f32) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_20:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%3, %[[VAL_68:.*]] : tensor<5x1xf32>, f32) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_21:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_22:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%[[VAL_19:.*]] : tensor<5x1xf32>) outs(%[[VAL_21:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_23:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_24:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%[[VAL_20:.*]] : tensor<5x1xf32>) outs(%[[VAL_23:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_25:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%9, %[[VAL_14:.*]] : tensor<5x1xf32>, f32) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_26:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%7, %[[VAL_25:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_27:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_13:.*]], %[[VAL_25:.*]] : f32, tensor<5x1xf32>) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_28:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%1, %[[VAL_27:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_29:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_13:.*]], %[[VAL_26:.*]] : f32, tensor<5x1xf32>) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_30:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%13, %[[VAL_29:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_31:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_12:.*]], %[[VAL_25:.*]] : f32, tensor<5x1xf32>) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_32:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%15, %[[VAL_31:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_33:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%17, %[[VAL_11:.*]] : tensor<5x1xf32>, f32) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_34:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_12:.*]], %[[VAL_26:.*]] : f32, tensor<5x1xf32>) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_35:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%18, %[[VAL_34:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_36:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_10:.*]], %[[VAL_25:.*]] : f32, tensor<5x1xf32>) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_37:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%20, %[[VAL_36:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_38:.*]]  = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_10:.*]], %[[VAL_26:.*]] : f32, tensor<5x1xf32>) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_39:.*]]  = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%22, %[[VAL_38:.*]]  : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_40:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_9:.*]], %[[VAL_25:.*]] : f32, tensor<5x1xf32>) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_41:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%24, %[[VAL_40:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_42:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_9:.*]], %[[VAL_26:.*]] : f32, tensor<5x1xf32>) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_43:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_41:.*]], %[[VAL_42:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_44:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_43:.*]], %[[VAL_8:.*]] : tensor<5x1xf32>, f32) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_45:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%7, %[[VAL_69:.*]] : tensor<5x1xf32>, f32) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_46:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_47:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<floor>} ins(%[[VAL_45:.*]] : tensor<5x1xf32>) outs(%[[VAL_46:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_48:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_47:.*]], %[[VAL_7:.*]] : tensor<5x1xf32>, f32) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_49:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%7, %[[VAL_6:.*]] : tensor<5x1xf32>, f32) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_50:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_48:.*]], %[[VAL_49:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_51:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_50:.*]], %[[VAL_5:.*]]  : tensor<5x1xf32>, f32) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_52:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_44:.*]], %[[VAL_44:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_53:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_52:.*]], %[[VAL_4:.*]] : tensor<5x1xf32>, f32) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_54:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_53:.*]], %[[VAL_3:.*]] : tensor<5x1xf32>, f32) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_55:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_54:.*]], %[[VAL_52:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_56:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_55:.*]], %[[VAL_2:.*]] : tensor<5x1xf32>, f32) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_57:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_56:.*]], %[[VAL_52:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_58:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_57:.*]], %[[VAL_1:.*]] : tensor<5x1xf32>, f32) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_59:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_58:.*]], %[[VAL_52:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_60:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_59:.*]], %[[VAL_5:.*]]  : tensor<5x1xf32>, f32) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_61:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_60:.*]], %[[VAL_44:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_62:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_61:.*]], %[[VAL_51:.*]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_17:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_63:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_64:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[VAL_62:.*]], %[[VAL_5:.*]]  : tensor<5x1xf32>, f32) outs(%[[VAL_63:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_65:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%[[VAL_64:.*]], %[[VAL_0:.*]] : tensor<5x1xf32>, f32) outs(%[[VAL_63:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_66:.*]] = tensor.empty() : tensor<5x1xf16>
// CHECK: %[[VAL_67:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%[[VAL_65:.*]] : tensor<5x1xf32>) outs(%[[VAL_66:.*]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: return %[[VAL_67:.*]] : tensor<5x1xf16>
// CHECK: }
func.func @test_hfusion_cos_ops(%arg0 : tensor<5x1xf16>) ->  tensor<5x1xf16> {
  %0 = tensor.empty() : tensor<5x1xf16>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>} ins(%arg0 : tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
  return %ret : tensor<5x1xf16>
}

// -----
// CHECK-LABEL: func.func @test_hfusion_atan_ops(
// CHECK-SAME: %[[VAL_0:.*]]: tensor<32xf32>) -> tensor<32xf32> {
// CHECK: %[[CONST_0:.*]] = arith.constant 2.16840434E-19 : f32
// CHECK: %[[CONST_1:.*]] = arith.constant 4.61168602E+18 : f32
// CHECK: %[[CST_0:.*]] = arith.constant 0.785398185 : f32
// CHECK: %[[NEGONE:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[CST_1:.*]] = arith.constant 0.392699093 : f32
// CHECK: %[[CST_2:.*]] = arith.constant 0.414213568 : f32
// CHECK: %[[CST_3:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[CST_4:.*]] = arith.constant -0.333333343 : f32
// CHECK: %[[CST_5:.*]] = arith.constant 2.000000e-01 : f32
// CHECK: %[[CST_6:.*]] = arith.constant -0.142857149 : f32
// CHECK: %[[CST_7:.*]] = arith.constant 0.111111112 : f32
// CHECK: %[[CST_8:.*]] = arith.constant -0.0909090936 : f32
// CHECK: %[[CST_9:.*]] = arith.constant 0.0769230798 : f32
// CHECK: %[[LOWER_BOUND:.*]] = arith.constant -1.000000e+04 : f32
// CHECK: %[[UPPER_BOUND:.*]] = arith.constant 1.000000e+04 : f32
// CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_2:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[VAL_0]], %[[UPPER_BOUND]] : tensor<32xf32>, f32) outs(%[[VAL_1]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_3:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%[[VAL_2]], %[[LOWER_BOUND]] : tensor<32xf32>, f32) outs(%[[VAL_1]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_4:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[VAL_3]] : tensor<32xf32>) outs(%[[EMPTY1]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_6:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_4]], %[[VAL_4]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_7:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_6]], %[[CST_9]] : tensor<32xf32>, f32) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_8:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_7]], %[[CST_8]] : tensor<32xf32>, f32) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_9:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_8]], %[[VAL_6]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_10:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_9]], %[[CST_7]] : tensor<32xf32>, f32) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_11:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_10]], %[[VAL_6]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_12:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_11]], %[[CST_6]] : tensor<32xf32>, f32) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_13:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_12]], %[[VAL_6]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_14:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_13]], %[[CST_5]] : tensor<32xf32>, f32) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_15:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_14]], %[[VAL_6]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_16:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_15]], %[[CST_4]] : tensor<32xf32>, f32) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_17:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_16]], %[[VAL_6]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_18:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_17]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_19:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_18]], %[[VAL_4]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_20:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_21:.*]] = linalg.fill ins(%[[CST_2]] : f32) outs(%[[VAL_20]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_22:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_23:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_4]], %[[VAL_21]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_22]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_24:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_23]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_22]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_25:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_26:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_4]], %[[VAL_21]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_25]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_27:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[VAL_26]], %[[VAL_24]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_25]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_28:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[VAL_27]] : tensor<32xf32>) outs(%[[VAL_25]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_29:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_30:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_28]], %[[VAL_28]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_31:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_30]], %[[CST_9]] : tensor<32xf32>, f32) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_32:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_31]], %[[CST_8]] : tensor<32xf32>, f32) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_33:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_32]], %[[VAL_30]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_34:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_33]], %[[CST_7]] : tensor<32xf32>, f32) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_35:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_34]], %[[VAL_30]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_36:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_35]], %[[CST_6]] : tensor<32xf32>, f32) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_37:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_36]], %[[VAL_30]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_38:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_37]], %[[CST_5]] : tensor<32xf32>, f32) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_39:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_38]], %[[VAL_30]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_40:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_39]], %[[CST_4]] : tensor<32xf32>, f32) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_41:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_40]], %[[VAL_30]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_42:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_41]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_43:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_42]], %[[VAL_28]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_44:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_43]], %[[CST_1]] : tensor<32xf32>, f32) outs(%[[VAL_25]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_45:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[VAL_19]], %[[VAL_44]] : tensor<32xf32>, tensor<32xf32>)
// CHECK: %[[VAL_46:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_47:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_4]], %[[NEGONE]] : tensor<32xf32>, f32)
// CHECK: %[[VAL_48:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_51:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_4]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_48]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_52:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_53:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[VAL_47]], %[[VAL_51]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_52]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_54:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[VAL_53]] : tensor<32xf32>) outs(%[[VAL_52]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_55:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_56:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_54]], %[[VAL_54]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_57:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_56]], %[[CST_9]] : tensor<32xf32>, f32) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_58:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_57]], %[[CST_8]] : tensor<32xf32>, f32) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_59:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_58]], %[[VAL_56]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_60:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_59]], %[[CST_7]] : tensor<32xf32>, f32) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_61:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_60]], %[[VAL_56]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_62:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_61]], %[[CST_6]] : tensor<32xf32>, f32) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_63:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_62]], %[[VAL_56]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_64:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_63]], %[[CST_5]] : tensor<32xf32>, f32) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_65:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_64]], %[[VAL_56]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_66:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_65]], %[[CST_4]] : tensor<32xf32>, f32) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_67:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_66]], %[[VAL_56]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_68:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_67]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_69:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_68]], %[[VAL_54]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_70:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_71:.*]] = linalg.fill ins(%[[CST_2]] : f32) outs(%[[VAL_70]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_72:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_73:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_54]], %[[VAL_71]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_72]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_74:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_73]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_72]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_75:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_76:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_54]], %[[VAL_71]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_75]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_77:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[VAL_76]], %[[VAL_74]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_75]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_78:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[VAL_77]] : tensor<32xf32>) outs(%[[VAL_75]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_79:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_80:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_78]], %[[VAL_78]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_81:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_80]], %[[CST_9]] : tensor<32xf32>, f32) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_82:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_81]], %[[CST_8]] : tensor<32xf32>, f32) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_83:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_82]], %[[VAL_80]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_84:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_83]], %[[CST_7]] : tensor<32xf32>, f32) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_85:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_84]], %[[VAL_80]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_86:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_85]], %[[CST_6]] : tensor<32xf32>, f32) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_87:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_86]], %[[VAL_80]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_88:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_87]], %[[CST_5]] : tensor<32xf32>, f32) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_89:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_88]], %[[VAL_80]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_90:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_89]], %[[CST_4]] : tensor<32xf32>, f32) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_91:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_90]], %[[VAL_80]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_92:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_91]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_93:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_92]], %[[VAL_78]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_94:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_93]], %[[CST_1]] : tensor<32xf32>, f32) outs(%[[VAL_75]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_95:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[VAL_69]], %[[VAL_94]] : tensor<32xf32>, tensor<32xf32>)
// CHECK: %[[VAL_96:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_97:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_95]], %[[CST_0]] : tensor<32xf32>, f32) outs(%[[VAL_96]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_98:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[VAL_45]], %[[VAL_97]] : tensor<32xf32>, tensor<32xf32>) outs(%[[EMPTY2]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_99:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_100:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CONST_1]] : tensor<32xf32>, f32) outs(%[[VAL_99]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_101:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_102:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[VAL_100]] : tensor<32xf32>) outs(%[[VAL_101]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_103:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_102]], %[[CONST_0]] : tensor<32xf32>, f32) outs(%[[VAL_101]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_104:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[VAL_100]], %[[VAL_103]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_99]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_105:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_98]], %[[VAL_104]] : tensor<32xf32>, tensor<32xf32>) outs(%[[EMPTY3]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: return %[[VAL_105]] : tensor<32xf32>
// CHECK: }
func.func @test_hfusion_atan_ops(%arg0 : tensor<32xf32>) ->  tensor<32xf32> {
  %0 = tensor.empty() : tensor<32xf32>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<atan>} ins(%arg0 : tensor<32xf32>) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  return %ret : tensor<32xf32>
}

// -----
// CHECK-LABEL: func.func @test_hfusion_atan2_ops(
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vgt>}
// CHECK: hfusion.select
// CHECK: hfusion.select
// CHECK: linalg.fill
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
// CHECK: hfusion.select
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: linalg.fill
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
// CHECK: hfusion.select
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vge>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
// CHECK: hfusion.select
// CHECK: hfusion.select
// CHECK: hfusion.select
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
// CHECK: hfusion.select
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
// CHECK: hfusion.select
// CHECK: return
func.func @test_hfusion_atan2_ops(%arg0 : tensor<32xf32>, %arg1 : tensor<32xf32>) -> tensor<32xf32> {
  %0 = tensor.empty() : tensor<32xf32>
  %ret = hfusion.elemwise_binary {fun = #hfusion.binary_fn<atan2>} ins(%arg0, %arg1 : tensor<32xf32>, tensor<32xf32>) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  return %ret : tensor<32xf32>
}

// -----
// CHECK-LABEL: func.func @test_hfusion_atanh_ops(
// CHECK-SAME: %[[ARG0:.*]]: tensor<32xf32>) -> tensor<32xf32> {
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[HALF:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG: %[[TWO:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG: %[[NEG_ONE:.*]] = arith.constant -1.000000e+00 : f32
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32

// Abs
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[ABS:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[ARG0]] : tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>) -> tensor<32xf32>

// 1 - |x|
// CHECK: %[[ABS_NEG:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ABS]], %[[NEG_ONE]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[ONE_MINUS_ABS:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[ONE]], %[[ABS_NEG]] : f32, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>) -> tensor<32xf32>

// 2 * |x|
// CHECK: %[[TWO_ABS:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[TWO]], %[[ABS]] : f32, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>) -> tensor<32xf32>

// Path 1 (x < 0.5)
// CHECK: %[[TWO_ABS_MUL_ABS:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[TWO_ABS]], %[[ABS]] : tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[DIV_1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[TWO_ABS_MUL_ABS]], %[[ONE_MINUS_ABS]] : tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[ADD_1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[TWO_ABS]], %[[DIV_1]] : tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[ADD_1_P1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[ADD_1]], %[[ONE]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[LOG_1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[ADD_1_P1]] : tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[RES_1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[HALF]], %[[LOG_1]] : f32, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>) -> tensor<32xf32>

// Path 2 (x >= 0.5)
// CHECK: %[[DIV_2:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[TWO_ABS]], %[[ONE_MINUS_ABS]] : tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[ADD_2_P1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[DIV_2]], %[[ONE]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[LOG_2:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[ADD_2_P1]] : tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[RES_2:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[HALF]], %[[LOG_2]] : f32, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>) -> tensor<32xf32>

// Select inner
// CHECK: %[[COMP_LT_HALF:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>} ins(%[[ABS]], %[[HALF]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xi1>) -> tensor<32xi1>
// CHECK: %[[SELECT_INNER:.*]] = hfusion.select ins(%[[COMP_LT_HALF]], %[[RES_1]], %[[RES_2]] : tensor<32xi1>, tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>) -> tensor<32xf32>

// sign restore
// CHECK: %[[SELECT_INNER_NEG:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[SELECT_INNER]], %[[NEG_ONE]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[COND_LT_ZERO:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>} ins(%[[ARG0]], %[[ZERO]] : tensor<32xf32>, f32) outs(%{{.*}} : tensor<32xi1>) -> tensor<32xi1>
// CHECK: %[[FINAL_RESULT:.*]] = hfusion.select ins(%[[COND_LT_ZERO]], %[[SELECT_INNER_NEG]], %[[SELECT_INNER]] : tensor<32xi1>, tensor<32xf32>, tensor<32xf32>) outs(%{{.*}} : tensor<32xf32>) -> tensor<32xf32>

// CHECK: return %[[FINAL_RESULT]] : tensor<32xf32>
// CHECK: }
func.func @test_hfusion_atanh_ops(%arg0 : tensor<32xf32>) -> tensor<32xf32> {
  %0 = tensor.empty() : tensor<32xf32>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<atanh>} ins(%arg0 : tensor<32xf32>) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  return %1 : tensor<32xf32>
}

// -----
// CHECK-LABEL: func.func @test_hfusion_tan_ops(
// CHECK-SAME: %[[ARG0:.*]]: tensor<32xf32>) -> tensor<32xf32> {
// CHECK: %[[CST:.*]] = arith.constant -24.8048935 : f32
// CHECK: %[[CST_0:.*]] = arith.constant 61.2036247 : f32
// CHECK: %[[CST_1:.*]] = arith.constant -6.87115717 : f32
// CHECK: %[[CST_2:.*]] = arith.constant 0.0698520839 : f32
// CHECK: %[[CST_3:.*]] = arith.constant -1.02906229E-13 : f32
// CHECK: %[[CST_4:.*]] = arith.constant 1.21644916E-10 : f32
// CHECK: %[[CST_5:.*]] = arith.constant 4.37113883E-8 : f32
// CHECK: %[[CST_6:.*]] = arith.constant -4.37113883E-8 : f32
// CHECK: %[[CST_7:.*]] = arith.constant 6.27711415E-7 : f32
// CHECK: %[[CST_8:.*]] = arith.constant -1.57079637 : f32
// CHECK: %[[CST_9:.*]] = arith.constant 1.57079637 : f32
// CHECK: %[[CST_10:.*]] = arith.constant 9.67025756E-4 : f32
// CHECK: %[[CST_11:.*]] = arith.constant 3.140625 : f32
// CHECK: %[[CST_12:.*]] = arith.constant 0.318309873 : f32
// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ARG0]], %[[CST_12]] : tensor<32xf32>, f32) outs(%[[VAL_0]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_3:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%[[VAL_1]] : tensor<32xf32>) outs(%[[VAL_2]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_11]] : tensor<32xf32>, f32) outs(%[[VAL_4]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_6:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[ARG0]], %[[VAL_5]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_4]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_7:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_10]] : tensor<32xf32>, f32) outs(%[[VAL_4]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_8:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_6]], %[[VAL_7]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_4]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_9:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_10:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_8]], %[[CST_9]] : tensor<32xf32>, f32) outs(%[[VAL_9]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_11:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_12:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_8]], %[[CST_8]] : tensor<32xf32>, f32) outs(%[[VAL_11]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_13:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_14:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_7]] : tensor<32xf32>, f32) outs(%[[VAL_13]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_15:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_10]], %[[VAL_14]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_13]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_16:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_17:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_7]] : tensor<32xf32>, f32) outs(%[[VAL_16]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_18:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_12]], %[[VAL_17]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_16]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_19:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_20:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_15]], %[[CST_6]] : tensor<32xf32>, f32) outs(%[[VAL_19]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_21:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_22:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_18]], %[[CST_5]] : tensor<32xf32>, f32) outs(%[[VAL_21]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_23:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_24:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_4]] : tensor<32xf32>, f32) outs(%[[VAL_23]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_25:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_20]], %[[VAL_24]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_23]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_26:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_23]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_27:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_25]], %[[VAL_26]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_23]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_28:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_29:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_4]] : tensor<32xf32>, f32) outs(%[[VAL_28]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_30:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_22]], %[[VAL_29]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_28]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_31:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_28]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_32:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_30]], %[[VAL_31]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_28]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_33:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_34:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_7]] : tensor<32xf32>, f32) outs(%[[VAL_33]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_35:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_8]], %[[VAL_34]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_33]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_36:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_4]] : tensor<32xf32>, f32) outs(%[[VAL_33]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_37:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_35]], %[[VAL_36]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_33]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_38:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_33]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_39:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_37]], %[[VAL_38]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_33]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_40:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_41:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_39]], %[[VAL_39]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_40]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_42:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_43:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_41]], %[[CST_2]] : tensor<32xf32>, f32) outs(%[[VAL_42]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_44:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_45:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_43]], %[[CST_1]] : tensor<32xf32>, f32) outs(%[[VAL_44]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_46:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_45]], %[[VAL_41]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_44]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_47:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_46]], %[[CST_0]] : tensor<32xf32>, f32) outs(%[[VAL_46]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_48:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_47]], %[[VAL_39]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_46]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_49:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_50:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_41]], %[[CST]] : tensor<32xf32>, f32) outs(%[[VAL_49]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_51:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_50]], %[[VAL_27]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_49]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_52:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_51]], %[[VAL_32]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_49]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_53:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_54:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[VAL_48]], %[[VAL_52]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_53]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: return %[[VAL_54]] : tensor<32xf32>
// CHECK: }
func.func @test_hfusion_tan_ops(%arg0 : tensor<32xf32>) ->  tensor<32xf32> {
  %0 = tensor.empty() : tensor<32xf32>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<tan>} ins(%arg0 : tensor<32xf32>) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  return %ret : tensor<32xf32>
}
// -----

// CHECK-LABEL: func.func @test_hfusion_elemwise_erf
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
func.func @test_hfusion_elemwise_erf(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<erf>} ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
}

// -----
// CHECK-LABEL: func.func @test_i1_cast_i64
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<200x200xf32>
// CHECK: %[[CAST_F32:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[ARG0:.*]] : tensor<200x200xi1>) outs(%[[EMPTY1:.*]] : tensor<200x200xf32>) -> tensor<200x200xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<200x200xi64>
// CHECK: %[[CAST_I64:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[CAST_F32:.*]] : tensor<200x200xf32>) outs(%[[EMPTY2:.*]] : tensor<200x200xi64>) -> tensor<200x200xi64>
func.func @test_i1_cast_i64(%arg0: tensor<200x200xi1>) -> tensor<200x200xi64>{
  %0 = tensor.empty() : tensor<200x200xi64>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<200x200xi1>) outs(%0 : tensor<200x200xi64>) -> tensor<200x200xi64>
  return %1 : tensor<200x200xi64>
}

// -----

// CHECK-LABEL: @test_i16_cast_i32(
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<4x4xi16>) outs({{.*}} : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<4x4xf32>) outs({{.*}} : tensor<4x4xi32>) -> tensor<4x4xi32>
func.func @test_i16_cast_i32(%arg0: tensor<4x4xi16>) -> tensor<4x4xi32> {
  %0 = tensor.empty() : tensor<4x4xi32>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<4x4xi16>) outs(%0 : tensor<4x4xi32>) -> tensor<4x4xi32>
  return %1 : tensor<4x4xi32>
}

// -----

// CHECK-LABEL: @test_i8_cast_i32(
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<4x4xi8>) outs({{.*}} : tensor<4x4xf16>) -> tensor<4x4xf16>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<4x4xf16>) outs({{.*}} : tensor<4x4xi32>) -> tensor<4x4xi32>
func.func @test_i8_cast_i32(%arg0: tensor<4x4xi8>) -> tensor<4x4xi32> {
  %0 = tensor.empty() : tensor<4x4xi32>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<4x4xi8>) outs(%0 : tensor<4x4xi32>) -> tensor<4x4xi32>
  return %1 : tensor<4x4xi32>
}

// -----
// CHECK-LABEL: func.func @test_i64_cast_i1
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<20x20xi64>) outs({{.*}} : tensor<20x20xf32>) -> tensor<20x20xf32>
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins({{.*}}, %[[cst_0:.*]] : tensor<20x20xf32>, f32) outs({{.*}} : tensor<20x20xi1>) -> tensor<20x20xi1>
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[Veq:.*]] : tensor<20x20xi1>)
func.func @test_i64_cast_i1(%arg0: tensor<20x20xi64>) -> tensor<20x20xi1>{
  %0 = tensor.empty() : tensor<20x20xi1>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<20x20xi64>) outs(%0 : tensor<20x20xi1>) -> tensor<20x20xi1>
  return %1 : tensor<20x20xi1>
}

// -----

// CHECK-LABEL: @test_i1_cast_f32(
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<4x256xi1>) outs({{.*}} : tensor<4x256xf32>) -> tensor<4x256xf32>
func.func @test_i1_cast_f32(%arg0: tensor<4x256xi1>) -> tensor<4x256xf32> {
  %0 = tensor.empty() : tensor<4x256xf32>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x256xi1>) outs(%0 : tensor<4x256xf32>) -> tensor<4x256xf32>
  return %1 : tensor<4x256xf32>
}

// -----
// CHECK-LABEL: func.func @test_i8_cast_i1
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<4x256xi8>) outs({{.*}} : tensor<4x256xf16>) -> tensor<4x256xf16>
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins({{.*}}, %[[cst_0:.*]] : tensor<4x256xf16>, f16) outs({{.*}} : tensor<4x256xi1>) -> tensor<4x256xi1>
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[Veq:.*]] : tensor<4x256xi1>)
func.func @test_i8_cast_i1(%arg0: tensor<4x256xi8>) -> tensor<4x256xi1>{
  %0 = tensor.empty() : tensor<4x256xi1>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x256xi8>) outs(%0 : tensor<4x256xi1>) -> tensor<4x256xi1>
  return %1 : tensor<4x256xi1>
}

// -----

// CHECK-LABEL: @test_dyn_rec_mul
// CHECK: %[[c0:.*]] = arith.constant 0 : index
// CHECK: %[[dim:.*]] = tensor.dim {{.*}}, %[[c0]] : tensor<?x14336xf16>
// CHECK: %[[empty:.*]] = tensor.empty(%[[dim]]) : tensor<?x14336xf16>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins({{.*}}, {{.*}} : tensor<?x14336xf16>, tensor<?x14336xf16>)
// CHECK-SAME: outs(%[[empty]] : tensor<?x14336xf16>) -> tensor<?x14336xf16>
func.func @test_dyn_rec_mul(%arg0: tensor<?x4096xf16>, %arg1: tensor<14336x4096xf16>, %arg2: tensor<?x14336xf16>) -> tensor<?x14336xf16> {
    %cst = arith.constant 1.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x4096xf16>
    %0 = tensor.empty(%dim) : tensor<?x14336xf16>
    %1 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<?x4096xf16>, tensor<14336x4096xf16>) outs(%0 : tensor<?x14336xf16>) -> tensor<?x14336xf16>
    %2 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>, rec} ins(%arg2 : tensor<?x14336xf16>) outs(%0 : tensor<?x14336xf16>) -> tensor<?x14336xf16>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>, mul} ins(%1, %2 : tensor<?x14336xf16>, tensor<?x14336xf16>) outs(%0 : tensor<?x14336xf16>) -> tensor<?x14336xf16>
    return %3 : tensor<?x14336xf16>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_frexp
// CHECK: %[[CSTTWO:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[CSTZERO:.*]] = arith.constant 0.000000e+00 : f16
// CHECK: %[[CSTONE:.*]] = arith.constant 1.000000e+00 : f16
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<10xf16>
// CHECK: %[[FILLONE:.*]] = linalg.fill ins(%[[CSTONE:.*]] : f16) outs(%[[EMPTY0:.*]] : tensor<10xf16>) -> tensor<10xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<10xf16>
// CHECK: %[[ABS:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[ARG0:.*]] : tensor<10xf16>) outs(%[[EMPTY1:.*]] : tensor<10xf16>) -> tensor<10xf16>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<10xf16>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<10xf32>
// CHECK: %[[CAST1:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[ABS:.*]] : tensor<10xf16>) outs(%[[EMPTY3:.*]] : tensor<10xf32>) -> tensor<10xf32>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<10xf32>
// CHECK: %[[EMPTY4:.*]] = tensor.empty() : tensor<10xf32>
// CHECK: %[[LOG1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[CAST1:.*]] : tensor<10xf32>) outs(%[[EMPTY3:.*]] : tensor<10xf32>) -> tensor<10xf32>
// CHECK: %[[FILLTWO:.*]] = linalg.fill ins(%[[CSTTWO:.*]] : f32) outs(%[[EMPTY3:.*]] : tensor<10xf32>) -> tensor<10xf32>
// CHECK: %[[LOG2:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[FILLTWO:.*]] : tensor<10xf32>) outs(%[[EMPTY3:.*]] : tensor<10xf32>) -> tensor<10xf32>
// CHECK: %[[DIV1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LOG1:.*]], %[[LOG2:.*]] : tensor<10xf32>, tensor<10xf32>) outs(%[[EMPTY4:.*]] : tensor<10xf32>) -> tensor<10xf32>
// CHECK: %[[CAST2:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%[[DIV1:.*]] : tensor<10xf32>) outs(%[[EMPTY2:.*]] : tensor<10xf16>) -> tensor<10xf16>
// CHECK: %[[EMPTY5:.*]] = tensor.empty() : tensor<10xf16>
// CHECK: %[[EMPTY6:.*]] = tensor.empty() : tensor<10xf32>
// CHECK: %[[CAST3:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[CAST2:.*]] : tensor<10xf16>) outs(%[[EMPTY6:.*]] : tensor<10xf32>) -> tensor<10xf32>
// CHECK: %[[EMPTY7:.*]] = tensor.empty() : tensor<10xf32>
// CHECK: %[[CAST4:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<floor>} ins(%[[CAST3:.*]] : tensor<10xf32>) outs(%[[EMPTY7:.*]] : tensor<10xf32>) -> tensor<10xf32>
// CHECK: %[[CAST5:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<floor>} ins(%[[CAST4:.*]] : tensor<10xf32>) outs(%[[EMPTY5:.*]] : tensor<10xf16>) -> tensor<10xf16>
// CHECK: %[[EMPTY8:.*]] = tensor.empty() : tensor<10xf16>
// CHECK: %[[ADD:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[CAST5:.*]], %[[FILLONE:.*]] : tensor<10xf16>, tensor<10xf16>) outs(%[[EMPTY8:.*]] : tensor<10xf16>) -> tensor<10xf16>
// CHECK: %[[EMPTY9:.*]] = tensor.empty() : tensor<10xf16>
// CHECK: %[[SUB:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%3, %23 : tensor<10xf16>, tensor<10xf16>) outs(%[[EMPTY9:.*]] : tensor<10xf16>) -> tensor<10xf16>
// CHECK: %[[EMPTY10:.*]] = tensor.empty() : tensor<10xf16>
// CHECK: %[[MUL:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ARG0:.*]], %[[SUB:.*]] : tensor<10xf16>, tensor<10xf16>) outs(%[[EMPTY10:.*]] : tensor<10xf16>) -> tensor<10xf16>
func.func @test_hfusion_frexp(%arg0: tensor<10xf16>) -> tensor<10xf16>{
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

// CHECK-LABEL: func.func @test_linalg_sub_sv_to_muls_and_adds
// CHECK-SAME: (%[[arg0:.*]]: tensor<16xf32>)
// CHECK: %[[N1:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[F5:.*]] = arith.constant 5.000000e+00 : f32
// CHECK: %[[INIT0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[INIT1:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[MUL:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[arg0]], %[[N1]] : tensor<16xf32>, f32) outs(%[[INIT1]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[ADD:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[F5]], %[[MUL]] : f32, tensor<16xf32>) outs(%[[INIT0]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: return %[[ADD]]
func.func @test_linalg_sub_sv_to_muls_and_adds(%arg0: tensor<16xf32>) -> tensor<16xf32>{
  %0 = tensor.empty(): tensor<16xf32>
  %cst = arith.constant 5.0 : f32
  %d = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%cst ,%arg0: f32, tensor<16xf32>) outs(%0: tensor<16xf32>) -> tensor<16xf32>
  return %d : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_powf_5
// CHECK: %[[cst_5:.*]] = arith.constant 5.000000e+00 : f32
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[fill:.*]] = linalg.fill ins(%[[cst_5:.*]] : f32) outs(%[[empty:.*]] : tensor<1xf32>) -> tensor<1xf32>
func.func @test_hfusion_powf_5(%arg0: tensor<1xf32>) -> tensor<1xf32>{
  %0 = tensor.empty(): tensor<1xf32>
  %cst_5 = arith.constant dense<5.000000e+00> : tensor<1xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0, %cst_5: tensor<1xf32>, tensor<1xf32>) outs(%0: tensor<1xf32>) -> tensor<1xf32>
  return %res : tensor<1xf32>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_powf_0
// CHECK: %[[cst_0:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[res:.*]] = linalg.fill ins(%[[cst_0:.*]] : f32) outs(%[[empty:.*]] : tensor<1xf32>) -> tensor<1xf32>
func.func @test_hfusion_powf_0(%arg0: tensor<1xf32>) -> tensor<1xf32>{
  %0 = tensor.empty(): tensor<1xf32>
  %cst_0 = arith.constant dense<0.0> : tensor<1xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0, %cst_0: tensor<1xf32>, tensor<1xf32>) outs(%0: tensor<1xf32>) -> tensor<1xf32>
  return %res : tensor<1xf32>
}

// -----
// CHECK-LABEL: func.func @test_normalize_hfusion_powi_i64
func.func @test_normalize_hfusion_powi_i64(%arg0 : tensor<4x2x32xi64>, %arg1 : tensor<4x2x32xi64>) -> tensor<4x2x32xi64> {
  %0 = tensor.empty() : tensor<4x2x32xi64>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powi>} ins(%arg0,  %arg1: tensor<4x2x32xi64>, tensor<4x2x32xi64>) outs(%0: tensor<4x2x32xi64>) -> tensor<4x2x32xi64>
  return %1 : tensor<4x2x32xi64>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_powf_f32
// CHECK: %[[cst_nan:.*]] = arith.constant 0x7FC00000 : f32
// CHECK: %[[cst:.*]] = arith.constant 2.13909504E+9 : f32
// CHECK: %[[cst_1:.*]] = arith.constant -2.000000e+00 : f32
// CHECK: %[[cst_2:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[cst_3:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[c1_i32:.*]] = arith.constant -1 : i32
// CHECK: %[[c31_i32:.*]] = arith.constant 31 : i32
// CHECK: %[[empty0:.*]] = tensor.empty() : tensor<16xi32>
// CHECK: %[[bitcast:.*]] = hfusion.bitcast ins(%arg0 : tensor<16xf32>) outs(%[[empty0:.*]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[empty1:.*]] = tensor.empty() : tensor<16xi32>
// CHECK: %[[shift:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%[[bitcast:.*]],  %[[c31_i32:.*]] : tensor<16xi32>, i32) outs(%[[empty1]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[empty2:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq0:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[shift:.*]], %[[c1_i32:.*]] : tensor<16xi32>, i32) outs(%[[empty2]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty4:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[cast1:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<floor>} ins(%arg1 : tensor<16xf32>) outs(%[[empty4]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty5:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq1:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[cast1]], %arg1 : tensor<16xf32>, tensor<16xf32>) outs(%[[empty5]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty6:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vand0:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[cmp_eq0]], %[[cmp_eq1]] : tensor<16xi1>, tensor<16xi1>) outs(%[[empty6]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty7:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty8:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty9:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs0:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%arg1 : tensor<16xf32>) outs(%[[empty9]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty10:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[mul0:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[abs0]], %[[cst_2]] : tensor<16xf32>, f32) outs(%[[empty10]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty11:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[cast2:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[mul0]] : tensor<16xf32>) outs(%[[empty11]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty12:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins({{.*}}, {{.*}} : f32, tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty13:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins({{.*}}, {{.*}} : tensor<16xf32>, tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty14:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins({{.*}}, {{.*}} : tensor<16xf32>, f32) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty15:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins({{.*}}, {{.*}} : tensor<16xf32>, f32) outs({{.*}} : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty16:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[empty17:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins({{.*}}, {{.*}} : tensor<16xi1>, tensor<16xi1>) outs({{.*}} : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty18:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: hfusion.select ins({{.*}}, {{.*}}, {{.*}} : tensor<16xi1>, f32, tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty19:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins({{.*}} : tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty20:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty21:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty22:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty23:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: return {{.*}} : tensor<16xf32>
func.func @test_hfusion_powf_f32(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32>{
  %0 = tensor.empty(): tensor<16xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0, %arg1: tensor<16xf32>, tensor<16xf32>) outs(%0: tensor<16xf32>) -> tensor<16xf32>
  return %res : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_powf_cast_fill
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>}
func.func @test_hfusion_powf_cast_fill(%arg0: tensor<16xf32>) -> tensor<16xf32>{
  %0 = tensor.empty(): tensor<16xf32>
  %cst_1 = arith.constant 0.5 : f16
  %1 = tensor.empty(): tensor<16xf16>
  %2 = linalg.fill ins(%cst_1 : f16) outs(%1 : tensor<16xf16>) -> tensor<16xf16>
  %3 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%2 : tensor<16xf16>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0, %3: tensor<16xf32>, tensor<16xf32>) outs(%0: tensor<16xf32>) -> tensor<16xf32>
  return %res : tensor<16xf32>
}

// -----
// CHECK-LABEL: func.func @test_hfusion_powf_f16(
// CHECK: %[[cst:.*]] = arith.constant 2.13909504E+9 : f32
// CHECK: %[[cst_1:.*]] = arith.constant -2.000000e+00 : f32
// CHECK: %[[cst_2:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[cst_3:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[c1_i32:.*]] = arith.constant -1 : i32
// CHECK: %[[c31_i32:.*]] = arith.constant 31 : i32
// CHECK: %[[tmp0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[cast0_f32:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<16xf16>) outs(%[[tmp0]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[tmp1:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[cast1_f32:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<16xf16>) outs(%[[tmp1]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty0:.*]] = tensor.empty() : tensor<16xi32>
// CHECK: %[[bitcast:.*]] = hfusion.bitcast ins(%[[cast0_f32]] : tensor<16xf32>) outs(%[[empty0]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[empty1:.*]] = tensor.empty() : tensor<16xi32>
// CHECK: %[[shift:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%[[bitcast]], %[[c31_i32]] : tensor<16xi32>, i32) outs(%[[empty1]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[empty2:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq0:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[shift]], %[[c1_i32]] : tensor<16xi32>, i32) outs(%[[empty2]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty4:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[cast1:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<floor>} ins(%[[cast1_f32]] : tensor<16xf32>) outs(%[[empty4]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty5:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq1:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[cast1]], %[[cast1_f32]] : tensor<16xf32>, tensor<16xf32>) outs(%[[empty5]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty6:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vand0:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[cmp_eq0]], %[[cmp_eq1]] : tensor<16xi1>, tensor<16xi1>) outs(%[[empty6]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty7:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty8:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty9:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs0:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[cast1_f32]] : tensor<16xf32>) outs(%[[empty9]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty10:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[mul0:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[abs0]], %[[cst_2]] : tensor<16xf32>, f32) outs(%[[empty10]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty11:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[cast2:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[mul0]] : tensor<16xf32>) outs(%[[empty11]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty12:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins({{.*}}, {{.*}} : f32, tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty13:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins({{.*}}, {{.*}} : tensor<16xf32>, tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty14:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins({{.*}}, {{.*}} : tensor<16xf32>, f32) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty15:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins({{.*}}, {{.*}} : tensor<16xf32>, f32) outs({{.*}} : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty16:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[empty17:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins({{.*}}, {{.*}} : tensor<16xi1>, tensor<16xi1>) outs({{.*}} : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty18:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: hfusion.select ins({{.*}}, {{.*}}, {{.*}} : tensor<16xi1>, f32, tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty19:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins({{.*}} : tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty20:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty21:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty22:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty23:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: return {{.*}} : tensor<16xf16>
func.func @test_hfusion_powf_f16(%arg0: tensor<16xf16>, %arg1: tensor<16xf16>) -> tensor<16xf16> {
  %0 = tensor.empty() : tensor<16xf16>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0, %arg1 : tensor<16xf16>, tensor<16xf16>) outs(%0 : tensor<16xf16>) -> tensor<16xf16>
  return %res : tensor<16xf16>
}


// -----

// CHECK-LABEL: func.func @test_normalize_hfusion_powi_i8
// CHECK: %[[VAL_0:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x2x32xi8>) outs(%[[empty0:.*]] : tensor<4x2x32xf16>) -> tensor<4x2x32xf16>
// CHECK: %[[VAL_1:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_0]] : tensor<4x2x32xf16>) outs(%[[empty1:.*]] : tensor<4x2x32xf32>) -> tensor<4x2x32xf32>
// CHECK: %[[VAL_2:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<4x2x32xi8>) outs(%[[empty2:.*]] : tensor<4x2x32xf16>) -> tensor<4x2x32xf16>
// CHECK: %[[VAL_3:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_2:.*]] : tensor<4x2x32xf16>) outs(%[[empty3:.*]] : tensor<4x2x32xf32>) -> tensor<4x2x32xf32>
// CHECK: %[[VAL_4:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[_:.*]] : tensor<4x2x32xf32>) outs(%[[_:.*]] : tensor<4x2x32xi32>) -> tensor<4x2x32xi32>
// CHECK: %[[result:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} ins(%[[_:.*]] : tensor<4x2x32xi32>) outs(%[[empty4:.*]] : tensor<4x2x32xi8>) -> tensor<4x2x32xi8>
// CHECK: return %[[result:.*]] : tensor<4x2x32xi8>

func.func @test_normalize_hfusion_powi_i8(%arg0 : tensor<4x2x32xi8>, %arg1 : tensor<4x2x32xi8>) -> tensor<4x2x32xi8> {
  %0 = tensor.empty() : tensor<4x2x32xi8>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powi>} ins(%arg0,  %arg1: tensor<4x2x32xi8>, tensor<4x2x32xi8>) outs(%0: tensor<4x2x32xi8>) -> tensor<4x2x32xi8>
  return %1 : tensor<4x2x32xi8>
}

// -----

// CHECK-LABEL: func.func @test_normalize_hfusion_powi_i16
// CHECK: %[[VAL_0:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x2x32xi16>) outs(%[[empty0:.*]] : tensor<4x2x32xf32>) -> tensor<4x2x32xf32>
// CHECK: %[[VAL_1:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<4x2x32xi16>) outs(%[[empty1:.*]] : tensor<4x2x32xf32>) -> tensor<4x2x32xf32>
// CHECK: %[[Result:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[empty2:.*]] : tensor<4x2x32xf32>) outs(%[[empty4:.*]] : tensor<4x2x32xi32>) -> tensor<4x2x32xi32>
// CHECK: %[[Result:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} ins(%[[empty4:.*]] : tensor<4x2x32xi32>) outs(%[[empty3:.*]] : tensor<4x2x32xi16>) -> tensor<4x2x32xi16>
// CHECK: return %[[Result]] : tensor<4x2x32xi16>

func.func @test_normalize_hfusion_powi_i16(%arg0 : tensor<4x2x32xi16>, %arg1 : tensor<4x2x32xi16>) -> tensor<4x2x32xi16> {
  %0 = tensor.empty() : tensor<4x2x32xi16>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powi>} ins(%arg0,  %arg1: tensor<4x2x32xi16>, tensor<4x2x32xi16>) outs(%0: tensor<4x2x32xi16>) -> tensor<4x2x32xi16>
  return %1 : tensor<4x2x32xi16>
}

// -----

// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_1:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>}
// CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_3:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>}
// CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: %[[VAL_7:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<floor>}
func.func @test_floordivsi(%arg0: tensor<6x6xi32>, %arg1: tensor<6x6xi32>) -> tensor<6x6xi32> {
  %0 = tensor.empty() : tensor<6x6xi32>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<floordivsi>} ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0 : tensor<6x6xi32>) -> tensor<6x6xi32>
  return %1 : tensor<6x6xi32>
}


// -----

// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_1:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>}
// CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_3:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>}
// CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: %[[VAL_7:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<ceil>}
func.func @test_ceildivsi(%arg0: tensor<6x6xi32>, %arg1: tensor<6x6xi32>) -> tensor<6x6xi32> {
  %0 = tensor.empty() : tensor<6x6xi32>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivsi>} ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0 : tensor<6x6xi32>) -> tensor<6x6xi32>
  return %1 : tensor<6x6xi32>
}

// -----
// CHECK-LABEL: func.func @test_hfusion_powf_half
// CHECK: %[[empty0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[res:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[arg0:.*]]: tensor<16xf32>) outs(%[[empty0:.*]] : tensor<16xf32>) -> tensor<16xf32>
func.func @test_hfusion_powf_half(%arg0: tensor<16xf32>) -> tensor<16xf32>{
  %0 = tensor.empty(): tensor<16xf32>
  %cst_1 = arith.constant 0.500000e+00 : f32
  %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0,  %1: tensor<16xf32>, tensor<16xf32>) outs(%0: tensor<16xf32>) -> tensor<16xf32>
  return %res : tensor<16xf32>
}

// -----
// CHECK-LABEL: func.func @test_hfusion_powf_const_dense
// CHECK: %[[empty0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[res:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[arg0:.*]]: tensor<16xf32>) outs(%[[empty0:.*]] : tensor<16xf32>) -> tensor<16xf32>
func.func @test_hfusion_powf_const_dense(%arg0: tensor<16xf32>) -> tensor<16xf32>{
  %0 = tensor.empty(): tensor<16xf32>
  %cst_dense = arith.constant dense<0.500000e+00> : tensor<16xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0, %cst_dense: tensor<16xf32>, tensor<16xf32>) outs(%0: tensor<16xf32>) -> tensor<16xf32>
  return %res : tensor<16xf32>
}

// -----

// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<4x2x64xi1>
// CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<4x2x64xi1>
// CHECK: %[[VAL_3:.*]] = tensor.empty() : tensor<4x2x64xf16>
// CHECK: %[[VAL_4:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_0]] : tensor<4x2x64xi1>) outs(%[[VAL_3:.*]] : tensor<4x2x64xf16>) -> tensor<4x2x64xf16>
// CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<4x2x64xf16>
// CHECK: %[[VAL_6:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_1]] : tensor<4x2x64xi1>) outs(%[[VAL_5:.*]] : tensor<4x2x64xf16>) -> tensor<4x2x64xf16>
// CHECK: %[[VAL_7:.*]] = hfusion.interleave %[[VAL_4:.*]], %[[VAL_6:.*]] : tensor<4x2x64xf16>, tensor<4x2x64xf16> -> tensor<4x2x128xf16>
// CHECK: %[[VAL_8:.*]] = tensor.empty() : tensor<4x2x128xi1>
// CHECK: %[[VAL_10:.*]] = tensor.empty() : tensor<4x2x128xi1>
// CHECK: %[[VAL_9:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[VAL_7:.*]], %{{.*}} : tensor<4x2x128xf16>, f16) outs(%[[VAL_10:.*]] : tensor<4x2x128xi1>) -> tensor<4x2x128xi1>
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[VAL_9]] : tensor<4x2x128xi1>) outs(%[[VAL_8]] : tensor<4x2x128xi1>) -> tensor<4x2x128xi1>
func.func @test_interleave_i1() -> tensor<4x2x128xi1> {
  %0 = tensor.empty() : tensor<4x2x64xi1>
  %1 = tensor.empty() : tensor<4x2x64xi1>
  %2 = hfusion.interleave %0, %1 : tensor<4x2x64xi1>, tensor<4x2x64xi1> -> tensor<4x2x128xi1>
  return %2 : tensor<4x2x128xi1>
}

// -----

// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<4x2x64xi8>
// CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<4x2x64xi8>
// CHECK: %[[VAL_3:.*]] = tensor.empty() : tensor<4x2x64xf16>
// CHECK: %[[VAL_4:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_0]] : tensor<4x2x64xi8>) outs(%[[VAL_3:.*]] : tensor<4x2x64xf16>) -> tensor<4x2x64xf16>
// CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<4x2x64xf16>
// CHECK: %[[VAL_6:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_1]] : tensor<4x2x64xi8>) outs(%[[VAL_5:.*]] : tensor<4x2x64xf16>) -> tensor<4x2x64xf16>
// CHECK: %[[VAL_7:.*]] = hfusion.interleave %[[VAL_4:.*]], %[[VAL_6:.*]] : tensor<4x2x64xf16>, tensor<4x2x64xf16> -> tensor<4x2x128xf16>
// CHECK: %[[VAL_8:.*]] = tensor.empty() : tensor<4x2x128xi8>
// CHECK: %[[VAL_9:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins(%[[VAL_7:.*]] : tensor<4x2x128xf16>) outs(%[[VAL_8:.*]] : tensor<4x2x128xi8>) -> tensor<4x2x128xi8>
func.func @test_interleave_i8() -> tensor<4x2x128xi8> {
  %0 = tensor.empty() : tensor<4x2x64xi8>
  %1 = tensor.empty() : tensor<4x2x64xi8>
  %2 = hfusion.interleave %0, %1 : tensor<4x2x64xi8>, tensor<4x2x64xi8> -> tensor<4x2x128xi8>
  return %2 : tensor<4x2x128xi8>
}

// -----

// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<4x2x64xf16>
// CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<4x2x64xf16>
// CHECK: %[[VAL_2:.*]] = hfusion.interleave %[[VAL_0]], %[[VAL_1]] : tensor<4x2x64xf16>, tensor<4x2x64xf16> -> tensor<4x2x128xf16>
func.func @test_interleave_f16() -> tensor<4x2x128xf16> {
  %0 = tensor.empty() : tensor<4x2x64xf16>
  %1 = tensor.empty() : tensor<4x2x64xf16>
  %2 = hfusion.interleave %0, %1 : tensor<4x2x64xf16>, tensor<4x2x64xf16> -> tensor<4x2x128xf16>
  return %2 : tensor<4x2x128xf16>
}

// -----

// CHECK-LABEL: @test_normalize_reduce_with_index_ra_to_ar
// CHECK: %[[res0:.*]] = tensor.empty() : tensor<32x128xf32>
// CHECK: %[[res1:.*]] = tensor.empty() : tensor<32x128xi32>
// CHECK: %[[tmp_buf0:.*]] = tensor.empty() : tensor<32x128x32xf32>
// CHECK: %[[transposed0:.*]] = linalg.transpose ins(%[[arg0:.*]] : tensor<32x32x128xf32>) outs(%[[tmp_buf0]] : tensor<32x128x32xf32>) permutation = [0, 2, 1]
// CHECK: %[[tmp_buf1:.*]] = tensor.empty() : tensor<32x128x32xi32>
// CHECK: %[[transposed1:.*]] = linalg.transpose ins(%[[arg1:.*]] : tensor<32x32x128xi32>) outs(%[[tmp_buf1]] : tensor<32x128x32xi32>) permutation = [0, 2, 1]
// CHECK: hfusion.reduce_with_index {already_denaned, tie_break_left = true} <min> ins(%[[transposed0]], %[[transposed1]] : tensor<32x128x32xf32>, tensor<32x128x32xi32>) outs(%[[res0]], %[[res1]] : tensor<32x128xf32>, tensor<32x128xi32>) dimensions = [2]
func.func @test_normalize_reduce_with_index_ra_to_ar(%arg0: tensor<32x32x128xf32>, %arg1: tensor<32x32x128xi32>) -> tensor<32x128xi32> {
  %true = arith.constant true
  %0 = tensor.empty() : tensor<32x128xf32>
  %1 = tensor.empty() : tensor<32x128xi32>
  %reduced:2 = hfusion.reduce_with_index {already_denaned, tie_break_left = true} <min>
                ins(%arg0, %arg1 : tensor<32x32x128xf32>, tensor<32x32x128xi32>)
                outs(%0, %1 : tensor<32x128xf32>, tensor<32x128xi32>)
                dimensions = [1] -> tensor<32x128xf32>, tensor<32x128xi32>

  return %reduced#1 : tensor<32x128xi32>
}

// -----

// CHECK-LABEL: @test_normalize_reduce_denaned
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[MIN_INF:.*]] = arith.constant 0xFF800000 : f32
// CHECK: %[[REAL_REDUCE:.*]]:2 = hfusion.reduce_with_index {already_denaned, tie_break_left = true} <min> ins(%[[SRC:.*]], %[[IDXS:.*]] : tensor<32x128x32xf32>, tensor<32x128x32xi32>)
// CHECK: %[[NAN_MASK:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
// CHECK: %[[DENANED_T:.*]] = hfusion.select ins(%[[NAN_MASK]], %[[MIN_INF]], %[[ZERO]] : tensor<32x128x32xi1>, f32, f32)
// CHECK: %[[DENANED_REDUCE:.*]]:2 = hfusion.reduce_with_index {already_denaned, tie_break_left = true} <min> ins(%[[DENANED_T]], %[[IDXS]] : tensor<32x128x32xf32>, tensor<32x128x32xi32>)
// CHECK: %[[INF_MASK:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
// CHECK: %[[RES:.*]] = hfusion.select ins(%[[INF_MASK]], %[[DENANED_REDUCE]]#1, %[[REAL_REDUCE]]#1 : tensor<32x128xi1>, tensor<32x128xi32>, tensor<32x128xi32>)
// CHECK: return %[[RES]]
func.func @test_normalize_reduce_denaned(%arg0: tensor<32x32x128xf32>, %arg1: tensor<32x32x128xi32>) -> tensor<32x128xi32> {
  %true = arith.constant true
  %0 = tensor.empty() : tensor<32x128xf32>
  %1 = tensor.empty() : tensor<32x128xi32>
  %reduced:2 = hfusion.reduce_with_index {tie_break_left = true} <min>
                ins(%arg0, %arg1 : tensor<32x32x128xf32>, tensor<32x32x128xi32>)
                outs(%0, %1 : tensor<32x128xf32>, tensor<32x128xi32>)
                dimensions = [1] -> tensor<32x128xf32>, tensor<32x128xi32>

  return %reduced#1 : tensor<32x128xi32>
}

// -----

// CHECK-LABEL: func.func @opt_cast_IToF_fill
// CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<24x32xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<24x32xf32>) -> tensor<24x32xf32>
// CHECK: return %[[FILL]] : tensor<24x32xf32>
func.func @opt_cast_IToF_fill() -> tensor<24x32xf32>{
  %c1_i32 = arith.constant 1 : i32
  %0 = tensor.empty() : tensor<24x32xf32>
  %1 = tensor.empty() : tensor<24x32xi32>
  %2 = linalg.fill ins(%c1_i32 : i32) outs(%1 : tensor<24x32xi32>) -> tensor<24x32xi32>
  %3 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%2 : tensor<24x32xi32>) outs(%0 : tensor<24x32xf32>) -> tensor<24x32xf32>
  return %3 : tensor<24x32xf32>
}
// -----

// CHECK-LABEL: func.func @opt_cast_FToI_fill_rint
// CHECK: %[[CST:.*]] = arith.constant 1.500000e+00 : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<f32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY0]] : tensor<f32>) -> tensor<f32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<i32>
// CHECK: %[[CAST:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<ceil>} ins(%[[FILL]] : tensor<f32>) outs(%[[EMPTY1]] : tensor<i32>) -> tensor<i32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<24x32xi32>
// CHECK: %[[BRC:.*]] = linalg.broadcast ins(%[[CAST]] : tensor<i32>) outs(%[[EMPTY2]] : tensor<24x32xi32>) dimensions = [0, 1]
// CHECK: return %[[BRC]] : tensor<24x32xi32>
func.func @opt_cast_FToI_fill_rint() -> tensor<24x32xi32>{
  %c1_i32 = arith.constant 1.5 : f32
  %0 = tensor.empty() : tensor<24x32xf32>
  %1 = tensor.empty() : tensor<24x32xi32>
  %2 = linalg.fill ins(%c1_i32 : f32) outs(%0 : tensor<24x32xf32>) -> tensor<24x32xf32>
  %3 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<ceil>} ins(%2 : tensor<24x32xf32>) outs(%1 : tensor<24x32xi32>) -> tensor<24x32xi32>
  return %3 : tensor<24x32xi32>
}

// -----

// CHECK-LABEL: func.func @opt_cast_FToI_brc
// CHECK: %[[CST:.*]] = arith.constant dense<1.500000e+00> : tensor<32xf32>
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<32xi32>
// CHECK: %[[CAST:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[CST]] : tensor<32xf32>) outs(%[[EMPTY0]] : tensor<32xi32>) -> tensor<32xi32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<24x32xi32>
// CHECK: %[[BRC:.*]] = linalg.broadcast ins(%[[CAST]] : tensor<32xi32>) outs(%[[EMPTY1]] : tensor<24x32xi32>) dimensions = [0]
// CHECK: return %[[BRC]] : tensor<24x32xi32>
func.func @opt_cast_FToI_brc() -> tensor<24x32xi32>{
  %c1_f32 = arith.constant dense<1.5> : tensor<32xf32>
  %0 = tensor.empty() : tensor<24x32xi32>
  %1 = tensor.empty() : tensor<24x32xf32>
  %2 = linalg.broadcast ins(%c1_f32 : tensor<32xf32>) outs(%1 : tensor<24x32xf32>) dimensions=[0]
  %3 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%2 : tensor<24x32xf32>) outs(%0 : tensor<24x32xi32>) -> tensor<24x32xi32>
  return %3 : tensor<24x32xi32>
}

// -----

// CHECK-LABEL: @test_normalize_interleave_i1
// CHECK :  %[[res_f16:.*]] = hfusion.interleave %[[arg0_f16:.*]], %[[arg1_f16:.*]] : tensor<4x2x32xf16>, tensor<4x2x32xf16> -> tensor<4x2x64xf16>
func.func @test_normalize_interleave_i1(%arg0 : tensor<4x2x32xi1>, %arg1 : tensor<4x2x32xi1>) -> tensor<4x2x64xi1> {
  %0 = tensor.empty() : tensor<4x2x64xi1>
  %1 = hfusion.interleave %arg0, %arg1 : tensor<4x2x32xi1>, tensor<4x2x32xi1> -> tensor<4x2x64xi1>
  return %1 : tensor<4x2x64xi1>
}

// -----

// CHECK-LABEL: @test_normalize_interleave_i8
// CHECK :  %[[res_f16:.*]] = hfusion.interleave %[[arg0_f16:.*]], %[[arg1_f16:.*]] : tensor<4x2x32xf16>, tensor<4x2x32xf16> -> tensor<4x2x64xf16>
func.func @test_normalize_interleave_i8(%arg0 : tensor<4x2x32xi8>, %arg1 : tensor<4x2x32xi8>) -> tensor<4x2x64xi8> {
  %0 = tensor.empty() : tensor<4x2x64xi8>
  %1 = hfusion.interleave %arg0, %arg1 : tensor<4x2x32xi8>, tensor<4x2x32xi8> -> tensor<4x2x64xi8>
  return %1 : tensor<4x2x64xi8>
}

// -----

// CHECK-LABEL: @test_normalize_deinterleave_i8
// CHECK: %[[res_f16:.*]] = hfusion.deinterleave %[[cast_f16:.*]] channel<1> : tensor<4x2x128xf16> -> tensor<4x2x64xf16>
func.func @test_normalize_deinterleave_i8() -> tensor<4x2x64xi8> {
  %0 = tensor.empty() : tensor<4x2x128xi8>
  %1 = hfusion.deinterleave %0 channel<1> : tensor<4x2x128xi8> -> tensor<4x2x64xi8>
  return %1 : tensor<4x2x64xi8>
}

// -----


// CHECK-LABEL: func.func @test_hfusion_tanh_ops(
// CHECK-SAME: %[[VAL_0:.*]]: tensor<32xf32>) -> tensor<32xf32> {
// CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[CST0:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[CST1:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[CST2:.*]] = arith.constant -8.800000e+00 : f32
// CHECK: %[[CST3:.*]] = arith.constant 8.800000e+00 : f32
// CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_2:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[VAL_0]], %[[CST3]] : tensor<32xf32>, f32) outs(%[[VAL_1]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_3:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%[[VAL_2]], %[[CST2]] : tensor<32xf32>, f32) outs(%[[VAL_1]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST1]] : tensor<32xf32>, f32) outs(%[[VAL_4]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_6:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[VAL_5]] : tensor<32xf32>) outs(%[[VAL_4]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_7:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_8:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_6]], %[[CST0]] : tensor<32xf32>, f32) outs(%[[VAL_7]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_9:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_10:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_6]], %[[CST]] : tensor<32xf32>, f32) outs(%[[VAL_9]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_11:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[VAL_8]], %[[VAL_10]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_7]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: return %[[VAL_11]] : tensor<32xf32>
// CHECK: }
func.func @test_hfusion_tanh_ops(%arg0 : tensor<32xf32>) ->  tensor<32xf32> {
  %0 = tensor.empty() : tensor<32xf32>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<tanh>} ins(%arg0 : tensor<32xf32>) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  return %ret : tensor<32xf32>
}
// -----

// CHECK-LABEL: func.func @test_hfusion_tanh_ops_f16
// CHECK: %[[CST_1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[CST_NEG1:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[CST_2:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[CST_NEG8:.*]] = arith.constant -8.800000e+00 : f32
// CHECK: %[[CST_8DOT8:.*]] = arith.constant 8.800000e+00 : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[CAST0:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%[[ARG0:.*]] : tensor<32xf16>) outs(%[[EMPTY0:.*]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[MINF:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[CAST0:.*]], %[[CST_8DOT8:.*]] : tensor<32xf32>, f32) outs(%[[EMPTY1:.*]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[MAXF:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%[[MINF:.*]], %[[CST_NEG8:.*]] : tensor<32xf32>, f32) outs(%[[EMPTY1:.*]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[MUL:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[MAXF:.*]], %[[CST_2:.*]] : tensor<32xf32>, f32) outs(%[[EMPTY2:.*]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EXP:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[MUL:.*]] : tensor<32xf32>) outs(%[[EMPTY2:.*]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[ADD0:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[EXP:.*]], %[[CST_NEG1:.*]] : tensor<32xf32>, f32) outs(%[[EMPTY3:.*]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EMPTY4:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[ADD1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[EXP:.*]], %[[CST_1:.*]] : tensor<32xf32>, f32) outs(%[[EMPTY4:.*]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[DIV:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[ADD0:.*]], %[[ADD1:.*]] : tensor<32xf32>, tensor<32xf32>) outs(%[[EMPTY3:.*]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EMPTY5:.*]] = tensor.empty() : tensor<32xf16>
// CHECK: %[[RES:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%[[DIV:.*]] : tensor<32xf32>) outs(%[[EMPTY5:.*]] : tensor<32xf16>) -> tensor<32xf16>
func.func @test_hfusion_tanh_ops_f16(%arg0 : tensor<32xf16>) ->  tensor<32xf16> {
  %0 = tensor.empty() : tensor<32xf16>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<tanh>} ins(%arg0 : tensor<32xf16>) outs(%0 : tensor<32xf16>) -> tensor<32xf16>
  return %ret : tensor<32xf16>
}
// -----

// CHECK-LABEL: @normalize_mulext_i8_high_bits
// CHECK: %[[cst_8:.*]] = arith.constant 8 : i16
// CHECK: %[[empty0:.*]] = tensor.empty() : tensor<4x2xf16>
// CHECK: %[[arg0_f16:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[arg0:.*]] : tensor<4x2xi8>) outs(%[[empty0:.*]] : tensor<4x2xf16>) -> tensor<4x2xf16>
// CHECK: %[[empty1:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[arg0_i16:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[arg0_f16:.*]] : tensor<4x2xf16>) outs(%[[empty1:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
// CHECK: %[[empty2:.*]] = tensor.empty() : tensor<4x2xf16>
// CHECK: %[[arg1_f16:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[arg1:.*]] : tensor<4x2xi8>) outs(%[[empty2:.*]] : tensor<4x2xf16>) -> tensor<4x2xf16>
// CHECK: %[[empty3:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[arg1_i16:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[arg1_f16:.*]] : tensor<4x2xf16>) outs(%[[empty3:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
// CHECK: %[[empty4:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[mul:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[arg0_i16:.*]], %[[arg1_i16:.*]] : tensor<4x2xi16>, tensor<4x2xi16>) outs(%[[empty4:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
// CHECK: %[[empty5:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[res_i16:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%[[mul:.*]], %[[cst_8:.*]] : tensor<4x2xi16>, i16) outs(%[[empty5:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
func.func @normalize_mulext_i8_high_bits(%arg0: tensor<4x2xi8>, %arg1: tensor<4x2xi8>) -> tensor<4x2xi8> {
  %low, %high = hfusion.mulext %arg0, %arg1 : tensor<4x2xi8>
  return %high : tensor<4x2xi8>
}

// -----

// CHECK-LABEL: @normalize_mulext_i8_low_bits
// CHECK: %[[cst_8:.*]] = arith.constant 8 : i16
// CHECK: %[[empty0:.*]] = tensor.empty() : tensor<4x2xf16>
// CHECK: %[[arg0_f16:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[arg0:.*]] : tensor<4x2xi8>) outs(%[[empty0:.*]] : tensor<4x2xf16>) -> tensor<4x2xf16>
// CHECK: %[[empty1:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[arg0_i16:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[arg0_f16:.*]] : tensor<4x2xf16>) outs(%[[empty1:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
// CHECK: %[[empty2:.*]] = tensor.empty() : tensor<4x2xf16>
// CHECK: %[[arg1_f16:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[arg1:.*]] : tensor<4x2xi8>) outs(%[[empty2:.*]] : tensor<4x2xf16>) -> tensor<4x2xf16>
// CHECK: %[[empty3:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[arg1_i16:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[arg1_f16:.*]] : tensor<4x2xf16>) outs(%[[empty3:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
// CHECK: %[[empty4:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[mul:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[arg0_i16:.*]], %[[arg1_i16:.*]] : tensor<4x2xi16>, tensor<4x2xi16>) outs(%[[empty4:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
// CHECK: %[[empty5:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[shl:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shli>} ins(%[[mul:.*]], %[[cst_8:.*]] : tensor<4x2xi16>, i16) outs(%[[empty5:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
// CHECK: %[[empty6:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[res_i16:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%[[shl:.*]], %c8_i16 : tensor<4x2xi16>, i16) outs(%[[empty6:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
func.func @normalize_mulext_i8_low_bits(%arg0: tensor<4x2xi8>, %arg1: tensor<4x2xi8>) -> tensor<4x2xi8> {
  %low, %high = hfusion.mulext %arg0, %arg1 : tensor<4x2xi8>
  return  %low : tensor<4x2xi8>
}

// CHECK-LABEL: @normalize_vlog_f16_to_f32
// CHECK: %[[a0:.*]] = tensor.empty() : tensor<17x256xf16>
// CHECK: %[[a1:.*]] = tensor.empty() : tensor<17x256xf32>
// CHECK: %[[a2:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[arg0:.*]] : tensor<17x256xf16>) outs(%[[a1]] : tensor<17x256xf32>) -> tensor<17x256xf32>
// CHECK: %[[a3:.*]] = tensor.empty() : tensor<17x256xf32>
// CHECK: %[[a4:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[a0]] : tensor<17x256xf16>) outs(%[[a3]] : tensor<17x256xf32>) -> tensor<17x256xf32>
// CHECK: %[[a5:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[a2]] : tensor<17x256xf32>) outs(%[[a4]] : tensor<17x256xf32>) -> tensor<17x256xf32>
// CHECK: %[[a6:.*]] = tensor.empty() : tensor<17x256xf16>
// CHECK: %[[a7:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[a5]] : tensor<17x256xf32>) outs(%[[a6]] : tensor<17x256xf16>) -> tensor<17x256xf16>
func.func @normalize_vlog_f16_to_f32(%arg0: tensor<17x256xf16>) -> tensor<17x256xf16> {
  %0 = tensor.empty() : tensor<17x256xf16>
  %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg0 : tensor<17x256xf16>) outs(%0 : tensor<17x256xf16>) -> tensor<17x256xf16>
  return %1 : tensor<17x256xf16>
}

// -----
// CHECK-LABEL: func.func @test_isnan
// CHECK: %[[INPUT:.*]] = tensor.empty() : tensor<8192xf32>
// CHECK: %[[OUT1:.*]] = tensor.empty() : tensor<8192xi1>
// CHECK: %[[OUT2:.*]] = tensor.empty() : tensor<8192xi1>
// CHECK: %[[RES:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[INPUT]], %[[INPUT]] : tensor<8192xf32>, tensor<8192xf32>) outs(%[[OUT2]] : tensor<8192xi1>) -> tensor<8192xi1>
// CHECK: %[[RES2:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[RES]] : tensor<8192xi1>) outs(%[[OUT1]] : tensor<8192xi1>) -> tensor<8192xi1>
func.func @test_isnan() -> tensor<8192xi1> {
  %0 = tensor.empty() : tensor<8192xf32>
  %2 = hfusion.isnan %0 : tensor<8192xf32> -> tensor<8192xi1>
  return %2 : tensor<8192xi1>
}

// -----
// CHECK-LABEL: func.func @test_divui
// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_1:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>}
// CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_3:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>}
// CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: %[[VAL_6:.*]] = tensor.empty() : tensor<6x6xi32>
// CHECK: %[[VAL_7:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>}
func.func @test_divui(%arg0: tensor<6x6xi32>, %arg1: tensor<6x6xi32>) -> tensor<6x6xi32> {
  %0 = tensor.empty() : tensor<6x6xi32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0 : tensor<6x6xi32>) -> tensor<6x6xi32>
  return %1 : tensor<6x6xi32>
}

// -----
// CHECK-LABEL: func.func @test_divui_vs
func.func @test_divui_vs(%arg0: tensor<48xi32>, %arg1: i32) -> tensor<48xi32> {
    %res = tensor.empty() : tensor<48xi32>
    // CHECK: %[[LHS:.*]] = tensor.empty() : tensor<48xf32>
    // CHECK: %[[LHSFP:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<48xi32>) outs(%[[LHS]] : tensor<48xf32>) -> tensor<48xf32>
    // CHECK: %[[RHSCASTED:.*]] = arith.uitofp %arg1 : i32 to f32
    // CHECK: %[[RES:.*]] = tensor.empty() : tensor<48xf32>
    // CHECK: %[[RESFP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LHSFP]], %[[RHSCASTED]] : tensor<48xf32>, f32) outs(%[[RES]] : tensor<48xf32>) -> tensor<48xf32>
    // CHECK: %[[RESINT:.*]] = tensor.empty() : tensor<48xi32>
    // CHECK: %[[RET:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[RESFP]] : tensor<48xf32>) outs(%[[RESINT]] : tensor<48xi32>) -> tensor<48xi32>
    %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%arg0, %arg1 : tensor<48xi32>, i32) outs(%res : tensor<48xi32>) -> tensor<48xi32>
    return %0 : tensor<48xi32>
}

// -----
// CHECK-LABEL: func.func @test_divui_sv
func.func @test_divui_sv(%arg0: i32, %arg1: tensor<48xi32>) -> tensor<48xi32> {
    %res = tensor.empty() : tensor<48xi32>
    // CHECK: %[[LHSCASTED:.*]] = arith.uitofp %arg0 : i32 to f32
    // CHECK: %[[RHS:.*]] = tensor.empty() : tensor<48xf32>
    // CHECK: %[[RHSFP:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<48xi32>) outs(%[[RHS]] : tensor<48xf32>) -> tensor<48xf32>
    // CHECK: %[[RES:.*]] = tensor.empty() : tensor<48xf32>
    // CHECK: %[[RESFP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LHSCASTED]], %[[RHSFP]] : f32, tensor<48xf32>) outs(%[[RES]] : tensor<48xf32>) -> tensor<48xf32>
    // CHECK: %[[RESINT:.*]] = tensor.empty() : tensor<48xi32>
    // CHECK: %[[RET:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[RESFP]] : tensor<48xf32>) outs(%[[RESINT]] : tensor<48xi32>) -> tensor<48xi32>
    %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%arg0, %arg1 : i32, tensor<48xi32>) outs(%res : tensor<48xi32>) -> tensor<48xi32>
    return %0 : tensor<48xi32>
}

// -----
// CHECK-LABEL: @test_cast_f32_to_i16
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<4x4xf32>) outs({{.*}} : tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} ins({{.*}} : tensor<4x4xi32>) outs({{.*}} : tensor<4x4xi16>) -> tensor<4x4xi16>
func.func @test_cast_f32_to_i16(%arg0: tensor<4x4xf32>) -> tensor<4x4xi16> {
  %0 = tensor.empty() : tensor<4x4xi16>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} ins(%arg0 : tensor<4x4xf32>) outs(%0 : tensor<4x4xi16>) -> tensor<4x4xi16>
  return %1 : tensor<4x4xi16>
}

// -----
// CHECK-LABEL: @test_cast_i64_to_i16
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} ins({{.*}} : tensor<4x4xi64>) outs({{.*}} : tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} ins({{.*}} : tensor<4x4xi32>) outs({{.*}} : tensor<4x4xi16>) -> tensor<4x4xi16>
func.func @test_cast_i64_to_i16(%arg0: tensor<4x4xi64>) -> tensor<4x4xi16> {
  %0 = tensor.empty() : tensor<4x4xi16>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x4xi64>) outs(%0 : tensor<4x4xi16>) -> tensor<4x4xi16>
  return %1 : tensor<4x4xi16>
}

// -----
// CHECK-LABEL: @test_cast_i64_to_i8
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} ins({{.*}} : tensor<4x4xi64>) outs({{.*}} : tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} ins({{.*}} : tensor<4x4xi32>) outs({{.*}} : tensor<4x4xi8>) -> tensor<4x4xi8>
func.func @test_cast_i64_to_i8(%arg0: tensor<4x4xi64>) -> tensor<4x4xi8> {
  %0 = tensor.empty() : tensor<4x4xi8>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x4xi64>) outs(%0 : tensor<4x4xi8>) -> tensor<4x4xi8>
  return %1 : tensor<4x4xi8>
}

// -----
// CHECK-LABEL: @test_broadcast_i1
// CHECK: %[[CAST16:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<8xi1>) outs({{.*}} : tensor<8xf16>) -> tensor<8xf16>
// CHECK: %[[BROADCAST16:.*]] = linalg.broadcast ins(%[[CAST16]] : tensor<8xf16>) outs({{.*}} : tensor<8x16xf16>) dimensions = [1]
// CHECK: %[[VEQ:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[BROADCAST16]], {{.*}} : tensor<8x16xf16>, f16) outs({{.*}} : tensor<8x16xi1>) -> tensor<8x16xi1>
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[VEQ]] : tensor<8x16xi1>) outs(%{{.*}} : tensor<8x16xi1>) -> tensor<8x16xi1>
func.func @test_broadcast_i1(%arg0: tensor<8xi1>, %arg1: tensor<8x16xi1>) -> tensor<8x16xi1> {
  %0 = tensor.empty() : tensor<8x16xi1>
  %1 = linalg.broadcast
    ins(%arg0 : tensor<8xi1>)
    outs(%0 : tensor<8x16xi1>)
    dimensions = [1]
  return %1 : tensor<8x16xi1>
}

// -----
// CHECK-LABEL: @test_broadcast_i8
// CHECK: %[[CAST16:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<8xi8>) outs({{.*}} : tensor<8xf16>) -> tensor<8xf16>
// CHECK: %[[BROADCAST16:.*]] = linalg.broadcast ins(%[[CAST16]] : tensor<8xf16>) outs({{.*}} : tensor<8x16xf16>) dimensions = [1]
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins(%[[BROADCAST16]] : tensor<8x16xf16>) outs({{.*}} : tensor<8x16xi8>) -> tensor<8x16xi8>
func.func @test_broadcast_i8(%arg0: tensor<8xi8>, %arg1: tensor<8x16xi8>) -> tensor<8x16xi8> {
  %0 = tensor.empty() : tensor<8x16xi8>
  %1 = linalg.broadcast
    ins(%arg0 : tensor<8xi8>)
    outs(%0 : tensor<8x16xi8>)
    dimensions = [1]
  return %1 : tensor<8x16xi8>
}

// -----

// CHECK-LABEL: @test_cumsum_f16
// CHECK: hfusion.cumsum %[[INPUT0:.*]] : tensor<4x64xf32> cum_dims = [0] reverse = false -> tensor<4x32xf32>
module {
  func.func @test_cumsum_f16(%arg0: tensor<4x64xf16>) -> tensor<4x32xf16> {
    %0 = tensor.empty() : tensor<4x32xf16>
    %1 = hfusion.cumsum %arg0 : tensor<4x64xf16> cum_dims = [0] reverse = false -> tensor<4x32xf16>
    return %1 : tensor<4x32xf16>
  }
}

// -----

// CHECK-LABEL: @test_cumprod_f16
// CHECK: hfusion.cumprod %[[INPUT0:.*]] : tensor<4x64xf32> cum_dims = [1] reverse = false -> tensor<4x32xf32>
module {
  func.func @test_cumprod_f16(%arg0: tensor<4x64xf16>) -> tensor<4x32xf16> {
    %0 = tensor.empty() : tensor<4x32xf16>
    %1 = hfusion.cumprod %arg0 : tensor<4x64xf16> cum_dims = [1] reverse = false -> tensor<4x32xf16>
    return %1 : tensor<4x32xf16>
  }
}

// -----
// CHECK-LABEL: @test_reduce_with_index_i1_return_value
func.func @test_reduce_with_index_i1_return_value(%arg0: tensor<32x32x128xi1>, %arg1: tensor<32x32x128xi32>) -> tensor<32x128xi1> {
  %0 = tensor.empty() : tensor<32x128xi1>
  %1 = tensor.empty() : tensor<32x128xi32>
  // CHECK %[[reduce:.*]] = hfusion.reduce_with_index {tie_break_left = true} <min>  ins(%[[arg0:.*]], %[[arg1:.*] : tensor<32x32x128xf16>
  %reduced:2 = hfusion.reduce_with_index {tie_break_left = true} <min>
                ins(%arg0, %arg1 : tensor<32x32x128xi1>, tensor<32x32x128xi32>)
                outs(%0, %1 : tensor<32x128xi1>, tensor<32x128xi32>)
                dimensions = [1] -> tensor<32x128xi1>, tensor<32x128xi32>

  return %reduced#0 : tensor<32x128xi1>
}

// -----
// CHECK-LABEL: @test_reduce_with_index_i1_return_index
func.func @test_reduce_with_index_i1_return_index(%arg0: tensor<32x32x128xi1>, %arg1: tensor<32x32x128xi32>) -> tensor<32x128xi32> {
  %0 = tensor.empty() : tensor<32x128xi1>
  %1 = tensor.empty() : tensor<32x128xi32>
  // CHECK %[[reduce:.*]] = hfusion.reduce_with_index {tie_break_left = true} <min>  ins(%[[arg0:.*]], %[[arg1:.*] : tensor<32x32x128xf16>
  %reduced:2 = hfusion.reduce_with_index {tie_break_left = true} <min>
                ins(%arg0, %arg1 : tensor<32x32x128xi1>, tensor<32x32x128xi32>)
                outs(%0, %1 : tensor<32x128xi1>, tensor<32x128xi32>)
                dimensions = [1] -> tensor<32x128xi1>, tensor<32x128xi32>

  return %reduced#1 : tensor<32x128xi32>
}

// -----
// CHECK-LABEL: @test_reduce_with_index_i64_return_index
func.func @test_reduce_with_index_i64_return_index(%arg0: tensor<32x32x128xf32>, %arg1: tensor<32x32x128xi64>) -> tensor<32x128xf32> {
  %0 = tensor.empty() : tensor<32x128xf32>
  %1 = tensor.empty() : tensor<32x128xi64>
  // CHECK: %[[reduce:.*]] = hfusion.reduce_with_index {already_denaned, tie_break_left = true} <min>  ins(%[[arg0:.*]], %[[arg1:.*]] : tensor<32x128x32xf32>, tensor<32x128x32xi32>
  %reduced:2 = hfusion.reduce_with_index {tie_break_left = true} <min>
                ins(%arg0, %arg1 : tensor<32x32x128xf32>, tensor<32x32x128xi64>)
                outs(%0, %1 : tensor<32x128xf32>, tensor<32x128xi64>)
                dimensions = [1] -> tensor<32x128xf32>, tensor<32x128xi64>

  return %reduced#0 : tensor<32x128xf32>
}

// -----
// CHECK-LABEL: @test_reduce_i1_addi
func.func @test_reduce_i1_addi(%arg0: tensor<16x32x64xi1>) -> tensor<16x64xi1> {
  %0 = tensor.empty() : tensor<16x64xi1>
  // CHECK: %[[reduce:.*]] = linalg.reduce { arith.ori }
  %reduce = linalg.reduce { arith.addi } ins(%arg0 : tensor<16x32x64xi1>) outs(%0 : tensor<16x64xi1>) dimensions = [1]
  return %reduce : tensor<16x64xi1>
}

// -----
// CHECK-LABEL: @test_reduce_i1_muli
func.func @test_reduce_i1_muli(%arg0: tensor<16x32x64xi1>) -> tensor<16x64xi1> {
  %0 = tensor.empty() : tensor<16x64xi1>
  // CHECK: %[[reduce:.*]] = linalg.reduce { arith.andi }
  %reduce = linalg.reduce { arith.muli } ins(%arg0 : tensor<16x32x64xi1>) outs(%0 : tensor<16x64xi1>) dimensions = [1]
  return %reduce : tensor<16x64xi1>
}

// -----
// CHECK-LABEL: @test_reduce_i1_maxi
func.func @test_reduce_i1_maxi(%arg0: tensor<16x32x64xi1>) -> (tensor<16x64xi1>, tensor<16x64xi1>) {
  %0 = tensor.empty() : tensor<16x64xi1>
  // CHECK: %[[reduce:.*]] = linalg.reduce { arith.ori }
  // CHECK: %[[reduce:.*]] = linalg.reduce { arith.ori }
  %reduce = linalg.reduce { arith.maxui } ins(%arg0 : tensor<16x32x64xi1>) outs(%0 : tensor<16x64xi1>) dimensions = [1]
  %reduce_0 = linalg.reduce { arith.maxsi } ins(%arg0 : tensor<16x32x64xi1>) outs(%0 : tensor<16x64xi1>) dimensions = [1]
  return %reduce, %reduce_0 : tensor<16x64xi1>, tensor<16x64xi1>
}

// -----
// CHECK-LABEL: @test_reduce_i1_mini
func.func @test_reduce_i1_mini(%arg0: tensor<16x32x64xi1>) -> (tensor<16x64xi1>, tensor<16x64xi1>) {
  %0 = tensor.empty() : tensor<16x64xi1>
  // CHECK: %[[reduce:.*]] = linalg.reduce { arith.andi }
  // CHECK: %[[reduce:.*]] = linalg.reduce { arith.andi }
  %reduce = linalg.reduce { arith.minui } ins(%arg0 : tensor<16x32x64xi1>) outs(%0 : tensor<16x64xi1>) dimensions = [1]
  %reduce_0 = linalg.reduce { arith.minsi } ins(%arg0 : tensor<16x32x64xi1>) outs(%0 : tensor<16x64xi1>) dimensions = [1]
  return %reduce, %reduce_0 : tensor<16x64xi1>, tensor<16x64xi1>
}

// -----
// CHECK-LABEL: @test_compare_i1
func.func @test_compare_i1(%arg0: tensor<16x32xi1>,%arg1: tensor<16x32xi1>,  %dst : tensor<16x32xi1>) -> (tensor<16x32xi1>) {
  // CHECK: %[[ret:.*]] = hfusion.compare  {compare_fn = #hfusion.compare_fn<vlt>} ins(%[[in1:.*]], %[[in2:.*]] : tensor<16x32xf16>, tensor<16x32xf16>)
  %ret = hfusion.compare {compare_fn  = #hfusion.compare_fn<vlt>}
    ins(%arg0, %arg1 : tensor<16x32xi1>, tensor<16x32xi1>)
    outs(%dst : tensor<16x32xi1>)
    -> tensor<16x32xi1>
  return %ret : tensor<16x32xi1>
}

// -----
// CHECK-LABEL: @test_concat_i1
func.func @test_concat_i1(%arg0: tensor<2048xi1>, %arg1: tensor<2048xi1>) -> tensor<4096xi1> {
  // CHECK: %[[concat:.*]] = tensor.concat dim(0) %[[in1:.*]], %[[in2:.*]] : (tensor<2048xf16>, tensor<2048xf16>) -> tensor<4096xf16>
  %0 = tensor.concat dim(0) %arg0, %arg1 : (tensor<2048xi1>, tensor<2048xi1>) -> tensor<4096xi1>
  return %0 : tensor<4096xi1>
}

// -----
// CHECK-LABEL: @test_transpose_i1
func.func @test_transpose_i1() -> tensor<8x32xi1> {
  %src = tensor.empty() : tensor<32x8xi1>
  %dst = tensor.empty() : tensor<8x32xi1>
  // CHECK: %[[transposed:.*]] = linalg.transpose ins(%[[in1:.*]] : tensor<32x8xf16>) outs(%[[in2:.*]] : tensor<8x32xf16>) permutation = [1, 0]
  %transposed = linalg.transpose ins(%src : tensor<32x8xi1>) outs(%dst : tensor<8x32xi1>) permutation = [1, 0]
  return %transposed : tensor<8x32xi1>
}

// -----
// CHECK-LABEL: func.func @test_normalize_compare_neq_to_Not_eq
// CHECK-SAME: (%[[arg0:.*]]: tensor<1024xi64>, %[[arg1:.*]]: tensor<1024xi64>, %[[arg2:.*]]: tensor<1024xi1>)
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<1024xi1>
// CHECK: %[[veq:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[arg0]], %[[arg1]] : tensor<1024xi64>, tensor<1024xi64>) outs(%[[empty]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: %[[notOp:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[veq]] : tensor<1024xi1>) outs(%[[arg2]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: return %[[notOp]]
func.func @test_normalize_compare_neq_to_Not_eq(
  %src1 : tensor<1024xi64>, %src2 : tensor<1024xi64>,  %dst : tensor<1024xi1>) ->  tensor<1024xi1> {
  %ret = hfusion.compare {compare_fn  = #hfusion.compare_fn<vne>}
    ins(%src1, %src2 : tensor<1024xi64>, tensor<1024xi64>)
    outs(%dst : tensor<1024xi1>)
    -> tensor<1024xi1>
  return %ret : tensor<1024xi1>
}

// -----
// CHECK-LABEL: @cast_f32_to_i16_with_overflow_mode(
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xf32>) outs({{.*}} : tensor<16xi16>) -> tensor<16xi16>
func.func @cast_f32_to_i16_with_overflow_mode(%arg0: tensor<16xf32>) -> tensor<16xi16> {
  %0 = tensor.empty() : tensor<16xi16>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xf32>) outs(%0 : tensor<16xi16>) -> tensor<16xi16>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi16>
  return %1 : tensor<16xi16>
}

// -----
// CHECK-LABEL: @cast_f32_to_i8_with_overflow_mode(
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xf32>) outs({{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xf16>) outs({{.*}} : tensor<16xi8>) -> tensor<16xi8>
func.func @cast_f32_to_i8_with_overflow_mode(%arg0: tensor<16xf32>) -> tensor<16xi8> {
  %0 = tensor.empty() : tensor<16xi8>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xf32>) outs(%0 : tensor<16xi8>) -> tensor<16xi8>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi8>
  return %1 : tensor<16xi8>
}

// -----
// CHECK-LABEL: @cast_f16_to_i8_with_overflow_mode(
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xf16>) outs({{.*}} : tensor<16xi8>) -> tensor<16xi8>
func.func @cast_f16_to_i8_with_overflow_mode(%arg0: tensor<16xf16>) -> tensor<16xi8> {
  %0 = tensor.empty() : tensor<16xi8>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xf16>) outs(%0 : tensor<16xi8>) -> tensor<16xi8>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi8>
  return %1 : tensor<16xi8>
}

// -----
// CHECK-LABEL: @cast_i64_to_i32_with_overflow_mode(
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<16xi64>) outs({{.*}} : tensor<16xi32>) -> tensor<16xi32>
func.func @cast_i64_to_i32_with_overflow_mode(%arg0: tensor<16xi64>) -> tensor<16xi32> {
  %0 = tensor.empty() : tensor<16xi32>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xi64>) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi32>
  return %1 : tensor<16xi32>
}

// -----
// CHECK-LABEL: @cast_i64_to_i16_with_overflow_mode(
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xi64>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xf32>) outs({{.*}} : tensor<16xi16>) -> tensor<16xi16>
func.func @cast_i64_to_i16_with_overflow_mode(%arg0: tensor<16xi64>) -> tensor<16xi16> {
  %0 = tensor.empty() : tensor<16xi16>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xi64>) outs(%0 : tensor<16xi16>) -> tensor<16xi16>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi16>
  return %1 : tensor<16xi16>
}

// -----
// CHECK-LABEL: @cast_i64_to_i8_with_overflow_mode(
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xi64>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xf32>) outs({{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xf16>) outs({{.*}} : tensor<16xi8>) -> tensor<16xi8>
func.func @cast_i64_to_i8_with_overflow_mode(%arg0: tensor<16xi64>) -> tensor<16xi8> {
  %0 = tensor.empty() : tensor<16xi8>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xi64>) outs(%0 : tensor<16xi8>) -> tensor<16xi8>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi8>
  return %1 : tensor<16xi8>
}

// -----
// CHECK-LABEL: @cast_i32_to_i16_with_overflow_mode(
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<16xi32>) outs({{.*}} : tensor<16xi16>) -> tensor<16xi16>
func.func @cast_i32_to_i16_with_overflow_mode(%arg0: tensor<16xi32>) -> tensor<16xi16> {
  %0 = tensor.empty() : tensor<16xi16>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xi32>) outs(%0 : tensor<16xi16>) -> tensor<16xi16>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi16>
  return %1 : tensor<16xi16>
}

// -----
// CHECK-LABEL: @cast_i32_to_i8_with_overflow_mode(
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xi32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xf32>) outs({{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xf16>) outs({{.*}} : tensor<16xi8>) -> tensor<16xi8>
func.func @cast_i32_to_i8_with_overflow_mode(%arg0: tensor<16xi32>) -> tensor<16xi8> {
  %0 = tensor.empty() : tensor<16xi8>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xi32>) outs(%0 : tensor<16xi8>) -> tensor<16xi8>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi8>
  return %1 : tensor<16xi8>
}

// -----
// CHECK-LABEL: @cast_i16_to_i8_with_overflow_mode(
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xi16>) outs({{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xf16>) outs({{.*}} : tensor<16xi8>) -> tensor<16xi8>
func.func @cast_i16_to_i8_with_overflow_mode(%arg0: tensor<16xi16>) -> tensor<16xi8> {
  %0 = tensor.empty() : tensor<16xi8>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xi16>) outs(%0 : tensor<16xi8>) -> tensor<16xi8>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi8>
  return %1 : tensor<16xi8>
}

// CHECK-LABEL: func.func @test_hfusion_muli_i1
// CHECK-SAME: (%[[arg0:.*]]: tensor<32xi1>, %[[arg1:.*]]: tensor<32xi1>)
// CHECK: %[[ZERO:.*]] : tensor<32xi1>
// CHECK: %[[ONE:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[arg0:.*]], %[[arg1:.*]]: tensor<32xi1>, tensor<32xi1>) outs(%[[ZERO:.*]] : tensor<32xi1>) -> tensor<32xi1>
// CHECK: return %[[ONE:.*]]
func.func @test_hfusion_muli_i1(%src1: tensor<32xi1>, %src2: tensor<32xi1>) -> tensor<32xi1> {
  %x = tensor.empty() : tensor<32xi1>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%src1, %src2 : tensor<32xi1>, tensor<32xi1>) outs(%x : tensor<32xi1>) -> tensor<32xi1>
  return %1 : tensor<32xi1>
}

// CHECK-LABEL: func.func @test_hfusion_select_cast_i1_to_i16
// CHECK-SAME: %[[arg0:.*]]: tensor<32xi1>
// CHECK: %[[EMPTYI1:.*]] : tensor<32xi1>
// CHECK: %[[EMPTY1:.*]] : tensor<32xi16>
// CHECK: %[[CASTRES1:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[arg1:.*]] : tensor<32xi1>) outs(%[[EMPTY1:.*]] : tensor<32xi16>) -> tensor<32xi16>
// CHECK: %[[EMPTY2:.*]] : tensor<32xi16>
// CHECK: %[[CASTRES2:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[arg2:.*]] : tensor<32xi1>) outs(%[[EMPTY2:.*]] : tensor<32xi16>) -> tensor<32xi16>
// CHECK: %[[EMPTYI16:.*]] : tensor<32xi16>
// CHECK: %[[CASTRES0:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[EMPTYI1:.*]] : tensor<32xi1>) outs(%[[EMPTYI16:.*]] : tensor<32xi16>) -> tensor<32xi16>
// CHECK: %[[ONE:.*]] = hfusion.select ins(%[[arg0:.*]], %[[CASTRES1:.*]], %[[CASTRES2:.*]] : tensor<32xi1>, tensor<32xi16>, tensor<32xi16>) outs(%[[EMPTYI16:.*]] : tensor<32xi16>) -> tensor<32xi16>
func.func @test_hfusion_select_cast_i1_to_i16(%src0: tensor<32xi1>, %src1: tensor<32xi1>, %src2: tensor<32xi1>) -> tensor<32xi1>{
  %x = tensor.empty() : tensor<32xi1>
  %1 = hfusion.select ins(%src0, %src1, %src2 : tensor<32xi1>, tensor<32xi1>, tensor<32xi1>) outs(%x : tensor<32xi1>) -> tensor<32xi1>
  return %1 : tensor<32xi1>
}

// -----
// CHECK-LABEL: func.func @test_minnumf_normalize
// CHECK-DAG: %[[C0:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%arg0, %arg0 : tensor<4xf32>, tensor<4xf32>) outs(%{{.*}} : tensor<4xi1>) -> tensor<4xi1>
// CHECK-DAG: %[[M0:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[C0]] : tensor<4xi1>) outs(%{{.*}} : tensor<4xi1>) -> tensor<4xi1>
// CHECK-DAG: %[[C1:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%arg1, %arg1 : tensor<4xf32>, tensor<4xf32>) outs(%{{.*}} : tensor<4xi1>) -> tensor<4xi1>
// CHECK-DAG: %[[M1:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[C1]] : tensor<4xi1>) outs(%{{.*}} : tensor<4xi1>) -> tensor<4xi1>
// CHECK: %[[EBASE:.*]] = tensor.empty() : tensor<4xf32>
// CHECK: %[[BASE:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>) outs(%[[EBASE]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[ETMP:.*]] = tensor.empty() : tensor<4xf32>
// CHECK: %[[TMP:.*]] = hfusion.select ins(%[[M0]], %arg1, %[[BASE]] : tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) outs(%[[ETMP]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[ERES:.*]] = tensor.empty() : tensor<4xf32>
// CHECK: %[[RES:.*]] = hfusion.select ins(%[[M1]], %arg0, %[[TMP]] : tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) outs(%[[ERES]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK: return %[[RES]] : tensor<4xf32>
func.func @test_minnumf_normalize(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %out = tensor.empty() : tensor<4xf32>
  %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minnumf>}
       ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>)
       outs(%out : tensor<4xf32>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_maxnumf_normalize
// CHECK-DAG: %[[C0:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%arg0, %arg0 : tensor<4xf32>, tensor<4xf32>) outs(%{{.*}} : tensor<4xi1>) -> tensor<4xi1>
// CHECK-DAG: %[[M0:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[C0]] : tensor<4xi1>) outs(%{{.*}} : tensor<4xi1>) -> tensor<4xi1>
// CHECK-DAG: %[[C1:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%arg1, %arg1 : tensor<4xf32>, tensor<4xf32>) outs(%{{.*}} : tensor<4xi1>) -> tensor<4xi1>
// CHECK-DAG: %[[M1:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[C1]] : tensor<4xi1>) outs(%{{.*}} : tensor<4xi1>) -> tensor<4xi1>
// CHECK: %[[EBASE:.*]] = tensor.empty() : tensor<4xf32>
// CHECK: %[[BASE:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>) outs(%[[EBASE]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[ETMP:.*]] = tensor.empty() : tensor<4xf32>
// CHECK: %[[TMP:.*]] = hfusion.select ins(%[[M0]], %arg1, %[[BASE]] : tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) outs(%[[ETMP]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[ERES:.*]] = tensor.empty() : tensor<4xf32>
// CHECK: %[[RES:.*]] = hfusion.select ins(%[[M1]], %arg0, %[[TMP]] : tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) outs(%[[ERES]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK: return %[[RES]] : tensor<4xf32>
func.func @test_maxnumf_normalize(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %out = tensor.empty() : tensor<4xf32>
  %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxnumf>}
       ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>)
       outs(%out : tensor<4xf32>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_unsigned_cast_for_uint8_triton_maximum(
// CHECK: %[[VAL_13:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_8:.*]] : tensor<1024xi8>) outs(%[[VAL_12:.*]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: %[[VAL_15:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_10:.*]] : tensor<1024xi8>) outs(%[[VAL_14:.*]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: %[[VAL_17:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_11:.*]] : tensor<1024xi8>) outs(%[[VAL_16:.*]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: %[[VAL_18:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%[[VAL_13]], %[[VAL_15]] : tensor<1024xf16>, tensor<1024xf16>) outs(%[[VAL_17]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: %[[VAL_20:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins(%[[VAL_18]] : tensor<1024xf16>) outs(%[[VAL_19:.*]] : tensor<1024xi8>) -> tensor<1024xi8>
// CHECK: %[[VAL_21:.*]] = memref.reinterpret_cast %[[VAL_0:.*]] to offset: {{\[}}%[[VAL_4:.*]]], sizes: [1024], strides: [1] : memref<?xi8> to memref<1024xi8, strided<[1], offset: ?>>
// CHECK: %[[VAL_22:.*]] = tensor.extract_slice %[[VAL_20]][0] {{\[}}%[[VAL_6:.*]]] [1] : tensor<1024xi8> to tensor<?xi8>
// CHECK: %[[VAL_23:.*]] = memref.subview %[[VAL_21]][0] {{\[}}%[[VAL_6]]] [1] : memref<1024xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
// CHECK: bufferization.materialize_in_destination %[[VAL_22]] in writable %[[VAL_23]] : (tensor<?xi8>, memref<?xi8, strided<[1], offset: ?>>) -> ()
func.func @test_unsigned_cast_for_uint8_triton_maximum(%arg4: memref<?xi8>, %arg9: i32) {
  %arg12 = arith.constant 0 : i32
  %c32768_i32 = arith.constant 32768 : i32
  %0 = arith.muli %arg9, %c32768_i32 : i32
  %1 = arith.addi %0, %arg12 : i32
  %2 = arith.index_cast %1 : i32 to index
  %6 = arith.index_cast %1 : i32 to index
  %7 = arith.subi %6, %2 : index
  %alloc = memref.alloc() : memref<1024xi8>
  %9 = bufferization.to_tensor %alloc restrict writable : memref<1024xi8>
  %alloc_2 = memref.alloc() : memref<1024xi8>
  %10 = bufferization.to_tensor %alloc_2 restrict writable : memref<1024xi8>    %11 = tensor.empty() : tensor<1024xi8>
  %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_unsigned>} ins(%9, %10 : tensor<1024xi8>, tensor<1024xi8>) outs(%11 : tensor<1024xi8>) -> tensor<1024xi8>
  %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%2], sizes: [1024], strides: [1] : memref<?xi8> to memref<1024xi8, strided<[1], offset: ?>>
  %extracted_slice = tensor.extract_slice %12[0] [%7] [1] : tensor<1024xi8> to tensor<?xi8>
  %subview_6 = memref.subview %reinterpret_cast_5[0] [%7] [1] : memref<1024xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
  bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<?xi8>, memref<?xi8, strided<[1], offset: ?>>) -> ()
  return
}

// -----
// CHECK-LABEL:   func.func @test_gt_uint8
// CHECK:           %[[VAL_11:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_12:.*]] = arith.constant 32 : i32
// CHECK:           %[[VAL_13:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_14:.*]] = arith.constant 1024 : i32
// CHECK:           %[[VAL_15:.*]] = arith.constant 32768 : i32
// CHECK:           %[[VAL_16:.*]] = arith.muli %[[VAL_8:.*]], %[[VAL_15]] : i32
// CHECK:           scf.for %[[VAL_17:.*]] = %[[VAL_13]] to %[[VAL_12]] step %[[VAL_11]]  : i32 {
// CHECK:             %[[VAL_18:.*]] = arith.muli %[[VAL_17]], %[[VAL_14]] : i32
// CHECK:             %[[VAL_19:.*]] = arith.addi %[[VAL_16]], %[[VAL_18]] : i32
// CHECK:             %[[VAL_20:.*]] = arith.index_cast %[[VAL_19]] : i32 to index
// CHECK:             %[[VAL_21:.*]] = memref.reinterpret_cast %[[VAL_2:.*]] to offset: {{\[}}%[[VAL_20]]], sizes: [1024], strides: [1] : memref<?xi8> to memref<1024xi8, strided<[1], offset: ?>>
// CHECK:             %[[VAL_22:.*]] = memref.alloc() : memref<1024xi8>
// CHECK:             memref.copy %[[VAL_21]], %[[VAL_22]] : memref<1024xi8, strided<[1], offset: ?>> to memref<1024xi8>
// CHECK:             %[[VAL_23:.*]] = bufferization.to_tensor %[[VAL_22]] restrict writable : memref<1024xi8>
// CHECK:             %[[VAL_24:.*]] = memref.reinterpret_cast %[[VAL_3:.*]] to offset: {{\[}}%[[VAL_20]]], sizes: [1024], strides: [1] : memref<?xi8> to memref<1024xi8, strided<[1], offset: ?>>
// CHECK:             %[[VAL_25:.*]] = memref.alloc() : memref<1024xi8>
// CHECK:             memref.copy %[[VAL_24]], %[[VAL_25]] : memref<1024xi8, strided<[1], offset: ?>> to memref<1024xi8>
// CHECK:             %[[VAL_26:.*]] = bufferization.to_tensor %[[VAL_25]] restrict writable : memref<1024xi8>
// CHECK:             %[[VAL_27:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK:             %[[VAL_28:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_23]] : tensor<1024xi8>) outs(%[[VAL_27]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK:             %[[VAL_29:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK:             %[[VAL_30:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_26]] : tensor<1024xi8>) outs(%[[VAL_29]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK:             %[[VAL_31:.*]] = tensor.empty() : tensor<1024xi1>
// CHECK:             %[[VAL_32:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vgt>} ins(%[[VAL_28]], %[[VAL_30]] : tensor<1024xf16>, tensor<1024xf16>) outs(%[[VAL_31]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK:             %[[VAL_33:.*]] = memref.reinterpret_cast %[[VAL_4:.*]] to offset: {{\[}}%[[VAL_20]]], sizes: [1024], strides: [1] : memref<?xi8> to memref<1024xi8, strided<[1], offset: ?>>
// CHECK:             %[[VAL_34:.*]] = tensor.empty() : tensor<1024xi8>
// CHECK:             %[[VAL_35:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_32]] : tensor<1024xi1>) outs(%[[VAL_34]] : tensor<1024xi8>) -> tensor<1024xi8>
// CHECK:             bufferization.materialize_in_destination %[[VAL_35]] in writable %[[VAL_33]] : (tensor<1024xi8>, memref<1024xi8, strided<[1], offset: ?>>) -> ()
// CHECK:           }
func.func @test_gt_uint8(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xi8>, %arg4: memref<?xi8>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32){
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %c32768_i32 = arith.constant 32768 : i32
  %0 = arith.muli %arg8, %c32768_i32 : i32
  scf.for %arg11 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
    %1 = arith.muli %arg11, %c1024_i32 : i32
    %2 = arith.addi %0, %1 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%3], sizes: [1024], strides: [1] : memref<?xi8> to memref<1024xi8, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<1024xi8>
    memref.copy %reinterpret_cast, %alloc : memref<1024xi8, strided<[1], offset: ?>> to memref<1024xi8>
    %4 = bufferization.to_tensor %alloc restrict writable : memref<1024xi8>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%3], sizes: [1024], strides: [1] : memref<?xi8> to memref<1024xi8, strided<[1], offset: ?>>
    %alloc_1 = memref.alloc() : memref<1024xi8>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<1024xi8, strided<[1], offset: ?>> to memref<1024xi8>
    %5 = bufferization.to_tensor %alloc_1 restrict writable : memref<1024xi8>
    %6 = tensor.empty() : tensor<1024xi1>
    %7 = hfusion.compare {compare_fn = #hfusion.compare_fn<vugt>} ins(%4, %5 : tensor<1024xi8>, tensor<1024xi8>) outs(%6 : tensor<1024xi1>) -> tensor<1024xi1>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%3], sizes: [1024], strides: [1] : memref<?xi8> to memref<1024xi8, strided<[1], offset: ?>>
    %8 = tensor.empty() : tensor<1024xi8>
    %9 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%7 : tensor<1024xi1>) outs(%8 : tensor<1024xi8>) -> tensor<1024xi8>
    bufferization.materialize_in_destination %9 in writable %reinterpret_cast_2 : (tensor<1024xi8>, memref<1024xi8, strided<[1], offset: ?>>) -> ()
  }
  return
}

// -----
// CHECK-LABEL: func.func @test_elementwise_shrui
func.func @test_elementwise_shrui(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi8> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<32xf16>
  // CHECK: %[[VAL_1:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<32xi8>) outs(%[[VAL_0:.*]] : tensor<32xf16>) -> tensor<32xf16>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<32xi16>
  // CHECK: %[[VAL_3:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[VAL_1:.*]] : tensor<32xf16>) outs(%[[VAL_2:.*]] : tensor<32xi16>) -> tensor<32xi16>
  // CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<32xf16>
  // CHECK: %[[VAL_5:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<32xi8>) outs(%[[VAL_4:.*]] : tensor<32xf16>) -> tensor<32xf16>
  // CHECK: %[[VAL_6:.*]] = tensor.empty() : tensor<32xi16>
  // CHECK: %[[VAL_7:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[VAL_5:.*]] : tensor<32xf16>) outs(%[[VAL_6:.*]] : tensor<32xi16>) -> tensor<32xi16>
  // CHECK: %[[VAL_8:.*]] = tensor.empty() : tensor<32xi16>
  // CHECK: %[[VAL_9:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%[[VAL_3:.*]], %[[VAL_7:.*]] : tensor<32xi16>, tensor<32xi16>) outs(%[[VAL_8:.*]] : tensor<32xi16>) -> tensor<32xi16>
  // CHECK: %[[VAL_10:.*]] = tensor.empty() : tensor<32xi8>
  // CHECK: %[[VAL_11:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} ins(%[[VAL_9:.*]] : tensor<32xi16>) outs(%[[VAL_10:.*]] : tensor<32xi8>) -> tensor<32xi8>
  %0 = tensor.empty() : tensor<32xi8>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%arg0, %arg1 : tensor<32xi8>, tensor<32xi8>) outs(%0 : tensor<32xi8>) -> tensor<32xi8>
  return %1 : tensor<32xi8>
}

// -----
// CHECK-LABEL: func.func @test_elementwise_modui
func.func @test_elementwise_modui(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi8> {
  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<32xf16>
  // CHECK: %[[VAL_1:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<32xi8>) outs(%[[VAL_0:.*]] : tensor<32xf16>) -> tensor<32xf16>
  // CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<32xf16>
  // CHECK: %[[VAL_3:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<32xi8>) outs(%[[VAL_2:.*]] : tensor<32xf16>) -> tensor<32xf16>
  // CHECK: %[[RET_0:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
  // CHECK: %[[RET_1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
  // CHECK: %[[RET_2:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
  %0 = tensor.empty() : tensor<32xi8>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>} ins(%arg0, %arg1 : tensor<32xi8>, tensor<32xi8>) outs(%0 : tensor<32xi8>) -> tensor<32xi8>
  return %1 : tensor<32xi8>
}

// -----

// CHECK: tensor.empty() : tensor<6x6xf16>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>}
// CHECK: tensor.empty() : tensor<6x6xf32>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>}
// CHECK: tensor.empty() : tensor<6x6xf16>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>}
// CHECK: tensor.empty() : tensor<6x6xf32>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>}
// CHECK: tensor.empty() : tensor<6x6xf32>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: tensor.empty() : tensor<6x6xi32>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<ceil>}
// CHECK: tensor.empty() : tensor<6x6xf16>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>}
// CHECK: tensor.empty() : tensor<6x6xi8>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = false, round_mode = #hfusion.round_mode<trunc>}
func.func @test_ceildivui_ui8(%arg0: tensor<6x6xi8>, %arg1: tensor<6x6xi8>) -> tensor<6x6xi8> {
  %0 = tensor.empty() : tensor<6x6xi8>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivui>} ins(%arg0, %arg1 : tensor<6x6xi8>, tensor<6x6xi8>) outs(%0 : tensor<6x6xi8>) -> tensor<6x6xi8>
  return %1 : tensor<6x6xi8>
}

// -----
// CHECK-LABEL:   func.func @test_normalize_reduce_ui8_to_f16(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<16x2x2x2xi8>) -> tensor<16x2x2xi8> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0.000000e+00 : f16
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<16x2x2x2xf16>
// CHECK:           %[[VAL_2:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[ARG0]] : tensor<16x2x2x2xi8>) outs(%[[VAL_1]] : tensor<16x2x2x2xf16>) -> tensor<16x2x2x2xf16>
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<16x2x2xf16>
// CHECK:           %[[VAL_4:.*]] = linalg.fill ins(%[[VAL_0]] : f16) outs(%[[VAL_3]] : tensor<16x2x2xf16>) -> tensor<16x2x2xf16>
// CHECK:           %[[VAL_5:.*]] = linalg.reduce ins(%[[VAL_2]] : tensor<16x2x2x2xf16>) outs(%[[VAL_4]] : tensor<16x2x2xf16>) dimensions = [3]
// CHECK:             (%[[VAL_6:.*]]: f16, %[[VAL_7:.*]]: f16) {
// CHECK:               %[[VAL_8:.*]] = arith.maximumf %[[VAL_6]], %[[VAL_7]] : f16
// CHECK:               linalg.yield %[[VAL_8]] : f16
// CHECK:             }
// CHECK:           %[[VAL_9:.*]] = tensor.empty() : tensor<16x2x2xi8>
// CHECK:           %[[VAL_10:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins(%[[VAL_5]] : tensor<16x2x2xf16>) outs(%[[VAL_9]] : tensor<16x2x2xi8>) -> tensor<16x2x2xi8>
// CHECK:           return %[[VAL_10]] : tensor<16x2x2xi8>
// CHECK:         }
func.func @test_normalize_reduce_ui8_to_f16(%arg0: tensor<16x2x2x2xi8>) -> tensor<16x2x2xi8> {
  %c0_i8 = arith.constant 0 : i8
  %init = tensor.empty() : tensor<16x2x2xi8>
  %filled = linalg.fill ins(%c0_i8 : i8) outs(%init : tensor<16x2x2xi8>) -> tensor<16x2x2xi8>
  %reduced = linalg.reduce ins(%arg0 : tensor<16x2x2x2xi8>) outs(%filled : tensor<16x2x2xi8>) dimensions = [3]
    (%in: i8, %init_val: i8) {
      %max = arith.maxui %in, %init_val : i8
      linalg.yield %max : i8
    }
  return %reduced : tensor<16x2x2xi8>
}

// -----
// CHECK-LABEL: func.func @triton_not_8d
func.func @triton_not_8d(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi64>, %arg3: memref<?xi64>) {
  %c-1_i64 = arith.constant -1 : i64
  %0 = tensor.empty() : tensor<2x2x2x2x2x2x2x2xi64>
  %1 = linalg.fill ins(%c-1_i64 : i64) outs(%0 : tensor<2x2x2x2x2x2x2x2xi64>) -> tensor<2x2x2x2x2x2x2x2xi64>
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [2, 2, 2, 2, 2, 2, 2, 2], strides: [128, 64, 32, 16, 8, 4, 2, 1] : memref<?xi64> to memref<2x2x2x2x2x2x2x2xi64, strided<[128, 64, 32, 16, 8, 4, 2, 1]>>
  %alloc = memref.alloc() : memref<2x2x2x2x2x2x2x2xi64>
  memref.copy %reinterpret_cast, %alloc : memref<2x2x2x2x2x2x2x2xi64, strided<[128, 64, 32, 16, 8, 4, 2, 1]>> to memref<2x2x2x2x2x2x2x2xi64>
  %2 = bufferization.to_tensor %alloc restrict writable : memref<2x2x2x2x2x2x2x2xi64>
  // CHECK:  %[[VAL_2:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
  // CHECK-NOT: hfusion.binary_fn<vxor>
  // CHECK-NOT: hfusion.binary_fn<vor>
  // CHECK-NOT: hfusion.binary_fn<vand>
  %3 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vxor>} ins(%2, %1 : tensor<2x2x2x2x2x2x2x2xi64>, tensor<2x2x2x2x2x2x2x2xi64>) outs(%0 : tensor<2x2x2x2x2x2x2x2xi64>) -> tensor<2x2x2x2x2x2x2x2xi64>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [2, 2, 2, 2, 2, 2, 2, 2], strides: [128, 64, 32, 16, 8, 4, 2, 1] : memref<?xi64> to memref<2x2x2x2x2x2x2x2xi64, strided<[128, 64, 32, 16, 8, 4, 2, 1]>>
  bufferization.materialize_in_destination %3 in writable %reinterpret_cast_0 : (tensor<2x2x2x2x2x2x2x2xi64>,  memref<2x2x2x2x2x2x2x2xi64, strided<[128, 64, 32, 16, 8, 4, 2, 1]>>) -> ()
  return
}

// -----
// CHECK-LABEL: func.func @test_gathermask
// CHECK: arith.constant -1.280000e+02 : f16
// CHECK: arith.constant 2.560000e+02 : f32
// CHECK: arith.constant 2.560000e+02 : f16
// CHECK: arith.constant 1.280000e+02 : f16
// CHECK: arith.constant 0 : i32
// CHECK: arith.constant 0.000000e+00 : f16
// CHECK: memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [16], strides: [1] : memref<?xi8> to memref<16xi8, strided<[1]>>
// CHECK: memref.alloc() : memref<16xi8>
// CHECK: memref.copy
// CHECK: bufferization.to_tensor
// CHECK: memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [16], strides: [1] : memref<?xi8> to memref<16xi8, strided<[1]>>
// CHECK: memref.alloc() : memref<16xi8>
// CHECK: memref.copy
// CHECK: bufferization.to_tensor
// CHECK: tensor.empty() : tensor<1xi32>
// CHECK: tensor.empty() : tensor<16xi8>
// CHECK: tensor.empty() : tensor<16xf16>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%{{.*}} : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: tensor.empty() : tensor<16xf16>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%{{.*}} : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: tensor.empty() : tensor<16xf16>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%{{.*}} : tensor<16xi8>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: tensor.empty() : tensor<16xi1>
// CHECK: tensor.empty() : tensor<16xi1>
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%{{.*}}, %{{.*}} : tensor<16xf16>, f16) outs(%{{.*}} : tensor<16xi1>) -> tensor<16xi1>
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%{{.*}} : tensor<16xi1>) outs(%{{.*}} : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %{{.*}}:2 = hfusion.gather_mask {operandSegmentSizes = array<i32: 2, 2>} ins(%{{.*}}, %{{.*}} : tensor<16xf16>, tensor<16xi1>) outs(%{{.*}}, %{{.*}} : tensor<16xf16>, tensor<1xi32>) -> (tensor<16xf16>, tensor<1xi32>)
// CHECK: tensor.empty() : tensor<16xf16>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%{{.*}}#0, %{{.*}} : tensor<16xf16>, f16) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: tensor.empty() : tensor<16xf32>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%{{.*}} : tensor<16xf16>) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: tensor.empty() : tensor<16xf32>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%{{.*}}, %{{.*}} : tensor<16xf32>, f32) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: tensor.empty() : tensor<16xf32>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%{{.*}} : tensor<16xf32>) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: tensor.empty() : tensor<16xf16>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<rint>} ins(%{{.*}} : tensor<16xf32>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: tensor.empty() : tensor<16xf16>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%{{.*}}, %{{.*}} : f16, tensor<16xf16>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: tensor.empty() : tensor<16xf16>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%{{.*}}, %{{.*}} : tensor<16xf16>, tensor<16xf16>) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%{{.*}}, %{{.*}} : tensor<16xf16>, f16) outs(%{{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: tensor.empty() : tensor<16xi8>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins(%{{.*}} : tensor<16xf16>) outs(%{{.*}} : tensor<16xi8>) -> tensor<16xi8>
// CHECK: memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [16], strides: [1] : memref<?xi8> to memref<16xi8, strided<[1]>>
// CHECK: bufferization.materialize_in_destination %{{.*}} in writable %{{.*}} : (tensor<16xi8>, memref<16xi8, strided<[1]>>) -> ()
// CHECK: memref.alloc() : memref<1xi32>
// CHECK: bufferization.materialize_in_destination %{{.*}}#1 in writable %{{.*}} : (tensor<1xi32>, memref<1xi32>) -> ()
// CHECK: return
func.func @test_gathermask(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xi8>, %arg4: memref<?xi8>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16], strides: [1] : memref<?xi8> to memref<16xi8, strided<[1]>>
  %alloc = memref.alloc() : memref<16xi8>
  memref.copy %reinterpret_cast, %alloc : memref<16xi8, strided<[1]>> to memref<16xi8>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<16xi8>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [16], strides: [1] : memref<?xi8> to memref<16xi8, strided<[1]>>
  %alloc_1 = memref.alloc() : memref<16xi8>
  memref.copy %reinterpret_cast_0, %alloc_1 : memref<16xi8, strided<[1]>> to memref<16xi8>
  %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<16xi8>
  %4 = tensor.empty() : tensor<1xi32>
  %2 = tensor.empty() : tensor<16xi8>
  %3:2 = hfusion.gather_mask ins(%0, %1 : tensor<16xi8>, tensor<16xi8>) outs(%2, %4 : tensor<16xi8>, tensor<1xi32>) -> (tensor<16xi8>, tensor<1xi32>)
  %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [16], strides: [1] : memref<?xi8> to memref<16xi8, strided<[1]>>
  bufferization.materialize_in_destination %3#0 in writable %reinterpret_cast_2 : (tensor<16xi8>, memref<16xi8, strided<[1]>>) -> ()
  %alloc_size = memref.alloc() : memref<1xi32>
  bufferization.materialize_in_destination %3#1 in writable %alloc_size : (tensor<1xi32>, memref<1xi32>) -> ()
  return
}

// -----
// CHECK-LABEL: func.func @test_hfusion_signbit_f32
// CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[CST_MASK:.*]] = arith.constant -2147483648 : i32
// CHECK: %[[BITCAST:.*]] = hfusion.bitcast ins(%{{.*}} : tensor<4xf32>)
// CHECK: %[[AND:.*]] = hfusion.elemwise_binary {{.*}} ins(%[[BITCAST]], %[[CST_MASK]] : tensor<4xi32>, i32)
// CHECK: %[[CMP:.*]] = hfusion.compare {{.*}} ins(%[[AND]], %[[CST0]] : tensor<4xi32>, i32)
// CHECK: %[[NOT:.*]] = hfusion.elemwise_unary {{.*}} ins(%[[CMP]] : tensor<4xi1>)
// CHECK: return %[[NOT]] : tensor<4xi1>
func.func @test_hfusion_signbit_f32(%arg0 : tensor<4xf32>) -> tensor<4xi1> {
  %ret = "hfusion.signbit"(%arg0) : (tensor<4xf32>) -> tensor<4xi1>
  return %ret : tensor<4xi1>
}

// -----
// CHECK-LABEL: func.func @test_hfusion_signbit_f16
// CHECK-DAG: %[[CST0_H:.*]] = arith.constant 0 : i16
// CHECK-DAG: %[[CST_MASK_H:.*]] = arith.constant -32768 : i16
// CHECK: %[[BITCAST_H:.*]] = hfusion.bitcast ins(%{{.*}} : tensor<4xf16>)
// CHECK: %[[AND_H:.*]] = hfusion.elemwise_binary {{.*}} ins(%[[BITCAST_H]], %[[CST_MASK_H]] : tensor<4xi16>, i16)
// CHECK: %[[CMP_H:.*]] = hfusion.compare {{.*}} ins(%[[AND_H]], %[[CST0_H]] : tensor<4xi16>, i16)
// CHECK: %[[NOT_H:.*]] = hfusion.elemwise_unary {{.*}} ins(%[[CMP_H]] : tensor<4xi1>)
// CHECK: return %[[NOT_H]] : tensor<4xi1>
func.func @test_hfusion_signbit_f16(%arg0 : tensor<4xf16>) -> tensor<4xi1> {
  %ret = "hfusion.signbit"(%arg0) : (tensor<4xf16>) -> tensor<4xi1>
  return %ret : tensor<4xi1>
}

// CHECK-LABEL: func.func @test_compare_i32_lt
// CHECK-SAME: ([[ARG0:%.*]]: tensor<16x32xi32>, [[ARG1:%.*]]: tensor<16x32xi32>, [[DST:%.*]]: tensor<16x32xi1>)
func.func @test_compare_i32_lt(%arg0: tensor<16x32xi32>, %arg1: tensor<16x32xi32>,  %dst : tensor<16x32xi1>) -> (tensor<16x32xi1>) {
  %ret = hfusion.compare {compare_fn  = #hfusion.compare_fn<vlt>} ins(%arg0, %arg1 : tensor<16x32xi32>, tensor<16x32xi32>) outs(%dst : tensor<16x32xi1>) -> tensor<16x32xi1>
  return %ret : tensor<16x32xi1>
// CHECK-DAG: [[EMPTY_I32:%.*]] = tensor.empty() : tensor<16x32xi32>
// CHECK:     [[MAX_VAL:%.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins([[ARG0]], [[ARG1]] : tensor<16x32xi32>, tensor<16x32xi32>) outs([[EMPTY_I32]] : tensor<16x32xi32>) -> tensor<16x32xi32>
// CHECK-DAG: [[EMPTY_I1_OUT:%.*]] = tensor.empty() : tensor<16x32xi1>
// CHECK-DAG: [[EMPTY_I1_MID:%.*]] = tensor.empty() : tensor<16x32xi1>
// CHECK:     [[CMP_EQ:%.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins([[MAX_VAL]], [[ARG0]] : tensor<16x32xi32>, tensor<16x32xi32>) outs([[EMPTY_I1_MID]] : tensor<16x32xi1>) -> tensor<16x32xi1>
// CHECK:     [[FINAL_RES:%.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins([[CMP_EQ]] : tensor<16x32xi1>) outs([[EMPTY_I1_OUT]] : tensor<16x32xi1>) -> tensor<16x32xi1>
// CHECK:     return [[FINAL_RES]] : tensor<16x32xi1>
}

// -----
// CHECK-LABEL: func.func @test_compare_i32_le
// CHECK-SAME: ([[ARG0:%.*]]: tensor<16x32xi32>, [[ARG1:%.*]]: tensor<16x32xi32>, [[DST:%.*]]: tensor<16x32xi1>)
func.func @test_compare_i32_le(%arg0: tensor<16x32xi32>, %arg1: tensor<16x32xi32>,  %dst : tensor<16x32xi1>) -> (tensor<16x32xi1>) {
  %ret = hfusion.compare {compare_fn  = #hfusion.compare_fn<vle>} ins(%arg0, %arg1 : tensor<16x32xi32>, tensor<16x32xi32>) outs(%dst : tensor<16x32xi1>) -> tensor<16x32xi1>
  return %ret : tensor<16x32xi1>
// CHECK-DAG: [[EMPTY_I32:%.*]] = tensor.empty() : tensor<16x32xi32>
// CHECK:     [[MAX_VAL:%.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins([[ARG0]], [[ARG1]] : tensor<16x32xi32>, tensor<16x32xi32>) outs([[EMPTY_I32]] : tensor<16x32xi32>) -> tensor<16x32xi32>
// CHECK-DAG: [[EMPTY_I1_MID:%.*]] = tensor.empty() : tensor<16x32xi1>
// CHECK:     [[FINAL_RES:%.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins([[MAX_VAL]], [[ARG1]] : tensor<16x32xi32>, tensor<16x32xi32>) outs([[EMPTY_I1_MID]] : tensor<16x32xi1>) -> tensor<16x32xi1>
// CHECK:     return [[FINAL_RES]] : tensor<16x32xi1>
}

// -----
// CHECK-LABEL: func.func @test_compare_i32_gt
// CHECK-SAME: ([[ARG0:%.*]]: tensor<16x32xi32>, [[ARG1:%.*]]: tensor<16x32xi32>, [[DST:%.*]]: tensor<16x32xi1>)
func.func @test_compare_i32_gt(%arg0: tensor<16x32xi32>, %arg1: tensor<16x32xi32>,  %dst : tensor<16x32xi1>) -> (tensor<16x32xi1>) {
  %ret = hfusion.compare {compare_fn  = #hfusion.compare_fn<vgt>} ins(%arg0, %arg1 : tensor<16x32xi32>, tensor<16x32xi32>) outs(%dst : tensor<16x32xi1>) -> tensor<16x32xi1>
  return %ret : tensor<16x32xi1>
// CHECK-DAG: [[EMPTY_I32:%.*]] = tensor.empty() : tensor<16x32xi32>
// CHECK:     [[MAX_VAL:%.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins([[ARG0]], [[ARG1]] : tensor<16x32xi32>, tensor<16x32xi32>) outs([[EMPTY_I32]] : tensor<16x32xi32>) -> tensor<16x32xi32>
// CHECK-DAG: [[EMPTY_I1_OUT:%.*]] = tensor.empty() : tensor<16x32xi1>
// CHECK-DAG: [[EMPTY_I1_MID:%.*]] = tensor.empty() : tensor<16x32xi1>
// CHECK:     [[CMP_EQ:%.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins([[MAX_VAL]], [[ARG1]] : tensor<16x32xi32>, tensor<16x32xi32>) outs([[EMPTY_I1_MID]] : tensor<16x32xi1>) -> tensor<16x32xi1>
// CHECK:     [[FINAL_RES:%.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins([[CMP_EQ]] : tensor<16x32xi1>) outs([[EMPTY_I1_OUT]] : tensor<16x32xi1>) -> tensor<16x32xi1>
// CHECK:     return [[FINAL_RES]] : tensor<16x32xi1>
}

// -----
// CHECK-LABEL: func.func @test_compare_i32_ge
// CHECK-SAME: ([[ARG0:%.*]]: tensor<16x32xi32>, [[ARG1:%.*]]: tensor<16x32xi32>, [[DST:%.*]]: tensor<16x32xi1>)
func.func @test_compare_i32_ge(%arg0: tensor<16x32xi32>, %arg1: tensor<16x32xi32>,  %dst : tensor<16x32xi1>) -> (tensor<16x32xi1>) {
  %ret = hfusion.compare {compare_fn  = #hfusion.compare_fn<vge>} ins(%arg0, %arg1 : tensor<16x32xi32>, tensor<16x32xi32>) outs(%dst : tensor<16x32xi1>) -> tensor<16x32xi1>
  return %ret : tensor<16x32xi1>
// CHECK-DAG: [[EMPTY_I32:%.*]] = tensor.empty() : tensor<16x32xi32>
// CHECK:     [[MAX_VAL:%.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins([[ARG0]], [[ARG1]] : tensor<16x32xi32>, tensor<16x32xi32>) outs([[EMPTY_I32]] : tensor<16x32xi32>) -> tensor<16x32xi32>
// CHECK-DAG: [[EMPTY_I1_MID:%.*]] = tensor.empty() : tensor<16x32xi1>
// CHECK:     [[FINAL_RES:%.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins([[MAX_VAL]], [[ARG0]] : tensor<16x32xi32>, tensor<16x32xi32>) outs([[EMPTY_I1_MID]] : tensor<16x32xi1>) -> tensor<16x32xi1>
// CHECK:     return [[FINAL_RES]] : tensor<16x32xi1>
}

// -----
// CHECK-LABEL: func.func @triton_uint8_mod
// CHECK: %[[VAL:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = false, round_mode = #hfusion.round_mode<trunc>}
// CHECK-NEXT: bufferization.materialize_in_destination %[[VAL]]
func.func @triton_uint8_mod(%arg0: tensor<1x64x64xi8>, %arg1: tensor<1x64x64xi8>, %arg2: memref<1x64x64xi8, strided<[4096, 64, 1]>>) {
  %0 = tensor.empty() : tensor<1x64x64xi8>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>} ins(%arg0, %arg1 : tensor<1x64x64xi8>, tensor<1x64x64xi8>) outs(%0 : tensor<1x64x64xi8>) -> tensor<1x64x64xi8>
  bufferization.materialize_in_destination %1 in writable %arg2 : (tensor<1x64x64xi8>, memref<1x64x64xi8, strided<[4096, 64, 1]>>) -> ()
  return
}

// -----
// CHECK-LABEL: func.func @test_insert_slice_i1
// CHECK: %[[INSERTED:.*]] = tensor.insert_slice %arg0 into %arg1[%arg2] [256] [1] : tensor<256xi1> into tensor<1024xi1>
// CHECK: return %[[INSERTED]] : tensor<1024xi1>
func.func @test_insert_slice_i1(%arg0: tensor<256xi1>, %arg1: tensor<1024xi1>, %args2: index) -> (tensor<1024xi1>) {
  %ret = tensor.insert_slice %arg0 into %arg1[%args2] [256] [1] : tensor<256xi1> into tensor<1024xi1>
return %ret : tensor<1024xi1>
}

// -----
// CHECK-LABEL: func.func @test_insert_slice_i8
// CHECK: %[[INSERTED:.*]] = tensor.insert_slice %arg0 into %arg1[%arg2] [32] [1] : tensor<32xi8> into tensor<1024xi8>
// CHECK: return %[[INSERTED]] : tensor<1024xi8>
func.func @test_insert_slice_i8(%arg0: tensor<32xi8>, %arg1: tensor<1024xi8>, %args2: index) -> (tensor<1024xi8>) {
  %ret = tensor.insert_slice %arg0 into %arg1[%args2] [32] [1] : tensor<32xi8> into tensor<1024xi8>
  return %ret : tensor<1024xi8>
}

// -----
// CHECK-LABEL: func.func @test_batch_matmul_f32_output
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x16x32xf16>, %[[ARG1:.*]]: tensor<2x32x16xf16>)
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<2x16x16xf32>
// CHECK: %[[MATMUL:.*]] = linalg.batch_matmul ins(%[[ARG0]], %[[ARG1]] : tensor<2x16x32xf16>, tensor<2x32x16xf16>) outs(%[[EMPTY]] : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
// CHECK: return %[[MATMUL]]
func.func @test_batch_matmul_f32_output(%arg0: tensor<2x16x32xf16>, %arg1: tensor<2x32x16xf16>) -> tensor<2x16x16xf32> {
  %0 = tensor.empty() : tensor<2x16x16xf32>
  %1 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<2x16x32xf16>, tensor<2x32x16xf16>) outs(%0 : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
  return %1 : tensor<2x16x16xf32>
}

// -----
// CHECK-LABEL: func.func @test_matmul_f16_output
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x32xf16>, %[[ARG1:.*]]: tensor<32x16xf16>)
// CHECK-DAG: %[[EMPTY_F16:.*]] = tensor.empty() : tensor<16x16xf16>
// CHECK-DAG: %[[EMPTY_F32:.*]] = tensor.empty() : tensor<16x16xf32>
// CHECK: %[[CAST_OUT:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[EMPTY_F16]] : tensor<16x16xf16>) outs(%[[EMPTY_F32]] : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK: %[[MATMUL:.*]] = linalg.matmul ins(%[[ARG0]], %[[ARG1]] : tensor<16x32xf16>, tensor<32x16xf16>) outs(%[[CAST_OUT]] : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK-DAG: %[[EMPTY_F16_2:.*]] = tensor.empty() : tensor<16x16xf16>
// CHECK: %[[CAST_RES:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[MATMUL]] : tensor<16x16xf32>) outs(%[[EMPTY_F16_2]] : tensor<16x16xf16>) -> tensor<16x16xf16>
// CHECK: return %[[CAST_RES]]
func.func @test_matmul_f16_output(%arg0: tensor<16x32xf16>, %arg1: tensor<32x16xf16>) -> tensor<16x16xf16> {
  %0 = tensor.empty() : tensor<16x16xf16>
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<16x32xf16>, tensor<32x16xf16>) outs(%0 : tensor<16x16xf16>) -> tensor<16x16xf16>
  return %1 : tensor<16x16xf16>
}

// -----
// CHECK-LABEL: func.func @test_matmul_f32_output
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x32xf16>, %[[ARG1:.*]]: tensor<32x16xf16>)
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<16x16xf32>
// CHECK: %[[MATMUL:.*]] = linalg.matmul ins(%[[ARG0]], %[[ARG1]] : tensor<16x32xf16>, tensor<32x16xf16>) outs(%[[EMPTY]] : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK: return %[[MATMUL]]
func.func @test_matmul_f32_output(%arg0: tensor<16x32xf16>, %arg1: tensor<32x16xf16>) -> tensor<16x16xf32> {
  %0 = tensor.empty() : tensor<16x16xf32>
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<16x32xf16>, tensor<32x16xf16>) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %1 : tensor<16x16xf32>
}




// -----
// CHECK-LABEL: module {
// CHECK-NEXT:   func.func @test_hfusion_erfinv_ops(%arg0: tensor<5x1xf16>) -> tensor<5x1xf16> {
// CHECK-NEXT:     %cst = arith.constant 0x7F800000 : f32
// CHECK-NEXT:     %cst_0 = arith.constant 2.83297682 : f32
// CHECK-NEXT:     %cst_1 = arith.constant 1.50140941 : f32
// CHECK-NEXT:     %cst_2 = arith.constant 1.00167406 : f32
// CHECK-NEXT:     %cst_3 = arith.constant 0.246640727 : f32
// CHECK-NEXT:     %cst_4 = arith.constant 0.00943887047 : f32
// CHECK-NEXT:     %cst_5 = arith.constant -0.00417768164 : f32
// CHECK-NEXT:     %cst_6 = arith.constant -0.0076224613 : f32
// CHECK-NEXT:     %cst_7 = arith.constant -0.00125372503 : f32
// CHECK-NEXT:     %cst_8 = arith.constant 0.00573950773 : f32
// CHECK-NEXT:     %cst_9 = arith.constant 2.1858087E-4 : f32
// CHECK-NEXT:     %cst_10 = arith.constant -0.00367342844 : f32
// CHECK-NEXT:     %cst_11 = arith.constant -4.39150654E-6 : f32
// CHECK-NEXT:     %cst_12 = arith.constant 0.00134934322 : f32
// CHECK-NEXT:     %cst_13 = arith.constant -3.5233877E-6 : f32
// CHECK-NEXT:     %cst_14 = arith.constant 1.00950558E-4 : f32
// CHECK-NEXT:     %cst_15 = arith.constant 3.43273939E-7 : f32
// CHECK-NEXT:     %cst_16 = arith.constant -2.00214257E-4 : f32
// CHECK-NEXT:     %cst_17 = arith.constant 2.81022636E-8 : f32
// CHECK-NEXT:     %cst_18 = arith.constant -3.000000e+00 : f32
// CHECK-NEXT:     %cst_19 = arith.constant -2.500000e+00 : f32
// CHECK-NEXT:     %cst_20 = arith.constant 5.000000e+00 : f32
// CHECK-NEXT:     %cst_21 = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:     %cst_22 = arith.constant -1.000000e+00 : f32
// CHECK-NEXT:     %0 = tensor.empty() : tensor<5x1xf32>
// CHECK-NEXT:     %1 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%arg0 : tensor<5x1xf16>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %2 = tensor.empty() : tensor<5x1xf32>
// CHECK-NEXT:     %3 = tensor.empty() : tensor<5x1xi1>
// CHECK-NEXT:     %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %1 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %cst_22 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%5, %cst_21 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %7 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%6 : tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%7, %cst_22 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %9 = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>} ins(%8, %cst_20 : tensor<5x1xf32>, f32) outs(%3 : tensor<5x1xi1>) -> tensor<5x1xi1>
// CHECK-NEXT:     %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%8, %cst_19 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %11 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%8 : tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%11, %cst_18 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %13 = hfusion.select ins(%9, %10, %12 : tensor<5x1xi1>, tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %14 = hfusion.select ins(%9, %cst_17, %cst_16 : tensor<5x1xi1>, f32, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %15 = hfusion.select ins(%9, %cst_15, %cst_14 : tensor<5x1xi1>, f32, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %16 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%14, %13 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%16, %15 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %18 = hfusion.select ins(%9, %cst_13, %cst_12 : tensor<5x1xi1>, f32, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %19 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%17, %13 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %20 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%19, %18 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %21 = hfusion.select ins(%9, %cst_11, %cst_10 : tensor<5x1xi1>, f32, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%20, %13 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%22, %21 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %24 = hfusion.select ins(%9, %cst_9, %cst_8 : tensor<5x1xi1>, f32, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %25 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%23, %13 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %26 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%25, %24 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %27 = hfusion.select ins(%9, %cst_7, %cst_6 : tensor<5x1xi1>, f32, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %28 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%26, %13 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %29 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%28, %27 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %30 = hfusion.select ins(%9, %cst_5, %cst_4 : tensor<5x1xi1>, f32, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %31 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%29, %13 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %32 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%31, %30 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %33 = hfusion.select ins(%9, %cst_3, %cst_2 : tensor<5x1xi1>, f32, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %34 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%32, %13 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %35 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%34, %33 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %36 = hfusion.select ins(%9, %cst_1, %cst_0 : tensor<5x1xi1>, f32, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %37 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%35, %13 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %38 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%37, %36 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %39 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%38, %1 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %40 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%1 : tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %41 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%40, %cst_21 : tensor<5x1xf32>, f32) outs(%3 : tensor<5x1xi1>) -> tensor<5x1xi1>
// CHECK-NEXT:     %42 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %cst : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %43 = hfusion.select ins(%41, %42, %39 : tensor<5x1xi1>, tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %44 = tensor.empty() : tensor<5x1xf16>
// CHECK-NEXT:     %45 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%43 : tensor<5x1xf32>) outs(%44 : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK-NEXT:     return %45 : tensor<5x1xf16>
// CHECK-NEXT:   }
// CHECK-NEXT: }
func.func @test_hfusion_erfinv_ops(%arg0 : tensor<5x1xf16>) -> tensor<5x1xf16> {
  %ret = hfusion.erfinv %arg0 : tensor<5x1xf16> -> tensor<5x1xf16>
  return %ret : tensor<5x1xf16>
}


// -----
// CHECK-LABEL: module {
// CHECK-NEXT:   func.func @test_hfusion_hypot_2_inputs_bf16(%arg0: tensor<1024xbf16>, %arg1: tensor<1024xbf16>) -> tensor<1024xbf16> {
// CHECK-NEXT:     %c-65536_i32 = arith.constant -65536 : i32
// CHECK-NEXT:     %c32767_i32 = arith.constant 32767 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT:     %cst = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:     %cst_0 = arith.constant 1.30438176E+19 : f32
// CHECK-NEXT:     %cst_1 = arith.constant 1.08420217E-19 : f32
// CHECK-NEXT:     %cst_2 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %cst_3 = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:     %cst_4 = arith.constant 0x7F800000 : f32
// CHECK-NEXT:     %cst_5 = arith.constant 0x7FC00000 : f32
// CHECK-NEXT:     %0 = tensor.empty() : tensor<1024xf32>
// CHECK-NEXT:     %1 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%arg0 : tensor<1024xbf16>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %2 = tensor.empty() : tensor<1024xf32>
// CHECK-NEXT:     %3 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%arg1 : tensor<1024xbf16>) outs(%2 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %4 = tensor.empty() : tensor<1024xf32>
// CHECK-NEXT:     %5 = tensor.empty() : tensor<1024xi1>
// CHECK-NEXT:     %6 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%1 : tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %7 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%3 : tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %8 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%6, %cst_4 : tensor<1024xf32>, f32) outs(%5 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %9 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%7, %cst_4 : tensor<1024xf32>, f32) outs(%5 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %10 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%8, %9 : tensor<1024xi1>, tensor<1024xi1>) outs(%5 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %11 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%1, %1 : tensor<1024xf32>, tensor<1024xf32>) outs(%5 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %12 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%11 : tensor<1024xi1>) outs(%5 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %13 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%3, %3 : tensor<1024xf32>, tensor<1024xf32>) outs(%5 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %14 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%13 : tensor<1024xi1>) outs(%5 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %15 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%12, %14 : tensor<1024xi1>, tensor<1024xi1>) outs(%5 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %16 = hfusion.compare {compare_fn = #hfusion.compare_fn<vgt>} ins(%6, %7 : tensor<1024xf32>, tensor<1024xf32>) outs(%5 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %17 = hfusion.select ins(%16, %6, %7 : tensor<1024xi1>, tensor<1024xf32>, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %18 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%17, %cst_2 : tensor<1024xf32>, f32) outs(%5 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %19 = hfusion.select ins(%18, %cst_3, %17 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %20 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%6, %17 : tensor<1024xf32>, tensor<1024xf32>) outs(%5 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %21 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%6, %19 : tensor<1024xf32>, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%21, %21 : tensor<1024xf32>, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %23 = hfusion.select ins(%20, %cst_3, %22 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %24 = hfusion.select ins(%18, %cst_2, %23 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %25 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%7, %17 : tensor<1024xf32>, tensor<1024xf32>) outs(%5 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %26 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%7, %19 : tensor<1024xf32>, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %27 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%26, %26 : tensor<1024xf32>, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %28 = hfusion.select ins(%25, %cst_3, %27 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %29 = hfusion.select ins(%18, %cst_2, %28 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %30 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%24, %29 : tensor<1024xf32>, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %31 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%30 : tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %32 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %6 : tensor<1024xf32>, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %33 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%7, %7 : tensor<1024xf32>, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %34 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%32, %33 : tensor<1024xf32>, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %35 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%34 : tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %36 = hfusion.compare {compare_fn = #hfusion.compare_fn<vge>} ins(%17, %cst_1 : tensor<1024xf32>, f32) outs(%5 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %37 = hfusion.compare {compare_fn = #hfusion.compare_fn<vle>} ins(%17, %cst_0 : tensor<1024xf32>, f32) outs(%5 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %38 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%36, %37 : tensor<1024xi1>, tensor<1024xi1>) outs(%5 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %39 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%31, %cst_2 : tensor<1024xf32>, f32) outs(%5 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %40 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%18, %39 : tensor<1024xi1>, tensor<1024xi1>) outs(%5 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %41 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%30, %31 : tensor<1024xf32>, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %42 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%31, %41 : tensor<1024xf32>, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %43 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%42, %cst : tensor<1024xf32>, f32) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %44 = hfusion.select ins(%40, %31, %43 : tensor<1024xi1>, tensor<1024xf32>, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %45 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%17, %44 : tensor<1024xf32>, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %46 = hfusion.select ins(%38, %35, %45 : tensor<1024xi1>, tensor<1024xf32>, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %47 = hfusion.select ins(%15, %cst_5, %46 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %48 = hfusion.select ins(%10, %cst_4, %47 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %49 = tensor.empty() : tensor<1024xi32>
// CHECK-NEXT:     %50 = tensor.empty() : tensor<1024xf32>
// CHECK-NEXT:     %51 = hfusion.bitcast ins(%48 : tensor<1024xf32>) outs(%49 : tensor<1024xi32>) -> tensor<1024xi32>
// CHECK-NEXT:     %52 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%51, %c16_i32 : tensor<1024xi32>, i32) outs(%49 : tensor<1024xi32>) -> tensor<1024xi32>
// CHECK-NEXT:     %53 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%52, %c1_i32 : tensor<1024xi32>, i32) outs(%49 : tensor<1024xi32>) -> tensor<1024xi32>
// CHECK-NEXT:     %54 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%c32767_i32, %53 : i32, tensor<1024xi32>) outs(%49 : tensor<1024xi32>) -> tensor<1024xi32>
// CHECK-NEXT:     %55 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%51, %54 : tensor<1024xi32>, tensor<1024xi32>) outs(%49 : tensor<1024xi32>) -> tensor<1024xi32>
// CHECK-NEXT:     %56 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%55, %c-65536_i32 : tensor<1024xi32>, i32) outs(%49 : tensor<1024xi32>) -> tensor<1024xi32>
// CHECK-NEXT:     %57 = hfusion.bitcast ins(%56 : tensor<1024xi32>) outs(%50 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %58 = tensor.empty() : tensor<1024xbf16>
// CHECK-NEXT:     %59 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%57 : tensor<1024xf32>) outs(%58 : tensor<1024xbf16>) -> tensor<1024xbf16>
// CHECK-NEXT:     return %59 : tensor<1024xbf16>
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-EMPTY:

func.func @test_hfusion_hypot_2_inputs_bf16(%arg0: tensor<1024xbf16>, %arg1: tensor<1024xbf16>) -> tensor<1024xbf16> {
    %ret = hfusion.hypot %arg0, %arg1 : tensor<1024xbf16>, tensor<1024xbf16> -> tensor<1024xbf16>
    return %ret : tensor<1024xbf16>
  }

// -----

// CHECK-LABEL: module {
// CHECK-NEXT:   func.func @test_hfusion_hypot_2_inputs_f16(%arg0: tensor<1024xf16>, %arg1: tensor<1024xf16>) -> tensor<1024xf16> {
// CHECK-NEXT:     %cst = arith.constant 0x7F800000 : f32
// CHECK-NEXT:     %cst_0 = arith.constant 0x7FC00000 : f32
// CHECK-NEXT:     %0 = tensor.empty() : tensor<1024xf32>
// CHECK-NEXT:     %1 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%arg0 : tensor<1024xf16>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %2 = tensor.empty() : tensor<1024xf32>
// CHECK-NEXT:     %3 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%arg1 : tensor<1024xf16>) outs(%2 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %4 = tensor.empty() : tensor<1024xf32>
// CHECK-NEXT:     %5 = tensor.empty() : tensor<1024xi1>
// CHECK-NEXT:     %6 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%1 : tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %7 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%3 : tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %8 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%6, %cst : tensor<1024xf32>, f32) outs(%5 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %9 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%7, %cst : tensor<1024xf32>, f32) outs(%5 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %10 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%8, %9 : tensor<1024xi1>, tensor<1024xi1>) outs(%5 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %11 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%1, %1 : tensor<1024xf32>, tensor<1024xf32>) outs(%5 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %12 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%11 : tensor<1024xi1>) outs(%5 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %13 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%3, %3 : tensor<1024xf32>, tensor<1024xf32>) outs(%5 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %14 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%13 : tensor<1024xi1>) outs(%5 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %15 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%12, %14 : tensor<1024xi1>, tensor<1024xi1>) outs(%5 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %16 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %6 : tensor<1024xf32>, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%7, %7 : tensor<1024xf32>, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %18 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%16, %17 : tensor<1024xf32>, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %19 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%18 : tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %20 = hfusion.select ins(%15, %cst_0, %19 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %21 = hfusion.select ins(%10, %cst, %20 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %22 = tensor.empty() : tensor<1024xf16>
// CHECK-NEXT:     %23 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%21 : tensor<1024xf32>) outs(%22 : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK-NEXT:     return %23 : tensor<1024xf16>
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-EMPTY:

func.func @test_hfusion_hypot_2_inputs_f16(%arg0: tensor<1024xf16>, %arg1: tensor<1024xf16>) -> tensor<1024xf16> {
    %ret = hfusion.hypot %arg0, %arg1 : tensor<1024xf16>, tensor<1024xf16> -> tensor<1024xf16>
    return %ret : tensor<1024xf16>
  }

// -----

// CHECK-LABEL: module {
// CHECK-NEXT:   func.func @test_hfusion_hypot_2_inputs_f32(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
// CHECK-NEXT:     %cst = arith.constant 0x7FC00000 : f32
// CHECK-NEXT:     %cst_0 = arith.constant 0x7F800000 : f32
// CHECK-NEXT:     %cst_1 = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:     %cst_2 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %0 = tensor.empty() : tensor<1024xf32>
// CHECK-NEXT:     %1 = tensor.empty() : tensor<1024xi1>
// CHECK-NEXT:     %2 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %3 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%arg1 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %4 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%2, %cst_0 : tensor<1024xf32>, f32) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %5 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%3, %cst_0 : tensor<1024xf32>, f32) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %6 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%4, %5 : tensor<1024xi1>, tensor<1024xi1>) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %7 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%arg0, %arg0 : tensor<1024xf32>, tensor<1024xf32>) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %8 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%7 : tensor<1024xi1>) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %9 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%arg1, %arg1 : tensor<1024xf32>, tensor<1024xf32>) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %10 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%9 : tensor<1024xi1>) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %11 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%8, %10 : tensor<1024xi1>, tensor<1024xi1>) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %12 = hfusion.compare {compare_fn = #hfusion.compare_fn<vgt>} ins(%2, %3 : tensor<1024xf32>, tensor<1024xf32>) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %13 = hfusion.select ins(%12, %2, %3 : tensor<1024xi1>, tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %14 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%13, %cst_2 : tensor<1024xf32>, f32) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %15 = hfusion.select ins(%14, %cst_1, %13 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %16 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%2, %13 : tensor<1024xf32>, tensor<1024xf32>) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%2, %15 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %18 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%17, %17 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %19 = hfusion.select ins(%16, %cst_1, %18 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %20 = hfusion.select ins(%14, %cst_2, %19 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %21 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%3, %13 : tensor<1024xf32>, tensor<1024xf32>) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%3, %15 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%22, %22 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %24 = hfusion.select ins(%21, %cst_1, %23 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %25 = hfusion.select ins(%14, %cst_2, %24 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %26 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%20, %25 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %27 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%26 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %28 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%13, %27 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %29 = hfusion.select ins(%11, %cst, %28 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %30 = hfusion.select ins(%6, %cst_0, %29 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     return %30 : tensor<1024xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-EMPTY:

func.func @test_hfusion_hypot_2_inputs_f32(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %ret = hfusion.hypot %arg0, %arg1 : tensor<1024xf32>, tensor<1024xf32> -> tensor<1024xf32>
    return %ret : tensor<1024xf32>
  }

// -----

// CHECK-LABEL: module {
// CHECK-NEXT:   func.func @test_hfusion_hypot_3_inputs(%arg0: tensor<1024xf16>, %arg1: tensor<1024xf16>, %arg2: tensor<1024xf16>) -> tensor<1024xf16> {
// CHECK-NEXT:     %cst = arith.constant 0x7F800000 : f32
// CHECK-NEXT:     %cst_0 = arith.constant 0x7FC00000 : f32
// CHECK-NEXT:     %0 = tensor.empty() : tensor<1024xf32>
// CHECK-NEXT:     %1 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%arg0 : tensor<1024xf16>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %2 = tensor.empty() : tensor<1024xf32>
// CHECK-NEXT:     %3 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%arg1 : tensor<1024xf16>) outs(%2 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %4 = tensor.empty() : tensor<1024xf32>
// CHECK-NEXT:     %5 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%arg2 : tensor<1024xf16>) outs(%4 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %6 = tensor.empty() : tensor<1024xf32>
// CHECK-NEXT:     %7 = tensor.empty() : tensor<1024xi1>
// CHECK-NEXT:     %8 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%1 : tensor<1024xf32>) outs(%6 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %9 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%3 : tensor<1024xf32>) outs(%6 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %10 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%5 : tensor<1024xf32>) outs(%6 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %11 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%8, %cst : tensor<1024xf32>, f32) outs(%7 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %12 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%9, %cst : tensor<1024xf32>, f32) outs(%7 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %13 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%11, %12 : tensor<1024xi1>, tensor<1024xi1>) outs(%7 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %14 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%1, %1 : tensor<1024xf32>, tensor<1024xf32>) outs(%7 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %15 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%14 : tensor<1024xi1>) outs(%7 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %16 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%3, %3 : tensor<1024xf32>, tensor<1024xf32>) outs(%7 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %17 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%16 : tensor<1024xi1>) outs(%7 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %18 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%15, %17 : tensor<1024xi1>, tensor<1024xi1>) outs(%7 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %19 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%10, %cst : tensor<1024xf32>, f32) outs(%7 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %20 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%13, %19 : tensor<1024xi1>, tensor<1024xi1>) outs(%7 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %21 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%5, %5 : tensor<1024xf32>, tensor<1024xf32>) outs(%7 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %22 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%21 : tensor<1024xi1>) outs(%7 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %23 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%18, %22 : tensor<1024xi1>, tensor<1024xi1>) outs(%7 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %24 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%8, %8 : tensor<1024xf32>, tensor<1024xf32>) outs(%6 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %25 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%9, %9 : tensor<1024xf32>, tensor<1024xf32>) outs(%6 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %26 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%24, %25 : tensor<1024xf32>, tensor<1024xf32>) outs(%6 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %27 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%10, %10 : tensor<1024xf32>, tensor<1024xf32>) outs(%6 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %28 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%26, %27 : tensor<1024xf32>, tensor<1024xf32>) outs(%6 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %29 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%28 : tensor<1024xf32>) outs(%6 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %30 = hfusion.select ins(%23, %cst_0, %29 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%6 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %31 = hfusion.select ins(%20, %cst, %30 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%6 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %32 = tensor.empty() : tensor<1024xf16>
// CHECK-NEXT:     %33 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%31 : tensor<1024xf32>) outs(%32 : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK-NEXT:     return %33 : tensor<1024xf16>
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-EMPTY:

func.func @test_hfusion_hypot_3_inputs(%arg0: tensor<1024xf16>, %arg1: tensor<1024xf16>, %arg2: tensor<1024xf16>) -> tensor<1024xf16> {
    %ret = hfusion.hypot %arg0, %arg1, %arg2 : tensor<1024xf16>, tensor<1024xf16>, tensor<1024xf16> -> tensor<1024xf16>
    return %ret : tensor<1024xf16>
  }

// -----

// CHECK-LABEL: module {
// CHECK-NEXT:   func.func @test_hfusion_hypot_3_inputs(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>) -> tensor<1024xf32> {
// CHECK-NEXT:     %cst = arith.constant 0x7FC00000 : f32
// CHECK-NEXT:     %cst_0 = arith.constant 0x7F800000 : f32
// CHECK-NEXT:     %cst_1 = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:     %cst_2 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %0 = tensor.empty() : tensor<1024xf32>
// CHECK-NEXT:     %1 = tensor.empty() : tensor<1024xi1>
// CHECK-NEXT:     %2 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %3 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%arg1 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %4 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%arg2 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %5 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%2, %cst_0 : tensor<1024xf32>, f32) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %6 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%3, %cst_0 : tensor<1024xf32>, f32) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %7 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%5, %6 : tensor<1024xi1>, tensor<1024xi1>) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %8 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%arg0, %arg0 : tensor<1024xf32>, tensor<1024xf32>) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %9 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%8 : tensor<1024xi1>) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %10 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%arg1, %arg1 : tensor<1024xf32>, tensor<1024xf32>) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %11 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%10 : tensor<1024xi1>) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %12 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%9, %11 : tensor<1024xi1>, tensor<1024xi1>) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %13 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%4, %cst_0 : tensor<1024xf32>, f32) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %14 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%7, %13 : tensor<1024xi1>, tensor<1024xi1>) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %15 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%arg2, %arg2 : tensor<1024xf32>, tensor<1024xf32>) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %16 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%15 : tensor<1024xi1>) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %17 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%12, %16 : tensor<1024xi1>, tensor<1024xi1>) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %18 = hfusion.compare {compare_fn = #hfusion.compare_fn<vgt>} ins(%2, %3 : tensor<1024xf32>, tensor<1024xf32>) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %19 = hfusion.select ins(%18, %2, %3 : tensor<1024xi1>, tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %20 = hfusion.compare {compare_fn = #hfusion.compare_fn<vgt>} ins(%19, %4 : tensor<1024xf32>, tensor<1024xf32>) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %21 = hfusion.select ins(%20, %19, %4 : tensor<1024xi1>, tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %22 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%21, %cst_2 : tensor<1024xf32>, f32) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %23 = hfusion.select ins(%22, %cst_1, %21 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %24 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%2, %21 : tensor<1024xf32>, tensor<1024xf32>) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %25 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%2, %23 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %26 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%25, %25 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %27 = hfusion.select ins(%24, %cst_1, %26 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %28 = hfusion.select ins(%22, %cst_2, %27 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %29 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%3, %21 : tensor<1024xf32>, tensor<1024xf32>) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %30 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%3, %23 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %31 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%30, %30 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %32 = hfusion.select ins(%29, %cst_1, %31 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %33 = hfusion.select ins(%22, %cst_2, %32 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %34 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%28, %33 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %35 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%4, %21 : tensor<1024xf32>, tensor<1024xf32>) outs(%1 : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK-NEXT:     %36 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%4, %23 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %37 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%36, %36 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %38 = hfusion.select ins(%35, %cst_1, %37 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %39 = hfusion.select ins(%22, %cst_2, %38 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %40 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%34, %39 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %41 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%40 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %42 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%21, %41 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %43 = hfusion.select ins(%17, %cst, %42 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     %44 = hfusion.select ins(%14, %cst_0, %43 : tensor<1024xi1>, f32, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NEXT:     return %44 : tensor<1024xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-EMPTY:

func.func @test_hfusion_hypot_3_inputs(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>) -> tensor<1024xf32> {
    %ret = hfusion.hypot %arg0, %arg1, %arg2 : tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32> -> tensor<1024xf32>
    return %ret : tensor<1024xf32>
}


// -----

// CHECK-LABEL: module {
// CHECK-NEXT:   func.func @test_hfusion_cyl_bessel_i0_ops(%arg0: tensor<5x1xf16>) -> tensor<5x1xf16> {
// CHECK-NEXT:     %cst = arith.constant 0.804490387 : f32
// CHECK-NEXT:     %cst_0 = arith.constant 0.00336911646 : f32
// CHECK-NEXT:     %cst_1 = arith.constant 6.88975852E-5 : f32
// CHECK-NEXT:     %cst_2 = arith.constant 2.89137051E-6 : f32
// CHECK-NEXT:     %cst_3 = arith.constant 2.04891862E-7 : f32
// CHECK-NEXT:     %cst_4 = arith.constant 2.26666899E-8 : f32
// CHECK-NEXT:     %cst_5 = arith.constant 3.39623196E-9 : f32
// CHECK-NEXT:     %cst_6 = arith.constant 4.94060237E-10 : f32
// CHECK-NEXT:     %cst_7 = arith.constant 1.18891468E-11 : f32
// CHECK-NEXT:     %cst_8 = arith.constant -3.14991644E-11 : f32
// CHECK-NEXT:     %cst_9 = arith.constant -1.32158121E-11 : f32
// CHECK-NEXT:     %cst_10 = arith.constant -1.79417852E-12 : f32
// CHECK-NEXT:     %cst_11 = arith.constant 7.18012455E-13 : f32
// CHECK-NEXT:     %cst_12 = arith.constant 3.85277829E-13 : f32
// CHECK-NEXT:     %cst_13 = arith.constant 1.54008615E-14 : f32
// CHECK-NEXT:     %cst_14 = arith.constant -4.15056918E-14 : f32
// CHECK-NEXT:     %cst_15 = arith.constant -9.55484674E-15 : f32
// CHECK-NEXT:     %cst_16 = arith.constant 3.81168087E-15 : f32
// CHECK-NEXT:     %cst_17 = arith.constant 1.77256012E-15 : f32
// CHECK-NEXT:     %cst_18 = arith.constant -3.42548568E-16 : f32
// CHECK-NEXT:     %cst_19 = arith.constant -2.82762388E-16 : f32
// CHECK-NEXT:     %cst_20 = arith.constant 3.46122279E-17 : f32
// CHECK-NEXT:     %cst_21 = arith.constant 4.46562156E-17 : f32
// CHECK-NEXT:     %cst_22 = arith.constant -4.83050441E-18 : f32
// CHECK-NEXT:     %cst_23 = arith.constant -7.23318078E-18 : f32
// CHECK-NEXT:     %cst_24 = arith.constant 0.676795303 : f32
// CHECK-NEXT:     %cst_25 = arith.constant -0.304682672 : f32
// CHECK-NEXT:     %cst_26 = arith.constant 0.171620905 : f32
// CHECK-NEXT:     %cst_27 = arith.constant -9.490110e-02 : f32
// CHECK-NEXT:     %cst_28 = arith.constant 0.0493052825 : f32
// CHECK-NEXT:     %cst_29 = arith.constant -0.0237374157 : f32
// CHECK-NEXT:     %cst_30 = arith.constant 0.0105464607 : f32
// CHECK-NEXT:     %cst_31 = arith.constant -4.324310e-03 : f32
// CHECK-NEXT:     %cst_32 = arith.constant 0.00163947558 : f32
// CHECK-NEXT:     %cst_33 = arith.constant -5.76375576E-4 : f32
// CHECK-NEXT:     %cst_34 = arith.constant 1.88502891E-4 : f32
// CHECK-NEXT:     %cst_35 = arith.constant -5.75419508E-5 : f32
// CHECK-NEXT:     %cst_36 = arith.constant 1.64484482E-5 : f32
// CHECK-NEXT:     %cst_37 = arith.constant -4.41673819E-6 : f32
// CHECK-NEXT:     %cst_38 = arith.constant 1.11738757E-6 : f32
// CHECK-NEXT:     %cst_39 = arith.constant -2.67079372E-7 : f32
// CHECK-NEXT:     %cst_40 = arith.constant 6.04699508E-8 : f32
// CHECK-NEXT:     %cst_41 = arith.constant -1.30002498E-8 : f32
// CHECK-NEXT:     %cst_42 = arith.constant 2.65982369E-9 : f32
// CHECK-NEXT:     %cst_43 = arith.constant -5.18979582E-10 : f32
// CHECK-NEXT:     %cst_44 = arith.constant 9.67580876E-11 : f32
// CHECK-NEXT:     %cst_45 = arith.constant -1.72682632E-11 : f32
// CHECK-NEXT:     %cst_46 = arith.constant 2.95505265E-12 : f32
// CHECK-NEXT:     %cst_47 = arith.constant -4.85644673E-13 : f32
// CHECK-NEXT:     %cst_48 = arith.constant 7.67618526E-14 : f32
// CHECK-NEXT:     %cst_49 = arith.constant -1.16853328E-14 : f32
// CHECK-NEXT:     %cst_50 = arith.constant 1.71539133E-15 : f32
// CHECK-NEXT:     %cst_51 = arith.constant -2.431280e-16 : f32
// CHECK-NEXT:     %cst_52 = arith.constant 3.33079461E-17 : f32
// CHECK-NEXT:     %cst_53 = arith.constant -4.41534163E-18 : f32
// CHECK-NEXT:     %cst_54 = arith.constant 3.200000e+01 : f32
// CHECK-NEXT:     %cst_55 = arith.constant 8.000000e+00 : f32
// CHECK-NEXT:     %cst_56 = arith.constant 2.000000e+00 : f32
// CHECK-NEXT:     %cst_57 = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:     %0 = tensor.empty() : tensor<5x1xf32>
// CHECK-NEXT:     %1 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%arg0 : tensor<5x1xf16>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %2 = tensor.empty() : tensor<5x1xf32>
// CHECK-NEXT:     %3 = tensor.empty() : tensor<5x1xi1>
// CHECK-NEXT:     %4 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%1 : tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %cst_57 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%5, %cst_56 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %cst_53 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%7, %cst_52 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %8 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%9, %cst_53 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%10, %cst_51 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %11 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%12, %8 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %14 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%13, %cst_50 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %15 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %14 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %16 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%15, %11 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%16, %cst_49 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %18 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %17 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %19 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%18, %14 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %20 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%19, %cst_48 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %21 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %20 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%21, %17 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%22, %cst_47 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %24 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %23 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %25 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%24, %20 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %26 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%25, %cst_46 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %27 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %26 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %28 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%27, %23 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %29 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%28, %cst_45 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %30 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %29 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %31 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%30, %26 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %32 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%31, %cst_44 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %33 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %32 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %34 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%33, %29 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %35 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%34, %cst_43 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %36 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %35 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %37 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%36, %32 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %38 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%37, %cst_42 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %39 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %38 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %40 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%39, %35 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %41 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%40, %cst_41 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %42 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %41 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %43 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%42, %38 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %44 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%43, %cst_40 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %45 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %44 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %46 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%45, %41 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %47 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%46, %cst_39 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %48 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %47 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %49 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%48, %44 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %50 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%49, %cst_38 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %51 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %50 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %52 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%51, %47 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %53 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%52, %cst_37 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %54 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %53 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %55 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%54, %50 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %56 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%55, %cst_36 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %57 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %56 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %58 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%57, %53 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %59 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%58, %cst_35 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %60 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %59 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %61 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%60, %56 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %62 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%61, %cst_34 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %63 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %62 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %64 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%63, %59 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %65 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%64, %cst_33 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %66 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %65 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %67 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%66, %62 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %68 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%67, %cst_32 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %69 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %68 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %70 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%69, %65 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %71 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%70, %cst_31 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %72 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %71 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %73 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%72, %68 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %74 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%73, %cst_30 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %75 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %74 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %76 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%75, %71 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %77 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%76, %cst_29 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %78 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %77 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %79 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%78, %74 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %80 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%79, %cst_28 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %81 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %80 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %82 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%81, %77 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %83 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%82, %cst_27 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %84 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %83 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %85 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%84, %80 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %86 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%85, %cst_26 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %87 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %86 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %88 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%87, %83 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %89 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%88, %cst_25 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %90 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %89 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %91 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%90, %86 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %92 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%91, %cst_24 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %93 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%92, %86 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %94 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%93, %cst_57 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %95 = hfusion.compare {compare_fn = #hfusion.compare_fn<vle>} ins(%4, %cst_55 : tensor<5x1xf32>, f32) outs(%3 : tensor<5x1xi1>) -> tensor<5x1xi1>
// CHECK-NEXT:     %96 = hfusion.select ins(%95, %cst_55, %4 : tensor<5x1xi1>, f32, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %97 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst_54, %96 : f32, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %98 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%97, %cst_56 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %99 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%98, %cst_23 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %100 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%99, %cst_22 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %101 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%98, %100 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %102 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%101, %cst_23 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %103 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%102, %cst_21 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %104 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%98, %103 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %105 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%104, %100 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %106 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%105, %cst_20 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %107 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%98, %106 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %108 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%107, %103 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %109 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%108, %cst_19 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %110 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%98, %109 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %111 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%110, %106 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %112 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%111, %cst_18 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %113 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%98, %112 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %114 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%113, %109 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %115 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%114, %cst_17 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %116 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%98, %115 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %117 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%116, %112 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %118 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%117, %cst_16 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %119 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%98, %118 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %120 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%119, %115 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %121 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%120, %cst_15 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %122 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%98, %121 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %123 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%122, %118 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %124 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%123, %cst_14 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %125 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%98, %124 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %126 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%125, %121 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %127 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%126, %cst_13 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %128 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%98, %127 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %129 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%128, %124 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %130 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%129, %cst_12 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %131 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%98, %130 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %132 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%131, %127 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %133 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%132, %cst_11 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %134 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%98, %133 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %135 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%134, %130 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %136 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%135, %cst_10 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %137 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%98, %136 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %138 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%137, %133 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %139 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%138, %cst_9 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %140 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%98, %139 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %141 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%140, %136 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %142 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%141, %cst_8 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %143 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%98, %142 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %144 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%143, %139 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %145 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%144, %cst_7 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %146 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%98, %145 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %147 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%146, %142 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %148 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%147, %cst_6 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %149 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%98, %148 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %150 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%149, %145 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %151 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%150, %cst_5 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %152 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%98, %151 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %153 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%152, %148 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %154 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%153, %cst_4 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %155 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%98, %154 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %156 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%155, %151 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %157 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%156, %cst_3 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %158 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%98, %157 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %159 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%158, %154 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %160 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%159, %cst_2 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %161 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%98, %160 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %162 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%161, %157 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %163 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%162, %cst_1 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %164 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%98, %163 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %165 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%164, %160 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %166 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%165, %cst_0 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %167 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%98, %166 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %168 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%167, %163 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %169 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%168, %cst : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %170 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%169, %163 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %171 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%170, %cst_57 : tensor<5x1xf32>, f32) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %172 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%96 : tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %173 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%171, %172 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %174 = hfusion.select ins(%95, %94, %173 : tensor<5x1xi1>, tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %175 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%4 : tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %176 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%175, %174 : tensor<5x1xf32>, tensor<5x1xf32>) outs(%2 : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK-NEXT:     %177 = tensor.empty() : tensor<5x1xf16>
// CHECK-NEXT:     %178 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%176 : tensor<5x1xf32>) outs(%177 : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK-NEXT:     return %178 : tensor<5x1xf16>
// CHECK-NEXT:   }
// CHECK-NEXT: }

func.func @test_hfusion_cyl_bessel_i0_ops(%arg0 : tensor<5x1xf16>) -> tensor<5x1xf16> {
  %ret = hfusion.cyl_bessel_i0 %arg0 : tensor<5x1xf16> -> tensor<5x1xf16>
  return %ret : tensor<5x1xf16>
}


// -----

// CHECK-LABEL: module {
// CHECK-NEXT:   func.func @test_hfusion_nextafter_ops(%arg0: tensor<5x1xf16>, %arg1: tensor<5x1xf16>) -> tensor<5x1xf16> {
// CHECK-NEXT:     %c-1_i16 = arith.constant -1 : i16
// CHECK-NEXT:     %c1_i16 = arith.constant 1 : i16
// CHECK-NEXT:     %c0_i16 = arith.constant 0 : i16
// CHECK-NEXT:     %c32767_i16 = arith.constant 32767 : i16
// CHECK-NEXT:     %c-32768_i16 = arith.constant -32768 : i16
// CHECK-NEXT:     %0 = tensor.empty() : tensor<5x1xf16>
// CHECK-NEXT:     %1 = tensor.empty() : tensor<5x1xi16>
// CHECK-NEXT:     %2 = tensor.empty() : tensor<5x1xi1>
// CHECK-NEXT:     %3 = hfusion.bitcast ins(%arg0 : tensor<5x1xf16>) outs(%1 : tensor<5x1xi16>) -> tensor<5x1xi16>
// CHECK-NEXT:     %4 = hfusion.bitcast ins(%arg1 : tensor<5x1xf16>) outs(%1 : tensor<5x1xi16>) -> tensor<5x1xi16>
// CHECK-NEXT:     %5 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%arg0, %arg0 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%2 : tensor<5x1xi1>) -> tensor<5x1xi1>
// CHECK-NEXT:     %6 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%5 : tensor<5x1xi1>) outs(%2 : tensor<5x1xi1>) -> tensor<5x1xi1>
// CHECK-NEXT:     %7 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%arg1, %arg1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%2 : tensor<5x1xi1>) -> tensor<5x1xi1>
// CHECK-NEXT:     %8 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%7 : tensor<5x1xi1>) outs(%2 : tensor<5x1xi1>) -> tensor<5x1xi1>
// CHECK-NEXT:     %9 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%6, %8 : tensor<5x1xi1>, tensor<5x1xi1>) outs(%2 : tensor<5x1xi1>) -> tensor<5x1xi1>
// CHECK-NEXT:     %10 = hfusion.select ins(%6, %3, %4 : tensor<5x1xi1>, tensor<5x1xi16>, tensor<5x1xi16>) outs(%1 : tensor<5x1xi16>) -> tensor<5x1xi16>
// CHECK-NEXT:     %11 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%3, %c32767_i16 : tensor<5x1xi16>, i16) outs(%1 : tensor<5x1xi16>) -> tensor<5x1xi16>
// CHECK-NEXT:     %12 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%4, %c32767_i16 : tensor<5x1xi16>, i16) outs(%1 : tensor<5x1xi16>) -> tensor<5x1xi16>
// CHECK-NEXT:     %13 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%arg0, %arg1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%2 : tensor<5x1xi1>) -> tensor<5x1xi1>
// CHECK-NEXT:     %14 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%11, %c0_i16 : tensor<5x1xi16>, i16) outs(%2 : tensor<5x1xi1>) -> tensor<5x1xi1>
// CHECK-NEXT:     %15 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%12, %c0_i16 : tensor<5x1xi16>, i16) outs(%2 : tensor<5x1xi1>) -> tensor<5x1xi1>
// CHECK-NEXT:     %16 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%3, %c-32768_i16 : tensor<5x1xi16>, i16) outs(%1 : tensor<5x1xi16>) -> tensor<5x1xi16>
// CHECK-NEXT:     %17 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%4, %c-32768_i16 : tensor<5x1xi16>, i16) outs(%1 : tensor<5x1xi16>) -> tensor<5x1xi16>
// CHECK-NEXT:     %18 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%17, %c1_i16 : tensor<5x1xi16>, i16) outs(%1 : tensor<5x1xi16>) -> tensor<5x1xi16>
// CHECK-NEXT:     %19 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%16, %17 : tensor<5x1xi16>, tensor<5x1xi16>) outs(%2 : tensor<5x1xi1>) -> tensor<5x1xi1>
// CHECK-NEXT:     %20 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%19 : tensor<5x1xi1>) outs(%2 : tensor<5x1xi1>) -> tensor<5x1xi1>
// CHECK-NEXT:     %21 = hfusion.compare {compare_fn = #hfusion.compare_fn<vgt>} ins(%11, %12 : tensor<5x1xi16>, tensor<5x1xi16>) outs(%2 : tensor<5x1xi1>) -> tensor<5x1xi1>
// CHECK-NEXT:     %22 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%21, %20 : tensor<5x1xi1>, tensor<5x1xi1>) outs(%2 : tensor<5x1xi1>) -> tensor<5x1xi1>
// CHECK-NEXT:     %23 = hfusion.select ins(%22, %c-1_i16, %c1_i16 : tensor<5x1xi1>, i16, i16) outs(%1 : tensor<5x1xi16>) -> tensor<5x1xi16>
// CHECK-NEXT:     %24 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %23 : tensor<5x1xi16>, tensor<5x1xi16>) outs(%1 : tensor<5x1xi16>) -> tensor<5x1xi16>
// CHECK-NEXT:     %25 = hfusion.select ins(%15, %4, %18 : tensor<5x1xi1>, tensor<5x1xi16>, tensor<5x1xi16>) outs(%1 : tensor<5x1xi16>) -> tensor<5x1xi16>
// CHECK-NEXT:     %26 = hfusion.select ins(%14, %25, %24 : tensor<5x1xi1>, tensor<5x1xi16>, tensor<5x1xi16>) outs(%1 : tensor<5x1xi16>) -> tensor<5x1xi16>
// CHECK-NEXT:     %27 = hfusion.select ins(%13, %4, %26 : tensor<5x1xi1>, tensor<5x1xi16>, tensor<5x1xi16>) outs(%1 : tensor<5x1xi16>) -> tensor<5x1xi16>
// CHECK-NEXT:     %28 = hfusion.select ins(%9, %10, %27 : tensor<5x1xi1>, tensor<5x1xi16>, tensor<5x1xi16>) outs(%1 : tensor<5x1xi16>) -> tensor<5x1xi16>
// CHECK-NEXT:     %29 = hfusion.bitcast ins(%28 : tensor<5x1xi16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK-NEXT:     return %29 : tensor<5x1xf16>
// CHECK-NEXT:   }
// CHECK-NEXT: }
func.func @test_hfusion_nextafter_ops(%arg0 : tensor<5x1xf16>,
                                      %arg1 : tensor<5x1xf16>)
    -> tensor<5x1xf16> {
  %ret = hfusion.nextafter %arg0, %arg1
    : tensor<5x1xf16>, tensor<5x1xf16> -> tensor<5x1xf16>
  return %ret : tensor<5x1xf16>
}

// -----
// Test lgamma normalization for f32 input
// lgamma(x) = ln|Γ(x)| using Lanczos approximation

// CHECK-LABEL: func.func @test_hfusion_lgamma_f32(
// CHECK-SAME: %[[ARG:.*]]: tensor<8xf32>) -> tensor<8xf32> {

// Original lgamma op should be eliminated
// CHECK-NOT: #hfusion.unary_fn<lgamma>

// Constants
// CHECK-DAG: arith.constant 5.000000e-01 : f32
// CHECK-DAG: arith.constant 1.000000e+00 : f32
// CHECK-DAG: arith.constant -1.000000e+00 : f32
// CHECK-DAG: arith.constant 7.500000e+00 : f32

// Tensor init
// CHECK: tensor.empty

// needToReflect = x < 0.5
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}

// negX implemented as mul(x, -1)
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}

// x - 1
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}

// z = select(...)
// CHECK: hfusion.select

// abs(x)
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}

// Lanczos series
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}

// log1p expanded into add + log
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<log>}

// log(a(z))
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<log>}

// floor(abs(x)) implemented as cast(round_mode=floor)
// CHECK: hfusion.cast
// CHECK-SAME: round_mode = #hfusion.round_mode<floor>

// absFrac > 0.5
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vgt>}

// polynomial-based sin approximation exists
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}

// finiteReflectionDenom = abs(reflectionDenom) < inf
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}

// reflection finite select
// CHECK: hfusion.select

// Final reflection/main select
// CHECK: hfusion.select

// inf input handling
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vge>}
// CHECK: hfusion.select

// CHECK: return %{{.*}} : tensor<8xf32>

func.func @test_hfusion_lgamma_f32(%arg0 : tensor<8xf32>) -> tensor<8xf32> {
  %0 = tensor.empty() : tensor<8xf32>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<lgamma>}
      ins(%arg0 : tensor<8xf32>)
      outs(%0 : tensor<8xf32>)
      -> tensor<8xf32>
  return %ret : tensor<8xf32>
}

// -----
// Test lgamma normalization for f16 input (with F16->F32->F16 conversion)

// CHECK-LABEL: func.func @test_hfusion_lgamma_f16(
// CHECK-SAME: %[[ARG:.*]]: tensor<8xf16>) -> tensor<8xf16> {

// F16 -> F32 cast
// CHECK: tensor.empty() : tensor<8xf32>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>
// CHECK-SAME: round_mode = #hfusion.round_mode<round>}

// Reflection predicate
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}

// negf lowered to mul(-1)
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}

// select
// CHECK: hfusion.select

// Lanczos
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}

// log1p lowered to add + log
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<log>}

// log(a)
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<log>}

// floor lowered to cast(round_mode=floor)
// CHECK: hfusion.cast
// CHECK-SAME: round_mode = #hfusion.round_mode<floor>

// absFrac > 0.5
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vgt>}

// polynomial sin approximation
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}

// finite reflection check
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}

// reflection select
// CHECK: hfusion.select

// inf handling
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vge>}
// CHECK: hfusion.select

// F32 -> F16 cast
// CHECK: tensor.empty() : tensor<8xf16>
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>
// CHECK-SAME: round_mode = #hfusion.round_mode<round>}

// CHECK: return %{{.*}} : tensor<8xf16>

func.func @test_hfusion_lgamma_f16(%arg0 : tensor<8xf16>) -> tensor<8xf16> {
  %0 = tensor.empty() : tensor<8xf16>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<lgamma>}
      ins(%arg0 : tensor<8xf16>)
      outs(%0 : tensor<8xf16>)
      -> tensor<8xf16>
  return %ret : tensor<8xf16>
}

// -----
// CHECK-LABEL: func.func @ldexp_exp_f32_i32(
// CHECK-SAME: %[[ARG0:[^:]*]]: tensor<256xf32>,
// CHECK-SAME: %[[ARG1:[^:]*]]: tensor<256xi32>)
// CHECK: %[[CST0:[^ ]*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[CST_NEG:[^ ]*]] = arith.constant -2.13909504E+9 : f32
// CHECK: %[[CST_LN2:[^ ]*]] = arith.constant 0.693147182 : f32
// CHECK: %[[CAST:[^ ]*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[ARG1]] : tensor<256xi32>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[MUL_LN2:[^ ]*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[CAST]], %[[CST_LN2]] : tensor<256xf32>, f32) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[EXP:[^ ]*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[MUL_LN2]] : tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[NAN_CMP:[^ ]*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[CAST]], %[[CAST]] : tensor<256xf32>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[NOT_NAN:[^ ]*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[NAN_CMP]] : tensor<256xi1>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[SEL_NAN:[^ ]*]] = hfusion.select ins(%[[NOT_NAN]], %[[CAST]], %[[EXP]] : tensor<256xi1>, tensor<256xf32>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[FILL_NEG:[^ ]*]] = linalg.fill ins(%[[CST_NEG]] : f32) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[CMP_NEG:[^ ]*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[CAST]], %[[FILL_NEG]] : tensor<256xf32>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[SEL_NEG:[^ ]*]] = hfusion.select ins(%[[CMP_NEG]], %[[CAST]], %[[SEL_NAN]] : tensor<256xi1>, tensor<256xf32>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[ABS:[^ ]*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[CAST]] : tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[CMP_INF:[^ ]*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[ABS]], %[[FILL_NEG]] : tensor<256xf32>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[NOT_NEG:[^ ]*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[CMP_NEG]] : tensor<256xi1>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[AND_COND:[^ ]*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[NOT_NEG]], %[[CMP_INF]] : tensor<256xi1>, tensor<256xi1>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[FILL_ZERO:[^ ]*]] = linalg.fill ins(%[[CST0]] : f32) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[SEL_INF:[^ ]*]] = hfusion.select ins(%[[AND_COND]], %[[FILL_ZERO]], %[[SEL_NEG]] : tensor<256xi1>, tensor<256xf32>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[RES:[^ ]*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ARG0]], %[[SEL_INF]] : tensor<256xf32>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: return %[[RES]] : tensor<256xf32>
func.func @ldexp_exp_f32_i32(%arg0: tensor<256xf32>, %arg1: tensor<256xi32>) -> (tensor<256xf32>) {
  %0 = tensor.empty() : tensor<256xf32>
  %ret = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ldexp>} ins(%arg0, %arg1 : tensor<256xf32>, tensor<256xi32>) outs(%0 : tensor<256xf32>) -> tensor<256xf32>
  return %ret : tensor<256xf32>
}

// -----
// CHECK-LABEL: func.func @ldexp_exp_f32_f32(
// CHECK-SAME: %[[ARG0:[^:]*]]: tensor<256xf32>,
// CHECK-SAME: %[[ARG1:[^:]*]]: tensor<256xf32>)
// CHECK: %[[CST0:[^ ]*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[CST_NEG:[^ ]*]] = arith.constant -2.13909504E+9 : f32
// CHECK: %[[CST_LN2:[^ ]*]] = arith.constant 0.693147182 : f32
// CHECK: %[[MUL_LN2:[^ ]*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ARG1]], %[[CST_LN2]] : tensor<256xf32>, f32) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[EXP:[^ ]*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[MUL_LN2]] : tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[NAN_CMP:[^ ]*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[ARG1]], %[[ARG1]] : tensor<256xf32>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[NOT_NAN:[^ ]*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[NAN_CMP]] : tensor<256xi1>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[SEL_NAN:[^ ]*]] = hfusion.select ins(%[[NOT_NAN]], %[[ARG1]], %[[EXP]] : tensor<256xi1>, tensor<256xf32>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[FILL_NEG:[^ ]*]] = linalg.fill ins(%[[CST_NEG]] : f32) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[CMP_NEG:[^ ]*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[ARG1]], %[[FILL_NEG]] : tensor<256xf32>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[SEL_NEG:[^ ]*]] = hfusion.select ins(%[[CMP_NEG]], %[[ARG1]], %[[SEL_NAN]] : tensor<256xi1>, tensor<256xf32>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[ABS:[^ ]*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[ARG1]] : tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[CMP_INF:[^ ]*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[ABS]], %[[FILL_NEG]] : tensor<256xf32>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[NOT_NEG:[^ ]*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[CMP_NEG]] : tensor<256xi1>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[AND_COND:[^ ]*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[NOT_NEG]], %[[CMP_INF]] : tensor<256xi1>, tensor<256xi1>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[FILL_ZERO:[^ ]*]] = linalg.fill ins(%[[CST0]] : f32) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[SEL_INF:[^ ]*]] = hfusion.select ins(%[[AND_COND]], %[[FILL_ZERO]], %[[SEL_NEG]] : tensor<256xi1>, tensor<256xf32>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[RES:[^ ]*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ARG0]], %[[SEL_INF]] : tensor<256xf32>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: return %[[RES]] : tensor<256xf32>
func.func @ldexp_exp_f32_f32(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> (tensor<256xf32>) {
    %0 = tensor.empty() : tensor<256xf32>
    %ret = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ldexp>} ins(%arg0, %arg1 : tensor<256xf32>, tensor<256xf32>) outs(%0 : tensor<256xf32>) -> tensor<256xf32>
    return %ret : tensor<256xf32>
}


// -----
// CHECK-LABEL: func.func @ldexp_exp_f16_i32(
// CHECK-SAME: %[[ARG0:[^:]*]]: tensor<256xf16>,
// CHECK-SAME: %[[ARG1:[^:]*]]: tensor<256xi32>)
// CHECK: %[[CST0:[^ ]*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[CST_NEG:[^ ]*]] = arith.constant -2.13909504E+9 : f32
// CHECK: %[[CST_LN2:[^ ]*]] = arith.constant 0.693147182 : f32
// CHECK: %[[CAST_TO_F32:[^ ]*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[ARG1]] : tensor<256xi32>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[MUL_LN2:[^ ]*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[CAST_TO_F32]], %[[CST_LN2]] : tensor<256xf32>, f32) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[EXP:[^ ]*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[MUL_LN2]] : tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[NAN_CMP:[^ ]*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[CAST_TO_F32]], %[[CAST_TO_F32]] : tensor<256xf32>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[NOT_NAN:[^ ]*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[NAN_CMP]] : tensor<256xi1>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[SEL_NAN:[^ ]*]] = hfusion.select ins(%[[NOT_NAN]], %[[CAST_TO_F32]], %[[EXP]] : tensor<256xi1>, tensor<256xf32>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[FILL_NEG:[^ ]*]] = linalg.fill ins(%[[CST_NEG]] : f32) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[CMP_NEG:[^ ]*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[CAST_TO_F32]], %[[FILL_NEG]] : tensor<256xf32>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[SEL_NEG:[^ ]*]] = hfusion.select ins(%[[CMP_NEG]], %[[CAST_TO_F32]], %[[SEL_NAN]] : tensor<256xi1>, tensor<256xf32>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[ABS:[^ ]*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[CAST_TO_F32]] : tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[CMP_INF:[^ ]*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[ABS]], %[[FILL_NEG]] : tensor<256xf32>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[NOT_NEG:[^ ]*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[CMP_NEG]] : tensor<256xi1>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[AND_COND:[^ ]*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[NOT_NEG]], %[[CMP_INF]] : tensor<256xi1>, tensor<256xi1>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[FILL_ZERO:[^ ]*]] = linalg.fill ins(%[[CST0]] : f32) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[SEL_INF:[^ ]*]] = hfusion.select ins(%[[AND_COND]], %[[FILL_ZERO]], %[[SEL_NEG]] : tensor<256xi1>, tensor<256xf32>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[RES:[^ ]*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ARG0]], %[[SEL_INF]] : tensor<256xf16>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf16>) -> tensor<256xf16>
// CHECK: return %[[RES]] : tensor<256xf16>
func.func @ldexp_exp_f16_i32(%arg0: tensor<256xf16>, %arg1: tensor<256xi32>) -> (tensor<256xf16>) {
    %0 = tensor.empty() : tensor<256xf16>
    %ret = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ldexp>} ins(%arg0, %arg1 : tensor<256xf16>, tensor<256xi32>) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
    return %ret : tensor<256xf16>
}

// -----
// CHECK-LABEL: func.func @ldexp_exp_f16_f16(
// CHECK-SAME: %[[ARG0:[^:]*]]: tensor<256xf16>,
// CHECK-SAME: %[[ARG1:[^:]*]]: tensor<256xf16>)
// CHECK: %[[CST0:[^ ]*]] = arith.constant 0.000000e+00 : f16
// CHECK: %[[CST_NEG:[^ ]*]] = arith.constant -3.174400e+04 : f16
// CHECK: %[[CST_LN2:[^ ]*]] = arith.constant 0.693147182 : f32
// CHECK: %[[CAST_TO_F32:[^ ]*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%[[ARG1]] : tensor<256xf16>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[MUL_LN2:[^ ]*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[CAST_TO_F32]], %[[CST_LN2]] : tensor<256xf32>, f32) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[EXP:[^ ]*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[MUL_LN2]] : tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[CAST_BACK_TO_F16:[^ ]*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%[[EXP]] : tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf16>) -> tensor<256xf16>
// CHECK: %[[NAN_CMP:[^ ]*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[ARG1]], %[[ARG1]] : tensor<256xf16>, tensor<256xf16>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[NOT_NAN:[^ ]*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[NAN_CMP]] : tensor<256xi1>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[SEL_NAN:[^ ]*]] = hfusion.select ins(%[[NOT_NAN]], %[[ARG1]], %[[CAST_BACK_TO_F16]] : tensor<256xi1>, tensor<256xf16>, tensor<256xf16>) outs(%{{[^ ]*}} : tensor<256xf16>) -> tensor<256xf16>
// CHECK: %[[FILL_NEG:[^ ]*]] = linalg.fill ins(%[[CST_NEG]] : f16) outs(%{{[^ ]*}} : tensor<256xf16>) -> tensor<256xf16>
// CHECK: %[[CMP_NEG:[^ ]*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[ARG1]], %[[FILL_NEG]] : tensor<256xf16>, tensor<256xf16>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[SEL_NEG:[^ ]*]] = hfusion.select ins(%[[CMP_NEG]], %[[ARG1]], %[[SEL_NAN]] : tensor<256xi1>, tensor<256xf16>, tensor<256xf16>) outs(%{{[^ ]*}} : tensor<256xf16>) -> tensor<256xf16>
// CHECK: %[[ABS:[^ ]*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[ARG1]] : tensor<256xf16>) outs(%{{[^ ]*}} : tensor<256xf16>) -> tensor<256xf16>
// CHECK: %[[CMP_INF:[^ ]*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[ABS]], %[[FILL_NEG]] : tensor<256xf16>, tensor<256xf16>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[NOT_NEG:[^ ]*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[CMP_NEG]] : tensor<256xi1>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[AND_COND:[^ ]*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[NOT_NEG]], %[[CMP_INF]] : tensor<256xi1>, tensor<256xi1>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[FILL_ZERO:[^ ]*]] = linalg.fill ins(%[[CST0]] : f16) outs(%{{[^ ]*}} : tensor<256xf16>) -> tensor<256xf16>
// CHECK: %[[SEL_INF:[^ ]*]] = hfusion.select ins(%[[AND_COND]], %[[FILL_ZERO]], %[[SEL_NEG]] : tensor<256xi1>, tensor<256xf16>, tensor<256xf16>) outs(%{{[^ ]*}} : tensor<256xf16>) -> tensor<256xf16>
// CHECK: %[[RES:[^ ]*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ARG0]], %[[SEL_INF]] : tensor<256xf16>, tensor<256xf16>) outs(%{{[^ ]*}} : tensor<256xf16>) -> tensor<256xf16>
// CHECK: return %[[RES]] : tensor<256xf16>
func.func @ldexp_exp_f16_f16(%arg0: tensor<256xf16>, %arg1: tensor<256xf16>) -> (tensor<256xf16>) {
    %0 = tensor.empty() : tensor<256xf16>
    %ret = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ldexp>} ins(%arg0, %arg1 : tensor<256xf16>, tensor<256xf16>) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
    return %ret : tensor<256xf16>
}

// -----
// CHECK-LABEL: func.func @ldexp_exp_f32_f16(
// CHECK-SAME: %[[ARG0:[^:]*]]: tensor<256xf32>,
// CHECK-SAME: %[[ARG1:[^:]*]]: tensor<256xf16>)
// CHECK: %[[CST0:[^ ]*]] = arith.constant 0.000000e+00 : f16
// CHECK: %[[CST_NEG:[^ ]*]] = arith.constant -3.174400e+04 : f16
// CHECK: %[[CST_LN2:[^ ]*]] = arith.constant 0.693147182 : f32
// CHECK: %[[CAST_TO_F32:[^ ]*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%[[ARG1]] : tensor<256xf16>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[MUL_LN2:[^ ]*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[CAST_TO_F32]], %[[CST_LN2]] : tensor<256xf32>, f32) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[EXP:[^ ]*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[MUL_LN2]] : tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[CAST_BACK_TO_F16:[^ ]*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%[[EXP]] : tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf16>) -> tensor<256xf16>
// CHECK: %[[NAN_CMP:[^ ]*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[ARG1]], %[[ARG1]] : tensor<256xf16>, tensor<256xf16>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[NOT_NAN:[^ ]*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[NAN_CMP]] : tensor<256xi1>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[SEL_NAN:[^ ]*]] = hfusion.select ins(%[[NOT_NAN]], %[[ARG1]], %[[CAST_BACK_TO_F16]] : tensor<256xi1>, tensor<256xf16>, tensor<256xf16>) outs(%{{[^ ]*}} : tensor<256xf16>) -> tensor<256xf16>
// CHECK: %[[FILL_NEG:[^ ]*]] = linalg.fill ins(%[[CST_NEG]] : f16) outs(%{{[^ ]*}} : tensor<256xf16>) -> tensor<256xf16>
// CHECK: %[[CMP_NEG:[^ ]*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[ARG1]], %[[FILL_NEG]] : tensor<256xf16>, tensor<256xf16>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[SEL_NEG:[^ ]*]] = hfusion.select ins(%[[CMP_NEG]], %[[ARG1]], %[[SEL_NAN]] : tensor<256xi1>, tensor<256xf16>, tensor<256xf16>) outs(%{{[^ ]*}} : tensor<256xf16>) -> tensor<256xf16>
// CHECK: %[[ABS:[^ ]*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[ARG1]] : tensor<256xf16>) outs(%{{[^ ]*}} : tensor<256xf16>) -> tensor<256xf16>
// CHECK: %[[CMP_INF:[^ ]*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[ABS]], %[[FILL_NEG]] : tensor<256xf16>, tensor<256xf16>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[NOT_NEG:[^ ]*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[CMP_NEG]] : tensor<256xi1>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[AND_COND:[^ ]*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[NOT_NEG]], %[[CMP_INF]] : tensor<256xi1>, tensor<256xi1>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[FILL_ZERO:[^ ]*]] = linalg.fill ins(%[[CST0]] : f16) outs(%{{[^ ]*}} : tensor<256xf16>) -> tensor<256xf16>
// CHECK: %[[SEL_INF:[^ ]*]] = hfusion.select ins(%[[AND_COND]], %[[FILL_ZERO]], %[[SEL_NEG]] : tensor<256xi1>, tensor<256xf16>, tensor<256xf16>) outs(%{{[^ ]*}} : tensor<256xf16>) -> tensor<256xf16>
// CHECK: %[[RES:[^ ]*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ARG0]], %[[SEL_INF]] : tensor<256xf32>, tensor<256xf16>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: return %[[RES]] : tensor<256xf32>
func.func @ldexp_exp_f32_f16(%arg0: tensor<256xf32>, %arg1: tensor<256xf16>) -> (tensor<256xf32>) {
    %0 = tensor.empty() : tensor<256xf32>
    %ret = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ldexp>} ins(%arg0, %arg1 : tensor<256xf32>, tensor<256xf16>) outs(%0 : tensor<256xf32>) -> tensor<256xf32>
    return %ret : tensor<256xf32>
}

// -----
// CHECK-LABEL: func.func @ldexp_exp_f16_f32(
// CHECK-SAME: %[[ARG0:[^:]*]]: tensor<256xf16>,
// CHECK-SAME: %[[ARG1:[^:]*]]: tensor<256xf32>)
// CHECK: %[[CST0:[^ ]*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[CST_NEG:[^ ]*]] = arith.constant -2.13909504E+9 : f32
// CHECK: %[[CST_LN2:[^ ]*]] = arith.constant 0.693147182 : f32
// CHECK: %[[MUL_LN2:[^ ]*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ARG1]], %[[CST_LN2]] : tensor<256xf32>, f32) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[EXP:[^ ]*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[MUL_LN2]] : tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[NAN_CMP:[^ ]*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[ARG1]], %[[ARG1]] : tensor<256xf32>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[NOT_NAN:[^ ]*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[NAN_CMP]] : tensor<256xi1>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[SEL_NAN:[^ ]*]] = hfusion.select ins(%[[NOT_NAN]], %[[ARG1]], %[[EXP]] : tensor<256xi1>, tensor<256xf32>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[FILL_NEG:[^ ]*]] = linalg.fill ins(%[[CST_NEG]] : f32) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[CMP_NEG:[^ ]*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[ARG1]], %[[FILL_NEG]] : tensor<256xf32>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[SEL_NEG:[^ ]*]] = hfusion.select ins(%[[CMP_NEG]], %[[ARG1]], %[[SEL_NAN]] : tensor<256xi1>, tensor<256xf32>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[ABS:[^ ]*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[ARG1]] : tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[CMP_INF:[^ ]*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[ABS]], %[[FILL_NEG]] : tensor<256xf32>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[NOT_NEG:[^ ]*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[CMP_NEG]] : tensor<256xi1>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[AND_COND:[^ ]*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[NOT_NEG]], %[[CMP_INF]] : tensor<256xi1>, tensor<256xi1>) outs(%{{[^ ]*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[FILL_ZERO:[^ ]*]] = linalg.fill ins(%[[CST0]] : f32) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[SEL_INF:[^ ]*]] = hfusion.select ins(%[[AND_COND]], %[[FILL_ZERO]], %[[SEL_NEG]] : tensor<256xi1>, tensor<256xf32>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[RES:[^ ]*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ARG0]], %[[SEL_INF]] : tensor<256xf16>, tensor<256xf32>) outs(%{{[^ ]*}} : tensor<256xf16>) -> tensor<256xf16>
// CHECK: return %[[RES]] : tensor<256xf16>
func.func @ldexp_exp_f16_f32(%arg0: tensor<256xf16>, %arg1: tensor<256xf32>) -> (tensor<256xf16>) {
    %0 = tensor.empty() : tensor<256xf16>
    %ret = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ldexp>} ins(%arg0, %arg1 : tensor<256xf16>, tensor<256xf32>) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
    return %ret : tensor<256xf16>
}

// -----
// CHECK-LABEL: func.func @test_insert_slice_i1_misalignment
// CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f16
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<25xf16>
// CHECK: %[[CAST0:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<25xi1>) outs(%[[EMPTY0]] : tensor<25xf16>) -> tensor<25xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[CAST1:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<1024xi1>) outs(%[[EMPTY1]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: %[[INSERT:.*]] = tensor.insert_slice %[[CAST0]] into %[[CAST1]][%arg2] [25] [1] : tensor<25xf16> into tensor<1024xf16>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<1024xi1>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<1024xi1>
// CHECK: %[[CMP:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[INSERT]], %[[CST]] : tensor<1024xf16>, f16) outs(%[[EMPTY3]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: %[[VNOT:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[CMP]] : tensor<1024xi1>) outs(%[[EMPTY2]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: return %[[VNOT]] : tensor<1024xi1>
func.func @test_insert_slice_i1_misalignment(%arg0: tensor<25xi1>, %arg1: tensor<1024xi1>, %args2: index) -> (tensor<1024xi1>) {
  %ret = tensor.insert_slice %arg0 into %arg1[%args2] [25] [1] : tensor<25xi1> into tensor<1024xi1>
  return %ret : tensor<1024xi1>
}

// -----
// CHECK-LABEL: func.func @test_insert_slice_i8_misalignment
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<25xf16>
// CHECK: %[[CAST0:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<25xi8>) outs(%[[EMPTY0]] : tensor<25xf16>) -> tensor<25xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[CAST1:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<1024xi8>) outs(%[[EMPTY1]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: %[[INSERT:.*]] = tensor.insert_slice %[[CAST0]] into %[[CAST1]][%arg2] [25] [1] : tensor<25xf16> into tensor<1024xf16>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<1024xi8>
// CHECK: %[[CAST2:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins(%[[INSERT]] : tensor<1024xf16>) outs(%[[EMPTY2]] : tensor<1024xi8>) -> tensor<1024xi8>
// CHECK: return %[[CAST2]] : tensor<1024xi8>
func.func @test_insert_slice_i8_misalignment(%arg0: tensor<25xi8>, %arg1: tensor<1024xi8>, %args2: index) -> (tensor<1024xi8>) {
  %ret = tensor.insert_slice %arg0 into %arg1[%args2] [25] [1] : tensor<25xi8> into tensor<1024xi8>
  return %ret : tensor<1024xi8>
}

// -----
// CHECK-LABEL: func.func @test_insert_slice_i1_stride_2
// CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f16
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<32xf16>
// CHECK: %[[CAST0:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<32xi1>) outs(%[[EMPTY0]] : tensor<32xf16>) -> tensor<32xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[CAST1:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<1024xi1>) outs(%[[EMPTY1]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: %[[INSERT:.*]] = tensor.insert_slice %[[CAST0]] into %[[CAST1]][%arg2] [32] [2] : tensor<32xf16> into tensor<1024xf16>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<1024xi1>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<1024xi1>
// CHECK: %[[CMP:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[INSERT]], %[[CST]] : tensor<1024xf16>, f16) outs(%[[EMPTY3]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: %[[VNOT:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[CMP]] : tensor<1024xi1>) outs(%[[EMPTY2]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: return %[[VNOT]] : tensor<1024xi1>
func.func @test_insert_slice_i1_stride_2(%arg0: tensor<32xi1>, %arg1: tensor<1024xi1>, %args2: index) -> (tensor<1024xi1>) {
  %ret = tensor.insert_slice %arg0 into %arg1[%args2] [32] [2] : tensor<32xi1> into tensor<1024xi1>
  return %ret : tensor<1024xi1>
}

// -----
// CHECK-LABEL: func.func @test_insert_slice_i8_stride_2
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<32xf16>
// CHECK: %[[CAST0:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<32xi8>) outs(%[[EMPTY0]] : tensor<32xf16>) -> tensor<32xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[CAST1:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<1024xi8>) outs(%[[EMPTY1]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: %[[INSERT:.*]] = tensor.insert_slice %[[CAST0]] into %[[CAST1]][%arg2] [32] [2] : tensor<32xf16> into tensor<1024xf16>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<1024xi8>
// CHECK: %[[CAST2:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins(%[[INSERT]] : tensor<1024xf16>) outs(%[[EMPTY2]] : tensor<1024xi8>) -> tensor<1024xi8>
// CHECK: return %[[CAST2]] : tensor<1024xi8>
func.func @test_insert_slice_i8_stride_2(%arg0: tensor<32xi8>, %arg1: tensor<1024xi8>, %args2: index) -> (tensor<1024xi8>) {
  %ret = tensor.insert_slice %arg0 into %arg1[%args2] [32] [2] : tensor<32xi8> into tensor<1024xi8>
  return %ret : tensor<1024xi8>
}

// -----
// CHECK-LABEL: func.func @test_normalize_shift_left_i8
// CHECK-SAME:    (%[[VAL_0:.*]]: tensor<16x32xi8>, %[[VAL_1:.*]]: tensor<16x32xi8>, %[[VAL_2:.*]]: tensor<16x32xi8>) -> tensor<16x32xi8> {
// CHECK-DAG:     %[[BITS:.*]] = arith.constant 32 : i32
// CHECK-DAG:     %[[BASE:.*]] = arith.constant 2 : i32
// CHECK-DAG:     %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:         %[[LHS:.*]] = hfusion.cast {{.*}} ins(%{{.*}} : tensor<16x32xf16>) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[RHS:.*]] = hfusion.cast {{.*}} ins(%{{.*}} : tensor<16x32xf16>) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[GE0:.*]] = hfusion.compare {compare_fn = #{{.*}}<veq>} ins(%{{.*}}, %[[RHS]] : tensor<16x32xi32>, tensor<16x32xi32>) {{.*}} -> tensor<16x32xi1>
// CHECK:         %[[LT_NE:.*]] = hfusion.compare {compare_fn = #{{.*}}<veq>} ins(%{{.*}}, %[[RHS]] : tensor<16x32xi32>, tensor<16x32xi32>) {{.*}} -> tensor<16x32xi1>
// CHECK:         %[[LT:.*]] = hfusion.elemwise_unary {fun = #{{.*}}<vnot>} ins(%[[LT_NE]] : tensor<16x32xi1>) {{.*}} -> tensor<16x32xi1>
// CHECK:         %[[MASK:.*]] = hfusion.elemwise_binary {fun = #{{.*}}<vand>} ins(%[[GE0]], %[[LT]] : tensor<16x32xi1>, tensor<16x32xi1>) {{.*}} -> tensor<16x32xi1>
// CHECK:         %[[CLAMP:.*]] = hfusion.select ins(%[[MASK]], %[[RHS]], %[[ZERO]] : tensor<16x32xi1>, tensor<16x32xi32>, i32) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[POW:.*]] = hfusion.elemwise_binary {fun = #{{.*}}<powi>} ins(%[[BASE]], %[[CLAMP]] : i32, tensor<16x32xi32>) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[MUL:.*]] = linalg.elemwise_binary {fun = #{{.*}}<mul>} ins(%[[LHS]], %[[POW]] : tensor<16x32xi32>, tensor<16x32xi32>) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[SEL:.*]] = hfusion.select ins(%[[MASK]], %[[MUL]], %[[ZERO]] : tensor<16x32xi1>, tensor<16x32xi32>, i32) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[RES:.*]] = hfusion.cast {{.*}} ins(%[[SEL]] : tensor<16x32xi32>) {{.*}} -> tensor<16x32xi8>
// CHECK:         return %[[RES]] : tensor<16x32xi8>
func.func @test_normalize_shift_left_i8(%arg0: tensor<16x32xi8>, %arg1: tensor<16x32xi8>, %dst : tensor<16x32xi8>) -> (tensor<16x32xi8>) {
  %ret = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shli>} ins(%arg0, %arg1 : tensor<16x32xi8>, tensor<16x32xi8>) outs(%dst : tensor<16x32xi8>) -> tensor<16x32xi8>
  annotation.mark %ret {soft_simd_mode = true} : tensor<16x32xi8>
  return %ret : tensor<16x32xi8>
}

// -----
// CHECK-LABEL: func.func @test_normalize_shift_left_i16
// CHECK-SAME:    (%[[VAL_0:.*]]: tensor<16x32xi16>, %[[VAL_1:.*]]: tensor<16x32xi16>, %[[VAL_2:.*]]: tensor<16x32xi16>) -> tensor<16x32xi16> {
// CHECK-DAG:     %[[BITS:.*]] = arith.constant 32 : i32
// CHECK-DAG:     %[[BASE:.*]] = arith.constant 2 : i32
// CHECK-DAG:     %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:         %[[LHS:.*]] = hfusion.cast {{.*}} ins(%{{.*}} : tensor<16x32xf32>) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[RHS:.*]] = hfusion.cast {{.*}} ins(%{{.*}} : tensor<16x32xf32>) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[GE0:.*]] = hfusion.compare {compare_fn = #{{.*}}<veq>} ins(%{{.*}}, %[[RHS]] : tensor<16x32xi32>, tensor<16x32xi32>) {{.*}} -> tensor<16x32xi1>
// CHECK:         %[[LT_NE:.*]] = hfusion.compare {compare_fn = #{{.*}}<veq>} ins(%{{.*}}, %[[RHS]] : tensor<16x32xi32>, tensor<16x32xi32>) {{.*}} -> tensor<16x32xi1>
// CHECK:         %[[LT:.*]] = hfusion.elemwise_unary {fun = #{{.*}}<vnot>} ins(%[[LT_NE]] : tensor<16x32xi1>) {{.*}} -> tensor<16x32xi1>
// CHECK:         %[[MASK:.*]] = hfusion.elemwise_binary {fun = #{{.*}}<vand>} ins(%[[GE0]], %[[LT]] : tensor<16x32xi1>, tensor<16x32xi1>) {{.*}} -> tensor<16x32xi1>
// CHECK:         %[[CLAMP:.*]] = hfusion.select ins(%[[MASK]], %[[RHS]], %[[ZERO]] : tensor<16x32xi1>, tensor<16x32xi32>, i32) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[POW:.*]] = hfusion.elemwise_binary {fun = #{{.*}}<powi>} ins(%[[BASE]], %[[CLAMP]] : i32, tensor<16x32xi32>) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[MUL:.*]] = linalg.elemwise_binary {fun = #{{.*}}<mul>} ins(%[[LHS]], %[[POW]] : tensor<16x32xi32>, tensor<16x32xi32>) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[SEL:.*]] = hfusion.select ins(%[[MASK]], %[[MUL]], %[[ZERO]] : tensor<16x32xi1>, tensor<16x32xi32>, i32) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[RES:.*]] = hfusion.cast {{.*}} ins(%[[SEL]] : tensor<16x32xi32>) {{.*}} -> tensor<16x32xi16>
// CHECK:         return %[[RES]] : tensor<16x32xi16>

func.func @test_normalize_shift_left_i16(%arg0: tensor<16x32xi16>, %arg1: tensor<16x32xi16>, %dst : tensor<16x32xi16>) -> (tensor<16x32xi16>) {
  %ret = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shli>} ins(%arg0, %arg1 : tensor<16x32xi16>, tensor<16x32xi16>) outs(%dst : tensor<16x32xi16>) -> tensor<16x32xi16>
  annotation.mark %ret {soft_simd_mode = true} : tensor<16x32xi16>
  return %ret : tensor<16x32xi16>
}

// -----
// CHECK-LABEL: func.func @test_normalize_shift_left_i32
// CHECK-SAME:    (%[[VAL_0:.*]]: tensor<16x32xi32>, %[[VAL_1:.*]]: tensor<16x32xi32>, %[[VAL_2:.*]]: tensor<16x32xi32>) -> tensor<16x32xi32> {
// CHECK-DAG:     %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-DAG:     %[[BASE:.*]] = arith.constant 2 : i32
// CHECK-DAG:     %[[BITS:.*]] = arith.constant 32 : i32
// CHECK:         %[[GE0:.*]] = hfusion.compare {compare_fn = #{{.*}}<veq>} ins(%{{.*}}, %[[VAL_1]] : tensor<16x32xi32>, tensor<16x32xi32>) {{.*}} -> tensor<16x32xi1>
// CHECK:         %[[LT_NE:.*]] = hfusion.compare {compare_fn = #{{.*}}<veq>} ins(%{{.*}}, %[[VAL_1]] : tensor<16x32xi32>, tensor<16x32xi32>) {{.*}} -> tensor<16x32xi1>
// CHECK:         %[[LT:.*]] = hfusion.elemwise_unary {fun = #{{.*}}<vnot>} ins(%[[LT_NE]] : tensor<16x32xi1>) {{.*}} -> tensor<16x32xi1>
// CHECK:         %[[MASK:.*]] = hfusion.elemwise_binary {fun = #{{.*}}<vand>} ins(%[[GE0]], %[[LT]] : tensor<16x32xi1>, tensor<16x32xi1>) {{.*}} -> tensor<16x32xi1>
// CHECK:         %[[CLAMP:.*]] = hfusion.select ins(%[[MASK]], %[[VAL_1]], %[[ZERO]] : tensor<16x32xi1>, tensor<16x32xi32>, i32) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[POW:.*]] = hfusion.elemwise_binary {fun = #{{.*}}<powi>} ins(%[[BASE]], %[[CLAMP]] : i32, tensor<16x32xi32>) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[MUL:.*]] = linalg.elemwise_binary {fun = #{{.*}}<mul>} ins(%[[VAL_0]], %[[POW]] : tensor<16x32xi32>, tensor<16x32xi32>) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[RES:.*]] = hfusion.select ins(%[[MASK]], %[[MUL]], %[[ZERO]] : tensor<16x32xi1>, tensor<16x32xi32>, i32) {{.*}} -> tensor<16x32xi32>
// CHECK:         return %[[RES]] : tensor<16x32xi32>
func.func @test_normalize_shift_left_i32(%arg0: tensor<16x32xi32>, %arg1: tensor<16x32xi32>, %dst : tensor<16x32xi32>) -> (tensor<16x32xi32>) {
  %ret = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shli>} ins(%arg0, %arg1 : tensor<16x32xi32>, tensor<16x32xi32>) outs(%dst : tensor<16x32xi32>) -> tensor<16x32xi32>
  annotation.mark %ret {soft_simd_mode = true} : tensor<16x32xi32>
  return %ret : tensor<16x32xi32>
}

// -----
// CHECK-LABEL: func.func @test_normalize_shift_right_i8
// CHECK-SAME:    (%[[VAL_0:.*]]: tensor<16x32xi8>, %[[VAL_1:.*]]: tensor<16x32xi8>, %[[VAL_2:.*]]: tensor<16x32xi8>) -> tensor<16x32xi8> {
// CHECK-DAG:     %[[NEG1:.*]] = arith.constant -1 : i32
// CHECK-DAG:     %[[BITS:.*]] = arith.constant 31 : i32
// CHECK-DAG:     %[[BASE:.*]] = arith.constant 2 : i32
// CHECK-DAG:     %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:         %[[LHS:.*]] = hfusion.cast {{.*}} ins(%{{.*}} : tensor<16x32xf16>) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[RHS:.*]] = hfusion.cast {{.*}} ins(%{{.*}} : tensor<16x32xf16>) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[GE0:.*]] = hfusion.compare {compare_fn = #{{.*}}<veq>} ins(%{{.*}}, %[[RHS]] : tensor<16x32xi32>, tensor<16x32xi32>) {{.*}} -> tensor<16x32xi1>
// CHECK:         %[[LT_NE:.*]] = hfusion.compare {compare_fn = #{{.*}}<veq>} ins(%{{.*}}, %[[RHS]] : tensor<16x32xi32>, tensor<16x32xi32>) {{.*}} -> tensor<16x32xi1>
// CHECK:         %[[LT:.*]] = hfusion.elemwise_unary {fun = #{{.*}}<vnot>} ins(%[[LT_NE]] : tensor<16x32xi1>) {{.*}} -> tensor<16x32xi1>
// CHECK:         %[[MASK:.*]] = hfusion.elemwise_binary {fun = #{{.*}}<vand>} ins(%[[GE0]], %[[LT]] : tensor<16x32xi1>, tensor<16x32xi1>) {{.*}} -> tensor<16x32xi1>
// CHECK:         %[[CLAMP:.*]] = hfusion.select ins(%[[MASK]], %[[RHS]], %[[ZERO]] : tensor<16x32xi1>, tensor<16x32xi32>, i32) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[POW:.*]] = hfusion.elemwise_binary {fun = #{{.*}}<powi>} ins(%[[BASE]], %[[CLAMP]] : i32, tensor<16x32xi32>) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[LHS_F:.*]] = hfusion.cast {{.*}} ins(%[[LHS]] : tensor<16x32xi32>) {{.*}} -> tensor<16x32xf32>
// CHECK:         %[[POW_F:.*]] = hfusion.cast {{.*}} ins(%[[POW]] : tensor<16x32xi32>) {{.*}} -> tensor<16x32xf32>
// CHECK:         %[[DIV:.*]] = linalg.elemwise_binary {fun = #{{.*}}<div>} ins(%[[LHS_F]], %[[POW_F]] : tensor<16x32xf32>, tensor<16x32xf32>) {{.*}} -> tensor<16x32xf32>
// CHECK:         %[[FLOOR:.*]] = hfusion.cast {{.*}}<floor>{{.*}} ins(%[[DIV]] : tensor<16x32xf32>) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[POS:.*]] = hfusion.compare {compare_fn = #{{.*}}<veq>} ins(%{{.*}}, %[[LHS]] : tensor<16x32xi32>, tensor<16x32xi32>) {{.*}} -> tensor<16x32xi1>
// CHECK:         %[[OOB:.*]] = hfusion.select ins(%[[POS]], %[[ZERO]], %[[NEG1]] : tensor<16x32xi1>, i32, i32) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[SEL:.*]] = hfusion.select ins(%[[MASK]], %[[FLOOR]], %[[OOB]] : tensor<16x32xi1>, tensor<16x32xi32>, tensor<16x32xi32>) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[RES:.*]] = hfusion.cast {{.*}} ins(%[[SEL]] : tensor<16x32xi32>) {{.*}} -> tensor<16x32xi8>
// CHECK:         return %[[RES]] : tensor<16x32xi8>
func.func @test_normalize_shift_right_i8(%arg0: tensor<16x32xi8>, %arg1: tensor<16x32xi8>, %dst : tensor<16x32xi8>) -> (tensor<16x32xi8>) {
  %ret = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%arg0, %arg1 : tensor<16x32xi8>, tensor<16x32xi8>) outs(%dst : tensor<16x32xi8>) -> tensor<16x32xi8>
  annotation.mark %ret {soft_simd_mode = true} : tensor<16x32xi8>
  return %ret : tensor<16x32xi8>
}

// -----
// CHECK-LABEL: func.func @test_normalize_shift_right_i16
// CHECK-SAME:    (%[[VAL_0:.*]]: tensor<16x32xi16>, %[[VAL_1:.*]]: tensor<16x32xi16>, %[[VAL_2:.*]]: tensor<16x32xi16>) -> tensor<16x32xi16> {
// CHECK-DAG:     %[[NEG1:.*]] = arith.constant -1 : i32
// CHECK-DAG:     %[[BITS:.*]] = arith.constant 31 : i32
// CHECK-DAG:     %[[BASE:.*]] = arith.constant 2 : i32
// CHECK-DAG:     %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:         %[[LHS:.*]] = hfusion.cast {{.*}} ins(%{{.*}} : tensor<16x32xf32>) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[RHS:.*]] = hfusion.cast {{.*}} ins(%{{.*}} : tensor<16x32xf32>) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[GE0:.*]] = hfusion.compare {compare_fn = #{{.*}}<veq>} ins(%{{.*}}, %[[RHS]] : tensor<16x32xi32>, tensor<16x32xi32>) {{.*}} -> tensor<16x32xi1>
// CHECK:         %[[LT_NE:.*]] = hfusion.compare {compare_fn = #{{.*}}<veq>} ins(%{{.*}}, %[[RHS]] : tensor<16x32xi32>, tensor<16x32xi32>) {{.*}} -> tensor<16x32xi1>
// CHECK:         %[[LT:.*]] = hfusion.elemwise_unary {fun = #{{.*}}<vnot>} ins(%[[LT_NE]] : tensor<16x32xi1>) {{.*}} -> tensor<16x32xi1>
// CHECK:         %[[MASK:.*]] = hfusion.elemwise_binary {fun = #{{.*}}<vand>} ins(%[[GE0]], %[[LT]] : tensor<16x32xi1>, tensor<16x32xi1>) {{.*}} -> tensor<16x32xi1>
// CHECK:         %[[CLAMP:.*]] = hfusion.select ins(%[[MASK]], %[[RHS]], %[[ZERO]] : tensor<16x32xi1>, tensor<16x32xi32>, i32) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[POW:.*]] = hfusion.elemwise_binary {fun = #{{.*}}<powi>} ins(%[[BASE]], %[[CLAMP]] : i32, tensor<16x32xi32>) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[LHS_F:.*]] = hfusion.cast {{.*}} ins(%[[LHS]] : tensor<16x32xi32>) {{.*}} -> tensor<16x32xf32>
// CHECK:         %[[POW_F:.*]] = hfusion.cast {{.*}} ins(%[[POW]] : tensor<16x32xi32>) {{.*}} -> tensor<16x32xf32>
// CHECK:         %[[DIV:.*]] = linalg.elemwise_binary {fun = #{{.*}}<div>} ins(%[[LHS_F]], %[[POW_F]] : tensor<16x32xf32>, tensor<16x32xf32>) {{.*}} -> tensor<16x32xf32>
// CHECK:         %[[FLOOR:.*]] = hfusion.cast {{.*}}<floor>{{.*}} ins(%[[DIV]] : tensor<16x32xf32>) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[POS:.*]] = hfusion.compare {compare_fn = #{{.*}}<veq>} ins(%{{.*}}, %[[LHS]] : tensor<16x32xi32>, tensor<16x32xi32>) {{.*}} -> tensor<16x32xi1>
// CHECK:         %[[OOB:.*]] = hfusion.select ins(%[[POS]], %[[ZERO]], %[[NEG1]] : tensor<16x32xi1>, i32, i32) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[SEL:.*]] = hfusion.select ins(%[[MASK]], %[[FLOOR]], %[[OOB]] : tensor<16x32xi1>, tensor<16x32xi32>, tensor<16x32xi32>) {{.*}} -> tensor<16x32xi32>
// CHECK:         %[[RES:.*]] = hfusion.cast {{.*}} ins(%[[SEL]] : tensor<16x32xi32>) {{.*}} -> tensor<16x32xi16>
// CHECK:         return %[[RES]] : tensor<16x32xi16>
func.func @test_normalize_shift_right_i16(%arg0: tensor<16x32xi16>, %arg1: tensor<16x32xi16>, %dst : tensor<16x32xi16>) -> (tensor<16x32xi16>) {
  %ret = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%arg0, %arg1 : tensor<16x32xi16>, tensor<16x32xi16>) outs(%dst : tensor<16x32xi16>) -> tensor<16x32xi16>
  annotation.mark %ret {soft_simd_mode = true} : tensor<16x32xi16>
  return %ret : tensor<16x32xi16>
}

// -----
// CHECK-LABEL: func.func @test_normalize_shift_right_i32
// CHECK-SAME:    (%[[VAL_0:.*]]: tensor<16x32xi32>, %[[VAL_1:.*]]: tensor<16x32xi32>, %[[VAL_2:.*]]: tensor<16x32xi32>) -> tensor<16x32xi32> {
// CHECK:         %[[VAL_3:.*]] = hfusion.elemwise_binary {fun = #{{.*}}<shrsi>} ins(%[[VAL_0]], %[[VAL_1]] : tensor<16x32xi32>, tensor<16x32xi32>) outs(%[[VAL_2]] : tensor<16x32xi32>) -> tensor<16x32xi32>
// CHECK:         return %[[VAL_3]] : tensor<16x32xi32>
func.func @test_normalize_shift_right_i32(%arg0: tensor<16x32xi32>, %arg1: tensor<16x32xi32>, %dst : tensor<16x32xi32>) -> (tensor<16x32xi32>) {
  %ret = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%arg0, %arg1 : tensor<16x32xi32>, tensor<16x32xi32>) outs(%dst : tensor<16x32xi32>) -> tensor<16x32xi32>
  return %ret : tensor<16x32xi32>
}
