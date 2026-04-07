// RUN: bishengir-opt --hfusion-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_NormalizeCastLowering_cast_f32_to_i1
// CHECK: %[[arg4:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[arg2:.*]], %[[cst:.*]] : tensor<2x256x12x257xf32>, f32) outs(%[[arg5:.*]] : tensor<2x256x12x257xi1>) -> tensor<2x256x12x257xi1>
// ChECK: %[[arg6:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[arg4:.*]] : tensor<2x256x12x257xi1>) outs(%[[arg3:.*]] : tensor<2x256x12x257xi1>) -> tensor<2x256x12x257xi1>
func.func @test_NormalizeCastLowering_cast_f32_to_i1(%arg0: tensor<2x256x12x257xf32>) -> tensor<2x256x12x257xi1> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %1 = tensor.empty() : tensor<2x256x12x257xi1>
  %2 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<2x256x12x257xf32>) outs(%1 : tensor<2x256x12x257xi1>) -> tensor<2x256x12x257xi1>
  return %2 : tensor<2x256x12x257xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_cast_f16_to_i1
// CHECK: %[[arg4:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[arg2:.*]], %[[cst:.*]] : tensor<2x256x12x257xf16>, f16) outs(%[[arg5:.*]] : tensor<2x256x12x257xi1>) -> tensor<2x256x12x257xi1>
// ChECK: %[[arg6:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[arg4:.*]] : tensor<2x256x12x257xi1>) outs(%[[arg3:.*]] : tensor<2x256x12x257xi1>) -> tensor<2x256x12x257xi1>
func.func @test_NormalizeCastLowering_cast_f16_to_i1(%arg0: tensor<2x256x12x257xf16>) -> tensor<2x256x12x257xi1> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %1 = tensor.empty() : tensor<2x256x12x257xi1>
  %2 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<2x256x12x257xf16>) outs(%1 : tensor<2x256x12x257xi1>) -> tensor<2x256x12x257xi1>
  return %2 : tensor<2x256x12x257xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_cast_bf16_to_i1
// CHECK-SAME: (%[[arg0:.*]]: tensor<16xbf16>)
// CHECK: %[[arg1:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[arg2:.*]] = hfusion.cast {{.*}} ins(%[[arg0]] : tensor<16xbf16>) outs(%[[arg1]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[arg3:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[arg5:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[arg4:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[arg2]], %[[cst:.*]] : tensor<16xf32>, f32) outs(%[[arg5]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[arg6:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[arg4]] : tensor<16xi1>) outs(%[[arg3]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: return %[[arg6]]
func.func @test_NormalizeCastLowering_cast_bf16_to_i1(%arg0: tensor<16xbf16>) -> tensor<16xi1> {
  %0 = tensor.empty() : tensor<16xi1>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<16xbf16>) outs(%0 : tensor<16xi1>) -> tensor<16xi1>
  return %1 : tensor<16xi1>
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_cast_f32_to_i16
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xf32>) outs({{.*}} : tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xi32>) outs({{.*}} : tensor<4x4xi16>) -> tensor<4x4xi16>
func.func @test_NormalizeCastLowering_cast_f32_to_i16(%arg0: tensor<4x4xf32>) -> tensor<4x4xi16> {
  %0 = tensor.empty() : tensor<4x4xi16>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} ins(%arg0 : tensor<4x4xf32>) outs(%0 : tensor<4x4xi16>) -> tensor<4x4xi16>
  return %1 : tensor<4x4xi16>
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_cast_f32_to_i16_with_overflow_mode(
// CHECK: %[[OUT:.+]] = tensor.empty() : tensor<16xi16>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<16xf32>) outs(%[[OUT]] : tensor<16xi16>) -> tensor<16xi16>
// CHECK: return %[[CAST:.+]] : tensor<16xi16>
func.func @test_NormalizeCastLowering_cast_f32_to_i16_with_overflow_mode(%arg0: tensor<16xf32>) -> tensor<16xi16> {
  %0 = tensor.empty() : tensor<16xi16>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xf32>) outs(%0 : tensor<16xi16>) -> tensor<16xi16>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi16>
  return %1 : tensor<16xi16>
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_cast_f32_to_i8_with_overflow_mode(
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<16xf32>) outs({{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<16xf16>) outs({{.*}} : tensor<16xi8>) -> tensor<16xi8>
func.func @test_NormalizeCastLowering_cast_f32_to_i8_with_overflow_mode(%arg0: tensor<16xf32>) -> tensor<16xi8> {
  %0 = tensor.empty() : tensor<16xi8>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xf32>) outs(%0 : tensor<16xi8>) -> tensor<16xi8>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi8>
  return %1 : tensor<16xi8>
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_cast_f16_to_i8_with_overflow_mode(
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<16xf16>) outs({{.*}} : tensor<16xi8>) -> tensor<16xi8>
func.func @test_NormalizeCastLowering_cast_f16_to_i8_with_overflow_mode(%arg0: tensor<16xf16>) -> tensor<16xi8> {
  %0 = tensor.empty() : tensor<16xi8>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xf16>) outs(%0 : tensor<16xi8>) -> tensor<16xi8>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi8>
  return %1 : tensor<16xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_cast_op_tensor_i64_to_f16
// CHECK: %[[ZERO:.*]] = tensor.empty() : tensor<23xf16>
// CHECK: %[[ONE:.*]] = tensor.empty() : tensor<23xf32>
// CHECK: %[[TWO:.*]] = hfusion.cast {{.*}} ins(%[[arg0:.*]] : tensor<23xi64>) outs(%[[ONE:.*]]: tensor<23xf32>) -> tensor<23xf32>
// CHECK: %[[THREE:.*]] = hfusion.cast {{.*}} ins(%[[TWO:.*]] : tensor<23xf32>) outs(%[[ZERO:.*]] : tensor<23xf16>) -> tensor<23xf16>
func.func @test_NormalizeCastLowering_cast_op_tensor_i64_to_f16(%arg0: tensor<23xi64>, %arg1: tensor<f16>) -> tensor<23xf16> attributes {hacc.entry} {
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

// CHECK-LABEL: @test_NormalizeCastLowering_lowering_cast_i64_to_bf16(
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xi64>) outs({{.*}} : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xf32>) outs({{.*}} : tensor<4x4xbf16>) -> tensor<4x4xbf16>
func.func @test_NormalizeCastLowering_lowering_cast_i64_to_bf16(%arg0: tensor<4x4xi64>) -> tensor<4x4xbf16> {
  %0 = tensor.empty() : tensor<4x4xbf16>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<4x4xi64>) outs(%0 : tensor<4x4xbf16>) -> tensor<4x4xbf16>
  return %1 : tensor<4x4xbf16>
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_cast_i8_to_bf16
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xi8>) outs({{.*}} : tensor<4x4xf16>) -> tensor<4x4xf16>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xf16>) outs({{.*}} : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xf32>) outs({{.*}} : tensor<4x4xbf16>) -> tensor<4x4xbf16>
func.func @test_NormalizeCastLowering_cast_i8_to_bf16(%arg0: tensor<4x4xi8>) -> tensor<4x4xbf16> {
  %0 = tensor.empty() : tensor<4x4xbf16>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<4x4xi8>) outs(%0 : tensor<4x4xbf16>) -> tensor<4x4xbf16>
  return %1 : tensor<4x4xbf16>
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_i1_cast_f32(
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x256xi1>) outs({{.*}} : tensor<4x256xf16>) -> tensor<4x256xf16>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x256xf16>) outs({{.*}} : tensor<4x256xf32>) -> tensor<4x256xf32>
func.func @test_NormalizeCastLowering_i1_cast_f32(%arg0: tensor<4x256xi1>) -> tensor<4x256xf32> {
  %0 = tensor.empty() : tensor<4x256xf32>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x256xi1>) outs(%0 : tensor<4x256xf32>) -> tensor<4x256xf32>
  return %1 : tensor<4x256xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_i1_cast_i64
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<200x200xf16>
// CHECK: %[[CAST_F16:.*]] = hfusion.cast {{.*}} ins(%[[ARG0:.*]] : tensor<200x200xi1>) outs(%[[EMPTY0:.*]] : tensor<200x200xf16>) -> tensor<200x200xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<200x200xf32>
// CHECK: %[[CAST_F32:.*]] = hfusion.cast {{.*}} ins(%[[CAST_F16:.*]] : tensor<200x200xf16>) outs(%[[EMPTY1:.*]] : tensor<200x200xf32>) -> tensor<200x200xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<200x200xi64>
// CHECK: %[[CAST_I64:.*]] = hfusion.cast {{.*}} ins(%[[CAST_F32:.*]] : tensor<200x200xf32>) outs(%[[EMPTY2:.*]] : tensor<200x200xi64>) -> tensor<200x200xi64>
func.func @test_NormalizeCastLowering_i1_cast_i64(%arg0: tensor<200x200xi1>) -> tensor<200x200xi64>{
  %0 = tensor.empty() : tensor<200x200xi64>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<200x200xi1>) outs(%0 : tensor<200x200xi64>) -> tensor<200x200xi64>
  return %1 : tensor<200x200xi64>
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_i16_cast_i32(
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xi16>) outs({{.*}} : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xf32>) outs({{.*}} : tensor<4x4xi32>) -> tensor<4x4xi32>
func.func @test_NormalizeCastLowering_i16_cast_i32(%arg0: tensor<4x4xi16>) -> tensor<4x4xi32> {
  %0 = tensor.empty() : tensor<4x4xi32>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<4x4xi16>) outs(%0 : tensor<4x4xi32>) -> tensor<4x4xi32>
  return %1 : tensor<4x4xi32>
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_i8_cast_i32(
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xi8>) outs({{.*}} : tensor<4x4xf16>) -> tensor<4x4xf16>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xf16>) outs({{.*}} : tensor<4x4xi32>) -> tensor<4x4xi32>
func.func @test_NormalizeCastLowering_i8_cast_i32(%arg0: tensor<4x4xi8>) -> tensor<4x4xi32> {
  %0 = tensor.empty() : tensor<4x4xi32>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<4x4xi8>) outs(%0 : tensor<4x4xi32>) -> tensor<4x4xi32>
  return %1 : tensor<4x4xi32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_i64_cast_i1
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<20x20xi64>) outs({{.*}} : tensor<20x20xf32>) -> tensor<20x20xf32>
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins({{.*}}, %[[cst_0:.*]] : tensor<20x20xf32>, f32) outs({{.*}} : tensor<20x20xi1>) -> tensor<20x20xi1>
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[Veq:.*]] : tensor<20x20xi1>)
func.func @test_NormalizeCastLowering_i64_cast_i1(%arg0: tensor<20x20xi64>) -> tensor<20x20xi1>{
  %0 = tensor.empty() : tensor<20x20xi1>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<20x20xi64>) outs(%0 : tensor<20x20xi1>) -> tensor<20x20xi1>
  return %1 : tensor<20x20xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_i8_cast_i1
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x256xi8>) outs({{.*}} : tensor<4x256xf16>) -> tensor<4x256xf16>
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins({{.*}}, %[[cst_0:.*]] : tensor<4x256xf16>, f16) outs({{.*}} : tensor<4x256xi1>) -> tensor<4x256xi1>
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[Veq:.*]] : tensor<4x256xi1>)
func.func @test_NormalizeCastLowering_i8_cast_i1(%arg0: tensor<4x256xi8>) -> tensor<4x256xi1>{
  %0 = tensor.empty() : tensor<4x256xi1>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x256xi8>) outs(%0 : tensor<4x256xi1>) -> tensor<4x256xi1>
  return %1 : tensor<4x256xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_i32_cast_i1
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<8xi32>) outs({{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins({{.*}}, %[[cst_0:.*]] : tensor<8xf32>, f32) outs({{.*}} : tensor<8xi1>) -> tensor<8xi1>
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[Veq:.*]] : tensor<8xi1>)
func.func @test_NormalizeCastLowering_i32_cast_i1(%arg0: tensor<8xi32>) -> tensor<8xi1> {
  %0 = tensor.empty() : tensor<8xi1>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<8xi32>) outs(%0 : tensor<8xi1>) -> tensor<8xi1>
  return %1 : tensor<8xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_i8_cast_i64
// CHECK: hfusion.cast {{.*}} ins(%arg0 : tensor<8xi8>) outs({{.*}} : tensor<8xf16>) -> tensor<8xf16>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<8xf16>) outs({{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<8xf32>) outs({{.*}} : tensor<8xi64>) -> tensor<8xi64>
func.func @test_NormalizeCastLowering_i8_cast_i64(%arg0: tensor<8xi8>) -> tensor<8xi64> {
  %0 = tensor.empty() : tensor<8xi64>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<8xi8>) outs(%0 : tensor<8xi64>) -> tensor<8xi64>
  return %1 : tensor<8xi64>
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_cast_i64_to_i16
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xi64>) outs({{.*}} : tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xi32>) outs({{.*}} : tensor<4x4xi16>) -> tensor<4x4xi16>
func.func @test_NormalizeCastLowering_cast_i64_to_i16(%arg0: tensor<4x4xi64>) -> tensor<4x4xi16> {
  %0 = tensor.empty() : tensor<4x4xi16>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x4xi64>) outs(%0 : tensor<4x4xi16>) -> tensor<4x4xi16>
  return %1 : tensor<4x4xi16>
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_cast_i64_to_i8
// CHECK: %[[TMP1:.*]] = hfusion.cast {{.*}} ins(%arg0 : tensor<4x4xi64>) outs({{.*}} : tensor<4x4xi32>)
// CHECK: hfusion.cast {{.*}} ins(%[[TMP1]] : tensor<4x4xi32>) outs({{.*}} : tensor<4x4xi8>)
func.func @test_NormalizeCastLowering_cast_i64_to_i8(%arg0: tensor<4x4xi64>) -> tensor<4x4xi8> {
  %0 = tensor.empty() : tensor<4x4xi8>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x4xi64>) outs(%0 : tensor<4x4xi8>) -> tensor<4x4xi8>
  return %1 : tensor<4x4xi8>
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_cast_i64_to_i32_with_overflow_mode(
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<16xi64>) outs({{.*}} : tensor<16xi32>) -> tensor<16xi32>
func.func @test_NormalizeCastLowering_cast_i64_to_i32_with_overflow_mode(%arg0: tensor<16xi64>) -> tensor<16xi32> {
  %0 = tensor.empty() : tensor<16xi32>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xi64>) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi32>
  return %1 : tensor<16xi32>
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_cast_i64_to_i16_with_overflow_mode(
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<16xi64>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<16xf32>) outs({{.*}} : tensor<16xi16>) -> tensor<16xi16>
func.func @test_NormalizeCastLowering_cast_i64_to_i16_with_overflow_mode(%arg0: tensor<16xi64>) -> tensor<16xi16> {
  %0 = tensor.empty() : tensor<16xi16>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xi64>) outs(%0 : tensor<16xi16>) -> tensor<16xi16>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi16>
  return %1 : tensor<16xi16>
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_cast_i64_to_i8_with_overflow_mode(
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<16xi64>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<16xf32>) outs({{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<16xf16>) outs({{.*}} : tensor<16xi8>) -> tensor<16xi8>
func.func @test_NormalizeCastLowering_cast_i64_to_i8_with_overflow_mode(%arg0: tensor<16xi64>) -> tensor<16xi8> {
  %0 = tensor.empty() : tensor<16xi8>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xi64>) outs(%0 : tensor<16xi8>) -> tensor<16xi8>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi8>
  return %1 : tensor<16xi8>
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_cast_i32_to_i16_with_overflow_mode(
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<16xi32>) outs({{.*}} : tensor<16xi16>) -> tensor<16xi16>
func.func @test_NormalizeCastLowering_cast_i32_to_i16_with_overflow_mode(%arg0: tensor<16xi32>) -> tensor<16xi16> {
  %0 = tensor.empty() : tensor<16xi16>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xi32>) outs(%0 : tensor<16xi16>) -> tensor<16xi16>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi16>
  return %1 : tensor<16xi16>
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_cast_i32_to_i8_with_overflow_mode(
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<16xi32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<16xf32>) outs({{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<16xf16>) outs({{.*}} : tensor<16xi8>) -> tensor<16xi8>
func.func @test_NormalizeCastLowering_cast_i32_to_i8_with_overflow_mode(%arg0: tensor<16xi32>) -> tensor<16xi8> {
  %0 = tensor.empty() : tensor<16xi8>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xi32>) outs(%0 : tensor<16xi8>) -> tensor<16xi8>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi8>
  return %1 : tensor<16xi8>
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_cast_i16_to_i8_with_overflow_mode(
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<16xi16>) outs({{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<16xf16>) outs({{.*}} : tensor<16xi8>) -> tensor<16xi8>
func.func @test_NormalizeCastLowering_cast_i16_to_i8_with_overflow_mode(%arg0: tensor<16xi16>) -> tensor<16xi8> {
  %0 = tensor.empty() : tensor<16xi8>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xi16>) outs(%0 : tensor<16xi8>) -> tensor<16xi8>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi8>
  return %1 : tensor<16xi8>
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_fp8_to_fp16
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<7x31x7xf8E5M2>) outs({{.*}} : tensor<7x31x7xf32>) -> tensor<7x31x7xf32>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<7x31x7xf32>) outs({{.*}} : tensor<7x31x7xf16>) -> tensor<7x31x7xf16>
func.func @test_NormalizeCastLowering_fp8_to_fp16(%arg0: tensor<7x31x7xf8E5M2>) -> tensor<7x31x7xf16> {
    %0 = tensor.empty() : tensor<7x31x7xf16>
    %1 = hfusion.cast {enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<7x31x7xf8E5M2>) outs(%0 : tensor<7x31x7xf16>) -> tensor<7x31x7xf16>
    return %1 : tensor<7x31x7xf16>
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_fp16_to_fp8
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<7x31x7xf16>) outs({{.*}} : tensor<7x31x7xf32>) -> tensor<7x31x7xf32>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<7x31x7xf32>) outs({{.*}} : tensor<7x31x7xf8E5M2>) -> tensor<7x31x7xf8E5M2>
func.func @test_NormalizeCastLowering_fp16_to_fp8(%arg0: tensor<7x31x7xf16>) -> tensor<7x31x7xf8E5M2> {
    %0 = tensor.empty() : tensor<7x31x7xf8E5M2>
    %1 = hfusion.cast {enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<7x31x7xf16>) outs(%0 : tensor<7x31x7xf8E5M2>) -> tensor<7x31x7xf8E5M2>
    return %1 : tensor<7x31x7xf8E5M2>
}

// -----

// CHECK-LABEL: @test_NormalizetruncfBf16_triton_scalar_f32_to_bf16(
// CHECK:  %[[c0:.*]] = arith.constant 0 : index
// CHECK: %[[FROM_ELEMENTS:.*]] = tensor.from_elements %arg0 : tensor<1xf32>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1xbf16>
// CHECK: %[[CAST:.*]] = hfusion.cast {{.*}} ins(%[[FROM_ELEMENTS:.*]] : tensor<1xf32>) outs(%[[EMPTY:.*]] : tensor<1xbf16>) -> tensor<1xbf16>
// CHECK: %[[EXTRACTED:.*]] = tensor.extract %[[CAST:.*]][%[[c0:.*]]] : tensor<1xbf16>
// CHECK: return %[[EXTRACTED:.*]] : bf16
func.func @test_NormalizetruncfBf16_triton_scalar_f32_to_bf16(%arg0: f32) -> bf16 {
  %1 = arith.truncf %arg0 : f32 to bf16
  return %1 : bf16
}

// -----

// CHECK-LABEL: @test_NormalizetruncfExtf_test_arith_extf_scalar_bf16_to_f32
// CHECK: %[[ZERO:.*]] = tensor.from_elements %arg0 : tensor<1xbf16>
// CHECK: %[[ONE:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[TWO:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%from_elements : tensor<1xbf16>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
func.func @test_NormalizetruncfExtf_test_arith_extf_scalar_bf16_to_f32(%arg0: bf16) -> f32 {
    %0 = arith.extf %arg0 : bf16 to f32
    return %0 : f32
}

// -----

// CHECK-LABEL: @test_NormalizetruncfExtf_fold_truncf_extf
// CHECK: return %arg0 : f32
func.func @test_NormalizetruncfExtf_fold_truncf_extf(%arg0: f32) -> f32 {
  %0 = arith.truncf %arg0 : f32 to bf16
  %1 = arith.extf %0 : bf16 to f32
  return %1 : f32
}

// -----

// CHECK-LABEL: @test_NormalizeAnyToF32UnaryRecOp_f16
// CHECK: tensor.empty() : tensor<5x1xf32>
// CHECK: hfusion.cast {{.*}} ins(%arg0 : tensor<5x1xf16>) outs({{.*}} : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins({{.*}} : tensor<5x1xf32>) outs({{.*}} : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<5x1xf32>) outs({{.*}} : tensor<5x1xf16>) -> tensor<5x1xf16>
func.func @test_NormalizeAnyToF32UnaryRecOp_f16(%arg0: tensor<5x1xf16>) -> tensor<5x1xf16> {
  %0 = tensor.empty() : tensor<5x1xf16>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%arg0 : tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
  return %1 : tensor<5x1xf16>
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_triton_cast_f32_to_int8
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<13xf32>) outs({{.*}} : tensor<13xi32>) -> tensor<13xi32>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<13xi32>) outs({{.*}} : tensor<13xi8>) -> tensor<13xi8>
func.func @test_NormalizeCastLowering_triton_cast_f32_to_int8(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [13], strides: [1] : memref<?xf32> to memref<13xf32, strided<[1]>>
  %alloc = memref.alloc() : memref<13xf32>
  memref.copy %reinterpret_cast, %alloc : memref<13xf32, strided<[1]>> to memref<13xf32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<13xf32>
  %1 = tensor.empty() : tensor<13xi8>
  %2 = hfusion.cast {round_mode = #hfusion.round_mode<truncwithoverflow>} ins(%0 : tensor<13xf32>) outs(%1 : tensor<13xi8>) -> tensor<13xi8>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [13], strides: [1] : memref<?xi8> to memref<13xi8, strided<[1]>>
  bufferization.materialize_in_destination %2 in writable %reinterpret_cast_0 : (tensor<13xi8>, memref<13xi8, strided<[1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_cast_u32_to_f32
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%arg0 : tensor<16xi32>) outs({{.*}} : tensor<16xi64>) -> tensor<16xi64>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<16xi64>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
func.func @test_NormalizeCastLowering_cast_u32_to_f32(%arg0: tensor<16xi32>) -> tensor<16xf32> {
  %0 = tensor.empty() : tensor<16xf32>
  %1 = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<16xi32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
  return %1 : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_triton_cast_u32_to_bf16
// CHECK: %[[CAST1:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%0 : tensor<13xi32>) outs({{.*}} : tensor<13xi64>) -> tensor<13xi64>
// CHECK: %[[CAST2:.*]] = hfusion.cast {{.*}} ins(%[[CAST1]] : tensor<13xi64>) outs({{.*}} : tensor<13xf32>) -> tensor<13xf32>
// CHECK: %[[CAST3:.*]] = hfusion.cast {{.*}} ins(%[[CAST2]] : tensor<13xf32>) outs({{.*}} : tensor<13xbf16>) -> tensor<13xbf16>
func.func @test_NormalizeCastLowering_triton_cast_u32_to_bf16(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [13], strides: [1] : memref<?xi32> to memref<13xi32, strided<[1]>>
  %alloc = memref.alloc() : memref<13xi32>
  memref.copy %reinterpret_cast, %alloc : memref<13xi32, strided<[1]>> to memref<13xi32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<13xi32>
  %1 = tensor.empty() : tensor<13xbf16>
  %2 = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, round_mode = #hfusion.round_mode<rint>} ins(%0 : tensor<13xi32>) outs(%1 : tensor<13xbf16>) -> tensor<13xbf16>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [13], strides: [1] : memref<?xbf16> to memref<13xbf16, strided<[1]>>
  bufferization.materialize_in_destination %2 in writable %reinterpret_cast_0 : (tensor<13xbf16>, memref<13xbf16, strided<[1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_triton_cast_f32_to_uint32
// CHECK: %[[CAST1:.*]] = hfusion.cast {{.*}} ins({{.*}} : tensor<16xf32>) outs({{.*}} : tensor<16xi64>) -> tensor<16xi64>
// CHECK: %[[CAST2:.*]] = hfusion.cast {{.*}} ins(%[[CAST1]] : tensor<16xi64>) outs({{.*}} : tensor<16xi32>) -> tensor<16xi32>
func.func @test_NormalizeCastLowering_triton_cast_f32_to_uint32(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1]>>
  %alloc = memref.alloc() : memref<16xf32>
  memref.copy %reinterpret_cast, %alloc : memref<16xf32, strided<[1]>> to memref<16xf32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<16xf32>
  %1 = tensor.empty() : tensor<16xi32>
  %2 = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, round_mode = #hfusion.round_mode<trunc>} ins(%0 : tensor<16xf32>) outs(%1 : tensor<16xi32>) -> tensor<16xi32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [16], strides: [1] : memref<?xi32> to memref<16xi32, strided<[1]>>
  bufferization.materialize_in_destination %2 in writable %reinterpret_cast_0 : (tensor<16xi32>, memref<16xi32, strided<[1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_cast_i64_to_i32
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%0 : tensor<23xi64>) outs(%1 : tensor<23xi32>) -> tensor<23xi32>
func.func @test_NormalizeCastLowering_cast_i64_to_i32(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [23], strides: [1] : memref<?xi64> to memref<23xi64, strided<[1]>>
  %alloc = memref.alloc() : memref<23xi64>
  memref.copy %reinterpret_cast, %alloc : memref<23xi64, strided<[1]>> to memref<23xi64>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<23xi64>
  %1 = tensor.empty() : tensor<23xi32>
  %2 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%0 : tensor<23xi64>) outs(%1 : tensor<23xi32>) -> tensor<23xi32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [23], strides: [1] : memref<?xi32> to memref<23xi32, strided<[1]>>
  bufferization.materialize_in_destination %2 in writable %reinterpret_cast_0 : (tensor<23xi32>, memref<23xi32, strided<[1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_cast_i32_to_i16
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<truncwithoverflow>} ins(%0 : tensor<23xi32>) outs(%1 : tensor<23xi16>) -> tensor<23xi16>
func.func @test_NormalizeCastLowering_cast_i32_to_i16(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [23], strides: [1] : memref<?xi32> to memref<23xi32, strided<[1]>>
  %alloc = memref.alloc() : memref<23xi32>
  memref.copy %reinterpret_cast, %alloc : memref<23xi32, strided<[1]>> to memref<23xi32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<23xi32>
  %1 = tensor.empty() : tensor<23xi16>
  %2 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<truncwithoverflow>} ins(%0 : tensor<23xi32>) outs(%1 : tensor<23xi16>) -> tensor<23xi16>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [23], strides: [1] : memref<?xi16> to memref<23xi16, strided<[1]>>
  bufferization.materialize_in_destination %2 in writable %reinterpret_cast_0 : (tensor<23xi16>, memref<23xi16, strided<[1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_cast_i64_to_u32_sat
func.func @test_NormalizeCastLowering_cast_i64_to_u32_sat(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [25, 5], strides: [5, 1] : memref<?xi64> to memref<25x5xi64, strided<[5, 1]>>
  %alloc = memref.alloc() : memref<25x5xi64>
  memref.copy %reinterpret_cast, %alloc : memref<25x5xi64, strided<[5, 1]>> to memref<25x5xi64>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<25x5xi64>
  %1 = tensor.empty() : tensor<25x5xi32>
  // CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<25x5xi64>) outs({{.*}} : tensor<25x5xi32>) -> tensor<25x5xi32>
  %2 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%0 : tensor<25x5xi64>) outs(%1 : tensor<25x5xi32>) -> tensor<25x5xi32>
  annotation.mark %2 {saturate_src_unsigned} : tensor<25x5xi32>
  annotation.mark %2 {saturate_dst_unsigned = true} : tensor<25x5xi32>
  annotation.mark %2 {overflow_mode = "saturate"} : tensor<25x5xi32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [25, 5], strides: [5, 1] : memref<?xi32> to memref<25x5xi32, strided<[5, 1]>>
  bufferization.materialize_in_destination %2 in writable %reinterpret_cast_0 : (tensor<25x5xi32>, memref<25x5xi32, strided<[5, 1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_cast_to_nd_with_overflow
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<13xi32>) outs({{.*}} : tensor<13xi8>) -> tensor<13xi8>
func.func @test_NormalizeCastLowering_cast_to_nd_with_overflow(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: memref<?xi16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [13], strides: [1] : memref<?xi16> to memref<13xi16, strided<[1]>>
  %alloc = memref.alloc() : memref<13xi16>
  memref.copy %reinterpret_cast, %alloc : memref<13xi16, strided<[1]>> to memref<13xi16>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<13xi16>
  %1 = tensor.empty() : tensor<13xf32>
  %2 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%0 : tensor<13xi16>) outs(%1 : tensor<13xf32>) -> tensor<13xf32>
  %3 = tensor.empty() : tensor<13xi8>
  %4 = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, round_mode = #hfusion.round_mode<truncwithoverflow>} ins(%2 : tensor<13xf32>) outs(%3 : tensor<13xi8>) -> tensor<13xi8>
  annotation.mark %4 {overflow_mode = "trunc"} : tensor<13xi8>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [13], strides: [1] : memref<?xi8> to memref<13xi8, strided<[1]>>
  bufferization.materialize_in_destination %4 in writable %reinterpret_cast_0 : (tensor<13xi8>, memref<13xi8, strided<[1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @test_NormalizefillCastToTensorBrc_opt_cast_IToF_fill
// CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<24x32xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<24x32xf32>) -> tensor<24x32xf32>
// CHECK: return %[[FILL]] : tensor<24x32xf32>
func.func @test_NormalizefillCastToTensorBrc_opt_cast_IToF_fill() -> tensor<24x32xf32>{
  %c1_i32 = arith.constant 1 : i32
  %0 = tensor.empty() : tensor<24x32xf32>
  %1 = tensor.empty() : tensor<24x32xi32>
  %2 = linalg.fill ins(%c1_i32 : i32) outs(%1 : tensor<24x32xi32>) -> tensor<24x32xi32>
  %3 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%2 : tensor<24x32xi32>) outs(%0 : tensor<24x32xf32>) -> tensor<24x32xf32>
  return %3 : tensor<24x32xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizefillCastToTensorBrc_opt_cast_FToI_fill_rint
// CHECK: %[[CST:.*]] = arith.constant 1.500000e+00 : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<f32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY0]] : tensor<f32>) -> tensor<f32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<i32>
// CHECK: %[[CAST:.*]] = hfusion.cast {{.*}} ins(%[[FILL]] : tensor<f32>) outs(%[[EMPTY1]] : tensor<i32>) -> tensor<i32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<24x32xi32>
// CHECK: %[[BRC:.*]] = linalg.broadcast ins(%[[CAST]] : tensor<i32>) outs(%[[EMPTY2]] : tensor<24x32xi32>) dimensions = [0, 1]
// CHECK: return %[[BRC]] : tensor<24x32xi32>
func.func @test_NormalizefillCastToTensorBrc_opt_cast_FToI_fill_rint() -> tensor<24x32xi32>{
  %c1_i32 = arith.constant 1.5 : f32
  %0 = tensor.empty() : tensor<24x32xf32>
  %1 = tensor.empty() : tensor<24x32xi32>
  %2 = linalg.fill ins(%c1_i32 : f32) outs(%0 : tensor<24x32xf32>) -> tensor<24x32xf32>
  %3 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<ceil>} ins(%2 : tensor<24x32xf32>) outs(%1 : tensor<24x32xi32>) -> tensor<24x32xi32>
  return %3 : tensor<24x32xi32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeBrcCast_opt_cast_FToI_brc
// CHECK: %[[CST:.*]] = arith.constant dense<1.500000e+00> : tensor<32xf32>
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<32xi32>
// CHECK: %[[CAST:.*]] = hfusion.cast {{.*}} ins(%[[CST]] : tensor<32xf32>) outs(%[[EMPTY0]] : tensor<32xi32>) -> tensor<32xi32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<24x32xi32>
// CHECK: %[[BRC:.*]] = linalg.broadcast ins(%[[CAST]] : tensor<32xi32>) outs(%[[EMPTY1]] : tensor<24x32xi32>) dimensions = [0]
// CHECK: return %[[BRC]] : tensor<24x32xi32>
func.func @test_NormalizeBrcCast_opt_cast_FToI_brc() -> tensor<24x32xi32>{
  %c1_f32 = arith.constant dense<1.5> : tensor<32xf32>
  %0 = tensor.empty() : tensor<24x32xi32>
  %1 = tensor.empty() : tensor<24x32xf32>
  %2 = linalg.broadcast ins(%c1_f32 : tensor<32xf32>) outs(%1 : tensor<24x32xf32>) dimensions=[0]
  %3 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%2 : tensor<24x32xf32>) outs(%0 : tensor<24x32xi32>) -> tensor<24x32xi32>
  return %3 : tensor<24x32xi32>
}

// -----

// CHECK-LABEL: @test_NormalizeBrcCast_broadcast(
// CHECK-NOT: linalg.broadcast
// CHECK: linalg.fill
// CHECK: linalg.fill
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_NormalizeBrcCast_broadcast(%arg0: tensor<23xi64>, %arg1: tensor<f16>) -> tensor<23xf16> attributes {hacc.entry} {
    %cst = arith.constant 0.86956521739130432 : f64
    %0 = tensor.empty() : tensor<23xf16>
    %1 = tensor.empty() : tensor<23xf32>
    %2 = hfusion.cast {enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<23xi64>) outs(%1 : tensor<23xf32>) -> tensor<23xf32>
    %3 = hfusion.cast {enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>} ins(%2 : tensor<23xf32>) outs(%0 : tensor<23xf16>) -> tensor<23xf16>
    %broadcasted = linalg.broadcast ins(%arg1 : tensor<f16>) outs(%0 : tensor<23xf16>) dimensions = [0]
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%3, %broadcasted : tensor<23xf16>, tensor<23xf16>) outs(%0 : tensor<23xf16>) -> tensor<23xf16>
    %5 = arith.truncf %cst : f64 to f16
    %6 = linalg.fill ins(%5 : f16) outs(%0 : tensor<23xf16>) -> tensor<23xf16>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %6 : tensor<23xf16>, tensor<23xf16>) outs(%0 : tensor<23xf16>) -> tensor<23xf16>
    return %7 : tensor<23xf16>
  }
}

// -----

// CHECK-LABEL: @test_NormalizeBrcCast_broadcast_2(
// CHECK-NOT: linalg.broadcast
// CHECK: %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK: %[[VAL_6:.*]] = tensor.extract %{{.*}}{{\[}}%[[VAL_2]], %[[VAL_2]], %[[VAL_2]]] : tensor<1x1x1xf16>
// CHECK: %[[VAL_7:.*]] = linalg.fill ins(%[[VAL_6]] : f16) outs(%{{.*}} : tensor<1x1x23x1xf16>) -> tensor<1x1x23x1xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_NormalizeBrcCast_broadcast_2(%arg0: tensor<1x1x23x1xf32>, %arg1: tensor<1x1x1xf16>) -> tensor<1x1x23x1xf16> attributes {hacc.entry} {
    %cst = arith.constant 0.86956521739130432 : f64
    %0 = tensor.empty() : tensor<1x1x23x1xf16>
    %1 = tensor.empty() : tensor<1x1x23x1xf32>
    %3 = hfusion.cast {enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<1x1x23x1xf32>) outs(%0 : tensor<1x1x23x1xf16>) -> tensor<1x1x23x1xf16>
    %broadcasted = linalg.broadcast ins(%arg1 : tensor<1x1x1xf16>) outs(%0 : tensor<1x1x23x1xf16>) dimensions = [2]
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%3, %broadcasted : tensor<1x1x23x1xf16>, tensor<1x1x23x1xf16>) outs(%0 : tensor<1x1x23x1xf16>) -> tensor<1x1x23x1xf16>
    %5 = arith.truncf %cst : f64 to f16
    %6 = linalg.fill ins(%5 : f16) outs(%0 : tensor<1x1x23x1xf16>) -> tensor<1x1x23x1xf16>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %6 : tensor<1x1x23x1xf16>, tensor<1x1x23x1xf16>) outs(%0 : tensor<1x1x23x1xf16>) -> tensor<1x1x23x1xf16>
    return %7 : tensor<1x1x23x1xf16>
  }
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_cast_to_nd_with_overflow
// CHECK-NOT: annotation.mark
func.func @test_NormalizeCastLowering_cast_to_nd_with_overflow(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %c0_i64 = arith.constant 0 : i64
  %0 = tensor.empty() : tensor<300xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<300xi64>) -> tensor<300xi64>
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [300], strides: [1] : memref<?xi64> to memref<300xi64, strided<[1]>>
  %alloc = memref.alloc() : memref<300xi64>
  memref.copy %reinterpret_cast, %alloc : memref<300xi64, strided<[1]>> to memref<300xi64>
  %2 = bufferization.to_tensor %alloc restrict writable : memref<300xi64>
  %3 = tensor.empty() : tensor<300xi1>
  %4 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%2, %1 : tensor<300xi64>, tensor<300xi64>) outs(%3 : tensor<300xi1>) -> tensor<300xi1>
  annotation.mark %4 {overflow_mode = "trunc"} : tensor<300xi1>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [300], strides: [1] : memref<?xi8> to memref<300xi8, strided<[1]>>
  %5 = tensor.empty() : tensor<300xi8>
  %6 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%4 : tensor<300xi1>) outs(%5 : tensor<300xi8>) -> tensor<300xi8>
  bufferization.materialize_in_destination %6 in writable %reinterpret_cast_0 : (tensor<300xi8>, memref<300xi8, strided<[1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_tensor_i64_to_f16
// CHECK: %[[ZERO:.*]] = tensor.empty() : tensor<23xf16>
// CHECK: %[[ONE:.*]] = tensor.empty() : tensor<23xf32>
// CHECK: %[[TWO:.*]] = hfusion.cast {{.*}} ins(%[[arg0:.*]] : tensor<23xi64>) outs(%[[ONE:.*]] : tensor<23xf32>) -> tensor<23xf32>
// CHECK: %[[THREE:.*]] = hfusion.cast {{.*}} ins(%[[TWO:.*]] : tensor<23xf32>) outs(%[[ZERO:.*]] : tensor<23xf16>) -> tensor<23xf16>
module attributes {hacc.target = #hacc.target<"Ascend310B4">} {
  func.func @test_NormalizeCastLowering_tensor_i64_to_f16(%arg0: tensor<23xi64>, %arg1: tensor<f16>) -> tensor<23xf16> attributes {hacc.entry} {
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
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_lowering_i64_to_bf16(
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xi64>) outs({{.*}} : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xf32>) outs({{.*}} : tensor<4x4xbf16>) -> tensor<4x4xbf16>
module attributes {hacc.target = #hacc.target<"Ascend310B4">} {
  func.func @test_NormalizeCastLowering_lowering_i64_to_bf16(%arg0: tensor<4x4xi64>) -> tensor<4x4xbf16> {
    %0 = tensor.empty() : tensor<4x4xbf16>
    %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<4x4xi64>) outs(%0 : tensor<4x4xbf16>) -> tensor<4x4xbf16>
    return %1 : tensor<4x4xbf16>
  }
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_i16_to_i32(
// CHECK: %0 = tensor.empty() : tensor<4x4xi32>
// CHECK: %1 = hfusion.cast {{.*}} ins(%arg0 : tensor<4x4xi16>) outs(%0 : tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: return %1 : tensor<4x4xi32>
module attributes {hacc.target = #hacc.target<"Ascend310B4">} {
  func.func @test_NormalizeCastLowering_i16_to_i32(%arg0: tensor<4x4xi16>) -> tensor<4x4xi32> {
    %0 = tensor.empty() : tensor<4x4xi32>
    %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<4x4xi16>) outs(%0 : tensor<4x4xi32>) -> tensor<4x4xi32>
    return %1 : tensor<4x4xi32>
  }
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_i8_to_i32(
// CHECK: %[[CAST:.*]] = hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xi8>) outs({{.*}} : tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: return %[[CAST]] : tensor<4x4xi32>
module attributes {hacc.target = #hacc.target<"Ascend310B4">} {
  func.func @test_NormalizeCastLowering_i8_to_i32(%arg0: tensor<4x4xi8>) -> tensor<4x4xi32> {
    %0 = tensor.empty() : tensor<4x4xi32>
    %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<4x4xi8>) outs(%0 : tensor<4x4xi32>) -> tensor<4x4xi32>
    return %1 : tensor<4x4xi32>
  }
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_i8_to_bf16
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xi8>) outs({{.*}} : tensor<4x4xf16>) -> tensor<4x4xf16>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xf16>) outs({{.*}} : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xf32>) outs({{.*}} : tensor<4x4xbf16>) -> tensor<4x4xbf16>
module attributes {hacc.target = #hacc.target<"Ascend310B4">} {
  func.func @test_NormalizeCastLowering_i8_to_bf16(%arg0: tensor<4x4xi8>) -> tensor<4x4xbf16> {
    %0 = tensor.empty() : tensor<4x4xbf16>
    %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<4x4xi8>) outs(%0 : tensor<4x4xbf16>) -> tensor<4x4xbf16>
    return %1 : tensor<4x4xbf16>
  }
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_f32_to_i16
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xf32>) outs({{.*}} : tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<4x4xi32>) outs({{.*}} : tensor<4x4xi16>) -> tensor<4x4xi16>
module attributes {hacc.target = #hacc.target<"Ascend310B4">} {
  func.func @test_NormalizeCastLowering_f32_to_i16(%arg0: tensor<4x4xf32>) -> tensor<4x4xi16> {
    %0 = tensor.empty() : tensor<4x4xi16>
    %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} ins(%arg0 : tensor<4x4xf32>) outs(%0 : tensor<4x4xi16>) -> tensor<4x4xi16>
    return %1 : tensor<4x4xi16>
  }
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_i64_to_i16
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<4x4xi64>) outs({{.*}} : tensor<4x4xi16>) -> tensor<4x4xi16>
module attributes {hacc.target = #hacc.target<"Ascend310B4">} {
  func.func @test_NormalizeCastLowering_i64_to_i16(%arg0: tensor<4x4xi64>) -> tensor<4x4xi16> {
    %0 = tensor.empty() : tensor<4x4xi16>
    %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x4xi64>) outs(%0 : tensor<4x4xi16>) -> tensor<4x4xi16>
    return %1 : tensor<4x4xi16>
  }
}

// -----

// CHECK-LABEL: @test_NormalizeCastLowering_i64_to_i8
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<4x4xi64>) outs({{.*}} : tensor<4x4xi8>) -> tensor<4x4xi8>
module attributes {hacc.target = #hacc.target<"Ascend310B4">} {
  func.func @test_NormalizeCastLowering_i64_to_i8(%arg0: tensor<4x4xi64>) -> tensor<4x4xi8> {
    %0 = tensor.empty() : tensor<4x4xi8>
    %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x4xi64>) outs(%0 : tensor<4x4xi8>) -> tensor<4x4xi8>
    return %1 : tensor<4x4xi8>
  }
}
