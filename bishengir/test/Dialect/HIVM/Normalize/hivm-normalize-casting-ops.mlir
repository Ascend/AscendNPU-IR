// RUN: bishengir-opt --hivm-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_NormalizeCastLowering_cast_f32_to_i1
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x256x12x257xf32>) -> tensor<2x256x12x257xi1>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<2x256x12x257xf32>
// CHECK: %[[BRC:.*]] = hivm.hir.vbrc ins(%[[C0]] : f32) outs(%[[EMPTY1]] : tensor<2x256x12x257xf32>) -> tensor<2x256x12x257xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<2x256x12x257xi1>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<2x256x12x257xi1>
// CHECK: %[[CMP:.*]] = hivm.hir.vcmp ins(%[[ARG0]], %[[BRC]] : tensor<2x256x12x257xf32>, tensor<2x256x12x257xf32>) outs(%[[EMPTY3]] : tensor<2x256x12x257xi1>) -> tensor<2x256x12x257xi1>
// CHECK: %[[NOT:.*]] = hivm.hir.vnot ins(%[[CMP]] : tensor<2x256x12x257xi1>) outs(%[[EMPTY2]] : tensor<2x256x12x257xi1>) -> tensor<2x256x12x257xi1>
// CHECK: return %[[NOT]]
func.func @test_NormalizeCastLowering_cast_f32_to_i1(%arg0: tensor<2x256x12x257xf32>) -> tensor<2x256x12x257xi1> {
  %0 = tensor.empty() : tensor<2x256x12x257xi1>
  %1 = hivm.hir.vcast ins(%arg0 : tensor<2x256x12x257xf32>) outs(%0 : tensor<2x256x12x257xi1>)
      round_mode = <trunc> -> tensor<2x256x12x257xi1>
  return %1 : tensor<2x256x12x257xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_cast_i16_to_i1
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8xi16>) -> tensor<8xi1>
// CHECK: %[[C0:.*]] = arith.constant 0.000000e+00 : f16
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<8xf16>
// CHECK: %[[CAST0:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<8xi16>) outs(%[[EMPTY0]] : tensor<8xf16>) -> tensor<8xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<8xf16>
// CHECK: %[[BRC:.*]] = hivm.hir.vbrc ins(%[[C0]] : f16) outs(%[[EMPTY1]] : tensor<8xf16>) -> tensor<8xf16>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<8xi1>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<8xi1>
// CHECK: %[[CMP:.*]] = hivm.hir.vcmp ins(%[[CAST0]], %[[BRC]] : tensor<8xf16>, tensor<8xf16>) outs(%[[EMPTY3]] : tensor<8xi1>) -> tensor<8xi1>
// CHECK: %[[NOT:.*]] = hivm.hir.vnot ins(%[[CMP]] : tensor<8xi1>) outs(%[[EMPTY2]] : tensor<8xi1>) -> tensor<8xi1>
// CHECK: return %[[NOT]]
func.func @test_NormalizeCastLowering_cast_i16_to_i1(%arg0: tensor<8xi16>) -> tensor<8xi1> {
  %0 = tensor.empty() : tensor<8xi1>
  %1 = hivm.hir.vcast ins(%arg0 : tensor<8xi16>) outs(%0 : tensor<8xi1>)
      round_mode = <rint> -> tensor<8xi1>
  return %1 : tensor<8xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_cast_bf16_to_i1
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xbf16>) -> tensor<16xi1>
// CHECK: %[[C0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[CAST0:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<16xbf16>) outs(%[[EMPTY0]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[BRC:.*]] = hivm.hir.vbrc ins(%[[C0]] : f32) outs(%[[EMPTY1]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[CMP:.*]] = hivm.hir.vcmp ins(%[[CAST0]], %[[BRC]] : tensor<16xf32>, tensor<16xf32>) outs(%[[EMPTY3]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[NOT:.*]] = hivm.hir.vnot ins(%[[CMP]] : tensor<16xi1>) outs(%[[EMPTY2]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: return %[[NOT]]
func.func @test_NormalizeCastLowering_cast_bf16_to_i1(%arg0: tensor<16xbf16>) -> tensor<16xi1> {
  %0 = tensor.empty() : tensor<16xi1>
  %1 = hivm.hir.vcast ins(%arg0 : tensor<16xbf16>) outs(%0 : tensor<16xi1>)
      round_mode = <rint> -> tensor<16xi1>
  return %1 : tensor<16xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_cast_f32_to_i16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4xf32>) -> tensor<4x4xi16>
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<4x4xi32>
// CHECK: %[[CAST0:.*]] = hivm.hir.vcast
// CHECK-SAME: ins(%[[ARG0]] : tensor<4x4xf32>) outs(%[[EMPTY0]] : tensor<4x4xi32>) round_mode = <trunc> -> tensor<4x4xi32>
// CHECK: annotation.mark %[[CAST0]] {overflow_mode = "trunc"} : tensor<4x4xi32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<4x4xi16>
// CHECK: %[[CAST1:.*]] = hivm.hir.vcast ins(%[[CAST0]] : tensor<4x4xi32>) outs(%[[EMPTY1]] : tensor<4x4xi16>) round_mode = <truncwithoverflow> -> tensor<4x4xi16>
// CHECK: return %[[CAST1]]
func.func @test_NormalizeCastLowering_cast_f32_to_i16(%arg0: tensor<4x4xf32>) -> tensor<4x4xi16> {
  %0 = tensor.empty() : tensor<4x4xi16>
  %1 = hivm.hir.vcast ins(%arg0 : tensor<4x4xf32>) outs(%0 : tensor<4x4xi16>)
      round_mode = <trunc> -> tensor<4x4xi16>
  annotation.mark %1 {overflow_mode = "trunc"} : tensor<4x4xi16>
  return %1 : tensor<4x4xi16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_cast_f16_to_i8_with_overflow_mode
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf16>) -> tensor<16xi8>
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<16xi32>
// CHECK: %[[CAST0:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<16xf16>) outs(%[[EMPTY0]] : tensor<16xi32>) round_mode = <trunc> -> tensor<16xi32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<16xi8>
// CHECK: %[[CAST1:.*]] = hivm.hir.vcast ins(%[[CAST0]] : tensor<16xi32>) outs(%[[EMPTY1]] : tensor<16xi8>) round_mode = <truncwithoverflow> -> tensor<16xi8>
// CHECK: return %[[CAST1]]
func.func @test_NormalizeCastLowering_cast_f16_to_i8_with_overflow_mode(%arg0: tensor<16xf16>) -> tensor<16xi8> {
  %0 = tensor.empty() : tensor<16xi8>
  %1 = hivm.hir.vcast ins(%arg0 : tensor<16xf16>) outs(%0 : tensor<16xi8>)
      round_mode = <trunc> -> tensor<16xi8>
  annotation.mark %1 {overflow_mode = "trunc"} : tensor<16xi8>
  return %1 : tensor<16xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_cast_i64_to_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<23xi64>) -> tensor<23xf16>
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<23xf32>
// CHECK: %[[CAST0:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<23xi64>) outs(%[[EMPTY0]] : tensor<23xf32>) round_mode = <trunc> -> tensor<23xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<23xf16>
// CHECK: %[[CAST1:.*]] = hivm.hir.vcast ins(%[[CAST0]] : tensor<23xf32>) outs(%[[EMPTY1]] : tensor<23xf16>) -> tensor<23xf16>
// CHECK: return %[[CAST1]]
func.func @test_NormalizeCastLowering_cast_i64_to_f16(%arg0: tensor<23xi64>) -> tensor<23xf16> {
  %0 = tensor.empty() : tensor<23xf16>
  %1 = hivm.hir.vcast ins(%arg0 : tensor<23xi64>) outs(%0 : tensor<23xf16>)
      round_mode = <trunc> -> tensor<23xf16>
  return %1 : tensor<23xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_cast_i8_to_bf16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4xi8>) -> tensor<4x4xbf16>
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<4x4xf16>
// CHECK: %[[CAST0:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<4x4xi8>) outs(%[[EMPTY0]] : tensor<4x4xf16>) -> tensor<4x4xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<4x4xf32>
// CHECK: %[[CAST1:.*]] = hivm.hir.vcast ins(%[[CAST0]] : tensor<4x4xf16>) outs(%[[EMPTY1]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<4x4xbf16>
// CHECK: %[[CAST2:.*]] = hivm.hir.vcast ins(%[[CAST1]] : tensor<4x4xf32>) outs(%[[EMPTY2]] : tensor<4x4xbf16>) -> tensor<4x4xbf16>
// CHECK: return %[[CAST2]]
func.func @test_NormalizeCastLowering_cast_i8_to_bf16(%arg0: tensor<4x4xi8>) -> tensor<4x4xbf16> {
  %0 = tensor.empty() : tensor<4x4xbf16>
  %1 = hivm.hir.vcast ins(%arg0 : tensor<4x4xi8>) outs(%0 : tensor<4x4xbf16>)
      round_mode = <rint> -> tensor<4x4xbf16>
  return %1 : tensor<4x4xbf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_i1_cast_i64
// CHECK-SAME: (%[[ARG0:.*]]: tensor<200x200xi1>) -> tensor<200x200xi64>
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<200x200xf16>
// CHECK: %[[CAST0:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<200x200xi1>) outs(%[[EMPTY0]] : tensor<200x200xf16>) round_mode = <trunc> -> tensor<200x200xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<200x200xf32>
// CHECK: %[[CAST1:.*]] = hivm.hir.vcast ins(%[[CAST0]] : tensor<200x200xf16>) outs(%[[EMPTY1]] : tensor<200x200xf32>) -> tensor<200x200xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<200x200xi64>
// CHECK: %[[CAST2:.*]] = hivm.hir.vcast ins(%[[CAST1]] : tensor<200x200xf32>) outs(%[[EMPTY2]] : tensor<200x200xi64>) round_mode = <trunc> -> tensor<200x200xi64>
// CHECK: return %[[CAST2]]
func.func @test_NormalizeCastLowering_i1_cast_i64(%arg0: tensor<200x200xi1>) -> tensor<200x200xi64> {
  %0 = tensor.empty() : tensor<200x200xi64>
  %1 = hivm.hir.vcast ins(%arg0 : tensor<200x200xi1>) outs(%0 : tensor<200x200xi64>)
      round_mode = <rint> -> tensor<200x200xi64>
  return %1 : tensor<200x200xi64>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_i1_cast_f32
// CHECK-SAME: (%[[ARG0:.*]]: tensor<32xi1>) -> tensor<32xf32>
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<32xf16>
// CHECK: %[[CAST0:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<32xi1>) outs(%[[EMPTY0]] : tensor<32xf16>) round_mode = <trunc> -> tensor<32xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[CAST1:.*]] = hivm.hir.vcast ins(%[[CAST0]] : tensor<32xf16>) outs(%[[EMPTY1]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: return %[[CAST1]]
func.func @test_NormalizeCastLowering_i1_cast_f32(%arg0: tensor<32xi1>) -> tensor<32xf32> {
  %0 = tensor.empty() : tensor<32xf32>
  %1 = hivm.hir.vcast ins(%arg0 : tensor<32xi1>) outs(%0 : tensor<32xf32>)
      round_mode = <rint> -> tensor<32xf32>
  return %1 : tensor<32xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_i8_cast_i32
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4xi8>) -> tensor<4x4xi32>
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<4x4xf16>
// CHECK: %[[CAST0:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<4x4xi8>) outs(%[[EMPTY0]] : tensor<4x4xf16>) -> tensor<4x4xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<4x4xi32>
// CHECK: %[[CAST1:.*]] = hivm.hir.vcast ins(%[[CAST0]] : tensor<4x4xf16>) outs(%[[EMPTY1]] : tensor<4x4xi32>) round_mode = <trunc> -> tensor<4x4xi32>
// CHECK: return %[[CAST1]]
func.func @test_NormalizeCastLowering_i8_cast_i32(%arg0: tensor<4x4xi8>) -> tensor<4x4xi32> {
  %0 = tensor.empty() : tensor<4x4xi32>
  %1 = hivm.hir.vcast ins(%arg0 : tensor<4x4xi8>) outs(%0 : tensor<4x4xi32>)
      round_mode = <rint> -> tensor<4x4xi32>
  return %1 : tensor<4x4xi32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_i64_cast_i1
// CHECK-SAME: (%[[ARG0:.*]]: tensor<20x20xi64>) -> tensor<20x20xi1>
// CHECK: %[[C0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<20x20xf32>
// CHECK: %[[CAST0:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<20x20xi64>) outs(%[[EMPTY0]] : tensor<20x20xf32>) -> tensor<20x20xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<20x20xf32>
// CHECK: %[[BRC:.*]] = hivm.hir.vbrc ins(%[[C0]] : f32) outs(%[[EMPTY1]] : tensor<20x20xf32>) -> tensor<20x20xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<20x20xi1>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<20x20xi1>
// CHECK: %[[CMP:.*]] = hivm.hir.vcmp ins(%[[CAST0]], %[[BRC]] : tensor<20x20xf32>, tensor<20x20xf32>) outs(%[[EMPTY3]] : tensor<20x20xi1>) -> tensor<20x20xi1>
// CHECK: %[[NOT:.*]] = hivm.hir.vnot ins(%[[CMP]] : tensor<20x20xi1>) outs(%[[EMPTY2]] : tensor<20x20xi1>) -> tensor<20x20xi1>
// CHECK: return %[[NOT]]
func.func @test_NormalizeCastLowering_i64_cast_i1(%arg0: tensor<20x20xi64>) -> tensor<20x20xi1> {
  %0 = tensor.empty() : tensor<20x20xi1>
  %1 = hivm.hir.vcast ins(%arg0 : tensor<20x20xi64>) outs(%0 : tensor<20x20xi1>)
      round_mode = <rint> -> tensor<20x20xi1>
  return %1 : tensor<20x20xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_i32_cast_i1
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8xi32>) -> tensor<8xi1>
// CHECK: %[[C0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<8xf32>
// CHECK: %[[CAST0:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<8xi32>) outs(%[[EMPTY0]] : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<8xf32>
// CHECK: %[[BRC:.*]] = hivm.hir.vbrc ins(%[[C0]] : f32) outs(%[[EMPTY1]] : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<8xi1>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<8xi1>
// CHECK: %[[CMP:.*]] = hivm.hir.vcmp ins(%[[CAST0]], %[[BRC]] : tensor<8xf32>, tensor<8xf32>) outs(%[[EMPTY3]] : tensor<8xi1>) -> tensor<8xi1>
// CHECK: %[[NOT:.*]] = hivm.hir.vnot ins(%[[CMP]] : tensor<8xi1>) outs(%[[EMPTY2]] : tensor<8xi1>) -> tensor<8xi1>
// CHECK: return %[[NOT]]
func.func @test_NormalizeCastLowering_i32_cast_i1(%arg0: tensor<8xi32>) -> tensor<8xi1> {
  %0 = tensor.empty() : tensor<8xi1>
  %1 = hivm.hir.vcast ins(%arg0 : tensor<8xi32>) outs(%0 : tensor<8xi1>)
      round_mode = <rint> -> tensor<8xi1>
  return %1 : tensor<8xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_cast_i64_to_i16_with_overflow_mode
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xi64>) -> tensor<16xi16>
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[CAST0:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<16xi64>) outs(%[[EMPTY0]] : tensor<16xf32>) round_mode = <trunc> -> tensor<16xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<16xi16>
// CHECK: %[[CAST1:.*]] = hivm.hir.vcast ins(%[[CAST0]] : tensor<16xf32>) outs(%[[EMPTY1]] : tensor<16xi16>) round_mode = <trunc> -> tensor<16xi16>
// CHECK: return %[[CAST1]]
func.func @test_NormalizeCastLowering_cast_i64_to_i16_with_overflow_mode(%arg0: tensor<16xi64>) -> tensor<16xi16> {
  %0 = tensor.empty() : tensor<16xi16>
  %1 = hivm.hir.vcast ins(%arg0 : tensor<16xi64>) outs(%0 : tensor<16xi16>)
      round_mode = <rint> -> tensor<16xi16>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi16>
  return %1 : tensor<16xi16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_cast_i32_to_i8_with_overflow_mode
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xi32>) -> tensor<16xi8>
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[CAST0:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<16xi32>) outs(%[[EMPTY0]] : tensor<16xf32>) round_mode = <trunc> -> tensor<16xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<16xf16>
// CHECK: %[[CAST1:.*]] = hivm.hir.vcast ins(%[[CAST0]] : tensor<16xf32>) outs(%[[EMPTY1]] : tensor<16xf16>) round_mode = <trunc> -> tensor<16xf16>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<16xi8>
// CHECK: %[[CAST2:.*]] = hivm.hir.vcast ins(%[[CAST1]] : tensor<16xf16>) outs(%[[EMPTY2]] : tensor<16xi8>) round_mode = <trunc> -> tensor<16xi8>
// CHECK: return %[[CAST2]]
func.func @test_NormalizeCastLowering_cast_i32_to_i8_with_overflow_mode(%arg0: tensor<16xi32>) -> tensor<16xi8> {
  %0 = tensor.empty() : tensor<16xi8>
  %1 = hivm.hir.vcast ins(%arg0 : tensor<16xi32>) outs(%0 : tensor<16xi8>)
      round_mode = <trunc> -> tensor<16xi8>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi8>
  return %1 : tensor<16xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCastLowering_cast_i64_to_i8_with_overflow_mode
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8xi64>) -> tensor<8xi8>
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<8xf32>
// CHECK: %[[CAST0:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<8xi64>) outs(%[[EMPTY0]] : tensor<8xf32>) round_mode = <trunc> -> tensor<8xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<8xf16>
// CHECK: %[[CAST1:.*]] = hivm.hir.vcast ins(%[[CAST0]] : tensor<8xf32>) outs(%[[EMPTY1]] : tensor<8xf16>) round_mode = <trunc> -> tensor<8xf16>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<8xi8>
// CHECK: %[[CAST2:.*]] = hivm.hir.vcast ins(%[[CAST1]] : tensor<8xf16>) outs(%[[EMPTY2]] : tensor<8xi8>) round_mode = <trunc> -> tensor<8xi8>
// CHECK: return %[[CAST2]]
func.func @test_NormalizeCastLowering_cast_i64_to_i8_with_overflow_mode(%arg0: tensor<8xi64>) -> tensor<8xi8> {
  %0 = tensor.empty() : tensor<8xi8>
  %1 = hivm.hir.vcast ins(%arg0 : tensor<8xi64>) outs(%0 : tensor<8xi8>)
      round_mode = <rint> -> tensor<8xi8>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<8xi8>
  return %1 : tensor<8xi8>
}
