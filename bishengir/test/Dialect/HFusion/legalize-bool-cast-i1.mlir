// RUN: bishengir-opt %s -hfusion-legalize-bool -split-input-file -verify-diagnostics | FileCheck %s

// Test that hfusion.cast i1→f16 is converted to hfusion.select(cond, 1.0, 0.0)
// rather than to arith.uitofp which would later fail in ConvertArithToHIVMAVE
// (VFUIntToFpOp only accepts i8/i64 input element types).
// Uses Ascend950PR_957b target to activate the RegBasedArch-only pattern.
//
// CHECK-LABEL: func.func @cast_i1_to_f16
module attributes {hacc.target = #hacc.target<"Ascend950PR_957b">} {
  func.func @cast_i1_to_f16(%arg0: tensor<1xi1>) -> tensor<1xf16> {
    %0 = tensor.empty() : tensor<1xf16>
    // CHECK: linalg.fill
    // CHECK: linalg.fill
    // CHECK: hfusion.select
    // CHECK-NOT: uitofp
    %1 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<trunc>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%arg0 : tensor<1xi1>) outs(%0 : tensor<1xf16>) -> tensor<1xf16>
    return %1 : tensor<1xf16>
  }
}

// -----

// Test that hfusion.cast i1→f32 is also converted to hfusion.select.
//
// CHECK-LABEL: func.func @cast_i1_to_f32
module attributes {hacc.target = #hacc.target<"Ascend950PR_957b">} {
  func.func @cast_i1_to_f32(%arg0: tensor<1xi1>) -> tensor<1xf32> {
    %0 = tensor.empty() : tensor<1xf32>
    // CHECK: linalg.fill
    // CHECK: linalg.fill
    // CHECK: hfusion.select
    // CHECK-NOT: uitofp
    %1 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<trunc>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%arg0 : tensor<1xi1>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    return %1 : tensor<1xf32>
  }
}

// -----

// Test that hfusion.cast i1→f16 with cast_unsigned is also converted.
//
// CHECK-LABEL: func.func @cast_i1_to_f16_unsigned
module attributes {hacc.target = #hacc.target<"Ascend950PR_957b">} {
  func.func @cast_i1_to_f16_unsigned(%arg0: tensor<1xi1>) -> tensor<1xf16> {
    %0 = tensor.empty() : tensor<1xf16>
    // CHECK: linalg.fill
    // CHECK: linalg.fill
    // CHECK: hfusion.select
    %1 = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<trunc>, unsigned_mode = #hfusion.unsigned_mode<ui2ui>} ins(%arg0 : tensor<1xi1>) outs(%0 : tensor<1xf16>) -> tensor<1xf16>
    return %1 : tensor<1xf16>
  }
}

// -----

// Test that hfusion.cast i1→f16 with dynamic shape is also converted.
//
// CHECK-LABEL: func.func @cast_i1_to_f16_dynamic
module attributes {hacc.target = #hacc.target<"Ascend950PR_957b">} {
  func.func @cast_i1_to_f16_dynamic(%arg0: tensor<?x?xi1>, %arg1: index) -> tensor<?x?xf16> {
    %0 = tensor.empty(%arg1, %arg1) : tensor<?x?xf16>
    // CHECK: linalg.fill
    // CHECK: linalg.fill
    // CHECK: hfusion.select
    // CHECK-NOT: uitofp
    %1 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<trunc>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%arg0 : tensor<?x?xi1>) outs(%0 : tensor<?x?xf16>) -> tensor<?x?xf16>
    return %1 : tensor<?x?xf16>
  }
}
