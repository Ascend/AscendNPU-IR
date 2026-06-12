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

// -----

// CHECK-LABEL: func.func @legalize_cast_signed_to_select
// CHECK-NOT: hfusion.cast
// CHECK: %[[ZERO:.+]] = arith.constant 0.000000e+00 : f16
// CHECK: %[[NEGONE:.+]] = arith.constant -1.000000e+00 : f16
// CHECK: %[[NEG_BUF:.+]] = tensor.empty()
// CHECK: %[[NEG_FILL:.+]] = linalg.fill ins(%[[NEGONE]]
// CHECK: %[[ZERO_BUF:.+]] = tensor.empty()
// CHECK: %[[ZERO_FILL:.+]] = linalg.fill ins(%[[ZERO]]
// CHECK: %[[SELECT:.+]] = hfusion.select
// CHECK-SAME: ins(%[[CMP:.+]], %[[NEG_FILL]], %[[ZERO_FILL]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_957c">} {
  func.func @legalize_cast_signed_to_select(%arg0: tensor<4096xf32>, %arg1: tensor<4096xf32>) -> tensor<4096xi1> {
    %cst = arith.constant -1.000000e+00 : f16
    %0 = tensor.empty() : tensor<4096xi1>
    %1 = hfusion.compare {compare_fn = #hfusion.compare_fn<vgt>} ins(%arg0, %arg1 : tensor<4096xf32>, tensor<4096xf32>) outs(%0 : tensor<4096xi1>) -> tensor<4096xi1>
    %2 = tensor.empty() : tensor<4096xf16>
    %3 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<trunc>, unsigned_mode = #hfusion.unsigned_mode<si2si>} ins(%1 : tensor<4096xi1>) outs(%2 : tensor<4096xf16>) -> tensor<4096xf16>
    %4 = tensor.empty() : tensor<4096xf16>
    %5 = linalg.fill ins(%cst : f16) outs(%4 : tensor<4096xf16>) -> tensor<4096xf16>
    %6 = tensor.empty() : tensor<4096xi1>
    %7 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%3, %5 : tensor<4096xf16>, tensor<4096xf16>) outs(%6 : tensor<4096xi1>) -> tensor<4096xi1>
    return %7 : tensor<4096xi1>
  }
}