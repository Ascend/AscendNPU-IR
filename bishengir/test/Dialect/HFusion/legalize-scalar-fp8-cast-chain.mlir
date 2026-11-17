// RUN: bishengir-opt -hfusion-legalize-scalar --convert-arith-to-hfusion %s -split-input-file | FileCheck %s

// Pins the conversion chain the scalar fp8 cast legalization depends on:
// LegalizeScalar wraps the scalar cast into a length-1 tensor, and
// ArithToHFusion must then pick that tensor cast up and emit an hfusion.cast
// with round-to-nearest-even. Checking only the wrapping would still pass if
// the second step stopped matching, which is exactly the case that leaves the
// cast for the backend to mis-lower.

// CHECK-LABEL: func.func @chain_truncf_f32_to_f8e4m3
// CHECK: %[[FE:.*]] = tensor.from_elements %arg0 : tensor<1xf32>
// CHECK: %[[OUT:.*]] = tensor.empty() : tensor<1xf8E4M3FN>
// CHECK: %[[CAST:.*]] = hfusion.cast {{.*}}round_mode = #hfusion.round_mode<rint>{{.*}} ins(%[[FE]] : tensor<1xf32>) outs(%[[OUT]] : tensor<1xf8E4M3FN>) -> tensor<1xf8E4M3FN>
// CHECK: tensor.extract %[[CAST]]{{.*}} : tensor<1xf8E4M3FN>
// CHECK-NOT: arith.truncf
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @chain_truncf_f32_to_f8e4m3(%a: f32) -> f8E4M3FN {
    %0 = arith.truncf %a : f32 to f8E4M3FN
    return %0 : f8E4M3FN
  }
}

// -----

// CHECK-LABEL: func.func @chain_extf_f8e4m3_to_f32
// CHECK: %[[FE:.*]] = tensor.from_elements %arg0 : tensor<1xf8E4M3FN>
// CHECK: %[[OUT:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[CAST:.*]] = hfusion.cast {{.*}}round_mode = #hfusion.round_mode<rint>{{.*}} ins(%[[FE]] : tensor<1xf8E4M3FN>) outs(%[[OUT]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK: tensor.extract %[[CAST]]{{.*}} : tensor<1xf32>
// CHECK-NOT: arith.extf
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @chain_extf_f8e4m3_to_f32(%a: f8E4M3FN) -> f32 {
    %0 = arith.extf %a : f8E4M3FN to f32
    return %0 : f32
  }
}

// -----

// CHECK-LABEL: func.func @chain_truncf_f32_to_f8e5m2
// CHECK: hfusion.cast {{.*}}round_mode = #hfusion.round_mode<rint>{{.*}} ins({{.*}} : tensor<1xf32>) outs({{.*}} : tensor<1xf8E5M2>) -> tensor<1xf8E5M2>
// CHECK-NOT: arith.truncf
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @chain_truncf_f32_to_f8e5m2(%a: f32) -> f8E5M2 {
    %0 = arith.truncf %a : f32 to f8E5M2
    return %0 : f8E5M2
  }
}

// -----

// f64 <-> fp8 is outside the set ArithToHFusion::selectRoundMode supports, so
// LegalizeScalar must not wrap it. This pins that the pair stays a scalar cast
// instead of being promoted into that pass's report_fatal_error.
// CHECK-LABEL: func.func @chain_truncf_f64_to_f8e4m3_untouched
// CHECK-NOT: tensor.from_elements
// CHECK-NOT: hfusion.cast
// CHECK: arith.truncf {{.*}} : f64 to f8E4M3FN
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @chain_truncf_f64_to_f8e4m3_untouched(%a: f64) -> f8E4M3FN {
    %0 = arith.truncf %a : f64 to f8E4M3FN
    return %0 : f8E4M3FN
  }
}
