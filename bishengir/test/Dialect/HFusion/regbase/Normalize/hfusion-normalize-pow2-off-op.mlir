// RUN: bishengir-opt --hfusion-normalize-ops="use-regbase=true enable-div-rem-pow2=false" %s -split-input-file -verify-diagnostics | FileCheck %s
//
// option off: with enable-div-rem-pow2=false the strength reduction is fully disabled,
// mod/div are preserved bit-for-bit, verified on the A5 target.

// CHECK-LABEL: func.func @test_modui_pow2_off
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>}
// CHECK-NOT:   binary_fn<vand>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_modui_pow2_off(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 8 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// CHECK-LABEL: func.func @test_mod_pow2_off
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
// CHECK-NOT:   binary_fn<shrsi>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_mod_pow2_off(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 8 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// CHECK-LABEL: func.func @test_div_pow2_off
// CHECK:       linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK-NOT:   binary_fn<shr
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_div_pow2_off(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 8 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// divui with the option off: must not be rewritten to shrui.
// CHECK-LABEL: func.func @test_divui_pow2_off
// CHECK:       linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>}
// CHECK-NOT:   binary_fn<shrui>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_divui_pow2_off(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 8 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}
