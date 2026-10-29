// RUN: bishengir-opt --hfusion-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s
//
// Membased (Ascend910B1) control: the membased pipeline uses the legacy
// monolithic populateNormalizeHFusionPatterns (the RUN line has no use-regbase), so these pow2
// regbase patterns are not registered: mod is preserved, while signed i32
// division has its own membase constant-divisor fast path.
// Unlike the Ascend310B4 cases (regbased, patterns loaded but rejected by the
// 950 gate), 910B is membased: the patterns are never even registered.

// CHECK-LABEL: func.func @test_mod_pow2_membased
// the membased path annotates mod with already_handle_mod_zero, hence the {{.*}}
// CHECK:       binary_fn<mod>}
// mod surviving proves pow2 did not fire; the membased mod-zero machinery emits an i1 vand, hence no NOT here
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @test_mod_pow2_membased(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 8 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// Membase signed i32 division by 8 uses an exact integer fast path.
// Keep checking the behavior separately from the regbase pow2 optimization.
// CHECK-LABEL: func.func @test_div_pow2_membased
// CHECK-NOT:   f32
// CHECK:       arith.constant 3 : i32
// CHECK:       #hfusion.binary_fn<shrui>
// CHECK-NOT:   binary_fn<vand>
// CHECK-NOT:   f32
// CHECK:       return
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @test_div_pow2_membased(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 8 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}
