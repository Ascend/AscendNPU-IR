// RUN: bishengir-opt --hivm-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_scalar_like_tensor_vadd
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1xf32>) -> tensor<1xf32>
// CHECK: %[[CST:.*]] = arith.constant 5.000000e-01 : f32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[VADD:.*]] = hivm.hir.vadd ins(%[[ARG0]], %[[CST]] : tensor<1xf32>, f32) outs(%[[EMPTY]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK: return %[[VADD]] : tensor<1xf32>
func.func @test_scalar_like_tensor_vadd(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %cst = arith.constant dense<0.5> : tensor<1xf32>
  %0 = tensor.empty() : tensor<1xf32>
  %1 = hivm.hir.vadd ins(%arg0, %cst : tensor<1xf32>, tensor<1xf32>)
      outs(%0 : tensor<1xf32>) -> tensor<1xf32>
  return %1 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func.func @test_scalar_like_tensor_vmul
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1xf32>) -> tensor<1xf32>
// CHECK: %[[CST:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[VMUL:.*]] = hivm.hir.vmul ins(%[[ARG0]], %[[CST]] : tensor<1xf32>, f32) outs(%[[EMPTY]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK: return %[[VMUL]] : tensor<1xf32>
func.func @test_scalar_like_tensor_vmul(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %cst = arith.constant dense<2.0> : tensor<1xf32>
  %0 = tensor.empty() : tensor<1xf32>
  %1 = hivm.hir.vmul ins(%arg0, %cst : tensor<1xf32>, tensor<1xf32>)
      outs(%0 : tensor<1xf32>) -> tensor<1xf32>
  return %1 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func.func @test_scalar_like_tensor_vmax
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1xf32>) -> tensor<1xf32>
// CHECK: %[[CST:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[VMAX:.*]] = hivm.hir.vmax ins(%[[ARG0]], %[[CST]] : tensor<1xf32>, f32) outs(%[[EMPTY]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK: return %[[VMAX]] : tensor<1xf32>
func.func @test_scalar_like_tensor_vmax(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %cst = arith.constant dense<2.0> : tensor<1xf32>
  %0 = tensor.empty() : tensor<1xf32>
  %1 = hivm.hir.vmax ins(%arg0, %cst : tensor<1xf32>, tensor<1xf32>)
      outs(%0 : tensor<1xf32>) -> tensor<1xf32>
  return %1 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func.func @test_scalar_like_tensor_vmin
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1xf32>) -> tensor<1xf32>
// CHECK: %[[CST:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[VMIN:.*]] = hivm.hir.vmin ins(%[[ARG0]], %[[CST]] : tensor<1xf32>, f32) outs(%[[EMPTY]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK: return %[[VMIN]] : tensor<1xf32>
func.func @test_scalar_like_tensor_vmin(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %cst = arith.constant dense<2.0> : tensor<1xf32>
  %0 = tensor.empty() : tensor<1xf32>
  %1 = hivm.hir.vmin ins(%arg0, %cst : tensor<1xf32>, tensor<1xf32>)
      outs(%0 : tensor<1xf32>) -> tensor<1xf32>
  return %1 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func.func @test_scalar_like_tensor_vcast
// CHECK: %[[CST:.*]] = arith.constant dense<1.250000e+00> : tensor<1xf32>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1xf16>
// CHECK: %[[VCAST:.*]] = hivm.hir.vcast ins(%[[CST]] : tensor<1xf32>) outs(%[[EMPTY]] : tensor<1xf16>) round_mode = <trunc> -> tensor<1xf16>
// CHECK: return %[[VCAST]] : tensor<1xf16>
func.func @test_scalar_like_tensor_vcast() -> tensor<1xf16> {
  %cst = arith.constant dense<1.25> : tensor<1xf32>
  %0 = tensor.empty() : tensor<1xf16>
  %1 = hivm.hir.vcast ins(%cst : tensor<1xf32>) outs(%0 : tensor<1xf16>)
      round_mode = <trunc> -> tensor<1xf16>
  return %1 : tensor<1xf16>
}

// -----

// CHECK-LABEL: func.func @test_scalar_like_tensor_vcmp
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1xi64>) -> tensor<1xi1>
// CHECK: %[[CST:.*]] = arith.constant 5 : i64
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1xi1>
// CHECK: %[[VCMP:.*]] = hivm.hir.vcmp ins(%[[ARG0]], %[[CST]] : tensor<1xi64>, i64) outs(%[[EMPTY]] : tensor<1xi1>) -> tensor<1xi1>
// CHECK: return %[[VCMP]] : tensor<1xi1>
func.func @test_scalar_like_tensor_vcmp(%arg0: tensor<1xi64>) -> tensor<1xi1> {
  %cst = arith.constant dense<5> : tensor<1xi64>
  %0 = tensor.empty() : tensor<1xi1>
  %1 = hivm.hir.vcmp ins(%arg0, %cst : tensor<1xi64>, tensor<1xi64>)
      outs(%0 : tensor<1xi1>) compare_mode = <eq> -> tensor<1xi1>
  return %1 : tensor<1xi1>
}

// -----

// CHECK-LABEL: func.func @test_scalar_like_tensor_vshr
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1xi32>) -> tensor<1xi32>
// CHECK: %[[CST:.*]] = arith.constant 3 : i32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1xi32>
// CHECK: %[[VSHR:.*]] = hivm.hir.vshr ins(%[[ARG0]], %[[CST]] : tensor<1xi32>, i32) outs(%[[EMPTY]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK: return %[[VSHR]] : tensor<1xi32>
func.func @test_scalar_like_tensor_vshr(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  %cst = arith.constant dense<3> : tensor<1xi32>
  %0 = tensor.empty() : tensor<1xi32>
  %1 = hivm.hir.vshr ins(%arg0, %cst : tensor<1xi32>, tensor<1xi32>)
      outs(%0 : tensor<1xi32>) -> tensor<1xi32>
  return %1 : tensor<1xi32>
}

// -----

// `hivm.hir.vshl` is intentionally not part of HIVM scalar-normalize. Its rhs
// is `ScalarOnlyTrait<1>`, so a shaped rhs is not legal HIVM IR to begin with;
// there is no verifier-clean pure HIVM input on which this pass could "repair"
// operand 1 from tensor to scalar.

// -----

// CHECK-LABEL: func.func @test_scalar_like_tensor_vsel
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1xi1>) -> tensor<1xi32>
// CHECK: %[[FALSE:.+]] = arith.constant 7 : i32
// CHECK: %[[TRUE:.+]] = arith.constant 5 : i32
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<1xi32>
// CHECK: %[[VSEL:.+]] = hivm.hir.vsel ins(%[[ARG0]], %[[TRUE]], %[[FALSE]] : tensor<1xi1>, i32, i32) outs(%[[EMPTY]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK: return %[[VSEL]] : tensor<1xi32>
func.func @test_scalar_like_tensor_vsel(%arg0: tensor<1xi1>) -> tensor<1xi32> {
  %truev = arith.constant dense<5> : tensor<1xi32>
  %falsev = arith.constant dense<7> : tensor<1xi32>
  %0 = tensor.empty() : tensor<1xi32>
  %1 = hivm.hir.vsel ins(%arg0, %truev, %falsev : tensor<1xi1>, tensor<1xi32>,
      tensor<1xi32>) outs(%0 : tensor<1xi32>) -> tensor<1xi32>
  return %1 : tensor<1xi32>
}

// -----

// CHECK-LABEL: func.func @test_scalar_like_tensor_hivm_broadcast
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[CST:.*]] = arith.constant 5.000000e-01 : f32
// CHECK: %[[VBRC:.*]] = hivm.hir.vbrc ins(%[[CST]] : f32) outs(%[[ARG0]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK: return %[[VBRC]] : tensor<4xf32>
func.func @test_scalar_like_tensor_hivm_broadcast(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %cst = arith.constant dense<0.5> : tensor<1xf32>
  %0 = hivm.hir.vbrc ins(%cst : tensor<1xf32>)
      outs(%arg0 : tensor<4xf32>) broadcast_dims = [0] -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func.func @test_nondense_hivm_broadcast_no_rewrite_without_regbase
// CHECK: hivm.hir.vbrc ins(%arg0 : tensor<1x1xf32>) outs(%arg1 : tensor<2x2xf32>) broadcast_dims = [0, 1] -> tensor<2x2xf32>
// CHECK-NOT: tensor.extract
func.func @test_nondense_hivm_broadcast_no_rewrite_without_regbase(
    %arg0: tensor<1x1xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = hivm.hir.vbrc ins(%arg0 : tensor<1x1xf32>)
      outs(%arg1 : tensor<2x2xf32>) broadcast_dims = [0, 1] -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func.func @test_nondense_hivm_broadcast_regbase
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x1xf32>, %[[ARG1:.*]]: tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[EXTRACT:.*]] = tensor.extract %[[ARG0]][%[[C0]], %[[C0]]] : tensor<1x1xf32>
// CHECK: %[[VBRC:.*]] = hivm.hir.vbrc ins(%[[EXTRACT]] : f32) outs(%[[ARG1]] : tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK: return %[[VBRC]] : tensor<2x2xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_nondense_hivm_broadcast_regbase(
      %arg0: tensor<1x1xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = hivm.hir.vbrc ins(%arg0 : tensor<1x1xf32>)
        outs(%arg1 : tensor<2x2xf32>) broadcast_dims = [0, 1] -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}
