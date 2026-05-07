// RUN: bishengir-opt --hivm-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_NormalizeExp2Op_f32
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf32>, %[[DST:.*]]: tensor<16xf32>)
// CHECK: %[[LN2:.*]] = arith.constant
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[MUL:.*]] = hivm.hir.vmul ins(%[[ARG0]], %[[LN2]] : tensor<16xf32>, f32) outs(%[[EMPTY]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EXP_EMPTY:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[EXP:.*]] = hivm.hir.vexp ins(%[[MUL]] : tensor<16xf32>) outs(%[[EXP_EMPTY]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: return %[[EXP]]
func.func @test_NormalizeExp2Op_f32(%src : tensor<16xf32>, %dst : tensor<16xf32>) -> tensor<16xf32> {
  %ret = hivm.hir.vexp2 ins(%src : tensor<16xf32>) outs(%dst : tensor<16xf32>) -> tensor<16xf32>
  return %ret : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeExp2Op_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf16>, %[[DST:.*]]: tensor<16xf16>)
// CHECK: %[[LN2:.*]] = arith.constant
// CHECK: %[[CAST_DST:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[CAST:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<16xf16>) outs(%[[CAST_DST]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[MUL_EMPTY:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[MUL:.*]] = hivm.hir.vmul ins(%[[CAST]], %[[LN2]] : tensor<16xf32>, f32) outs(%[[MUL_EMPTY]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EXP_EMPTY:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[EXP:.*]] = hivm.hir.vexp ins(%[[MUL]] : tensor<16xf32>) outs(%[[EXP_EMPTY]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[CAST_BACK:.*]] = hivm.hir.vcast ins(%[[EXP]] : tensor<16xf32>) outs(%{{.*}} : tensor<16xf16>) round_mode = <round> -> tensor<16xf16>
// CHECK: return %[[CAST_BACK]]
func.func @test_NormalizeExp2Op_f16(%src : tensor<16xf16>, %dst : tensor<16xf16>) -> tensor<16xf16> {
  %ret = hivm.hir.vexp2 ins(%src : tensor<16xf16>) outs(%dst : tensor<16xf16>) -> tensor<16xf16>
  return %ret : tensor<16xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeErfOp_f32
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8xf32>, %[[DST:.*]]: tensor<8xf32>)
// CHECK-DAG: %[[LOWER:.*]] = arith.constant -3.920000
// CHECK-DAG: %[[UPPER:.*]] = arith.constant 3.920000
// CHECK: %[[MIN_EMPTY:.*]] = tensor.empty() : tensor<8xf32>
// CHECK: %[[MIN:.*]] = hivm.hir.vmin ins(%[[ARG0]], %[[UPPER]] : tensor<8xf32>, f32) outs(%[[MIN_EMPTY]] : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[MAX:.*]] = hivm.hir.vmax ins(%[[MIN]], %[[LOWER]] : tensor<8xf32>, f32) outs(%[[MIN_EMPTY]] : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[SQUARE_EMPTY:.*]] = tensor.empty() : tensor<8xf32>
// CHECK: %[[SQUARE:.*]] = hivm.hir.vmul ins(%[[MAX]], %[[MAX]] : tensor<8xf32>, tensor<8xf32>) outs(%[[SQUARE_EMPTY]] : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%{{.*}}, %{{.*}} : tensor<8xf32>, tensor<8xf32>) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK: return %[[DIV]]
func.func @test_NormalizeErfOp_f32(%src : tensor<8xf32>, %dst : tensor<8xf32>) -> tensor<8xf32> {
  %ret = hivm.hir.verf ins(%src : tensor<8xf32>) outs(%dst : tensor<8xf32>) -> tensor<8xf32>
  return %ret : tensor<8xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeErfOp_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8xf16>, %[[DST:.*]]: tensor<8xf16>)
// CHECK-DAG: %[[LOWER:.*]] = arith.constant -3.920000
// CHECK-DAG: %[[UPPER:.*]] = arith.constant 3.920000
// CHECK: %[[CAST_DST:.*]] = tensor.empty() : tensor<8xf32>
// CHECK: %[[CAST:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<8xf16>) outs(%[[CAST_DST]] : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[MIN_EMPTY:.*]] = tensor.empty() : tensor<8xf32>
// CHECK: %[[MIN:.*]] = hivm.hir.vmin ins(%[[CAST]], %[[UPPER]] : tensor<8xf32>, f32) outs(%[[MIN_EMPTY]] : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[MAX:.*]] = hivm.hir.vmax ins(%[[MIN]], %[[LOWER]] : tensor<8xf32>, f32) outs(%[[MIN_EMPTY]] : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[SQUARE_EMPTY:.*]] = tensor.empty() : tensor<8xf32>
// CHECK: %[[SQUARE:.*]] = hivm.hir.vmul ins(%[[MAX]], %[[MAX]] : tensor<8xf32>, tensor<8xf32>) outs(%[[SQUARE_EMPTY]] : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%{{.*}}, %{{.*}} : tensor<8xf32>, tensor<8xf32>) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[CAST_BACK:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<8xf32>) outs(%{{.*}} : tensor<8xf16>) round_mode = <round> -> tensor<8xf16>
// CHECK: return %[[CAST_BACK]]
func.func @test_NormalizeErfOp_f16(%src : tensor<8xf16>, %dst : tensor<8xf16>) -> tensor<8xf16> {
  %ret = hivm.hir.verf ins(%src : tensor<8xf16>) outs(%dst : tensor<8xf16>) -> tensor<8xf16>
  return %ret : tensor<8xf16>
}
