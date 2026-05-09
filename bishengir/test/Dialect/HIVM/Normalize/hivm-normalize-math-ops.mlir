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

// -----

// CHECK-LABEL: func.func @test_NormalizeVLog2_hivm_vlog2
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1024xf32>)
// CHECK: %[[CST:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[LN:.*]] = hivm.hir.vln ins(%[[ARG0]] : tensor<1024xf32>) outs(%[[EMPTY0]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[VBRC:.*]] = hivm.hir.vbrc ins(%[[CST]] : f32) outs(%[[EMPTY0]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[LNBASE:.*]] = hivm.hir.vln ins(%[[VBRC]] : tensor<1024xf32>) outs(%[[EMPTY0]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[LN]], %[[LNBASE]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[EMPTY1]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: return %[[DIV]]
func.func @test_NormalizeVLog2_hivm_vlog2(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %0 = tensor.empty() : tensor<1024xf32>
  %1 = hivm.hir.vlog2 ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
  return %1 : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeVLog2_hivm_vlog2_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1024xf16>)
// CHECK: %[[CST:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[CASTIN:.*]] = hivm.hir.vcast
// CHECK: %[[LN:.*]] = hivm.hir.vln ins(%[[CASTIN]] : tensor<1024xf32>)
// CHECK: %[[VBRC:.*]] = hivm.hir.vbrc ins(%[[CST]] : f32)
// CHECK: %[[LNBASE:.*]] = hivm.hir.vln ins(%[[VBRC]] : tensor<1024xf32>)
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[LN]], %[[LNBASE]] : tensor<1024xf32>, tensor<1024xf32>)
// CHECK: %[[CASTOUT:.*]] = hivm.hir.vcast
// CHECK: return %[[CASTOUT]]
func.func @test_NormalizeVLog2_hivm_vlog2_f16(%arg0: tensor<1024xf16>) -> tensor<1024xf16> {
  %0 = tensor.empty() : tensor<1024xf16>
  %1 = hivm.hir.vlog2 ins(%arg0 : tensor<1024xf16>) outs(%0 : tensor<1024xf16>) -> tensor<1024xf16>
  return %1 : tensor<1024xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeVLog10_hivm_vlog10
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1024xf32>)
// CHECK: %[[CST:.*]] = arith.constant 1.000000e+01 : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[LN:.*]] = hivm.hir.vln ins(%[[ARG0]] : tensor<1024xf32>) outs(%[[EMPTY0]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[VBRC:.*]] = hivm.hir.vbrc ins(%[[CST]] : f32) outs(%[[EMPTY0]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[LNBASE:.*]] = hivm.hir.vln ins(%[[VBRC]] : tensor<1024xf32>) outs(%[[EMPTY0]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[LN]], %[[LNBASE]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[EMPTY1]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: return %[[DIV]]
func.func @test_NormalizeVLog10_hivm_vlog10(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %0 = tensor.empty() : tensor<1024xf32>
  %1 = hivm.hir.vlog10 ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
  return %1 : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeVLog10_hivm_vlog10_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1024xf16>)
// CHECK: %[[CST:.*]] = arith.constant 1.000000e+01 : f32
// CHECK: %[[CASTIN:.*]] = hivm.hir.vcast
// CHECK: %[[LN:.*]] = hivm.hir.vln ins(%[[CASTIN]] : tensor<1024xf32>)
// CHECK: %[[VBRC:.*]] = hivm.hir.vbrc ins(%[[CST]] : f32)
// CHECK: %[[LNBASE:.*]] = hivm.hir.vln ins(%[[VBRC]] : tensor<1024xf32>)
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[LN]], %[[LNBASE]] : tensor<1024xf32>, tensor<1024xf32>)
// CHECK: %[[CASTOUT:.*]] = hivm.hir.vcast
// CHECK: return %[[CASTOUT]]
func.func @test_NormalizeVLog10_hivm_vlog10_f16(%arg0: tensor<1024xf16>) -> tensor<1024xf16> {
  %0 = tensor.empty() : tensor<1024xf16>
  %1 = hivm.hir.vlog10 ins(%arg0 : tensor<1024xf16>) outs(%0 : tensor<1024xf16>) -> tensor<1024xf16>
  return %1 : tensor<1024xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeVLog1p_hivm_vlog1p
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1024xf32>)
// CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[ADD:.*]] = hivm.hir.vadd ins(%[[ARG0]], %[[CST]] : tensor<1024xf32>, f32) outs(%[[EMPTY0]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[LN:.*]] = hivm.hir.vln ins(%[[ADD]] : tensor<1024xf32>) outs(%[[EMPTY1]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: return %[[LN]]
func.func @test_NormalizeVLog1p_hivm_vlog1p(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %0 = tensor.empty() : tensor<1024xf32>
  %1 = hivm.hir.vlog1p ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
  return %1 : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeVLog1p_hivm_vlog1p_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1024xf16>)
// CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f16
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[ADD:.*]] = hivm.hir.vadd ins(%[[ARG0]], %[[CST]] : tensor<1024xf16>, f16) outs(%[[EMPTY0]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[LN:.*]] = hivm.hir.vln ins(%[[ADD]] : tensor<1024xf16>) outs(%[[EMPTY1]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: return %[[LN]]
func.func @test_NormalizeVLog1p_hivm_vlog1p_f16(%arg0: tensor<1024xf16>) -> tensor<1024xf16> {
  %0 = tensor.empty() : tensor<1024xf16>
  %1 = hivm.hir.vlog1p ins(%arg0 : tensor<1024xf16>) outs(%0 : tensor<1024xf16>) -> tensor<1024xf16>
  return %1 : tensor<1024xf16>
}
