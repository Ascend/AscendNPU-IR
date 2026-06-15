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
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[CASTIN:.*]] = hivm.hir.vcast ins(%[[ADD]] : tensor<1024xf16>) outs(%[[EMPTY2]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[CASTOUT_INIT:.*]] = hivm.hir.vcast ins(%[[EMPTY1]] : tensor<1024xf16>) outs(%[[EMPTY3]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[LN:.*]] = hivm.hir.vln ins(%[[CASTIN]] : tensor<1024xf32>) outs(%[[CASTOUT_INIT]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY4:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[CASTBACK:.*]] = hivm.hir.vcast ins(%[[LN]] : tensor<1024xf32>) outs(%[[EMPTY4]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: return %[[CASTBACK]]
func.func @test_NormalizeVLog1p_hivm_vlog1p_f16(%arg0: tensor<1024xf16>) -> tensor<1024xf16> {
  %0 = tensor.empty() : tensor<1024xf16>
  %1 = hivm.hir.vlog1p ins(%arg0 : tensor<1024xf16>) outs(%0 : tensor<1024xf16>) -> tensor<1024xf16>
  return %1 : tensor<1024xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeVExpM1_hivm_vexpm1
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1024xf32>)
// CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[EXP:.*]] = hivm.hir.vexp ins(%[[ARG0]] : tensor<1024xf32>) outs(%[[EMPTY0]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[SUB:.*]] = hivm.hir.vsub ins(%[[EXP]], %[[CST]] : tensor<1024xf32>, f32) outs(%[[EMPTY1]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: return %[[SUB]]
func.func @test_NormalizeVExpM1_hivm_vexpm1(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %0 = tensor.empty() : tensor<1024xf32>
  %1 = hivm.hir.vexpm1 ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
  return %1 : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeVExpM1_hivm_vexpm1_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1024xf16>)
// CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[CASTIN:.*]] = hivm.hir.vcast
// CHECK: %[[EXP:.*]] = hivm.hir.vexp ins(%[[CASTIN]] : tensor<1024xf32>)
// CHECK: %[[SUB:.*]] = hivm.hir.vsub ins(%[[EXP]], %[[CST]] : tensor<1024xf32>, f32)
// CHECK: %[[CASTOUT:.*]] = hivm.hir.vcast
// CHECK: return %[[CASTOUT]]
func.func @test_NormalizeVExpM1_hivm_vexpm1_f16(%arg0: tensor<1024xf16>) -> tensor<1024xf16> {
  %0 = tensor.empty() : tensor<1024xf16>
  %1 = hivm.hir.vexpm1 ins(%arg0 : tensor<1024xf16>) outs(%0 : tensor<1024xf16>) -> tensor<1024xf16>
  return %1 : tensor<1024xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeVIlogb_hivm_vilogb
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf32>)
// CHECK: %[[CST:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[ABS_EMPTY:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[ABS:.*]] = hivm.hir.vabs ins(%[[ARG0]] : tensor<16xf32>) outs(%[[ABS_EMPTY]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[LOG_EMPTY0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[LOG_EMPTY1:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[LN1:.*]] = hivm.hir.vln ins(%[[ABS]] : tensor<16xf32>) outs(%[[LOG_EMPTY0]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[VBRC:.*]] = hivm.hir.vbrc ins(%[[CST]] : f32) outs(%[[LOG_EMPTY0]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[LN2:.*]] = hivm.hir.vln ins(%[[VBRC]] : tensor<16xf32>) outs(%[[LOG_EMPTY0]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[LN1]], %[[LN2]] : tensor<16xf32>, tensor<16xf32>) outs(%[[LOG_EMPTY1]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[CAST:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<16xf32>) outs({{.*}} : tensor<16xf32>) round_mode = <floor> -> tensor<16xf32>
// CHECK: return %[[CAST]]
func.func @test_NormalizeVIlogb_hivm_vilogb(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  %0 = tensor.empty() : tensor<16xf32>
  %1 = hivm.hir.vilogb ins(%arg0 : tensor<16xf32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
  return %1 : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeVIlogb_hivm_vilogb_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf16>)
// CHECK: %[[CST:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[ABS_EMPTY:.*]] = tensor.empty() : tensor<16xf16>
// CHECK: %[[ABS:.*]] = hivm.hir.vabs ins(%[[ARG0]] : tensor<16xf16>) outs(%[[ABS_EMPTY]] : tensor<16xf16>) -> tensor<16xf16>
// CHECK: %[[CAST_EMPTY0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[CASTIN:.*]] = hivm.hir.vcast ins(%[[ABS]] : tensor<16xf16>) outs(%[[CAST_EMPTY0]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[LOG_EMPTY0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[LOG_EMPTY1:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[LN1:.*]] = hivm.hir.vln ins(%[[CASTIN]] : tensor<16xf32>) outs(%[[LOG_EMPTY0]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[VBRC:.*]] = hivm.hir.vbrc ins(%[[CST]] : f32) outs(%[[LOG_EMPTY0]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[LN2:.*]] = hivm.hir.vln ins(%[[VBRC]] : tensor<16xf32>) outs(%[[LOG_EMPTY0]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[LN1]], %[[LN2]] : tensor<16xf32>, tensor<16xf32>) outs(%[[LOG_EMPTY1]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[CAST_EMPTY1:.*]] = tensor.empty() : tensor<16xf16>
// CHECK: %[[CASTROUND:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<16xf32>) outs(%[[CAST_EMPTY1]] : tensor<16xf16>) round_mode = <round> -> tensor<16xf16>
// CHECK: %[[RINT_EMPTY:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[CASTRINT:.*]] = hivm.hir.vcast ins(%[[CASTROUND]] : tensor<16xf16>) outs(%[[RINT_EMPTY]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[FLOOR_F32_EMPTY:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[CASTFLOOR_F32:.*]] = hivm.hir.vcast ins(%[[CASTRINT]] : tensor<16xf32>) outs(%[[FLOOR_F32_EMPTY]] : tensor<16xf32>) round_mode = <floor> -> tensor<16xf32>
// CHECK: %[[CAST_EMPTY2:.*]] = tensor.empty() : tensor<16xf16>
// CHECK: %[[CASTFLOOR_F16:.*]] = hivm.hir.vcast ins(%[[CASTFLOOR_F32]] : tensor<16xf32>) outs(%[[CAST_EMPTY2]] : tensor<16xf16>) round_mode = <floor> -> tensor<16xf16>
// CHECK: return %[[CASTFLOOR_F16]]
func.func @test_NormalizeVIlogb_hivm_vilogb_f16(%arg0: tensor<16xf16>) -> tensor<16xf16> {
  %0 = tensor.empty() : tensor<16xf16>
  %1 = hivm.hir.vilogb ins(%arg0 : tensor<16xf16>) outs(%0 : tensor<16xf16>) -> tensor<16xf16>
  return %1 : tensor<16xf16>
}

// -----
// NormalizeCeilandFloorOp tests for HIVM dialect.

// -----
// Test: VCastOp with ceil round_mode on f16 same-type (non-950)
// Pattern: f16->f32(rint), f32->f32(ceil), f32->f16(ceil)

// CHECK-LABEL: func.func @test_NormalizeVCeilOp_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf16>, %[[ARG1:.*]]: tensor<16xf16>)
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[RINT:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<16xf16>) outs(%[[EMPTY0]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[CEIL_F32:.*]] = hivm.hir.vcast ins(%[[RINT]] : tensor<16xf32>) outs(%[[EMPTY1]] : tensor<16xf32>) round_mode = <ceil> -> tensor<16xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<16xf16>
// CHECK: %[[CEIL_F16:.*]] = hivm.hir.vcast ins(%[[CEIL_F32]] : tensor<16xf32>) outs(%[[EMPTY2]] : tensor<16xf16>) round_mode = <ceil> -> tensor<16xf16>
// CHECK: return %[[CEIL_F16]]
func.func @test_NormalizeVCeilOp_f16(%src : tensor<16xf16>, %dst : tensor<16xf16>) -> tensor<16xf16> {
  %ret = hivm.hir.vcast ins(%src : tensor<16xf16>) outs(%dst : tensor<16xf16>) round_mode = <ceil> -> tensor<16xf16>
  return %ret : tensor<16xf16>
}

// -----
// Test: VCastOp with floor round_mode on f16 same-type (non-950)
// Pattern: f16->f32(rint), f32->f32(floor), f32->f16(floor)

// CHECK-LABEL: func.func @test_NormalizeVFloorOp_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf16>, %[[ARG1:.*]]: tensor<16xf16>)
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[RINT:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<16xf16>) outs(%[[EMPTY0]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[FLOOR_F32:.*]] = hivm.hir.vcast ins(%[[RINT]] : tensor<16xf32>) outs(%[[EMPTY1]] : tensor<16xf32>) round_mode = <floor> -> tensor<16xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<16xf16>
// CHECK: %[[FLOOR_F16:.*]] = hivm.hir.vcast ins(%[[FLOOR_F32]] : tensor<16xf32>) outs(%[[EMPTY2]] : tensor<16xf16>) round_mode = <floor> -> tensor<16xf16>
// CHECK: return %[[FLOOR_F16]]
func.func @test_NormalizeVFloorOp_f16(%src : tensor<16xf16>, %dst : tensor<16xf16>) -> tensor<16xf16> {
  %ret = hivm.hir.vcast ins(%src : tensor<16xf16>) outs(%dst : tensor<16xf16>) round_mode = <floor> -> tensor<16xf16>
  return %ret : tensor<16xf16>
}

// -----
// Test: VCastOp with ceil round_mode on bf16 same-type (non-950)
// Pattern: bf16->f32(rint), f32->f32(ceil), f32->bf16(ceil)

// CHECK-LABEL: func.func @test_NormalizeVCeilOp_bf16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xbf16>, %[[ARG1:.*]]: tensor<16xbf16>)
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[RINT:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<16xbf16>) outs(%[[EMPTY0]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[CEIL_F32:.*]] = hivm.hir.vcast ins(%[[RINT]] : tensor<16xf32>) outs(%[[EMPTY1]] : tensor<16xf32>) round_mode = <ceil> -> tensor<16xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<16xbf16>
// CHECK: %[[CEIL_BF16:.*]] = hivm.hir.vcast ins(%[[CEIL_F32]] : tensor<16xf32>) outs(%[[EMPTY2]] : tensor<16xbf16>) round_mode = <ceil> -> tensor<16xbf16>
// CHECK: return %[[CEIL_BF16]]
func.func @test_NormalizeVCeilOp_bf16(%src : tensor<16xbf16>, %dst : tensor<16xbf16>) -> tensor<16xbf16> {
  %ret = hivm.hir.vcast ins(%src : tensor<16xbf16>) outs(%dst : tensor<16xbf16>) round_mode = <ceil> -> tensor<16xbf16>
  return %ret : tensor<16xbf16>
}

// -----
// Test: VCastOp with floor round_mode on bf16 same-type (non-950)
// Pattern: bf16->f32(rint), f32->f32(floor), f32->bf16(floor)

// CHECK-LABEL: func.func @test_NormalizeVFloorOp_bf16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xbf16>, %[[ARG1:.*]]: tensor<16xbf16>)
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[RINT:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<16xbf16>) outs(%[[EMPTY0]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[FLOOR_F32:.*]] = hivm.hir.vcast ins(%[[RINT]] : tensor<16xf32>) outs(%[[EMPTY1]] : tensor<16xf32>) round_mode = <floor> -> tensor<16xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<16xbf16>
// CHECK: %[[FLOOR_BF16:.*]] = hivm.hir.vcast ins(%[[FLOOR_F32]] : tensor<16xf32>) outs(%[[EMPTY2]] : tensor<16xbf16>) round_mode = <floor> -> tensor<16xbf16>
// CHECK: return %[[FLOOR_BF16]]
func.func @test_NormalizeVFloorOp_bf16(%src : tensor<16xbf16>, %dst : tensor<16xbf16>) -> tensor<16xbf16> {
  %ret = hivm.hir.vcast ins(%src : tensor<16xbf16>) outs(%dst : tensor<16xbf16>) round_mode = <floor> -> tensor<16xbf16>
  return %ret : tensor<16xbf16>
}

// -----
// Test: VCastOp with ceil round_mode on f32 same-type - no normalization needed

// CHECK-LABEL: func.func @test_NoNormalizeVCeilOp_f32
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf32>, %[[ARG1:.*]]: tensor<16xf32>)
// CHECK: %[[RET:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<16xf32>) outs(%[[ARG1]] : tensor<16xf32>) round_mode = <ceil> -> tensor<16xf32>
// CHECK: return %[[RET]]
func.func @test_NoNormalizeVCeilOp_f32(%src : tensor<16xf32>, %dst : tensor<16xf32>) -> tensor<16xf32> {
  %ret = hivm.hir.vcast ins(%src : tensor<16xf32>) outs(%dst : tensor<16xf32>) round_mode = <ceil> -> tensor<16xf32>
  return %ret : tensor<16xf32>
}

// -----
// Test: VCastOp with floor round_mode on f32 same-type - no normalization needed

// CHECK-LABEL: func.func @test_NoNormalizeVFloorOp_f32
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf32>, %[[ARG1:.*]]: tensor<16xf32>)
// CHECK: %[[RET:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<16xf32>) outs(%[[ARG1]] : tensor<16xf32>) round_mode = <floor> -> tensor<16xf32>
// CHECK: return %[[RET]]
func.func @test_NoNormalizeVFloorOp_f32(%src : tensor<16xf32>, %dst : tensor<16xf32>) -> tensor<16xf32> {
  %ret = hivm.hir.vcast ins(%src : tensor<16xf32>) outs(%dst : tensor<16xf32>) round_mode = <floor> -> tensor<16xf32>
  return %ret : tensor<16xf32>
}

// -----
// Test: VCastOp with ceil round_mode on f16 same-type on Ascend950 - no normalization

// CHECK-LABEL: func.func @test_NoNormalizeVCeilOp_ascend950
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf16>, %[[ARG1:.*]]: tensor<16xf16>)
// CHECK: %[[RET:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<16xf16>) outs(%[[ARG1]] : tensor<16xf16>) round_mode = <ceil> -> tensor<16xf16>
// CHECK: return %[[RET]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NoNormalizeVCeilOp_ascend950(%src : tensor<16xf16>, %dst : tensor<16xf16>) -> tensor<16xf16> {
    %ret = hivm.hir.vcast ins(%src : tensor<16xf16>) outs(%dst : tensor<16xf16>) round_mode = <ceil> -> tensor<16xf16>
    return %ret : tensor<16xf16>
  }
}

// -----
// Test: VCastOp with floor round_mode on f16 same-type on Ascend950 - no normalization

// CHECK-LABEL: func.func @test_NoNormalizeVFloorOp_ascend950
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf16>, %[[ARG1:.*]]: tensor<16xf16>)
// CHECK: %[[RET:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<16xf16>) outs(%[[ARG1]] : tensor<16xf16>) round_mode = <floor> -> tensor<16xf16>
// CHECK: return %[[RET]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NoNormalizeVFloorOp_ascend950(%src : tensor<16xf16>, %dst : tensor<16xf16>) -> tensor<16xf16> {
    %ret = hivm.hir.vcast ins(%src : tensor<16xf16>) outs(%dst : tensor<16xf16>) round_mode = <floor> -> tensor<16xf16>
    return %ret : tensor<16xf16>
  }
}

// -----
// NormalizeModOp tests for HIVM dialect.

// -----
// Test: I1 (bool) mod - needs normalize (returns all zeros, since x % 1 = 0 for bool)

// CHECK-LABEL: func.func @test_NormalizeVModOp_i1_native
// CHECK-SAME: (%arg0: tensor<256xi1>, %arg1: tensor<256xi1>)
// CHECK: %[[CST_DENSE:.*]] = arith.constant dense<false> : tensor<1xi1>
// CHECK: %[[EMPTY_I1:.*]] = tensor.empty() : tensor<256xi1>
// CHECK: %[[EMPTY_F16_1:.*]] = tensor.empty() : tensor<1xf16>
// CHECK: %[[CAST_DENSE:.*]] = hivm.hir.vcast ins(%[[CST_DENSE]] : tensor<1xi1>) outs(%[[EMPTY_F16_1]] : tensor<1xf16>)
// CHECK: %[[EXTRACTED:.*]] = tensor.extract %[[CAST_DENSE]]
// CHECK: %[[EMPTY_F16_2:.*]] = tensor.empty() : tensor<256xf16>
// CHECK: %[[CAST_I1:.*]] = hivm.hir.vcast ins(%[[EMPTY_I1]] : tensor<256xi1>) outs(%[[EMPTY_F16_2]] : tensor<256xf16>)
// CHECK: %[[BRC:.*]] = hivm.hir.vbrc ins(%[[EXTRACTED]] : f16) outs(%[[CAST_I1]] : tensor<256xf16>)
// CHECK: %[[CMP:.*]] = hivm.hir.vcmp
// CHECK: %[[NOT:.*]] = hivm.hir.vnot ins(%[[CMP]] : tensor<256xi1>)
// CHECK: return %[[NOT]]
func.func @test_NormalizeVModOp_i1_native(%src0: tensor<256xi1>, %src1: tensor<256xi1>) -> tensor<256xi1> {
    %0 = tensor.empty() : tensor<256xi1>
    %1 = hivm.hir.vmod ins(%src0, %src1 : tensor<256xi1>, tensor<256xi1>) outs(%0 : tensor<256xi1>) -> tensor<256xi1>
    return %1 : tensor<256xi1>
}

// -----
// Test: I16 mod - native op, no normalization (returns original op)

// CHECK-LABEL: func.func @test_NormalizeVModOp_i16_native
// CHECK-SAME: (%arg0: tensor<256xi16>, %arg1: tensor<256xi16>)
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<256xi16>
// CHECK: %[[mod:.*]] = hivm.hir.vmod ins(%arg0, %arg1 : tensor<256xi16>, tensor<256xi16>) outs(%[[empty]] : tensor<256xi16>) -> tensor<256xi16>
// CHECK: return %[[mod]]
func.func @test_NormalizeVModOp_i16_native(%src0: tensor<256xi16>, %src1: tensor<256xi16>) -> tensor<256xi16> {
    %0 = tensor.empty() : tensor<256xi16>
    %1 = hivm.hir.vmod ins(%src0, %src1 : tensor<256xi16>, tensor<256xi16>) outs(%0 : tensor<256xi16>) -> tensor<256xi16>
    return %1 : tensor<256xi16>
}

// -----
// Test: I32 mod - native op, no normalization

// CHECK-LABEL: func.func @test_NormalizeVModOp_i32_native
// CHECK-SAME: (%arg0: tensor<256xi32>, %arg1: tensor<256xi32>)
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<256xi32>
// CHECK: %[[mod:.*]] = hivm.hir.vmod ins(%arg0, %arg1 : tensor<256xi32>, tensor<256xi32>) outs(%[[empty]] : tensor<256xi32>) -> tensor<256xi32>
// CHECK: return %[[mod]]
func.func @test_NormalizeVModOp_i32_native(%src0: tensor<256xi32>, %src1: tensor<256xi32>) -> tensor<256xi32> {
    %0 = tensor.empty() : tensor<256xi32>
    %1 = hivm.hir.vmod ins(%src0, %src1 : tensor<256xi32>, tensor<256xi32>) outs(%0 : tensor<256xi32>) -> tensor<256xi32>
    return %1 : tensor<256xi32>
}

// -----
// Test: I64 mod - native op, no normalization

// CHECK-LABEL: func.func @test_NormalizeVModOp_i64_native
// CHECK-SAME: (%arg0: tensor<256xi64>, %arg1: tensor<256xi64>)
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<256xi64>
// CHECK: %[[mod:.*]] = hivm.hir.vmod ins(%arg0, %arg1 : tensor<256xi64>, tensor<256xi64>) outs(%[[empty]] : tensor<256xi64>) -> tensor<256xi64>
// CHECK: return %[[mod]]
func.func @test_NormalizeVModOp_i64_native(%src0: tensor<256xi64>, %src1: tensor<256xi64>) -> tensor<256xi64> {
    %0 = tensor.empty() : tensor<256xi64>
    %1 = hivm.hir.vmod ins(%src0, %src1 : tensor<256xi64>, tensor<256xi64>) outs(%0 : tensor<256xi64>) -> tensor<256xi64>
    return %1 : tensor<256xi64>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeVLdexp_hivm_vldexp
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf32>, %[[ARG1:.*]]: tensor<16xf32>)
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[MUL:.*]] = hivm.hir.vmul ins(%[[ARG0]], %[[ARG1]] : tensor<16xf32>, tensor<16xf32>) outs(%[[EMPTY]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: return %[[MUL]]
func.func @test_NormalizeVLdexp_hivm_vldexp(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
  %0 = tensor.empty() : tensor<16xf32>
  %1 = hivm.hir.vldexp ins(%arg0, %arg1 : tensor<16xf32>, tensor<16xf32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
  return %1 : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeVLdexp_hivm_vldexp_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf16>, %[[ARG1:.*]]: tensor<16xf16>)
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<16xf16>
// CHECK: %[[MUL:.*]] = hivm.hir.vmul ins(%[[ARG0]], %[[ARG1]] : tensor<16xf16>, tensor<16xf16>) outs(%[[EMPTY]] : tensor<16xf16>) -> tensor<16xf16>
// CHECK: return %[[MUL]]
func.func @test_NormalizeVLdexp_hivm_vldexp_f16(%arg0: tensor<16xf16>, %arg1: tensor<16xf16>) -> tensor<16xf16> {
  %0 = tensor.empty() : tensor<16xf16>
  %1 = hivm.hir.vldexp ins(%arg0, %arg1 : tensor<16xf16>, tensor<16xf16>) outs(%0 : tensor<16xf16>) -> tensor<16xf16>
  return %1 : tensor<16xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeVModOp_i8
// CHECK-SAME: (%arg0: tensor<256xi8>, %arg1: tensor<256xi8>)
// CHECK: %[[cast0f16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<256xi8>) outs(%{{.*}} : tensor<256xf16>) -> tensor<256xf16>
// CHECK: %[[cast0:.*]] = hivm.hir.vcast ins(%[[cast0f16]] : tensor<256xf16>) outs(%{{.*}} : tensor<256xi16>) round_mode = <trunc> -> tensor<256xi16>
// CHECK: %[[cast1f16:.*]] = hivm.hir.vcast ins(%arg1 : tensor<256xi8>) outs(%{{.*}} : tensor<256xf16>) -> tensor<256xf16>
// CHECK: %[[cast1:.*]] = hivm.hir.vcast ins(%[[cast1f16]] : tensor<256xf16>) outs(%{{.*}} : tensor<256xi16>) round_mode = <trunc> -> tensor<256xi16>
// CHECK: %[[mod:.*]] = hivm.hir.vmod ins(%[[cast0]], %[[cast1]] : tensor<256xi16>, tensor<256xi16>) outs(%{{.*}} : tensor<256xi16>) -> tensor<256xi16>
// CHECK: %[[cast2:.*]] = hivm.hir.vcast ins(%[[mod]] : tensor<256xi16>) outs(%{{.*}} : tensor<256xi8>) round_mode = <truncwithoverflow>{{.*}} -> tensor<256xi8>
// CHECK: return %[[cast2]]
func.func @test_NormalizeVModOp_i8(%src0: tensor<256xi8>, %src1: tensor<256xi8>) -> tensor<256xi8> {
    %0 = tensor.empty() : tensor<256xi8>
    %1 = hivm.hir.vmod ins(%src0, %src1 : tensor<256xi8>, tensor<256xi8>) outs(%0 : tensor<256xi8>) -> tensor<256xi8>
    return %1 : tensor<256xi8>
}

// -----
// Test: VModUIOp stays native when already legal

// CHECK-LABEL: func.func @test_NormalizeVModUIOp_i1_native
// CHECK-SAME: (%arg0: tensor<256xi1>, %arg1: tensor<256xi1>)
// CHECK: %[[CST_DENSE:.*]] = arith.constant dense<false> : tensor<1xi1>
// CHECK: %[[EMPTY_I1:.*]] = tensor.empty() : tensor<256xi1>
// CHECK: %[[EMPTY_F16_1:.*]] = tensor.empty() : tensor<1xf16>
// CHECK: %[[CAST_DENSE:.*]] = hivm.hir.vcast ins(%[[CST_DENSE]] : tensor<1xi1>) outs(%[[EMPTY_F16_1]] : tensor<1xf16>)
// CHECK: %[[EXTRACTED:.*]] = tensor.extract %[[CAST_DENSE]]
// CHECK: %[[EMPTY_F16_2:.*]] = tensor.empty() : tensor<256xf16>
// CHECK: %[[CAST_I1:.*]] = hivm.hir.vcast ins(%[[EMPTY_I1]] : tensor<256xi1>) outs(%[[EMPTY_F16_2]] : tensor<256xf16>)
// CHECK: %[[BRC:.*]] = hivm.hir.vbrc ins(%[[EXTRACTED]] : f16) outs(%[[CAST_I1]] : tensor<256xf16>)
// CHECK: %[[CMP:.*]] = hivm.hir.vcmp
// CHECK: %[[NOT:.*]] = hivm.hir.vnot ins(%[[CMP]] : tensor<256xi1>)
// CHECK: return %[[NOT]]
func.func @test_NormalizeVModUIOp_i1_native(%src0: tensor<256xi1>, %src1: tensor<256xi1>) -> tensor<256xi1> {
    %0 = tensor.empty() : tensor<256xi1>
    %1 = hivm.hir.vmodui ins(%src0, %src1 : tensor<256xi1>, tensor<256xi1>) outs(%0 : tensor<256xi1>) -> tensor<256xi1>
    return %1 : tensor<256xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeVPowf_hivm_vpow_half
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf32>, %[[DST:.*]]: tensor<16xf32>)
// CHECK: %[[SQRT_EMPTY:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[SQRT:.*]] = hivm.hir.vsqrt ins(%[[ARG0]] : tensor<16xf32>) outs(%[[SQRT_EMPTY]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: return %[[SQRT]]
func.func @test_NormalizeVPowf_hivm_vpow_half(%arg0: tensor<16xf32>, %dst: tensor<16xf32>) -> tensor<16xf32> {
  %cst = arith.constant 5.000000e-01 : f32
  %0 = hivm.hir.vbrc ins(%cst : f32) outs(%dst : tensor<16xf32>) -> tensor<16xf32>
  %1 = hivm.hir.vpow ins(%arg0, %0 : tensor<16xf32>, tensor<16xf32>) outs(%dst : tensor<16xf32>) -> tensor<16xf32>
  return %1 : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeVPowf_hivm_vpow_two
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf32>, %[[DST:.*]]: tensor<16xf32>)
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[MUL:.*]] = hivm.hir.vmul ins(%[[ARG0]], %[[ARG0]] : tensor<16xf32>, tensor<16xf32>) outs(%[[EMPTY]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: return %[[MUL]]
func.func @test_NormalizeVPowf_hivm_vpow_two(%arg0: tensor<16xf32>, %dst: tensor<16xf32>) -> tensor<16xf32> {
  %cst = arith.constant 2.000000e+00 : f32
  %0 = hivm.hir.vbrc ins(%cst : f32) outs(%dst : tensor<16xf32>) -> tensor<16xf32>
  %1 = hivm.hir.vpow ins(%arg0, %0 : tensor<16xf32>, tensor<16xf32>) outs(%dst : tensor<16xf32>) -> tensor<16xf32>
  return %1 : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeVPowf_hivm_vpow_dynamic
// CHECK-SAME: (%[[BASE:.*]]: tensor<16xf32>, %[[EXPONENT:.*]]: tensor<16xf32>, %[[DST:.*]]: tensor<16xf32>)
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG: %[[NEG_TWO:.*]] = arith.constant -2.000000e+00 : f32
// CHECK-DAG: %[[POS_TWO:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG: %[[NAN:.*]] = arith.constant 0x7FC00000 : f32
// CHECK-DAG: %[[INF:.*]] = arith.constant 0x7F800000 : f32
// CHECK: %[[BASE_SIGN:.*]] = hivm.hir.bitcast %[[BASE]] : tensor<16xf32> -> tensor<16xi32>
// CHECK: %[[NEG_COND:.*]] = hivm.hir.vcmp
// CHECK: %[[ABS_EXP0:.*]] = hivm.hir.vabs ins(%[[EXPONENT]] : tensor<16xf32>) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[DIVHALF:.*]] = hivm.hir.vdiv ins(%[[ABS_EXP0]], %[[POS_TWO]] : tensor<16xf32>, f32) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[TRUNC:.*]] = hivm.hir.vcast ins(%[[DIVHALF]] : tensor<16xf32>) outs(%{{.*}} : tensor<16xf32>) round_mode = <trunc> -> tensor<16xf32>
// CHECK: %[[MULMOD:.*]] = hivm.hir.vmul ins(%[[TRUNC]], %[[POS_TWO]] : tensor<16xf32>, f32) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[MOD:.*]] = hivm.hir.vsub ins(%[[ABS_EXP0]], %[[MULMOD]] : tensor<16xf32>, tensor<16xf32>) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[NEG_MAG:.*]] = hivm.hir.vexp
// CHECK: %[[NEG_RESULT:.*]] = hivm.hir.vmul ins(%[[NEG_MAG]], %{{.*}} : tensor<16xf32>, tensor<16xf32>) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[POS_RESULT:.*]] = hivm.hir.vexp
// CHECK: %[[SELECT0:.*]] = hivm.hir.vsel ins(%{{.*}}, %[[NEG_RESULT]], %[[POS_RESULT]] : tensor<16xi1>, tensor<16xf32>, tensor<16xf32>) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[ABS_BASE0:.*]] = hivm.hir.vabs ins(%[[BASE]] : tensor<16xf32>) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[ABS_EXP1:.*]] = hivm.hir.vabs ins(%[[EXPONENT]] : tensor<16xf32>) outs(%{{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[SELECT1:.*]] = hivm.hir.vsel
// CHECK: %[[SELECT2:.*]] = hivm.hir.vsel
// CHECK: %[[SELECT3:.*]] = hivm.hir.vsel
// CHECK: return %[[SELECT3]]
func.func @test_NormalizeVPowf_hivm_vpow_dynamic(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>, %dst: tensor<16xf32>) -> tensor<16xf32> {
  %0 = hivm.hir.vpow ins(%arg0, %arg1 : tensor<16xf32>, tensor<16xf32>) outs(%dst : tensor<16xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeVModUIOp_i8
// CHECK-SAME: (%arg0: tensor<256xi8>, %arg1: tensor<256xi8>)
// CHECK: %[[cast0f16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<256xi8>) outs(%{{.*}} : tensor<256xf16>) cast = <cast_unsigned> -> tensor<256xf16>
// CHECK: %[[cast0:.*]] = hivm.hir.vcast ins(%[[cast0f16]] : tensor<256xf16>) outs(%{{.*}} : tensor<256xi16>) round_mode = <trunc> -> tensor<256xi16>
// CHECK: %[[cast1f16:.*]] = hivm.hir.vcast ins(%arg1 : tensor<256xi8>) outs(%{{.*}} : tensor<256xf16>) cast = <cast_unsigned> -> tensor<256xf16>
// CHECK: %[[cast1:.*]] = hivm.hir.vcast ins(%[[cast1f16]] : tensor<256xf16>) outs(%{{.*}} : tensor<256xi16>) round_mode = <trunc> -> tensor<256xi16>
// CHECK: %[[mod:.*]] = hivm.hir.vmodui ins(%[[cast0]], %[[cast1]] : tensor<256xi16>, tensor<256xi16>) outs(%{{.*}} : tensor<256xi16>) -> tensor<256xi16>
// CHECK: %[[cast2:.*]] = hivm.hir.vcast ins(%[[mod]] : tensor<256xi16>) outs(%{{.*}} : tensor<256xi8>) round_mode = <truncwithoverflow> cast = <cast_unsigned> -> tensor<256xi8>
// CHECK: return %[[cast2]]
func.func @test_NormalizeVModUIOp_i8(%src0: tensor<256xi8>, %src1: tensor<256xi8>) -> tensor<256xi8> {
    %0 = tensor.empty() : tensor<256xi8>
    %1 = hivm.hir.vmodui ins(%src0, %src1 : tensor<256xi8>, tensor<256xi8>) outs(%0 : tensor<256xi8>) -> tensor<256xi8>
    return %1 : tensor<256xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeVModUIOp_i16_native
// CHECK-SAME: (%arg0: tensor<256xi16>, %arg1: tensor<256xi16>)
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<256xi16>
// CHECK: %[[mod:.*]] = hivm.hir.vmodui ins(%arg0, %arg1 : tensor<256xi16>, tensor<256xi16>) outs(%[[empty]] : tensor<256xi16>) -> tensor<256xi16>
// CHECK: return %[[mod]]
func.func @test_NormalizeVModUIOp_i16_native(%src0: tensor<256xi16>, %src1: tensor<256xi16>) -> tensor<256xi16> {
    %0 = tensor.empty() : tensor<256xi16>
    %1 = hivm.hir.vmodui ins(%src0, %src1 : tensor<256xi16>, tensor<256xi16>) outs(%0 : tensor<256xi16>) -> tensor<256xi16>
    return %1 : tensor<256xi16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeVModUIOp_i32_native
// CHECK-SAME: (%arg0: tensor<256xi32>, %arg1: tensor<256xi32>)
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<256xi32>
// CHECK: %[[mod:.*]] = hivm.hir.vmodui ins(%arg0, %arg1 : tensor<256xi32>, tensor<256xi32>) outs(%[[empty]] : tensor<256xi32>) -> tensor<256xi32>
// CHECK: return %[[mod]]
func.func @test_NormalizeVModUIOp_i32_native(%src0: tensor<256xi32>, %src1: tensor<256xi32>) -> tensor<256xi32> {
    %0 = tensor.empty() : tensor<256xi32>
    %1 = hivm.hir.vmodui ins(%src0, %src1 : tensor<256xi32>, tensor<256xi32>) outs(%0 : tensor<256xi32>) -> tensor<256xi32>
    return %1 : tensor<256xi32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeVModUIOp_i64_native
// CHECK-SAME: (%arg0: tensor<256xi64>, %arg1: tensor<256xi64>)
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<256xi64>
// CHECK: %[[mod:.*]] = hivm.hir.vmodui ins(%arg0, %arg1 : tensor<256xi64>, tensor<256xi64>) outs(%[[empty]] : tensor<256xi64>) -> tensor<256xi64>
// CHECK: return %[[mod]]
func.func @test_NormalizeVModUIOp_i64_native(%src0: tensor<256xi64>, %src1: tensor<256xi64>) -> tensor<256xi64> {
    %0 = tensor.empty() : tensor<256xi64>
    %1 = hivm.hir.vmodui ins(%src0, %src1 : tensor<256xi64>, tensor<256xi64>) outs(%0 : tensor<256xi64>) -> tensor<256xi64>
    return %1 : tensor<256xi64>
}

// -----
// Test: VModOp with broadcast should not be normalized

// CHECK-LABEL: func.func @test_NormalizeVModOp_with_broadcast_no_normalize
// CHECK-SAME: (%arg0: tensor<256xi32>, %arg1: tensor<256xi32>)
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<256xi32>
// CHECK: %[[mod:.*]] = hivm.hir.vmod ins(%arg0, %arg1 : tensor<256xi32>, tensor<256xi32>) outs(%[[empty]] : tensor<256xi32>) broadcast = [1, 1] -> tensor<256xi32>
// CHECK: return %[[mod]]
func.func @test_NormalizeVModOp_with_broadcast_no_normalize(%src0: tensor<256xi32>, %src1: tensor<256xi32>) -> tensor<256xi32> {
    %0 = tensor.empty() : tensor<256xi32>
    %1 = hivm.hir.vmod ins(%src0, %src1 : tensor<256xi32>, tensor<256xi32>) outs(%0 : tensor<256xi32>) broadcast = [1, 1] -> tensor<256xi32>
    return %1 : tensor<256xi32>
}

// -----
// Test: VModOp with transpose should not be normalized

// CHECK-LABEL: func.func @test_NormalizeVModOp_with_transpose_no_normalize
// CHECK-SAME: (%arg0: tensor<256xi32>, %arg1: tensor<256xi32>)
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<256xi32>
// CHECK: %[[mod:.*]] = hivm.hir.vmod ins(%arg0, %arg1 : tensor<256xi32>, tensor<256xi32>) outs(%[[empty]] : tensor<256xi32>) transpose = [1, 0] -> tensor<256xi32>
// CHECK: return %[[mod]]
func.func @test_NormalizeVModOp_with_transpose_no_normalize(%src0: tensor<256xi32>, %src1: tensor<256xi32>) -> tensor<256xi32> {
    %0 = tensor.empty() : tensor<256xi32>
    %1 = hivm.hir.vmod ins(%src0, %src1 : tensor<256xi32>, tensor<256xi32>) outs(%0 : tensor<256xi32>) transpose = [1, 0] -> tensor<256xi32>
    return %1 : tensor<256xi32>
}

// -----
// Test: F16 mod - needs normalize (truncate_div normalization)
// Pattern: cast(F16->F32) -> div -> vcast(trunc back to F16) -> mul -> sub ->
// normalized isinf sequence -> vsel

// CHECK-LABEL: func.func @test_NormalizeVModOp_f16
// CHECK-SAME: (%arg0: tensor<256xf16>, %arg1: tensor<256xf16>)
// CHECK: %cst = arith.constant 0.000000e+00 : f16
// CHECK: %c-1_i16 = arith.constant -1 : i16
// CHECK: %c1_i16 = arith.constant 1 : i16
// CHECK: %c-31744_i16 = arith.constant -31744 : i16
// CHECK: %c32767_i16 = arith.constant 32767 : i16
// CHECK: %cst_0 = arith.constant 0x7E00 : f16
// CHECK: %[[empty0:.*]] = tensor.empty() : tensor<256xf32>
// CHECK: %[[cast0:.*]] = hivm.hir.vcast ins(%arg0 : tensor<256xf16>) outs(%[[empty0]] : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[empty1:.*]] = tensor.empty() : tensor<256xf32>
// CHECK: %[[cast1:.*]] = hivm.hir.vcast ins(%arg1 : tensor<256xf16>) outs(%[[empty1]] : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[empty2:.*]] = tensor.empty() : tensor<256xf32>
// CHECK: %[[div:.*]] = hivm.hir.vdiv ins(%[[cast0]], %[[cast1]] : tensor<256xf32>, tensor<256xf32>) outs(%[[empty2]] : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[empty3:.*]] = tensor.empty() : tensor<256xf16>
// CHECK: %[[trunc:.*]] = hivm.hir.vcast ins(%[[div]] : tensor<256xf32>) outs(%[[empty3]] : tensor<256xf16>){{( round_mode = <trunc>)?}} -> tensor<256xf16>
// CHECK: %[[empty4:.*]] = tensor.empty() : tensor<256xf16>
// CHECK: %[[mul:.*]] = hivm.hir.vmul ins(%[[trunc]], %arg1 : tensor<256xf16>, tensor<256xf16>) outs(%[[empty4]] : tensor<256xf16>) -> tensor<256xf16>
// CHECK: %[[empty5:.*]] = tensor.empty() : tensor<256xf16>
// CHECK: %[[sub:.*]] = hivm.hir.vsub ins(%arg0, %[[mul]] : tensor<256xf16>, tensor<256xf16>) outs(%[[empty5]] : tensor<256xf16>) -> tensor<256xf16>
// CHECK: %[[brc_empty:.*]] = tensor.empty() : tensor<256xi16>
// CHECK: %[[brc:.*]] = hivm.hir.vbrc ins(%c32767_i16 : i16) outs(%[[brc_empty]] : tensor<256xi16>) -> tensor<256xi16>
// CHECK: %[[bitcast0:.*]] = hivm.hir.bitcast %arg1 : tensor<256xf16> -> tensor<256xi16>
// CHECK: %[[and_empty:.*]] = tensor.empty() : tensor<256xi16>
// CHECK: %[[and:.*]] = hivm.hir.vand ins(%[[bitcast0]], %[[brc]] : tensor<256xi16>, tensor<256xi16>) outs(%[[and_empty]] : tensor<256xi16>) -> tensor<256xi16>
// CHECK: %[[empty8:.*]] = tensor.empty() : tensor<256xi16>
// CHECK: %[[add0:.*]] = hivm.hir.vadd ins(%[[and]], %c-31744_i16 : tensor<256xi16>, i16) outs(%[[empty8]] : tensor<256xi16>) -> tensor<256xi16>
// CHECK: %[[bitcast1:.*]] = hivm.hir.bitcast %[[add0]] : tensor<256xi16> -> tensor<256xf16>
// CHECK: %[[empty9:.*]] = tensor.empty() : tensor<256xf16>
// CHECK: %[[abs:.*]] = hivm.hir.vabs ins(%[[bitcast1]] : tensor<256xf16>) outs(%[[empty9]] : tensor<256xf16>) -> tensor<256xf16>
// CHECK: %[[bitcast2:.*]] = hivm.hir.bitcast %[[abs]] : tensor<256xf16> -> tensor<256xi16>
// CHECK: %[[empty10:.*]] = tensor.empty() : tensor<256xi16>
// CHECK: %[[min:.*]] = hivm.hir.vmin ins(%[[bitcast2]], %c1_i16 : tensor<256xi16>, i16) outs(%[[empty10]] : tensor<256xi16>) -> tensor<256xi16>
// CHECK: %[[mul1:.*]] = hivm.hir.vmul ins(%[[min]], %c-1_i16 : tensor<256xi16>, i16) outs(%[[min]] : tensor<256xi16>) -> tensor<256xi16>
// CHECK: %[[add1:.*]] = hivm.hir.vadd ins(%[[mul1]], %c1_i16 : tensor<256xi16>, i16) outs(%[[mul1]] : tensor<256xi16>) -> tensor<256xi16>
// CHECK: %[[empty11:.*]] = tensor.empty() : tensor<256xf16>
// CHECK: %[[cast2:.*]] = hivm.hir.vcast ins(%[[add1]] : tensor<256xi16>) outs(%[[empty11]] : tensor<256xf16>) -> tensor<256xf16>
// CHECK: %[[empty12:.*]] = tensor.empty() : tensor<256xi1>
// CHECK: %[[cmp:.*]] = hivm.hir.vcmp ins(%[[cast2]], %cst : tensor<256xf16>, f16) outs(%[[empty12]] : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[not:.*]] = hivm.hir.vnot ins(%[[cmp]] : tensor<256xi1>) outs(%{{.*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[empty14:.*]] = tensor.empty() : tensor<256xf16>
// CHECK: %[[sel:.*]] = hivm.hir.vsel ins(%[[not]], %cst_0, %[[sub]] : tensor<256xi1>, f16, tensor<256xf16>) outs(%[[empty14]] : tensor<256xf16>) -> tensor<256xf16>
// CHECK: return %[[sel]] : tensor<256xf16>
func.func @test_NormalizeVModOp_f16(%src0: tensor<256xf16>, %src1: tensor<256xf16>) -> tensor<256xf16> {
    %0 = tensor.empty() : tensor<256xf16>
    %1 = hivm.hir.vmod ins(%src0, %src1 : tensor<256xf16>, tensor<256xf16>) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
    return %1 : tensor<256xf16>
}

// -----
// Test: F32 mod - needs normalize (truncate_div normalization)
// Pattern: div -> vcast(trunc) -> mul -> sub -> normalized isinf sequence ->
// vsel

// CHECK-LABEL: func.func @test_NormalizeVModOp_f32
// CHECK-SAME: (%arg0: tensor<256xf32>, %arg1: tensor<256xf32>)
// CHECK: %cst = arith.constant 0.000000e+00 : f32
// CHECK: %c-1_i32 = arith.constant -1 : i32
// CHECK: %c1_i32 = arith.constant 1 : i32
// CHECK: %c-2139095040_i32 = arith.constant -2139095040 : i32
// CHECK: %c2147483647_i32 = arith.constant 2147483647 : i32
// CHECK: %cst_0 = arith.constant 0x7FC00000 : f32
// CHECK: %[[empty0:.*]] = tensor.empty() : tensor<256xf32>
// CHECK: %[[div:.*]] = hivm.hir.vdiv ins(%arg0, %arg1 : tensor<256xf32>, tensor<256xf32>) outs(%[[empty0]] : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[empty1:.*]] = tensor.empty() : tensor<256xf32>
// CHECK: %[[trunc:.*]] = hivm.hir.vcast ins(%[[div]] : tensor<256xf32>) outs(%[[empty1]] : tensor<256xf32>) round_mode = <trunc> -> tensor<256xf32>
// CHECK: %[[empty2:.*]] = tensor.empty() : tensor<256xf32>
// CHECK: %[[mul:.*]] = hivm.hir.vmul ins(%[[trunc]], %arg1 : tensor<256xf32>, tensor<256xf32>) outs(%[[empty2]] : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[empty3:.*]] = tensor.empty() : tensor<256xf32>
// CHECK: %[[sub:.*]] = hivm.hir.vsub ins(%arg0, %[[mul]] : tensor<256xf32>, tensor<256xf32>) outs(%[[empty3]] : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[brc_empty:.*]] = tensor.empty() : tensor<256xi32>
// CHECK: %[[brc:.*]] = hivm.hir.vbrc ins(%c2147483647_i32 : i32) outs(%[[brc_empty]] : tensor<256xi32>) -> tensor<256xi32>
// CHECK: %[[bitcast0:.*]] = hivm.hir.bitcast %arg1 : tensor<256xf32> -> tensor<256xi32>
// CHECK: %[[and_empty:.*]] = tensor.empty() : tensor<256xi32>
// CHECK: %[[and:.*]] = hivm.hir.vand ins(%[[bitcast0]], %[[brc]] : tensor<256xi32>, tensor<256xi32>) outs(%[[and_empty]] : tensor<256xi32>) -> tensor<256xi32>
// CHECK: %[[empty6:.*]] = tensor.empty() : tensor<256xi32>
// CHECK: %[[add0:.*]] = hivm.hir.vadd ins(%[[and]], %c-2139095040_i32 : tensor<256xi32>, i32) outs(%[[empty6]] : tensor<256xi32>) -> tensor<256xi32>
// CHECK: %[[bitcast1:.*]] = hivm.hir.bitcast %[[add0]] : tensor<256xi32> -> tensor<256xf32>
// CHECK: %[[empty7:.*]] = tensor.empty() : tensor<256xf32>
// CHECK: %[[abs:.*]] = hivm.hir.vabs ins(%[[bitcast1]] : tensor<256xf32>) outs(%[[empty7]] : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[bitcast2:.*]] = hivm.hir.bitcast %[[abs]] : tensor<256xf32> -> tensor<256xi32>
// CHECK: %[[empty8:.*]] = tensor.empty() : tensor<256xi32>
// CHECK: %[[min:.*]] = hivm.hir.vmin ins(%[[bitcast2]], %c1_i32 : tensor<256xi32>, i32) outs(%[[empty8]] : tensor<256xi32>) -> tensor<256xi32>
// CHECK: %[[mul1:.*]] = hivm.hir.vmul ins(%[[min]], %c-1_i32 : tensor<256xi32>, i32) outs(%[[min]] : tensor<256xi32>) -> tensor<256xi32>
// CHECK: %[[add1:.*]] = hivm.hir.vadd ins(%[[mul1]], %c1_i32 : tensor<256xi32>, i32) outs(%[[mul1]] : tensor<256xi32>) -> tensor<256xi32>
// CHECK: %[[empty9:.*]] = tensor.empty() : tensor<256xf32>
// CHECK: %[[cast2:.*]] = hivm.hir.vcast ins(%[[add1]] : tensor<256xi32>) outs(%[[empty9]] : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[empty10:.*]] = tensor.empty() : tensor<256xi1>
// CHECK: %[[cmp:.*]] = hivm.hir.vcmp ins(%[[cast2]], %cst : tensor<256xf32>, f32) outs(%[[empty10]] : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[not:.*]] = hivm.hir.vnot ins(%[[cmp]] : tensor<256xi1>) outs(%{{.*}} : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[empty12:.*]] = tensor.empty() : tensor<256xf32>
// CHECK: %[[sel:.*]] = hivm.hir.vsel ins(%[[not]], %cst_0, %[[sub]] : tensor<256xi1>, f32, tensor<256xf32>) outs(%[[empty12]] : tensor<256xf32>) -> tensor<256xf32>
// CHECK: return %[[sel]] : tensor<256xf32>
func.func @test_NormalizeVModOp_f32(%src0: tensor<256xf32>, %src1: tensor<256xf32>) -> tensor<256xf32> {
    %0 = tensor.empty() : tensor<256xf32>
    %1 = hivm.hir.vmod ins(%src0, %src1 : tensor<256xf32>, tensor<256xf32>) outs(%0 : tensor<256xf32>) -> tensor<256xf32>
    return %1 : tensor<256xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeVPowf_hivm_vpow_zero
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf32>, %[[DST:.*]]: tensor<16xf32>)
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[FILL:.*]] = hivm.hir.vbrc ins(%[[ONE]] : f32) outs(%[[EMPTY]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: return %[[FILL]]
func.func @test_NormalizeVPowf_hivm_vpow_zero(%arg0: tensor<16xf32>, %dst: tensor<16xf32>) -> tensor<16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hivm.hir.vbrc ins(%cst : f32) outs(%dst : tensor<16xf32>) -> tensor<16xf32>
  %1 = hivm.hir.vpow ins(%arg0, %0 : tensor<16xf32>, tensor<16xf32>) outs(%dst : tensor<16xf32>) -> tensor<16xf32>
  return %1 : tensor<16xf32>
}

// -----

// Ascend950: use math.fma instead of vmul+vadd for vpow coefficient
// CHECK-LABEL: func.func @test_NormalizeVPowf_hivm_vpow_dynamic_ascend950
// CHECK: %cst = arith.constant 0x7F800000 : f32
// CHECK: %[[INF:.*]] = arith.constant 2.13909504E+9 : f32
// CHECK: %[[NAN:.*]] = arith.constant 0x7FC00000 : f32
// CHECK: %[[NEG_TWO:.*]] = arith.constant -2.000000e+00 : f32
// CHECK: %[[POS_TWO:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[POS_ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[NEG_ONE:.*]] = arith.constant -1 : i32
// CHECK: %[[BITCAST0:.*]] = hivm.hir.bitcast %arg0 : tensor<16xf32> -> tensor<16xi32>
// CHECK: %[[SHIFT:.*]] = hivm.hir.vshr ins(%[[BITCAST0]], {{.*}} : tensor<16xi32>, i32) outs({{.*}} : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[NEG_COND:.*]] = hivm.hir.vcmp ins(%[[SHIFT]], %[[NEG_ONE]] : tensor<16xi32>, i32) outs({{.*}} : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[EXP_FLOOR:.*]] = hivm.hir.vcast ins(%arg1 : tensor<16xf32>) outs({{.*}} : tensor<16xf32>) round_mode = <floor> -> tensor<16xf32>
// CHECK: %[[IS_INT:.*]] = hivm.hir.vcmp ins(%[[EXP_FLOOR]], %arg1 : tensor<16xf32>, tensor<16xf32>) outs({{.*}} : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[NEG_AND_INT:.*]] = hivm.hir.vand ins(%[[NEG_COND]], %[[IS_INT]] : tensor<16xi1>, tensor<16xi1>) outs({{.*}} : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[ABS_EXP:.*]] = hivm.hir.vabs ins(%arg1 : tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[ABS_EXP]], %[[POS_TWO]] : tensor<16xf32>, f32) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[TRUNC:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<16xf32>) outs({{.*}} : tensor<16xf32>) round_mode = <trunc> -> tensor<16xf32>
// CHECK: %[[MUL_MOD:.*]] = hivm.hir.vmul ins(%[[TRUNC]], %[[POS_TWO]] : tensor<16xf32>, f32) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[MOD:.*]] = hivm.hir.vsub ins(%[[ABS_EXP]], %[[MUL_MOD]] : tensor<16xf32>, tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[MOD_SEL:.*]] = hivm.hir.vsel ins({{.*}}, {{.*}}, %[[MOD]] : tensor<16xi1>, f32, tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[VBRC_NEG:.*]] = hivm.hir.vbrc ins(%[[NEG_TWO]] : f32) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[VBRC_POS:.*]] = hivm.hir.vbrc ins(%[[POS_ONE]] : f32) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[FMA:.*]] = math.fma %[[MOD_SEL]], %[[VBRC_NEG]], %[[VBRC_POS]] : tensor<16xf32>
// CHECK: %[[ABS_BASE0:.*]] = hivm.hir.vabs ins(%arg0 : tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[LN0:.*]] = hivm.hir.vln ins(%[[ABS_BASE0]] : tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[MUL0:.*]] = hivm.hir.vmul ins(%[[LN0]], %arg1 : tensor<16xf32>, tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EXP0:.*]] = hivm.hir.vexp ins(%[[MUL0]] : tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[NEG_MAG:.*]] = hivm.hir.vmul ins(%[[EXP0]], %[[FMA]] : tensor<16xf32>, tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[ABS_BASE1:.*]] = hivm.hir.vabs ins(%arg0 : tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[LN1:.*]] = hivm.hir.vln ins(%[[ABS_BASE1]] : tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[MUL1:.*]] = hivm.hir.vmul ins(%[[LN1]], %arg1 : tensor<16xf32>, tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EXP1:.*]] = hivm.hir.vexp ins(%[[MUL1]] : tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[SEL0:.*]] = hivm.hir.vsel ins(%[[NEG_AND_INT]], %[[NEG_MAG]], %[[EXP1]] : tensor<16xi1>, tensor<16xf32>, tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[SEL1:.*]] = hivm.hir.vsel{{.*}} outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[SEL2:.*]] = hivm.hir.vsel{{.*}} outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[SEL3:.*]] = hivm.hir.vsel{{.*}} outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: return %[[SEL3]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeVPowf_hivm_vpow_dynamic_ascend950(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>, %dst: tensor<16xf32>) -> tensor<16xf32> {
    %0 = hivm.hir.vpow ins(%arg0, %arg1 : tensor<16xf32>, tensor<16xf32>) outs(%dst : tensor<16xf32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}
