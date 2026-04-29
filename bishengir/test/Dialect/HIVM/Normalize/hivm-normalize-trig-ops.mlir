// RUN: bishengir-opt --hivm-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_NormalizeSin_hivm_vsin_f32(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-DAG: %[[PI_REC:.*]] = arith.constant 0.318309873 : f32
// CHECK-DAG: %[[HALF:.*]] = arith.constant 5.000000e-01 : f32
// CHECK: %[[OUT:.*]] = tensor.empty() : tensor<4xf32>
// CHECK: %[[DIV_PI:.*]] = hivm.hir.vmul ins(%[[ARG0]], %[[PI_REC]] : tensor<4xf32>, f32) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[ROUND:.*]] = hivm.hir.vcast ins(%[[DIV_PI]] : tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) round_mode = <round> -> tensor<4xf32>
// CHECK: hivm.hir.vsub
// CHECK: hivm.hir.vmul ins(%{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[SIGN_PRE:.*]] = hivm.hir.vmul ins(%[[ROUND]], %[[HALF]] : tensor<4xf32>, f32) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[SIGN_FLOOR:.*]] = hivm.hir.vcast ins(%[[SIGN_PRE]] : tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) round_mode = <floor> -> tensor<4xf32>
// CHECK: %[[RES:.*]] = hivm.hir.vmul ins(%{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32>) outs(%[[OUT]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: return %[[RES]] : tensor<4xf32>
// CHECK-NOT: hivm.hir.vsin
func.func @test_NormalizeSin_hivm_vsin_f32(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = tensor.empty() : tensor<4xf32>
  %1 = hivm.hir.vsin ins(%arg0 : tensor<4xf32>) outs(%0 : tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeSin_hivm_vsin_f16(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4xf16>) -> tensor<4xf16> {
// CHECK: %[[EMPTY_F32:.*]] = tensor.empty() : tensor<4xf32>
// CHECK: %[[CAST_IN:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<4xf16>) outs(%[[EMPTY_F32]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[ROUND_INPUT:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) round_mode = <round> -> tensor<4xf32>
// CHECK: %[[SIGN_FLOOR:.*]] = hivm.hir.vcast
// CHECK-SAME: round_mode = <floor>
// CHECK: %[[EMPTY_F16:.*]] = tensor.empty() : tensor<4xf16>
// CHECK-NEXT: %[[CAST_OUT:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<4xf32>) outs(%[[EMPTY_F16]] : tensor<4xf16>) round_mode = <round> -> tensor<4xf16>
// CHECK-NEXT: return %[[CAST_OUT]] : tensor<4xf16>
// CHECK-NOT: hivm.hir.vsin
func.func @test_NormalizeSin_hivm_vsin_f16(%arg0 : tensor<4xf16>) -> tensor<4xf16> {
  %0 = tensor.empty() : tensor<4xf16>
  %1 = hivm.hir.vsin ins(%arg0 : tensor<4xf16>) outs(%0 : tensor<4xf16>) -> tensor<4xf16>
  return %1 : tensor<4xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCos_hivm_vcos_f32(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-DAG: %[[PI_REC:.*]] = arith.constant 0.318309873 : f32
// CHECK-DAG: %[[HALF:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG: %[[HALF_PI:.*]] = arith.constant 1.57079637 : f32
// CHECK: %[[OUT:.*]] = tensor.empty() : tensor<4xf32>
// CHECK: %[[DIV_PI:.*]] = hivm.hir.vmul ins(%[[ARG0]], %[[PI_REC]] : tensor<4xf32>, f32) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[ROUND_INPUT:.*]] = hivm.hir.vadd ins(%[[DIV_PI]], %[[HALF]] : tensor<4xf32>, f32) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[ROUND:.*]] = hivm.hir.vcast ins(%[[ROUND_INPUT]] : tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) round_mode = <round> -> tensor<4xf32>
// CHECK: %[[NORM:.*]] = hivm.hir.vadd ins(%{{.*}}, %[[HALF_PI]] : tensor<4xf32>, f32) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[SIGN_PRE:.*]] = hivm.hir.vmul ins(%[[ROUND]], %[[HALF]] : tensor<4xf32>, f32) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[SIGN_FLOOR:.*]] = hivm.hir.vcast ins(%[[SIGN_PRE]] : tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) round_mode = <floor> -> tensor<4xf32>
// CHECK: %[[RES:.*]] = hivm.hir.vmul ins(%{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32>) outs(%[[OUT]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: return %[[RES]] : tensor<4xf32>
// CHECK-NOT: hivm.hir.vcos
func.func @test_NormalizeCos_hivm_vcos_f32(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = tensor.empty() : tensor<4xf32>
  %1 = hivm.hir.vcos ins(%arg0 : tensor<4xf32>) outs(%0 : tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCos_hivm_vcos_f16(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4xf16>) -> tensor<4xf16> {
// CHECK: %[[EMPTY_F32:.*]] = tensor.empty() : tensor<4xf32>
// CHECK: %[[CAST_IN:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<4xf16>) outs(%[[EMPTY_F32]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[ROUND_INPUT:.*]] = hivm.hir.vadd
// CHECK: %[[ROUND:.*]] = hivm.hir.vcast ins(%[[ROUND_INPUT]] : tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) round_mode = <round> -> tensor<4xf32>
// CHECK: %[[SIGN_FLOOR:.*]] = hivm.hir.vcast
// CHECK-SAME: round_mode = <floor>
// CHECK: %[[EMPTY_F16:.*]] = tensor.empty() : tensor<4xf16>
// CHECK-NEXT: %[[CAST_OUT:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<4xf32>) outs(%[[EMPTY_F16]] : tensor<4xf16>) round_mode = <round> -> tensor<4xf16>
// CHECK-NEXT: return %[[CAST_OUT]] : tensor<4xf16>
// CHECK-NOT: hivm.hir.vcos
func.func @test_NormalizeCos_hivm_vcos_f16(%arg0 : tensor<4xf16>) -> tensor<4xf16> {
  %0 = tensor.empty() : tensor<4xf16>
  %1 = hivm.hir.vcos ins(%arg0 : tensor<4xf16>) outs(%0 : tensor<4xf16>) -> tensor<4xf16>
  return %1 : tensor<4xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAtan_hivm_vatan_f32(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-DAG: %[[CLIP_HI:.*]] = arith.constant 1.000000e+04 : f32
// CHECK-DAG: %[[CLIP_LO:.*]] = arith.constant -1.000000e+04 : f32
// CHECK-DAG: %[[PI_OVER8:.*]] = arith.constant 0.392699093 : f32
// CHECK-DAG: %[[PI_OVER4:.*]] = arith.constant 0.785398185 : f32
// CHECK: %[[CLAMP_HI:.*]] = hivm.hir.vmin ins(%[[ARG0]], %[[CLIP_HI]] : tensor<4xf32>, f32)
// CHECK: %[[CLAMP:.*]] = hivm.hir.vmax ins(%[[CLAMP_HI]], %[[CLIP_LO]] : tensor<4xf32>, f32)
// CHECK: %[[ABS:.*]] = hivm.hir.vabs ins(%[[CLAMP]] : tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[SHIFTED:.*]] = hivm.hir.vadd ins(%{{.*}}, %[[PI_OVER8]] : tensor<4xf32>, f32)
// CHECK: %[[RECIP:.*]] = hivm.hir.vadd ins(%{{.*}}, %[[PI_OVER4]] : tensor<4xf32>, f32)
// CHECK: %[[MAG:.*]] = hivm.hir.vmin ins(%{{.*}}, %[[RECIP]] : tensor<4xf32>, tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[SIGN:.*]] = hivm.hir.vdiv ins(%{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[RES:.*]] = hivm.hir.vmul ins(%[[MAG]], %[[SIGN]] : tensor<4xf32>, tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: return %[[RES]] : tensor<4xf32>
// CHECK-NOT: hivm.hir.vatan
func.func @test_NormalizeAtan_hivm_vatan_f32(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = tensor.empty() : tensor<4xf32>
  %1 = hivm.hir.vatan ins(%arg0 : tensor<4xf32>) outs(%0 : tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeAtan_hivm_vatan_f16(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4xf16>) -> tensor<4xf16> {
// CHECK: %[[EMPTY_F32:.*]] = tensor.empty() : tensor<4xf32>
// CHECK: %[[CAST_IN:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<4xf16>) outs(%[[EMPTY_F32]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[CLAMP_HI:.*]] = hivm.hir.vmin ins(%[[CAST_IN]], %{{.*}} : tensor<4xf32>, f32)
// CHECK: %[[ABS:.*]] = hivm.hir.vabs ins(%{{.*}} : tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: hivm.hir.vmin ins(%{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[SIGN_SCALE:.*]] = hivm.hir.vmul ins(%[[CAST_IN]], %{{.*}} : tensor<4xf32>, f32) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[SIGN:.*]] = hivm.hir.vdiv ins(%[[SIGN_SCALE]], %{{.*}} : tensor<4xf32>, tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[MUL:.*]] = hivm.hir.vmul ins(%{{.*}}, %[[SIGN]] : tensor<4xf32>, tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[EMPTY_F16:.*]] = tensor.empty() : tensor<4xf16>
// CHECK-NEXT: %[[CAST_OUT:.*]] = hivm.hir.vcast ins(%[[MUL]] : tensor<4xf32>) outs(%[[EMPTY_F16]] : tensor<4xf16>) round_mode = <round> -> tensor<4xf16>
// CHECK-NEXT: return %[[CAST_OUT]] : tensor<4xf16>
// CHECK-NOT: hivm.hir.vatan
func.func @test_NormalizeAtan_hivm_vatan_f16(%arg0 : tensor<4xf16>) -> tensor<4xf16> {
  %0 = tensor.empty() : tensor<4xf16>
  %1 = hivm.hir.vatan ins(%arg0 : tensor<4xf16>) outs(%0 : tensor<4xf16>) -> tensor<4xf16>
  return %1 : tensor<4xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeTan_hivm_vtan_f32(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-DAG: %[[PI_REC:.*]] = arith.constant 0.318309873 : f32
// CHECK: %[[ROUND_MUL:.*]] = hivm.hir.vmul ins(%[[ARG0]], %[[PI_REC]] : tensor<4xf32>, f32) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[ROUND:.*]] = hivm.hir.vcast ins(%[[ROUND_MUL]] : tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) round_mode = <round> -> tensor<4xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: return %[[DIV]] : tensor<4xf32>
// CHECK-NOT: hivm.hir.vtan
func.func @test_NormalizeTan_hivm_vtan_f32(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = tensor.empty() : tensor<4xf32>
  %1 = hivm.hir.vtan ins(%arg0 : tensor<4xf32>) outs(%0 : tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeTan_hivm_vtan_f16(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4xf16>) -> tensor<4xf16> {
// CHECK: %[[EMPTY_F32:.*]] = tensor.empty() : tensor<4xf32>
// CHECK: %[[CAST_IN:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<4xf16>) outs(%[[EMPTY_F32]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[EMPTY_F16:.*]] = tensor.empty() : tensor<4xf16>
// CHECK-NEXT: %[[CAST_OUT:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<4xf32>) outs(%[[EMPTY_F16]] : tensor<4xf16>) round_mode = <round> -> tensor<4xf16>
// CHECK-NEXT: return %[[CAST_OUT]] : tensor<4xf16>
// CHECK-NOT: hivm.hir.vtan
func.func @test_NormalizeTan_hivm_vtan_f16(%arg0 : tensor<4xf16>) -> tensor<4xf16> {
  %0 = tensor.empty() : tensor<4xf16>
  %1 = hivm.hir.vtan ins(%arg0 : tensor<4xf16>) outs(%0 : tensor<4xf16>) -> tensor<4xf16>
  return %1 : tensor<4xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeTanh_hivm_vtanh_f32(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG: %[[NEG_ONE:.*]] = arith.constant -1.000000e+00 : f32
// CHECK-DAG: %[[TWO:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG: %[[CLIP_LO:.*]] = arith.constant -8.800000e+00 : f32
// CHECK-DAG: %[[CLIP_HI:.*]] = arith.constant 8.800000e+00 : f32
// CHECK: %[[CLAMP_HI:.*]] = hivm.hir.vmin ins(%[[ARG0]], %[[CLIP_HI]] : tensor<4xf32>, f32)
// CHECK: %[[CLAMP:.*]] = hivm.hir.vmax ins(%[[CLAMP_HI]], %[[CLIP_LO]] : tensor<4xf32>, f32)
// CHECK: %[[MUL:.*]] = hivm.hir.vmul ins(%[[CLAMP]], %[[TWO]] : tensor<4xf32>, f32)
// CHECK: %[[EXP:.*]] = hivm.hir.vexp ins(%[[MUL]] : tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[NUM:.*]] = hivm.hir.vadd ins(%[[EXP]], %[[NEG_ONE]] : tensor<4xf32>, f32)
// CHECK: %[[DEN:.*]] = hivm.hir.vadd ins(%[[EXP]], %[[ONE]] : tensor<4xf32>, f32)
// CHECK: %[[RES:.*]] = hivm.hir.vdiv ins(%[[NUM]], %[[DEN]] : tensor<4xf32>, tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: return %[[RES]] : tensor<4xf32>
// CHECK-NOT: hivm.hir.vtanh
func.func @test_NormalizeTanh_hivm_vtanh_f32(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = tensor.empty() : tensor<4xf32>
  %1 = hivm.hir.vtanh ins(%arg0 : tensor<4xf32>) outs(%0 : tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeTanh_hivm_vtanh_f16(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4xf16>) -> tensor<4xf16> {
// CHECK: %[[EMPTY_F32:.*]] = tensor.empty() : tensor<4xf32>
// CHECK: %[[CAST_IN:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<4xf16>) outs(%[[EMPTY_F32]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[CLAMP_HI:.*]] = hivm.hir.vmin ins(%[[CAST_IN]], %{{.*}} : tensor<4xf32>, f32)
// CHECK: %[[CLAMP:.*]] = hivm.hir.vmax ins(%[[CLAMP_HI]], %{{.*}} : tensor<4xf32>, f32)
// CHECK: %[[MUL:.*]] = hivm.hir.vmul ins(%[[CLAMP]], %{{.*}} : tensor<4xf32>, f32)
// CHECK: %[[EXP:.*]] = hivm.hir.vexp ins(%[[MUL]] : tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[NUM:.*]] = hivm.hir.vadd ins(%[[EXP]], %{{.*}} : tensor<4xf32>, f32)
// CHECK: %[[DEN:.*]] = hivm.hir.vadd ins(%[[EXP]], %{{.*}} : tensor<4xf32>, f32)
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%[[NUM]], %[[DEN]] : tensor<4xf32>, tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[EMPTY_F16:.*]] = tensor.empty() : tensor<4xf16>
// CHECK-NEXT: %[[CAST_OUT:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<4xf32>) outs(%[[EMPTY_F16]] : tensor<4xf16>) round_mode = <round> -> tensor<4xf16>
// CHECK-NEXT: return %[[CAST_OUT]] : tensor<4xf16>
// CHECK-NOT: hivm.hir.vtanh
func.func @test_NormalizeTanh_hivm_vtanh_f16(%arg0 : tensor<4xf16>) -> tensor<4xf16> {
  %0 = tensor.empty() : tensor<4xf16>
  %1 = hivm.hir.vtanh ins(%arg0 : tensor<4xf16>) outs(%0 : tensor<4xf16>) -> tensor<4xf16>
  return %1 : tensor<4xf16>
}
