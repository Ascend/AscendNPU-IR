// RUN: bishengir-opt --convert-hfusion-to-hivm --hivm-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s
// The CHECK lines in this file are derived from the
// `--hfusion-normalize-ops --convert-hfusion-to-hivm` reference pipeline, but
// only the direct `--convert-hfusion-to-hivm --hivm-normalize-ops` pipeline is
// required to match them.
//
// Excluded source HFusion testcases from
// `bishengir/test/Dialect/HFusion/Normalize/hfusion-normalize-math-ops.mlir`:
// - `test_NormalizeExp2_hfusion_elemwise_unary_exp2`
//   Excluded because `convert-hfusion-to-hivm` does not
//   materialize `hivm.hir.vexp2`.
// - `test_NormalizeExp2_hfusion_elemwise_unary_exp2_f16`
//   Same reason as above.
// - `test_NormalizeLdexp_hfusion_frexp`
// - `test_NormalizeLdexp_hfusion_elemwise_binary_ldexp_f32`
// - `test_NormalizePowf_hfusion_powf_f32`
// - `test_NormalizePowf_hfusion_powf_f16`
// - `test_NormalizePowf_hfusion_powf_half`
// - `test_NormalizePowf_hfusion_powf_const_dense`
// - `test_NormalizePowf_hfusion_powf_cast_fill`
// - `test_NormalizePowf_hfusion_powf_5`
// - `test_NormalizePowf_hfusion_powf_0`
// - `test_NormalizePowf_hfusion_powf_2`
//   Excluded because the current `convert-hfusion-to-hivm` path still does not
//   materialize `hivm.hir.vldexp` or `hivm.hir.vpow`. Pure-HIVM coverage for
//   both migrated normalize rules lives in `hivm-normalize-math-ops.mlir`.
// CHECK-LABEL: func.func @test_NormalizeErf_hfusion_elemwise_erf
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1024xf32>)
// CHECK-DAG: %[[UPPER:.*]] = arith.constant 3.920000
// CHECK-DAG: %[[LOWER:.*]] = arith.constant -3.920000
// CHECK: %[[MIN_EMPTY:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[MIN:.*]] = hivm.hir.vmin ins(%[[ARG0]], %[[UPPER]] : tensor<1024xf32>, f32) outs(%[[MIN_EMPTY]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[MAX:.*]] = hivm.hir.vmax ins(%[[MIN]], %[[LOWER]] : tensor<1024xf32>, f32) outs(%[[MIN_EMPTY]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[SQUARE_EMPTY:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[SQUARE:.*]] = hivm.hir.vmul ins(%[[MAX]], %[[MAX]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[SQUARE_EMPTY]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%{{.*}}, %{{.*}} : tensor<1024xf32>, tensor<1024xf32>) outs(%{{.*}} : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: return %[[DIV]]
func.func @test_NormalizeErf_hfusion_elemwise_erf(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<erf>} ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeErf_hfusion_elemwise_erf_f16
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1024xf16>)
// CHECK-DAG: %[[UPPER:.*]] = arith.constant 3.920000
// CHECK-DAG: %[[LOWER:.*]] = arith.constant -3.920000
// CHECK: %[[EMPTY_F32:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[CAST:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<1024xf16>) outs(%[[EMPTY_F32]] : tensor<1024xf32>){{( round_mode = <round>)?}} -> tensor<1024xf32>
// CHECK: %[[MIN_EMPTY:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[MIN:.*]] = hivm.hir.vmin ins(%[[CAST]], %[[UPPER]] : tensor<1024xf32>, f32) outs(%[[MIN_EMPTY]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[MAX:.*]] = hivm.hir.vmax ins(%[[MIN]], %[[LOWER]] : tensor<1024xf32>, f32) outs(%[[MIN_EMPTY]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[SQUARE_EMPTY:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[SQUARE:.*]] = hivm.hir.vmul ins(%[[MAX]], %[[MAX]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[SQUARE_EMPTY]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[DIV:.*]] = hivm.hir.vdiv ins(%{{.*}}, %{{.*}} : tensor<1024xf32>, tensor<1024xf32>) outs(%{{.*}} : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[CAST_BACK:.*]] = hivm.hir.vcast ins(%[[DIV]] : tensor<1024xf32>) outs(%{{.*}} : tensor<1024xf16>) round_mode = <round> -> tensor<1024xf16>
// CHECK: return %[[CAST_BACK]]
func.func @test_NormalizeErf_hfusion_elemwise_erf_f16(%arg0: tensor<1024xf16>) -> tensor<1024xf16> {
    %0 = tensor.empty() : tensor<1024xf16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<erf>} ins(%arg0 : tensor<1024xf16>) outs(%0 : tensor<1024xf16>) -> tensor<1024xf16>
    return %1 : tensor<1024xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeMod_hfusion_mod_bool
// CHECK-SAME: (%arg0: tensor<2048xi1>, %arg1: tensor<2048xi1>)
// CHECK: %[[CST_DENSE:.*]] = arith.constant dense<false> : tensor<1xi1>
// CHECK: %[[EMPTY_I1:.*]] = tensor.empty() : tensor<2048xi1>
// CHECK: %[[EMPTY_F16_1:.*]] = tensor.empty() : tensor<1xf16>
// CHECK: %[[CAST_DENSE:.*]] = hivm.hir.vcast ins(%[[CST_DENSE]] : tensor<1xi1>) outs(%[[EMPTY_F16_1]] : tensor<1xf16>)
// CHECK: %[[EXTRACTED:.*]] = tensor.extract %[[CAST_DENSE]]
// CHECK: %[[EMPTY_F16_2:.*]] = tensor.empty() : tensor<2048xf16>
// CHECK: %[[CAST_I1:.*]] = hivm.hir.vcast ins(%[[EMPTY_I1]] : tensor<2048xi1>) outs(%[[EMPTY_F16_2]] : tensor<2048xf16>)
// CHECK: %[[BRC:.*]] = hivm.hir.vbrc ins(%[[EXTRACTED]] : f16) outs(%[[CAST_I1]] : tensor<2048xf16>)
// CHECK: %[[CMP:.*]] = hivm.hir.vcmp
// CHECK: %[[NOT:.*]] = hivm.hir.vnot ins(%[[CMP]] : tensor<2048xi1>)
// CHECK: return %[[NOT]] : tensor<2048xi1>
func.func @test_NormalizeMod_hfusion_mod_bool(%src0: tensor<2048xi1>, %src1: tensor<2048xi1>) -> tensor<2048xi1> {
    %3 = tensor.empty() : tensor<2048xi1>
    %4 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%src0, %src1 : tensor<2048xi1>, tensor<2048xi1>) outs(%3 : tensor<2048xi1>) -> tensor<2048xi1>
    return %4 : tensor<2048xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeMod_hfusion_mod_f16
// CHECK-SAME: (%arg0: tensor<2048xf16>, %arg1: tensor<2048xf16>)
// CHECK: %cst = arith.constant 0.000000e+00 : f16
// CHECK: %c-1_i16 = arith.constant -1 : i16
// CHECK: %c1_i16 = arith.constant 1 : i16
// CHECK: %c-31744_i16 = arith.constant -31744 : i16
// CHECK: %c32767_i16 = arith.constant 32767 : i16
// CHECK: %cst_0 = arith.constant 0x7E00 : f16
// CHECK: %[[empty0:.*]] = tensor.empty() : tensor<2048xf32>
// CHECK: %[[cast0:.*]] = hivm.hir.vcast ins(%arg0 : tensor<2048xf16>) outs(%[[empty0]] : tensor<2048xf32>){{( round_mode = <round>)?}} -> tensor<2048xf32>
// CHECK: %[[empty1:.*]] = tensor.empty() : tensor<2048xf32>
// CHECK: %[[cast1:.*]] = hivm.hir.vcast ins(%arg1 : tensor<2048xf16>) outs(%[[empty1]] : tensor<2048xf32>){{( round_mode = <round>)?}} -> tensor<2048xf32>
// CHECK: %[[empty2:.*]] = tensor.empty() : tensor<2048xf32>
// CHECK: %[[div:.*]] = hivm.hir.vdiv ins(%[[cast0]], %[[cast1]] : tensor<2048xf32>, tensor<2048xf32>) outs(%[[empty2]] : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK: %[[empty3:.*]] = tensor.empty() : tensor<2048xf16>
// CHECK: %[[trunc:.*]] = hivm.hir.vcast ins(%[[div]] : tensor<2048xf32>) outs(%[[empty3]] : tensor<2048xf16>){{( round_mode = <trunc>)?}} -> tensor<2048xf16>
// CHECK: %[[empty4:.*]] = tensor.empty() : tensor<2048xf16>
// CHECK: %[[mul:.*]] = hivm.hir.vmul ins(%[[trunc]], %arg1 : tensor<2048xf16>, tensor<2048xf16>) outs(%[[empty4]] : tensor<2048xf16>) -> tensor<2048xf16>
// CHECK: %[[empty5:.*]] = tensor.empty() : tensor<2048xf16>
// CHECK: %[[sub:.*]] = hivm.hir.vsub ins(%arg0, %[[mul]] : tensor<2048xf16>, tensor<2048xf16>) outs(%[[empty5]] : tensor<2048xf16>) -> tensor<2048xf16>
// CHECK: %[[brc_empty:.*]] = tensor.empty() : tensor<2048xi16>
// CHECK: %[[brc:.*]] = hivm.hir.vbrc ins(%c32767_i16 : i16) outs(%[[brc_empty]] : tensor<2048xi16>) -> tensor<2048xi16>
// CHECK: %[[bitcast0:.*]] = hivm.hir.bitcast %arg1 : tensor<2048xf16> -> tensor<2048xi16>
// CHECK: %[[and_empty:.*]] = tensor.empty() : tensor<2048xi16>
// CHECK: %[[and:.*]] = hivm.hir.vand ins(%[[bitcast0]], %[[brc]] : tensor<2048xi16>, tensor<2048xi16>) outs(%[[and_empty]] : tensor<2048xi16>) -> tensor<2048xi16>
// CHECK: %[[empty8:.*]] = tensor.empty() : tensor<2048xi16>
// CHECK: %[[add0:.*]] = hivm.hir.vadd ins(%[[and]], %c-31744_i16 : tensor<2048xi16>, i16) outs(%[[empty8]] : tensor<2048xi16>) -> tensor<2048xi16>
// CHECK: %[[bitcast1:.*]] = hivm.hir.bitcast %[[add0]] : tensor<2048xi16> -> tensor<2048xf16>
// CHECK: %[[empty9:.*]] = tensor.empty() : tensor<2048xf16>
// CHECK: %[[abs:.*]] = hivm.hir.vabs ins(%[[bitcast1]] : tensor<2048xf16>) outs(%[[empty9]] : tensor<2048xf16>) -> tensor<2048xf16>
// CHECK: %[[bitcast2:.*]] = hivm.hir.bitcast %[[abs]] : tensor<2048xf16> -> tensor<2048xi16>
// CHECK: %[[empty10:.*]] = tensor.empty() : tensor<2048xi16>
// CHECK: %[[min:.*]] = hivm.hir.vmin ins(%[[bitcast2]], %c1_i16 : tensor<2048xi16>, i16) outs(%[[empty10]] : tensor<2048xi16>) -> tensor<2048xi16>
// CHECK: %[[mul1:.*]] = hivm.hir.vmul ins(%[[min]], %c-1_i16 : tensor<2048xi16>, i16) outs(%[[min]] : tensor<2048xi16>) -> tensor<2048xi16>
// CHECK: %[[add1:.*]] = hivm.hir.vadd ins(%[[mul1]], %c1_i16 : tensor<2048xi16>, i16) outs(%[[mul1]] : tensor<2048xi16>) -> tensor<2048xi16>
// CHECK: %[[empty11:.*]] = tensor.empty() : tensor<2048xf16>
// CHECK: %[[cast2:.*]] = hivm.hir.vcast ins(%[[add1]] : tensor<2048xi16>) outs(%[[empty11]] : tensor<2048xf16>) -> tensor<2048xf16>
// CHECK: %[[empty12:.*]] = tensor.empty() : tensor<2048xi1>
// CHECK: %[[cmp:.*]] = hivm.hir.vcmp ins(%[[cast2]], %cst : tensor<2048xf16>, f16) outs(%{{.*}} : tensor<2048xi1>) -> tensor<2048xi1>
// CHECK: %[[not:.*]] = hivm.hir.vnot ins(%[[cmp]] : tensor<2048xi1>) outs(%{{.*}} : tensor<2048xi1>) -> tensor<2048xi1>
// CHECK: %[[empty14:.*]] = tensor.empty() : tensor<2048xf16>
// CHECK: %[[sel:.*]] = hivm.hir.vsel ins(%[[not]], %cst_0, %[[sub]] : tensor<2048xi1>, f16, tensor<2048xf16>) outs(%[[empty14]] : tensor<2048xf16>) -> tensor<2048xf16>
// CHECK: return %[[sel]] : tensor<2048xf16>
func.func @test_NormalizeMod_hfusion_mod_f16(%src0: tensor<2048xf16>, %src1: tensor<2048xf16>) -> tensor<2048xf16> {
    %3 = tensor.empty() : tensor<2048xf16>
    %4 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%src0, %src1 : tensor<2048xf16>, tensor<2048xf16>) outs(%3 : tensor<2048xf16>) -> tensor<2048xf16>
    return %4 : tensor<2048xf16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeMod_hfusion_mod_f32
// CHECK-SAME: (%arg0: tensor<2048xf32>, %arg1: tensor<2048xf32>)
// CHECK: %cst = arith.constant 0.000000e+00 : f32
// CHECK: %c-1_i32 = arith.constant -1 : i32
// CHECK: %c1_i32 = arith.constant 1 : i32
// CHECK: %c-2139095040_i32 = arith.constant -2139095040 : i32
// CHECK: %c2147483647_i32 = arith.constant 2147483647 : i32
// CHECK: %cst_0 = arith.constant 0x7FC00000 : f32
// CHECK: %[[empty0:.*]] = tensor.empty() : tensor<2048xf32>
// CHECK: %[[div:.*]] = hivm.hir.vdiv ins(%arg0, %arg1 : tensor<2048xf32>, tensor<2048xf32>) outs(%[[empty0]] : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK: %[[empty1:.*]] = tensor.empty() : tensor<2048xf32>
// CHECK: %[[trunc:.*]] = hivm.hir.vcast ins(%[[div]] : tensor<2048xf32>) outs(%[[empty1]] : tensor<2048xf32>) round_mode = <trunc> -> tensor<2048xf32>
// CHECK: %[[empty2:.*]] = tensor.empty() : tensor<2048xf32>
// CHECK: %[[mul:.*]] = hivm.hir.vmul ins(%[[trunc]], %arg1 : tensor<2048xf32>, tensor<2048xf32>) outs(%[[empty2]] : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK: %[[empty3:.*]] = tensor.empty() : tensor<2048xf32>
// CHECK: %[[sub:.*]] = hivm.hir.vsub ins(%arg0, %[[mul]] : tensor<2048xf32>, tensor<2048xf32>) outs(%[[empty3]] : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK: %[[brc_empty:.*]] = tensor.empty() : tensor<2048xi32>
// CHECK: %[[brc:.*]] = hivm.hir.vbrc ins(%c2147483647_i32 : i32) outs(%[[brc_empty]] : tensor<2048xi32>) -> tensor<2048xi32>
// CHECK: %[[bitcast0:.*]] = hivm.hir.bitcast %arg1 : tensor<2048xf32> -> tensor<2048xi32>
// CHECK: %[[and_empty:.*]] = tensor.empty() : tensor<2048xi32>
// CHECK: %[[and:.*]] = hivm.hir.vand ins(%[[bitcast0]], %[[brc]] : tensor<2048xi32>, tensor<2048xi32>) outs(%[[and_empty]] : tensor<2048xi32>) -> tensor<2048xi32>
// CHECK: %[[empty6:.*]] = tensor.empty() : tensor<2048xi32>
// CHECK: %[[add0:.*]] = hivm.hir.vadd ins(%[[and]], %c-2139095040_i32 : tensor<2048xi32>, i32) outs(%[[empty6]] : tensor<2048xi32>) -> tensor<2048xi32>
// CHECK: %[[bitcast1:.*]] = hivm.hir.bitcast %[[add0]] : tensor<2048xi32> -> tensor<2048xf32>
// CHECK: %[[empty7:.*]] = tensor.empty() : tensor<2048xf32>
// CHECK: %[[abs:.*]] = hivm.hir.vabs ins(%[[bitcast1]] : tensor<2048xf32>) outs(%[[empty7]] : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK: %[[bitcast2:.*]] = hivm.hir.bitcast %[[abs]] : tensor<2048xf32> -> tensor<2048xi32>
// CHECK: %[[empty8:.*]] = tensor.empty() : tensor<2048xi32>
// CHECK: %[[min:.*]] = hivm.hir.vmin ins(%[[bitcast2]], %c1_i32 : tensor<2048xi32>, i32) outs(%[[empty8]] : tensor<2048xi32>) -> tensor<2048xi32>
// CHECK: %[[mul1:.*]] = hivm.hir.vmul ins(%[[min]], %c-1_i32 : tensor<2048xi32>, i32) outs(%[[min]] : tensor<2048xi32>) -> tensor<2048xi32>
// CHECK: %[[add1:.*]] = hivm.hir.vadd ins(%[[mul1]], %c1_i32 : tensor<2048xi32>, i32) outs(%[[mul1]] : tensor<2048xi32>) -> tensor<2048xi32>
// CHECK: %[[empty9:.*]] = tensor.empty() : tensor<2048xf32>
// CHECK: %[[cast2:.*]] = hivm.hir.vcast ins(%[[add1]] : tensor<2048xi32>) outs(%[[empty9]] : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK: %[[empty10:.*]] = tensor.empty() : tensor<2048xi1>
// CHECK: %[[cmp:.*]] = hivm.hir.vcmp ins(%[[cast2]], %cst : tensor<2048xf32>, f32) outs(%{{.*}} : tensor<2048xi1>) -> tensor<2048xi1>
// CHECK: %[[not:.*]] = hivm.hir.vnot ins(%[[cmp]] : tensor<2048xi1>) outs(%{{.*}} : tensor<2048xi1>) -> tensor<2048xi1>
// CHECK: %[[empty12:.*]] = tensor.empty() : tensor<2048xf32>
// CHECK: %[[sel:.*]] = hivm.hir.vsel ins(%[[not]], %cst_0, %[[sub]] : tensor<2048xi1>, f32, tensor<2048xf32>) outs(%[[empty12]] : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK: return %[[sel]] : tensor<2048xf32>
func.func @test_NormalizeMod_hfusion_mod_f32(%src0: tensor<2048xf32>, %src1: tensor<2048xf32>) -> tensor<2048xf32> {
    %3 = tensor.empty() : tensor<2048xf32>
    %4 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%src0, %src1 : tensor<2048xf32>, tensor<2048xf32>) outs(%3 : tensor<2048xf32>) -> tensor<2048xf32>
    return %4 : tensor<2048xf32>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeMod_hfusion_mod_i64
// CHECK-SAME: (%arg0: tensor<2048xi64>, %arg1: tensor<2048xi64>)
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<2048xi64>
// CHECK: %[[vmod:.*]] = hivm.hir.vmod ins(%arg0, %arg1 : tensor<2048xi64>, tensor<2048xi64>) outs(%[[empty]] : tensor<2048xi64>) -> tensor<2048xi64>
// CHECK: return %[[vmod]] : tensor<2048xi64>
func.func @test_NormalizeMod_hfusion_mod_i64(%src0: tensor<2048xi64>, %src1: tensor<2048xi64>) -> tensor<2048xi64> {
    %3 = tensor.empty() : tensor<2048xi64>
    %4 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%src0, %src1 : tensor<2048xi64>, tensor<2048xi64>) outs(%3 : tensor<2048xi64>) -> tensor<2048xi64>
    return %4 : tensor<2048xi64>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeMod_hfusion_mod_i8
// CHECK-SAME: (%arg0: tensor<2048xi8>, %arg1: tensor<2048xi8>)
// CHECK: %[[cast0f16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<2048xi8>) outs(%{{.*}} : tensor<2048xf16>) -> tensor<2048xf16>
// CHECK: %[[cast0:.*]] = hivm.hir.vcast ins(%[[cast0f16]] : tensor<2048xf16>) outs(%{{.*}} : tensor<2048xi16>) round_mode = <trunc> -> tensor<2048xi16>
// CHECK: %[[cast1f16:.*]] = hivm.hir.vcast ins(%arg1 : tensor<2048xi8>) outs(%{{.*}} : tensor<2048xf16>) -> tensor<2048xf16>
// CHECK: %[[cast1:.*]] = hivm.hir.vcast ins(%[[cast1f16]] : tensor<2048xf16>) outs(%{{.*}} : tensor<2048xi16>) round_mode = <trunc> -> tensor<2048xi16>
// CHECK: %[[mod:.*]] = hivm.hir.vmod ins(%[[cast0]], %[[cast1]] : tensor<2048xi16>, tensor<2048xi16>) outs(%{{.*}} : tensor<2048xi16>) -> tensor<2048xi16>
// CHECK: %[[cast2:.*]] = hivm.hir.vcast ins(%[[mod]] : tensor<2048xi16>) outs(%{{.*}} : tensor<2048xi8>) round_mode = <truncwithoverflow> -> tensor<2048xi8>
// CHECK: return %[[cast2]] : tensor<2048xi8>
func.func @test_NormalizeMod_hfusion_mod_i8(%src0: tensor<2048xi8>, %src1: tensor<2048xi8>) -> tensor<2048xi8> {
    %3 = tensor.empty() : tensor<2048xi8>
    %4 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%src0, %src1 : tensor<2048xi8>, tensor<2048xi8>) outs(%3 : tensor<2048xi8>) -> tensor<2048xi8>
    return %4 : tensor<2048xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeMod_hfusion_modui_i8
// CHECK-SAME: (%arg0: tensor<2048xi8>, %arg1: tensor<2048xi8>)
// CHECK: %[[cast0f16:.*]] = hivm.hir.vcast ins(%arg0 : tensor<2048xi8>) outs(%{{.*}} : tensor<2048xf16>) cast = <cast_unsigned> -> tensor<2048xf16>
// CHECK: %[[cast0:.*]] = hivm.hir.vcast ins(%[[cast0f16]] : tensor<2048xf16>) outs(%{{.*}} : tensor<2048xi16>) round_mode = <trunc> -> tensor<2048xi16>
// CHECK: %[[cast1f16:.*]] = hivm.hir.vcast ins(%arg1 : tensor<2048xi8>) outs(%{{.*}} : tensor<2048xf16>) cast = <cast_unsigned> -> tensor<2048xf16>
// CHECK: %[[cast1:.*]] = hivm.hir.vcast ins(%[[cast1f16]] : tensor<2048xf16>) outs(%{{.*}} : tensor<2048xi16>) round_mode = <trunc> -> tensor<2048xi16>
// CHECK: %[[mod:.*]] = hivm.hir.vmodui ins(%[[cast0]], %[[cast1]] : tensor<2048xi16>, tensor<2048xi16>) outs(%{{.*}} : tensor<2048xi16>) -> tensor<2048xi16>
// CHECK: %[[cast2:.*]] = hivm.hir.vcast ins(%[[mod]] : tensor<2048xi16>) outs(%{{.*}} : tensor<2048xi8>) round_mode = <truncwithoverflow> cast = <cast_unsigned> -> tensor<2048xi8>
// CHECK: return %[[cast2]] : tensor<2048xi8>
func.func @test_NormalizeMod_hfusion_modui_i8(%src0: tensor<2048xi8>, %src1: tensor<2048xi8>) -> tensor<2048xi8> {
    %3 = tensor.empty() : tensor<2048xi8>
    %4 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>} ins(%src0, %src1 : tensor<2048xi8>, tensor<2048xi8>) outs(%3 : tensor<2048xi8>) -> tensor<2048xi8>
    return %4 : tensor<2048xi8>
}

// -----

// CHECK-LABEL: @test_NormalizeMod_i8_elemwise_mod
// CHECK: hivm.hir.vcast ins(%arg0 : tensor<64xi8>) outs(%{{.*}} : tensor<64xf16>)
// CHECK: hivm.hir.vcast ins(%{{.*}} : tensor<64xf16>) outs(%{{.*}} : tensor<64xi16>) round_mode = <trunc>
// CHECK: hivm.hir.vcast ins(%arg1 : tensor<64xi8>) outs(%{{.*}} : tensor<64xf16>)
// CHECK: hivm.hir.vcast ins(%{{.*}} : tensor<64xf16>) outs(%{{.*}} : tensor<64xi16>) round_mode = <trunc>
// CHECK: hivm.hir.vmod ins(%{{.*}}, %{{.*}} : tensor<64xi16>, tensor<64xi16>)
// CHECK: hivm.hir.vcast ins(%{{.*}} : tensor<64xi16>) outs(%{{.*}} : tensor<64xi8>) round_mode = <truncwithoverflow>
func.func @test_NormalizeMod_i8_elemwise_mod(%arg0: tensor<64xi8>, %arg1: tensor<64xi8>) -> tensor<64xi8> {
  %0 = tensor.empty() : tensor<64xi8>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
                                ins(%arg0, %arg1 : tensor<64xi8>, tensor<64xi8>)
                                outs(%0 : tensor<64xi8>) -> tensor<64xi8>
  return %1 : tensor<64xi8>
}
