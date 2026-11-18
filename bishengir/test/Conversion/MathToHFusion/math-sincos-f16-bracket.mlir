// RUN: bishengir-opt -convert-math-to-hfusion %s -split-input-file | FileCheck %s

// An f16-bracketed math.sin/math.cos pair is fused into one inline
// low-precision recipe: a single pi reduction is shared by both results, the
// sin and cos Taylor polynomials are evaluated on the same reduced angle, and
// no SIMT call and no 2/pi table are needed. The f32 SIMT dispatch lives in
// math-sincos-template-lib.mlir.
//
// This split mirrors the coverage math-cos-f16-bracket.mlir gives the per-op
// path: every supported way the shared input can be derived from f16 and the
// results narrowed back to f16, plus the routing cases where the pair must NOT
// be fused.
//
// CHECK-NOT: memref.global @tbl
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // arith.extf / arith.truncf bracket on both results.
  func.func @bracket_extf_truncf(%arg0: tensor<8x16xf16>) -> (tensor<8x16xf16>, tensor<8x16xf16>) {
    %0 = arith.extf %arg0 : tensor<8x16xf16> to tensor<8x16xf32>
    %s = math.sin %0 : tensor<8x16xf32>
    %c = math.cos %0 : tensor<8x16xf32>
    %s16 = arith.truncf %s : tensor<8x16xf32> to tensor<8x16xf16>
    %c16 = arith.truncf %c : tensor<8x16xf32> to tensor<8x16xf16>
    return %s16, %c16 : tensor<8x16xf16>, tensor<8x16xf16>
  }
  // hfusion.cast bracket on both sides.
  func.func @bracket_hfusion_cast(%arg0: tensor<8x16xf16>) -> (tensor<8x16xf16>, tensor<8x16xf16>) {
    %e0 = tensor.empty() : tensor<8x16xf32>
    %0 = hfusion.cast ins(%arg0 : tensor<8x16xf16>) outs(%e0 : tensor<8x16xf32>) -> tensor<8x16xf32>
    %s = math.sin %0 : tensor<8x16xf32>
    %c = math.cos %0 : tensor<8x16xf32>
    %es = tensor.empty() : tensor<8x16xf16>
    %s16 = hfusion.cast ins(%s : tensor<8x16xf32>) outs(%es : tensor<8x16xf16>) -> tensor<8x16xf16>
    %ec = tensor.empty() : tensor<8x16xf16>
    %c16 = hfusion.cast ins(%c : tensor<8x16xf32>) outs(%ec : tensor<8x16xf16>) -> tensor<8x16xf16>
    return %s16, %c16 : tensor<8x16xf16>, tensor<8x16xf16>
  }
  // f16 provenance through tensor.extract_slice.
  func.func @bracket_input_extract_slice(%arg0: tensor<16x16xf16>) -> (tensor<8x16xf16>, tensor<8x16xf16>) {
    %sl = tensor.extract_slice %arg0[0, 0] [8, 16] [1, 1] : tensor<16x16xf16> to tensor<8x16xf16>
    %0 = arith.extf %sl : tensor<8x16xf16> to tensor<8x16xf32>
    %s = math.sin %0 : tensor<8x16xf32>
    %c = math.cos %0 : tensor<8x16xf32>
    %s16 = arith.truncf %s : tensor<8x16xf32> to tensor<8x16xf16>
    %c16 = arith.truncf %c : tensor<8x16xf32> to tensor<8x16xf16>
    return %s16, %c16 : tensor<8x16xf16>, tensor<8x16xf16>
  }
  // Narrowing through a real tensor.extract_slice before truncf.
  func.func @bracket_output_extract_slice(%arg0: tensor<8x32xf16>) -> (tensor<8x16xf16>, tensor<8x16xf16>) {
    %0 = arith.extf %arg0 : tensor<8x32xf16> to tensor<8x32xf32>
    %s = math.sin %0 : tensor<8x32xf32>
    %c = math.cos %0 : tensor<8x32xf32>
    %ss = tensor.extract_slice %s[0, 0] [8, 16] [1, 1] : tensor<8x32xf32> to tensor<8x16xf32>
    %cs = tensor.extract_slice %c[0, 0] [8, 16] [1, 1] : tensor<8x32xf32> to tensor<8x16xf32>
    %s16 = arith.truncf %ss : tensor<8x16xf32> to tensor<8x16xf16>
    %c16 = arith.truncf %cs : tensor<8x16xf32> to tensor<8x16xf16>
    return %s16, %c16 : tensor<8x16xf16>, tensor<8x16xf16>
  }
  // Rank 6 is above the template-library limit, so an f16-bracketed pair falls
  // back to the HFusion Normalize expansion like any other unsupported rank.
  func.func @bracket_rank6(%arg0: tensor<1x1x1x1x1x2xf16>) -> (tensor<1x1x1x1x1x2xf16>, tensor<1x1x1x1x1x2xf16>) {
    %0 = arith.extf %arg0 : tensor<1x1x1x1x1x2xf16> to tensor<1x1x1x1x1x2xf32>
    %s = math.sin %0 : tensor<1x1x1x1x1x2xf32>
    %c = math.cos %0 : tensor<1x1x1x1x1x2xf32>
    %s16 = arith.truncf %s : tensor<1x1x1x1x1x2xf32> to tensor<1x1x1x1x1x2xf16>
    %c16 = arith.truncf %c : tensor<1x1x1x1x1x2xf32> to tensor<1x1x1x1x1x2xf16>
    return %s16, %c16 : tensor<1x1x1x1x1x2xf16>, tensor<1x1x1x1x1x2xf16>
  }
}

// One shared reduction feeds both polynomials: k = round(x / pi) computed once
// (0.318309873 is 1/pi), then the sin degree-9 polynomial (1/9! =
// 2.75573188E-6, its x^9 coefficient) and the cos even polynomial (1/24 =
// 0.0416666679, its x^4 coefficient) evaluated on the same reduced angle, then
// one (-1)^k sign from a floor-rounded cast.
//
// The recipe is pinned by its positive checks only: if a hivm.hir.custom
// replaced this expansion, the 1/9! and 1/24 constants and the round/floor
// casts below would not match at all, which is why no per-function CHECK-NOT is
// needed here (same convention as math-cos-f16-bracket.mlir).
//
// The greedy folder hoists constants, so match them in any order.
// CHECK-LABEL: func.func @bracket_extf_truncf
// CHECK-DAG: arith.constant 0.318309873 : f32
// CHECK-DAG: arith.constant 3.140625 : f32
// CHECK-DAG: arith.constant 2.75573188E-6 : f32
// CHECK-DAG: arith.constant 0.0416666679 : f32
// -0.5 is the even cos polynomial's x^2 coefficient, which the odd sin
// polynomial never produces, so pinning it proves the fused recipe evaluated the
// even cosine polynomial rather than the solo cos formulation (that one builds
// sin(x - k*pi + pi/2) and has no -0.5 coefficient at all).
// CHECK-DAG: arith.constant -5.000000e-01 : f32
// CHECK: hfusion.cast {{.*}} round_mode = #hfusion.round_mode<round>
// CHECK: hfusion.cast {{.*}} round_mode = #hfusion.round_mode<floor>
// CHECK: arith.truncf
// CHECK: arith.truncf
// CHECK: return

// CHECK-LABEL: func.func @bracket_hfusion_cast
// CHECK-DAG: arith.constant 2.75573188E-6 : f32
// CHECK-DAG: arith.constant 0.0416666679 : f32
// CHECK: hfusion.cast {{.*}} round_mode = #hfusion.round_mode<round>
// CHECK: hfusion.cast {{.*}} round_mode = #hfusion.round_mode<floor>
// CHECK: return

// CHECK-LABEL: func.func @bracket_input_extract_slice
// CHECK-DAG: arith.constant 2.75573188E-6 : f32
// CHECK-DAG: arith.constant 0.0416666679 : f32
// CHECK: hfusion.cast {{.*}} round_mode = #hfusion.round_mode<round>
// CHECK: arith.truncf
// CHECK: arith.truncf
// CHECK: return

// CHECK-LABEL: func.func @bracket_output_extract_slice
// CHECK-DAG: arith.constant 2.75573188E-6 : f32
// CHECK-DAG: arith.constant 0.0416666679 : f32
// CHECK: arith.truncf
// CHECK: arith.truncf
// CHECK: return

// CHECK-LABEL: func.func @bracket_rank6
// CHECK-NOT: hivm.hir.custom
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<sin>}
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>}
// CHECK: return

// -----

// Only one half of the pair is f16-bracketed: the sine is narrowed back to f16
// but the cosine is returned as f32, so the pair must NOT be fused. The sine
// keeps the inline low-precision recipe and the cosine takes the
// template-library path with its own 2/pi table.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @sin_bracketed_cos_not(%arg0: tensor<8x16xf16>) -> (tensor<8x16xf16>, tensor<8x16xf32>) {
    %0 = arith.extf %arg0 : tensor<8x16xf16> to tensor<8x16xf32>
    %s = math.sin %0 : tensor<8x16xf32>
    %c = math.cos %0 : tensor<8x16xf32>
    %s16 = arith.truncf %s : tensor<8x16xf32> to tensor<8x16xf16>
    return %s16, %c : tensor<8x16xf16>, tensor<8x16xf32>
  }
}

// CHECK-LABEL: func.func @sin_bracketed_cos_not
// CHECK: hfusion.cast {{.*}} round_mode = #hfusion.round_mode<round>
// CHECK: hivm.hir.custom {{.*}} symbol = "cos_fp32_simt_2d"
// CHECK-NOT: symbol = "sincos_fp32_simt_2d"
// CHECK: return
