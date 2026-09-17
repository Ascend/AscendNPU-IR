// RUN: bishengir-opt %s --hivm-bubble-up-extract-slice | FileCheck %s

// CHECK-DAG: #[[$NZ_SCORE:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[$NZ_STATE:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1 * 16 + d2)>
// CHECK-DAG: #[[$NZ_LUT:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d4)>
// CHECK-DAG: #[[$NZ_P:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0 floordiv 2, d1, d2, (d0 mod 2) * 16 + d3)>

// CHECK-LABEL: func.func @nz_softmax_m128_static(
// CHECK: %[[CALL:.*]]:3 = hivm.hir.custom
// CHECK-SAME: indexing_map = [#[[$NZ_SCORE]], #[[$NZ_STATE]], #[[$NZ_LUT]], #[[$NZ_P]], #[[$NZ_STATE]], #[[$NZ_STATE]]]
// CHECK-SAME: iterator_types = [#hivm.iterator_type<opaque>, #hivm.iterator_type<parallel>, #hivm.iterator_type<opaque>, #hivm.iterator_type<opaque>, #hivm.iterator_type<opaque>]
// CHECK-SAME: ins({{.*}} : tensor<32x4x16x16xf16>, tensor<64xf16>, tensor<256xi8>)
// CHECK-SAME: outs({{.*}} : tensor<16x4x16x32xf8E4M3FN>, tensor<64xf16>, tensor<64xf32>)
// CHECK-NOT: hivm.hir.custom
// CHECK: return %[[CALL]]#0, %[[CALL]]#1, %[[CALL]]#2
#nz_score = affine_map<(n1, m1, m0, n0, lut) -> (n1, m1, m0, n0)>
#nz_state = affine_map<(n1, m1, m0, n0, lut) -> (16 * m1 + m0)>
#nz_lut = affine_map<(n1, m1, m0, n0, lut) -> (lut)>
#nz_p = affine_map<(n1, m1, m0, n0, lut) -> (n1 floordiv 2, m1, m0, (n1 mod 2) * 16 + n0)>

func.func @nz_softmax_m128_static(%score: tensor<32x8x16x16xf16>, %m: tensor<128xf16>, %lut: tensor<256xi8>, %id: index) -> (tensor<16x4x16x32xf8E4M3FN>, tensor<64xf16>, tensor<64xf32>) {
  %p = tensor.empty() : tensor<16x8x16x32xf8E4M3FN>
  %mo = tensor.empty() : tensor<128xf16>
  %bo = tensor.empty() : tensor<128xf32>
  %r:3 = hivm.hir.custom {iterator_types = [#hivm.iterator_type<opaque>, #hivm.iterator_type<parallel>, #hivm.iterator_type<opaque>, #hivm.iterator_type<opaque>, #hivm.iterator_type<opaque>], indexing_map = [#nz_score, #nz_state, #nz_lut, #nz_p, #nz_state, #nz_state], symbol = "__builtin_online_softmax_nz", hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>} "__builtin_online_softmax_nz" ins(%score, %m, %lut : tensor<32x8x16x16xf16>, tensor<128xf16>, tensor<256xi8>) outs(%p, %mo, %bo : tensor<16x8x16x32xf8E4M3FN>, tensor<128xf16>, tensor<128xf32>) -> (tensor<16x8x16x32xf8E4M3FN>, tensor<128xf16>, tensor<128xf32>)
  %ps = tensor.extract_slice %r#0[0, 4, 0, 0] [16, 4, 16, 32] [1, 1, 1, 1] {to_be_bubbled_slice} : tensor<16x8x16x32xf8E4M3FN> to tensor<16x4x16x32xf8E4M3FN>
  %ms = tensor.extract_slice %r#1[64] [64] [1] {to_be_bubbled_slice} : tensor<128xf16> to tensor<64xf16>
  %ss = tensor.extract_slice %r#2[64] [64] [1] {to_be_bubbled_slice} : tensor<128xf32> to tensor<64xf32>
  return %ps, %ms, %ss : tensor<16x4x16x32xf8E4M3FN>, tensor<64xf16>, tensor<64xf32>
}
