// RUN: bishengir-opt %s --canonicalize --one-shot-bufferize="bufferize-function-boundaries" --canonicalize | FileCheck %s
// RUN: bishengir-opt %s --one-shot-bufferize="bufferize-function-boundaries" | FileCheck %s --check-prefix=DIRECT --implicit-check-not=memref.copy
#nz_score = affine_map<(n1, m1, m0, n0, lut) -> (n1, m1, m0, n0)>
#nz_state = affine_map<(n1, m1, m0, n0, lut) -> (16 * m1 + m0)>
#nz_lut = affine_map<(n1, m1, m0, n0, lut) -> (lut)>
#nz_p = affine_map<(n1, m1, m0, n0, lut) -> (n1 floordiv 2, m1, m0, (n1 mod 2) * 16 + n0)>

// Preserve the declared same-SSA output during bufferization.
// CHECK-LABEL: func.func @nz_old_m_explicit_inplace(
// CHECK-SAME: %[[OLD:[a-zA-Z0-9_]+]]: memref<64xf16>
// CHECK-NOT: memref.copy
// CHECK: hivm.hir.custom
// CHECK-SAME: ins(%{{[^,]+}}, %[[OLD]],
// CHECK-SAME: inplace_outs = 1 : i32
// CHECK-SAME: outs(%{{[^,]+}}, %[[OLD]],
// DIRECT-LABEL: func.func @nz_old_m_explicit_inplace(
// DIRECT-SAME: %[[INIT:[a-zA-Z0-9_]+]]: memref<16x4x16x32xf8E4M3FN,
// DIRECT: %[[P_RESULT:.*]] = memref.reinterpret_cast
// DIRECT-SAME: strides: [4096, 512, 32, 1]
// DIRECT: hivm.hir.custom
// DIRECT-SAME: outs(%[[P_RESULT]],
// DIRECT: return %[[P_RESULT]], {{.*}}, %[[INIT]] :
func.func @nz_old_m_explicit_inplace(%old: memref<64xf16>, %indices: memref<256xi8>, %p: tensor<16x4x16x32xf8E4M3FN, {formalization = "preserve"}>) -> (tensor<16x4x16x32xf8E4M3FN, {formalization = "preserve"}>, tensor<64xf16>, tensor<64xf32>, tensor<16x4x16x32xf8E4M3FN, {formalization = "preserve"}>) {
  %buf = memref.alloc() : memref<32x4x16x16xf16>
  %s = bufferization.to_tensor %buf restrict writable : memref<32x4x16x16xf16>
  %m = bufferization.to_tensor %old restrict writable : memref<64xf16>
  %lut = bufferization.to_tensor %indices restrict : memref<256xi8>
  %zs = arith.constant 0.0 : f32
  %se = tensor.empty() : tensor<64xf32>
  %sum = linalg.fill ins(%zs : f32) outs(%se : tensor<64xf32>) -> tensor<64xf32>
  %r:3 = hivm.hir.custom {iterator_types = [#hivm.iterator_type<opaque>, #hivm.iterator_type<parallel>, #hivm.iterator_type<opaque>, #hivm.iterator_type<opaque>, #hivm.iterator_type<opaque>], indexing_map = [#nz_score, #nz_state, #nz_lut, #nz_p, #nz_state, #nz_state], symbol = "__builtin_online_softmax_nz", hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>} "__builtin_online_softmax_nz" ins(%s, %m, %lut : tensor<32x4x16x16xf16> {inplace_outs = [0 : i32, affine_map<(d0, d1, d2, d3) -> (d0 * 2, d1, d2, d3)>]}, tensor<64xf16> {inplace_outs = 1 : i32}, tensor<256xi8>) outs(%p, %m, %sum : tensor<16x4x16x32xf8E4M3FN, {formalization = "preserve"}>, tensor<64xf16>, tensor<64xf32>) -> (tensor<16x4x16x32xf8E4M3FN, {formalization = "preserve"}>, tensor<64xf16>, tensor<64xf32>)
  return %r#0, %r#1, %r#2, %p : tensor<16x4x16x32xf8E4M3FN, {formalization = "preserve"}>, tensor<64xf16>, tensor<64xf32>, tensor<16x4x16x32xf8E4M3FN, {formalization = "preserve"}>
}
