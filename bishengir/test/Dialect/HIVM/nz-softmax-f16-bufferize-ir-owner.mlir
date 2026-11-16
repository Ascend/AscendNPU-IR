// RUN: bishengir-opt %s --one-shot-bufferize | FileCheck %s --check-prefix=SHARED-P --implicit-check-not=memref.copy --implicit-check-not=memref.alloc

// Both calls ignore their shared P placeholder and reuse distinct score roots.
// SHARED-P-LABEL: func.func @shared_empty_p_init
// SHARED-P: %[[ROOT_E0:.*]] = memref.alloc() {{.*}} : memref<65536xi8>
// SHARED-P: %[[SCORE_E0:.*]] = memref.view %[[ROOT_E0]]{{.*}} to memref<32x4x16x16xf16>
// SHARED-P: %[[ROOT_E1:.*]] = memref.alloc() {{.*}} : memref<65536xi8>
// SHARED-P: %[[SCORE_E1:.*]] = memref.view %[[ROOT_E1]]{{.*}} to memref<32x4x16x16xf16>
// SHARED-P: %[[FLAT_E0:.*]] = memref.view %[[ROOT_E0]]{{.*}} to memref<65536xf8E4M3FN>
// SHARED-P: %[[P_E0:.*]] = memref.reinterpret_cast %[[FLAT_E0]] to offset: [0], sizes: [16, 4, 16, 32], strides: [4096, 512, 32, 1]
// SHARED-P: hivm.hir.custom
// SHARED-P-SAME: ins(%[[SCORE_E0]],
// SHARED-P-SAME: outs(%[[P_E0]],
// SHARED-P: %[[FLAT_E1:.*]] = memref.view %[[ROOT_E1]]{{.*}} to memref<65536xf8E4M3FN>
// SHARED-P: %[[P_E1:.*]] = memref.reinterpret_cast %[[FLAT_E1]] to offset: [0], sizes: [16, 4, 16, 32], strides: [4096, 512, 32, 1]
// SHARED-P: hivm.hir.custom
// SHARED-P-SAME: ins(%[[SCORE_E1]],
// SHARED-P-SAME: outs(%[[P_E1]],
// SHARED-P: return
#nz_score = affine_map<(n1, m1, m0, n0, lut) -> (n1, m1, m0, n0)>
#nz_state = affine_map<(n1, m1, m0, n0, lut) -> (16 * m1 + m0)>
#nz_lut = affine_map<(n1, m1, m0, n0, lut) -> (lut)>
#nz_p = affine_map<(n1, m1, m0, n0, lut) -> (n1 floordiv 2, m1, m0, (n1 mod 2) * 16 + n0)>

func.func @shared_empty_p_init(%old0: memref<64xf16>, %old1: memref<64xf16>, %lut: memref<256xi8>, %new0: memref<64xf16>, %new1: memref<64xf16>, %sum0: memref<64xf32>, %sum1: memref<64xf32>) {
  %buf0 = memref.alloc() : memref<32x4x16x16xf16>
  %score0 = bufferization.to_tensor %buf0 restrict writable : memref<32x4x16x16xf16>
  %buf1 = memref.alloc() : memref<32x4x16x16xf16>
  %score1 = bufferization.to_tensor %buf1 restrict writable : memref<32x4x16x16xf16>
  %m0 = bufferization.to_tensor %old0 restrict : memref<64xf16>
  %m1 = bufferization.to_tensor %old1 restrict : memref<64xf16>
  %lt = bufferization.to_tensor %lut restrict : memref<256xi8>
  %mn0 = bufferization.to_tensor %new0 restrict writable : memref<64xf16>
  %mn1 = bufferization.to_tensor %new1 restrict writable : memref<64xf16>
  %st0 = bufferization.to_tensor %sum0 restrict writable : memref<64xf32>
  %st1 = bufferization.to_tensor %sum1 restrict writable : memref<64xf32>
  %p = tensor.empty() : tensor<16x4x16x32xf8E4M3FN>
  %r0:3 = hivm.hir.custom {iterator_types = [#hivm.iterator_type<opaque>, #hivm.iterator_type<parallel>, #hivm.iterator_type<opaque>, #hivm.iterator_type<opaque>, #hivm.iterator_type<opaque>], indexing_map = [#nz_score, #nz_state, #nz_lut, #nz_p, #nz_state, #nz_state], symbol = "__builtin_online_softmax_nz", hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>} "__builtin_online_softmax_nz" ins(%score0, %m0, %lt : tensor<32x4x16x16xf16> {inplace_outs = [0 : i32, affine_map<(d0, d1, d2, d3) -> (d0 * 2, d1, d2, d3)>]}, tensor<64xf16>, tensor<256xi8>) outs(%p, %mn0, %st0 : tensor<16x4x16x32xf8E4M3FN>, tensor<64xf16>, tensor<64xf32>) -> (tensor<16x4x16x32xf8E4M3FN>, tensor<64xf16>, tensor<64xf32>)
  %r1:3 = hivm.hir.custom {iterator_types = [#hivm.iterator_type<opaque>, #hivm.iterator_type<parallel>, #hivm.iterator_type<opaque>, #hivm.iterator_type<opaque>, #hivm.iterator_type<opaque>], indexing_map = [#nz_score, #nz_state, #nz_lut, #nz_p, #nz_state, #nz_state], symbol = "__builtin_online_softmax_nz", hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>} "__builtin_online_softmax_nz" ins(%score1, %m1, %lt : tensor<32x4x16x16xf16> {inplace_outs = [0 : i32, affine_map<(d0, d1, d2, d3) -> (d0 * 2, d1, d2, d3)>]}, tensor<64xf16>, tensor<256xi8>) outs(%p, %mn1, %st1 : tensor<16x4x16x32xf8E4M3FN>, tensor<64xf16>, tensor<64xf32>) -> (tensor<16x4x16x32xf8E4M3FN>, tensor<64xf16>, tensor<64xf32>)
  return
}
