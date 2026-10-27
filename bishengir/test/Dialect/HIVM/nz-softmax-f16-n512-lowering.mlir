// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9589 -canonicalize -convert-hivm-to-std="mark-libcall-noinline=true" -split-input-file -verify-diagnostics -o %t
// RUN: FileCheck %s < %t
// RUN: FileCheck %s --check-prefix=DECL < %t
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9589 -convert-hivm-to-std="mark-libcall-noinline=true" -split-input-file -verify-diagnostics -o %t.direct
// RUN: FileCheck %s < %t.direct
// DECL: func.func private @online_softmax_nz_f16_f8e4m3fn(
// DECL-SAME: hacc.always_inline
// CHECK: call @online_softmax_nz_f16_f8e4m3fn
func.func @f16_n512_memref(%score: memref<32x4x16x16xf16>, %m: memref<64xf16>, %lut: memref<256xi8>, %p: memref<16x4x16x32xf8E4M3FN, strided<[4096,512,32,1]>>, %mn: memref<64xf16>, %sum: memref<64xf32>) {
  hivm.hir.custom {symbol = "__builtin_online_softmax_nz", hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>} "__builtin_online_softmax_nz" ins(%score, %m, %lut : memref<32x4x16x16xf16>, memref<64xf16>, memref<256xi8>) outs(%p, %mn, %sum : memref<16x4x16x32xf8E4M3FN, strided<[4096,512,32,1]>>, memref<64xf16>, memref<64xf32>)
  return
}

// CHECK-LABEL: func.func @f16_n512_erased_unit_strides
// CHECK: call @online_softmax_nz_f16_f8e4m3fn_causal
func.func @f16_n512_erased_unit_strides(%score: memref<32x4x16x16xf16>, %m: memref<64xf16>, %lut: memref<256xi8>, %limits: memref<64xi32>, %p: memref<16x4x16x32xf8E4M3FN, strided<[4096,512,32,1]>>, %mn: memref<64xf16>, %sum: memref<64xf32>) {
  %m_dynamic = memref.cast %m : memref<64xf16> to memref<64xf16, strided<[?], offset: ?>>
  %lut_dynamic = memref.cast %lut : memref<256xi8> to memref<256xi8, strided<[?], offset: ?>>
  %limits_dynamic = memref.cast %limits : memref<64xi32> to memref<64xi32, strided<[?], offset: ?>>
  %mn_dynamic = memref.cast %mn : memref<64xf16> to memref<64xf16, strided<[?], offset: ?>>
  %sum_dynamic = memref.cast %sum : memref<64xf32> to memref<64xf32, strided<[?], offset: ?>>
  hivm.hir.custom {symbol = "__builtin_online_softmax_nz", hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>} "__builtin_online_softmax_nz" ins(%score, %m_dynamic, %lut_dynamic, %limits_dynamic : memref<32x4x16x16xf16>, memref<64xf16, strided<[?], offset: ?>>, memref<256xi8, strided<[?], offset: ?>>, memref<64xi32, strided<[?], offset: ?>>) outs(%p, %mn_dynamic, %sum_dynamic : memref<16x4x16x32xf8E4M3FN, strided<[4096,512,32,1]>>, memref<64xf16, strided<[?], offset: ?>>, memref<64xf32, strided<[?], offset: ?>>)
  return
}

// CHECK-LABEL: func.func @f16_n512_dynamic_unit_subview
// CHECK: call @online_softmax_nz_f16_f8e4m3fn
func.func @f16_n512_dynamic_unit_subview(%score: memref<32x4x16x16xf16>, %storage: memref<128xf16>, %lut: memref<256xi8>, %p: memref<16x4x16x32xf8E4M3FN, strided<[4096,512,32,1]>>, %mn: memref<64xf16>, %sum: memref<64xf32>) {
  %c1 = arith.constant 1 : index
  %m = memref.subview %storage[0] [64] [%c1] : memref<128xf16> to memref<64xf16, strided<[?]>>
  hivm.hir.custom {symbol = "__builtin_online_softmax_nz", hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>} "__builtin_online_softmax_nz" ins(%score, %m, %lut : memref<32x4x16x16xf16>, memref<64xf16, strided<[?]>>, memref<256xi8>) outs(%p, %mn, %sum : memref<16x4x16x32xf8E4M3FN, strided<[4096,512,32,1]>>, memref<64xf16>, memref<64xf32>)
  return
}

// CHECK-LABEL: func.func @f16_n512_unit_view_strides
// CHECK: call @online_softmax_nz_f16_f8e4m3fn
func.func @f16_n512_unit_view_strides(%score: memref<32x4x16x16xf16>, %storage: memref<1x128xf16>, %lut: memref<256xi8>, %p: memref<16x4x16x32xf8E4M3FN, strided<[4096,512,32,1]>>, %new_storage: memref<128xf16>, %sum: memref<64xf32>, %offset: index) {
  %c1 = arith.constant 1 : index
  %m = memref.subview %storage[0, 0] [1, 64] [1, %c1] : memref<1x128xf16> to memref<64xf16, strided<[?]>>
  %mn = memref.reinterpret_cast %new_storage to offset: [%offset], sizes: [64], strides: [%c1] : memref<128xf16> to memref<64xf16, strided<[?], offset: ?>>
  hivm.hir.custom {symbol = "__builtin_online_softmax_nz", hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>} "__builtin_online_softmax_nz" ins(%score, %m, %lut : memref<32x4x16x16xf16>, memref<64xf16, strided<[?]>>, memref<256xi8>) outs(%p, %mn, %sum : memref<16x4x16x32xf8E4M3FN, strided<[4096,512,32,1]>>, memref<64xf16, strided<[?], offset: ?>>, memref<64xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @f16_n512_explicit_noinline
// CHECK: call @online_softmax_nz_f16_f8e4m3fn
// DECL: func.func private @online_softmax_nz_f16_f8e4m3fn(
// DECL-SAME: hacc.noinline
// DECL-LABEL: func.func @f16_n512_explicit_noinline
func.func @f16_n512_explicit_noinline(%score: memref<32x4x16x16xf16>, %m: memref<64xf16>, %lut: memref<256xi8>, %p: memref<16x4x16x32xf8E4M3FN, strided<[4096,512,32,1]>>, %mn: memref<64xf16>, %sum: memref<64xf32>) {
  hivm.hir.custom {symbol = "__builtin_online_softmax_nz", hivm.inline_mode = #hivm.inline_mode<no_inline>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>} "__builtin_online_softmax_nz" ins(%score, %m, %lut : memref<32x4x16x16xf16>, memref<64xf16>, memref<256xi8>) outs(%p, %mn, %sum : memref<16x4x16x32xf8E4M3FN, strided<[4096,512,32,1]>>, memref<64xf16>, memref<64xf32>)
  return
}
