// RUN: bishengir-opt %s -convert-hivm-to-std | FileCheck %s
// Span ABI: src, dst, mode, pad, lp, rp, eviction, axes. No allocation
// metadata is read merely to issue a load from an already legal source view.
// CHECK-LABEL: func.func @dynamic(
// CHECK-NOT: memref.extract_strided_metadata
// CHECK: call @load_gm_to_ubuf_2d_float_span({{.*}}, %arg4, %arg2, %arg3, %c0_i32, %c1_i32) :
// CHECK-NOT: hivm.hir.load
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @dynamic(%src: memref<1x?xf32, strided<[32, 1]>, #hivm.address_space<gm>>, %dst: memref<?x?xf32, strided<[?, 1]>, #hivm.address_space<ub>>, %lp: index, %rp: index, %p: f32) {
  hivm.hir.load ins(%src : memref<1x?xf32, strided<[32, 1]>, #hivm.address_space<gm>>) outs(%dst : memref<?x?xf32, strided<[?, 1]>, #hivm.address_space<ub>>) {pad_anchor = #hivm.pad_anchor<span>} pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index broadcast_axes = [0]
  return
 }
// Static shapes also use the same eight-operand ABI.
// CHECK-LABEL: func.func @static_3d(
// CHECK: call @load_gm_to_ubuf_3d_float_span({{.*}}, %arg2, %c2, %c3, %c0_i32, %c1_i32) :
// CHECK-NOT: hivm.hir.load
 func.func @static_3d(%src: memref<1x3x4xf32, #hivm.address_space<gm>>, %dst: memref<2x3x9xf32, #hivm.address_space<ub>>, %p: f32) {
  %lp = arith.constant 2 : index
  %rp = arith.constant 3 : index
  hivm.hir.load ins(%src : memref<1x3x4xf32, #hivm.address_space<gm>>) outs(%dst : memref<2x3x9xf32, #hivm.address_space<ub>>) {pad_anchor = #hivm.pad_anchor<span>} pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index broadcast_axes = [0]
  return
 }
// A 2-byte element type reuses the same span ABI with an f16 pad operand
// instead of the f32 bit carrier the FP8 path needs, and the row padding stays
// 2 / 4 for a 5-wide source row inside an 11-wide destination row. The template
// then packs the low 16 bits of that pad value into the descriptor.
// CHECK-LABEL: func.func @span_f16_2(
// CHECK: call @load_gm_to_ubuf_2d_half_span({{.*}}, %arg2, %c2, %c4, %c0_i32, %c1_i32) : ({{.*}}, i32, f16, index, index, i32, i32) -> ()
// CHECK-NOT: hivm.hir.load
 func.func @span_f16_2(%src: memref<1x5xf16, #hivm.address_space<gm>>, %dst: memref<3x11xf16, #hivm.address_space<ub>>, %value: f16) {
  %lp = arith.constant 2 : index
  %rp = arith.constant 4 : index
  hivm.hir.load ins(%src : memref<1x5xf16, #hivm.address_space<gm>>) outs(%dst : memref<3x11xf16, #hivm.address_space<ub>>) {pad_anchor = #hivm.pad_anchor<span>} pad_mode = <PadValue> pad_value = %value : f16 left_padding_num = %lp : index right_padding_num = %rp : index broadcast_axes = [0]
  return
 }
}
