// RUN: bishengir-opt %s -convert-hivm-to-std | FileCheck %s
// FP8 keeps the f32 pad bit carrier. The remaining ABI tail is lp, rp,
// eviction and axes; no allocation-size parameters are introduced.
module attributes {hacc.target = #hacc.target<"Ascend960">} {
// CHECK-LABEL: func.func @span_f8E4M3FN_2(
// CHECK: %[[FP8:.*]] = arith.truncf %arg2 : f32 to f8E4M3FN
// CHECK: %[[BYTE:.*]] = arith.bitcast %[[FP8]] : f8E4M3FN to i8
// CHECK: %[[BITS:.*]] = arith.extui %[[BYTE]] : i8 to i32
// CHECK: %[[CARRIER:.*]] = arith.bitcast %[[BITS]] : i32 to f32
// CHECK: call @load_gm_to_ubuf_2d_float8_e4m3_t_span({{.*}}, %[[CARRIER]], %c2, %c3, %c0_i32, %c1_i32) :
// CHECK-NOT: hivm.hir.load
 func.func @span_f8E4M3FN_2(%src: memref<1x4xf8E4M3FN, #hivm.address_space<gm>>, %dst: memref<2x9xf8E4M3FN, #hivm.address_space<ub>>, %value: f32) {
  %p = arith.truncf %value : f32 to f8E4M3FN
  %lp = arith.constant 2 : index
  %rp = arith.constant 3 : index
  hivm.hir.load ins(%src : memref<1x4xf8E4M3FN, #hivm.address_space<gm>>) outs(%dst : memref<2x9xf8E4M3FN, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f8E4M3FN left_padding_num = %lp : index right_padding_num = %rp : index broadcast_axes = [0]
  return
 }
// CHECK-LABEL: func.func @span_f8E4M3FN_3(
// CHECK: %[[FP8:.*]] = arith.truncf %arg2 : f32 to f8E4M3FN
// CHECK: %[[BYTE:.*]] = arith.bitcast %[[FP8]] : f8E4M3FN to i8
// CHECK: %[[BITS:.*]] = arith.extui %[[BYTE]] : i8 to i32
// CHECK: %[[CARRIER:.*]] = arith.bitcast %[[BITS]] : i32 to f32
// CHECK: call @load_gm_to_ubuf_3d_float8_e4m3_t_span({{.*}}, %[[CARRIER]], %c2, %c3, %c0_i32, %c3_i32) :
// CHECK-NOT: hivm.hir.load
 func.func @span_f8E4M3FN_3(%src: memref<1x1x4xf8E4M3FN, #hivm.address_space<gm>>, %dst: memref<2x2x9xf8E4M3FN, #hivm.address_space<ub>>, %value: f32) {
  %p = arith.truncf %value : f32 to f8E4M3FN
  %lp = arith.constant 2 : index
  %rp = arith.constant 3 : index
  hivm.hir.load ins(%src : memref<1x1x4xf8E4M3FN, #hivm.address_space<gm>>) outs(%dst : memref<2x2x9xf8E4M3FN, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f8E4M3FN left_padding_num = %lp : index right_padding_num = %rp : index broadcast_axes = [0, 1]
  return
 }
// CHECK-LABEL: func.func @span_f8E5M2_2(
// CHECK: %[[FP8:.*]] = arith.truncf %arg2 : f32 to f8E5M2
// CHECK: %[[BYTE:.*]] = arith.bitcast %[[FP8]] : f8E5M2 to i8
// CHECK: %[[BITS:.*]] = arith.extui %[[BYTE]] : i8 to i32
// CHECK: %[[CARRIER:.*]] = arith.bitcast %[[BITS]] : i32 to f32
// CHECK: call @load_gm_to_ubuf_2d_float8_e5m2_t_span({{.*}}, %[[CARRIER]], %c2, %c3, %c0_i32, %c1_i32) :
// CHECK-NOT: hivm.hir.load
 func.func @span_f8E5M2_2(%src: memref<1x4xf8E5M2, #hivm.address_space<gm>>, %dst: memref<2x9xf8E5M2, #hivm.address_space<ub>>, %value: f32) {
  %p = arith.truncf %value : f32 to f8E5M2
  %lp = arith.constant 2 : index
  %rp = arith.constant 3 : index
  hivm.hir.load ins(%src : memref<1x4xf8E5M2, #hivm.address_space<gm>>) outs(%dst : memref<2x9xf8E5M2, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f8E5M2 left_padding_num = %lp : index right_padding_num = %rp : index broadcast_axes = [0]
  return
 }
// CHECK-LABEL: func.func @span_f8E5M2_3(
// CHECK: %[[FP8:.*]] = arith.truncf %arg2 : f32 to f8E5M2
// CHECK: %[[BYTE:.*]] = arith.bitcast %[[FP8]] : f8E5M2 to i8
// CHECK: %[[BITS:.*]] = arith.extui %[[BYTE]] : i8 to i32
// CHECK: %[[CARRIER:.*]] = arith.bitcast %[[BITS]] : i32 to f32
// CHECK: call @load_gm_to_ubuf_3d_float8_e5m2_t_span({{.*}}, %[[CARRIER]], %c2, %c3, %c0_i32, %c3_i32) :
// CHECK-NOT: hivm.hir.load
 func.func @span_f8E5M2_3(%src: memref<1x1x4xf8E5M2, #hivm.address_space<gm>>, %dst: memref<2x2x9xf8E5M2, #hivm.address_space<ub>>, %value: f32) {
  %p = arith.truncf %value : f32 to f8E5M2
  %lp = arith.constant 2 : index
  %rp = arith.constant 3 : index
  hivm.hir.load ins(%src : memref<1x1x4xf8E5M2, #hivm.address_space<gm>>) outs(%dst : memref<2x2x9xf8E5M2, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f8E5M2 left_padding_num = %lp : index right_padding_num = %rp : index broadcast_axes = [0, 1]
  return
 }
}
