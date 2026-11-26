// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9589 -convert-hivm-to-std | FileCheck %s
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend960 -convert-hivm-to-std | FileCheck %s --check-prefix=SPAN
// An FP8 pad value reaches the library in an f32 bit carrier
// (PadValueType<float8_*>::type == float, handed verbatim to set_mov_pad_val),
// so the FP8 byte is moved into the low bits of that carrier and is neither
// numerically extended nor replaced by zero. That holds for the default anchor
// (Ascend950, and Ascend960 without a right padding count) as well as for the
// span anchor: dropping the value on the default anchor made every `other=`
// operand of an fp8 load pad with 0.0 instead, clobbering the tail of the last
// 32-byte block of an unaligned load.

// CHECK-LABEL: func.func @fp8_e4m3_pad_default_anchor(
// CHECK-SAME: %[[PAD:[^:]*]]: f8E4M3FN)
// CHECK: %[[BYTE:.*]] = arith.bitcast %[[PAD]] : f8E4M3FN to i8
// CHECK: %[[BITS:.*]] = arith.extui %[[BYTE]] : i8 to i32
// CHECK: %[[CARRIER:.*]] = arith.bitcast %[[BITS]] : i32 to f32
// CHECK: call @load_gm_to_ubuf_2d_float8_e4m3_t({{.*}}, %[[CARRIER]], {{.*}})
// CHECK-NOT: hivm.hir.load
func.func @fp8_e4m3_pad_default_anchor(%src: memref<1x8xf8E4M3FN, #hivm.address_space<gm>>, %dst: memref<1x10xf8E4M3FN, #hivm.address_space<ub>>, %pad: f8E4M3FN) {
  %lp = arith.constant 1 : index
  hivm.hir.load ins(%src : memref<1x8xf8E4M3FN, #hivm.address_space<gm>>) outs(%dst : memref<1x10xf8E4M3FN, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %pad : f8E4M3FN left_padding_num = %lp : index
  return
}

// A constant pad value takes the same route, and the bitcast chain then folds
// into one f32 constant holding the FP8 byte. `1.0 : f8E4M3FN` is the
// issue-701 case: its bit pattern 0x38 read back as an f32 is the denormal
// 56 * 2^-149 == 7.847270e-44, so that constant -- not the f32 value 1.0, and
// above all not 0.0 -- is what the call must receive.
// CHECK-LABEL: func.func @fp8_e4m3_constant_pad(
// CHECK-NOT: arith.constant 0.000000e+00 : f32
// CHECK: %[[C:.*]] = arith.constant 7.847270e-44 : f32
// CHECK: call @load_gm_to_ubuf_1d_float8_e4m3_t({{.*}}, %[[C]], {{.*}})
func.func @fp8_e4m3_constant_pad(%src: memref<17xf8E4M3FN, #hivm.address_space<gm>>, %dst: memref<32xf8E4M3FN, #hivm.address_space<ub>>) {
  %pad = arith.constant 1.0 : f8E4M3FN
  hivm.hir.load ins(%src : memref<17xf8E4M3FN, #hivm.address_space<gm>>) outs(%dst : memref<32xf8E4M3FN, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %pad : f8E4M3FN
  return
}

// The other FP8 format shares the carrier.
// CHECK-LABEL: func.func @fp8_e5m2_pad_default_anchor(
// CHECK: %[[B2:.*]] = arith.bitcast %{{.*}} : f8E5M2 to i8
// CHECK: %[[E2:.*]] = arith.extui %[[B2]] : i8 to i32
// CHECK: %[[C2:.*]] = arith.bitcast %[[E2]] : i32 to f32
// CHECK: call @load_gm_to_ubuf_1d_float8_e5m2_t({{.*}}, %[[C2]], {{.*}})
func.func @fp8_e5m2_pad_default_anchor(%src: memref<17xf8E5M2, #hivm.address_space<gm>>, %dst: memref<32xf8E5M2, #hivm.address_space<ub>>, %pad: f8E5M2) {
  hivm.hir.load ins(%src : memref<17xf8E5M2, #hivm.address_space<gm>>) outs(%dst : memref<32xf8E5M2, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %pad : f8E5M2
  return
}

// A load with no pad_value operand keeps its zero f32 default: there are no FP8
// bits to carry, and zero is the right pad there.
// CHECK-LABEL: func.func @fp8_no_pad_value(
// CHECK-NOT: arith.bitcast
// CHECK: arith.constant 0.000000e+00 : f32
// CHECK: call @load_gm_to_ubuf_1d_float8_e4m3_t(
func.func @fp8_no_pad_value(%src: memref<32xf8E4M3FN, #hivm.address_space<gm>>, %dst: memref<32xf8E4M3FN, #hivm.address_space<ub>>) {
  hivm.hir.load ins(%src : memref<32xf8E4M3FN, #hivm.address_space<gm>>) outs(%dst : memref<32xf8E4M3FN, #hivm.address_space<ub>>)
  return
}

// The span anchor, which already carried the value, is unchanged. A non-zero
// right padding count selects it, and only on Ascend960.
// SPAN-LABEL: func.func @fp8_e4m3_pad_span_anchor(
// SPAN: %[[SB:.*]] = arith.bitcast %{{.*}} : f8E4M3FN to i8
// SPAN: %[[SE:.*]] = arith.extui %[[SB]] : i8 to i32
// SPAN: %[[SC:.*]] = arith.bitcast %[[SE]] : i32 to f32
// SPAN: call @load_gm_to_ubuf_2d_float8_e4m3_t_span({{.*}}, %[[SC]], {{.*}})
func.func @fp8_e4m3_pad_span_anchor(%src: memref<1x8xf8E4M3FN, #hivm.address_space<gm>>, %dst: memref<1x10xf8E4M3FN, #hivm.address_space<ub>>, %pad: f8E4M3FN) {
  %lp = arith.constant 1 : index
  %rp = arith.constant 1 : index
  hivm.hir.load ins(%src : memref<1x8xf8E4M3FN, #hivm.address_space<gm>>) outs(%dst : memref<1x10xf8E4M3FN, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %pad : f8E4M3FN left_padding_num = %lp : index right_padding_num = %rp : index
  return
}
