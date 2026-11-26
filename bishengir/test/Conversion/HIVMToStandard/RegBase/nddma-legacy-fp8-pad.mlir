// RUN: bishengir-opt %s -convert-hivm-to-std | FileCheck %s
// The legacy anchor keeps the plain symbol, without the _span suffix. Its FP8
// pad value rides the same f32 bit carrier as the span anchor's, so a zero pad
// yields a zero carrier. A non-zero pad on this anchor must survive too; see
// nddma-fp8-pad-value.mlir.
module attributes {hacc.target = #hacc.target<"Ascend950">} {
 // CHECK-LABEL: func.func @legacy_fp8_pad(
 // CHECK: arith.constant 0.000000e+00 : f32
 // CHECK: call @load_gm_to_ubuf_2d_float8_e4m3_t({{.*}})
 func.func @legacy_fp8_pad(%src: memref<1x8xf8E4M3FN, #hivm.address_space<gm>>, %dst: memref<1x10xf8E4M3FN, #hivm.address_space<ub>>) {
  %p = arith.constant 0.0 : f8E4M3FN
  %lp = arith.constant 1 : index
  hivm.hir.load ins(%src : memref<1x8xf8E4M3FN, #hivm.address_space<gm>>) outs(%dst : memref<1x10xf8E4M3FN, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f8E4M3FN left_padding_num = %lp : index
  return
 }
}
