// RUN: bishengir-opt %s -convert-hivm-to-std | FileCheck %s
// On a non-A6 target the in-row form does not exist, so the right count is
// dropped and the legacy six-operand ABI is used. This pins that behaviour:
// the right padding is only carried by the A6 span templates.
module attributes {hacc.target = #hacc.target<"Ascend950">} {
 // CHECK-LABEL: func.func @a5_with_right_pad(
 // CHECK: call @load_gm_to_ubuf_2d_float({{.*}}, %c2, %c0_i32) :
 // CHECK-NOT: _span
 func.func @a5_with_right_pad(%src: memref<1x8xf32, #hivm.address_space<gm>>, %dst: memref<1x12xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 2 : index
  %rp = arith.constant 2 : index
  hivm.hir.load ins(%src : memref<1x8xf32, #hivm.address_space<gm>>) outs(%dst : memref<1x12xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index
  return
 }
}
