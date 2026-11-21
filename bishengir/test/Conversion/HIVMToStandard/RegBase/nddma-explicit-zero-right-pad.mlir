// RUN: bishengir-opt %s -split-input-file -canonicalize | FileCheck %s --check-prefix=CANON
// RUN: bishengir-opt %s -split-input-file -canonicalize -convert-hivm-to-std | FileCheck %s --check-prefix=LOWER


// CANON-LABEL: func.func @a6_no_right
// CANON: hivm.hir.load
// CANON-NOT: right_padding_num
// CANON: return
// LOWER-LABEL: func.func @a6_no_right
// LOWER: call @load_gm_to_ubuf_2d_float(
// LOWER: return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @a6_no_right(%src: memref<4x8xf32, #hivm.address_space<gm>>, %dst: memref<4x8xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 0 : index
  %z = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %r = arith.subi %c8, %c8 : index
  hivm.hir.load ins(%src : memref<4x8xf32, #hivm.address_space<gm>>) outs(%dst : memref<4x8xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index
  return
}
}

// -----
// CANON-LABEL: func.func @a5_zero
// CANON: hivm.hir.load
// CANON-NOT: right_padding_num
// CANON: return
// LOWER-LABEL: func.func @a5_zero
// LOWER: call @load_gm_to_ubuf_2d_float(
// LOWER: return
module attributes {hacc.target = #hacc.target<"Ascend950">} {
func.func @a5_zero(%src: memref<4x8xf32, #hivm.address_space<gm>>, %dst: memref<4x8xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 0 : index
  %z = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %r = arith.subi %c8, %c8 : index
  hivm.hir.load ins(%src : memref<4x8xf32, #hivm.address_space<gm>>) outs(%dst : memref<4x8xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %z : index
  return
}
}

// -----
// CANON-LABEL: func.func @a3_zero
// CANON: hivm.hir.load
// CANON-NOT: right_padding_num
// CANON: return
// Non-RegBase conversion visits device functions only; this is not one.
// LOWER-LABEL: func.func @a3_zero
// LOWER-NOT: call @load_gm_to_ubuf
// LOWER: hivm.hir.load
// LOWER-NOT: right_padding_num
// LOWER-NOT: call @load_gm_to_ubuf
// LOWER: return
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
func.func @a3_zero(%src: memref<4x8xf32, #hivm.address_space<gm>>, %dst: memref<4x8xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 0 : index
  %z = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %r = arith.subi %c8, %c8 : index
  hivm.hir.load ins(%src : memref<4x8xf32, #hivm.address_space<gm>>) outs(%dst : memref<4x8xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %z : index
  return
}
}

// -----
// CANON-LABEL: func.func @no_target_zero
// CANON: hivm.hir.load
// CANON-NOT: right_padding_num
// CANON: return
// Non-RegBase conversion visits device functions only; this is not one.
// LOWER-LABEL: func.func @no_target_zero
// LOWER-NOT: call @load_gm_to_ubuf
// LOWER: hivm.hir.load
// LOWER-NOT: right_padding_num
// LOWER-NOT: call @load_gm_to_ubuf
// LOWER: return
module {
func.func @no_target_zero(%src: memref<4x8xf32, #hivm.address_space<gm>>, %dst: memref<4x8xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 0 : index
  %z = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %r = arith.subi %c8, %c8 : index
  hivm.hir.load ins(%src : memref<4x8xf32, #hivm.address_space<gm>>) outs(%dst : memref<4x8xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %z : index
  return
}
}
