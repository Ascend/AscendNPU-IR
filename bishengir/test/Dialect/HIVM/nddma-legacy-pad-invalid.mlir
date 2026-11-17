// RUN: bishengir-opt %s -split-input-file -verify-diagnostics
// The destination-anchored (legacy) form keeps its own shape and pad_mode
// contract. These cases pin the diagnostics on that path, which the in-row
// form does not own.
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @non_last_dim_mismatch(%src: memref<2x8xf32, #hivm.address_space<gm>>, %dst: memref<4x10xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 1 : index
  // expected-error @+1 {{only the last dimension can be different}}
  hivm.hir.load ins(%src : memref<2x8xf32, #hivm.address_space<gm>>) outs(%dst : memref<4x10xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @dst_narrower(%src: memref<2x8xf32, #hivm.address_space<gm>>, %dst: memref<2x6xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 1 : index
  // expected-error @+1 {{dst last dimension cannot be less than src's}}
  hivm.hir.load ins(%src : memref<2x8xf32, #hivm.address_space<gm>>) outs(%dst : memref<2x6xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @first_elem_with_value(%src: memref<1x8xf32, #hivm.address_space<gm>>, %dst: memref<1x10xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  // expected-error @+1 {{if padmode is PadFirstElem, pad_value should not be set!}}
  hivm.hir.load ins(%src : memref<1x8xf32, #hivm.address_space<gm>>) outs(%dst : memref<1x10xf32, #hivm.address_space<ub>>) pad_mode = <PadFirstElem> pad_value = %p : f32
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @value_without_mode(%src: memref<1x8xf32, #hivm.address_space<gm>>, %dst: memref<1x8xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  // expected-error @+1 {{pad_mode unset or PadNull: pad_value must not be set!}}
  hivm.hir.load ins(%src : memref<1x8xf32, #hivm.address_space<gm>>) outs(%dst : memref<1x8xf32, #hivm.address_space<ub>>) pad_value = %p : f32
  return
 }
}

// -----
// rank-0 keeps the pad_value contract on every target.
module attributes {hacc.target = #hacc.target<"Ascend950">} {
 func.func @rank0_pad_value(%src: memref<f32, #hivm.address_space<gm>>, %dst: memref<f32, #hivm.address_space<ub>>, %p: f32) {
  // expected-error @+1 {{pad_mode unset or PadNull: pad_value must not be set!}}
  hivm.hir.load ins(%src : memref<f32, #hivm.address_space<gm>>) outs(%dst : memref<f32, #hivm.address_space<ub>>) pad_value = %p : f32
  return
 }
}

// -----
// rank-0 pairs PadValue with a pad value and PadFirstElem without one.
module attributes {hacc.target = #hacc.target<"Ascend950">} {
 func.func @rank0_pad_value_missing(%src: memref<f32, #hivm.address_space<gm>>, %dst: memref<f32, #hivm.address_space<ub>>) {
  // expected-error @+1 {{if padmode is PadValue, pad_value is required!}}
  hivm.hir.load ins(%src : memref<f32, #hivm.address_space<gm>>) outs(%dst : memref<f32, #hivm.address_space<ub>>) pad_mode = <PadValue>
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950">} {
 func.func @rank0_first_elem_with_value(%src: memref<f32, #hivm.address_space<gm>>, %dst: memref<f32, #hivm.address_space<ub>>, %p: f32) {
  // expected-error @+1 {{if padmode is PadFirstElem, pad_value should not be set!}}
  hivm.hir.load ins(%src : memref<f32, #hivm.address_space<gm>>) outs(%dst : memref<f32, #hivm.address_space<ub>>) pad_mode = <PadFirstElem> pad_value = %p : f32
  return
 }
}

