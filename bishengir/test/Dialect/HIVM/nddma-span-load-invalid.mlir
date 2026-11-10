// RUN: bishengir-opt %s -split-input-file -verify-diagnostics
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @negative(%src: memref<1x32xf32, #hivm.address_space<gm>>, %dst: memref<3x40xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant -1 : index
  %rp = arith.constant 9 : index
  // expected-error @+1 {{span padding counts must be nonnegative}}
  hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<3x40xf32, #hivm.address_space<ub>>) {pad_anchor = #hivm.pad_anchor<span>} pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index broadcast_axes = [0]
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @bad_sum(%src: memref<1x32xf32, #hivm.address_space<gm>>, %dst: memref<3x40xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 3 : index
  %rp = arith.constant 6 : index
  // expected-error @+1 {{span requires D = lp + W + rp without overflow}}
  hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<3x40xf32, #hivm.address_space<ub>>) {pad_anchor = #hivm.pad_anchor<span>} pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index broadcast_axes = [0]
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @bad_axis(%src: memref<1x32xf32, #hivm.address_space<gm>>, %dst: memref<3x40xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 3 : index
  %rp = arith.constant 5 : index
  // expected-error @+1 {{span broadcast axes must be unique non-last axes}}
  hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<3x40xf32, #hivm.address_space<ub>>) {pad_anchor = #hivm.pad_anchor<span>} pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index broadcast_axes = [1]
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @bad_shape(%src: memref<2x32xf32, #hivm.address_space<gm>>, %dst: memref<3x40xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 3 : index
  %rp = arith.constant 5 : index
  // expected-error @+1 {{span non-broadcast dimensions must match}}
  hivm.hir.load ins(%src : memref<2x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<3x40xf32, #hivm.address_space<ub>>) {pad_anchor = #hivm.pad_anchor<span>} pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @overlap(%src: memref<1x32xf32, #hivm.address_space<gm>>, %dst: memref<3x40xf32, strided<[39, 1]>, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 3 : index
  %rp = arith.constant 5 : index
  // expected-error @+1 {{span rows or planes overlap}}
  hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<3x40xf32, strided<[39, 1]>, #hivm.address_space<ub>>) {pad_anchor = #hivm.pad_anchor<span>} pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index broadcast_axes = [0]
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950">} {
 func.func @bad_target(%src: memref<1x32xf32, #hivm.address_space<gm>>, %dst: memref<3x40xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 3 : index
  %rp = arith.constant 5 : index
  // expected-error @+1 {{span padding requires A6}}
  hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<3x40xf32, #hivm.address_space<ub>>) {pad_anchor = #hivm.pad_anchor<span>} pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index broadcast_axes = [0]
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @bad_mode(%src: memref<1x32xf32, #hivm.address_space<gm>>, %dst: memref<3x40xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 3 : index
  %rp = arith.constant 5 : index
  // expected-error @+1 {{span padding requires PadValue}}
  hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<3x40xf32, #hivm.address_space<ub>>) {pad_anchor = #hivm.pad_anchor<span>} pad_mode = <PadFirstElem> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index broadcast_axes = [0]
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @init(%src: memref<1x32xf32, #hivm.address_space<gm>>, %dst: memref<3x40xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 3 : index
  %rp = arith.constant 5 : index
  // expected-error @+1 {{span padding does not support initialization or transpose}}
  hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<3x40xf32, #hivm.address_space<ub>>) {pad_anchor = #hivm.pad_anchor<span>, init_out_buffer = true} pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index broadcast_axes = [0]
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @backing_range(%src: memref<1x32xf32, #hivm.address_space<gm>>) {
  %root = memref.alloc() : memref<100xf32, #hivm.address_space<ub>>
  %dst = memref.reinterpret_cast %root to offset: [0], sizes: [3, 40], strides: [40, 1] : memref<100xf32, #hivm.address_space<ub>> to memref<3x40xf32, #hivm.address_space<ub>>
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 3 : index
  %rp = arith.constant 5 : index
  // expected-error @+1 {{span view exceeds its backing allocation}}
  hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<3x40xf32, #hivm.address_space<ub>>) {pad_anchor = #hivm.pad_anchor<span>} pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index broadcast_axes = [0]
  return
 }
}


// -----
// Span requires rank 2 or 3.
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @span_bad_rank(%src: memref<32xf32, #hivm.address_space<gm>>, %dst: memref<40xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 3 : index
  %rp = arith.constant 5 : index
  // expected-error @+1 {{span padding requires rank 2 or 3 memrefs}}
  hivm.hir.load ins(%src : memref<32xf32, #hivm.address_space<gm>>) outs(%dst : memref<40xf32, #hivm.address_space<ub>>) {pad_anchor = #hivm.pad_anchor<span>} pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index
  return
 }
}

// -----
// Span is GM -> UB only.
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @span_bad_space(%src: memref<1x32xf32, #hivm.address_space<ub>>, %dst: memref<3x40xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 3 : index
  %rp = arith.constant 5 : index
  // expected-error @+1 {{span padding requires GM -> UB}}
  hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<ub>>) outs(%dst : memref<3x40xf32, #hivm.address_space<ub>>) {pad_anchor = #hivm.pad_anchor<span>} pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index
  return
 }
}

// -----
// Span supports i8/i16/i32/f16/bf16/f32/fp8 only; i64 must be rejected.
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @span_bad_element(%src: memref<1x32xi64, #hivm.address_space<gm>>, %dst: memref<3x40xi64, #hivm.address_space<ub>>) {
  %p = arith.constant 7 : i64
  %lp = arith.constant 3 : index
  %rp = arith.constant 5 : index
  // expected-error @+1 {{span padding has an unsupported element type}}
  hivm.hir.load ins(%src : memref<1x32xi64, #hivm.address_space<gm>>) outs(%dst : memref<3x40xi64, #hivm.address_space<ub>>) {pad_anchor = #hivm.pad_anchor<span>} pad_mode = <PadValue> pad_value = %p : i64 left_padding_num = %lp : index right_padding_num = %rp : index
  return
 }
}

// -----
// Span needs both lp and rp explicitly; rp cannot be inferred.
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @span_missing_rp(%src: memref<1x32xf32, #hivm.address_space<gm>>, %dst: memref<3x40xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 3 : index
  // expected-error @+1 {{span padding requires explicit index lp and rp}}
  hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<3x40xf32, #hivm.address_space<ub>>) {pad_anchor = #hivm.pad_anchor<span>} pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index
  return
 }
}
