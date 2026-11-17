// RUN: bishengir-opt %s -split-input-file -verify-diagnostics
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @negative(%src: memref<1x32xf32, #hivm.address_space<gm>>, %dst: memref<1x40xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant -1 : index
  %rp = arith.constant 9 : index
  // expected-error @+1 {{span padding counts must be nonnegative}}
  hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<1x40xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @overflow_row(%src: memref<1x32xf32, #hivm.address_space<gm>>, %dst: memref<1x40xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 3 : index
  %rp = arith.constant 6 : index
  // expected-error @+1 {{span requires D = lp + W + rp without overflow}}
  hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<1x40xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @negative_rp(%src: memref<1x32xf32, #hivm.address_space<gm>>, %dst: memref<1x40xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 1 : index
  %rp = arith.constant -1 : index
  // expected-error @+1 {{span padding counts must be nonnegative}}
  hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<1x40xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @broadcast_padded_last_axis(%src: memref<2x1xf32, #hivm.address_space<gm>>, %dst: memref<2x3xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 1 : index
  %rp = arith.constant 1 : index
  // expected-error @+1 {{span broadcast axes must be unique non-last axes}}
  hivm.hir.load ins(%src : memref<2x1xf32, #hivm.address_space<gm>>) outs(%dst : memref<2x3xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index broadcast_axes = [1]
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @noncontiguous_rows(%src: memref<2x8xf32, strided<[16, 2]>, #hivm.address_space<gm>>, %dst: memref<2x10xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %rp = arith.constant 2 : index
  // expected-error @+1 {{span requires contiguous rows and nonnegative offsets}}
  hivm.hir.load ins(%src : memref<2x8xf32, strided<[16, 2]>, #hivm.address_space<gm>>) outs(%dst : memref<2x10xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 right_padding_num = %rp : index
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @with_init(%src: memref<2x8xf32, #hivm.address_space<gm>>, %dst: memref<2x10xf32, #hivm.address_space<ub>>, %cond: i1) {
  %p = arith.constant 7.0 : f32
  %rp = arith.constant 2 : index
  // expected-error @+1 {{span padding does not support initialization or transpose}}
  hivm.hir.load ins(%src : memref<2x8xf32, #hivm.address_space<gm>>) outs(%dst : memref<2x10xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 init_out_buffer = true right_padding_num = %rp : index init_condition = %cond : i1
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @into_cbuf(%src: memref<2x8xf32, #hivm.address_space<gm>>, %dst: memref<2x10xf32, #hivm.address_space<cbuf>>) {
  %p = arith.constant 7.0 : f32
  %rp = arith.constant 2 : index
  // expected-error @+1 {{span padding requires GM -> UB}}
  hivm.hir.load ins(%src : memref<2x8xf32, #hivm.address_space<gm>>) outs(%dst : memref<2x10xf32, #hivm.address_space<cbuf>>) pad_mode = <PadValue> pad_value = %p : f32 right_padding_num = %rp : index
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @rank4(%src: memref<1x2x2x8xf32, #hivm.address_space<gm>>, %dst: memref<1x2x2x12xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %rp = arith.constant 2 : index
  // expected-error @+1 {{span padding requires rank 2 or 3 memrefs}}
  hivm.hir.load ins(%src : memref<1x2x2x8xf32, #hivm.address_space<gm>>) outs(%dst : memref<1x2x2x12xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 right_padding_num = %rp : index
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @first_elem(%src: memref<1x8xf32, #hivm.address_space<gm>>, %dst: memref<1x10xf32, #hivm.address_space<ub>>) {
  %rp = arith.constant 2 : index
  // expected-error @+1 {{span padding requires PadValue and a matching value}}
  hivm.hir.load ins(%src : memref<1x8xf32, #hivm.address_space<gm>>) outs(%dst : memref<1x10xf32, #hivm.address_space<ub>>) pad_mode = <PadFirstElem> right_padding_num = %rp : index
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @transpose(%src: memref<1x8xf32, #hivm.address_space<gm>>, %dst: memref<1x10xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %rp = arith.constant 2 : index
  // expected-error @+1 {{span padding does not support initialization or transpose}}
  hivm.hir.load ins(%src : memref<1x8xf32, #hivm.address_space<gm>>) outs(%dst : memref<1x10xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 right_padding_num = %rp : index may_implicit_transpose_with_last_axis = true
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @rows_overlap(%src: memref<2x8xf32, strided<[4, 1]>, #hivm.address_space<gm>>, %dst: memref<2x10xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %rp = arith.constant 2 : index
  // expected-error @+1 {{span rows or planes overlap}}
  hivm.hir.load ins(%src : memref<2x8xf32, strided<[4, 1]>, #hivm.address_space<gm>>) outs(%dst : memref<2x10xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 right_padding_num = %rp : index
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @address_range(%src: memref<2x8xf32, strided<[4611686018427387904, 1]>, #hivm.address_space<gm>>, %dst: memref<2x10xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %rp = arith.constant 2 : index
  // expected-error @+1 {{span address range overflows}}
  hivm.hir.load ins(%src : memref<2x8xf32, strided<[4611686018427387904, 1]>, #hivm.address_space<gm>>) outs(%dst : memref<2x10xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 right_padding_num = %rp : index
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @exceeds_backing(%src: memref<2x8xf32, #hivm.address_space<gm>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 4 : index
  %rp = arith.constant 4 : index
  %alloc = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
  %view = memref.reinterpret_cast %alloc to offset: [0], sizes: [2, 16], strides: [16, 1] : memref<16xf32, #hivm.address_space<ub>> to memref<2x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
  // expected-error @+1 {{span view exceeds its backing allocation}}
  hivm.hir.load ins(%src : memref<2x8xf32, #hivm.address_space<gm>>) outs(%view : memref<2x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @negative_offset(%src: memref<2x8xf32, strided<[8, 1], offset: -1>, #hivm.address_space<gm>>, %dst: memref<2x10xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %rp = arith.constant 2 : index
  // expected-error @+1 {{span requires contiguous rows and nonnegative offsets}}
  hivm.hir.load ins(%src : memref<2x8xf32, strided<[8, 1], offset: -1>, #hivm.address_space<gm>>) outs(%dst : memref<2x10xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 right_padding_num = %rp : index
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @empty_axes(%src: memref<2x8xf32, #hivm.address_space<gm>>, %dst: memref<2x10xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %rp = arith.constant 2 : index
  // expected-error @+1 {{broadcast_axes expects at least one axis}}
  hivm.hir.load ins(%src : memref<2x8xf32, #hivm.address_space<gm>>) outs(%dst : memref<2x10xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 right_padding_num = %rp : index broadcast_axes = []
  return
 }
}
