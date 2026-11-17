// RUN: bishengir-opt %s -split-input-file -verify-diagnostics
// The span templates are instantiated for rank 2/3 and exclude 64-bit element
// types; the verifier rejects both before the lowering picks the symbol.
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @span_rank1(%src: memref<8xf32, #hivm.address_space<gm>>, %dst: memref<40xf32, #hivm.address_space<ub>>) {
  %p = arith.constant 7.0 : f32
  %lp = arith.constant 1 : index
  %rp = arith.constant 1 : index
  // expected-error @+1 {{span padding requires rank 2 or 3 memrefs}}
  hivm.hir.load ins(%src : memref<8xf32, #hivm.address_space<gm>>) outs(%dst : memref<40xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : f32 left_padding_num = %lp : index right_padding_num = %rp : index
  return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
 func.func @span_i64(%src: memref<1x8xi64, #hivm.address_space<gm>>, %dst: memref<1x12xi64, #hivm.address_space<ub>>) {
  %p = arith.constant 7 : i64
  %lp = arith.constant 2 : index
  %rp = arith.constant 2 : index
  // expected-error @+1 {{span padding has an unsupported element type}}
  hivm.hir.load ins(%src : memref<1x8xi64, #hivm.address_space<gm>>) outs(%dst : memref<1x12xi64, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %p : i64 left_padding_num = %lp : index right_padding_num = %rp : index
  return
 }
}
