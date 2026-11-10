// RUN: not bishengir-opt %s -convert-hivm-to-std -split-input-file 2>&1 | FileCheck %s

// The legacy template has no broadcast+pad combination; the in-row padded
// form does.
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @brc_with_pad(%src: memref<1x32xf32, #hivm.address_space<gm>>) {
  %ub = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
  %c0 = arith.constant 0 : index
  // The verifier rejects the combination first; the identical guard in the
  // lowering is defensive and unreachable while this check exists.
  // CHECK: broadcast_axes cannot be combined with padding in the legacy load form
  hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%ub : memref<4x32xf32, #hivm.address_space<ub>>) left_padding_num = %c0 : index broadcast_axes = [0]
  return
}
}
