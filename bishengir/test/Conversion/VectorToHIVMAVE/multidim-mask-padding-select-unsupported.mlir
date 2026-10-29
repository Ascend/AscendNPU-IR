// RUN: bishengir-opt %s -hacc-append-device-spec -convert-vector-to-hivmave \
// RUN: -split-input-file -verify-diagnostics

// Review finding 10-23 [P2]: a rank>1 mask with no defining op (here a
// `vector<2x16xi1>` block argument) reaches the padding select, and for an
// element type wider than 32 bits neither the splat nor the select has a
// lowering: `HIVMBroadCastScalarOpLowering` covers 8/16/32-bit elements only,
// and `vsel` is defined for b16/b32 sources.  Emitting them anyway left
// `ave.hir.scalar_broadcast` and `ave.hir.vsel` in the IR with a zero exit
// code, so the compile only failed at LLVM translation.  The read is now
// rejected where the decision is made, with the reason attached.
//
// This is a diagnostic, not a lowering: these inputs still do not compile.
// Making them compile needs either a 64-bit broadcast/select (the ISA has
// none) or a proof that the padding is unobservable, which is a cross-op
// analysis -- see the TODO on `canLowerPaddingSelect`.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @i64_blockarg_mask_rejected(
      %src: memref<2x16xi64, #hivm.address_space<ub>>,
      %dst: memref<32xi64, #hivm.address_space<ub>>,
      %mask: vector<2x16xi1>,
      %padding: i64) attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    // expected-error @below {{needs a padding select, which has no AVE lowering for element types wider than 32 bits}}
    %v = vector.transfer_read %src[%c0, %c0], %padding, %mask
        {in_bounds = [true, true],
         permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
        : memref<2x16xi64, #hivm.address_space<ub>>, vector<2x16xi64>
    %flat = vector.shape_cast %v : vector<2x16xi64> to vector<32xi64>
    %flat_mask = vector.shape_cast %mask : vector<2x16xi1> to vector<32xi1>
    vector.maskedstore %dst[%c0], %flat_mask, %flat
        : memref<32xi64, #hivm.address_space<ub>>, vector<32xi1>, vector<32xi64>
    return
  }
}
