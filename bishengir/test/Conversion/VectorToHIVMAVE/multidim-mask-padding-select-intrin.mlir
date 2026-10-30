// RUN: bishengir-opt %s -hacc-append-device-spec -convert-vector-to-hivmave \
// RUN: -convert-hivmave-to-ave-intrin -split-input-file | FileCheck %s

// Review finding 10-23 [P2]: the rank>1 mask padding select is reachable for a
// mask with no defining op -- a `vector<...xi1>` block argument -- because no
// pattern ever converts a block argument.  For an element type wider than 32
// bits the splat and the select have no lowering at all
// (`HIVMBroadCastScalarOpLowering` covers 8/16/32-bit elements, and `vsel` is
// defined for b16/b32 sources), so they used to survive every later pass
// un-lowered with a zero exit code, failing only at LLVM translation -- the
// same failure mode that made 122 test_quantile cases fail before the i1
// branch existed.  Such reads are now rejected up front with their real
// reason.
//
// These cases run through to the intrinsic layer on purpose: checking only the
// vector-to-ave output is what let the un-lowered ops slip through.

// -----

// A supported element width really does lower: the splat and the select both
// become intrinsics and no `ave.hir.*` op is left behind.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @f32_blockarg_mask_lowers
  // CHECK-NOT: ave.hir.
  // CHECK: hivm_regbaseintrins.intr.hivm.
  func.func @f32_blockarg_mask_lowers(
      %src: memref<2x16xf32, #hivm.address_space<ub>>,
      %dst: memref<32xf32, #hivm.address_space<ub>>,
      %mask: vector<2x16xi1>,
      %padding: f32) attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %v = vector.transfer_read %src[%c0, %c0], %padding, %mask
        {in_bounds = [true, true],
         permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
        : memref<2x16xf32, #hivm.address_space<ub>>, vector<2x16xf32>
    %flat = vector.shape_cast %v : vector<2x16xf32> to vector<32xf32>
    %flat_mask = vector.shape_cast %mask : vector<2x16xi1> to vector<32xi1>
    vector.maskedstore %dst[%c0], %flat_mask, %flat
        : memref<32xf32, #hivm.address_space<ub>>, vector<32xi1>, vector<32xf32>
    return
  }
}

// -----

// An i8 element also reaches the intrinsic layer cleanly, so the accepted
// range is not accidentally narrowed to the 16/32-bit `vsel` widths.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @i8_blockarg_mask_lowers
  // CHECK-NOT: ave.hir.
  // CHECK: hivm_regbaseintrins.intr.hivm.
  func.func @i8_blockarg_mask_lowers(
      %src: memref<2x128xi8, #hivm.address_space<ub>>,
      %dst: memref<256xi8, #hivm.address_space<ub>>,
      %mask: vector<2x128xi1>,
      %padding: i8) attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %v = vector.transfer_read %src[%c0, %c0], %padding, %mask
        {in_bounds = [true, true],
         permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
        : memref<2x128xi8, #hivm.address_space<ub>>, vector<2x128xi8>
    %flat = vector.shape_cast %v : vector<2x128xi8> to vector<256xi8>
    %flat_mask = vector.shape_cast %mask : vector<2x128xi1> to vector<256xi1>
    vector.maskedstore %dst[%c0], %flat_mask, %flat
        : memref<256xi8, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
    return
  }
}
