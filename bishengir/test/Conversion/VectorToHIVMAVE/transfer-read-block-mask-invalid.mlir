// RUN: bishengir-opt %s -convert-vector-to-hivmave -verify-diagnostics

// E2B/BLK reads must not retain a mask after empty/full mask folding.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @unsupported_e2b_mask(%src: memref<4xf16, #hivm.address_space<ub>>, %mask: vector<4xi1>) -> vector<4x16xf16> {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0.0 : f16
    // expected-error@+2 {{E2B/BLK transfer_read does not support a mask}}
    // expected-error@+1 {{failed to legalize operation 'vector.transfer_read' that was explicitly marked illegal}}
    %read = vector.transfer_read %src[%c0], %pad, %mask {in_bounds = [true, true], permutation_map = affine_map<(d0) -> (d0, 0)>} : memref<4xf16, #hivm.address_space<ub>>, vector<4x16xf16>
    return %read : vector<4x16xf16>
  }
}
