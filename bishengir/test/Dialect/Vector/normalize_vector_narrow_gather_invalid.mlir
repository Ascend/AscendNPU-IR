// RUN: bishengir-opt %s -normalize-vector -vector-transfer-lowering -cse -canonicalize -split-input-file -verify-diagnostics

// A permuted i8 read wider than the physical 256-lane gather width is
// unsupported -- nothing above one full physical register is implemented,
// regardless of what consumes the result.
func.func @oversized_i8_transpose_still_not_gathered(%arg0: memref<512x16xi8, #hivm.address_space<ub>>, %arg1: memref<16x512xi8, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %subview = memref.subview %arg0[0, 0] [512, 1] [1, 1] : memref<512x16xi8, #hivm.address_space<ub>> to memref<512x1xi8, strided<[16, 1]>, #hivm.address_space<ub>>
  %subview_0 = memref.subview %arg1[0, 0] [1, 512] [1, 1] : memref<16x512xi8, #hivm.address_space<ub>> to memref<1x512xi8, strided<[512, 1]>, #hivm.address_space<ub>>
  // expected-error @+1 {{unsupported B8 transpose gather width 512: expected <= 256}}
  %0 = vector.transfer_read %subview[%c0, %c0], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<512x1xi8, strided<[16, 1]>, #hivm.address_space<ub>>, vector<1x512xi8>
  vector.transfer_write %0, %subview_0[%c0, %c0] {in_bounds = [true, true]} : vector<1x512xi8>, memref<1x512xi8, strided<[512, 1]>, #hivm.address_space<ub>>
  return
}
