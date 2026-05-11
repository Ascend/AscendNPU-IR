// RUN: bishengir-opt %s -convert-vector-to-hivmave -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

func.func @fn_npu_2d_outlined_vf_0(%arg0: i32, %arg1: memref<16x17xi32, strided<[24, 1]>, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c16 step %c1 {
    %subview = memref.subview %arg1[%arg2, 0] [1, 17] [1, 1] : memref<16x17xi32, strided<[24, 1]>, #hivm.address_space<ub>> to memref<1x17xi32, strided<[24, 1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.constant_mask [17] : vector<64xi1>
    annotation.mark %0 {mask_op_idx = 0 : i32} : vector<64xi1>
    %subview_0 = memref.subview %subview[0, 0] [1, 17] [1, 1] : memref<1x17xi32, strided<[24, 1], offset: ?>, #hivm.address_space<ub>> to memref<17xi32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    // CHECK: %0 = ave.hir.scalar_broadcast %arg0 : i32 -> vector<64xi32>
    %1 = vector.broadcast %arg0 : i32 to vector<64xi32>
    annotation.mark %1 {reached_mask_ops_idx = 0 : i32} : vector<64xi32>
    vector.transfer_write %1, %subview_0[%c0], %0 {in_bounds = [true]} : vector<64xi32>, memref<17xi32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
  }
  return
}