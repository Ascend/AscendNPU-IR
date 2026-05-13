// RUN: bishengir-opt %s -convert-vector-to-hivmave -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

func.func @triton_broadcast_to_3d_outlined_vf_0(%arg0: memref<7xi64, #hivm.address_space<ub>>, %arg1: memref<7x3xi64, #hivm.address_space<ub>>, %arg2: memref<7x8xi64, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0_i64 = arith.constant 0 : i64
  %c7 = arith.constant 7 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  scf.for %arg3 = %c0 to %c7 step %c1 {
    %subview = memref.subview %arg0[%arg3] [1] [1] : memref<7xi64, #hivm.address_space<ub>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg3, 0] [1, 3] [1, 1] : memref<7x3xi64, #hivm.address_space<ub>> to memref<1x3xi64, strided<[3, 1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.constant_mask [3] : vector<64xi1>
    %subview_1 = memref.subview %subview_0[0, 0] [1, 3] [1, 1] : memref<1x3xi64, strided<[3, 1], offset: ?>, #hivm.address_space<ub>> to memref<3xi64, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %1 = vector.transfer_read %subview_1[%c0], %c0_i64, %0 {in_bounds = [true]} : memref<3xi64, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi64>
    %subview_2 = memref.subview %subview[0] [1] [1] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<i64, strided<[], offset: ?>, #hivm.address_space<ub>>
    %2 = vector.transfer_read %subview_2[], %c0_i64 : memref<i64, strided<[], offset: ?>, #hivm.address_space<ub>>, vector<i64>
    %3 = vector.shape_cast %2 : vector<i64> to vector<1xi64>
    %4 = math.absi %1 : vector<64xi64>
    %5 = builtin.unrealized_conversion_cast %3 : vector<1xi64> to i64
    %6 = vector.mask %0 { vector.reduction <add>, %4, %5 : vector<64xi64> into i64 } : vector<64xi1> -> i64
    %7 = builtin.unrealized_conversion_cast %6 : i64 to vector<1xi64>
    %subview_3 = memref.subview %arg2[%arg3, 0] [1, 8] [1, 1] : memref<7x8xi64, #hivm.address_space<ub>> to memref<1x8xi64, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    // CHECK: ave.hir.vector_broadcast %[[VAL_1:.*]], %[[VAL_2:.*]], true : vector<64xi64>, vector<64xi1> -> vector<64xi64>
    %8 = vector.broadcast %7 : vector<1xi64> to vector<1x64xi64>
    %9 = vector.constant_mask [8] : vector<64xi1>
    annotation.mark %9 {mask_op_idx = 1 : i32} : vector<64xi1>
    %subview_4 = memref.subview %subview_3[0, 0] [1, 8] [1, 1] : memref<1x8xi64, strided<[8, 1], offset: ?>, #hivm.address_space<ub>> to memref<8xi64, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %10 = vector.shape_cast %8 : vector<1x64xi64> to vector<64xi64>
    annotation.mark %10 {reached_mask_ops_idx = 1 : i32} : vector<64xi64>
    vector.transfer_write %10, %subview_4[%c0], %9 {in_bounds = [true]} : vector<64xi64>, memref<8xi64, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
  }
  return
}