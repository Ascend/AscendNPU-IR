// RUN: bishengir-opt %s -convert-vector-to-hivmave -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// CHECK-LABEL: func.func @triton_sum_3D_dim0_outlined_vf_1
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @triton_sum_3D_dim0_outlined_vf_1(%arg0: memref<3x514xi32, #hivm.address_space<ub>>, %arg1: memref<514xi32, #hivm.address_space<ub>>, %arg2: memref<256xi8, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant dense<0> : vector<64xi32>
    %c0_i32 = arith.constant 0 : i32
    %c64 = arith.constant 64 : index
    %c514 = arith.constant 514 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    scf.for %arg3 = %c0 to %c514 step %c64 {
      %0 = affine.min affine_map<(d0) -> (-d0 + 514, 64)>(%arg3)
      %subview = memref.subview %arg1[%arg3] [%0] [1] : memref<514xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %view = memref.view %arg2[%c0][%0] : memref<256xi8, #hivm.address_space<ub>> to memref<1x?xi32, #hivm.address_space<ub>>
      %1 = vector.create_mask %0 : vector<64xi1>
      annotation.mark %1 {mask_op_idx = 0 : i32} : vector<64xi1>
      %subview_0 = memref.subview %view[0, 0] [1, %0] [1, 1] : memref<1x?xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
      vector.transfer_write %cst, %subview_0[%c0], %1 {in_bounds = [true]} : vector<64xi32>, memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %view[0, 0] [1, %0] [1, 1] : memref<1x?xi32, #hivm.address_space<ub>> to memref<1x?xi32, strided<[?, 1]>, #hivm.address_space<ub>>
      %subview_2 = memref.subview %subview_1[0, 0] [1, %0] [1, 1] : memref<1x?xi32, strided<[?, 1]>, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
      %2 = vector.transfer_read %subview_2[%c0], %c0_i32, %1 {in_bounds = [true]} : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>, vector<64xi32>
      annotation.mark %2 {reached_mask_ops_idx = 0 : i32} : vector<64xi32>
      %3 = vector.shape_cast %2 : vector<64xi32> to vector<1x64xi32>
      annotation.mark %3 {reached_mask_ops_idx = 0 : i32} : vector<1x64xi32>
      %4:2 = scf.for %arg4 = %c0 to %c3 step %c1 iter_args(%arg5 = %3, %arg6 = %2) -> (vector<1x64xi32>, vector<64xi32>) {
        %subview_3 = memref.subview %arg0[%arg4, %arg3] [1, %0] [1, 1] : memref<3x514xi32, #hivm.address_space<ub>> to memref<1x?xi32, strided<[514, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_4 = memref.subview %subview_3[0, 0] [1, %0] [1, 1] : memref<1x?xi32, strided<[514, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xi32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
        // CHECK: %[[LOAD:.*]] = ave.hir.vload <NORM> %{{.*}}[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access}
        %9 = vector.transfer_read %subview_4[%c0], %c0_i32, %1 {in_bounds = [true]} : memref<?xi32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi32>
        %10 = arith.addi %9, %arg6 : vector<64xi32>
        %11 = vector.shape_cast %10 : vector<64xi32> to vector<1x64xi32>
        scf.yield %11, %10 : vector<1x64xi32>, vector<64xi32>
      }
      annotation.mark %4#1 {reached_mask_ops_idx = 0 : i32} : vector<64xi32>
      vector.transfer_write %4#1, %subview_2[%c0], %1 {in_bounds = [true]} : vector<64xi32>, memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
      %5 = vector.transfer_read %subview_0[%c0], %c0_i32, %1 {in_bounds = [true]} : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>, vector<64xi32>
      annotation.mark %5 {reached_mask_ops_idx = 0 : i32} : vector<64xi32>
      %6 = vector.transfer_read %subview[%c0], %c0_i32, %1 {in_bounds = [true]} : memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi32>
      annotation.mark %6 {reached_mask_ops_idx = 0 : i32} : vector<64xi32>
      %7 = arith.addi %6, %5 : vector<64xi32>
      annotation.mark %7 {reached_mask_ops_idx = 0 : i32} : vector<64xi32>
      %8 = arith.select %1, %7, %5 : vector<64xi1>, vector<64xi32>
      annotation.mark %8 {reached_mask_ops_idx = 0 : i32} : vector<64xi32>
      vector.transfer_write %8, %subview[%c0], %1 {in_bounds = [true]} : vector<64xi32>, memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
}
