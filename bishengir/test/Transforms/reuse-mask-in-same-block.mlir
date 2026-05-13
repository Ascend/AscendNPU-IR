// RUN: bishengir-opt --convert-arith-to-hivmave %s | FileCheck %s

func.func @triton_unk_fused_cat_7_outlined_vf_1(%arg0: memref<16x408xf32, #hivm.address_space<ub>>, %arg1: i32, %arg2: memref<16x408xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
  %cst_0 = arith.constant dense<1.000000e+00> : vector<64xf32>
  %cst_1 = arith.constant dense<0> : vector<64xi64>
  %cst_2 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]> : vector<64xi32>
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c408 = arith.constant 408 : index
  scf.for %arg3 = %c0 to %c16 step %c1 {
    %0 = arith.index_cast %arg3 : index to i32
    annotation.mark %0 {reached_mask_ops_idx = 0 : i32} : i32
    %1 = ave.hir.scalar_broadcast %arg1 : i32 -> vector<64xi32>
    annotation.mark %1 {reached_mask_ops_idx = 0 : i32} : vector<64xi32>
    %2 = ave.hir.scalar_broadcast %0 : i32 -> vector<64xi32>
    annotation.mark %2 {reached_mask_ops_idx = 0 : i32} : vector<64xi32>
    // CHECK: ave.hir.pge <ALL>
    %3 = arith.addi %1, %2 : vector<64xi32>
    annotation.mark %3 {reached_mask_ops_idx = 0 : i32} : vector<64xi32>
    scf.for %arg4 = %c0 to %c408 step %c64 {
      %4 = affine.min affine_map<(d0) -> (-d0 + 408, 64)>(%arg4)
      %subview = memref.subview %arg0[%arg3, %arg4] [1, %4] [1, 1] : memref<16x408xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[408, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_3 = memref.subview %arg2[%arg3, %arg4] [1, %4] [1, 1] : memref<16x408xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[408, 1], offset: ?>, #hivm.address_space<ub>>
      %res, %new_true_shape = ave.hir.plt %4 {mask_op_idx = 0 : i32} : vector<64xi1>, index
      annotation.mark %res {mask_op_idx = 0 : i32} : vector<64xi1>
      %subview_4 = memref.subview %subview[0, 0] [1, %4] [1, 1] : memref<1x?xf32, strided<[408, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      %5 = ave.hir.vload <NORM> %subview_4[%c0] : memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
      annotation.mark %5 {reached_mask_ops_idx = 0 : i32} : vector<64xf32>
      %6 = arith.index_cast %arg4 : index to i32
      annotation.mark %6 {reached_mask_ops_idx = 0 : i32} : i32
      %7 = ave.hir.scalar_broadcast %6 : i32 -> vector<64xi32>
      annotation.mark %7 {reached_mask_ops_idx = 0 : i32} : vector<64xi32>
      %8 = arith.addi %7, %cst_2 : vector<64xi32>
      annotation.mark %8 {reached_mask_ops_idx = 0 : i32} : vector<64xi32>
      %9 = arith.subi %8, %3 : vector<64xi32>
      annotation.mark %9 {reached_mask_ops_idx = 0 : i32} : vector<64xi32>
      %10 = arith.extsi %9 {round_mode = #hfusion.round_mode<rint>} : vector<64xi32> to vector<64xi64>
      annotation.mark %10 {reached_mask_ops_idx = 0 : i32} : vector<64xi64>
      %11 = arith.cmpi sge, %10, %cst_1 : vector<64xi64>
      annotation.mark %11 {reached_mask_ops_idx = 0 : i32} : vector<64xi1>
      %12 = arith.select %11, %cst_0, %cst : vector<64xi1>, vector<64xf32>
      annotation.mark %12 {reached_mask_ops_idx = 0 : i32} : vector<64xf32>
      %13 = arith.cmpf une, %12, %cst : vector<64xf32>
      annotation.mark %13 {reached_mask_ops_idx = 0 : i32} : vector<64xi1>
      %14 = arith.select %13, %cst, %5 : vector<64xi1>, vector<64xf32>
      annotation.mark %14 {reached_mask_ops_idx = 0 : i32} : vector<64xf32>
      %subview_5 = memref.subview %subview_3[0, 0] [1, %4] [1, 1] : memref<1x?xf32, strided<[408, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      ave.hir.masked_store <NORM_B32> %subview_5[%c0], %res, %14 : memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    }
  }
  return
}

