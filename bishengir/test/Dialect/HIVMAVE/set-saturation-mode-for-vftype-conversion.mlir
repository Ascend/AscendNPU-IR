// RUN: bishengir-opt -convert-arith-to-hivmave %s | FileCheck %s

func.func @triton_mod_outlined_vf_0(%arg0: memref<1x1x23xf32, strided<[24, 24, 1]>, #hivm.address_space<ub>>, %arg1: memref<1x1x23xf32, strided<[24, 24, 1]>, #hivm.address_space<ub>>, %arg2: memref<1x1x23xi16, strided<[32, 32, 1]>, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c23 = arith.constant 23 : index
  %res, %new_true_shape = ave.hir.plt %c23 {mask_op_idx = 0 : i32} : vector<64xi1>, index
  annotation.mark %res {mask_op_idx = 0 : i32} : vector<64xi1>
  %subview = memref.subview %arg0[0, 0, 0] [1, 1, 23] [1, 1, 1] : memref<1x1x23xf32, strided<[24, 24, 1]>, #hivm.address_space<ub>> to memref<23xf32, strided<[1]>, #hivm.address_space<ub>>
  %0 = ave.hir.vload <NORM> %subview[%c0] : memref<23xf32, strided<[1]>, #hivm.address_space<ub>> into vector<64xf32>
  annotation.mark %0 {reached_mask_ops_idx = 0 : i32} : vector<64xf32>
  %subview_4 = memref.subview %arg1[0, 0, 0] [1, 1, 23] [1, 1, 1] : memref<1x1x23xf32, strided<[24, 24, 1]>, #hivm.address_space<ub>> to memref<23xf32, strided<[1]>, #hivm.address_space<ub>>
  %1 = ave.hir.vload <NORM> %subview_4[%c0] : memref<23xf32, strided<[1]>, #hivm.address_space<ub>> into vector<64xf32>
  annotation.mark %1 {reached_mask_ops_idx = 0 : i32} : vector<64xf32>
  %2 = arith.divf %0, %1 : vector<64xf32>
  annotation.mark %2 {reached_mask_ops_idx = 0 : i32} : vector<64xf32>
  %3 = math.round %2 {round_mode = #hfusion.round_mode<floor>} : vector<64xf32>
  annotation.mark %3 {reached_mask_ops_idx = 0 : i32} : vector<64xf32>
  %4 = arith.mulf %3, %1 : vector<64xf32>
  annotation.mark %4 {reached_mask_ops_idx = 0 : i32} : vector<64xf32>
  %5 = arith.subf %0, %4 : vector<64xf32>
  annotation.mark %5 {reached_mask_ops_idx = 0 : i32} : vector<64xf32>
  %6 = arith.fptosi %5 {round_mode = #hfusion.round_mode<truncwithoverflow>} : vector<64xf32> to vector<64xi16>
  annotation.mark %6 {reached_mask_ops_idx = 0 : i32} : vector<64xi16>
  %subview_5 = memref.subview %arg2[0, 0, 0] [1, 1, 23] [1, 1, 1] : memref<1x1x23xi16, strided<[32, 32, 1]>, #hivm.address_space<ub>> to memref<23xi16, strided<[1]>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B16> %subview_5[%c0], %res, %6 : memref<23xi16, strided<[1]>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi16>
  return
}

// CHECK-LABEL: func.func @triton_mod_outlined_vf_0
// CHECK: ave.hir.vfptosi %{{.*}}, <truncwithoverflow>, %res {part = #ave.vcvt_part_type<part_even>, sat = false}
// CHECK-NOT: ave.hir.vfptosi %{{.*}}, <truncwithoverflow>, %res {part = #ave.vcvt_part_type<part_even>, sat = true}
