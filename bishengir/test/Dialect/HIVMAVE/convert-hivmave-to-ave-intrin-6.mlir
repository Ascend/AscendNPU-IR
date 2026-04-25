 // RUN: bishengir-opt -data-layout-analyze -append-vector-layout -annotate-dist-op-layout \
 // RUN: -eliminate-vector-layout -convert-hivmave-to-ave-intrin -cse %s | FileCheck %s
 // CHECK-LABEL: @test_long_vector_len_case
func.func @test_long_vector_len_case(%arg0: memref<1x1x135xi8, strided<[160, 160, 1]>, #hivm.address_space<ub>>, %arg1: memref<1x1x135xi8, strided<[160, 160, 1]>, #hivm.address_space<ub>>, %arg2: memref<1x1x135xi8, strided<[160, 160, 1]>, #hivm.address_space<ub>>, %arg3: memref<1x1x135xi8, strided<[160, 160, 1]>, #hivm.address_space<ub>>, %arg4: memref<1x1x135xi8, strided<[160, 160, 1]>, #hivm.address_space<ub>>, %arg5: memref<1x1x135xi8, strided<[160, 160, 1]>, #hivm.address_space<ub>>, %arg6: memref<1x1x135xi8, strided<[160, 160, 1]>, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c135 = arith.constant 135 : index
  %c0_i32 = arith.constant 0 : i32
  // CHECK : llvm.mlir.constant(0 : i32) : i32
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK : "hivm_regbaseintrins.intr.hivm.pge.b32"
  %1 = ave.hir.broadcast %c0_i32, %0 : i32, vector<64xi1> -> vector<64xi32>
  // CHECK : "hivm_regbaseintrins.intr.hivm.vdups.z"
  %res, %new_true_shape = ave.hir.plt %c135 {mask_op_idx = 0 : i32} : vector<256xi1>, index
  // CHECK : "hivm_regbaseintrins.intr.hivm.plt.b8.v300"
  %subview = memref.subview %arg0[0, 0, 0] [1, 1, 135] [1, 1, 1] : memref<1x1x135xi8, strided<[160, 160, 1]>, #hivm.address_space<ub>> to memref<135xi8, strided<[1]>, #hivm.address_space<ub>>
  %res_0 = ave.hir.vload <NORM> %subview[%c0] : memref<135xi8, strided<[1]>, #hivm.address_space<ub>> into vector<256xi8>
  // CHECK : "hivm_regbaseintrins.intr.hivm.vldsx1.v256s8"
  %subview_1 = memref.subview %arg1[0, 0, 0] [1, 1, 135] [1, 1, 1] : memref<1x1x135xi8, strided<[160, 160, 1]>, #hivm.address_space<ub>> to memref<135xi8, strided<[1]>, #hivm.address_space<ub>>
  %res_2 = ave.hir.vload <NORM> %subview_1[%c0] : memref<135xi8, strided<[1]>, #hivm.address_space<ub>> into vector<256xi8>
  // CHECK : "hivm_regbaseintrins.intr.hivm.vldsx1.v256s8"
  %2 = ave.hir.vsub %res_0, %res_2, %res : vector<256xi8>, vector<256xi1>
  %subview_3 = memref.subview %arg2[0, 0, 0] [1, 1, 135] [1, 1, 1] : memref<1x1x135xi8, strided<[160, 160, 1]>, #hivm.address_space<ub>> to memref<135xi8, strided<[1]>, #hivm.address_space<ub>>
  ave.hir.masked_store <NORM_B8> %subview_3[%c0], %res, %2 : memref<135xi8, strided<[1]>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  scf.for %arg7 = %c0 to %c135 step %c64 {
    %3 = affine.min affine_map<(d0) -> (-d0 + 135, 64)>(%arg7)
    %subview_4 = memref.subview %arg3[0, 0, %arg7] [1, 1, %3] [1, 1, 1] : memref<1x1x135xi8, strided<[160, 160, 1]>, #hivm.address_space<ub>> to memref<1x1x?xi8, strided<[160, 160, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_5 = memref.subview %arg4[0, 0, %arg7] [1, 1, %3] [1, 1, 1] : memref<1x1x135xi8, strided<[160, 160, 1]>, #hivm.address_space<ub>> to memref<1x1x?xi8, strided<[160, 160, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_6 = memref.subview %arg5[0, 0, %arg7] [1, 1, %3] [1, 1, 1] : memref<1x1x135xi8, strided<[160, 160, 1]>, #hivm.address_space<ub>> to memref<1x1x?xi8, strided<[160, 160, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_7 = memref.subview %arg6[0, 0, %arg7] [1, 1, %3] [1, 1, 1] : memref<1x1x135xi8, strided<[160, 160, 1]>, #hivm.address_space<ub>> to memref<1x1x?xi8, strided<[160, 160, 1], offset: ?>, #hivm.address_space<ub>>
    %4 = ave.hir.pltm %arg7, %c135 : vector<64xi1>
    %subview_8 = memref.subview %subview_4[0, 0, 0] [1, 1, %3] [1, 1, 1] : memref<1x1x?xi8, strided<[160, 160, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res_9 = ave.hir.vload <NORM> %subview_8[%c0] : memref<?xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xi8>
    %subview_10 = memref.subview %subview_5[0, 0, 0] [1, 1, %3] [1, 1, 1] : memref<1x1x?xi8, strided<[160, 160, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res_11 = ave.hir.vload <NORM> %subview_10[%c0] : memref<?xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xi8>
    %subview_12 = memref.subview %subview_6[0, 0, 0] [1, 1, %3] [1, 1, 1] : memref<1x1x?xi8, strided<[160, 160, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res_13 = ave.hir.vload <NORM> %subview_12[%c0] : memref<?xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xi8>
    %5 = ave.hir.vextsi %res_9, %4 {pp = #ave.vcvt_pp_type<pp0>} : vector<64xi8>, vector<64xi32>, vector<64xi1>
    %6 = ave.hir.vcmp <NE> %5, %1, %4 : vector<64xi32>, vector<64xi1> -> vector<64xi1>
    %7 = ave.hir.vsel %6, %res_11, %res_13 : vector<64xi1>, vector<64xi8>
    %subview_14 = memref.subview %subview_7[0, 0, 0] [1, 1, %3] [1, 1, 1] : memref<1x1x?xi8, strided<[160, 160, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    ave.hir.masked_store <NORM_B8> %subview_14[%c0], %4, %7 {hivm.is_continuous} : memref<?xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi8>
  } {element_alignment_bit_width = 32 : i32}
  return
}