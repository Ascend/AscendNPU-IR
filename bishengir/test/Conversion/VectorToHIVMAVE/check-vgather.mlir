// RUN: bishengir-opt %s -convert-vector-to-hivmave -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

func.func @fn_npu_021_outlined_vf_0(%arg0: memref<2x13xi8, strided<[32, 1]>, #hivm.address_space<ub>>, %arg1: memref<13x2xi8, strided<[32, 1]>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %cst = arith.constant dense<0> : vector<256xi8>
  %cst_0 = arith.constant dense<0> : vector<128xi16>
  %c1 = arith.constant 1 : index
  %c13 = arith.constant 13 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c13 step %c1 {
    %subview = memref.subview %arg0[0, %arg2] [2, 1] [1, 1] : memref<2x13xi8, strided<[32, 1]>, #hivm.address_space<ub>> to memref<2x1xi8, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg1[%arg2, 0] [1, 2] [1, 1] : memref<13x2xi8, strided<[32, 1]>, #hivm.address_space<ub>> to memref<1x2xi8, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.constant_mask [2] : vector<256xi1>
    annotation.mark %0 {mask_op_idx = 0 : i32} : vector<256xi1>
    // CHECK: {{.*}} = ave.hir.vgather %subview[%c0, %c0] [%cst_0], %0 : memref<2x1xi8, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1> into vector<256xi8>
    // CHECK: {{.*}} = ave.hir.vgather %subview[%c0, %c0] [%cst_2], %0 : memref<2x1xi8, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1> into vector<256xi8>
    %1 = vector.gather %subview[%c0, %c0] [%cst_0], %0, %cst {secondary_index = dense<"0x20000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"> : vector<128xi16>} : memref<2x1xi8, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xi8> into vector<256xi8>
    annotation.mark %1 {reached_mask_ops_idx = 0 : i32} : vector<256xi8>
    %subview_2 = memref.subview %subview_1[0, 0] [1, 2] [1, 1] : memref<1x2xi8, strided<[32, 1], offset: ?>, #hivm.address_space<ub>> to memref<2xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    vector.transfer_write %1, %subview_2[%c0], %0 {in_bounds = [true]} : vector<256xi8>, memref<2xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
  }
  return
}

func.func @triton_permute_2d_outlined_vf_0(%arg0: memref<37x3xf8E5M2, #hivm.address_space<ub>>, %arg1: memref<3x37xf8E5M2, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %cst = arith.constant dense<0.000000e+00> : vector<256xf8E5M2>
  %cst_0 = arith.constant dense<"0x000006000C00120018001E0024002A00300036003C00420048004E0054005A00600066006C000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"> : vector<128xi16>
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c3 step %c1 {
    %subview = memref.subview %arg0[0, %arg2] [37, 1] [1, 1] : memref<37x3xf8E5M2, #hivm.address_space<ub>> to memref<37x1xf8E5M2, strided<[3, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg1[%arg2, 0] [1, 37] [1, 1] : memref<3x37xf8E5M2, #hivm.address_space<ub>> to memref<1x37xf8E5M2, strided<[37, 1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.constant_mask [37] : vector<256xi1>
    annotation.mark %0 {mask_op_idx = 0 : i32} : vector<256xi1>
    // CHECK: ave.hir.vgather %subview[%c0, %c0] [%cst_0], %res : memref<37x1xf8E5M2, strided<[3, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1> into vector<256xf8E5M2>
    // CHECK: ave.hir.vgather %subview[%c0, %c0] [%cst_2], %res : memref<37x1xf8E5M2, strided<[3, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1> into vector<256xf8E5M2> 
    %1 = vector.gather %subview[%c0, %c0] [%cst_0], %0, %cst {secondary_index = dense<"0x030009000F0015001B00210027002D00330039003F0045004B00510057005D006300690000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"> : vector<128xi16>} : memref<37x1xf8E5M2, strided<[3, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1>, vector<256xf8E5M2> into vector<256xf8E5M2>
    annotation.mark %1 {reached_mask_ops_idx = 0 : i32} : vector<256xf8E5M2>
    %subview_2 = memref.subview %subview_1[0, 0] [1, 37] [1, 1] : memref<1x37xf8E5M2, strided<[37, 1], offset: ?>, #hivm.address_space<ub>> to memref<37xf8E5M2, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %2 = vector.shape_cast %1 : vector<256xf8E5M2> to vector<256xf8E5M2>
    annotation.mark %2 {reached_mask_ops_idx = 0 : i32} : vector<256xf8E5M2>
    vector.transfer_write %2, %subview_2[%c0], %0 {in_bounds = [true]} : vector<256xf8E5M2>, memref<37xf8E5M2, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
  }
  return
}