// RUN: bishengir-opt %s -convert-arith-to-hivmave -split-input-file -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// CHECK-LABEL: @triton_cast_1d_outlined_vf_0
func.func @triton_cast_1d_outlined_vf_0(%arg0: memref<13xi16, #hivm.address_space<ub>>, %arg1: memref<13xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c13 = arith.constant 13 : index
  %res, %new_true_shape = ave.hir.plt %c13 {mask_op_idx = 0 : i32} : vector<64xi1>, index
  annotation.mark %res {mask_op_idx = 0 : i32} : vector<64xi1>
  %0 = ave.hir.vload <NORM> %arg0[%c0] : memref<13xi16, #hivm.address_space<ub>> into vector<64xi16>
  annotation.mark %0 {reached_mask_ops_idx = 0 : i32} : vector<64xi16>
  // CHECK: %[[EXT1:.*]] = ave.hir.vextui %{{.*}}, %{{.*}} {part = #ave.vcvt_part_type<part_even>} : vector<64xi16>, vector<64xi32>, vector<64xi1>
  // CHECK: %[[PG:.*]] = ave.hir.pge <ALL> : vector<64xi1>
  %1 = arith.uitofp %0 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<64xi16> to vector<64xf32>
  annotation.mark %1 {reached_mask_ops_idx = 0 : i32} : vector<64xf32>
  ave.hir.masked_store <NORM_B32> %arg1[%c0], %res, %1 : memref<13xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
  return
}

// -----

// CHECK-LABEL: @triton_trans_3d_outlined_vf_0
func.func @triton_trans_3d_outlined_vf_0(%arg0: memref<3x2x257xi16, strided<[544, 272, 1]>, #hivm.address_space<ub>>, %arg1: memref<2x3x257xi16, strided<[816, 272, 1]>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  // CHECK: %[[C0_I16:.*]] = arith.constant 0 : i16
  // CHECK: %[[IDX_VEC:.*]] = ave.hir.vci %[[C0_I16]], <INCREASE> : i16, vector<128xi16>
  %cst = arith.constant dense<"0x00000100020003000400050006000700080009000A000B000C000D000E000F0010001100120013001400150016001700180019001A001B001C001D001E001F0020002100220023002400250026002700280029002A002B002C002D002E002F0030003100320033003400350036003700380039003A003B003C003D003E003F0040004100420043004400450046004700480049004A004B004C004D004E004F0050005100520053005400550056005700580059005A005B005C005D005E005F0060006100620063006400650066006700680069006A006B006C006D006E006F0070007100720073007400750076007700780079007A007B007C007D007E007F00"> : vector<128xi16>
  %c256 = arith.constant 256 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c128 = arith.constant 128 : index
  scf.for %arg2 = %c0 to %c2 step %c1 {
    scf.for %arg3 = %c0 to %c3 step %c1 {
      scf.for %arg4 = %c0 to %c256 step %c128 {
        %subview_1 = memref.subview %arg0[%arg3, %arg2, %arg4] [1, 1, 128] [1, 1, 1] : memref<3x2x257xi16, strided<[544, 272, 1]>, #hivm.address_space<ub>> to memref<1x1x128xi16, strided<[544, 272, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_2 = memref.subview %arg1[%arg2, %arg3, %arg4] [1, 1, 128] [1, 1, 1] : memref<2x3x257xi16, strided<[816, 272, 1]>, #hivm.address_space<ub>> to memref<1x1x128xi16, strided<[816, 272, 1], offset: ?>, #hivm.address_space<ub>>
        %3 = ave.hir.pge <ALL> : vector<128xi1>
        %4 = ave.hir.vgather %subview_1[%c0, %c0, %c0] [%cst], %3 : memref<1x1x128xi16, strided<[544, 272, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi16>, vector<128xi1> into vector<128xi16>
        %subview_3 = memref.subview %subview_2[0, 0, 0] [1, 1, 128] [1, 1, 1] : memref<1x1x128xi16, strided<[816, 272, 1], offset: ?>, #hivm.address_space<ub>> to memref<128xi16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
        %5 = ave.hir.pge <ALL> : vector<128xi1>
        ave.hir.masked_store <NORM_B16> %subview_3[%c0], %5, %4 : memref<128xi16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi16>
      }
      %subview = memref.subview %arg0[%arg3, %arg2, 256] [1, 1, 1] [1, 1, 1] : memref<3x2x257xi16, strided<[544, 272, 1]>, #hivm.address_space<ub>> to memref<1x1x1xi16, strided<[544, 272, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[%arg2, %arg3, 256] [1, 1, 1] [1, 1, 1] : memref<2x3x257xi16, strided<[816, 272, 1]>, #hivm.address_space<ub>> to memref<1x1x1xi16, strided<[816, 272, 1], offset: ?>, #hivm.address_space<ub>>
      %0 = ave.hir.pge <VL1> : vector<128xi1>
      %1 = builtin.unrealized_conversion_cast %0 : vector<128xi1> to vector<1x1x128xi1>
      annotation.mark %1 {mask_op_idx = -1 : i32} : vector<1x1x128xi1>
      %2 = ave.hir.vload <BRC_B16> %subview[%c0, %c0, %c0] : memref<1x1x1xi16, strided<[544, 272, 1], offset: ?>, #hivm.address_space<ub>> into vector<128xi16>
      ave.hir.masked_store <ONEPT_B16> %subview_0[%c0, %c0, %c0], %0, %2 : memref<1x1x1xi16, strided<[816, 272, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi16>
    }
  }
  return
}


// -----

// CHECK-LABEL: @triton_mod_4d_outlined_vf_0
func.func @triton_mod_4d_outlined_vf_0(%arg0: memref<32xi64, #hivm.address_space<ub>>, %arg1: memref<32xi64, #hivm.address_space<ub>>, %arg2: memref<32xi64, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  // %[[PGE:.*]] = ave.hir.pge <VL32> : vector<64xi1>
  // %[[VMOD_VEC:.*]] = ave.hir.vmod %1, %2, %[[PGE:.*]] : vector<64xi64>, vector<64xi1>
  %c0 = arith.constant 0 : index
  %0 = ave.hir.pge <VL32> {mask_op_idx = 0 : i32} : vector<64xi1>
  annotation.mark %0 {mask_op_idx = 0 : i32} : vector<64xi1>
  %1 = ave.hir.vload <NORM> %arg0[%c0] : memref<32xi64, #hivm.address_space<ub>> into vector<64xi64>
  annotation.mark %1 {reached_mask_ops_idx = 0 : i32} : vector<64xi64>
  %2 = ave.hir.vload <NORM> %arg1[%c0] : memref<32xi64, #hivm.address_space<ub>> into vector<64xi64>
  annotation.mark %2 {reached_mask_ops_idx = 0 : i32} : vector<64xi64>
  %3 = arith.remsi %1, %2 : vector<64xi64>
  annotation.mark %3 {reached_mask_ops_idx = 0 : i32} : vector<64xi64>
  ave.hir.masked_store <NORM_B64> %arg2[%c0], %0, %3 : memref<32xi64, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi64>
  return
}

// -----

// CHECK-LABEL: @triton_indirect_store_1D_contiguous_kernel_outlined_vf_0
func.func @triton_indirect_store_1D_contiguous_kernel_outlined_vf_0(%arg0: memref<5xi8, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  // CHECK: %[[CONSTANT5:.*]] = arith.constant 5 : index
  // CHECK: %[[RES:.*]], %[[NEWSHAPE:.*]] = ave.hir.plt %[[CONSTANT5]] : vector<64xi1>, index
  // CHECK: %[[PGE:.*]] = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: %[[CONSTANT1:.*]] = arith.constant 1 : i8
  // CHECK: %[[BRC1:.*]] = ave.hir.broadcast %[[CONSTANT1]], %[[PGE]] : i8, vector<64xi1> -> vector<64xi8>
  // CHECK: %[[CONSTANT0:.*]] = arith.constant 0 : i8
  // CHECK: %[[BRC2:.*]] = ave.hir.broadcast %[[CONSTANT0]], %[[PGE]] : i8, vector<64xi1> -> vector<64xi8>
  // CHECK: %[[VSEL:.*]] = ave.hir.vsel %[[RES]], %[[BRC1]], %[[BRC2]] : vector<64xi1>, vector<64xi8>
  %cst = arith.constant dense<[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : vector<64xi8>
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  %res, %new_true_shape = ave.hir.plt %c5 {mask_op_idx = 0 : i32} : vector<64xi1>, index
  annotation.mark %res {mask_op_idx = 0 : i32} : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %arg0[%c0], %res, %cst : memref<5xi8, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi8>
  return
}

// -----

// CHECK-LABEL: @handwrite_case
func.func @handwrite_case(%arg0: memref<15xi8, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  // CHECK: arith.constant 7 : index
  // CHECK: ave.hir.plt
  // CHECK: ave.hir.pge <ALL>
  // CHECK: arith.constant 5 : i32
  // CHECK: ave.hir.broadcast
  // CHECK: arith.constant 6 : i32
  // CHECK: ave.hir.vsel
  %cst = arith.constant dense<[5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]> : vector<64xi32>
  %c0 = arith.constant 0 : index
  %c15 = arith.constant 15 : index
  %res, %new_true_shape = ave.hir.plt %c15 {mask_op_idx = 0 : i32} : vector<64xi1>, index
  annotation.mark %res {mask_op_idx = 0 : i32} : vector<64xi1>
  ave.hir.masked_store <NORM_B8> %arg0[%c0], %res, %cst : memref<15xi8, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>
  return
}


// -----

// CHECK-LABEL: @triton_ge_1D_outlined_vf_0
// CHECK: ave.hir.vcmp <ULE> {{.*}} {{.*}} {{.*}} : vector<256xi8>, vector<256xi1> -> vector<256xi1>
// CHECK: ave.hir.vcmp <UGE> {{.*}} {{.*}} {{.*}} : vector<256xi8>, vector<256xi1> -> vector<256xi1>
// CHECK: ave.hir.vcmp <UGT> {{.*}} {{.*}} {{.*}} : vector<256xi8>, vector<256xi1> -> vector<256xi1>
// CHECK: ave.hir.vcmp <ULT> {{.*}} {{.*}} {{.*}} : vector<256xi8>, vector<256xi1> -> vector<256xi1>
func.func @triton_ge_1D_outlined_vf_0(%arg0: memref<5xi8>, %arg1: memref<5xi8>) -> (vector<256xi1>, vector<256xi1>, vector<256xi1>, vector<256xi1>) {
  %c0 = arith.constant 0 : index
  %0 = ave.hir.vload <NORM> %arg0[%c0] : memref<5xi8> into vector<256xi8>
  annotation.mark %0 {reached_mask_ops_idx = 0 : i32} : vector<256xi8>
  %1 = ave.hir.vload <NORM> %arg1[%c0] : memref<5xi8> into vector<256xi8>
  annotation.mark %1 {reached_mask_ops_idx = 0 : i32} : vector<256xi8>
  %2 = arith.cmpi ule, %0, %1 : vector<256xi8>
  %3 = arith.cmpi uge, %0, %1 : vector<256xi8>
  %4 = arith.cmpi ugt, %0, %1 : vector<256xi8>
  %5 = arith.cmpi ult, %0, %1 : vector<256xi8>
  return %2, %3, %4, %5 : vector<256xi1>, vector<256xi1>, vector<256xi1>, vector<256xi1>
}

// -----

// CHECK-LABEL: @triton_unk_fused__npu_dtype_cast__unsafe_index_0_outlined_vf_8
func.func @triton_unk_fused__npu_dtype_cast__unsafe_index_0_outlined_vf_8(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>) -> tensor<32xf32> attributes {hivm.vector_function} {
  // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[VCI:.*]] = ave.hir.vci %[[CST]], <INCREASE> : f32, vector<64xf32>
  // CHECK: %[[PGE:.*]] = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: %[[K:.*]] = arith.constant 1.562500e+00 : f32
  // CHECK: %[[KX:.*]] = ave.hir.vmuls %[[VCI]], %[[K]], %[[PGE]] : vector<64xf32>, f32, vector<64xi1>
  // CHECK: %[[B:.*]] = arith.constant 2.812500e-01 : f32
  // CHECK: %[[RES:.*]] = ave.hir.vadds %[[KX]], %[[B]], %[[PGE]] : vector<64xf32>, f32, vector<64xi1>
  %cst = arith.constant dense<[2.812500e-01, 1.843750e+00, 3.406250e+00, 4.968750e+00, 6.531250e+00, 8.093750e+00, 9.656250e+00, 11.21875, 12.78125, 14.34375, 15.90625, 17.46875, 19.03125, 20.59375, 22.15625, 23.71875, 25.28125, 26.84375, 28.40625, 29.96875, 31.53125, 33.09375, 34.65625, 36.21875, 37.78125, 39.34375, 40.90625, 42.46875, 44.03125, 45.59375, 47.15625, 48.71875, 50.28125, 51.84375, 53.40625, 54.96875, 56.53125, 58.09375, 59.65625, 61.21875, 62.78125, 64.34375, 65.90625, 67.46875, 69.03125, 70.59375, 72.15625, 73.71875, 75.28125, 76.84375, 78.40625, 79.96875, 81.53125, 83.09375, 84.65625, 86.21875, 87.78125, 89.34375, 90.90625, 92.46875, 94.03125, 95.59375, 97.15625, 98.71875]> : vector<64xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [32] : vector<64xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst_0, %0 {in_bounds = [true]} : tensor<32xf32>, vector<64xf32>
  %2 = arith.maximumf %1, %cst : vector<64xf32>
  %3 = vector.transfer_write %2, %arg1[%c0], %0 {in_bounds = [true]} : vector<64xf32>, tensor<32xf32>
  return %3 : tensor<32xf32>
}

// -----

// CHECK-LABEL: @fn_npu_021_outlined_vf_0
func.func @fn_npu_021_outlined_vf_0(%arg0: memref<256x256xi8, #hivm.address_space<ub>>, %arg1: memref<256x256xi8, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0_i8 = arith.constant 0 : i8
  // CHECK: arith.constant 0 : i16
  // CHECK: ave.hir.vci
  // CHECK: ave.hir.pge <ALL> : vector<128xi1>
  // CHECK: arith.constant 512 : i16
  // CHECK: ave.hir.vmuls
  // CHECK: arith.constant 256 : i16
  // CHECK: ave.hir.vadds
  // CHECK: arith.constant 0 : i16
  // CHECK: ave.hir.vci
  // CHECK: ave.hir.pge <ALL> : vector<128xi1>
  // CHECK: arith.constant 512 : i16
  // CHECK: ave.hir.vmuls
  %cst = arith.constant dense<"0x00010003000500070009000B000D000F00110013001500170019001B001D001F00210023002500270029002B002D002F00310033003500370039003B003D003F00410043004500470049004B004D004F00510053005500570059005B005D005F00610063006500670069006B006D006F00710073007500770079007B007D007F00810083008500870089008B008D008F00910093009500970099009B009D009F00A100A300A500A700A900AB00AD00AF00B100B300B500B700B900BB00BD00BF00C100C300C500C700C900CB00CD00CF00D100D300D500D700D900DB00DD00DF00E100E300E500E700E900EB00ED00EF00F100F300F500F700F900FB00FD00FF"> : vector<128xi16>
  %cst_0 = arith.constant dense<"0x00000002000400060008000A000C000E00100012001400160018001A001C001E00200022002400260028002A002C002E00300032003400360038003A003C003E00400042004400460048004A004C004E00500052005400560058005A005C005E00600062006400660068006A006C006E00700072007400760078007A007C007E00800082008400860088008A008C008E00900092009400960098009A009C009E00A000A200A400A600A800AA00AC00AE00B000B200B400B600B800BA00BC00BE00C000C200C400C600C800CA00CC00CE00D000D200D400D600D800DA00DC00DE00E000E200E400E600E800EA00EC00EE00F000F200F400F600F800FA00FC00FE"> : vector<128xi16>
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c256 step %c1 {
    %subview = memref.subview %arg0[0, %arg2] [256, 1] [1, 1] : memref<256x256xi8, #hivm.address_space<ub>> to memref<256x1xi8, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg1[%arg2, 0] [1, 256] [1, 1] : memref<256x256xi8, #hivm.address_space<ub>> to memref<1x256xi8, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>
    %0 = ave.hir.pge <ALL> : vector<256xi1>
    %1 = ave.hir.vgather %subview[%c0, %c0] [%cst_0], %0 : memref<256x1xi8, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1> into vector<256xi8>
    %2 = ave.hir.vgather %subview[%c0, %c0] [%cst], %0 : memref<256x1xi8, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi16>, vector<256xi1> into vector<256xi8>
    %3 = ave.hir.scalar_broadcast %c0_i8 : i8 -> vector<256xi8>
    %res1, %res2 = ave.hir.vdintlv %1, %3 {layout_change = #ave<layout_change DENSE>}: vector<256xi8>, vector<256xi8>
    %res1_2, %res2_3 = ave.hir.vdintlv %2, %3 {layout_change = #ave<layout_change DENSE>}: vector<256xi8>, vector<256xi8>
    %res1_4, %res2_5 = ave.hir.vintlv %res1, %res1_2 : vector<256xi8>, vector<256xi8>
    %subview_6 = memref.subview %subview_1[0, 0] [1, 256] [1, 1] : memref<1x256xi8, strided<[256, 1], offset: ?>, #hivm.address_space<ub>> to memref<256xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %4 = ave.hir.pge <ALL> : vector<256xi1>
    ave.hir.masked_store <NORM_B8> %subview_6[%c0], %4, %res1_4 : memref<256xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  }
  return
}

// -----

// CHECK-LABEL: @bitcast
func.func @bitcast(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64xi32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %0 = ave.hir.vload <NORM> %arg0[%c0]: memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
  // CHECK: vector.bitcast
  %1 = arith.bitcast %0: vector<64xf32> to vector<64xi32>
  %2 = ave.hir.pge <ALL> : vector<256xi1>
  ave.hir.masked_store <NORM_B32> %arg1[%c0], %2, %1 : memref<64xi32, #hivm.address_space<ub>>, vector<256xi1>, vector<64xi32>
  return
}

// -----

// CHECK-LABEL: @divsi_int64
func.func @divsi_int64(%arg0: memref<64xi64, #hivm.address_space<ub>>, %arg1: memref<64xi64, #hivm.address_space<ub>>, %arg2: memref<64xi64, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %0 = ave.hir.vload <NORM> %arg0[%c0] : memref<64xi64, #hivm.address_space<ub>> into vector<64xi64>
  %1 = ave.hir.vload <NORM> %arg1[%c0] : memref<64xi64, #hivm.address_space<ub>> into vector<64xi64>
  // CHECK: ave.hir.vdiv
  %2 = arith.divsi %1, %0 : vector<64xi64>
  %3 = ave.hir.pge <ALL> : vector<64xi1>
  ave.hir.masked_store <NORM_B64> %arg2[%c0], %3, %2 : memref<64xi64, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi64>
  return
}

// -----

// CHECK-LABEL: @divui_uint64
func.func @divui_uint64(%arg0: memref<64xi64, #hivm.address_space<ub>>, %arg1: memref<64xi64, #hivm.address_space<ub>>, %arg2: memref<64xi64, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %0 = ave.hir.vload <NORM> %arg0[%c0] : memref<64xi64, #hivm.address_space<ub>> into vector<64xi64>
  %1 = ave.hir.vload <NORM> %arg1[%c0] : memref<64xi64, #hivm.address_space<ub>> into vector<64xi64>
  // CHECK: ave.hir.vdiv {{.*}}{cast = #hivm.cast<cast_unsigned>}
  %2 = arith.divui %1, %0 : vector<64xi64>
  %3 = ave.hir.pge <ALL> : vector<64xi1>
  ave.hir.masked_store <NORM_B64> %arg2[%c0], %3, %2 : memref<64xi64, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi64>
  return
}

// -----
// CHECK-LABEL:   func.func @triton_load_mask_outlined_vf_0(
func.func @triton_load_mask_outlined_vf_0(%arg0: memref<32xi8, #hivm.address_space<ub>>, %arg1: memref<32xf8E5M2, #hivm.address_space<ub>>, %arg2: memref<32xf8E5M2, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f8E5M2
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i8
// CHECK:           %[[VAL_5:.*]] = ave.hir.pge <ALL> : vector<64xi1>
// CHECK:           %[[VAL_6:.*]] = ave.hir.broadcast %[[VAL_4:.*]], %[[VAL_5:.*]] : i8, vector<64xi1> -> vector<64xi8>
// CHECK:           %[[VAL_7:.*]] = vector.bitcast %[[VAL_6:.*]] : vector<64xi8> to vector<64xf8E5M2>
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_9:.*]] = ave.hir.pge <ALL> : vector<64xi1>
// CHECK:           %[[VAL_10:.*]] = ave.hir.broadcast %[[VAL_8:.*]], %[[VAL_9:.*]] : i32, vector<64xi1> -> vector<64xi32>
// CHECK:           %[[VAL_11:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_12:.*]] = ave.hir.vload <NORM> %[[VAL_0:.*]]{{\[}}%[[VAL_11:.*]]] : memref<32xi8, #hivm.address_space<ub>> into vector<64xi8>
// CHECK:           %[[VAL_13:.*]] = ave.hir.vload <NORM> %[[VAL_1:.*]]{{\[}}%[[VAL_11:.*]]] : memref<32xf8E5M2, #hivm.address_space<ub>> into vector<64xf8E5M2>
// CHECK:           %[[VAL_14:.*]] = ave.hir.pge <ALL> : vector<64xi1>
  %cst = arith.constant dense<0.000000e+00> : vector<64xf8E5M2>
  %cst_0 = arith.constant dense<0> : vector<64xi32>
  %c0 = arith.constant 0 : index
  %res = ave.hir.vload <NORM> %arg0[%c0] : memref<32xi8, #hivm.address_space<ub>> into vector<64xi8>
  %res_1 = ave.hir.vload <NORM> %arg1[%c0] : memref<32xf8E5M2, #hivm.address_space<ub>> into vector<64xf8E5M2>
  %1 = arith.extsi %res {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<64xi8> to vector<64xi32>
  // CHECK:           %[[VAL_16:.*]] = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK:           %[[VAL_17:.*]] = ave.hir.vcmp <NE> %{{.*}}, %[[VAL_10:.*]], %[[VAL_16:.*]] : vector<64xi32>, vector<64xi1> -> vector<64xi1>
  // CHECK:           %[[VAL_18:.*]] = vector.bitcast %[[VAL_13:.*]] : vector<64xf8E5M2> to vector<64xi8>
  // CHECK:           %[[VAL_19:.*]] = vector.bitcast %[[VAL_7:.*]] : vector<64xf8E5M2> to vector<64xi8>
  // CHECK:           %[[VAL_20:.*]] = ave.hir.vsel %[[VAL_17:.*]], %[[VAL_18:.*]], %[[VAL_19:.*]] : vector<64xi1>, vector<64xi8>
  // CHECK:           %[[VAL_21:.*]] = vector.bitcast %[[VAL_20:.*]] : vector<64xi8> to vector<64xf8E5M2>
  %2 = arith.cmpi ne, %1, %cst_0 : vector<64xi32>
  %3 = arith.select %2, %res_1, %cst : vector<64xi1>, vector<64xf8E5M2>
  return
}

// CHECK-LABEL: @arith_two_dim
func.func @arith_two_dim(%arg0: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<4x16xf32>
// CHECK: %[[cst:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[pge:.*]] = ave.hir.pge <ALL> : vector<64xi1>
// CHECK-NEXT: %[[brc:.*]] = ave.hir.broadcast %[[cst]], %[[pge]] : f32, vector<64xi1> -> vector<64xf32>
// CHECK-NEXT: %[[ucc:.*]] = builtin.unrealized_conversion_cast %[[brc]] : vector<64xf32> to vector<4x16xf32>
  %1 = ave.hir.pge <ALL> : vector<64xi1>
  %2 = builtin.unrealized_conversion_cast %cst : vector<4x16xf32> to vector<64xf32>
  ave.hir.masked_store <NORM_B32> %arg0[%c0], %1, %2 : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
  return
}

// -----

// CHECK-LABEL: func.func @test_unsigned_i1_cmp
// CHECK: ave.hir.preg.not
// CHECK: ave.hir.preg.or
func.func @test_unsigned_i1_cmp() -> vector<256xi1> {
  %lhs = vector.constant_mask [128] : vector<256xi1>
  %rhs = vector.constant_mask [64] : vector<256xi1>

  // i1 unsigned comparison, should be converted to logical gate combination
  %cmp = arith.cmpi uge, %lhs, %rhs : vector<256xi1>

  return %cmp : vector<256xi1>
}

// CHECK-LABEL: @test_mulsi_extended
// CHECK: ave.hir.mull
func.func @test_mulsi_extended(%arg0: vector<64xi64>, %arg1: vector<64xi64>) -> (vector<64xi64>, vector<64xi64>){
  %low, %high = arith.mulsi_extended  %arg0, %arg1 : vector<64xi64>
  return %low, %high : vector<64xi64>, vector<64xi64>
}

// -----

// CHECK-LABEL: @test_mului_extended
// CHECK: ave.hir.mull
func.func @test_mului_extended(%arg0: vector<64xi64>, %arg1: vector<64xi64>) -> (vector<64xi64>, vector<64xi64>){
  %low, %high = arith.mului_extended  %arg0, %arg1 : vector<64xi64>
  return %low, %high : vector<64xi64>, vector<64xi64>
}

// -----

// CHECK-LABEL: @lower_fma
// CHECK: ave.hir.vmula
// CHECK-SAME: vector<8xf32>
func.func @lower_fma(%a: vector<8xf32>, %b: vector<8xf32>, %c: vector<8xf32>) -> vector<8xf32> {
  %result = math.fma %a, %b, %c : vector<8xf32>
  return %result: vector<8xf32>
}

// -----

// CHECK-LABEL: @test_create_preg_from_constant_op
// Test for createPRegFromConstantOp function with multi-dimensional vector
func.func @test_create_preg_from_constant_op(%arg0: memref<16x2x1xi1, strided<[65536, 256, 1]>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  // This constant will trigger createPRegFromConstantOp
  %cst = arith.constant dense<true> : vector<1x1x64xi1>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    scf.for %j = %c0 to %c2 step %c1 {
      %subview = memref.subview %arg0[%i, %j, 0] [1, 1, 1] [1, 1, 1] : memref<16x2x1xi1, strided<[65536, 256, 1]>, #hivm.address_space<ub>> to memref<1x1x1xi1, strided<[65536, 256, 1], offset: ?>, #hivm.address_space<ub>>
      %mask = ave.hir.pge <VL1> : vector<64xi1> 
      // CHECK: %[[PG:.*]] = ave.hir.pge <ALL> : vector<64xi1>
      // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[PG]] : vector<64xi1> to vector<1x1x64xi1>
      // CHECK: %[[CAST2:.*]] = builtin.unrealized_conversion_cast %[[CAST]] : vector<1x1x64xi1> to vector<64xi1> 
      %0 = builtin.unrealized_conversion_cast %cst : vector<1x1x64xi1> to vector<64xi1>
      ave.hir.masked_store <NORM_B8> %subview[%c0, %c0, %c0], %mask, %0 : memref<1x1x1xi1, strided<[65536, 256, 1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
    }
  }
  return
}
// -----