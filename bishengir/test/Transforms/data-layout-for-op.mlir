// RUN: bishengir-opt --data-layout-analyze -split-input-file %s | FileCheck %s

// CHECK: #map = affine_map<(d0) -> (-d0 + 130, 64)>
// CHECK: module {
// CHECK: func.func @test_data_layout_for(%arg0: memref<130xi16, #hivm.address_space<ub>>, %arg1: memref<130xi16, #hivm.address_space<ub>>, %arg2: memref<130xi16, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
func.func @test_data_layout_for(%arg0: memref<130xi16, #hivm.address_space<ub>>, %arg1: memref<130xi16, #hivm.address_space<ub>>, %arg2: memref<130xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c130 = arith.constant 130 : index
  %c64 = arith.constant 64 : index
  %c1_i16 = arith.constant 1 : i16
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  %1 = ave.hir.broadcast %c1_i16, %0 : i16, vector<64xi1> -> vector<64xi16>
  scf.for %arg3 = %c0 to %c130 step %c64 {
    %2 = affine.min affine_map<(d0) -> (-d0 + 130, 64)>(%arg3)
    %subview = memref.subview %arg0[%arg3] [%2] [1] : memref<130xi16, #hivm.address_space<ub>> to memref<?xi16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg3] [%2] [1] : memref<130xi16, #hivm.address_space<ub>> to memref<?xi16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg2[%arg3] [%2] [1] : memref<130xi16, #hivm.address_space<ub>> to memref<?xi16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %res, %new_true_shape = ave.hir.plt %2 {mask_op_idx = 0 : i32} : vector<64xi1>, index
    %3 = ave.hir.vload <NORM> %subview[%c0] : memref<?xi16, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi16>
    %4 = ave.hir.vload <NORM> %subview_0[%c0] : memref<?xi16, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi16>
    %5 = ave.hir.vsitofp %4, %res {part = #ave.vcvt_part_type<part_even>} : vector<64xi16>, vector<64xi1>, vector<64xf32>
    %6 = ave.hir.vadd %3, %4, %res : vector<64xi16>, vector<64xi1>
    %7 = ave.hir.vsub %6, %1, %res : vector<64xi16>, vector<64xi1>
    %8 = ave.hir.vsitofp %7, %res {part = #ave.vcvt_part_type<part_even>} : vector<64xi16>, vector<64xi1>, vector<64xf32>
    %9 = ave.hir.vdiv %8, %5, %res : vector<64xf32>, vector<64xi1>
    %10 = ave.hir.vfptosi %9, <trunc>, %res {sat = false} : vector<64xf32>, vector<64xi1>, vector<64xi32>
    %11 = ave.hir.vtrunci %10, false, %res {part = #ave.vcvt_part_type<part_even>} : vector<64xi32>, vector<64xi16>, vector<64xi1>
    ave.hir.masked_store <NORM_B16> %subview_1[%c0], %res, %11 : memref<?xi16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi16>
  }
  return
}

// -----
// CHECK-LABEL: @_attn_fwd_mix_aiv_outlined_vf_7
#map = affine_map<(d0) -> (-d0 + 80, 64)>
#map1 = affine_map<(d0)[s0] -> (d0 + s0)>
module {
  func.func @_attn_fwd_mix_aiv_outlined_vf_7(%arg0: memref<16xi32, #hivm.address_space<ub>>, %arg1: memref<16xi32, #hivm.address_space<ub>>, %arg2: memref<16xf32, #hivm.address_space<ub>>, %arg3: memref<16xi32, #hivm.address_space<ub>>, %arg4: memref<16xf32, #hivm.address_space<ub>>, %arg5: memref<16xf32, #hivm.address_space<ub>>, %arg6: memref<16xf32, #hivm.address_space<ub>>, %arg7: memref<16x80xf32, #hivm.address_space<ub>>, %arg8: memref<16x80xf32, #hivm.address_space<ub>>, %arg9: memref<16x512xf32, #hivm.address_space<ub>>, %arg10: memref<16x512xf32, #hivm.address_space<ub>>, %arg11: memref<16x80xbf16, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %c80 = arith.constant 80 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c-2139095040_i32 = arith.constant -2139095040 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.693147182 : f32
    %0 = ave.hir.pge <ALL> : vector<64xi1>
    %1 = ave.hir.broadcast %cst, %0 : f32, vector<64xi1> -> vector<64xf32>
    %2 = ave.hir.pge <ALL> : vector<64xi1>
    %3 = ave.hir.broadcast %c0_i32, %2 : i32, vector<64xi1> -> vector<64xi32>
    %4, %new_true_shape = ave.hir.plt %c80 {mask_op_idx = 0 : i32} : vector<64xi1>, index
    %res = ave.hir.vload <NORM> %arg0[%c0] : memref<16xi32, #hivm.address_space<ub>> into vector<64xi32>
    %res_1 = ave.hir.vload <NORM> %arg1[%c0] : memref<16xi32, #hivm.address_space<ub>> into vector<64xi32>
    %5 = ave.hir.vand %res, %res_1, %4 : vector<64xi32>, vector<64xi1>
    %res_2 = ave.hir.vload <NORM> %arg3[%c0] : memref<16xi32, #hivm.address_space<ub>> into vector<64xi32>
    %6 = ave.hir.vand %res_2, %res_1, %4 : vector<64xi32>, vector<64xi1>
    %res_3 = ave.hir.vload <NORM> %arg2[%c0] : memref<16xf32, #hivm.address_space<ub>> into vector<64xf32>
    %res_4 = ave.hir.vload <NORM> %arg4[%c0] : memref<16xf32, #hivm.address_space<ub>> into vector<64xf32>
    %7 = ave.hir.vadds %6, %c-2139095040_i32, %4 : vector<64xi32>, i32, vector<64xi1>
    %8 = ave.hir.vsmins %7, %c1_i32, %4 : vector<64xi32>, i32, vector<64xi1>
    %9 = ave.hir.vsmaxs %8, %c0_i32, %4 : vector<64xi32>, i32, vector<64xi1>
    %10 = ave.hir.vcmp <NE> %9, %3, %4 : vector<64xi32>, vector<64xi1> -> vector<64xi1>
    %11 = ave.hir.vsel %10, %1, %res_4 : vector<64xi1>, vector<64xf32>
    %12 = ave.hir.vadds %5, %c-2139095040_i32, %4 : vector<64xi32>, i32, vector<64xi1>
    %13 = ave.hir.vsmins %12, %c1_i32, %4 : vector<64xi32>, i32, vector<64xi1>
    %14 = ave.hir.vsmaxs %13, %c0_i32, %4 : vector<64xi32>, i32, vector<64xi1>
    %15 = ave.hir.vcmp <NE> %14, %3, %4 : vector<64xi32>, vector<64xi1> -> vector<64xi1>
    %16 = ave.hir.vsel %15, %1, %res_3 : vector<64xi1>, vector<64xf32>
    %17 = ave.hir.vmax %16, %11, %4 : vector<64xf32>, vector<64xi1>
    ave.hir.masked_store <NORM_B32> %arg5[%c0], %4, %17 : memref<16xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    %18 = ave.hir.vsub %res_3, %17, %4 : vector<64xf32>, vector<64xi1>
    %19 = ave.hir.vmuls %18, %cst_0, %4 : vector<64xf32>, f32, vector<64xi1>
    %20 = ave.hir.vexp %19, %4 : vector<64xf32>, vector<64xi1>
    scf.for %arg12 = %c0 to %c16 step %c1 {
      // CHECK: ave.hir.masked_store <NORM_B32> %arg6[%c0], %res, %19 : memref<16xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
      ave.hir.masked_store <NORM_B32> %arg6[%c0], %4, %20 : memref<16xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
      %subview = memref.subview %arg5[%arg12] [1] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      scf.for %arg13 = %c0 to %c80 step %c64 {
        %21 = affine.min affine_map<(d0) -> (-d0 + 80, 64)>(%arg13)
        %subview_5 = memref.subview %arg7[%arg12, %arg13] [1, %21] [1, 1] : memref<16x80xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[80, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_6 = memref.subview %arg8[%arg12, %arg13] [1, %21] [1, 1] : memref<16x80xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[80, 1], offset: ?>, #hivm.address_space<ub>>
        %22 = ave.hir.pltm %arg13, %c80 : vector<64xi1>
        %subview_7 = memref.subview %subview_5[0, 0] [1, %21] [1, 1] : memref<1x?xf32, strided<[80, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
        %res_8 = ave.hir.vload <NORM> %subview_7[%c0] : memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
        %res_9 = ave.hir.vload <BRC_B32> %subview[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
        %23 = ave.hir.vsub %res_8, %res_9, %22 : vector<64xf32>, vector<64xi1>
        %24 = ave.hir.vmuls %23, %cst_0, %22 : vector<64xf32>, f32, vector<64xi1>
        %25 = ave.hir.vexp %24, %22 : vector<64xf32>, vector<64xi1>
        %subview_10 = memref.subview %subview_6[0, 0] [1, %21] [1, 1] : memref<1x?xf32, strided<[80, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
        ave.hir.masked_store <NORM_B32> %subview_10[%c0], %22, %25 {hivm.is_continuous} : memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
      } {element_alignment_bit_width = 32 : i32}
    } {element_alignment_bit_width = 32 : i32}
    scf.for %arg12 = %c0 to %c16 step %c1 {
      %subview = memref.subview %arg6[%arg12] [1] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      scf.for %arg13 = %c0 to %c512 step %c64 {
        %subview_5 = memref.subview %arg9[%arg12, %arg13] [1, 64] [1, 1] : memref<16x512xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_6 = memref.subview %arg10[%arg12, %arg13] [1, 64] [1, 1] : memref<16x512xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_7 = memref.subview %subview_5[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
        %res_8 = ave.hir.vload <NORM> %subview_7[%c0] : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
        %res_9 = ave.hir.vload <BRC_B32> %subview[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
        %21 = ave.hir.pge <ALL> : vector<64xi1>
        %22 = ave.hir.vmul %res_8, %res_9, %21 : vector<64xf32>, vector<64xi1>
        %subview_10 = memref.subview %subview_6[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[512, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
        %23 = ave.hir.pge <ALL> : vector<64xi1>
        ave.hir.masked_store <NORM_B32> %subview_10[%c0], %23, %22 {hivm.is_continuous} : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
      } {element_alignment_bit_width = 32 : i32}
      scf.for %arg13 = %c0 to %c80 step %c64 {
        %21 = affine.min affine_map<(d0) -> (-d0 + 80, 64)>(%arg13)
        %subview_5 = memref.subview %arg8[%arg12, %arg13] [1, %21] [1, 1] : memref<16x80xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[80, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_6 = memref.subview %arg11[%arg12, %arg13] [1, %21] [1, 1] : memref<16x80xbf16, #hivm.address_space<ub>> to memref<1x?xbf16, strided<[80, 1], offset: ?>, #hivm.address_space<ub>>
        %22 = ave.hir.pltm %arg13, %c80 : vector<64xi1>
        %subview_7 = memref.subview %subview_5[0, 0] [1, %21] [1, 1] : memref<1x?xf32, strided<[80, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
        %res_8 = ave.hir.vload <NORM> %subview_7[%c0] : memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
        %23 = ave.hir.vtruncf %res_8, <rint>, false, <part_even>, %22 : vector<64xf32>, vector<64xbf16>, vector<64xi1>
        %subview_9 = memref.subview %subview_6[0, 0] [1, %21] [1, 1] : memref<1x?xbf16, strided<[80, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xbf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
        ave.hir.masked_store <NORM_B16> %subview_9[%c0], %22, %23 {hivm.is_continuous} : memref<?xbf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xbf16>
      } {element_alignment_bit_width = 32 : i32}
    } {element_alignment_bit_width = 32 : i32}
    return
  }
}