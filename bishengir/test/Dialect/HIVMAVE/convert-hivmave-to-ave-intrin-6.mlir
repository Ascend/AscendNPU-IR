 // RUN: bishengir-opt -data-layout-analyze -append-vector-layout -annotate-dist-op-layout \
 // RUN: -eliminate-vector-layout -convert-hivmave-to-ave-intrin -cse -split-input-file %s | FileCheck %s
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

// -----

// CHECK-LABEL: @test_broadcast_mask
#map = affine_map<()[s0] -> (s0 * 128)>
#map1 = affine_map<(d0)[s0] -> (d0 + s0)>
#map2 = affine_map<()[s0] -> (s0 * 64)>
#map3 = affine_map<()[s0, s1] -> (s0 * 128 + s1)>
#map4 = affine_map<()[s0, s1, s2] -> (s0 * 1024 + s1 * 64 + s2 * 16)>
#map5 = affine_map<()[s0, s1, s2, s3] -> (s0 * s1 + s2 + s3)>
#map6 = affine_map<()[s0, s1] -> (s0 - s1)>
#map7 = affine_map<()[s0] -> ((s0 + 15) floordiv 16)>
#map8 = affine_map<()[s0, s1, s2] -> (s0 * 2048 + s1 * 256 + s2 * 16)>
#map9 = affine_map<()[s0, s1, s2] -> (s0 * 2048 + s1 * 128 + s2 * 16)>
#map10 = affine_map<()[s0, s1, s2] -> (s0 * 1024 + s1 * 256 + s2 * 16)>
#map11 = affine_map<()[s0, s1, s2] -> (s0 * 512 + s1 * 256 + s2 * 16)>
#map12 = affine_map<()[s0] -> (s0 + 64)>
#map13 = affine_map<()[s0, s1] -> (s0 + s1)>
#map14 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>
#map15 = affine_map<()[s0] -> (s0 + 32)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @test_broadcast_mask(%arg0: memref<32xf32, #hivm.address_space<ub>>, %arg1: memref<32xf32, #hivm.address_space<ub>>, %arg2: memref<64x128xf16, #hivm.address_space<ub>>, %arg3: memref<128x64xf16, #hivm.address_space<ub>>, %arg4: memref<32x128xf32, #hivm.address_space<ub>>, %arg5: memref<32x64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %cst_1 = arith.constant 0.000000e+00 : f16
    %0 = ave.hir.pge <ALL> : vector<128xi1>
    %1 = ave.hir.broadcast %cst_1, %0: f16, vector<128xi1> -> vector<128xf16>
    // CHECK : "hivm_regbaseintrins.intr.hivm.pge.b16"
    // CHECK : "hivm_regbaseintrins.intr.hivm.vdups.z"
    // CHECK : "hivm_regbaseintrins.intr.hivm.pge.b32"
    // CHECK : "hivm_regbaseintrins.intr.hivm.vdups.z"
    // CHECK : "hivm_regbaseintrins.intr.hivm.pge.b32"
    // CHECK : "hivm_regbaseintrins.intr.hivm.vdups.z"
    // CHECK : "hivm_regbaseintrins.intr.hivm.pge.b32"
    // CHECK : "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"
    // CHECK : "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"
    %2 = ave.hir.pge <ALL> : vector<64xi1>
    %3 = ave.hir.broadcast %cst_0, %2  : f32, vector<64xi1> -> vector<64xf32>
    %4 = ave.hir.pge <ALL> : vector<64xi1>
    %5 = ave.hir.broadcast %cst, %4  : f32, vector<64xi1> -> vector<64xf32>
    %6 = ave.hir.pge <VL32> {mask_op_idx = 0 : i32} : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %arg0[%c0], %6, %5 : memref<32xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    ave.hir.masked_store <NORM_B32> %arg1[%c0], %6, %3 : memref<32xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    scf.for %arg6 = %c0 to %c64 step %c1 {
      %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg2 : memref<64x128xf16, #hivm.address_space<ub>> -> memref<f16, #hivm.address_space<ub>>, index, index, index, index, index
      %7 = affine.apply #map()[%arg6]
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%7], sizes: [128], strides: [1] : memref<f16, #hivm.address_space<ub>> to memref<128xf16, #map1, #hivm.address_space<ub>>
      %8 = ave.hir.pge <ALL> : vector<128xi1>
      ave.hir.masked_store <NORM_B16> %reinterpret_cast[%c0], %8, %1 {hivm.is_continuous} : memref<128xf16, #map1, #hivm.address_space<ub>>, vector<128xi1>, vector<128xf16>
    }
    scf.for %arg6 = %c0 to %c128 step %c1 {
      %7 = ave.hir.pge <VL64> {mask_op_idx = 1 : i32} : vector<128xi1>
      %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg3 : memref<128x64xf16, #hivm.address_space<ub>> -> memref<f16, #hivm.address_space<ub>>, index, index, index, index, index
      %8 = affine.apply #map2()[%arg6]
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%8], sizes: [64], strides: [1] : memref<f16, #hivm.address_space<ub>> to memref<64xf16, #map1, #hivm.address_space<ub>>
      ave.hir.masked_store <NORM_B16> %reinterpret_cast[%c0], %7, %1 : memref<64xf16, #map1, #hivm.address_space<ub>>, vector<128xi1>, vector<128xf16>
    }
    scf.for %arg6 = %c0 to %c32 step %c1 {
      scf.for %arg7 = %c0 to %c128 step %c64 {
        %base_buffer_2, %offset_3, %sizes_4:2, %strides_5:2 = memref.extract_strided_metadata %arg4 : memref<32x128xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
        %9 = affine.apply #map3()[%arg6, %arg7]
        %reinterpret_cast_6 = memref.reinterpret_cast %base_buffer_2 to offset: [%9], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, #map1, #hivm.address_space<ub>>
        %10 = ave.hir.pge <ALL> : vector<64xi1>
        ave.hir.masked_store <NORM_B32> %reinterpret_cast_6[%c0], %10, %3 {hivm.is_continuous} : memref<64xf32, #map1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
      }
      %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg5 : memref<32x64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
      %7 = affine.apply #map2()[%arg6]
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%7], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, #map1, #hivm.address_space<ub>>
      %8 = ave.hir.pge <ALL> : vector<64xi1>
      ave.hir.masked_store <NORM_B32> %reinterpret_cast[%c0], %8, %5 {hivm.is_continuous} : memref<64xf32, #map1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    }
    return
  }
}