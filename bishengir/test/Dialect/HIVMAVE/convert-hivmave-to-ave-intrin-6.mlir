 // RUN: bishengir-opt  -analyze-vector-layout -analyze-alignment-bitwidth \
 // RUN: -remove-vector-layout-attr -convert-hivmave-to-ave-intrin -cse -split-input-file %s | FileCheck %s
 // CHECK-LABEL: @test_long_vector_len_case
func.func @test_long_vector_len_case(%arg0: memref<1x1x135xi8, strided<[160, 160, 1]>, #hivm.address_space<ub>>, %arg1: memref<1x1x135xi8, strided<[160, 160, 1]>, #hivm.address_space<ub>>, %arg2: memref<1x1x135xi8, strided<[160, 160, 1]>, #hivm.address_space<ub>>, %arg3: memref<1x1x135xi8, strided<[160, 160, 1]>, #hivm.address_space<ub>>, %arg4: memref<1x1x135xi8, strided<[160, 160, 1]>, #hivm.address_space<ub>>, %arg5: memref<1x1x135xi8, strided<[160, 160, 1]>, #hivm.address_space<ub>>, %arg6: memref<1x1x135xi8, strided<[160, 160, 1]>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
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
  }
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

// -----

// CHECK-LABEL: @test_deintlv
#map = affine_map<()[s0] -> (s0 * 92)>
#map1 = affine_map<(d0)[s0] -> (d0 + s0)>
#map2 = affine_map<(d0) -> (-d0 + 92, 64)>
#map3 = affine_map<()[s0] -> ((s0 floordiv 2) * 2)>
#map4 = affine_map<(d0)[s0] -> (64, -d0 + (s0 floordiv 2) * 2)>
#map5 = affine_map<()[s0] -> (s0 floordiv 2)>
#map6 = affine_map<()[s0, s1, s2] -> (s0 * 46 + s1 floordiv 2 + s2 floordiv 2)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @test_deintlv(%arg0: memref<64x92xbf16, #hivm.address_space<ub>>, %arg1: memref<64x46xf32, #hivm.address_space<ub>>, %arg2: memref<64x46xf32, #hivm.address_space<ub>>, %arg3: memref<46xf32, #hivm.address_space<ub>>, %arg4: memref<46xf32, #hivm.address_space<ub>>, %arg5: memref<64x92xbf16, #hivm.address_space<ub>>, %arg6: memref<1x46xf32, #hivm.address_space<ub>>, %arg7: memref<1x46xf32, #hivm.address_space<ub>>, %arg8: memref<256xi8, #hivm.address_space<ub>>, %arg9: memref<128xi8, #hivm.address_space<ub>>, %arg10: memref<128xi8, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c46 = arith.constant 46 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c92 = arith.constant 92 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant -1.000000e+00 : f32
    scf.for %arg11 = %c0 to %c64 step %c1 {
      %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<64x92xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index
      %0 = affine.apply #map()[%arg11]
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%0], sizes: [1, 92], strides: [92, 1] : memref<bf16, #hivm.address_space<ub>> to memref<1x92xbf16, strided<[92, 1], offset: ?>, #hivm.address_space<ub>>
      %res = ave.hir.vload <NORM> %reinterpret_cast[%c0, %c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, element_alignment_bit_width = 16 : i32} : memref<1x92xbf16, strided<[92, 1], offset: ?>, #hivm.address_space<ub>> into vector<128xbf16>
      %res_0 = ave.hir.vload <NORM> %reinterpret_cast[%c0, %c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, element_alignment_bit_width = 16 : i32} : memref<1x92xbf16, strided<[92, 1], offset: ?>, #hivm.address_space<ub>> into vector<128xbf16>
      %res1, %res2 = ave.hir.vdintlv %res, %res_0 {element_alignment_bit_width = 16 : i32} : vector<128xbf16>, vector<128xbf16>
      // CHECK: "hivm_regbaseintrins.intr.hivm.vdintlv"
      // CHECK: "hivm_regbaseintrins.intr.hivm.vintlv"
      // CHECK: "hivm_regbaseintrins.intr.hivm.vintlv"
      // CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.bf162f32.x"
      // CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.bf162f32.x"
      %1 = builtin.unrealized_conversion_cast %res1 : vector<128xbf16> to vector<1x128xbf16>
      %2 = builtin.unrealized_conversion_cast %res2 : vector<128xbf16> to vector<1x128xbf16>
      %3 = builtin.unrealized_conversion_cast %1 : vector<1x128xbf16> to vector<1x64xbf16>
      %4 = builtin.unrealized_conversion_cast %3 : vector<1x64xbf16> to vector<64xbf16>
      %5 = ave.hir.pge <ALL> {element_alignment_bit_width = -1 : i32} : vector<64xi1>
      %6 = ave.hir.vextf %4, <part_even>, %5 {element_alignment_bit_width = 32 : i32} : vector<64xbf16>, vector<64xf32>, vector<64xi1>
      %7 = builtin.unrealized_conversion_cast %2 : vector<1x128xbf16> to vector<1x64xbf16>
      %8 = builtin.unrealized_conversion_cast %7 : vector<1x64xbf16> to vector<64xbf16>
      %9 = ave.hir.pge <ALL> {element_alignment_bit_width = -1 : i32} : vector<64xi1>
      %10 = ave.hir.vextf %8, <part_even>, %9 {element_alignment_bit_width = 32 : i32} : vector<64xbf16>, vector<64xf32>, vector<64xi1>
      scf.for %arg12 = %c0 to %c92 step %c64 {
        %11 = affine.min #map2(%arg12)
        %12 = affine.apply #map3()[%11]
        %view = memref.view %arg8[%c0][%12] : memref<256xi8, #hivm.address_space<ub>> to memref<1x?xf32, #hivm.address_space<ub>>
        scf.for %arg13 = %c0 to %12 step %c64 {
          %15 = affine.min #map4(%arg13)[%11]
          %16 = affine.apply #map5()[%15]
          %base_buffer_13, %offset_14, %sizes_15:2, %strides_16:2 = memref.extract_strided_metadata %arg2 : memref<64x46xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
          %17 = affine.apply #map6()[%arg11, %arg13, %arg12]
          %reinterpret_cast_17 = memref.reinterpret_cast %base_buffer_13 to offset: [%17], sizes: [1, %16], strides: [46, 1] : memref<f32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[46, 1], offset: ?>, #hivm.address_space<ub>>
          %base_buffer_18, %offset_19, %sizes_20:2, %strides_21:2 = memref.extract_strided_metadata %arg1 : memref<64x46xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
          %18 = affine.apply #map6()[%arg11, %arg13, %arg12]
          %reinterpret_cast_22 = memref.reinterpret_cast %base_buffer_18 to offset: [%18], sizes: [1, %16], strides: [46, 1] : memref<f32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[46, 1], offset: ?>, #hivm.address_space<ub>>
          %res_23, %new_true_shape_24 = ave.hir.plt %c46 {mask_op_idx = 2 : i32} : vector<64xi1>, index
          %base_buffer_25, %offset_26, %sizes_27:2, %strides_28:2 = memref.extract_strided_metadata %arg7 : memref<1x46xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
          %reinterpret_cast_29 = memref.reinterpret_cast %base_buffer_25 to offset: [0], sizes: [46], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<46xf32, strided<[1]>, #hivm.address_space<ub>>
          ave.hir.masked_store <NORM_B32> %reinterpret_cast_29[%c0], %res_23, %6 {element_alignment_bit_width = 32 : i32} : memref<46xf32, strided<[1]>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
          %base_buffer_30, %offset_31, %sizes_32:2, %strides_33:2 = memref.extract_strided_metadata %arg6 : memref<1x46xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
          %reinterpret_cast_34 = memref.reinterpret_cast %base_buffer_30 to offset: [0], sizes: [46], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<46xf32, strided<[1]>, #hivm.address_space<ub>>
          ave.hir.masked_store <NORM_B32> %reinterpret_cast_34[%c0], %res_23, %10 {element_alignment_bit_width = 32 : i32} : memref<46xf32, strided<[1]>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
        }
      }
    }
    return
  }
}
