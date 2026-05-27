// RUN: bishengir-opt --legalize-opt-hivmave %s | FileCheck %s
 
#map = affine_map<()[s0, s1] -> (s0 * 16 + s1 * 8)>
#map1 = affine_map<(d0)[s0] -> (d0 + s0)>
#map2 = affine_map<()[s0] -> (s0 * 256)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2097152 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @triton_unk_fused__npu_dtype_cast_ones_like_triu_where_zeros_like_4_outlined_vf_5(%arg0: memref<2x2xi1, strided<[256, 1]>, #hivm.address_space<ub>>, %arg1: memref<64x2x2xf32, strided<[16, 8, 1]>, #hivm.address_space<ub>>, %arg2: memref<64x2x2xf32, strided<[16, 8, 1]>, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = ave.hir.pge <ALL> : vector<64xi1>
    %1 = ave.hir.broadcast %cst, %0 : f32, vector<64xi1> -> vector<64xf32>
    %2 = ave.hir.pge <ALLF> : vector<64xi1>
    %3 = ave.hir.pge <VL2> {mask_op_idx = 0 : i32} : vector<64xi1>
    scf.for %arg3 = %c0 to %c64 step %c1 {
      scf.for %arg4 = %c0 to %c2 step %c1 {
        %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<2x2xi1, strided<[256, 1]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index, index, index
        %4 = affine.apply #map2()[%arg4]
        %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%4], sizes: [1, 2], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<1x2xi1, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>
        %5 = ave.hir.vload <NORM> %reinterpret_cast[%c0, %c0] : memref<1x2xi1, strided<[256, 1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
        %base_buffer_0, %offset_1, %sizes_2:3, %strides_3:3 = memref.extract_strided_metadata %arg1 : memref<64x2x2xf32, strided<[16, 8, 1]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index, index, index
        %6 = affine.apply #map()[%arg3, %arg4]
        %reinterpret_cast_4 = memref.reinterpret_cast %base_buffer_0 to offset: [%6], sizes: [2], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<2xf32, #map1, #hivm.address_space<ub>>
        %7 = ave.hir.vload <NORM> %reinterpret_cast_4[%c0] : memref<2xf32, #map1, #hivm.address_space<ub>> into vector<64xf32>
        // CHECK: ave.hir.preg.xor
        %8 = ave.hir.vcmp <NE> %5, %2, %3 : vector<64xi1>, vector<64xi1> -> vector<64xi1>
        // CHECK: ave.hir.preg.xor
        // CHECK: ave.hir.preg.not
        %test = ave.hir.vcmp <EQ> %8, %2, %3 : vector<64xi1>, vector<64xi1> -> vector<64xi1>
        %9 = ave.hir.vsel %test, %1, %7 : vector<64xi1>, vector<64xf32>
        %base_buffer_5, %offset_6, %sizes_7:3, %strides_8:3 = memref.extract_strided_metadata %arg2 : memref<64x2x2xf32, strided<[16, 8, 1]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index, index, index
        %10 = affine.apply #map()[%arg3, %arg4]
        %reinterpret_cast_9 = memref.reinterpret_cast %base_buffer_5 to offset: [%10], sizes: [2], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<2xf32, #map1, #hivm.address_space<ub>>
        ave.hir.masked_store <NORM_B32> %reinterpret_cast_9[%c0], %3, %9 : memref<2xf32, #map1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
      }
    }
    return
  }
}
