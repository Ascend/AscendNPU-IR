// RUN: bishengir-opt -append-vector-layout -annotate-dist-op-layout \
// RUN: -eliminate-vector-layout -ave-normalize-ops -convert-hivmave-to-ave-intrin -cse %s | FileCheck %s

// CHECK-LABEL: @sort_kernel_2d_outlined_vf_0
#map = affine_map<()[s0] -> (s0 * 12)>
#map1 = affine_map<(d0)[s0] -> (d0 + s0)>
#map2 = affine_map<()[s0] -> (s0 * 32)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @sort_kernel_2d_outlined_vf_0(%arg0: memref<10x12xf8E4M3FN, #hivm.address_space<ub>>, %arg1: memref<10x12xf32, strided<[32, 1]>, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c12 = arith.constant 12 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    scf.for %arg2 = %c0 to %c10 step %c1 {
      %res, %new_true_shape = ave.hir.plt %c12 {mask_op_idx = 0 : i32} : vector<64xi1>, index
      %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<10x12xf8E4M3FN, #hivm.address_space<ub>> -> memref<f8E4M3FN, #hivm.address_space<ub>>, index, index, index, index, index
        %0 = affine.apply #map()[%arg2]
        %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%0], sizes: [12], strides: [1] : memref<f8E4M3FN, #hivm.address_space<ub>> to memref<12xf8E4M3FN, #map1, #hivm.address_space<ub>>
        // CHECK: hivm_regbaseintrins.intr.hivm.vintlv
        // CHECK: llvm.extractvalue {{.*}}[0] : !llvm.struct<(vector<256xf8E4M3FN>, vector<256xf8E4M3FN>)>
        // CHECK: hivm_regbaseintrins.intr.hivm.vintlv
        // CHECK: llvm.extractvalue {{.*}}[0] : !llvm.struct<(vector<256xf8E4M3FN>, vector<256xf8E4M3FN>)>
        %res_0 = ave.hir.vload <NORM> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<12xf8E4M3FN, #map1, #hivm.address_space<ub>> into vector<64xf8E4M3FN>
      %1 = ave.hir.vextf %res_0, <part_even>, %res : vector<64xf8E4M3FN>, vector<64xf32>, vector<64xi1>
      %base_buffer_1, %offset_2, %sizes_3:2, %strides_4:2 = memref.extract_strided_metadata %arg1 : memref<10x12xf32, strided<[32, 1]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
      %2 = affine.apply #map2()[%arg2]
      %reinterpret_cast_5 = memref.reinterpret_cast %base_buffer_1 to offset: [%2], sizes: [12], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<12xf32, #map1, #hivm.address_space<ub>>
      ave.hir.masked_store <NORM_B32> %reinterpret_cast_5[%c0], %res, %1 : memref<12xf32, #map1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    } {element_alignment_bit_width = 32 : i32}
    return
  }
// CHECK-LABEL: @sort_kernel_2d_outlined_vf_1
  func.func @sort_kernel_2d_outlined_vf_1(%arg0: memref<10x12xf32, strided<[32, 1]>, #hivm.address_space<ub>>, %arg1: memref<10x12xf8E4M3FN, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c12 = arith.constant 12 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    scf.for %arg2 = %c0 to %c10 step %c1 {
      %res, %new_true_shape = ave.hir.plt %c12 {mask_op_idx = 0 : i32} : vector<64xi1>, index
      %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<10x12xf32, strided<[32, 1]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
      %0 = affine.apply #map2()[%arg2]
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%0], sizes: [12], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<12xf32, #map1, #hivm.address_space<ub>>
      %res_0 = ave.hir.vload <NORM> %reinterpret_cast[%c0] : memref<12xf32, #map1, #hivm.address_space<ub>> into vector<64xf32>
      %1 = ave.hir.vtruncf %res_0, <round>, false, <part_even>, %res : vector<64xf32>, vector<64xf8E4M3FN>, vector<64xi1>
        %base_buffer_1, %offset_2, %sizes_3:2, %strides_4:2 = memref.extract_strided_metadata %arg1 : memref<10x12xf8E4M3FN, #hivm.address_space<ub>> -> memref<f8E4M3FN, #hivm.address_space<ub>>, index, index, index, index, index
        %2 = affine.apply #map()[%arg2]
        %reinterpret_cast_5 = memref.reinterpret_cast %base_buffer_1 to offset: [%2], sizes: [12], strides: [1] : memref<f8E4M3FN, #hivm.address_space<ub>> to memref<12xf8E4M3FN, #map1, #hivm.address_space<ub>>
        // CHECK: hivm_regbaseintrins.intr.hivm.vdintlv
        // CHECK: llvm.extractvalue {{.*}}[0] : !llvm.struct<(vector<256xf8E4M3FN>, vector<256xf8E4M3FN>)>
        // CHECK: hivm_regbaseintrins.intr.hivm.vdintlv
        // CHECK: llvm.extractvalue {{.*}}[0] : !llvm.struct<(vector<256xf8E4M3FN>, vector<256xf8E4M3FN>)>
        ave.hir.masked_store <NORM_B8> %reinterpret_cast_5[%c0], %res, %1 {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<12xf8E4M3FN, #map1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf8E4M3FN>
    } {element_alignment_bit_width = 32 : i32}
    return
  }
}