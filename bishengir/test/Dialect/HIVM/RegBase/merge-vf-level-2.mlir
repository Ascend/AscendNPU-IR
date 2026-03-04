// RUN: bishengir-opt --hfusion-merge-vf="merge-level=2" -split-input-file %s | FileCheck %s

// CHECK: %[[VAL:.*]]:4 = func.call @rms_norm_f32_split_n_with_vf_merge_outlined_merged__merged_vf_0
#map = affine_map<(d0)[s0] -> (-d0 + s0, 64)>
#map1 = affine_map<(d0) -> (0, d0)>
#map2 = affine_map<(d0) -> (d0, 0)>
#map3 = affine_map<()[s0] -> (s0 * 29)>
#map4 = affine_map<()[s0] -> (s0 * -29 + 256, 29)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 1 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 1 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 1 : i32>, #dlti.dl_entry<"UB_SIZE", 2097152 : i32>, #dlti.dl_entry<"L1_SIZE", 8388608 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 0 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 0 : i32>, #dlti.dl_entry<"ARCH", "dav-m300">>>, hacc.target = #hacc.target<"Ascend310B4">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @rms_norm_f32_split_n_with_vf_merge_get_tiling_struct_size_function() -> i64 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<get_tiling_struct_size_function>} {
    %c0_i64 = arith.constant 0 : i64
    return %c0_i64 : i64
  }
  func.func @rms_norm_f32_split_n_with_vf_merge_outlined_vf_0(%arg0: index, %arg1: memref<?xf32>) -> memref<?xf32> attributes {hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg2 = %c0 to %arg0 step %c64 iter_args(%arg3 = %arg1) -> (memref<?xf32>) {
      %1 = affine.min #map(%arg2)[%arg0]
      %subview = memref.subview %arg3[%arg2] [%1] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
      %2 = vector.create_mask %1 : vector<64xi1>
      vector.transfer_write %cst, %subview[%c0], %2 {in_bounds = [true]} : vector<64xf32>, memref<?xf32, strided<[1], offset: ?>>
      %subview_0 = memref.subview %arg3[%arg2] [%1] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
      memref.copy %subview, %subview_0 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      scf.yield %arg3 : memref<?xf32>
    }
    return %0 : memref<?xf32>
  }
  func.func @rms_norm_f32_split_n_with_vf_merge_outlined_vf_1(%arg0: memref<1x64xf32>, %arg1: memref<?x128xf32>, %arg2: index, %arg3: memref<?xf32>) -> memref<?xf32> attributes {hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0.000000e+00> : vector<1x64xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg4 = %c0 to %arg2 step %c1 iter_args(%arg5 = %arg3) -> (memref<?xf32>) {
      %subview = memref.subview %arg5[%arg4] [1] [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %1 = scf.for %arg6 = %c0 to %c128 step %c64 iter_args(%arg7 = %cst) -> (vector<1x64xf32>) {
        %subview_2 = memref.subview %arg1[%arg4, %arg6] [1, 64] [1, 1] : memref<?x128xf32> to memref<1x64xf32, strided<[128, 1], offset: ?>>
        %4 = vector.transfer_read %subview_2[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[128, 1], offset: ?>>, vector<1x64xf32>
        %5 = arith.mulf %4, %4 : vector<1x64xf32>
        %6 = arith.addf %arg7, %5 {reductionOp} : vector<1x64xf32>
        scf.yield %6 : vector<1x64xf32>
      } {reductionLoop}
      %2 = vector.transfer_read %subview[%c0], %cst_0 {in_bounds = [true]} : memref<1xf32, strided<[1], offset: ?>>, vector<1xf32>
      %3 = vector.multi_reduction <add>, %1, %2 [1] : vector<1x64xf32> to vector<1xf32>
      vector.transfer_write %3, %subview[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, strided<[1], offset: ?>>
      %subview_1 = memref.subview %arg5[%arg4] [1] [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      memref.copy %subview, %subview_1 : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32, strided<[1], offset: ?>>
      scf.yield %arg5 : memref<?xf32>
    }
    return %0 : memref<?xf32>
  }
  func.func @rms_norm_f32_split_n_with_vf_merge_outlined_vf_2(%arg0: index, %arg1: memref<?xf32>, %arg2: memref<?xf32>) -> memref<?xf32> attributes {hivm.vector_function, no_inline} {
    %cst = arith.constant dense<1.000000e+00> : vector<64xf32>
    %cst_0 = arith.constant dense<1.000000e-01> : vector<64xf32>
    %cst_1 = arith.constant dense<7.812500e-03> : vector<64xf32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg3 = %c0 to %arg0 step %c64 iter_args(%arg4 = %arg2) -> (memref<?xf32>) {
      %1 = affine.min #map(%arg3)[%arg0]
      %subview = memref.subview %arg1[%arg3] [%1] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
      %subview_3 = memref.subview %arg4[%arg3] [%1] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
      %2 = vector.create_mask %1 : vector<64xi1>
      %3 = vector.transfer_read %subview[%c0], %cst_2, %2 {in_bounds = [true]} : memref<?xf32, strided<[1], offset: ?>>, vector<64xf32>
      %4 = arith.mulf %3, %cst_1 : vector<64xf32>
      %5 = arith.addf %4, %cst_0 : vector<64xf32>
      %6 = math.sqrt %5 : vector<64xf32>
      %7 = arith.divf %cst, %6 : vector<64xf32>
      vector.transfer_write %7, %subview_3[%c0], %2 {in_bounds = [true]} : vector<64xf32>, memref<?xf32, strided<[1], offset: ?>>
      %subview_4 = memref.subview %arg4[%arg3] [%1] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
      memref.copy %subview_3, %subview_4 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      scf.yield %arg4 : memref<?xf32>
    }
    return %0 : memref<?xf32>
  }
  func.func @rms_norm_f32_split_n_with_vf_merge_outlined_vf_3(%arg0: memref<?xf32>, %arg1: memref<128xf32>, %arg2: memref<?x128xf32>, %arg3: index, %arg4: memref<?x128xf32>) -> memref<?x128xf32> attributes {hivm.vector_function, no_inline} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg5 = %c0 to %arg3 step %c1 iter_args(%arg6 = %arg4) -> (memref<?x128xf32>) {
      %subview = memref.subview %arg0[%arg5] [1] [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %1 = scf.for %arg7 = %c0 to %c128 step %c64 iter_args(%arg8 = %arg6) -> (memref<?x128xf32>) {
        %subview_0 = memref.subview %arg1[%arg7] [64] [1] : memref<128xf32> to memref<64xf32, strided<[1], offset: ?>>
        %subview_1 = memref.subview %arg2[%arg5, %arg7] [1, 64] [1, 1] : memref<?x128xf32> to memref<1x64xf32, strided<[128, 1], offset: ?>>
        %subview_2 = memref.subview %arg8[%arg5, %arg7] [1, 64] [1, 1] : memref<?x128xf32> to memref<1x64xf32, strided<[128, 1], offset: ?>>
        %2 = vector.transfer_read %subview_0[%c0], %cst {in_bounds = [true, true], permutation_map = #map1} : memref<64xf32, strided<[1], offset: ?>>, vector<1x64xf32>
        %3 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true, true], permutation_map = #map2} : memref<1xf32, strided<[1], offset: ?>>, vector<1x64xf32>
        %4 = vector.transfer_read %subview_1[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[128, 1], offset: ?>>, vector<1x64xf32>
        %5 = arith.mulf %3, %4 : vector<1x64xf32>
        %6 = arith.mulf %2, %5 : vector<1x64xf32>
        vector.transfer_write %6, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[128, 1], offset: ?>>
        %subview_3 = memref.subview %arg8[%arg5, %arg7] [1, 64] [1, 1] : memref<?x128xf32> to memref<1x64xf32, strided<[128, 1], offset: ?>>
        memref.copy %subview_2, %subview_3 : memref<1x64xf32, strided<[128, 1], offset: ?>> to memref<1x64xf32, strided<[128, 1], offset: ?>>
        scf.yield %arg8 : memref<?x128xf32>
      }
      scf.yield %1 : memref<?x128xf32>
    }
    return %0 : memref<?x128xf32>
  }
  func.func @rms_norm_f32_split_n_with_vf_merge(%arg0: memref<256x128xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<128xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>}, %arg2: memref<256x128xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}, %arg3: memref<256xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<1>}) -> (memref<256x128xf32>, memref<256xf32>) attributes {enable_auto_mark_buffer_size, hacc.block_dim = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.get_tiling_struct_size_function = #hacc.get_tiling_struct_size_function<@rms_norm_f32_split_n_with_vf_merge_get_tiling_struct_size_function>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>, hivm.vf_mode = #hivm.vf_mode<SIMD>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x64xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
    hivm.hir.load ins(%arg1 : memref<128xf32>) outs(%alloc_0 : memref<128xf32>)
    %0:2 = scf.for %arg4 = %c0 to %c9 step %c1 iter_args(%arg5 = %arg2, %arg6 = %arg3) -> (memref<256x128xf32>, memref<256xf32>) {
      %1 = affine.apply #map3()[%arg4]
      %2 = affine.min #map4()[%arg4]
      %alloc_1 = memref.alloc(%2) {alignment = 64 : i64} : memref<?x128xf32>
      %subview = memref.subview %arg0[%1, 0] [%2, 128] [1, 1] : memref<256x128xf32> to memref<?x128xf32, strided<[128, 1], offset: ?>>
      %alloc_2 = memref.alloc(%2) {alignment = 64 : i64} : memref<?x128xf32>
      hivm.hir.load ins(%subview : memref<?x128xf32, strided<[128, 1], offset: ?>>) outs(%alloc_2 : memref<?x128xf32>)
      %alloc_3 = memref.alloc(%2) {alignment = 64 : i64} : memref<?xf32>
      %alloc_4 = memref.alloc(%2) {alignment = 64 : i64} : memref<?xf32>
      %3 = func.call @rms_norm_f32_split_n_with_vf_merge_outlined_vf_0(%2, %alloc_4) {hivm.vector_function, no_inline} : (index, memref<?xf32>) -> memref<?xf32>
      %subview_5 = memref.subview %arg5[%1, 0] [%2, 128] [1, 1] : memref<256x128xf32> to memref<?x128xf32, strided<[128, 1], offset: ?>>
      %subview_6 = memref.subview %arg6[%1] [%2] [1] : memref<256xf32> to memref<?xf32, strided<[1], offset: ?>>
      annotation.mark %alloc_2 {buffer_size_in_byte = 15104 : i64} : memref<?x128xf32>
      annotation.mark %3 {buffer_size_in_byte = 15104 : i64} : memref<?xf32>
      %4 = func.call @rms_norm_f32_split_n_with_vf_merge_outlined_vf_1(%alloc, %alloc_2, %2, %3) {hivm.vector_function, no_inline} : (memref<1x64xf32>, memref<?x128xf32>, index, memref<?xf32>) -> memref<?xf32>
      annotation.mark %4 {buffer_size_in_byte = 15104 : i64} : memref<?xf32>
      %dim = memref.dim %4, %c0 : memref<?xf32>
      %5 = func.call @rms_norm_f32_split_n_with_vf_merge_outlined_vf_2(%dim, %4, %alloc_3) {hivm.vector_function, no_inline} : (index, memref<?xf32>, memref<?xf32>) -> memref<?xf32>
      annotation.mark %5 {buffer_size_in_byte = 15104 : i64} : memref<?xf32>
      hivm.hir.store ins(%5 : memref<?xf32>) outs(%subview_6 : memref<?xf32, strided<[1], offset: ?>>) atomic = <none>
      %subview_7 = memref.subview %arg6[%1] [%2] [1] : memref<256xf32> to memref<?xf32, strided<[1], offset: ?>>
      memref.copy %subview_6, %subview_7 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %dim_8 = memref.dim %5, %c0 : memref<?xf32>
      %6 = func.call @rms_norm_f32_split_n_with_vf_merge_outlined_vf_3(%5, %alloc_0, %alloc_2, %dim_8, %alloc_1) {hivm.vector_function, no_inline} : (memref<?xf32>, memref<128xf32>, memref<?x128xf32>, index, memref<?x128xf32>) -> memref<?x128xf32>
      annotation.mark %6 {buffer_size_in_byte = 15104 : i64} : memref<?x128xf32>
      hivm.hir.store ins(%6 : memref<?x128xf32>) outs(%subview_5 : memref<?x128xf32, strided<[128, 1], offset: ?>>) atomic = <none>
      %subview_9 = memref.subview %arg5[%1, 0] [%2, 128] [1, 1] : memref<256x128xf32> to memref<?x128xf32, strided<[128, 1], offset: ?>>
      memref.copy %subview_5, %subview_9 : memref<?x128xf32, strided<[128, 1], offset: ?>> to memref<?x128xf32, strided<[128, 1], offset: ?>>
      scf.yield %arg5, %arg6 : memref<256x128xf32>, memref<256xf32>
    } {__tiled_for___3}
    return %0#0, %0#1 : memref<256x128xf32>, memref<256xf32>
  }
}