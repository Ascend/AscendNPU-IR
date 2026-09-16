// RUN: bishengir-opt -hivm-remove-copy-ops %s | FileCheck %s


// CASE 1:
#map = affine_map<(d0)[s0] -> (-d0 + s0, 64)>
#map1 = affine_map<(d0) -> (0, d0)>
#map2 = affine_map<(d0) -> (d0, 0)>
#map3 = affine_map<(d0) -> (-d0 + 4194304, 5456)>
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, transform.with_named_sequence} {
  func.func @rms_norm_f32_outlined_vf_1(%arg0: index, %arg1: memref<1x?xf32, #hivm.address_space<ub>>, %arg2: memref<1x?xf32, #hivm.address_space<ub>>, %arg3: index, %arg4: index, %arg5: index, %arg6: memref<1x?xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : f32
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    scf.for %arg7 = %c0 to %arg0 step %c64 {
      %0 = affine.min #map(%arg7)[%arg0]
      %subview = memref.subview %arg6[0, %arg7] [1, %0] [1, 1] : memref<1x?xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[0, %arg7] [1, %0] [1, 1] : memref<1x?xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
      %1 = vector.create_mask %c1, %0 : vector<1x64xi1>
      %2 = vector.transfer_read %subview_0[%c0, %c0], %cst, %1 {in_bounds = [true, true]} : memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %3 = arith.mulf %2, %2 : vector<1x64xf32>
      %4 = vector.transfer_read %subview[%c0, %c0], %cst, %1 {in_bounds = [true, true]} : memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %5 = arith.addf %4, %3 : vector<1x64xf32>
      vector.transfer_write %5, %subview[%c0, %c0], %1 {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @rms_norm_f32(%arg0: memref<8x4194304xf32, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<4194304xf32, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>}, %arg2: memref<8x4194304xf32, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}, %arg3: memref<8xf32, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<1>}) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.get_tiling_struct_size_function = #hacc.get_tiling_struct_size_function<@rms_norm_f32_get_tiling_struct_size_function>, hfusion.block_dim = 1 : i64, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 2.38418579E-7 : f32
    %cst_1 = arith.constant 1.000000e-01 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4194304 = arith.constant 4194304 : index
    %c8 = arith.constant 8 : index
    %c5456 = arith.constant 5456 : index
    hivm.hir.set_mask_norm
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x5456xf32, #hivm.address_space<ub>>
    %cast = memref.cast %alloc : memref<1x5456xf32, #hivm.address_space<ub>> to memref<1x?xf32, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1xf32, #hivm.address_space<ub>>
    scf.for %arg4 = %c0 to %c8 step %c1 {
      %subview = memref.subview %arg3[%arg4] [1] [1] : memref<8xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %arg2[%arg4, 0] [1, 4194304] [1, 1] : memref<8x4194304xf32, #hivm.address_space<gm>> to memref<1x4194304xf32, strided<[4194304, 1], offset: ?>, #hivm.address_space<gm>>
      annotation.mark %cast {buffer_size_in_byte = 21824 : i64} : memref<1x?xf32, #hivm.address_space<ub>>
      scf.for %arg5 = %c0 to %c4194304 step %c5456 {
        %4 = affine.min #map3(%arg5)
        %subview_6 = memref.subview %arg0[%arg4, %arg5] [1, %4] [1, 1] : memref<8x4194304xf32, #hivm.address_space<gm>> to memref<1x?xf32, strided<[4194304, 1], offset: ?>, #hivm.address_space<gm>>
        %alloc_7 = memref.alloc(%4) {alignment = 64 : i64} : memref<1x?xf32, #hivm.address_space<ub>>
        %alloc_8 = memref.alloc(%4) {alignment = 64 : i64} : memref<1x?xf32, #hivm.address_space<ub>>
        hivm.hir.load ins(%subview_6 : memref<1x?xf32, strided<[4194304, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_8 : memref<1x?xf32, #hivm.address_space<ub>>)
        %subview_9 = memref.subview %alloc[0, 0] [1, %4] [1, 1] : memref<1x5456xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[5456, 1]>, #hivm.address_space<ub>>
        annotation.mark %alloc_8 {buffer_size_in_byte = 21824 : i64} : memref<1x?xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_8 {buffer_size_in_byte = 21824 : i64} : memref<1x?xf32, #hivm.address_space<ub>>
        %alloc_10 = memref.alloc(%4) {alignment = 64 : i64} : memref<1x?xf32, #hivm.address_space<ub>>
        // CHECK-NOT: hivm.hir.copy
        hivm.hir.copy ins(%subview_9 : memref<1x?xf32, strided<[5456, 1]>, #hivm.address_space<ub>>) outs(%alloc_10 : memref<1x?xf32, #hivm.address_space<ub>>)
        func.call @rms_norm_f32_outlined_vf_1(%4, %alloc_8, %alloc_7, %c0, %c1, %c1, %alloc_10) {hivm.vector_function} : (index, memref<1x?xf32, #hivm.address_space<ub>>, memref<1x?xf32, #hivm.address_space<ub>>, index, index, index, memref<1x?xf32, #hivm.address_space<ub>>) -> ()
        annotation.mark %alloc_10 {buffer_size_in_byte = 21824 : i64} : memref<1x?xf32, #hivm.address_space<ub>>
        // CHECK-NOT: hivm.hir.copy
        hivm.hir.copy ins(%alloc_10 : memref<1x?xf32, #hivm.address_space<ub>>) outs(%subview_9 : memref<1x?xf32, strided<[5456, 1]>, #hivm.address_space<ub>>)
      } {__reduction_loop__}
    } {__tiled_for___4}
    return
  }
}
