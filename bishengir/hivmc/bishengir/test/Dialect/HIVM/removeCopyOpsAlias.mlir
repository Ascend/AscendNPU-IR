// RUN: bishengir-opt -hivm-remove-copy-ops %s | FileCheck %s

// CASE 2:
// CASE 3:
#map = affine_map<(d0)[s0] -> (-d0 + s0, 64)>
#map1 = affine_map<(d0) -> (0, d0)>
#map2 = affine_map<(d0) -> (d0, 0)>
#map3 = affine_map<(d0) -> (-d0 + 65536, 5456)>
#map4 = affine_map<()[s0] -> (s0 * 5456 - (s0 floordiv 13) * 70928)>
#map5 = affine_map<()[s0] -> (s0 * -5456 + (s0 floordiv 13) * 70928 + 65536, 5456)>
#map6 = affine_map<()[s0] -> (s0 floordiv 13)>
#map7 = affine_map<()[s0, s1] -> (s0 + s1 floordiv 13)>
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, transform.with_named_sequence} {
  func.func @rms_norm_f32_outlined_vf_0(%arg0: index, %arg1: memref<1x?xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %cst = arith.constant dense<0.000000e+00> : vector<1x64xf32>
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    scf.for %arg2 = %c0 to %arg0 step %c64 {
      %0 = affine.min #map(%arg2)[%arg0]
      %subview = memref.subview %arg1[0, %arg2] [1, %0] [1, 1] : memref<1x?xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
      %1 = vector.create_mask %c1, %0 : vector<1x64xi1>
      vector.transfer_write %cst, %subview[%c0, %c0], %1 {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @rms_norm_f32_outlined_vf_1(%arg0: index, %arg1: memref<?x?xf32, #hivm.address_space<ub>>, %arg2: index, %arg3: memref<?x?xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : f32
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %arg2 step %c1 {
      scf.for %arg5 = %c0 to %arg0 step %c64 {
        %0 = affine.min #map(%arg5)[%arg0]
        %subview = memref.subview %arg3[%arg4, %arg5] [1, %0] [1, 1] : memref<?x?xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_0 = memref.subview %arg1[%arg4, %arg5] [1, %0] [1, 1] : memref<?x?xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
        %1 = vector.create_mask %c1, %0 : vector<1x64xi1>
        %2 = vector.transfer_read %subview_0[%c0, %c0], %cst, %1 {in_bounds = [true, true]} : memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
        %3 = arith.mulf %2, %2 : vector<1x64xf32>
        %4 = vector.transfer_read %subview[%c0, %c0], %cst, %1 {in_bounds = [true, true]} : memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
        %5 = arith.addf %4, %3 : vector<1x64xf32>
        vector.transfer_write %5, %subview[%c0, %c0], %1 {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
      }
    }
    return
  }
  func.func @rms_norm_f32_outlined_vf_2(%arg0: index, %arg1: memref<1x?xf32, #hivm.address_space<ub>>, %arg2: memref<1xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : f32
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg2[%c0], %cst {in_bounds = [true]} : memref<1xf32, #hivm.address_space<ub>>, vector<1xf32>
    %1 = scf.for %arg3 = %c0 to %arg0 step %c64 iter_args(%arg4 = %0) -> (vector<1xf32>) {
      %2 = affine.min #map(%arg3)[%arg0]
      %subview = memref.subview %arg1[0, %arg3] [1, %2] [1, 1] : memref<1x?xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
      %3 = vector.create_mask %c1, %2 : vector<1x64xi1>
      %4 = vector.transfer_read %subview[%c0, %c0], %cst, %3 {in_bounds = [true, true]} : memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %5 = vector.mask %3 { vector.multi_reduction <add>, %4, %arg4 [1] : vector<1x64xf32> to vector<1xf32> } : vector<1x64xi1> -> vector<1xf32>
      scf.yield %5 : vector<1xf32>
    }
    vector.transfer_write %1, %arg2[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, #hivm.address_space<ub>>
    return
  }
  func.func @rms_norm_f32_outlined_vf_3(%arg0: index, %arg1: memref<?xf32, #hivm.address_space<ub>>, %arg2: memref<?xf32, #hivm.address_space<ub>>, %arg3: memref<?x?xf32, #hivm.address_space<ub>>, %arg4: index, %arg5: memref<?x?xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : f32
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    scf.for %arg6 = %c0 to %arg4 step %c1 {
      %subview = memref.subview %arg2[%arg6] [1] [1] : memref<?xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      scf.for %arg7 = %c0 to %arg0 step %c64 {
        %0 = affine.min #map(%arg7)[%arg0]
        %subview_0 = memref.subview %arg5[%arg6, %arg7] [1, %0] [1, 1] : memref<?x?xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
        %1 = vector.create_mask %0 : vector<64xi1>
        %subview_1 = memref.subview %arg1[%arg7] [%0] [1] : memref<?xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %subview_2 = memref.subview %arg3[%arg6, %arg7] [1, %0] [1, 1] : memref<?x?xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
        %2 = vector.create_mask %c1, %0 : vector<1x64xi1>
        %3 = vector.transfer_read %subview_1[%c0], %cst, %1 {in_bounds = [true, true], permutation_map = #map1} : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
        %4 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true, true], permutation_map = #map2} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
        %5 = vector.transfer_read %subview_2[%c0, %c0], %cst, %2 {in_bounds = [true, true]} : memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
        %6 = arith.mulf %4, %5 : vector<1x64xf32>
        %7 = arith.mulf %3, %6 : vector<1x64xf32>
        vector.transfer_write %7, %subview_0[%c0, %c0], %2 {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
      }
    }
    return
  }
  func.func @rms_norm_f32_get_tiling_struct_size_function() -> i64 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<get_tiling_struct_size_function>} {
    %c0_i64 = arith.constant 0 : i64
    return %c0_i64 : i64
  }
  func.func @rms_norm_f32(%arg0: memref<8x65536xf32, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<65536xf32, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>}, %arg2: memref<8x65536xf32, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}, %arg3: memref<8xf32, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<1>}) attributes {enable_auto_mark_buffer_size, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.get_tiling_struct_size_function = #hacc.get_tiling_struct_size_function<@rms_norm_f32_get_tiling_struct_size_function>, hfusion.block_dim = 1 : i64, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 1.52587891E-5 : f32
    %cst_1 = arith.constant 1.000000e-01 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c65536 = arith.constant 65536 : index
    %c8 = arith.constant 8 : index
    %c5456 = arith.constant 5456 : index
    %c13 = arith.constant 13 : index
    hivm.hir.set_mask_norm
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x5456xf32, #hivm.address_space<ub>>
    %cast = memref.cast %alloc : memref<1x5456xf32, #hivm.address_space<ub>> to memref<1x?xf32, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1xf32, #hivm.address_space<ub>>
    scf.for %arg4 = %c0 to %c8 step %c1 {
      %subview = memref.subview %arg3[%arg4] [1] [1] : memref<8xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %arg2[%arg4, 0] [1, 65536] [1, 1] : memref<8x65536xf32, #hivm.address_space<gm>> to memref<1x65536xf32, strided<[65536, 1], offset: ?>, #hivm.address_space<gm>>
      func.call @rms_norm_f32_outlined_vf_0(%c5456, %cast) {hivm.vector_function} : (index, memref<1x?xf32, #hivm.address_space<ub>>) -> ()
      scf.for %arg5 = %c0 to %c65536 step %c5456 {
        %4 = affine.min #map3(%arg5)
        %subview_6 = memref.subview %arg0[%arg4, %arg5] [1, %4] [1, 1] : memref<8x65536xf32, #hivm.address_space<gm>> to memref<1x?xf32, strided<[65536, 1], offset: ?>, #hivm.address_space<gm>>
        %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<21824xi8, #hivm.address_space<ub>>
        %view = memref.view %alloc_7[%c0][%4] : memref<21824xi8, #hivm.address_space<ub>> to memref<1x?xf32, #hivm.address_space<ub>>
        %subview_8 = memref.subview %subview_6[0, 0] [1, %4] [1, 1] : memref<1x?xf32, strided<[65536, 1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_9 = memref.subview %view[0, 0] [1, %4] [1, 1] : memref<1x?xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.load ins(%subview_8 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_9 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
        %cast_10 = memref.cast %view : memref<1x?xf32, #hivm.address_space<ub>> to memref<?x?xf32, #hivm.address_space<ub>>
        %subview_11 = memref.subview %alloc[0, 0] [1, %4] [1, 1] : memref<1x5456xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[5456, 1]>, #hivm.address_space<ub>>
        %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<21824xi8, #hivm.address_space<ub>>
        %view_13 = memref.view %alloc_12[%c0][%4] : memref<21824xi8, #hivm.address_space<ub>> to memref<1x?xf32, #hivm.address_space<ub>>
        %cast_14 = memref.cast %view_13 : memref<1x?xf32, #hivm.address_space<ub>> to memref<?x?xf32, #hivm.address_space<ub>>
        %subview_15 = memref.subview %subview_11[0, 0] [1, %4] [1, 1] : memref<1x?xf32, strided<[5456, 1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %subview_16 = memref.subview %view_13[0, 0] [1, %4] [1, 1] : memref<1x?xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        // CHECK: memref.cast
        // CHECK-NOT: hivm.hir.copy
        hivm.hir.copy ins(%subview_15 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_16 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
        func.call @rms_norm_f32_outlined_vf_1(%4, %cast_10, %c1, %cast_14) {hivm.vector_function} : (index, memref<?x?xf32, #hivm.address_space<ub>>, index, memref<?x?xf32, #hivm.address_space<ub>>) -> ()
        // CHECK-NOT: hivm.hir.copy
        hivm.hir.copy ins(%subview_16 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_15 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      }
      func.call @rms_norm_f32_outlined_vf_2(%c5456, %cast, %alloc_2) {hivm.vector_function} : (index, memref<1x?xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) -> ()
      %0 = memref.load %alloc_2[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %1 = arith.mulf %0, %cst_0 : f32
      %2 = arith.addf %1, %cst_1 : f32
      %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1xf32, #hivm.address_space<ub>>
      %3 = math.sqrt %2 : f32
      memref.store %3, %alloc_4[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_5 : memref<1xf32, #hivm.address_space<ub>>)
      hivm.hir.vdiv ins(%alloc_5, %alloc_4 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%alloc_5 : memref<1xf32, #hivm.address_space<ub>>)
      hivm.hir.store ins(%alloc_5 : memref<1xf32, #hivm.address_space<ub>>) outs(%subview : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) atomic = <none>
      scf.for %arg5 = %c0 to %c13 step %c1 {
        %4 = affine.apply #map4()[%arg5]
        %5 = affine.min #map5()[%arg5]
        %subview_6 = memref.subview %arg1[%4] [%5] [1] : memref<65536xf32, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<21824xi8, #hivm.address_space<ub>>
        %view = memref.view %alloc_7[%c0][%5] : memref<21824xi8, #hivm.address_space<ub>> to memref<?xf32, #hivm.address_space<ub>>
        hivm.hir.load ins(%subview_6 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%view : memref<?xf32, #hivm.address_space<ub>>)
        %6 = affine.apply #map6()[%arg5]
        %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<21824xi8, #hivm.address_space<ub>>
        %view_9 = memref.view %alloc_8[%c0][%5] : memref<21824xi8, #hivm.address_space<ub>> to memref<1x?xf32, #hivm.address_space<ub>>
        %cast_10 = memref.cast %view_9 : memref<1x?xf32, #hivm.address_space<ub>> to memref<?x?xf32, #hivm.address_space<ub>>
        %7 = affine.apply #map7()[%arg4, %arg5]
        %subview_11 = memref.subview %arg0[%7, %4] [1, %5] [1, 1] : memref<8x65536xf32, #hivm.address_space<gm>> to memref<1x?xf32, strided<[65536, 1], offset: ?>, #hivm.address_space<gm>>
        %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<21824xi8, #hivm.address_space<ub>>
        %view_13 = memref.view %alloc_12[%c0][%5] : memref<21824xi8, #hivm.address_space<ub>> to memref<1x?xf32, #hivm.address_space<ub>>
        %subview_14 = memref.subview %subview_11[0, 0] [1, %5] [1, 1] : memref<1x?xf32, strided<[65536, 1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_15 = memref.subview %view_13[0, 0] [1, %5] [1, 1] : memref<1x?xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.load ins(%subview_14 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_15 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
        %cast_16 = memref.cast %view_13 : memref<1x?xf32, #hivm.address_space<ub>> to memref<?x?xf32, #hivm.address_space<ub>>
        %subview_17 = memref.subview %alloc_5[%6] [1] [1] : memref<1xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %subview_18 = memref.subview %subview_3[%6, %4] [1, %5] [1, 1] : memref<1x65536xf32, strided<[65536, 1], offset: ?>, #hivm.address_space<gm>> to memref<1x?xf32, strided<[65536, 1], offset: ?>, #hivm.address_space<gm>>
        %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<1xf32, #hivm.address_space<ub>>
        %cast_20 = memref.cast %alloc_19 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, #hivm.address_space<ub>>
        // CHECK: memref.cast
        // CHECK-NOT: hivm.hir.copy
        hivm.hir.copy ins(%subview_17 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>) outs(%alloc_19 : memref<1xf32, #hivm.address_space<ub>>)
        func.call @rms_norm_f32_outlined_vf_3(%5, %view, %cast_20, %cast_16, %c1, %cast_10) {hivm.vector_function} : (index, memref<?xf32, #hivm.address_space<ub>>, memref<?xf32, #hivm.address_space<ub>>, memref<?x?xf32, #hivm.address_space<ub>>, index, memref<?x?xf32, #hivm.address_space<ub>>) -> ()
        %subview_21 = memref.subview %view_9[0, 0] [1, %5] [1, 1] : memref<1x?xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %subview_22 = memref.subview %subview_18[0, 0] [1, %5] [1, 1] : memref<1x?xf32, strided<[65536, 1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.store ins(%subview_21 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_22 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) atomic = <none>
      } {__coalesced_loop___1, __tiled_for___4}
    } {__tiled_for___3}
    return
  }
}
