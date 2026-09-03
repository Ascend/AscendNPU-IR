// RUN: bishengir-opt -hivm-remove-copy-ops %s | FileCheck %s

// CASE 4:
func.func @entry_outlined_vf_0(%arg0: index, %arg1: index, %arg2: memref<?x?xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant dense<0.000000e+00> : vector<1x64xf32>
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  scf.for %arg3 = %c0 to %arg1 step %c1 {
    scf.for %arg4 = %c0 to %arg0 step %c64 {
      %0 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 64)>(%arg4)[%arg0]
      %subview = memref.subview %arg2[%arg3, %arg4] [1, %0] [1, 1] : memref<?x?xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
      %1 = vector.create_mask %c1, %0 : vector<1x64xi1>
      vector.transfer_write %cst, %subview[%c0, %c0], %1 {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
    }
  }
  return
}

func.func @entry_outlined_vf_1(%arg0: index, %arg1: memref<?x?xf32, #hivm.address_space<ub>>, %arg2: index, %arg3: memref<?x?xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  scf.for %arg4 = %c0 to %arg2 step %c1 {
    scf.for %arg5 = %c0 to %arg0 step %c64 {
      %0 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 64)>(%arg5)[%arg0]
      %subview = memref.subview %arg3[%arg5, %arg4] [%0, 1] [1, 1] : memref<?x?xf32, #hivm.address_space<ub>> to memref<?x1xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[%arg4, %arg5] [1, %0] [1, 1] : memref<?x?xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
      %1 = vector.create_mask %c1, %0 : vector<1x64xi1>
      %2 = vector.create_mask %0, %c1 : vector<64x1xi1>
      %3 = vector.transfer_read %subview_0[%c0, %c0], %cst, %1 {in_bounds = [true, true]} : memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %4 = vector.transfer_read %subview[%c0, %c0], %cst, %2 {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<?x1xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %5 = arith.addf %4, %3 : vector<1x64xf32>
      vector.transfer_write %5, %subview[%c0, %c0], %2 {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : vector<1x64xf32>, memref<?x1xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
    }
  }
  return
}

func.func @entry_get_tiling_struct_size_function() -> i64 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<get_tiling_struct_size_function>} {
  %c0_i64 = arith.constant 0 : i64
  return %c0_i64 : i64
}

func.func @entry_outlined_vf_2(%arg0: index, %arg1: memref<?x?xf32, #hivm.address_space<ub>>, %arg2: index, %arg3: memref<?xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  scf.for %arg4 = %c0 to %arg2 step %c1 {
    scf.for %arg5 = %c0 to %arg0 step %c64 {
      %0 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 64)>(%arg5)[%arg0]
      %subview = memref.subview %arg3[%arg5] [%0] [1] : memref<?xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %1 = vector.create_mask %0 : vector<64xi1>
      %subview_0 = memref.subview %arg1[%arg5, %arg4] [%0, 1] [1, 1] : memref<?x?xf32, #hivm.address_space<ub>> to memref<?x1xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
      %2 = vector.create_mask %0, %c1 : vector<64x1xi1>
      %3 = vector.create_mask %c1, %0 : vector<1x64xi1>
      %4 = vector.shape_cast %3 : vector<1x64xi1> to vector<64xi1>
      %5 = vector.transfer_read %subview_0[%c0, %c0], %cst, %2 {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<?x1xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %6 = vector.shape_cast %5 : vector<1x64xf32> to vector<64xf32>
      %7 = vector.transfer_read %subview[%c0], %cst, %1 {in_bounds = [true]} : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xf32>
      %8 = arith.addf %7, %6 : vector<64xf32>
      %9 = arith.select %4, %8, %6 : vector<64xi1>, vector<64xf32>
      vector.transfer_write %9, %subview[%c0], %1 {in_bounds = [true]} : vector<64xf32>, memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    }
  }
  return
}

func.func @entry(%arg0: memref<256x128xf32, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<128xf32, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}) attributes {enable_auto_mark_buffer_size, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.get_tiling_struct_size_function = #hacc.get_tiling_struct_size_function<@entry_get_tiling_struct_size_function>, hfusion.block_dim = 1 : i64, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c76 = arith.constant 76 : index
  %c128 = arith.constant 128 : index
  hivm.hir.set_mask_norm
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32, #hivm.address_space<ub>>
  %cast = memref.cast %alloc : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<76x128xf32, #hivm.address_space<ub>>
  %cast_1 = memref.cast %alloc_0 : memref<76x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, #hivm.address_space<ub>>
  call @entry_outlined_vf_0(%c128, %c76, %cast_1) {hivm.vector_function} : (index, index, memref<?x?xf32, #hivm.address_space<ub>>) -> ()
  scf.for %arg2 = %c0 to %c256 step %c76 {
    %0 = affine.min affine_map<(d0) -> (-d0 + 256, 76)>(%arg2)
    %subview = memref.subview %arg0[%arg2, 0] [%0, 128] [1, 1] : memref<256x128xf32, #hivm.address_space<gm>> to memref<?x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<39296xi8, #hivm.address_space<ub>>
    %view = memref.view %alloc_2[%c0][%0] : memref<39296xi8, #hivm.address_space<ub>> to memref<?x128xf32, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %view [[0, 1]] : memref<?x128xf32, #hivm.address_space<ub>> into memref<?xf32, #hivm.address_space<ub>>
    %collapse_shape_3 = memref.collapse_shape %subview [[0, 1]] : memref<?x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.load ins(%collapse_shape_3 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape : memref<?xf32, #hivm.address_space<ub>>)
    %cast_4 = memref.cast %view : memref<?x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, #hivm.address_space<ub>>
    %subview_5 = memref.subview %alloc_0[0, 0] [128, %0] [1, 1] : memref<76x128xf32, #hivm.address_space<ub>> to memref<128x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    %1 = affine.apply affine_map<()[s0] -> (((s0 + 7) floordiv 8) * 8)>()[%0]
    %alloc_6 = memref.alloc() : memref<39296xi8, #hivm.address_space<ub>>
    %view_7 = memref.view %alloc_6[%c0][%1] : memref<39296xi8, #hivm.address_space<ub>> to memref<128x?x1xf32, #hivm.address_space<ub>>
    %subview_8 = memref.subview %view_7[0, 0, 0] [128, %0, 1] [1, 1, 1] : memref<128x?x1xf32, #hivm.address_space<ub>> to memref<128x?xf32, strided<[?, 1]>, #hivm.address_space<ub>>
    %cast_9 = memref.cast %subview_8 : memref<128x?xf32, strided<[?, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, #hivm.address_space<ub>>
    // CHECK-NOT: hivm.hir.copy
    hivm.hir.copy ins(%subview_5 : memref<128x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%subview_8 : memref<128x?xf32, strided<[?, 1]>, #hivm.address_space<ub>>)
    func.call @entry_outlined_vf_1(%c128, %cast_4, %0, %cast_9) {hivm.vector_function} : (index, memref<?x?xf32, #hivm.address_space<ub>>, index, memref<?x?xf32, #hivm.address_space<ub>>) -> ()
    // CHECK-NOT: hivm.hir.copy
    hivm.hir.copy ins(%subview_8 : memref<128x?xf32, strided<[?, 1]>, #hivm.address_space<ub>>) outs(%subview_5 : memref<128x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
  } {__reduction_loop__}
  call @entry_outlined_vf_2(%c76, %cast_1, %c128, %cast) {hivm.vector_function} : (index, memref<?x?xf32, #hivm.address_space<ub>>, index, memref<?xf32, #hivm.address_space<ub>>) -> ()
  hivm.hir.store ins(%alloc : memref<128xf32, #hivm.address_space<ub>>) outs(%arg1 : memref<128xf32, #hivm.address_space<gm>>) atomic = <none>
  return
}
