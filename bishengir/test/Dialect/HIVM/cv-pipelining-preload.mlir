// RUN: bishengir-opt -cv-pipelining="enable-auto-preload" -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: func.func @_attn_fwd
// CHECK: scf.for %{{.*}} = %{{.*}} to %c160_i32 step %{{.*}}
// CHECK:   scope.scope : () -> i32 {
// CHECK:     hivm.hir.mmadL1 {b_transpose, fixpipe_already_inserted = true}
// CHECK:     hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
// CHECK:     scope.return
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
// CHECK:   scope.scope : () -> (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) {
// CHECK:     hivm.hir.vreduce <max>
// CHECK:     hivm.hir.vreduce <sum>
// CHECK:     scope.return
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
// CHECK:   scope.scope : () -> i32 {
// CHECK:     hivm.hir.mmadL1 {fixpipe_already_inserted = true, tile_cube_loop}
// CHECK:     hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
// CHECK:     scope.return
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
// CHECK:   scope.scope : () -> tensor<128x128xf32> {
// CHECK:     hivm.hir.vmul
// CHECK:     hivm.hir.vadd
// CHECK:     scope.return
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}

func.func @_attn_fwd(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix", parallel_mode = "simd"} {
  %c512 = arith.constant 512 : index
  %true = arith.constant true
  %c128 = arith.constant 128 : index
  %c80_i32 = arith.constant 80 : i32
  %c2_i32 = arith.constant 2 : i32
  %c0_i32 = arith.constant 0 : i32
  %c2621440_i64 = arith.constant 2621440 : i64
  %c1310720_i64 = arith.constant 1310720 : i64
  %c128_i32 = arith.constant 128 : i32
  %c10240_i32 = arith.constant 10240 : i32
  %c160_i32 = arith.constant 160 : i32
  %cst = arith.constant 5.000000e-01 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c512_i32 = arith.constant 512 : i32
  %cst_1 = arith.constant 0xFF800000 : f32
  %cst_2 = arith.constant 1.000000e+00 : f32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg8, %arg9 : i32
  %1 = arith.muli %0, %arg10 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg10, %arg9 : i32
  %5 = arith.divsi %3, %4 : i32
  %6 = arith.remsi %5, %arg8 : i32
  %7 = tensor.empty() : tensor<128xf32>
  %8 = hivm.hir.vbrc ins(%cst_2 : f32) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
  %9 = hivm.hir.vbrc ins(%cst_1 : f32) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
  %10 = tensor.empty() : tensor<128x512xf32>
  %11 = tensor.empty() : tensor<128x128xf32>
  %12 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%11 : tensor<128x128xf32>) -> tensor<128x128xf32>
  scf.for %arg11 = %6 to %c160_i32 step %arg8  : i32 {
    %13 = arith.divsi %arg11, %c80_i32 : i32
    %14 = arith.remsi %arg11, %c80_i32 : i32
    %15 = arith.divsi %13, %c2_i32 : i32
    %16 = arith.remsi %13, %c2_i32 : i32
    %17 = arith.extsi %15 : i32 to i64
    %18 = arith.muli %17, %c2621440_i64 : i64
    %19 = arith.extsi %16 : i32 to i64
    %20 = arith.muli %19, %c1310720_i64 : i64
    %21 = arith.addi %18, %20 : i64
    %22 = arith.index_cast %21 : i64 to index
    %23 = arith.muli %14, %c128_i32 : i32
    %24 = arith.index_cast %23 : i32 to index
    %25 = arith.muli %24, %c128 : index
    %26 = arith.addi %25, %22 : index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%26], sizes: [128, 128], strides: [128, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[128, 1], offset: ?>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg7 to offset: [%26], sizes: [128, 128], strides: [128, 1] : memref<?xbf16> to memref<128x128xbf16, strided<[128, 1], offset: ?>>
    %alloc = memref.alloc() : memref<128x128xbf16>
    hivm.hir.load ins(%reinterpret_cast : memref<128x128xbf16, strided<[128, 1], offset: ?>>) outs(%alloc : memref<128x128xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %27 = bufferization.to_tensor %alloc restrict writable : memref<128x128xbf16>
    %28:5 = scf.for %arg12 = %c0_i32 to %c10240_i32 step %c512_i32 iter_args(%arg13 = %8, %arg14 = %12, %arg15 = %9, %arg16 = %c0_i32, %arg17 = %c0_i32) -> (tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, i32, i32)  : i32 {
      %37 = arith.index_cast %arg17 : i32 to index
      %38 = arith.muli %37, %c128 : index
      %39 = arith.addi %38, %22 : index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%39], sizes: [512, 128], strides: [128, 1] : memref<?xbf16> to memref<512x128xbf16, strided<[128, 1], offset: ?>>
      %40 = arith.index_cast %arg16 : i32 to index
      %41 = arith.muli %40, %c128 : index
      %42 = arith.addi %41, %22 : index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%42], sizes: [512, 128], strides: [128, 1] : memref<?xbf16> to memref<512x128xbf16, strided<[128, 1], offset: ?>>
      %alloc_7 = memref.alloc() : memref<512x128xbf16>
      hivm.hir.load ins(%reinterpret_cast_5 : memref<512x128xbf16, strided<[128, 1], offset: ?>>) outs(%alloc_7 : memref<512x128xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %43 = bufferization.to_tensor %alloc_7 restrict writable : memref<512x128xbf16>
      %44 = tensor.empty() : tensor<128x512xbf16>
      %45 = tensor.empty() : tensor<128x512xf32>
      %46 = hivm.hir.mmadL1 {b_transpose, fixpipe_already_inserted = true} ins(%27, %43, %true, %c128, %c128, %c512 : tensor<128x128xbf16>, tensor<512x128xbf16>, i1, index, index, index) outs(%45 : tensor<128x512xf32>) -> tensor<128x512xf32>
      %47 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8> to memref<128x512xf32>
      annotation.mark %47 {hivm.multi_buffer = 4 : i32} : memref<128x512xf32>
      %48 = bufferization.to_tensor %47 restrict writable : memref<128x512xf32>
      %49 = hivm.hir.fixpipe {enable_nz2nd} ins(%46 : tensor<128x512xf32>) outs(%48 : tensor<128x512xf32>) -> tensor<128x512xf32>
      %50 = tensor.empty() : tensor<128x512xf32>
      %51 = hivm.hir.load ins(%49 : tensor<128x512xf32>) outs(%50 : tensor<128x512xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<128x512xf32>
      %52 = hivm.hir.vmul ins(%51, %cst : tensor<128x512xf32>, f32) outs(%10 : tensor<128x512xf32>) -> tensor<128x512xf32>
      %53 = tensor.empty() : tensor<128x1xf32>
      %54 = hivm.hir.vreduce <max> ins(%52 : tensor<128x512xf32>) outs(%53 : tensor<128x1xf32>) reduce_dims = [1] -> tensor<128x1xf32>
      %collapsed = tensor.collapse_shape %54 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
      %55 = hivm.hir.vmax ins(%arg15, %collapsed : tensor<128xf32>, tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
      %expanded_8 = tensor.expand_shape %55 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
      %56 = hivm.hir.vsub ins(%52, %expanded_8 : tensor<128x512xf32>, tensor<128x1xf32>) outs(%10 : tensor<128x512xf32>) broadcast = [1] -> tensor<128x512xf32>
      %57 = hivm.hir.vexp ins(%56 : tensor<128x512xf32>) outs(%10 : tensor<128x512xf32>) -> tensor<128x512xf32>
      %58 = hivm.hir.vcast ins(%57 : tensor<128x512xf32>) outs(%44 : tensor<128x512xbf16>) -> tensor<128x512xbf16>
      %59 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8> to memref<128x512xbf16>
      annotation.mark %59 {hivm.multi_buffer = 4 : i32} : memref<128x512xbf16>
      %60 = bufferization.to_tensor %59 restrict writable : memref<128x512xbf16>
      %61 = hivm.hir.store ins(%58 : tensor<128x512xbf16>) outs(%60 : tensor<128x512xbf16>) -> tensor<128x512xbf16>
      %62 = tensor.empty() : tensor<128x512xbf16>
      %63 = hivm.hir.load ins(%61 : tensor<128x512xbf16>) outs(%62 : tensor<128x512xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<128x512xbf16>
      %alloc_9 = memref.alloc() : memref<512x128xbf16>
      hivm.hir.load ins(%reinterpret_cast_6 : memref<512x128xbf16, strided<[128, 1], offset: ?>>) outs(%alloc_9 : memref<512x128xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %64 = bufferization.to_tensor %alloc_9 restrict writable : memref<512x128xbf16>
      %65 = tensor.empty() : tensor<128x128xf32>
      %66 = hivm.hir.mmadL1 {fixpipe_already_inserted = true, tile_cube_loop} ins(%63, %64, %true, %c128, %c512, %c128 : tensor<128x512xbf16>, tensor<512x128xbf16>, i1, index, index, index) outs(%65 : tensor<128x128xf32>) -> tensor<128x128xf32>
      %67 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8> to memref<128x128xf32>
      annotation.mark %67 {hivm.multi_buffer = 4 : i32} : memref<128x128xf32>
      %68 = bufferization.to_tensor %67 restrict writable : memref<128x128xf32>
      %69 = hivm.hir.fixpipe {enable_nz2nd} ins(%66 : tensor<128x128xf32>) outs(%68 : tensor<128x128xf32>) -> tensor<128x128xf32>
      %70 = tensor.empty() : tensor<128x128xf32>
      %71 = hivm.hir.load ins(%69 : tensor<128x128xf32>) outs(%70 : tensor<128x128xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<128x128xf32>
      %72 = tensor.empty() : tensor<128x1xf32>
      %73 = hivm.hir.vreduce <sum> ins(%57 : tensor<128x512xf32>) outs(%72 : tensor<128x1xf32>) reduce_dims = [1] -> tensor<128x1xf32>
      %collapsed_10 = tensor.collapse_shape %73 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
      %74 = hivm.hir.vsub ins(%arg15, %55 : tensor<128xf32>, tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
      %75 = hivm.hir.vexp ins(%74 : tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
      %76 = hivm.hir.vmul ins(%arg13, %75 : tensor<128xf32>, tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
      %77 = hivm.hir.vadd ins(%76, %collapsed_10 : tensor<128xf32>, tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
      %expanded_11 = tensor.expand_shape %75 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
      %78 = hivm.hir.vmul ins(%arg14, %expanded_11 : tensor<128x128xf32>, tensor<128x1xf32>) outs(%11 : tensor<128x128xf32>) broadcast = [1] -> tensor<128x128xf32>
      %79 = hivm.hir.vadd ins(%78, %71 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%11 : tensor<128x128xf32>) -> tensor<128x128xf32>
      %80 = arith.addi %arg16, %c512_i32 : i32
      %81 = arith.addi %arg17, %c512_i32 : i32
      scf.yield %77, %79, %55, %80, %81 : tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, i32, i32
    }
    %29 = hivm.hir.vln ins(%28#0 : tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
    %30 = hivm.hir.vadd ins(%28#2, %29 : tensor<128xf32>, tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
    %expanded = tensor.expand_shape %28#0 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
    %31 = hivm.hir.vdiv ins(%28#1, %expanded : tensor<128x128xf32>, tensor<128x1xf32>) outs(%11 : tensor<128x128xf32>) broadcast = [1] -> tensor<128x128xf32>
    %32 = arith.muli %13, %c10240_i32 : i32
    %33 = arith.index_cast %32 : i32 to index
    %34 = arith.addi %33, %24 : index
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%34], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
    hivm.hir.store ins(%30 : tensor<128xf32>) outs(%reinterpret_cast_4 : memref<128xf32, strided<[1], offset: ?>>)
    %35 = tensor.empty() : tensor<128x128xbf16>
    %36 = hivm.hir.vcast ins(%31 : tensor<128x128xf32>) outs(%35 : tensor<128x128xbf16>) -> tensor<128x128xbf16>
    hivm.hir.store ins(%36 : tensor<128x128xbf16>) outs(%reinterpret_cast_3 : memref<128x128xbf16, strided<[128, 1], offset: ?>>)
  }
  return
}