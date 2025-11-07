// REQUIRES: asserts
// RUN: bishengir-opt %s --propagate-reshape --cse --canonicalize-ext --valid-propagate --debug-only="propagate-valid-check" -split-input-file | FileCheck %s

// CHECK: Valid
// CHECK-LABEL: func.func @scf_for_propagate
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.yield
// CHECK-SAME: : tensor<64x1xf32>,
// CHECK-NOT: expand_shape
// CHECK: hivm.hir.store
#map = affine_map<(d0) -> (d0 * 28672)>
#map1 = affine_map<(d0) -> (d0 * 28672 + 8192)>
#map2 = affine_map<(d0) -> (d0 * 28672 + 12288)>
module {
  func.func @scf_for_propagate(%arg0: memref<?xi8>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: f32, %arg4: i32, %arg5: tensor<64x64xf32>, %arg6: tensor<64x64xf32>, %arg7: tensor<64x32xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64xf32>, %arg10: tensor<64xf32>, %arg11: i64, %arg12: index, %arg13: i32, %arg14: f32, %arg15: i32, %arg16: i64, %arg17: i64, %arg18: i32, %arg19: i32, %arg20: index, %arg21: index, %arg22: f32, %arg23: f32, %arg24: i32) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false]> : vector<11xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    scf.for %arg25 = %arg19 to %arg18 step %arg24  : i32 {
      %0 = arith.divsi %arg25, %arg18 : i32
      %1 = arith.remsi %arg25, %arg18 : i32
      %2 = arith.extsi %0 : i32 to i64
      %3 = arith.muli %2, %arg17 : i64
      %4 = arith.extsi %1 : i32 to i64
      %5 = arith.muli %4, %arg16 : i64
      %6 = arith.addi %3, %5 : i64
      %7 = arith.index_cast %6 : i64 to index
      %8 = arith.index_cast %arg4 : i32 to index
      %9 = arith.muli %8, %arg20 : index
      %10 = arith.addi %9, %7 : index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%10], sizes: [64, 64], strides: [64, 1] : memref<?xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
      %11:7 = scf.for %arg26 = %arg19 to %arg15 step %arg13 iter_args(%arg27 = %arg9, %arg28 = %arg5, %arg29 = %arg8, %arg30 = %7, %arg31 = %arg12, %arg32 = %7, %arg33 = %arg12) -> (tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, index, index, index, index)  : i32 {
        %18 = tensor.empty() : tensor<64x32xf16>
        %19 = arith.index_cast %arg11 : i64 to index
        %20 = affine.apply #map(%19)
        %view = memref.view %arg0[%20][] : memref<?xi8> to memref<64x32xf32>
        %21 = bufferization.to_tensor %view restrict writable : memref<64x32xf32>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 1
        %22 = hivm.hir.load ins(%21 : tensor<64x32xf32>) outs(%arg7 : tensor<64x32xf32>) init_out_buffer = false -> tensor<64x32xf32>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 0
        %23 = tensor.empty() : tensor<64x1xf32>
        %24 = hivm.hir.vreduce <max> ins(%22 : tensor<64x32xf32>) outs(%23 : tensor<64x1xf32>) reduce_dims = [1] -> tensor<64x1xf32>
        %collapsed = tensor.collapse_shape %24 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
        %25 = hivm.hir.vmul ins(%collapsed, %arg3 : tensor<64xf32>, f32) outs(%arg10 : tensor<64xf32>) -> tensor<64xf32>
        %26 = hivm.hir.vmax ins(%arg29, %25 : tensor<64xf32>, tensor<64xf32>) outs(%arg10 : tensor<64xf32>) -> tensor<64xf32>
        %27 = hivm.hir.vmul ins(%22, %arg3 : tensor<64x32xf32>, f32) outs(%arg7 : tensor<64x32xf32>) -> tensor<64x32xf32>
        %expanded_0 = tensor.expand_shape %26 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
        %28 = hivm.hir.vbrc ins(%expanded_0 : tensor<64x1xf32>) outs(%arg7 : tensor<64x32xf32>) broadcast_dims = [1] -> tensor<64x32xf32>
        %29 = hivm.hir.vsub ins(%27, %28 : tensor<64x32xf32>, tensor<64x32xf32>) outs(%arg7 : tensor<64x32xf32>) -> tensor<64x32xf32>
        %30 = hivm.hir.vmul ins(%29, %arg22 : tensor<64x32xf32>, f32) outs(%arg7 : tensor<64x32xf32>) -> tensor<64x32xf32>
        %31 = hivm.hir.vexp ins(%30 : tensor<64x32xf32>) outs(%arg7 : tensor<64x32xf32>) -> tensor<64x32xf32>
        %32 = tensor.empty() : tensor<64x1xf32>
        %33 = hivm.hir.vreduce <sum> ins(%31 : tensor<64x32xf32>) outs(%32 : tensor<64x1xf32>) reduce_dims = [1] -> tensor<64x1xf32>
        %collapsed_1 = tensor.collapse_shape %33 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
        %34 = hivm.hir.vsub ins(%arg29, %26 : tensor<64xf32>, tensor<64xf32>) outs(%arg10 : tensor<64xf32>) -> tensor<64xf32>
        %35 = hivm.hir.vmul ins(%34, %arg22 : tensor<64xf32>, f32) outs(%arg10 : tensor<64xf32>) -> tensor<64xf32>
        %36 = hivm.hir.vexp ins(%35 : tensor<64xf32>) outs(%arg10 : tensor<64xf32>) -> tensor<64xf32>
        %37 = hivm.hir.vmul ins(%arg27, %36 : tensor<64xf32>, tensor<64xf32>) outs(%arg10 : tensor<64xf32>) -> tensor<64xf32>
        %38 = hivm.hir.vadd ins(%37, %collapsed_1 : tensor<64xf32>, tensor<64xf32>) outs(%arg10 : tensor<64xf32>) -> tensor<64xf32>
        %expanded_2 = tensor.expand_shape %36 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
        %39 = hivm.hir.vbrc ins(%expanded_2 : tensor<64x1xf32>) outs(%arg6 : tensor<64x64xf32>) broadcast_dims = [1] -> tensor<64x64xf32>
        %40 = hivm.hir.vmul ins(%arg28, %39 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg6 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %41 = hivm.hir.vcast ins(%31 : tensor<64x32xf32>) outs(%18 : tensor<64x32xf16>) -> tensor<64x32xf16>
        %42 = affine.apply #map1(%19)
        %view_3 = memref.view %arg0[%42][] : memref<?xi8> to memref<64x32xf16>
        %43 = bufferization.to_tensor %view_3 restrict writable : memref<64x32xf16>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 2
        %44 = hivm.hir.store ins(%41 : tensor<64x32xf16>) outs(%43 : tensor<64x32xf16>) -> tensor<64x32xf16>
        annotation.mark %44 : tensor<64x32xf16>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE2>] flag = 1
        %45 = affine.apply #map2(%19)
        %view_4 = memref.view %arg0[%45][] : memref<?xi8> to memref<64x64xf32>
        %46 = bufferization.to_tensor %view_4 restrict writable : memref<64x64xf32>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 1
        %47 = hivm.hir.load ins(%46 : tensor<64x64xf32>) outs(%arg6 : tensor<64x64xf32>) init_out_buffer = false -> tensor<64x64xf32>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_FIX>] flag = 3
        %48 = hivm.hir.vadd ins(%47, %40 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg6 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %49 = hivm.hir.vmul ins(%26, %arg3 : tensor<64xf32>, f32) outs(%arg10 : tensor<64xf32>) -> tensor<64xf32>
        %50 = hivm.hir.vdiv ins(%49, %arg14 : tensor<64xf32>, f32) outs(%arg10 : tensor<64xf32>) -> tensor<64xf32>
        %51 = arith.addi %arg30, %arg21 : index
        %52 = arith.addi %51, %arg31 : index
        %53 = arith.addi %arg32, %arg21 : index
        %54 = arith.addi %53, %arg33 : index
        scf.yield %38, %48, %50, %52, %arg12, %54, %arg12 : tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, index, index, index, index
      }
      %expanded = tensor.expand_shape %11#0 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
      %17 = hivm.hir.vbrc ins(%expanded : tensor<64x1xf32>) outs(%arg6 : tensor<64x64xf32>) broadcast_dims = [1] -> tensor<64x64xf32>
      hivm.hir.store ins(%17 : tensor<64x64xf32>) outs(%reinterpret_cast : memref<64x64xf32, strided<[64, 1], offset: ?>>)
    }
    return
  }
}