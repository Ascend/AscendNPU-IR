// RUN: bishengir-opt --one-shot-bufferize -split-input-file %s | FileCheck %s
// CHECK: scope.return %alloc_1 : memref<128x128xf32>
func.func @debug_scope_original(%arg0: tensor<128x128xf32>, %arg1: tensor<128xf32>) -> tensor<128x128xf32> {
  %0 = scope.scope : () -> tensor<128x128xf32> {
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
    %49 = tensor.empty() : tensor<128x128xf32>
    %50 = hivm.hir.load ins(%arg0 : tensor<128x128xf32>) outs(%49 : tensor<128x128xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<128x128xf32>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
    %expanded = tensor.expand_shape %arg1 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
    %51 = tensor.empty() : tensor<128x128xf32>
    %52 = hivm.hir.vmul ins(%50, %expanded : tensor<128x128xf32>, tensor<128x1xf32>) outs(%51 : tensor<128x128xf32>) broadcast = [1] -> tensor<128x128xf32>
    %53 = tensor.empty() : tensor<128x128xf32>
    %54 = hivm.hir.vadd ins(%52, %50 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%53 : tensor<128x128xf32>) -> tensor<128x128xf32>
    scope.return %54 : tensor<128x128xf32>
  } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, preload_num = 0 : i32}
  return %0 : tensor<128x128xf32>
}