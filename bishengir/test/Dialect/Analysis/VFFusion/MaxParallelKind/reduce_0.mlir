// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="fusion-mode=max-parallel enable-ar enable-ra enable-ar" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func private @triton_sum_2D_dim1_1x64_mul_reduce_fused_0
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<64xf32>, %[[ARG1:.*]]: tensor<64xf32>, %[[ARG2:.*]]: f32, %[[ARG3:.*]]: tensor<f32>) -> tensor<f32>
// CHECK:         %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[MUL:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ARG0]], %[[ARG0]] : tensor<64xf32>, tensor<64xf32>) outs(%[[ARG1]] : tensor<64xf32>) -> tensor<64xf32>
// CHECK:         %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[ARG3]] : tensor<f32>) -> tensor<f32>
// CHECK:         %[[REDUCED:.*]] = linalg.reduce ins(%[[MUL]] : tensor<64xf32>) outs(%[[FILL]] : tensor<f32>) dimensions = [0]
// CHECK:           (%[[IN:.*]]: f32, %[[INIT:.*]]: f32) {
// CHECK:             %[[ADD:.*]] = arith.addf %[[IN]], %[[INIT]] : f32
// CHECK:             linalg.yield %[[ADD]] : f32
// CHECK:           }
// CHECK:         return %[[REDUCED]] : tensor<f32>
// CHECK-LABEL: func.func @triton_sum_2D_dim1_1x64_mul_reduce
// CHECK:         %[[CALL_RES:.*]] = call @triton_sum_2D_dim1_1x64_mul_reduce_fused_0
// CHECK-SAME:      (tensor<64xf32>, tensor<64xf32>, f32, tensor<f32>) -> tensor<f32>
// CHECK:         tensor.expand_shape %[[CALL_RES]] [] output_shape [1] : tensor<f32> into tensor<1xf32>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9579">} {
  func.func @triton_sum_2D_dim1_1x64_mul_reduce(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1]>>
    %alloc = memref.alloc() : memref<64xf32>
    memref.copy %reinterpret_cast, %alloc : memref<64xf32, strided<[1]>> to memref<64xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<64xf32>
    %1 = tensor.empty() : tensor<64xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%0, %0 : tensor<64xf32>, tensor<64xf32>) outs(%1 : tensor<64xf32>) -> tensor<64xf32>
    %3 = tensor.empty() : tensor<f32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<f32>) -> tensor<f32>
    %reduced = linalg.reduce ins(%2 : tensor<64xf32>) outs(%4 : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %5 = arith.addf %in, %init : f32
        linalg.yield %5 : f32
      }
    %expanded = tensor.expand_shape %reduced [] output_shape [1] : tensor<f32> into tensor<1xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1]>>
    bufferization.materialize_in_destination %expanded in writable %reinterpret_cast_0 : (tensor<1xf32>, memref<1xf32, strided<[1]>>) -> ()
    return
  }
}

// CHECK-LABEL: func.func private @triton_sum_2D_dim1_fused_0
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<64x64xf32>, %[[ARG1:.*]]: tensor<64xf32>) -> tensor<64xf32>
// CHECK-NOT:     linalg.elemwise_binary
// CHECK-NOT:     linalg.fill
// CHECK:         %[[REDUCED:.*]] = linalg.reduce ins(%[[ARG0]] : tensor<64x64xf32>) outs(%[[ARG1]] : tensor<64xf32>) dimensions = [1]
// CHECK:           (%[[IN:.*]]: f32, %[[INIT:.*]]: f32) {
// CHECK:             %[[ADD:.*]] = arith.addf %[[IN]], %[[INIT]] : f32
// CHECK:             linalg.yield %[[ADD]] : f32
// CHECK:           }
// CHECK-NOT:     linalg.reduce
// CHECK-NOT:     linalg.elemwise_binary
// CHECK:         return %[[REDUCED]] : tensor<64xf32>

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9579">} {
  func.func @triton_sum_2D_dim1(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xf32> to memref<64x64xf32, strided<[64, 1]>>
    %alloc = memref.alloc() : memref<64x64xf32>
    memref.copy %reinterpret_cast, %alloc : memref<64x64xf32, strided<[64, 1]>> to memref<64x64xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32>
    %1 = tensor.empty() : tensor<64x64xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%0, %0 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %3 = tensor.empty() : tensor<64xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<64xf32>) -> tensor<64xf32>
    %reduced = linalg.reduce ins(%2 : tensor<64x64xf32>) outs(%4 : tensor<64xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %5 = arith.addf %in, %init : f32
        linalg.yield %5 : f32
      }
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1]>>
    bufferization.materialize_in_destination %reduced in writable %reinterpret_cast_0 : (tensor<64xf32>, memref<64xf32, strided<[1]>>) -> ()
    return
  }
}

// CHECK-LABEL: func.func private @triton_sum_2D_dim0_fused_0
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<64x64xf32>, %[[ARG1:.*]]: tensor<64xf32>) -> tensor<64xf32>
// CHECK-NOT:     linalg.elemwise_binary
// CHECK-NOT:     linalg.fill
// CHECK:         %[[REDUCED:.*]] = linalg.reduce ins(%[[ARG0]] : tensor<64x64xf32>) outs(%[[ARG1]] : tensor<64xf32>) dimensions = [0]
// CHECK:           (%[[IN:.*]]: f32, %[[INIT:.*]]: f32) {
// CHECK:             %[[ADD:.*]] = arith.addf %[[IN]], %[[INIT]] : f32
// CHECK:             linalg.yield %[[ADD]] : f32
// CHECK:           }
// CHECK-NOT:     linalg.reduce
// CHECK-NOT:     linalg.elemwise_binary
// CHECK:         return %[[REDUCED]] : tensor<64xf32>

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9579">} {
  func.func @triton_sum_2D_dim0(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xf32> to memref<64x64xf32, strided<[64, 1]>>
    %alloc = memref.alloc() : memref<64x64xf32>
    memref.copy %reinterpret_cast, %alloc : memref<64x64xf32, strided<[64, 1]>> to memref<64x64xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32>
    %1 = tensor.empty() : tensor<64x64xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%0, %0 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %3 = tensor.empty() : tensor<64xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<64xf32>) -> tensor<64xf32>
    %reduced = linalg.reduce ins(%2 : tensor<64x64xf32>) outs(%4 : tensor<64xf32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %5 = arith.addf %in, %init : f32
        linalg.yield %5 : f32
      }
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1]>>
    bufferization.materialize_in_destination %reduced in writable %reinterpret_cast_0 : (tensor<64xf32>, memref<64xf32, strided<[1]>>) -> ()
    return
  }
}
