// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="fusion-mode=max-parallel" --split-input-file %s | FileCheck %s

// Stage 0 special pattern: predicate (producer) + select (consumer) co-location.
// All i1-producing ops (compare, isinf, isnan, isfinite) must be fused into the
// same group as the select that consumes their result, bypassing the cost model.

// CHECK-LABEL: func.func private @compare_select_kernel_fused_0(
// CHECK: hfusion.compare
// CHECK-NOT: func.func
// CHECK: hfusion.select
// CHECK: return

// CHECK-LABEL: func.func @compare_select_kernel(
// CHECK: call @compare_select_kernel_fused_0
// CHECK-NOT: call @compare_select_kernel_fused_{{[1-9]}}

module {
  func.func @compare_select_kernel(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1]>>
    %alloc = memref.alloc() : memref<8xf32>
    memref.copy %reinterpret_cast, %alloc : memref<8xf32, strided<[1]>> to memref<8xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<8xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1]>>
    %alloc_0 = memref.alloc() : memref<8xf32>
    memref.copy %reinterpret_cast_0, %alloc_0 : memref<8xf32, strided<[1]>> to memref<8xf32>
    %1 = bufferization.to_tensor %alloc_0 restrict writable : memref<8xf32>
    %2 = tensor.empty() : tensor<8xi1>
    %3 = tensor.empty() : tensor<8xf32>
    %4 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%0, %1 : tensor<8xf32>, tensor<8xf32>) outs(%2 : tensor<8xi1>) -> tensor<8xi1>
    %5 = hfusion.select ins(%4, %0, %1 : tensor<8xi1>, tensor<8xf32>, tensor<8xf32>) outs(%3 : tensor<8xf32>) -> tensor<8xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1]>>
    bufferization.materialize_in_destination %5 in writable %reinterpret_cast_1 : (tensor<8xf32>, memref<8xf32, strided<[1]>>) -> ()
    return
  }
}

// -----

// Chained compare->select->broadcast->compare->select (from rmsnorm kernel)

// CHECK-LABEL: func.func private @chained_predicate_select_kernel_fused_0(
// CHECK: hfusion.compare
// CHECK-NOT: func.func
// CHECK: hfusion.select
// CHECK: hfusion.compare
// CHECK-NOT: func.func
// CHECK: hfusion.select
// CHECK: hfusion.compare
// CHECK-NOT: func.func
// CHECK: hfusion.select
// CHECK: hfusion.compare
// CHECK-NOT: func.func
// CHECK: hfusion.select
// CHECK: return

// CHECK-LABEL: func.func @chained_predicate_select_kernel(
// CHECK: call @chained_predicate_select_kernel_fused_0
// CHECK-NOT: call @chained_predicate_select_kernel_fused_{{[1-9]}}

module {
  func.func @chained_predicate_select_kernel(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant -1.000000e+00 : f16
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1]>>
    %alloc = memref.alloc() : memref<8xf32>
    memref.copy %reinterpret_cast, %alloc : memref<8xf32, strided<[1]>> to memref<8xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<8xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1]>>
    %alloc_0 = memref.alloc() : memref<8xf32>
    memref.copy %reinterpret_cast_0, %alloc_0 : memref<8xf32, strided<[1]>> to memref<8xf32>
    %1 = bufferization.to_tensor %alloc_0 restrict writable : memref<8xf32>
    %2 = tensor.empty() : tensor<8xi32>
    %3 = tensor.empty() : tensor<8xi1>
    %4 = tensor.empty() : tensor<8xf16>
    %5 = tensor.empty() : tensor<8x16xf16>
    %6 = tensor.empty() : tensor<8x16xi1>
    %7 = tensor.empty() : tensor<8x16xf32>
    %8 = linalg.fill ins(%arg4 : i32) outs(%2 : tensor<8xi32>) -> tensor<8xi32>
    %9 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%8, %c0_i32 : tensor<8xi32>, i32) outs(%3 : tensor<8xi1>) -> tensor<8xi1>
    %10 = hfusion.select ins(%9, %cst, %cst_0 : tensor<8xi1>, f16, f16) outs(%4 : tensor<8xf16>) -> tensor<8xf16>
    %broadcasted = linalg.broadcast ins(%10 : tensor<8xf16>) outs(%5 : tensor<8x16xf16>) dimensions = [1]
    %11 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%broadcasted, %cst_0 : tensor<8x16xf16>, f16) outs(%6 : tensor<8x16xi1>) -> tensor<8x16xi1>
    %broadcasted_0 = linalg.broadcast ins(%0 : tensor<8xf32>) outs(%7 : tensor<8x16xf32>) dimensions = [1]
    %broadcasted_1 = linalg.broadcast ins(%1 : tensor<8xf32>) outs(%7 : tensor<8x16xf32>) dimensions = [1]
    %12 = hfusion.select ins(%11, %broadcasted_1, %broadcasted_0 : tensor<8x16xi1>, tensor<8x16xf32>, tensor<8x16xf32>) outs(%7 : tensor<8x16xf32>) -> tensor<8x16xf32>
    %13 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%8, %c1_i32 : tensor<8xi32>, i32) outs(%3 : tensor<8xi1>) -> tensor<8xi1>
    %14 = hfusion.select ins(%13, %cst, %cst_0 : tensor<8xi1>, f16, f16) outs(%4 : tensor<8xf16>) -> tensor<8xf16>
    %broadcasted_2 = linalg.broadcast ins(%14 : tensor<8xf16>) outs(%5 : tensor<8x16xf16>) dimensions = [1]
    %15 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%broadcasted_2, %cst_0 : tensor<8x16xf16>, f16) outs(%6 : tensor<8x16xi1>) -> tensor<8x16xi1>
    %16 = hfusion.select ins(%15, %broadcasted_0, %12 : tensor<8x16xi1>, tensor<8x16xf32>, tensor<8x16xf32>) outs(%7 : tensor<8x16xf32>) -> tensor<8x16xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8, 16], strides: [16, 1] : memref<?xf32> to memref<8x16xf32, strided<[16, 1]>>
    bufferization.materialize_in_destination %16 in writable %reinterpret_cast_1 : (tensor<8x16xf32>, memref<8x16xf32, strided<[16, 1]>>) -> ()
    return
  }
}
