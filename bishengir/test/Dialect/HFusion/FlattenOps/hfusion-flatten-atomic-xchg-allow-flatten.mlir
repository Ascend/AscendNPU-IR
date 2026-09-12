// RUN: bishengir-opt --hfusion-flatten-ops="flatten-mode=tidy register-based=true skip-scope=true multi-dynamic-shape=false" %s | FileCheck %s

// Regression: software atomic_xchg scopes marked hivm.allow_flatten must not
// block FlattenOps. Collapsing the unit leading dim (and further) avoids the
// padded 5D UB alloc that overflows on Ascend950.

// CHECK-LABEL: func.func @triton_atomic_xchg_5D_allow_flatten
// FlattenOps must collapse multi-rank copies (incl. inside allow_flatten scope).
// CHECK: scope.scope
// CHECK-DAG: memref.copy %{{.*}}, %{{.*}} : memref<24576xf8E4M3FN{{.*}}> to memref<24576xf8E4M3FN{{.*}}>
// CHECK: } {hivm.allow_flatten, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9599">} {
  func.func @triton_atomic_xchg_5D_allow_flatten(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf8E4M3FN> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf8E4M3FN> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg4: memref<?xf8E4M3FN> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %c6144 = arith.constant 6144 : index
    %c48 = arith.constant 48 : index
    %c24576_i32 = arith.constant 24576 : i32
    %c4_i32 = arith.constant 4 : i32
    %c128_i32 = arith.constant 128 : i32
    %0 = arith.muli %arg9, %c4_i32 : i32
    %1 = arith.muli %arg10, %c128_i32 : i32
    %2 = arith.muli %arg8, %c24576_i32 : i32
    %3 = arith.index_cast %1 : i32 to index
    %4 = arith.muli %3, %c48 : index
    %5 = arith.index_cast %0 : i32 to index
    %6 = arith.muli %5, %c6144 : index
    %7 = arith.index_cast %2 : i32 to index
    %8 = arith.addi %7, %6 : index
    %9 = arith.addi %8, %4 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%9], sizes: [4, 128, 16, 3], strides: [6144, 48, 3, 1] : memref<?xf8E4M3FN> to memref<4x128x16x3xf8E4M3FN, strided<[6144, 48, 3, 1], offset: ?>>
    %alloc = memref.alloc() : memref<4x128x16x3xf8E4M3FN>
    %expand_shape = memref.expand_shape %alloc [[0, 1], [2], [3], [4]] output_shape [1, 4, 128, 16, 3] : memref<4x128x16x3xf8E4M3FN> into memref<1x4x128x16x3xf8E4M3FN>
    memref.copy %reinterpret_cast, %alloc : memref<4x128x16x3xf8E4M3FN, strided<[6144, 48, 3, 1], offset: ?>> to memref<4x128x16x3xf8E4M3FN>
    %10 = bufferization.to_tensor %alloc restrict writable : memref<4x128x16x3xf8E4M3FN>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [4, 128, 16, 3], strides: [6144, 48, 3, 1] : memref<?xf8E4M3FN> to memref<4x128x16x3xf8E4M3FN, strided<[6144, 48, 3, 1], offset: ?>>
    %expand_shape_1 = memref.expand_shape %reinterpret_cast_0 [[0, 1], [2], [3], [4]] output_shape [1, 4, 128, 16, 3] : memref<4x128x16x3xf8E4M3FN, strided<[6144, 48, 3, 1], offset: ?>> into memref<1x4x128x16x3xf8E4M3FN, strided<[24576, 6144, 48, 3, 1], offset: ?>>
    scope.scope : () -> () {
      %alloc_3 = memref.alloc() : memref<1x4x128x16x3xf8E4M3FN>
      %11 = hivm.hir.create_sync_block_lock : memref<1xi64>
      hivm.hir.sync_block_lock lock_var(%11 : memref<1xi64>)
      memref.copy %expand_shape_1, %alloc_3 : memref<1x4x128x16x3xf8E4M3FN, strided<[24576, 6144, 48, 3, 1], offset: ?>> to memref<1x4x128x16x3xf8E4M3FN>
      memref.copy %expand_shape, %expand_shape_1 : memref<1x4x128x16x3xf8E4M3FN> to memref<1x4x128x16x3xf8E4M3FN, strided<[24576, 6144, 48, 3, 1], offset: ?>>
      memref.copy %alloc_3, %expand_shape : memref<1x4x128x16x3xf8E4M3FN> to memref<1x4x128x16x3xf8E4M3FN>
      hivm.hir.sync_block_unlock lock_var(%11 : memref<1xi64>)
      scope.return
    } {hivm.allow_flatten, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%9], sizes: [4, 128, 16, 3], strides: [6144, 48, 3, 1] : memref<?xf8E4M3FN> to memref<4x128x16x3xf8E4M3FN, strided<[6144, 48, 3, 1], offset: ?>>
    bufferization.materialize_in_destination %10 in writable %reinterpret_cast_2 : (tensor<4x128x16x3xf8E4M3FN>, memref<4x128x16x3xf8E4M3FN, strided<[6144, 48, 3, 1], offset: ?>>) -> ()
    return
  }
}
