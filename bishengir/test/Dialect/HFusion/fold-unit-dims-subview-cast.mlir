// RUN: bishengir-opt -hfusion-fold-unit-dims -split-input-file %s | FileCheck %s

// -----

// CHECK-LABEL: func.func @test_subview_cast_8d
// CHECK:       memref.subview
// CHECK:       memref.cast
// CHECK:       memref.memory_space_cast
// CHECK:       bufferization.to_tensor
func.func @test_subview_cast_8d(
    %arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
    %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
    %arg2: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
    %arg3: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
    %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32
) attributes {
    SyncBlockLockArgIdx = 0 : i64,
    WorkspaceArgIdx = 1 : i64,
    hacc.entry,
    hacc.function_kind = #hacc.function_kind<DEVICE>,
    mix_mode = "aiv",
    parallel_mode = "simd"
} {
  %reinterpret_cast = memref.reinterpret_cast %arg2
      to offset: [0], sizes: [1, 2, 3, 4, 5, 6, 7, 8],
      strides: [40320, 20160, 6720, 1680, 336, 56, 8, 1]
      : memref<?xi32> to memref<1x2x3x4x5x6x7x8xi32, strided<[40320, 20160, 6720, 1680, 336, 56, 8, 1]>>
  %alloc = memref.alloc() : memref<1x2x3x4x5x6x7x8xi32>
  memref.copy %reinterpret_cast, %alloc
      : memref<1x2x3x4x5x6x7x8xi32, strided<[40320, 20160, 6720, 1680, 336, 56, 8, 1]>>
      to memref<1x2x3x4x5x6x7x8xi32>
  %memspacecast = memref.memory_space_cast %alloc
      : memref<1x2x3x4x5x6x7x8xi32> to memref<1x2x3x4x5x6x7x8xi32, #hivm.address_space<ub>>
  %subview = memref.subview %memspacecast[0, 0, 0, 0, 0, 0, 0, 0] [1, 1, 1, 1, 1, 1, 1, 2] [1, 1, 1, 1, 1, 1, 1, 1]
      : memref<1x2x3x4x5x6x7x8xi32, #hivm.address_space<ub>>
      to memref<1x1x1x1x1x1x1x2xi32, strided<[40320, 20160, 6720, 1680, 336, 56, 8, 1]>, #hivm.address_space<ub>>
  %cast = memref.cast %subview
      : memref<1x1x1x1x1x1x1x2xi32, strided<[40320, 20160, 6720, 1680, 336, 56, 8, 1]>, #hivm.address_space<ub>>
      to memref<1x1x1x1x1x1x1x2xi32, strided<[40320, 20160, 6720, 1680, 336, 56, 8, 1], offset: ?>, #hivm.address_space<ub>>
  %memspacecast_0 = memref.memory_space_cast %cast
      : memref<1x1x1x1x1x1x1x2xi32, strided<[40320, 20160, 6720, 1680, 336, 56, 8, 1], offset: ?>, #hivm.address_space<ub>>
      to memref<1x1x1x1x1x1x1x2xi32, strided<[40320, 20160, 6720, 1680, 336, 56, 8, 1], offset: ?>>
  %0 = bufferization.to_tensor %memspacecast_0 restrict writable
      : memref<1x1x1x1x1x1x1x2xi32, strided<[40320, 20160, 6720, 1680, 336, 56, 8, 1], offset: ?>>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg3
      to offset: [0], sizes: [1, 1, 1, 1, 1, 1, 1, 2],
      strides: [2, 2, 2, 2, 2, 2, 2, 1]
      : memref<?xi32> to memref<1x1x1x1x1x1x1x2xi32, strided<[2, 2, 2, 2, 2, 2, 2, 1]>>
  bufferization.materialize_in_destination %0 in writable %reinterpret_cast_1
      : (tensor<1x1x1x1x1x1x1x2xi32>,
         memref<1x1x1x1x1x1x1x2xi32, strided<[2, 2, 2, 2, 2, 2, 2, 1]>>) -> ()
  return
}
