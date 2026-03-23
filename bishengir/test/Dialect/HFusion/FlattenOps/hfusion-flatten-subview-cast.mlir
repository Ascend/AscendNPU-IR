// RUN: bishengir-opt %s -hfusion-flatten-ops="flatten-mode=tidy register-based=true multi-dynamic-shape=false" -split-input-file | FileCheck %s

// -----

// Verify that a 3-D subview -> memref.cast chain is correctly flattened
// without producing a cast-incompatible pair.  Before the fix, the subview
// was collapsed to 2-D but memref.cast still expected a 3-D operand,
// causing "'memref.cast' op operand type ... and result type ... are cast
// incompatible".

// CHECK-LABEL: func.func @test_subview_cast_3d
// CHECK-DAG:     memref.collapse_shape {{.*}} {{\[}}[0], [1, 2]{{\]}} : memref<30x40x64xi64, {{.*}}> into memref<30x2560xi64
// CHECK-DAG:     memref.collapse_shape {{.*}} {{\[}}[0], [1, 2]{{\]}} : memref<30x40x64xi64> into memref<30x2560xi64>
// CHECK:         memref.copy
// CHECK:         memref.subview {{.*}} : memref<30x2560xi64> to memref<15x1280xi64
// CHECK:         memref.cast {{.*}} : memref<15x1280xi64, {{.*}}> to memref<15x1280xi64
// CHECK:         bufferization.to_tensor {{.*}} : memref<15x1280xi64
// CHECK:         memref.collapse_shape {{.*}} {{\[}}[0], [1, 2]{{\]}} : memref<15x20x64xi64, {{.*}}> into memref<15x1280xi64
// CHECK:         bufferization.materialize_in_destination
func.func @test_subview_cast_3d(
    %arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
    %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
    %arg2: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
    %arg3: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
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
      to offset: [0], sizes: [30, 40, 64], strides: [2560, 64, 1]
      : memref<?xi64> to memref<30x40x64xi64, strided<[2560, 64, 1]>>
  %alloc = memref.alloc() : memref<30x40x64xi64>
  memref.copy %reinterpret_cast, %alloc
      : memref<30x40x64xi64, strided<[2560, 64, 1]>> to memref<30x40x64xi64>
  %subview = memref.subview %alloc[2, 3, 0] [15, 20, 64] [1, 1, 1]
      : memref<30x40x64xi64>
      to memref<15x20x64xi64, strided<[2560, 64, 1], offset: 5312>>
  %cast = memref.cast %subview
      : memref<15x20x64xi64, strided<[2560, 64, 1], offset: 5312>>
      to memref<15x20x64xi64, strided<[2560, 64, 1], offset: ?>>
  %0 = bufferization.to_tensor %cast restrict writable
      : memref<15x20x64xi64, strided<[2560, 64, 1], offset: ?>>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3
      to offset: [0], sizes: [15, 20, 64], strides: [1280, 64, 1]
      : memref<?xi64> to memref<15x20x64xi64, strided<[1280, 64, 1]>>
  bufferization.materialize_in_destination %0 in writable %reinterpret_cast_0
      : (tensor<15x20x64xi64>,
         memref<15x20x64xi64, strided<[1280, 64, 1]>>) -> ()
  return
}
