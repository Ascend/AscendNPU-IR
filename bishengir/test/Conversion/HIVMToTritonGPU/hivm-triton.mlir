// RUN: bishengir-opt -convert-hivm-to-tritongpu %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: tt.func @simple_indirect_load_kernel_scope_0(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i64>, %arg2: index, %arg3: index, %arg4: index, %arg5: !tt.ptr<i64>, %arg6: !tt.ptr<i64>, %arg7: index, %arg8: index, %arg9: index, %arg10: !tt.ptr<f32>, %arg11: !tt.ptr<f32>, %arg12: index, %arg13: index, %arg14: index, %arg15: i32, %arg16: !tt.ptr<f32>, %arg17: !tt.ptr<f32>, %arg18: index, %arg19: index, %arg20: index)
// CHECK-NEXT: %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK-NEXT: %1 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<8x!tt.ptr<i64>>
// CHECK-NEXT: %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<i64>>, tensor<8xi32>
// CHECK-NEXT: %3 = tt.load %2 evictionPolicy = evict_first : tensor<8x!tt.ptr<i64>>
// CHECK-NEXT: %4 = tt.splat %arg5 : !tt.ptr<i64> -> tensor<8x!tt.ptr<i64>>
// CHECK-NEXT: %5 = tt.addptr %4, %0 : tensor<8x!tt.ptr<i64>>, tensor<8xi32>
// CHECK-NEXT: tt.store %5, %3 : tensor<8x!tt.ptr<i64>>
// CHECK-NEXT: %6 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK-NEXT: %7 = tt.splat %arg5 : !tt.ptr<i64> -> tensor<8x!tt.ptr<i64>>
// CHECK-NEXT: %8 = tt.addptr %7, %6 : tensor<8x!tt.ptr<i64>>, tensor<8xi32>
// CHECK-NEXT: %9 = tt.load %8 : tensor<8x!tt.ptr<i64>>
// CHECK-NEXT: %10 = tt.splat %arg10 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: %11 = tt.addptr %10, %9 : tensor<8x!tt.ptr<f32>>, tensor<8xi64>
// CHECK-NEXT: %12 = tt.load %11 : tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: %13 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK-NEXT: %14 = tt.splat %arg16 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: %15 = tt.addptr %14, %13 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
// CHECK-NEXT: tt.store %15, %12 : tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: tt.return

module {
  func.func @simple_indirect_load_kernel_scope_0(%arg0: memref<?xi64>, %arg1: memref<8xi64>, %arg2: memref<?xf32>, %arg3: i32, %arg4: memref<8xf32>) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
    hivm.hir.load ins(%reinterpret_cast : memref<8xi64, strided<[1]>>) outs(%arg1 : memref<8xi64>) eviction_policy = <EvictFirst>
    %0 = bufferization.to_tensor %arg1 restrict writable : memref<8xi64>
    %1 = hivm.hir.gather_load ins(%arg2 : memref<?xf32>, %0 : tensor<8xi64>, %arg3 : i32) {cache = 1 : i32, evict = #hivm.evictionpolicy<EvictLast>, isVolatile = false} -> tensor<8xf32>
    hivm.hir.local_store ins(%arg4 : memref<8xf32>, %1 : tensor<8xf32>)
    return
  }
}

// CHECK-LABEL: tt.func @simple_indirect_store_kernel_scope_0(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i64>, %arg2: index, %arg3: index, %arg4: index, %arg5: !tt.ptr<i64>, %arg6: !tt.ptr<i64>, %arg7: index, %arg8: index, %arg9: index, %arg10: !tt.ptr<f32>, %arg11: !tt.ptr<f32>, %arg12: index, %arg13: index, %arg14: index, %arg15: !tt.ptr<f32>, %arg16: !tt.ptr<f32>, %arg17: index, %arg18: index, %arg19: index, %arg20: i32)
// CHECK-NEXT: %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK-NEXT: %1 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<8x!tt.ptr<i64>>
// CHECK-NEXT: %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<i64>>, tensor<8xi32>
// CHECK-NEXT: %3 = tt.load %2 evictionPolicy = evict_first : tensor<8x!tt.ptr<i64>>
// CHECK-NEXT: %4 = tt.splat %arg5 : !tt.ptr<i64> -> tensor<8x!tt.ptr<i64>>
// CHECK-NEXT: %5 = tt.addptr %4, %0 : tensor<8x!tt.ptr<i64>>, tensor<8xi32>
// CHECK-NEXT: tt.store %5, %3 : tensor<8x!tt.ptr<i64>>
// CHECK-NEXT: %6 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK-NEXT: %7 = tt.splat %arg5 : !tt.ptr<i64> -> tensor<8x!tt.ptr<i64>>
// CHECK-NEXT: %8 = tt.addptr %7, %6 : tensor<8x!tt.ptr<i64>>, tensor<8xi32>
// CHECK-NEXT: %9 = tt.load %8 : tensor<8x!tt.ptr<i64>>
// CHECK-NEXT: %10 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK-NEXT: %11 = tt.splat %arg10 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: %12 = tt.addptr %11, %10 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
// CHECK-NEXT: %13 = tt.load %12 : tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: %14 = tt.splat %arg15 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: %15 = tt.addptr %14, %9 : tensor<8x!tt.ptr<f32>>, tensor<8xi64>
// CHECK-NEXT: tt.store %15, %13 : tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: tt.return

module {
  func.func @simple_indirect_store_kernel_scope_0(%arg0: memref<?xi64>, %arg1: memref<8xi64>, %arg2: memref<8xf32>, %arg3: memref<?xf32>, %arg4: i32) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
    hivm.hir.load ins(%reinterpret_cast : memref<8xi64, strided<[1]>>) outs(%arg1 : memref<8xi64>) eviction_policy = <EvictFirst>
    %0 = bufferization.to_tensor %arg1 restrict writable : memref<8xi64>
    %1 = hivm.hir.local_load ins(%arg2 : memref<8xf32>) -> tensor<8xf32>
    hivm.hir.scatter_store ins(%arg3 : memref<?xf32>, %0 : tensor<8xi64>, %1 : tensor<8xf32>, %arg4 : i32) {cache = 1 : i32, evict = #hivm.evictionpolicy<EvictLast>}
    return
  }
}
