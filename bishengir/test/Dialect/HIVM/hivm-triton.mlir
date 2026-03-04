// RUN: bishengir-opt -convert-hivm-to-tritongpu %s -split-input-file -verify-diagnostics | FileCheck %s

//  CHECK: tt.func @simt_vf(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>, %arg3: !tt.ptr<f32>) {
//  CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
//  CHECK-NEXT:     %0 = tt.make_range {end = 8192 : i32, start = 0 : i32} : tensor<8192xi32>
//  CHECK-NEXT:     %1 = tt.reshape %0 : tensor<8192xi32> -> tensor<16x512xi32>
//  CHECK-NEXT:     %2 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<16x512x!tt.ptr<i32>>
//  CHECK-NEXT:     %3 = tt.addptr %2, %1 : tensor<16x512x!tt.ptr<i32>>, tensor<16x512xi32>
//  CHECK-NEXT:     %4 = tt.load %3 evictionPolicy = evict_first : tensor<16x512x!tt.ptr<i32>>
//  CHECK-NEXT:     %5 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x512x!tt.ptr<i32>>
//  CHECK-NEXT:     %6 = tt.addptr %5, %1 : tensor<16x512x!tt.ptr<i32>>, tensor<16x512xi32>
//  CHECK-NEXT:     tt.store %6, %4 : tensor<16x512x!tt.ptr<i32>>
//  CHECK-NEXT:     %7 = tt.make_range {end = 8192 : i32, start = 0 : i32} : tensor<8192xi32>
//  CHECK-NEXT:     %8 = tt.reshape %7 : tensor<8192xi32> -> tensor<16x512xi32>
//  CHECK-NEXT:     %9 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x512x!tt.ptr<i32>>
//  CHECK-NEXT:     %10 = tt.addptr %9, %8 : tensor<16x512x!tt.ptr<i32>>, tensor<16x512xi32>
//  CHECK-NEXT:     %11 = tt.load %10 : tensor<16x512x!tt.ptr<i32>>
//  CHECK-NEXT:     %12 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x512x!tt.ptr<f32>>
//  CHECK-NEXT:     %13 = tt.addptr %12, %11 : tensor<16x512x!tt.ptr<f32>>, tensor<16x512xi32>
//  CHECK-NEXT:     %14 = tt.load %13 : tensor<16x512x!tt.ptr<f32>>
//  CHECK-NEXT:     %15 = tt.make_range {end = 8192 : i32, start = 0 : i32} : tensor<8192xi32>
//  CHECK-NEXT:     %16 = tt.reshape %15 : tensor<8192xi32> -> tensor<16x512xi32>
//  CHECK-NEXT:     %17 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<16x512x!tt.ptr<f32>>
//  CHECK-NEXT:     %18 = tt.addptr %17, %16 : tensor<16x512x!tt.ptr<f32>>, tensor<16x512xi32>
//  CHECK-NEXT:     tt.store %18, %14 : tensor<16x512x!tt.ptr<f32>>
//  CHECK-NEXT:     tt.return


module {
  func.func @simt_vf(%arg0 : memref<?xf32>, %arg1: memref<16x512xi32>, %arg2: memref<16x512xi32>, %arg3: memref<16x512xf32>) {
    %c1_i64 = arith.constant 1 : i64
    hivm.hir.load ins(%arg2 : memref<16x512xi32>) outs(%arg1 : memref<16x512xi32>) eviction_policy = <EvictFirst>
    %0 = bufferization.to_tensor %arg1 restrict writable : memref<16x512xi32>
    %1 = hivm.hir.gather_load ins(%arg0 : memref<?xf32>, %0: tensor<16x512xi32>, %c1_i64: i64)  -> tensor<16x512xf32>
    hivm.hir.store ins(%1 : tensor<16x512xf32>) outs(%arg3 : memref<16x512xf32>)
    func.return
  }
}


// -----

// RUN: bishengir-opt -convert-hivm-to-tritongpu %s -split-input-file -verify-diagnostics | FileCheck %s
 
//  CHECK: tt.func @simt_vf(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i64>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>, %arg4: i32)
//  CHECK-NEXT:  %0 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32>
//  CHECK-NEXT:  %1 = tt.reshape %0 : tensor<4096xi32> -> tensor<16x256xi32>
//  CHECK-NEXT:  %2 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<16x256x!tt.ptr<i64>>
//  CHECK-NEXT:  %3 = tt.addptr %2, %1 : tensor<16x256x!tt.ptr<i64>>, tensor<16x256xi32>
//  CHECK-NEXT:  %4 = tt.load %3 evictionPolicy = evict_first : tensor<16x256x!tt.ptr<i64>>
//  CHECK-NEXT:  %5 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<16x256x!tt.ptr<i64>>
//  CHECK-NEXT:  %6 = tt.addptr %5, %1 : tensor<16x256x!tt.ptr<i64>>, tensor<16x256xi32>
//  CHECK-NEXT:  tt.store %6, %4 : tensor<16x256x!tt.ptr<i64>>
//  CHECK-NEXT:  %7 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32>
//  CHECK-NEXT:  %8 = tt.reshape %7 : tensor<4096xi32> -> tensor<16x256xi32>
//  CHECK-NEXT:  %9 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<16x256x!tt.ptr<i64>>
//  CHECK-NEXT:  %10 = tt.addptr %9, %8 : tensor<16x256x!tt.ptr<i64>>, tensor<16x256xi32>
//  CHECK-NEXT:  %11 = tt.load %10 : tensor<16x256x!tt.ptr<i64>>
//  CHECK-NEXT:  %12 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32>
//  CHECK-NEXT:  %13 = tt.reshape %12 : tensor<4096xi32> -> tensor<16x256xi32>
//  CHECK-NEXT:  %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<16x256x!tt.ptr<f32>>
//  CHECK-NEXT:  %15 = tt.addptr %14, %13 : tensor<16x256x!tt.ptr<f32>>, tensor<16x256xi32>
//  CHECK-NEXT:  %16 = tt.load %15 : tensor<16x256x!tt.ptr<f32>>
//  CHECK-NEXT:  %17 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<16x256x!tt.ptr<f32>>
//  CHECK-NEXT:  %18 = tt.addptr %17, %11 : tensor<16x256x!tt.ptr<f32>>, tensor<16x256xi64>
//  CHECK-NEXT:  tt.store %18, %16 : tensor<16x256x!tt.ptr<f32>>
//  CHECK-NEXT:  tt.return

module {
  func.func @simt_vf(%arg0: memref<?xi64>, %arg1: memref<16x256xi64>, %arg2: memref<16x256xf32>, %arg3: memref<?xf32>, %arg4: i32) {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16,256], strides: [256,1] : memref<?xi64> to memref<16x256xi64, strided<[256,1]>>
    hivm.hir.load ins(%reinterpret_cast : memref<16x256xi64, strided<[256,1]>>) outs(%arg1 : memref<16x256xi64>) eviction_policy = <EvictFirst>
    %0 = bufferization.to_tensor %arg1 restrict writable : memref<16x256xi64>
    %1 = bufferization.to_tensor %arg2 : memref<16x256xf32>
    hivm.hir.scatter_store ins(%arg3 : memref<?xf32>, %0 : tensor<16x256xi64>, %1 : tensor<16x256xf32>, %arg4 : i32) {cache = 1 : i32, evict = #hivm.evictionpolicy<EvictLast>}
    return
  }
}
