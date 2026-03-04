// RUN: bishengir-compile -target=Ascend950PR_9589 -enable-hfusion-compile=true -enable-triton-kernel-compile -mix-simt -enable-lir-compile=false %s | FileCheck %s
// XFAIL: *
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @gather_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1024_i32 = arith.constant 1024 : i32
    %0 = arith.muli %arg10, %c1024_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%1], sizes: [1024], strides: [1] : memref<?xi64> to memref<1024xi64, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<1024xi64>
    memref.copy %reinterpret_cast, %alloc : memref<1024xi64, strided<[1], offset: ?>> to memref<1024xi64>
    %2 = bufferization.to_tensor %reinterpret_cast restrict writable : memref<1024xi64, strided<[1], offset: ?>>
    %3 = tensor.empty() : tensor<1024xf32>
    %4 = scf.for %arg13 = %c0 to %c1024 step %c1 iter_args(%arg14 = %3) -> (tensor<1024xf32>) {
      %extracted = tensor.extract %2[%arg13] : tensor<1024xi64>
      %5 = arith.index_cast %extracted : i64 to index
      //////////////////////////
      // Problematic:
      // %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%5], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      // %6 = memref.load %reinterpret_cast_1[%c0] : memref<1xf32, strided<[1], offset: ?>>

      // Replace with following:
      %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32>
      %6 = memref.load %reinterpret_cast_1[%5] : memref<1024xf32>
      //////////////////////////
      %inserted = tensor.insert %6 into %arg14[%arg13] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    } {map_for_to_forall,  mapping = [#hivm.block]}
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%1], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %4 in writable %reinterpret_cast_0 : (tensor<1024xf32>, memref<1024xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}

// CHECK: define dso_local void @gather_kernel
// CHECK: define internal simt_entry void @gather_kernel_kernel
