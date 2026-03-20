// RUN: rm -rf %t.dir && mkdir -p %t.dir && cd %t.dir && bishengir-compile -enable-hfusion-compile=true -target=Ascend910_9589 -enable-triton-kernel-compile=true %s -o %t

// CHECK: LLVMDialectModule
module {
  func.func @triton_mod(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [92, 2, 23, 10], strides: [460, 230, 10, 1] : memref<?xi16> to memref<92x2x23x10xi16, strided<[460, 230, 10, 1]>>
    %alloc = memref.alloc() : memref<92x2x23x10xi16>
    memref.copy %reinterpret_cast, %alloc : memref<92x2x23x10xi16, strided<[460, 230, 10, 1]>> to memref<92x2x23x10xi16>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<92x2x23x10xi16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [92, 2, 23, 10], strides: [460, 230, 10, 1] : memref<?xi16> to memref<92x2x23x10xi16, strided<[460, 230, 10, 1]>>
    %alloc_1 = memref.alloc() : memref<92x2x23x10xi16>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<92x2x23x10xi16, strided<[460, 230, 10, 1]>> to memref<92x2x23x10xi16>
    %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<92x2x23x10xi16>
    %2 = arith.remsi %0, %1 : tensor<92x2x23x10xi16>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [92, 2, 23, 10], strides: [460, 230, 10, 1] : memref<?xi16> to memref<92x2x23x10xi16, strided<[460, 230, 10, 1]>>
    bufferization.materialize_in_destination %2 in writable %reinterpret_cast_2 : (tensor<92x2x23x10xi16>, memref<92x2x23x10xi16, strided<[460, 230, 10, 1]>>) -> ()
    return
  }
}
