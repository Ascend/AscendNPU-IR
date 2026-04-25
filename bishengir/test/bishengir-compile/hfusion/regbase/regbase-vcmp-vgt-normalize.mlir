// RUN: bishengir-compile -enable-lir-compile=false -enable-hfusion-compile=true -block-dim=1 -target=Ascend310B4  \
// RUN: -enable-triton-kernel-compile=true %s | FileCheck %s

// ----- Case From #2476
// CHECK-LABEL: @triton_gt_outlined_vf_0
module {
  func.func @triton_gt(%arg0: memref<?xi8>, %arg1: memref<?xi32> {tt.divisibility = 16 : i32}, %arg2: memref<?xi32> {tt.divisibility = 16 : i32}, %arg3: memref<?xi32> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {WorkspaceArgIdx = 0 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = arith.muli %arg7, %c1024_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%1], sizes: [1024], strides: [1] : memref<?xi32> to memref<1024xi32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<1024xi32>
    memref.copy %reinterpret_cast, %alloc : memref<1024xi32, strided<[1], offset: ?>> to memref<1024xi32>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<1024xi32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [1024], strides: [1] : memref<?xi32> to memref<1024xi32, strided<[1], offset: ?>>
    %alloc_1 = memref.alloc() : memref<1024xi32>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<1024xi32, strided<[1], offset: ?>> to memref<1024xi32>
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<1024xi32>
    // CHECK: llvm.hivm.vcmp.gt.s.z
    %4 = arith.cmpi sgt, %2, %3 : tensor<1024xi32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%1], sizes: [1024], strides: [1] : memref<?xi32> to memref<1024xi32, strided<[1], offset: ?>>
    %5 = arith.extui %4 : tensor<1024xi1> to tensor<1024xi32>
    bufferization.materialize_in_destination %5 in writable %reinterpret_cast_2 : (tensor<1024xi32>, memref<1024xi32, strided<[1], offset: ?>>) -> ()
    return
  }
}
