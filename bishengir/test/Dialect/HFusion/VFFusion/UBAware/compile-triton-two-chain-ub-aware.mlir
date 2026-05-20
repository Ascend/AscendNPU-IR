// REQUIRES: hivmc-a5
//
// Full-pipeline test: Two independent chains (a*b, c*d) from Triton kernel.
// Without ub-aware-op mode, both chains are merged into 1 VF that overflows UB.
// With ub-aware-op mode, chains are split into 2 VFs that each fit within UB.
//
// RUN: bishengir-compile %s -target=Ascend950PR_9589 -enable-hfusion-compile=true -enable-triton-kernel-compile=true -enable-vf-fusion -vf-fusion-mode=ub-aware-op -o %t

#loc = loc("test_two_chain.py":1:0)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @two_chain_kernel(%arg0: memref<?xi8> loc("test_two_chain.py":1:0), %arg1: memref<?xi8> loc("test_two_chain.py":1:0), %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} loc("a_ptr"(#loc)), %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} loc("b_ptr"(#loc)), %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} loc("c_ptr"(#loc)), %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} loc("d_ptr"(#loc)), %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32} loc("out1_ptr"(#loc)), %arg7: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32} loc("out2_ptr"(#loc)), %arg8: i32 {tt.divisibility = 16 : i32} loc("stride"(#loc)), %arg9: i32 loc("test_two_chain.py":1:0), %arg10: i32 loc("test_two_chain.py":1:0), %arg11: i32 loc("test_two_chain.py":1:0), %arg12: i32 loc("test_two_chain.py":1:0), %arg13: i32 loc("test_two_chain.py":1:0), %arg14: i32 loc("test_two_chain.py":1:0)) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = arith.muli %arg12, %arg8 : i32
    %1 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [16384], strides: [1] : memref<?xf32> to memref<16384xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<16384xf32>
    memref.copy %reinterpret_cast, %alloc : memref<16384xf32, strided<[1], offset: ?>> to memref<16384xf32>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<16384xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%1], sizes: [16384], strides: [1] : memref<?xf32> to memref<16384xf32, strided<[1], offset: ?>>
    %alloc_1 = memref.alloc() : memref<16384xf32>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<16384xf32, strided<[1], offset: ?>> to memref<16384xf32>
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<16384xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%1], sizes: [16384], strides: [1] : memref<?xf32> to memref<16384xf32, strided<[1], offset: ?>>
    %alloc_3 = memref.alloc() : memref<16384xf32>
    memref.copy %reinterpret_cast_2, %alloc_3 : memref<16384xf32, strided<[1], offset: ?>> to memref<16384xf32>
    %4 = bufferization.to_tensor %alloc_3 restrict writable : memref<16384xf32>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%1], sizes: [16384], strides: [1] : memref<?xf32> to memref<16384xf32, strided<[1], offset: ?>>
    %alloc_5 = memref.alloc() : memref<16384xf32>
    memref.copy %reinterpret_cast_4, %alloc_5 : memref<16384xf32, strided<[1], offset: ?>> to memref<16384xf32>
    %5 = bufferization.to_tensor %alloc_5 restrict writable : memref<16384xf32>
    %6 = arith.mulf %2, %3 : tensor<16384xf32>
    %7 = arith.mulf %4, %5 : tensor<16384xf32>
    %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%1], sizes: [16384], strides: [1] : memref<?xf32> to memref<16384xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %6 in writable %reinterpret_cast_6 : (tensor<16384xf32>, memref<16384xf32, strided<[1], offset: ?>>) -> ()
    %reinterpret_cast_7 = memref.reinterpret_cast %arg7 to offset: [%1], sizes: [16384], strides: [1] : memref<?xf32> to memref<16384xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %7 in writable %reinterpret_cast_7 : (tensor<16384xf32>, memref<16384xf32, strided<[1], offset: ?>>) -> ()
    return
  }
} // module
