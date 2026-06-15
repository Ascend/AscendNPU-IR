// REQUIRES: enable-lir-compile
// RUN: bishengir-compile -enable-lir-compile=false --enable-tree-reduce-mode=all -enable-hfusion-compile=true -enable-hivm-compile=true --enable-triton-kernel-compile=true --target=Ascend950PR_9589 %s

// CHECK-LABEL: func.func @triton_sum_2D_dim0_fused_0_outlined_vf_0
// CHECK-NOT: vector.multi_reduction <add>
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     arith.divui
// CHECK:     scf.for
// CHECK:       vector.create_mask
// CHECK:       scf.for
// CHECK:         vector.mask {{.*}} { vector.transfer_read {{.*}} : tensor<1x?xf32>, vector<1x64xf32> }
// CHECK:         arith.addf {{.*}} : vector<1x64xf32>
// CHECK:         vector.mask {{.*}} { vector.transfer_write {{.*}} : vector<1x64xf32>, tensor<1x?xf32> }
// CHECK:   scf.for
// CHECK:     vector.create_mask {{.*}} : vector<1x64xi1>
// CHECK:     vector.mask {{.*}} { vector.transfer_read {{.*}} : tensor<1x?xf32>, vector<1x64xf32> }
// CHECK:     arith.addf {{.*}} : vector<1x64xf32>
// CHECK:     vector.shape_cast {{.*}} : vector<1x64xf32> to vector<64xf32>
// CHECK:     vector.create_mask {{.*}} : vector<64xi1>
// CHECK:     vector.mask {{.*}} { vector.transfer_read {{.*}} : tensor<?xf32>, vector<64xf32> }
// CHECK:     arith.addf {{.*}} : vector<64xf32>
// CHECK:     vector.mask {{.*}} { vector.transfer_write {{.*}} : vector<64xf32>, tensor<?xf32> }
// CHECK: return

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @triton_sum_2D_dim0(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<?xf32> to memref<32x32xf32, strided<[32, 1]>>
    
    %alloc = memref.alloc() : memref<32x32xf32>
    memref.copy %reinterpret_cast, %alloc : memref<32x32xf32, strided<[32, 1]>> to memref<32x32xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf32>
    %1 = tensor.empty() : tensor<32xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<32xf32>) -> tensor<32xf32>
    
    %reduced = linalg.reduce ins(%0 : tensor<32x32xf32>) outs(%2 : tensor<32xf32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %3 = arith.addf %in, %init : f32
        linalg.yield %3 : f32
      }
      
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1]>>
    bufferization.materialize_in_destination %reduced in writable %reinterpret_cast_0 : (tensor<32xf32>, memref<32xf32, strided<[1]>>) -> ()
    
    return
  }
}
