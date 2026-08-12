// RUN: bishengir-opt %s "--one-shot-bufferize=allow-return-allocs-from-loops=true allow-unknown-ops=true bufferize-function-boundaries=true function-boundary-type-conversion=identity-layout-map analysis-heuristic=top-down" -split-input-file | FileCheck %s

// Regression for HIVMLoadOpInterface::getBufferType infinite recursion during
// one-shot-bufferize with HIVM pipeline options (membase and regbase).
// CHECK-LABEL: func.func @test_hivm_load_get_buffer_type
// CHECK: hivm.hir.store ins({{.*}} : memref{{.*}}) outs({{.*}} : memref{{.*}})
// CHECK: hivm.hir.load ins({{.*}} : memref{{.*}}) outs({{.*}} : memref{{.*}})
module attributes {hacc.target = #hacc.target<"Ascend910B3">} {
  func.func @test_hivm_load_get_buffer_type(%arg2: memref<?xi8>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %init = tensor.empty() : tensor<64x64xf32>
    scf.for %i = %c0_i32 to %c1_i32 step %c1_i32 iter_args(%arg = %init) -> (tensor<64x64xf32>) : i32 {
      %ws0 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8> to memref<128x64xf32>
      %sv0 = memref.subview %ws0[0, 0] [64, 64] [1, 1] : memref<128x64xf32> to memref<64x64xf32, strided<[64, 1]>>
      %t0 = bufferization.to_tensor %sv0 restrict writable : memref<64x64xf32, strided<[64, 1]>>
      hivm.hir.store ins(%arg : tensor<64x64xf32>) outs(%t0 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %ws1 = memref_ext.alloc_workspace() from %arg2 : from memref<?xi8> to memref<128x64xf32>
      %sv1 = memref.subview %ws1[0, 0] [64, 64] [1, 1] : memref<128x64xf32> to memref<64x64xf32, strided<[64, 1]>>
      %t1 = bufferization.to_tensor %sv1 restrict writable : memref<64x64xf32, strided<[64, 1]>>
      %dst = tensor.empty() : tensor<64x64xf32>
      %loaded = hivm.hir.load ins(%t1 : tensor<64x64xf32>) outs(%dst : tensor<64x64xf32>) -> tensor<64x64xf32>
      scf.yield %loaded : tensor<64x64xf32>
    }
    return
  }
}
