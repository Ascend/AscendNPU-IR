// REQUIRES: execution-engine
// RUN: bishengir-opt %s \
// RUN:   --hfusion-decompose="hfusion-decompose-phase=before-lower-to-loops" \
// RUN:   --convert-linalg-to-loops \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries" \
// RUN:   --convert-scf-to-cf \
// RUN:   --convert-arith-to-llvm \
// RUN:   --finalize-memref-to-llvm \
// RUN:   --convert-func-to-llvm \
// RUN:   --reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

// CHECK: [1, 2, 3, 4]
// CHECK: [5, -1, 0, 2]
// CHECK: [1, 3, 6, 10]
// CHECK: [5, 4, 4, 6]
// CHECK: [6, 1, 3, 6]
// CHECK: [5, -1, 0, 2]

module {
  func.func @cumsum_kernel_2d_dim1(%arg0: memref<2x4xf32>) -> tensor<2x4xf32> {
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<2x4xf32>
    %1 = hfusion.cumsum %0 : tensor<2x4xf32> cum_dims = [1] reverse = false -> tensor<2x4xf32>
    return %1 : tensor<2x4xf32>
  }

  func.func @cumsum_kernel_2d_dim0_reverse(%arg0: memref<2x4xf32>) -> tensor<2x4xf32> {
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<2x4xf32>
    %1 = hfusion.cumsum %0 : tensor<2x4xf32> cum_dims = [0] reverse = true -> tensor<2x4xf32>
    return %1 : tensor<2x4xf32>
  }

  func.func @main() {
    %m = memref.alloc() : memref<2x4xf32>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %f_n1 = arith.constant -1.0 : f32
    %f0 = arith.constant 0.0 : f32
    %f1 = arith.constant 1.0 : f32
    %f2 = arith.constant 2.0 : f32
    %f3 = arith.constant 3.0 : f32
    %f4 = arith.constant 4.0 : f32
    %f5 = arith.constant 5.0 : f32

    memref.store %f1, %m[%c0, %c0] : memref<2x4xf32>
    memref.store %f2, %m[%c0, %c1] : memref<2x4xf32>
    memref.store %f3, %m[%c0, %c2] : memref<2x4xf32>
    memref.store %f4, %m[%c0, %c3] : memref<2x4xf32>

    memref.store %f5, %m[%c1, %c0] : memref<2x4xf32>
    memref.store %f_n1, %m[%c1, %c1] : memref<2x4xf32>
    memref.store %f0, %m[%c1, %c2] : memref<2x4xf32>
    memref.store %f2, %m[%c1, %c3] : memref<2x4xf32>

    %unranked_orig = memref.cast %m : memref<2x4xf32> to memref<*xf32>
    func.call @printMemrefF32(%unranked_orig) : (memref<*xf32>) -> ()

    %dim1_tensor = func.call @cumsum_kernel_2d_dim1(%m) : (memref<2x4xf32>) -> tensor<2x4xf32>
    %dim1_memref = bufferization.to_memref %dim1_tensor : memref<2x4xf32>
    %unranked_dim1 = memref.cast %dim1_memref : memref<2x4xf32> to memref<*xf32>
    func.call @printMemrefF32(%unranked_dim1) : (memref<*xf32>) -> ()

    %reverse_tensor = func.call @cumsum_kernel_2d_dim0_reverse(%m) : (memref<2x4xf32>) -> tensor<2x4xf32>
    %reverse_memref = bufferization.to_memref %reverse_tensor : memref<2x4xf32>
    %unranked_reverse = memref.cast %reverse_memref : memref<2x4xf32> to memref<*xf32>
    func.call @printMemrefF32(%unranked_reverse) : (memref<*xf32>) -> ()

    memref.dealloc %m : memref<2x4xf32>
    return
  }

  func.func private @printMemrefF32(memref<*xf32>)
}
