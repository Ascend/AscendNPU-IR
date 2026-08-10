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

// CHECK: [1, 3, 5, 7]
// CHECK: [2, 4, 6, 8]
// CHECK: [1, 2, 3, 4, 5, 6, 7, 8]

module {
  func.func @interleave_kernel(%arg0: memref<4xf32>, %arg1: memref<4xf32>) -> tensor<8xf32> {
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<4xf32>
    %1 = bufferization.to_tensor %arg1 restrict writable : memref<4xf32>
    %2 = hfusion.interleave %0, %1 : tensor<4xf32>, tensor<4xf32> -> tensor<8xf32>
    return %2 : tensor<8xf32>
  }

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %a = memref.alloc() : memref<4xf32>
    %b = memref.alloc() : memref<4xf32>

    %f1 = arith.constant 1.0 : f32
    %f2 = arith.constant 2.0 : f32
    %f3 = arith.constant 3.0 : f32
    %f4 = arith.constant 4.0 : f32
    %f5 = arith.constant 5.0 : f32
    %f6 = arith.constant 6.0 : f32
    %f7 = arith.constant 7.0 : f32
    %f8 = arith.constant 8.0 : f32

    memref.store %f1, %a[%c0] : memref<4xf32>
    memref.store %f3, %a[%c1] : memref<4xf32>
    memref.store %f5, %a[%c2] : memref<4xf32>
    memref.store %f7, %a[%c3] : memref<4xf32>

    memref.store %f2, %b[%c0] : memref<4xf32>
    memref.store %f4, %b[%c1] : memref<4xf32>
    memref.store %f6, %b[%c2] : memref<4xf32>
    memref.store %f8, %b[%c3] : memref<4xf32>

    %a_unranked = memref.cast %a : memref<4xf32> to memref<*xf32>
    func.call @printMemrefF32(%a_unranked) : (memref<*xf32>) -> ()

    %b_unranked = memref.cast %b : memref<4xf32> to memref<*xf32>
    func.call @printMemrefF32(%b_unranked) : (memref<*xf32>) -> ()

    %result_tensor = func.call @interleave_kernel(%a, %b) : (memref<4xf32>, memref<4xf32>) -> tensor<8xf32>
    %result_memref = bufferization.to_memref %result_tensor : memref<8xf32>
    %result_unranked = memref.cast %result_memref : memref<8xf32> to memref<*xf32>
    func.call @printMemrefF32(%result_unranked) : (memref<*xf32>) -> ()

    memref.dealloc %a : memref<4xf32>
    memref.dealloc %b : memref<4xf32>
    return
  }

  func.func private @printMemrefF32(memref<*xf32>)
}
