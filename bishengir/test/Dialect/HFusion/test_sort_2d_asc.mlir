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
// RUN:   -shared-libs=%S/../../../../build/lib/libmlir_runner_utils.so \
// RUN:   -shared-libs=%S/../../../../build/lib/libmlir_c_runner_utils.so | \
// RUN: FileCheck %s

// CHECK: [1, inf, nan, -inf]
// CHECK: [9, 0, 5, 2]
// CHECK: [-inf, 1, inf, nan]
// CHECK: [0, 2, 5, 9]
module {
    // 1. Sorting function
    func.func @sort_kernel_2d_orig(%arg0: memref<4x4xf32>) -> tensor<4x4xf32> {
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %0 = bufferization.to_tensor %arg0 restrict writable : memref<4x4xf32>
        %1 = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%arg2 = %0) -> (tensor<4x4xf32>) {
        %2 = scf.for %arg3 = %c0 to %c3 step %c1 iter_args(%arg4 = %arg2) -> (tensor<4x4xf32>) {
            %3 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg4) -> (tensor<4x4xf32>) {
            %4 = arith.addi %arg5, %c1 : index
            %extracted = tensor.extract %arg6[%arg1, %arg5] : tensor<4x4xf32>
            %extracted_0 = tensor.extract %arg6[%arg1, %4] : tensor<4x4xf32>
            %5 = arith.cmpf ugt, %extracted, %extracted_0 : f32
            %6 = scf.if %5 -> (tensor<4x4xf32>) {
                %inserted = tensor.insert %extracted_0 into %arg6[%arg1, %arg5] : tensor<4x4xf32>
                %inserted_1 = tensor.insert %extracted into %inserted[%arg1, %4] : tensor<4x4xf32>
                scf.yield %inserted_1 : tensor<4x4xf32>
            } else {
                scf.yield %arg6 : tensor<4x4xf32>
            }
            scf.yield %6 : tensor<4x4xf32>
            }
            scf.yield %3 : tensor<4x4xf32>
        }
        scf.yield %2 : tensor<4x4xf32>
        }
        return %1 : tensor<4x4xf32>
    }

    // 2. Entry point (main)
    func.func @main() {
        %m = memref.alloc() : memref<4x4xf32>
        
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index

        %f0 = arith.constant 0.0 : f32
        %f1 = arith.constant 1.0 : f32
        %f2 = arith.constant 2.0 : f32
        %f3 = arith.constant 3.0 : f32
        %f4 = arith.constant 4.0 : f32
        %f5 = arith.constant 5.0 : f32
        %f8 = arith.constant 8.0 : f32
        %f9 = arith.constant 9.0 : f32
        // Define special floating point constants
        %f_inf = arith.constant 0x7F800000 : f32 // Positive Infinity
        %f_ninf = arith.constant 0xFF800000 : f32 // Negative Infinity
        %f_nan = arith.constant 0x7FC00000 : f32 // Quiet NaN
        // Data initialization
        memref.store %f1, %m[%c0, %c0] : memref<4x4xf32>
        memref.store %f_inf, %m[%c0, %c1] : memref<4x4xf32>
        memref.store %f_nan, %m[%c0, %c2] : memref<4x4xf32>
        memref.store %f_ninf, %m[%c0, %c3] : memref<4x4xf32>

        memref.store %f9, %m[%c1, %c0] : memref<4x4xf32>
        memref.store %f0, %m[%c1, %c1] : memref<4x4xf32>
        memref.store %f5, %m[%c1, %c2] : memref<4x4xf32>
        memref.store %f2, %m[%c1, %c3] : memref<4x4xf32>

        memref.store %f1, %m[%c2, %c0] : memref<4x4xf32>
        memref.store %f5, %m[%c2, %c1] : memref<4x4xf32>
        memref.store %f2, %m[%c2, %c2] : memref<4x4xf32>
        memref.store %f8, %m[%c2, %c3] : memref<4x4xf32>

        memref.store %f0, %m[%c3, %c0] : memref<4x4xf32>
        memref.store %f9, %m[%c3, %c1] : memref<4x4xf32>
        memref.store %f3, %m[%c3, %c2] : memref<4x4xf32>
        memref.store %f1, %m[%c3, %c3] : memref<4x4xf32>

        // --- Print original tensor ---
        %unranked_orig = memref.cast %m : memref<4x4xf32> to memref<*xf32>
        func.call @printMemrefF32(%unranked_orig) : (memref<*xf32>) -> ()

        // Invoke sorting kernel
        %result_tensor = func.call @sort_kernel_2d_orig(%m) : (memref<4x4xf32>) -> tensor<4x4xf32>

        // Print result tensor
        %res_m = bufferization.to_memref %result_tensor : memref<4x4xf32>
        %unranked_res = memref.cast %res_m : memref<4x4xf32> to memref<*xf32>
        func.call @printMemrefF32(%unranked_res) : (memref<*xf32>) -> ()

        memref.dealloc %m : memref<4x4xf32>
        return
    }

    func.func private @printMemrefF32(memref<*xf32>)
}