// RUN: bishengir-opt %s \
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

// CHECK: [0, 7, 1, 8]
// CHECK: [8, 7, 1, 0]

module {
    // 1. Sorting function
    func.func @sort_kernel_3d_desc(%arg0: memref<2x3x4xf32>) -> tensor<2x3x4xf32> {
        %c3 = arith.constant 3 : index
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %0 = bufferization.to_tensor %arg0 restrict writable : memref<2x3x4xf32>
        %1 = scf.for %arg1 = %c0 to %c2 step %c1 iter_args(%arg2 = %0) -> (tensor<2x3x4xf32>) {
        %2 = scf.for %arg3 = %c0 to %c3 step %c1 iter_args(%arg4 = %arg2) -> (tensor<2x3x4xf32>) {
            %3 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg4) -> (tensor<2x3x4xf32>) {
            %4 = scf.for %arg7 = %c0 to %c3 step %c1 iter_args(%arg8 = %arg6) -> (tensor<2x3x4xf32>) {
                %5 = arith.addi %arg7, %c1 : index
                %extracted = tensor.extract %arg8[%arg1, %arg3, %arg7] : tensor<2x3x4xf32>
                %extracted_0 = tensor.extract %arg8[%arg1, %arg3, %5] : tensor<2x3x4xf32>
                %6 = arith.cmpf ult, %extracted, %extracted_0 : f32
                %7 = scf.if %6 -> (tensor<2x3x4xf32>) {
                %inserted = tensor.insert %extracted_0 into %arg8[%arg1, %arg3, %arg7] : tensor<2x3x4xf32>
                %inserted_1 = tensor.insert %extracted into %inserted[%arg1, %arg3, %5] : tensor<2x3x4xf32>
                scf.yield %inserted_1 : tensor<2x3x4xf32>
                } else {
                scf.yield %arg8 : tensor<2x3x4xf32>
                }
                scf.yield %7 : tensor<2x3x4xf32>
            }
            scf.yield %4 : tensor<2x3x4xf32>
            }
            scf.yield %3 : tensor<2x3x4xf32>
        }
        scf.yield %2 : tensor<2x3x4xf32>
        }
        return %1 : tensor<2x3x4xf32>
    }

    // 2. Entry point (main)
    func.func @main() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        
        %m = memref.alloc() : memref<2x3x4xf32>

        // Data initialization
        scf.for %i = %c0 to %c2 step %c1 {
            scf.for %j = %c0 to %c3 step %c1 {
                scf.for %k = %c0 to %c4 step %c1 {
                    %i_i = arith.index_cast %i : index to i64
                    %j_i = arith.index_cast %j : index to i64
                    %k_i = arith.index_cast %k : index to i64
                    
                    %c5 = arith.constant 5 : i64
                    %c2_64 = arith.constant 2 : i64
                    %c7 = arith.constant 7 : i64
                    %c13 = arith.constant 13 : i64
                    
                    %t1 = arith.muli %i_i, %c5 : i64
                    %t2 = arith.muli %j_i, %c2_64 : i64
                    %t3 = arith.muli %k_i, %c7 : i64
                    %sum1 = arith.addi %t1, %t2 : i64
                    %sum2 = arith.addi %sum1, %t3 : i64
                    %rem = arith.remsi %sum2, %c13 : i64
                    
                    %fval = arith.sitofp %rem : i64 to f32
                    memref.store %fval, %m[%i, %j, %k] : memref<2x3x4xf32>
                }
            }
        }

        // --- Print original tensor ---
        %unranked_orig = memref.cast %m : memref<2x3x4xf32> to memref<*xf32>
        func.call @printMemrefF32(%unranked_orig) : (memref<*xf32>) -> ()

        // Invoke sorting kernel
        %result_tensor = func.call @sort_kernel_3d_desc(%m) : (memref<2x3x4xf32>) -> tensor<2x3x4xf32>

        // Print result tensor
        %res_m = bufferization.to_memref %result_tensor : memref<2x3x4xf32>
        %unranked_res = memref.cast %res_m : memref<2x3x4xf32> to memref<*xf32>
        func.call @printMemrefF32(%unranked_res) : (memref<*xf32>) -> ()

        memref.dealloc %m : memref<2x3x4xf32>
        return
    }

    func.func private @printMemrefF32(memref<*xf32>)
}