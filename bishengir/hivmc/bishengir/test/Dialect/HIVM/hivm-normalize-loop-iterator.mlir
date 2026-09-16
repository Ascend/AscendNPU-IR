// RUN: bishengir-opt --hivm-normalize-loop-iterator -split-input-file %s | FileCheck %s

// CHECK-LABEL: @NormalizeLoopIterator
func.func @NormalizeLoopIterator() {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  // CHECK: %[[OUT_ALLOC:.*]] = memref.alloc() : memref<32xf32>
  // CHECK: scf.for
  // CHECK-SAME: iter_args(%[[ITER:.*]] = %[[OUT_ALLOC]])
  %1 = memref.alloc() : memref<32xf32>
  %2 = scf.for %arg0 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg1 = %1) -> (memref<32xf32>) : i32 {
    // CHECK: %[[INNER_ALLOC:.*]] = memref.alloc() : memref<32xf32>
    %3 = memref.alloc() : memref<32xf32>
    hivm.hir.vmax ins(%3, %arg1 : memref<32xf32>, memref<32xf32>) outs(%3 : memref<32xf32>)
    hivm.hir.vsub ins(%arg1, %3 : memref<32xf32>, memref<32xf32>) outs(%3 : memref<32xf32>)
    // CHECK: hivm.hir.copy ins(%[[INNER_ALLOC]] : memref<32xf32>) outs(%[[ITER]] : memref<32xf32>)
    // CHECK: scf.yield %[[ITER]]
    scf.yield %3 : memref<32xf32>
  }

  return
}
