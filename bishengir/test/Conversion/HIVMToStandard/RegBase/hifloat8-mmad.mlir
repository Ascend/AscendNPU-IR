// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9589 -convert-hivm-to-std | FileCheck %s

// CHECK-LABEL: func.func @matmul
// CHECK: call @mma_tile_hifloat8_t_to_float
func.func @matmul(%a: memref<16x32xf8HIF>, %b: memref<32x16xf8HIF>, %c: memref<16x16xf32>) {
  %init = arith.constant true
  %m = arith.constant 16 : index
  %k = arith.constant 32 : index
  %n = arith.constant 16 : index
  hivm.hir.mmadL1 ins(%a, %b, %init, %m, %k, %n : memref<16x32xf8HIF>, memref<32x16xf8HIF>, i1, index, index, index) outs(%c : memref<16x16xf32>)
  return
}
