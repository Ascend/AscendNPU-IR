// RUN: bishengir-opt -split-input-file -convert-hivmave-to-ave-intrin %s | FileCheck %s

// Check that IV is narrowed to i32 by default.
// CHECK-LABEL: func.func @test_for
// CHECK: scf.for %[[ARG:.*]] = %{{.*}} to %{{.*}} step %{{.*}} : i32 {
// CHECK: llvm.sext %[[ARG]] : i32 to i64
func.func @test_for() attributes {hivm.vector_function} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<6xi32>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<6xi32>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<6xi32>
  %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<6xi32>
  scf.for %arg0 = %c0 to %c6 step %c1 {
    %0 = memref.load %alloc[%arg0] : memref<6xi32>
    %1 = memref.load %alloc_0[%arg0] : memref<6xi32>
    %low, %high = arith.mului_extended %0, %1 : i32
    memref.store %low, %alloc_1[%arg0] : memref<6xi32>
    memref.store %high, %alloc_2[%arg0] : memref<6xi32>
  }
  return
}

