// RUN: bishengir-opt %s -cse -canonicalize | FileCheck %s

llvm.func @test_redundant_thread_id() -> i32 {
  %0 = ascend_dpx.thread_id_x
  %1 = ascend_dpx.thread_id_x
  %2 = llvm.mlir.constant(32 : i32) : i32
  %3 = llvm.mlir.constant(32 : i32) : i32
  %4 = llvm.urem %0, %2 : i32
  %5 = llvm.urem %1, %3 : i32
  %6 = llvm.add %4, %5 : i32
  llvm.return %6 : i32
}

// CHECK-LABEL: llvm.func @test_redundant_thread_id() -> i32
// CHECK-NEXT:    %[[CST:.*]] = llvm.mlir.constant(32 : i32) : i32
// CHECK-NEXT:    %[[TID:.*]] = ascend_dpx.thread_id_x
// CHECK-NEXT:    %[[UREM:.*]] = llvm.urem %[[TID]], %[[CST]] : i32
// CHECK-NEXT:    %[[ADD:.*]] = llvm.add %[[UREM]], %[[UREM]] : i32
// CHECK-NEXT:    llvm.return %[[ADD]] : i32