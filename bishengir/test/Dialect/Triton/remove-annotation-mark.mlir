// RUN: bishengir-opt -split-input-file %s -remove-annotation-mark | FileCheck %s

// All annotation.mark ops are erased; everything else is left untouched.
// CHECK-LABEL: @remove_marks
// CHECK-NOT:   annotation.mark
// CHECK:       %[[ADD:.*]] = arith.addi %arg0, %arg1
// CHECK-NEXT:  return %[[ADD]]
func.func @remove_marks(%arg0: i32, %arg1: i32) -> i32 {
  annotation.mark %arg0 {reached_mask_ops_idx = 0 : i32} : i32
  %0 = arith.addi %arg0, %arg1 : i32
  annotation.mark %0 {reached_mask_ops_idx = 0 : i32} : i32
  return %0 : i32
}

// -----

// A function with no marks is a no-op.
// CHECK-LABEL: @no_marks
// CHECK-NOT:   annotation.mark
// CHECK:       return
func.func @no_marks(%arg0: i32) -> i32 {
  return %arg0 : i32
}
