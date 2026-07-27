// REQUIRES: regbase
// RUN: bishengir-opt -cv-pipelining="pipeline-depth=0" %s | FileCheck %s

// Verify that the compile-level workspace depth value 0 disables CV
// pipelining instead of falling back to automatic depth inference.

// CHECK-LABEL: func.func @cv_pipeline_disabled
// CHECK: scf.for
// CHECK-NOT: hivm.loop_core_type
// CHECK: return
func.func @cv_pipeline_disabled() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  scf.for %i = %c0 to %c4 step %c1 : i32 {
    %unused = arith.addi %i, %c1 : i32
  }
  return
}
