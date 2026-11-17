// RUN: bishengir-opt %s -save-cv-pipeline-snapshot | FileCheck %s
// RUN: bishengir-opt %s -save-cv-pipeline-snapshot -erase-cv-pipeline-snapshots | FileCheck %s --check-prefix=ERASE

// CHECK-LABEL: func.func @with_loop
// CHECK-NOT: hivm.cv_pipeline_snapshot
// CHECK: return
// CHECK: func.func private @with_loop.cv_pipeline_snapshot
// CHECK-SAME: hivm.cv_pipeline_snapshot = @with_loop
// CHECK-NOT: hacc.entry
// CHECK: arith.addi
// ERASE-LABEL: func.func @with_loop
// ERASE: return
// ERASE-NOT: func.func
module {
  func.func @with_loop(%in: index) -> index attributes {hacc.entry} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %out = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %in) -> (index) {
      %added = arith.addi %acc, %c1 : index
      scf.yield %added : index
    }
    return %out : index
  }
}
