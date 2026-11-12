// RUN: bishengir-opt %s --outline-cv-scopes --split-input-file | FileCheck %s

func.func @foreign_scope_in_loop(%lb: index, %ub: index, %step: index) {
  // CHECK-LABEL: func.func @foreign_scope_in_loop
  // CHECK: {"hivm.legacy-cv-pipeline"}
  scf.for %i = %lb to %ub step %step {
    scope.scope : () -> () {
      "test.opaque"() : () -> ()
      scope.return
    }
    scf.yield
  }
  return
}

// -----

func.func @foreign_scope_outside_loop(%lb: index, %ub: index, %step: index) {
  // CHECK-LABEL: func.func @foreign_scope_outside_loop
  // CHECK: {"hivm.legacy-cv-pipeline"}
  scope.scope : () -> () {
    "test.opaque"() : () -> ()
    scope.return
  }
  scf.for %i = %lb to %ub step %step {
    "test.opaque"() : () -> ()
    scf.yield
  }
  return
}

// -----

func.func @dirty(%lb: index, %ub: index, %step: index) {
  // CHECK-LABEL: func.func @dirty
  // CHECK: {"hivm.legacy-cv-pipeline"}
  scf.for %i = %lb to %ub step %step {
    scope.scope : () -> () {
      "test.opaque"() : () -> ()
      scope.return
    }
    scf.yield
  }
  return
}

func.func @clean(%lb: index, %ub: index, %step: index) {
  // CHECK-LABEL: func.func @clean
  // CHECK-NOT: hivm.legacy_cv_pipeline
  scf.for %i = %lb to %ub step %step {
    "test.opaque"() : () -> ()
    scf.yield
  }
  return
}

// -----

func.func @foreign_scope_in_if(%lb: index, %ub: index, %step: index, %c: i1) {
  // CHECK-LABEL: func.func @foreign_scope_in_if
  // CHECK: {"hivm.legacy-cv-pipeline"}
  scf.for %i = %lb to %ub step %step {
    scf.if %c {
      scope.scope : () -> () {
        "test.opaque"() : () -> ()
        scope.return
      }
    }
    scf.yield
  }
  return
}
