// RUN: bishengir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

// CHECK-LABEL: func.func @invalid_scope_return_parent(
func.func @invalid_scope_return_parent() {
  // expected-error@below {{'scope.return' op expects parent op 'scope.scope'}}
  scope.return
  return
}

// -----

// CHECK-LABEL: func.func @no_returned_value(
func.func @no_returned_value() -> f32 {
  %cst = arith.constant 0.000000e+00 : f32
  // expected-error@below {{'scope.scope' op  region control flow edge from Region #0 to parent results: source has 0 operands, but target successor needs 1}}
  %0 = scope.scope : () -> f32 {
    scope.return
  }
  return %0 : f32
}

// -----

// CHECK-LABEL: func.func @no_result_type(
func.func @no_result_type() -> () {
  %cst = arith.constant 0.000000e+00 : f32
  // expected-error@below {{'scope.scope' op  region control flow edge from Region #0 to parent results: source has 1 operands, but target successor needs 0}}
  scope.scope : () -> () {
    scope.return %cst : f32
  }
  return
}

// -----

// CHECK-LABEL: func.func @mismatch_result_type(
func.func @mismatch_result_type() {
  %cst = arith.constant 0.000000e+00 : f32
  // expected-error@below {{'scope.scope' op  along control flow edge from Region #0 to parent results: source type #0 'f32' should match input type #0 'i32'}}
  scope.scope : () -> i32 {
    scope.return %cst : f32
  }
  return
}

// -----

// CHECK-LABEL: func.func @need_one_region(
func.func @need_one_region() {
  // expected-error@below {{'scope.scope' op region #0 ('region') failed to verify constraint: region with 1 blocks}}
  "scope.scope"() ({
  }): () -> ()
  return
}

// -----

// CHECK-LABEL: func.func @multiple_blocks(
func.func @multiple_blocks() {
  // expected-error@below {{'scope.scope' op expects region #0 to have 0 or 1 blocks}}
  "scope.scope"() ({
    ^bb0:
      scope.return
    ^bb1:
      scope.return
  }): () -> ()

  return
}

// -----

// CHECK-LABEL: func.func @multiple_regions(
func.func @multiple_regions() {
  // expected-error@below {{'scope.scope' op requires one region}}
  "scope.scope"() ({
    ^bb0:
      scope.return
  }, {}): () -> ()
  return
}
