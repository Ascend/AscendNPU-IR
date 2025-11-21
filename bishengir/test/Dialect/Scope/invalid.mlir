// RUN: bishengir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

// CHECK-LABEL: test_scope_return
func.func @test_scope_return() {
  // expected-error@below {{'scope.return' op expects parent op 'scope.scope'}}
  scope.return
  return
}
