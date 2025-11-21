// RUN: bishengir-opt -allow-unregistered-dialect %s -split-input-file | FileCheck %s
// Verify the printed output can be parsed.
// RUN: bishengir-opt -allow-unregistered-dialect %s -split-input-file | bishengir-opt -allow-unregistered-dialect -split-input-file | FileCheck %s
// Verify the generic form can be parsed.
// RUN: bishengir-opt -allow-unregistered-dialect -mlir-print-op-generic %s -split-input-file | bishengir-opt -allow-unregistered-dialect -split-input-file | FileCheck %s

// CHECK-LABEL: test_scope_scope
func.func @test_scope_scope() {
  // CHECK: scope.scope
  scope.scope : () -> () {
    // CHECK: scope.return
    scope.return
  // CHECK: {tcore_type = #hivm.tcore_type<CUBE>}
  } {tcore_type = #hivm.tcore_type<CUBE>}
  return
}
