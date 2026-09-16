// RUN: bishengir-opt %s -hacc-rename-func -allow-unregistered-dialect -verify-diagnostics -split-input-file | FileCheck %s

// CHECK: func.func @foo
func.func @test_standalone_func() attributes {hacc.rename_func = #hacc.rename_func<@foo>} {
  return
}

// -----

// CHECK-NOT: bar
func.func @bar() attributes {hacc.rename_func = #hacc.rename_func<@foo>} {
  return
}

func.func @caller() {
  "some_op"() { callee=@bar } : () -> ()
  func.call @bar() : () -> ()
  return
}

// -----

// expected-error@below {{failed to rename function to @foo because there is already a function with the same name!}}
func.func @bar() attributes {hacc.rename_func = #hacc.rename_func<@foo>} {
  return
}

func.func @foo() {
  return
}
