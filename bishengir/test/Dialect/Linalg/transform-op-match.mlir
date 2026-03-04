// RUN: bishengir-opt -transform-interpreter -verify-diagnostics -allow-unregistered-dialect -split-input-file %s

// CHECK: foo
module attributes {transform.with_named_sequence} {
  func.func @foo() {
    // expected-remark @below {{0}}
    "some_op"() {__0__} : () -> ()
    "some_op"() {__1__} : () -> ()
    return
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match attributes{__0__} optional_attributes{} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %0, "0" : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK: foo
module attributes {transform.with_named_sequence} {
  func.func @foo() {
    // expected-remark @below {{0 or 1}}
    "some_op"() {__0__} : () -> ()
    // expected-remark @below {{0 or 1}}
    "some_op"() {__1__} : () -> ()
    return
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match optional_attributes {__0__, __1__} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %0, "0 or 1" : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-NOT: remark: 0 and 1
module attributes {transform.with_named_sequence} {
  func.func @foo() {
    "some_op"() {__0__} : () -> ()
    "some_op"() {__1__} : () -> ()
    return
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match attributes {__0__, __1__} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %0, "0 and 1" : !transform.any_op
    transform.yield
  }
}