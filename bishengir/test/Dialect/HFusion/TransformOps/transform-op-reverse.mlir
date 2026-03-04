// RUN: bishengir-opt -transform-interpreter -verify-diagnostics -allow-unregistered-dialect %s
// RUN: bishengir-opt %s -transform-interpreter -allow-unregistered-dialect 2>&1 | FileCheck %s -check-prefix=CHECK-NOTE

module attributes { transform.with_named_sequence } {
func.func @reverse() {
  // expected-remark @+1 {{matched op}}
  "some_op"() {op0} : () -> ()
  // expected-remark @+1 {{matched op}}
  "some_op"() {op1} : () -> ()
}

transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
  %0 = transform.structured.match ops{["some_op"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  %1 = transform.reverse %0 : (!transform.any_op) -> !transform.any_op
  transform.debug.emit_remark_at %1, "matched op" : !transform.any_op
  transform.yield 
}
}

// CHECK-NOTE: note: see current operation: "some_op"() {op1} : () -> ()
// CHECK-NOTE: note: see current operation: "some_op"() {op0} : () -> ()