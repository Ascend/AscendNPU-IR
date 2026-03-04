// RUN: bishengir-opt %s --hfusion-auto-schedule-interpreter=kernel-name=foo --verify-diagnostics | FileCheck %s
// RUN: bishengir-opt %s --hfusion-auto-schedule-interpreter="debug-payload-root-tag=foo_payload debug-transform-root-tag=foo_transform" \
// RUN:                  --verify-diagnostics | FileCheck %s

// CHECK: foo
// expected-remark @below {{foo}}
func.func @foo() attributes {transform.target_tag = "foo_payload"} {
  %0 = arith.constant 0 : i32
  return
}

transform.sequence failures(propagate) attributes {transform.target_tag = "foo_transform"} {
  ^bb0(%arg0: !transform.any_op):
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %f, "foo" : !transform.any_op
}