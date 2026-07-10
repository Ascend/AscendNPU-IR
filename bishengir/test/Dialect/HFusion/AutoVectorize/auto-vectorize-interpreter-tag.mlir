// RUN: bishengir-opt %s -hfusion-auto-vectorize-interpreter --verify-diagnostics | FileCheck %s

// CHECK: func.func @plain_a() attributes {transform.target_tag = "auto_vectorize_v2.payload.plain_a"}
// CHECK: func.func @plain_b() attributes {transform.target_tag = "auto_vectorize_v2.payload.plain_b"}
// CHECK: transform.sequence
// CHECK-SAME: transform.target_tag = "auto_vectorize_v2.transform.plain_a"
// CHECK: transform.sequence
// CHECK-SAME: transform.target_tag = "auto_vectorize_v2.transform.plain_b"
module {
  // expected-remark @below {{tagged sequence a}}
  func.func @plain_a() attributes {transform.target_tag = "auto_vectorize_v2.payload.plain_a"} {
    return
  }
  // expected-remark @below {{tagged sequence b}}
  func.func @plain_b() attributes {transform.target_tag = "auto_vectorize_v2.payload.plain_b"} {
    return
  }
  transform.sequence failures(propagate) attributes {transform.target_tag = "auto_vectorize_v2.transform.plain_a"} {
  ^bb0(%arg0: !transform.any_op):
    transform.debug.emit_remark_at %arg0, "tagged sequence a" : !transform.any_op
  }
  transform.sequence failures(propagate) attributes {transform.target_tag = "auto_vectorize_v2.transform.plain_b"} {
  ^bb0(%arg0: !transform.any_op):
    transform.debug.emit_remark_at %arg0, "tagged sequence b" : !transform.any_op
  }
}
