// RUN: bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload" \
// RUN:   2>&1 | FileCheck %s

// The annotate stage marks vector functions, marks their call sites, and
// corrects inherited core metadata to AIV. Functions outside the selected
// handle keep their attributes.

module attributes {transform.target_tag = "payload"} {
  func.func private @vf(%a: f32) -> f32
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>,
                  mix_mode = "mix"} {
    return %a : f32
  }

  func.func @caller(%a: f32) -> f32 {
    %0 = call @vf(%a) : (f32) -> f32
    return %0 : f32
  }

  func.func @cpu(%a: f32) -> f32 {
    return %a : f32
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %vfs = transform.structured.match ops{["func.func"]}
        attributes {mix_mode = "mix"} in %root
        : (!transform.any_op) -> !transform.any_op
    %annotated = transform.hivm.annotate_vf %vfs
        : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %annotated, "annotated" : !transform.any_op
    transform.yield
  }
}

// CHECK: remark: annotated
// CHECK-LABEL: func.func private @vf(%arg0: f32) -> f32 attributes {
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-SAME: hivm.vector_function
// CHECK-SAME: mix_mode = "aiv"
// CHECK-SAME: no_inline
// CHECK-LABEL: func.func @caller(
// CHECK: call @vf(%{{.*}}) {hivm.vector_function, no_inline}
// CHECK-LABEL: func.func @cpu(
// CHECK-NOT: hivm.vector_function
// CHECK: return
