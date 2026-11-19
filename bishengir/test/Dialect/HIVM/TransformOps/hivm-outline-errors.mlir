// RUN: not bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload" -split-input-file 2>&1 | FileCheck %s
// RUN: not bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload disable-expensive-checks" -split-input-file 2>&1 | FileCheck %s


// CHECK: expected a payload module root
module attributes {transform.target_tag = "payload"} {
  func.func @non_module_root(%a: f32) -> f32 {
    return %a : f32
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %function = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %groups = transform.param.constant #hivm_vectorize.groups<[]> -> !transform.any_param
    %newRoot, %funcs = transform.hivm.outline %function, %groups
        : (!transform.any_op, !transform.any_param)
          -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----


// CHECK: expected a #hivm_vectorize.groups attribute
module attributes {transform.target_tag = "payload"} {
  func.func @non_groups(%a: f32) -> f32 {
    return %a : f32
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %groups = transform.param.constant 3 : i64 -> !transform.any_param
    %newRoot, %funcs = transform.hivm.outline %root, %groups
        : (!transform.any_op, !transform.any_param)
          -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----


// CHECK: group ordinal 99 is outside the payload root
module attributes {transform.target_tag = "payload"} {
  func.func @malformed_groups(%a: f32) -> f32 {
    return %a : f32
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %groups = transform.param.constant #hivm_vectorize.groups<[array<i64: 99>]>
        -> !transform.any_param
    %newRoot, %funcs = transform.hivm.outline %root, %groups
        : (!transform.any_op, !transform.any_param)
          -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----


// CHECK: expected only function operations
module attributes {transform.target_tag = "payload"} {
  func.func @annotate_non_function(%a: f32) -> f32 {
    return %a : f32
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %annotated = transform.hivm.annotate_vf %root
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


// CHECK: expected a #hivm_vectorize.plan attribute
module attributes {transform.target_tag = "payload"} {
  func.func @non_plan(%a: f32) -> f32 {
    return %a : f32
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %plan = transform.param.constant 3 : i64 -> !transform.any_param
    %groups = transform.hivm.extract_groups %plan
        : (!transform.any_param) -> !transform.any_param
    transform.yield
  }
}
