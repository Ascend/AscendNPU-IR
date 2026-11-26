// RUN: not bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload" -split-input-file 2>&1 | FileCheck %s
// RUN: not bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload disable-expensive-checks" -split-input-file 2>&1 | FileCheck %s


// CHECK: requires at least one model
module attributes {transform.target_tag = "payload"} {
  func.func @empty_models() {
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %options = transform.param.constant {} -> !transform.any_param
    %groups = transform.hivm.partition %root, %options {models = []}
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: duplicate model 'SameBlock'
module attributes {transform.target_tag = "payload"} {
  func.func @duplicate_models() {
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %options = transform.param.constant {} -> !transform.any_param
    %groups = transform.hivm.partition %root, %options
        {models = ["SameBlock", "SameBlock"]}
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: unknown partition model 'missing'
module attributes {transform.target_tag = "payload"} {
  func.func @unknown_model() {
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %options = transform.param.constant {} -> !transform.any_param
    %groups = transform.hivm.partition %root, %options {models = ["missing"]}
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: expected a dictionary options attribute
module attributes {transform.target_tag = "payload"} {
  func.func @malformed_options() {
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %options = transform.param.constant 3 : i64 -> !transform.any_param
    %groups = transform.hivm.partition %root, %options {models = ["SameBlock"]}
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: expected a payload module root
module attributes {transform.target_tag = "payload"} {
  func.func @non_module_root() {
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %function = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %options = transform.param.constant {} -> !transform.any_param
    %groups = transform.hivm.partition %function, %options
        {models = ["SameBlock"]}
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: expected exactly one payload root
module attributes {transform.target_tag = "payload"} {
  module @one {
  }
  module @two {
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %roots = transform.structured.match ops{["builtin.module"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %options = transform.param.constant {} -> !transform.any_param
    %groups = transform.hivm.partition %roots, %options {models = ["SameBlock"]}
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    transform.yield
  }
}
