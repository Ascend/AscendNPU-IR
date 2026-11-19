// RUN: not bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload" -split-input-file 2>&1 | FileCheck %s
// RUN: not bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload disable-expensive-checks" -split-input-file 2>&1 | FileCheck %s


// CHECK: requires a selector
module attributes {transform.target_tag = "payload"} {
  func.func @empty_selector() {
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %options = transform.param.constant {} -> !transform.any_param
    %groups = transform.param.constant #hivm_vectorize.groups<[]> -> !transform.any_param
    %plan = transform.hivm.agglomerate %root, %options, %groups
        {selector = "", cost = ["uniform_cost"]}
        : (!transform.any_op, !transform.any_param, !transform.any_param)
          -> !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: requires at least one cost model
module attributes {transform.target_tag = "payload"} {
  func.func @empty_cost() {
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %options = transform.param.constant {} -> !transform.any_param
    %groups = transform.param.constant #hivm_vectorize.groups<[]> -> !transform.any_param
    %plan = transform.hivm.agglomerate %root, %options, %groups
        {selector = "preserve_groups", cost = []}
        : (!transform.any_op, !transform.any_param, !transform.any_param)
          -> !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: duplicate cost model 'uniform_cost'
module attributes {transform.target_tag = "payload"} {
  func.func @duplicate_cost() {
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %options = transform.param.constant {} -> !transform.any_param
    %groups = transform.param.constant #hivm_vectorize.groups<[]> -> !transform.any_param
    %plan = transform.hivm.agglomerate %root, %options, %groups
        {selector = "preserve_groups", cost = ["uniform_cost", "uniform_cost"]}
        : (!transform.any_op, !transform.any_param, !transform.any_param)
          -> !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: unknown selector 'missing'
module attributes {transform.target_tag = "payload"} {
  func.func @unknown_selector() {
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %options = transform.param.constant {} -> !transform.any_param
    %groups = transform.param.constant #hivm_vectorize.groups<[]> -> !transform.any_param
    %plan = transform.hivm.agglomerate %root, %options, %groups
        {selector = "missing", cost = ["uniform_cost"]}
        : (!transform.any_op, !transform.any_param, !transform.any_param)
          -> !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: unknown cost model 'missing'
module attributes {transform.target_tag = "payload"} {
  func.func @unknown_cost() {
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %options = transform.param.constant {} -> !transform.any_param
    %groups = transform.param.constant #hivm_vectorize.groups<[]> -> !transform.any_param
    %plan = transform.hivm.agglomerate %root, %options, %groups
        {selector = "preserve_groups", cost = ["missing"]}
        : (!transform.any_op, !transform.any_param, !transform.any_param)
          -> !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: expected a #hivm_vectorize.groups attribute
module attributes {transform.target_tag = "payload"} {
  func.func @non_groups() {
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %options = transform.param.constant {} -> !transform.any_param
    %groups = transform.param.constant 3 : i64 -> !transform.any_param
    %plan = transform.hivm.agglomerate %root, %options, %groups
        {selector = "preserve_groups", cost = ["uniform_cost"]}
        : (!transform.any_op, !transform.any_param, !transform.any_param)
          -> !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: group ordinal 99 is outside the payload root
module attributes {transform.target_tag = "payload"} {
  func.func @malformed_groups() {
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %options = transform.param.constant {} -> !transform.any_param
    %groups = transform.param.constant #hivm_vectorize.groups<[array<i64: 99>]>
        -> !transform.any_param
    %plan = transform.hivm.agglomerate %root, %options, %groups
        {selector = "preserve_groups", cost = ["uniform_cost"]}
        : (!transform.any_op, !transform.any_param, !transform.any_param)
          -> !transform.any_param
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
    %groups = transform.param.constant #hivm_vectorize.groups<[]> -> !transform.any_param
    %plan = transform.hivm.agglomerate %function, %options, %groups
        {selector = "preserve_groups", cost = ["uniform_cost"]}
        : (!transform.any_op, !transform.any_param, !transform.any_param)
          -> !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: initial agglomeration blocks must be disjoint
module attributes {transform.target_tag = "payload"} {
  func.func @overlapping_groups(%a: memref<8xf32>, %b: memref<8xf32>,
                                %c: memref<8xf32>) {
    hivm.hir.vadd ins(%a, %b : memref<8xf32>, memref<8xf32>)
        outs(%c : memref<8xf32>)
    return
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %options = transform.param.constant {} -> !transform.any_param
    %groups = transform.param.constant #hivm_vectorize.groups<[array<i64: 2>, array<i64: 2>]>
        -> !transform.any_param
    %plan = transform.hivm.agglomerate %root, %options, %groups
        {selector = "preserve_groups", cost = ["uniform_cost"]}
        : (!transform.any_op, !transform.any_param, !transform.any_param)
          -> !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: group member must not be a module
module attributes {transform.target_tag = "payload"} {
  func.func @module_member(%a: f32, %b: f32) -> f32 {
    %0 = arith.addf %a, %b : f32
    return %0 : f32
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %options = transform.param.constant {} -> !transform.any_param
    %groups = transform.param.constant #hivm_vectorize.groups<[array<i64: 0>]>
        -> !transform.any_param
    %plan = transform.hivm.agglomerate %root, %options, %groups
        {selector = "preserve_groups", cost = ["uniform_cost"]}
        : (!transform.any_op, !transform.any_param, !transform.any_param)
          -> !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: group member must not be a function
module attributes {transform.target_tag = "payload"} {
  func.func @function_member(%a: f32, %b: f32) -> f32 {
    %0 = arith.addf %a, %b : f32
    return %0 : f32
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %options = transform.param.constant {} -> !transform.any_param
    %groups = transform.param.constant #hivm_vectorize.groups<[array<i64: 1>]>
        -> !transform.any_param
    %plan = transform.hivm.agglomerate %root, %options, %groups
        {selector = "preserve_groups", cost = ["uniform_cost"]}
        : (!transform.any_op, !transform.any_param, !transform.any_param)
          -> !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: agglomeration groups must belong to one function
module attributes {transform.target_tag = "payload"} {
  func.func @first(%a: f32, %b: f32) -> f32 {
    %0 = arith.addf %a, %b : f32
    return %0 : f32
  }

  func.func @second(%a: f32, %b: f32) -> f32 {
    %0 = arith.mulf %a, %b : f32
    return %0 : f32
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %options = transform.param.constant {} -> !transform.any_param
    %groups = transform.param.constant #hivm_vectorize.groups<[array<i64: 2, 5>]>
        -> !transform.any_param
    %plan = transform.hivm.agglomerate %root, %options, %groups
        {selector = "preserve_groups", cost = ["uniform_cost"]}
        : (!transform.any_op, !transform.any_param, !transform.any_param)
          -> !transform.any_param
    transform.yield
  }
}
