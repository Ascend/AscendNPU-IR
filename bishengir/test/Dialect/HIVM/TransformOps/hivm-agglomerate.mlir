// RUN: bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload" -split-input-file 2>&1 | FileCheck %s
// RUN: bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload disable-expensive-checks" -split-input-file 2>&1 | FileCheck %s


// CHECK: remark: plan #hivm_vectorize.plan<cost = 1.000000e+00 : f64, groups = #hivm_vectorize.groups<[array<i64: 2, 3>]>>
module attributes {transform.target_tag = "payload"} {
  func.func @connected(%a: tensor<8xf32>, %b: tensor<8xf32>,
                       %c: tensor<8xf32>) -> tensor<8xf32> {
    %sum = hivm.hir.vadd ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
        outs(%c : tensor<8xf32>) -> tensor<8xf32>
    %product = hivm.hir.vmul ins(%a, %sum : tensor<8xf32>, tensor<8xf32>)
        outs(%c : tensor<8xf32>) -> tensor<8xf32>
    return %product : tensor<8xf32>
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %options = transform.param.constant {} -> !transform.any_param
    %groups = transform.hivm.partition %root, %options
        {models = ["SameBlock", "UnseparatedByBarrier"]}
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    %plan = transform.hivm.agglomerate %root, %options, %groups
        {selector = "preserve_groups", cost = ["uniform_cost"]}
        : (!transform.any_op, !transform.any_param, !transform.any_param)
          -> !transform.any_param
    transform.debug.emit_param_as_remark %plan, "plan" : !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: remark: plan #hivm_vectorize.plan<cost = 0.000000e+00 : f64, groups = #hivm_vectorize.groups<[]>>
module attributes {transform.target_tag = "payload"} {
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %options = transform.param.constant {} -> !transform.any_param
    %groups = transform.hivm.partition %root, %options
        {models = ["SameBlock", "UnseparatedByBarrier"]}
        : (!transform.any_op, !transform.any_param) -> !transform.any_param
    %plan = transform.hivm.agglomerate %root, %options, %groups
        {selector = "preserve_groups", cost = ["uniform_cost"]}
        : (!transform.any_op, !transform.any_param, !transform.any_param)
          -> !transform.any_param
    transform.debug.emit_param_as_remark %plan, "plan" : !transform.any_param
    transform.yield
  }
}
