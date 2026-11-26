// RUN: bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload" \
// RUN:   2>&1 | FileCheck %s

// The agglomerate stage combines partition groups into a plan. The shipped
// sequence selects "preserve_groups" with the uniform cost model, so groups
// stay separate and the plan reports a uniform cost per group.

module attributes {transform.target_tag = "payload"} {
  func.func private @opaque()

  func.func @twogroups(%a: tensor<64xf32>, %b: tensor<64xf32>,
                       %init: tensor<64xf32>) -> tensor<64xf32> {
    %sum = hivm.hir.vadd ins(%a, %b : tensor<64xf32>, tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    func.call @opaque() : () -> ()
    %product = hivm.hir.vmul ins(%sum, %b : tensor<64xf32>, tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    return %product : tensor<64xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
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

// CHECK: remark: plan #hivm_vectorize.plan<cost = 2.000000e+00 : f64,
// CHECK-SAME: groups = #hivm_vectorize.groups<[array<i64: 3>, array<i64: 5>]>>
