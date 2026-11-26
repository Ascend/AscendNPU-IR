// RUN: bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload" -split-input-file 2>&1 | FileCheck %s
// RUN: bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload disable-expensive-checks" -split-input-file 2>&1 | FileCheck %s


// CHECK: remark: groups #hivm_vectorize.groups<[array<i64: 2, 3>, array<i64: 5>]>
module attributes {transform.target_tag = "payload"} {
  func.func @simple(%a: tensor<8xf32>, %b: tensor<8xf32>,
                    %c: tensor<8xf32>) -> tensor<8xf32> {
    %sum = hivm.hir.vadd ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
        outs(%c : tensor<8xf32>) -> tensor<8xf32>
    %product = hivm.hir.vmul ins(%sum, %b : tensor<8xf32>, tensor<8xf32>)
        outs(%c : tensor<8xf32>) -> tensor<8xf32>
    %barrier = arith.constant 0 : index
    %absolute = hivm.hir.vabs ins(%product : tensor<8xf32>)
        outs(%c : tensor<8xf32>) -> tensor<8xf32>
    return %absolute : tensor<8xf32>
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
    transform.debug.emit_param_as_remark %groups, "groups" : !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: remark: groups #hivm_vectorize.groups<[]>
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
    transform.debug.emit_param_as_remark %groups, "groups" : !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: remark: groups #hivm_vectorize.groups<[array<i64: 3>, array<i64: 6>]>
module attributes {transform.target_tag = "payload"} {
  func.func @nested(%a: tensor<8xf32>, %b: tensor<8xf32>,
                    %c: tensor<8xf32>, %cond: i1) {
    scf.if %cond {
      %sum = hivm.hir.vadd ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
          outs(%c : tensor<8xf32>) -> tensor<8xf32>
    }
    scf.if %cond {
      %product = hivm.hir.vmul ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
          outs(%c : tensor<8xf32>) -> tensor<8xf32>
    }
    return
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
    transform.debug.emit_param_as_remark %groups, "groups" : !transform.any_param
    transform.yield
  }
}
