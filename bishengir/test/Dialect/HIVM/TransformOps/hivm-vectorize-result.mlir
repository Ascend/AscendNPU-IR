// RUN: bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload" -split-input-file 2>&1 | FileCheck %s
// RUN: bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload" -split-input-file -verify-diagnostics | FileCheck %s


// CHECK-LABEL: func.func private @forward
// CHECK-COUNT-1: vector.transfer_read
// CHECK-COUNT-1: vector.transfer_write
// CHECK-NOT: arith.addf
module attributes {transform.target_tag = "payload"} {
  func.func private @forward(%arg: tensor<8xf32>, %init: tensor<8xf32>,
                             %init2: tensor<8xf32>) -> tensor<8xf32>
      attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg[%c0], %cst : tensor<8xf32>, vector<8xf32>
    %1 = vector.transfer_write %0, %init[%c0] : vector<8xf32>, tensor<8xf32>
    %2 = vector.transfer_read %1[%c0], %cst : tensor<8xf32>, vector<8xf32>
    %3 = arith.addf %cst, %cst : f32
    %4 = math.exp2 %2 : vector<8xf32>
    %5 = vector.transfer_write %4, %init2[%c0] : vector<8xf32>, tensor<8xf32>
    return %5 : tensor<8xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %cleaned = transform.hivm.cleanup %funcs
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


module attributes {transform.target_tag = "payload"} {
  func.func private @checked(%arg: tensor<8xf32>, %init: tensor<8xf32>)
      -> tensor<8xf32> attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg[%c0], %cst : tensor<8xf32>, vector<8xf32>
    %1 = math.exp2 %0 : vector<8xf32>
    %2 = vector.transfer_write %1, %init[%c0] : vector<8xf32>, tensor<8xf32>
    return %2 : tensor<8xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %cost = transform.hivm.check_result %funcs
        : (!transform.any_op) -> !transform.any_param
    // expected-remark @below {{cost 1 : i64}}
    transform.debug.emit_param_as_remark %cost, "cost"
        : !transform.any_param
    transform.yield
  }
}

// -----


module attributes {transform.target_tag = "payload"} {
  func.func @empty() {
    return
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match attributes {not_a_vector_function} in %root
        : (!transform.any_op) -> !transform.any_op
    %cost = transform.hivm.check_result %funcs
        : (!transform.any_op) -> !transform.any_param
    // expected-remark @below {{cost 0 : i64}}
    transform.debug.emit_param_as_remark %cost, "cost"
        : !transform.any_param
    transform.yield
  }
}
