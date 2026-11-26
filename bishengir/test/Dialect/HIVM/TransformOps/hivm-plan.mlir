// RUN: bishengir-opt %s -transform-interpreter -split-input-file 2>&1 | FileCheck %s
// RUN: bishengir-opt %s -transform-interpreter="disable-expensive-checks" -split-input-file 2>&1 | FileCheck %s


// CHECK: remark: plan #hivm_vectorize.plan<cost = 2.500000e+00 : f64, groups = #hivm_vectorize.groups<[array<i64: 2>, array<i64: 3>]>>
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %plan = transform.param.constant
        #hivm_vectorize.plan<cost = 2.5 : f64, groups = #hivm_vectorize.groups<[array<i64: 2>, array<i64: 3>]>>
        -> !transform.any_param
    transform.debug.emit_param_as_remark %plan, "plan" : !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: remark: plan #hivm_vectorize.plan<cost = 0.000000e+00 : f64, groups = #hivm_vectorize.groups<[]>>
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %plan = transform.param.constant
        #hivm_vectorize.plan<cost = 0.0 : f64, groups = #hivm_vectorize.groups<[]>>
        -> !transform.any_param
    transform.debug.emit_param_as_remark %plan, "plan" : !transform.any_param
    transform.yield
  }
}
