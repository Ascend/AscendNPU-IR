// RUN: not bishengir-opt %s -transform-interpreter -split-input-file 2>&1 | FileCheck %s
// RUN: not bishengir-opt %s -transform-interpreter="disable-expensive-checks" -split-input-file 2>&1 | FileCheck %s


// CHECK: plan cost must be non-negative
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %plan = transform.param.constant
        #hivm_vectorize.plan<cost = -1.0 : f64, groups = #hivm_vectorize.groups<[]>>
        -> !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: plan cost must be f64
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %plan = transform.param.constant
        #hivm_vectorize.plan<cost = 1.0 : f32, groups = #hivm_vectorize.groups<[]>>
        -> !transform.any_param
    transform.yield
  }
}

// -----


// CHECK: plan cost must be finite
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %plan = transform.param.constant
        #hivm_vectorize.plan<cost = 0x7FF0000000000000 : f64, groups = #hivm_vectorize.groups<[]>>
        -> !transform.any_param
    transform.yield
  }
}
