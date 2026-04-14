// REQUIRES: hivmc
// UNSUPPORTED: bishengir_published

// RUN: bishengir-opt %s \
// RUN:   -transform-interpreter="debug-payload-root-tag=payload" \
// RUN:   -test-transform-dialect-erase-schedule -cse \
// RUN: | FileCheck %s

// RUN: bishengir-opt %s \
// RUN:   -transform-interpreter="debug-payload-root-tag=payload" \
// RUN:   -test-transform-dialect-erase-schedule -cse | \
// RUN: bishengir-compile \
// RUN:   -enable-lir-compile=false -enable-hfusion-compile=false \
// RUN:   -enable-hivm-compile=true

// CHECK-NOT: main_0_tiling_func
// CHECK: main_0_0
// CHECK: main

module attributes { transform.target_tag="payload"} {
  func.func @main(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>) -> tensor<8xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %0 = tensor.empty() : tensor<8xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %arg2 : tensor<8xf32>, tensor<8xf32>) outs(%arg3 : tensor<8xf32>) -> tensor<8xf32>
    return %2 : tensor<8xf32>
  }
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%toplevel_module: !transform.any_op {transform.consumed}) {
    %0 = transform.apply_registered_pass "lower-hfusion-pipeline" to %toplevel_module : (!transform.any_op) -> !transform.any_op
    %1 = transform.apply_registered_pass "convert-hfusion-to-hivm" to %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
