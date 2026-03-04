// RUN: bishengir-opt -transform-interpreter -verify-diagnostics %s | FileCheck %s

module attributes { transform.with_named_sequence } {
// CHECK-LABEL: @cache_read
// CHECK: %[[ARG_0_INIT:.*]] = tensor.empty({{.*}}) : tensor<?xf32>
// CHECK: %[[ARG_0_CACHE:.*]] = hfusion.load ins({{.*}}) outs(%[[ARG_0_INIT]] : tensor<?xf32>)
// CHECK: %[[RES:.*]] = linalg.elemwise_unary ins(%[[ARG_0_CACHE]] : tensor<?xf32>)
func.func @cache_read(%arg0: tensor<?xf32>, %dim : index) -> tensor<?xf32> {
  %empty = tensor.empty(%dim) : tensor<?xf32>
  %0 = linalg.elemwise_unary ins(%arg0: tensor<?xf32>) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
  %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  %1 = transform.func.get_func_argument %0[all] : (!transform.any_op) -> !transform.any_value
  %2 = transform.structured.cache_read %1 : (!transform.any_value) -> !transform.any_op
  transform.yield 
}
}