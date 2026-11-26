// RUN: bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload" -split-input-file 2>&1 | FileCheck %s
// RUN: bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload disable-expensive-checks" -split-input-file 2>&1 | FileCheck %s


// CHECK: remark: outline root
// CHECK-LABEL: func.func private @f_fused_0(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT: %0 = arith.addf %arg0, %arg1 : f32
// CHECK-NEXT: %1 = arith.mulf %0, %arg1 : f32
// CHECK-NEXT: return %1 : f32
// CHECK-NEXT: }
// CHECK-NOT: hivm.vector_function
// CHECK-NOT: no_inline
// CHECK-LABEL: func.func @f(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT: %0 = call @f_fused_0(%arg0, %arg1) : (f32, f32) -> f32
// CHECK-NEXT: %1 = arith.subf %0, %arg0 : f32
// CHECK-NEXT: return %1 : f32
module attributes {transform.target_tag = "payload"} {
  func.func @f(%a: f32, %b: f32) -> f32 {
    %0 = arith.addf %a, %b : f32
    %1 = arith.mulf %0, %b : f32
    %2 = arith.subf %1, %a : f32
    return %2 : f32
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %groups = transform.param.constant #hivm_vectorize.groups<[array<i64: 2, 3>]>
        -> !transform.any_param
    %newRoot, %funcs = transform.hivm.outline %root, %groups
        : (!transform.any_op, !transform.any_param)
          -> (!transform.any_op, !transform.any_op)
    transform.debug.emit_remark_at %newRoot, "outline root" : !transform.any_op
    transform.yield
  }
}

// -----


// CHECK-LABEL: func.func private @g_fused_0(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT: %0 = arith.addf %arg0, %arg1 : f32
// CHECK-NEXT: %1 = arith.subf %0, %arg0 : f32
// CHECK-NEXT: return %1 : f32
// CHECK-NEXT: }
// CHECK-LABEL: func.func @g(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT: %0 = arith.mulf %arg0, %arg1 : f32
// CHECK-NEXT: %[[CALL:.*]] = call @g_fused_0(%arg0, %arg1) : (f32, f32) -> f32
// CHECK-NEXT: %{{.*}} = arith.addf %[[CALL]], %0 : f32
// CHECK-NEXT: return
module attributes {transform.target_tag = "payload"} {
  func.func @g(%a: f32, %b: f32) -> f32 {
    %0 = arith.addf %a, %b : f32
    %1 = arith.mulf %a, %b : f32
    %2 = arith.subf %0, %a : f32
    %3 = arith.addf %2, %1 : f32
    return %3 : f32
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %groups = transform.param.constant #hivm_vectorize.groups<[array<i64: 2, 4>]>
        -> !transform.any_param
    %newRoot, %funcs = transform.hivm.outline %root, %groups
        : (!transform.any_op, !transform.any_param)
          -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----


// CHECK-LABEL: func.func private @pipe_fused_0
// CHECK-SAME: attributes {hivm.vector_function, no_inline}
// CHECK-NEXT: %[[SUM:.*]] = hivm.hir.vadd
// CHECK-NEXT: %[[PRODUCT:.*]] = hivm.hir.vmul ins(%[[SUM]],
// CHECK-NEXT: return %[[PRODUCT]]
// CHECK-NEXT: }
// CHECK-NOT: @pipe_fused_1
// CHECK-LABEL: func.func @pipe(
// CHECK-NEXT: %{{.*}} = call @pipe_fused_0
// CHECK-NEXT: return
module attributes {transform.target_tag = "payload"} {
  func.func @pipe(%a: tensor<8xf32>, %b: tensor<8xf32>,
                 %c: tensor<8xf32>) -> tensor<8xf32> {
    %sum = hivm.hir.vadd ins(%a, %b : tensor<8xf32>, tensor<8xf32>)
        outs(%c : tensor<8xf32>) -> tensor<8xf32>
    %product = hivm.hir.vmul ins(%sum, %b : tensor<8xf32>, tensor<8xf32>)
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
    %selected = transform.hivm.extract_groups %plan
        : (!transform.any_param) -> !transform.any_param
    %newRoot, %funcs = transform.hivm.outline %root, %selected
        : (!transform.any_op, !transform.any_param)
          -> (!transform.any_op, !transform.any_op)
    %annotated = transform.hivm.annotate_vf %funcs
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


// CHECK-LABEL: func.func @empty(%arg0: f32) -> f32 {
// CHECK-NEXT: return %arg0 : f32
// CHECK-NEXT: }
module attributes {transform.target_tag = "payload"} {
  func.func @empty(%a: f32) -> f32 {
    return %a : f32
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %groups = transform.param.constant #hivm_vectorize.groups<[]> -> !transform.any_param
    %newRoot, %funcs = transform.hivm.outline %root, %groups
        : (!transform.any_op, !transform.any_param)
          -> (!transform.any_op, !transform.any_op)
    %annotated = transform.hivm.annotate_vf %funcs
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
