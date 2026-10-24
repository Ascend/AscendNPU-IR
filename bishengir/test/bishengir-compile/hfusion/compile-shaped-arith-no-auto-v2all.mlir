// UNSUPPORTED: bishengir_published
// RUN: bishengir-compile -enable-hfusion-compile=true -enable-hivm-compile=true -enable-lir-compile=false -enable-tree-reduce-v2=true -enable-tree-reduce-mode=ra -target=Ascend950PR_9579 -debug-only=bishengir-compile-regbase,tree-reduce-v2 %s 2>&1 | FileCheck %s

// CHECK-NOT: Detected reduce-only
// CHECK: Reduction rewrite plan: kind=legacy-ra

func.func @shaped_arith_before_reduce(
    %lhs: tensor<64x128xf32>, %rhs: tensor<64x128xf32>)
    -> tensor<128xf32>
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %product = arith.mulf %lhs, %rhs : tensor<64x128xf32>
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<128xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<128xf32>)
      -> tensor<128xf32>
  %reduced = linalg.reduce ins(%product : tensor<64x128xf32>)
      outs(%init : tensor<128xf32>) dimensions = [0]
      (%in: f32, %acc: f32) {
        %sum = arith.addf %in, %acc : f32
        linalg.yield %sum : f32
      }
  return %reduced : tensor<128xf32>
}
