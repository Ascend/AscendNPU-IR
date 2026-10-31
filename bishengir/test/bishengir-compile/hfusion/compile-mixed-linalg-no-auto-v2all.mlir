// REQUIRES: asserts
// UNSUPPORTED: bishengir_published
// RUN: bishengir-compile -enable-hfusion-compile=true -enable-hivm-compile=true -enable-lir-compile=false -enable-tree-reduce-v2=true -enable-tree-reduce-mode=ra -target=Ascend950PR_9579 -debug-only=bishengir-compile-regbase,tree-reduce-v2 %s 2>&1 | FileCheck %s

// CHECK: Not reduce-only: module contains a non-reduction compute op
// CHECK-NOT: Detected reduce-only
// CHECK: [tree-reduce-v2] === Processing func: None enableRA=1 enableAR=0 ===

func.func @mixed_linalg(%arg0: tensor<64x128xf32>) -> tensor<128xf32>
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<64x128xf32>
  %mapped = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
      ins(%arg0, %arg0 : tensor<64x128xf32>, tensor<64x128xf32>)
      outs(%empty : tensor<64x128xf32>) -> tensor<64x128xf32>
  %result = tensor.empty() : tensor<128xf32>
  %init = linalg.fill ins(%cst : f32) outs(%result : tensor<128xf32>)
      -> tensor<128xf32>
  %reduced = linalg.reduce ins(%mapped : tensor<64x128xf32>)
      outs(%init : tensor<128xf32>) dimensions = [0]
      (%in: f32, %acc: f32) {
        %sum = arith.addf %in, %acc : f32
        linalg.yield %sum : f32
      }
  return %reduced : tensor<128xf32>
}
