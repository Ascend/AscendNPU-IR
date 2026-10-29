// REQUIRES: asserts
// UNSUPPORTED: bishengir_published
// RUN: bishengir-compile -enable-hfusion-compile=true -enable-hivm-compile=true -enable-lir-compile=false -target=Ascend950PR_9579 -debug-only=bishengir-compile-regbase,tree-reduce-v2 %s 2>&1 | FileCheck %s

// CHECK: Detected reduce-only data flow; forcing TreeReduceV2 mode=all with Ascend-style RA
// CHECK: [tree-reduce-v2] === Processing func: None enableRA=1 enableAR=1 ===
// CHECK: Reduction rewrite plan: kind=ascend-style-ra

func.func @reduce_only_first(%arg0: tensor<64x128xf32>) -> tensor<128xf32>
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<128xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<128xf32>)
      -> tensor<128xf32>
  %reduced = linalg.reduce ins(%arg0 : tensor<64x128xf32>)
      outs(%init : tensor<128xf32>) dimensions = [0]
      (%in: f32, %acc: f32) {
        %sum = arith.addf %in, %acc : f32
        linalg.yield %sum : f32
      }
  return %reduced : tensor<128xf32>
}
func.func @reduce_only_second(%arg0: tensor<32x128xf32>) -> tensor<128xf32>
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<128xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<128xf32>)
      -> tensor<128xf32>
  %reduced = linalg.reduce ins(%arg0 : tensor<32x128xf32>)
      outs(%init : tensor<128xf32>) dimensions = [0]
      (%in: f32, %acc: f32) {
        %sum = arith.addf %in, %acc : f32
        linalg.yield %sum : f32
      }
  return %reduced : tensor<128xf32>
}

func.func @reduce_only_structural_outputs(%arg0: tensor<64x128xf32>)
    -> (tensor<128xf32>, tensor<1x128xf32>)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<128xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<128xf32>)
      -> tensor<128xf32>
  %reduced = linalg.reduce ins(%arg0 : tensor<64x128xf32>)
      outs(%init : tensor<128xf32>) dimensions = [0]
      (%in: f32, %acc: f32) {
        %sum = arith.addf %in, %acc : f32
        linalg.yield %sum : f32
      }
  %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [1, 128] :
      tensor<128xf32> into tensor<1x128xf32>
  return %reduced, %expanded : tensor<128xf32>, tensor<1x128xf32>
}
