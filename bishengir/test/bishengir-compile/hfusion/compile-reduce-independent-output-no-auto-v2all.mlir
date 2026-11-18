// REQUIRES: asserts
// UNSUPPORTED: bishengir_published
// RUN: bishengir-compile -enable-hfusion-compile=true -enable-hivm-compile=true -enable-lir-compile=false -target=Ascend950PR_9579 -debug-only=bishengir-compile-regbase,tree-reduce-v2 %s 2>&1 | FileCheck %s

// CHECK: Not reduce-only: external output has a non-reduction data contributor
// CHECK-NOT: Detected reduce-only data flow
// CHECK-NOT: [tree-reduce-v2] === Processing func: None enableRA=1 enableAR=1 ===

func.func @reduce_with_passthrough_output(
    %arg0: tensor<64x128xf32>, %reducedDest: memref<128xf32>,
    %passthroughDest: memref<64x128xf32>)
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
  bufferization.materialize_in_destination %reduced in writable %reducedDest :
      (tensor<128xf32>, memref<128xf32>) -> ()
  bufferization.materialize_in_destination %arg0 in writable %passthroughDest :
      (tensor<64x128xf32>, memref<64x128xf32>) -> ()
  return
}
