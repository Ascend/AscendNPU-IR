// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2 2>&1 | FileCheck %s

// UNSUPPORTED: bishengir_published

// CHECK: warning: AutoVectorizeV2 failed; falling back to legacy HFusionAutoVectorize pass

#map = affine_map<(d0) -> (d0)>
module {
  func.func @kernel(%arg0: memref<64xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<64xf32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : tensor<64xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<64xf32>
    %empty0 = tensor.empty() : tensor<64xf32>
    %2 = scope.scope : () -> tensor<64xf32> {
      %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%1 : tensor<64xf32>) outs(%empty0 : tensor<64xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %cst : f32
      } -> tensor<64xf32>
      scope.return %4 : tensor<64xf32>
    }
    %empty1 = tensor.empty() : tensor<64xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%2 : tensor<64xf32>) outs(%empty1 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %cst : f32
    } -> tensor<64xf32>
    hivm.hir.store ins(%3 : tensor<64xf32>) outs(%arg0 : memref<64xf32>) {tiled_op}
    return
  }
}
