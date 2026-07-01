// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2 2>&1 | FileCheck %s

// CHECK-NOT: falling back to legacy HFusionAutoVectorize pass

#map = affine_map<(d0) -> (d0)>
module {
  func.func @kernel(%arg0: memref<64xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %a = tensor.empty() : tensor<64xf32>
    %b = tensor.empty() : tensor<64xf32>
    %result = scope.scope : () -> tensor<64xf32> {
      %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%a : tensor<64xf32>) outs(%b : tensor<64xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1 = arith.addf %in, %in : f32
        linalg.yield %1 : f32
      } -> tensor<64xf32>
      scope.return %0 : tensor<64xf32>
    }
    hivm.hir.store ins(%result : tensor<64xf32>) outs(%arg0 : memref<64xf32>)
    return
  }
}
