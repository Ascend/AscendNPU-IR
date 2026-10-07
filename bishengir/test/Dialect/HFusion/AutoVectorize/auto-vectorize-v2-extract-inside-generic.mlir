// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2 2>&1 | FileCheck %s
// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2="emit-transform-sequence=true" | FileCheck %s --check-prefix=SEQ

// A generic whose only user is tensor.extract nested in another generic is
// not an independent sibling. loop.fuse_sibling rejects that pair.

// CHECK-NOT: AutoVectorizeV2 failed
// CHECK-NOT: user of results of target should be properly dominated by source
// CHECK-LABEL: func.func @gather_extract_not_sibling
// CHECK: scf.for
// CHECK: arith.sitofp
// CHECK: scf.for

// SEQ-LABEL: transform.sequence {{.*}}gather_extract_not_sibling
// SEQ: transform.structured.tile_using_for
// SEQ: transform.structured.tile_using_for
// SEQ-NOT: transform.loop.fuse_sibling

#map = affine_map<(d0) -> (d0)>
module {
  func.func @gather_extract_not_sibling(
      %idx: tensor<256xi32>, %vals: tensor<256xi32>) -> tensor<256xi32>
      attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %empty_f32 = tensor.empty() : tensor<256xf32>
    %casted = linalg.generic
        {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
        ins(%vals : tensor<256xi32>) outs(%empty_f32 : tensor<256xf32>) {
      ^bb0(%in: i32, %out: f32):
        %c = arith.sitofp %in : i32 to f32
        linalg.yield %c : f32
    } -> tensor<256xf32>
    %empty_i32 = tensor.empty() : tensor<256xi32>
    %gathered = linalg.generic
        {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
        ins(%idx : tensor<256xi32>) outs(%empty_i32 : tensor<256xi32>) {
      ^bb0(%in: i32, %out: i32):
        %i = arith.index_castui %in : i32 to index
        %extracted = tensor.extract %casted[%i] : tensor<256xf32>
        %v = arith.fptosi %extracted : f32 to i32
        linalg.yield %v : i32
    } -> tensor<256xi32>
    return %gathered : tensor<256xi32>
  }
}
