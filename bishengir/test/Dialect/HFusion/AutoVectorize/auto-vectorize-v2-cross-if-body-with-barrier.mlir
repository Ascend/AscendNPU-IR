// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2="enable-cross-if-fusion=true enable-multiple-consumer-fusion=false" -split-input-file | FileCheck %s
// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2="enable-cross-if-fusion=false enable-multiple-consumer-fusion=false" -split-input-file | FileCheck %s --check-prefix=CLOSE

// -----
// body_with_sync_barrier: scf.for body contains sync set.
//
// CLOSE-LABEL: func.func @body_with_sync_barrier
// CLOSE-COUNT-3: outlined-loop-target

// CHECK-LABEL: func.func @body_with_sync_barrier
// CHECK-COUNT-3: outlined-loop-target

#map = affine_map<(d0) -> (d0)>
module {
  func.func @body_with_sync_barrier(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %empty = tensor.empty() : tensor<128xf32>
    %producer = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<128xf32>) outs(%empty : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = math.exp %in : f32
      linalg.yield %v : f32
    } -> tensor<128xf32>

    %user0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%producer, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %v = arith.addf %in, %in_0 : f32
      linalg.yield %v : f32
    } -> tensor<128xf32>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c1 step %c1 {
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
    }

    %user1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%producer, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %v = arith.subf %in, %in_0 : f32
      linalg.yield %v : f32
    } -> tensor<128xf32>

    return %user0, %user1 : tensor<128xf32>, tensor<128xf32>
  }
}

// -----
// body_with_anchor_barrier: scf.if body contains anchor.
//
// CHECK-LABEL: func.func @body_with_anchor_barrier
// CHECK-COUNT-3: outlined-loop-target

// CLOSE-LABEL: func.func @body_with_anchor_barrier
// CLOSE-COUNT-3: outlined-loop-target

#map = affine_map<(d0) -> (d0)>
module {
  func.func @body_with_anchor_barrier(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %cond: i1) -> (tensor<128xf32>, tensor<128xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %empty = tensor.empty() : tensor<128xf32>
    %producer = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<128xf32>) outs(%empty : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = math.exp %in : f32
      linalg.yield %v : f32
    } -> tensor<128xf32>

    %user0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%producer, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %v = arith.addf %in, %in_0 : f32
      linalg.yield %v : f32
    } -> tensor<128xf32>

    %if = scf.if %cond -> (tensor<128xf32>) {
      hivm.hir.anchor {id = 0 : i64}
      scf.yield %producer : tensor<128xf32>
    } else {
      scf.yield %producer : tensor<128xf32>
    }

    %user1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%producer, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %v = arith.subf %in, %in_0 : f32
      linalg.yield %v : f32
    } -> tensor<128xf32>

    return %user0, %user1 : tensor<128xf32>, tensor<128xf32>
  }
}

// -----
// body_with_copy_barrier: scf.for body contains copy.
//
// CHECK-LABEL: func.func @body_with_copy_barrier
// CHECK-COUNT-1: outlined-loop-target
// CHECK-NOT: outlined-loop-target

// CLOSE-LABEL: func.func @body_with_copy_barrier
// CLOSE-COUNT-3: outlined-loop-target

#map = affine_map<(d0) -> (d0)>
module {
  func.func @body_with_copy_barrier(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %empty = tensor.empty() : tensor<128xf32>
    %producer = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<128xf32>) outs(%empty : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = math.exp %in : f32
      linalg.yield %v : f32
    } -> tensor<128xf32>

    %user0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%producer, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %v = arith.addf %in, %in_0 : f32
      linalg.yield %v : f32
    } -> tensor<128xf32>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %empty_1 = tensor.empty() : tensor<128xf32>
    %alloc = memref.alloc() : memref<128xf32, #hivm.address_space<cbuf>>
    %memspacecast = memref.memory_space_cast %alloc : memref<128xf32, #hivm.address_space<cbuf>> to memref<128xf32>
    scf.for %i = %c0 to %c1 step %c1 {
      hivm.hir.copy ins(%empty_1 : tensor<128xf32>) outs(%memspacecast : memref<128xf32>)
    }

    %user1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%producer, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %v = arith.subf %in, %in_0 : f32
      linalg.yield %v : f32
    } -> tensor<128xf32>

    return %user0, %user1 : tensor<128xf32>, tensor<128xf32>
  }
}
