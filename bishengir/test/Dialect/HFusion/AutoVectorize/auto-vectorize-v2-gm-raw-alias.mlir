// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2 --outline-vector-function \
// RUN: -split-input-file | FileCheck %s

// This test verifies that AutoVectorizeV2 does NOT fuse a producer
// (linalg.generic subf) and a consumer (linalg.generic mul) into the same VF
// when they are connected through a GM-level RAW alias:
//   producer → bufferization.materialize_in_destination → GM[addr]
//   GM[addr] → memref.load → consumer
//
// Without the GM RAW alias check in computeConflictLists, AutoVectorizeV2
// would fuse producer and consumer into one VF, creating a cyclic dependency:
// the VF's input (memref.load) must be filled before the VF runs, but the VF's
// output (materialize_in_destination) is only available after the VF runs,
// and both access the same GM address — the load reads stale (uninitialized)
// data.

// CHECK-LABEL: func.func @gm_raw_alias_no_fuse
// CHECK-COUNT-2: func.call @gm_raw_alias_no_fuse_outlined_vf_
func.func @gm_raw_alias_no_fuse(
    %arg0: memref<1xf32>,        // GM: max_score
    %arg1: memref<1xf32>,        // GM: bias
    %arg2: memref<1xf32>,        // GM: topk_weights (read-write!)
    %scale: f32
) -> () attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  %f1 = arith.constant 1.0 : f32

  // LOOP1: compute w = max_score - bias, store to GM.
  %empty = tensor.empty() : tensor<1xf32>
  %w_tensor = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } outs(%empty : tensor<1xf32>) {
  ^bb0(%out: f32):
    %w = arith.subf %f1, %f0 : f32
    linalg.yield %w : f32
  } -> tensor<1xf32>
  // materialize w into GM[%arg2] (LOOP1 store)
  bufferization.materialize_in_destination %w_tensor in writable %arg2 : (tensor<1xf32>, memref<1xf32>) -> ()

  // LOOP2: load w from GM, multiply by scale, store back.
  // memref.load from the SAME GM address that was just written → RAW alias
  %loaded = memref.load %arg2[%c0] : memref<1xf32>
  %loaded_tensor = tensor.insert %loaded into %w_tensor[%c0] : tensor<1xf32>
  %scale_tensor = tensor.insert %scale into %w_tensor[%c0] : tensor<1xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%loaded_tensor, %scale_tensor : tensor<1xf32>, tensor<1xf32>)
    outs(%w_tensor : tensor<1xf32>) {
    ^bb0(%in: f32, %sc: f32, %out: f32):
      %r = arith.mulf %in, %sc : f32
      linalg.yield %r : f32
  } -> tensor<1xf32>
  // materialize result into GM[%arg2] (LOOP2 store)
  bufferization.materialize_in_destination %result in writable %arg2 : (tensor<1xf32>, memref<1xf32>) -> ()
  return
}

// CHECK-LABEL: func.func @gm_raw_alias_distinct_views_no_fuse
// CHECK-COUNT-2: func.call @gm_raw_alias_distinct_views_no_fuse_outlined_vf_
func.func @gm_raw_alias_distinct_views_no_fuse(
    %arg0: memref<?xf32>,
    %scale: f32
) -> () attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  %f1 = arith.constant 1.0 : f32
  %write_view = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1], strides: [1]
      : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
  %read_view = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1], strides: [1]
      : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>

  %empty = tensor.empty() : tensor<1xf32>
  %producer = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } outs(%empty : tensor<1xf32>) {
  ^bb0(%out: f32):
    %w = arith.subf %f1, %f0 : f32
    linalg.yield %w : f32
  } -> tensor<1xf32>
  bufferization.materialize_in_destination %producer in writable %write_view
      : (tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>) -> ()

  %loaded = memref.load %read_view[%c0] : memref<1xf32, strided<[1], offset: ?>>
  %loaded_tensor = tensor.insert %loaded into %producer[%c0] : tensor<1xf32>
  %scale_tensor = tensor.insert %scale into %producer[%c0] : tensor<1xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%loaded_tensor, %scale_tensor : tensor<1xf32>, tensor<1xf32>)
    outs(%producer : tensor<1xf32>) {
    ^bb0(%in: f32, %sc: f32, %out: f32):
      %r = arith.mulf %in, %sc : f32
      linalg.yield %r : f32
  } -> tensor<1xf32>
  bufferization.materialize_in_destination %result in writable %read_view
      : (tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>) -> ()
  return
}
