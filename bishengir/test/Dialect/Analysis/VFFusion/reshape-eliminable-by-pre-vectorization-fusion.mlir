// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="fusion-mode=max-parallel" --split-input-file %s | FileCheck %s

// Test: tensor.collapse_shape feeding linalg.broadcast should not block
// VFFusion. PreVectorizationFusion's generalizeBroadcastOp would create an
// inverse expand_shape on the collapse; MLIR fold then cancels
// expand_shape(collapse_shape(x)) -> x, leaving the collapse dead.
// isReshapeEliminableByPreVectorizationFusion detects this pattern and
// admits the collapse into the fusion group.

// CHECK-LABEL: func.func private @collapse_broadcast_fused_0(
// CHECK: tensor.collapse_shape
// CHECK-LABEL: func.func @collapse_broadcast(
// CHECK: call @collapse_broadcast_fused_0
// CHECK-NOT: call @collapse_broadcast_fused_{{[1-9]}}
module {
  func.func @collapse_broadcast(%arg0: tensor<64x1xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = tensor.empty() : tensor<64x1xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg0 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%0 : tensor<64x1xf32>) -> tensor<64x1xf32>
    %collapsed = tensor.collapse_shape %1 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
    %2 = tensor.empty() : tensor<64x64xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<64xf32>) outs(%2 : tensor<64x64xf32>) dimensions = [1]
    %3 = tensor.empty() : tensor<64x64xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%broadcasted, %arg1 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%3 : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %4 : tensor<64x64xf32>
  }
}

// -----

// Test: tensor.expand_shape consumed by linalg.generic where the expand
// only inserts a unit dim and that unit dim is indexed by constant 0 in the
// generic's indexing map. PreVectorizationFusion's
// ExpandShapeToImplicitBrcInGenericPattern would fold the expand into the
// indexing map. isReshapeEliminableByPreVectorizationFusion detects this
// pattern and admits the expand into the fusion group.

// CHECK-LABEL: func.func private @expand_generic_fused_0(
// CHECK: tensor.expand_shape
// CHECK-LABEL: func.func @expand_generic(
// CHECK: call @expand_generic_fused_0
// CHECK-NOT: call @expand_generic_fused_{{[1-9]}}
module {
  func.func @expand_generic(%arg0: tensor<64xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = tensor.empty() : tensor<64xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg0 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %expanded = tensor.expand_shape %1 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
    %2 = tensor.empty() : tensor<64x64xf32>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<64x1xf32>) outs(%2 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x64xf32>
    return %3 : tensor<64x64xf32>
  }
}

// -----

// Test: expand_shape whose source is from collapse_shape (inverse reshape
// pair). VFFusion::preProcess() calls applyPatternsGreedily, whose
// GreedyPatternRewriteDriver fold step invokes ExpandShapeOp::fold(),
// canceling expand(collapse) -> x BEFORE the fusion phase. The reshapes
// are gone by the time areReshapesValidIfFused runs, so the surrounding
// ops fuse into a single VF without any reshape blocking.

// CHECK-LABEL: func.func private @expand_from_collapse_fused_0(
// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: tensor.expand_shape
// CHECK-LABEL: func.func @expand_from_collapse(
// CHECK: call @expand_from_collapse_fused_0
// CHECK-NOT: call @expand_from_collapse_fused_{{[1-9]}}
module {
  func.func @expand_from_collapse(%arg0: tensor<64x1xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = tensor.empty() : tensor<64x1xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg0 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%0 : tensor<64x1xf32>) -> tensor<64x1xf32>
    %collapsed = tensor.collapse_shape %1 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
    %expanded = tensor.expand_shape %collapsed [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
    %2 = tensor.empty() : tensor<64x64xf32>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<64x1xf32>) outs(%2 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x64xf32>
    return %3 : tensor<64x64xf32>
  }
}

// -----

// Negative: expand_shape consumed by hfusion.cast. hfusion.cast implements the
// LinalgOp interface but is a type-conversion op that
// ExpandShapeToImplicitBrcInGenericPattern does not fold (identity map, no
// constant-0 broadcast axis). isExpandShapeEliminable rejects it, so the expand
// is NOT fused into a VF and stays in the caller.

// CHECK-LABEL: func.func @expand_cast
// CHECK: tensor.expand_shape
module {
  func.func @expand_cast(%arg0: tensor<64xf32>, %arg1: tensor<64x1xf16>) -> tensor<64x1xf16> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
    %0 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%expanded : tensor<64x1xf32>) outs(%arg1 : tensor<64x1xf16>) -> tensor<64x1xf16>
    return %0 : tensor<64x1xf16>
  }
}

// -----

// Negative: expand_shape consumed by hfusion.bitcast. Same rationale as cast:
// bitcast is a type-conversion op that the reshape-folding pattern does not fold.

// CHECK-LABEL: func.func @expand_bitcast
// CHECK: tensor.expand_shape
module {
  func.func @expand_bitcast(%arg0: tensor<64xf32>, %arg1: tensor<64x1xi32>) -> tensor<64x1xi32> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
    %0 = hfusion.bitcast ins(%expanded : tensor<64x1xf32>) outs(%arg1 : tensor<64x1xi32>) -> tensor<64x1xi32>
    return %0 : tensor<64x1xi32>
  }
}

// -----

// Negative: expand_shape has TWO users — a foldable linalg.generic (broadcast
// map, which alone WOULD be admitted) and a func.return (not foldable). all_of
// rejects the expand because not EVERY user is foldable, so it stays in the
// caller even though one user is foldable.

// CHECK-LABEL: func.func @expand_multi_user
// CHECK: tensor.expand_shape
module {
  func.func @expand_multi_user(%arg0: tensor<64xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x1xf32>) -> (tensor<64x64xf32>, tensor<64x1xf32>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<64x1xf32>) outs(%0 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x64xf32>
    return %1, %expanded : tensor<64x64xf32>, tensor<64x1xf32>
  }
}
