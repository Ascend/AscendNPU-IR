// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="fusion-mode=max-parallel" --split-input-file %s | FileCheck %s

// Test: tensor.collapse_shape feeding linalg.broadcast should not block
// VFFusion. PreVectorizationFusion's generalizeBroadcastOp would create an
// inverse expand_shape on the collapse; MLIR fold then cancels
// expand_shape(collapse_shape(x)) -> x, leaving the collapse dead.
// isReshapeEliminableByPreVectorizationFusion detects this pattern and
// admits the collapse into the fusion group.

// CHECK-LABEL: func.func private @collapse_broadcast_fused_0(
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: tensor.collapse_shape
// CHECK: linalg.broadcast
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: return
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

// Positive: expand_shape consumed by a named elementwise op (linalg.mul) that
// has a constant operand. The constant broadcasts, so the expand's unit dim is
// removable; isExpandShapeEliminable admits the expand into the VF.

// CHECK-LABEL: func.func private @expand_named_const_fused_0(
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: tensor.expand_shape
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: return
// CHECK-LABEL: func.func @expand_named_const(
// CHECK: call @expand_named_const_fused_0
// CHECK-NOT: call @expand_named_const_fused_{{[1-9]}}
module {
  func.func @expand_named_const(%arg0: tensor<64xf32>, %arg1: tensor<64x1xf32>) -> tensor<64x1xf32> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = tensor.empty() : tensor<64xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg0 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %expanded = tensor.expand_shape %1 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
    %cst = arith.constant 1.000000e+00 : f32
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expanded, %cst : tensor<64x1xf32>, f32) outs(%arg1 : tensor<64x1xf32>) -> tensor<64x1xf32>
    return %2 : tensor<64x1xf32>
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
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.generic
// CHECK: return
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

// Negative: expand_shape consumed by a linalg.generic with NO constant
// operand. Under the constant-operand gate the generic is rejected (even
// though its map has a constant-0 broadcast axis), so the expand stays in
// the caller and is NOT fused into a VF.

// CHECK-LABEL: func.func @expand_generic
// CHECK: tensor.expand_shape
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

// Negative: expand_shape consumed by a named elementwise op (mul) with only
// tensor operands (no constant). Every operand carries the unit dim, so the
// expand is not removable; isExpandShapeEliminable rejects it.

// CHECK-LABEL: func.func @expand_mul_no_const
// CHECK: tensor.expand_shape
module {
  func.func @expand_mul_no_const(%arg0: tensor<64xf32>, %arg1: tensor<64x1xf32>) -> tensor<64x1xf32> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
    %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expanded, %arg1 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%arg1 : tensor<64x1xf32>) -> tensor<64x1xf32>
    return %0 : tensor<64x1xf32>
  }
}

// -----

// Negative: expand_shape consumed by hfusion.cast. cast is a type-conversion
// op with a single input (the expand) and no constant operand, so the
// constant-operand gate rejects it; the expand stays in the caller.

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
// single input, no constant operand -> rejected.

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

// Negative: expand_shape has TWO users — an admissible mul (with a constant
// operand) and a func.return (not a LinalgOp). all_of rejects the expand
// because not EVERY user is admissible; the return keeps it alive, so it
// stays in the caller even though the mul alone would be admissible.

// CHECK-LABEL: func.func @expand_multi_user
// CHECK: tensor.expand_shape
module {
  func.func @expand_multi_user(%arg0: tensor<64xf32>, %arg1: tensor<64x1xf32>) -> (tensor<64x1xf32>, tensor<64x1xf32>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
    %cst = arith.constant 1.000000e+00 : f32
    %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expanded, %cst : tensor<64x1xf32>, f32) outs(%arg1 : tensor<64x1xf32>) -> tensor<64x1xf32>
    return %0, %expanded : tensor<64x1xf32>, tensor<64x1xf32>
  }
}
