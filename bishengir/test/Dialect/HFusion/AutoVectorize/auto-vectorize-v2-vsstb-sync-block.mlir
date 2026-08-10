// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2 -outline-vector-function -split-input-file | FileCheck %s

// The expand_shape has multiple users in block order:
//   1 expand_shape
//   2/3 unrelated ops
//   4 first user
//   5 unrelated op
//   6 sync_block_set/sync_block_wait
//   7 vsstb-pattern transpose
// The sync op before the later transpose must prevent expand_shape from being
// treated as fusible into the vsstb transpose path.

// CHECK-LABEL: func.func @expand_shape_not_fused_when_later_vsstb_user_blocked_by_sync(
// CHECK: %[[EXPANDED_SET:.*]] = tensor.expand_shape
// CHECK: tensor.extract %[[EXPANDED_SET]]
// CHECK: hivm.hir.sync_block_set
// CHECK: func.call @{{.*}}(%[[EXPANDED_SET]],
// CHECK: %[[EXPANDED_WAIT:.*]] = tensor.expand_shape
// CHECK: tensor.extract %[[EXPANDED_WAIT]]
// CHECK: hivm.hir.sync_block_wait
// CHECK: func.call @{{.*}}(%[[EXPANDED_WAIT]],
module {
  func.func @expand_shape_not_fused_when_later_vsstb_user_blocked_by_sync(
      %arg0: tensor<64x64xbf16>, %arg1: tensor<64x64xbf16>)
      -> (bf16, tensor<4x64x16xbf16>, bf16, tensor<4x64x16xbf16>)
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %c0 = arith.constant 0 : index
    %transpose_empty0 = tensor.empty() : tensor<4x64x16xbf16>
    %transpose_empty1 = tensor.empty() : tensor<4x64x16xbf16>

    %expanded_set = tensor.expand_shape %arg0 [[0], [1, 2]] output_shape [64, 4, 16] : tensor<64x64xbf16> into tensor<64x4x16xbf16>
    %first_user_set = tensor.extract %expanded_set[%c0, %c0, %c0] : tensor<64x4x16xbf16>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
    %transposed_set = linalg.transpose ins(%expanded_set : tensor<64x4x16xbf16>) outs(%transpose_empty0 : tensor<4x64x16xbf16>) permutation = [1, 0, 2]

    %expanded_wait = tensor.expand_shape %arg1 [[0], [1, 2]] output_shape [64, 4, 16] : tensor<64x64xbf16> into tensor<64x4x16xbf16>
    %first_user_wait = tensor.extract %expanded_wait[%c0, %c0, %c0] : tensor<64x4x16xbf16>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
    %transposed_wait = linalg.transpose ins(%expanded_wait : tensor<64x4x16xbf16>) outs(%transpose_empty1 : tensor<4x64x16xbf16>) permutation = [1, 0, 2]

    return %first_user_set, %transposed_set, %first_user_wait, %transposed_wait
        : bf16, tensor<4x64x16xbf16>, bf16, tensor<4x64x16xbf16>
  }
}
