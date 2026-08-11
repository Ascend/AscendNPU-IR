// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="fusion-mode=max-parallel" --split-input-file %s | FileCheck %s

// Test: tensor.insert_slice is in isInFusionWhiteList, so it is admitted into
// the same fused VF group as its producer/consumer linalg ops instead of
// splitting the group. Before the white-list change, the insert_slice edge
// blocked fusion across it.

// CHECK-LABEL: func.func private @insert_slice_in_group_fused_0(
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: tensor.insert_slice
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: return
// CHECK-LABEL: func.func @insert_slice_in_group(
// CHECK: call @insert_slice_in_group_fused_0
// CHECK-NOT: call @insert_slice_in_group_fused_{{[1-9]}}

module {
  func.func @insert_slice_in_group(%arg0: tensor<64x1xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = tensor.empty() : tensor<64x1xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg0 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%0 : tensor<64x1xf32>) -> tensor<64x1xf32>
    %2 = tensor.empty() : tensor<64x64xf32>
    %3 = tensor.insert_slice %1 into %2[0, 0] [64, 1] [1, 1] : tensor<64x1xf32> into tensor<64x64xf32>
    %4 = tensor.empty() : tensor<64x64xf32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %arg1 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %5 : tensor<64x64xf32>
  }
}
