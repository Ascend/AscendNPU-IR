// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="fusion-mode=max-parallel" --split-input-file %s | FileCheck %s

// Test: insert_slice is in isInFusionWhiteList. isInsertSliceFusionAllowed
// rejects insert_slice as producer (always). As consumer it is admitted via
// the bridge case: when the producer also feeds a later linalg user. Here mul
// feeds both insert_slice and sub (after insert_slice in block order), so the
// bridge allows mul+insert_slice to fuse into fused_0. add (consuming
// insert_slice's result) and sub stay in the caller because
// insert_slice-as-producer is always rejected.

// CHECK-LABEL: func.func private @insert_slice_bridge_fused_0(
// CHECK: linalg.elemwise_binary
// CHECK: tensor.insert_slice
// CHECK: return

// CHECK-LABEL: func.func @insert_slice_bridge(
// CHECK: call @insert_slice_bridge_fused_0
// CHECK: linalg.elemwise_binary
// CHECK: linalg.elemwise_binary
// CHECK: return

module {
  func.func @insert_slice_bridge(%arg0: tensor<64x1xf32>, %arg1: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x1xf32>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = tensor.empty() : tensor<64x1xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg0 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%0 : tensor<64x1xf32>) -> tensor<64x1xf32>
    %2 = tensor.empty() : tensor<64x64xf32>
    %3 = tensor.insert_slice %1 into %2[0, 0] [64, 1] [1, 1] : tensor<64x1xf32> into tensor<64x64xf32>
    %4 = tensor.empty() : tensor<64x64xf32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %arg1 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %6 = tensor.empty() : tensor<64x1xf32>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%1, %arg0 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%6 : tensor<64x1xf32>) -> tensor<64x1xf32>
    return %5, %7 : tensor<64x64xf32>, tensor<64x1xf32>
  }
}
