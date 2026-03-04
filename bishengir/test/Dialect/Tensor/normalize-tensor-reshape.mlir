// RUN: bishengir-opt --canonicalize-tensor-reshape -split-input-file %s | FileCheck %s

// CHECK-LABEL:   func.func @reshape_normalize(
// CHECK:           %[[VAL_5:.*]] = tensor.collapse_shape %{{.*}} {{\[\[}}0, 1], [2]] : tensor<1x?x4096xf32> into tensor<?x4096xf32>
// CHECK:           return %[[VAL_5]] : tensor<?x4096xf32>
// CHECK:         }
func.func @reshape_normalize(%arg0: tensor<1x?x4096xbf16>, %arg1: i64, %arg2: tensor<1x24576xbf16>, %arg3: i64, %arg4: tensor<1x?x4096xf32>) -> tensor<?x4096xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %from_elements = tensor.from_elements %arg1, %arg3 : tensor<2xi64>
    %reshape = tensor.reshape %arg4(%from_elements) : (tensor<1x?x4096xf32>, tensor<2xi64>) -> tensor<?x4096xf32>
    return %reshape : tensor<?x4096xf32>
}

// -----
// CHECK-LABEL: all_static
// CHECK: collapse_shape
// CHECK-SAME: {{\[\[}}0, 1], [2]]
func.func @all_static(%arg0: tensor<16x4x64xf16>, %arg1: tensor<2xi64>) -> tensor<64x64xf16> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix"} {
  %reshape = tensor.reshape %arg0(%arg1) : (tensor<16x4x64xf16>, tensor<2xi64>) -> tensor<64x64xf16>
  return %reshape : tensor<64x64xf16>
}

// -----
// CHECK-LABEL: non_inferrable_dynamic
// CHECK-LABEL: reshape
func.func @non_inferrable_dynamic(%arg0: tensor<16x4x64xf16>, %arg1: tensor<2xi64>) -> tensor<?x64xf16> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix"} {
  %reshape = tensor.reshape %arg0(%arg1) : (tensor<16x4x64xf16>, tensor<2xi64>) -> tensor<?x64xf16>
  return %reshape : tensor<?x64xf16>
}