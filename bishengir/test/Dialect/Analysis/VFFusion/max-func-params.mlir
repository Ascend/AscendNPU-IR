// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="max-vf-params=4" --split-input-file %s | FileCheck %s --check-prefix=FUSE
// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="max-vf-params=3" --split-input-file %s | FileCheck %s --check-prefix=LIMIT
// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="max-vf-params=-1" --split-input-file %s | FileCheck %s --check-prefix=DISABLE
// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="max-vf-params=6" --split-input-file %s | FileCheck %s --check-prefix=SPLIT
// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="max-vf-params=4" --split-input-file %s | FileCheck %s --check-prefix=LIMIT-SPLIT
// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="max-vf-params=-1" --split-input-file %s | FileCheck %s --check-prefix=DISABLE-SPLIT

// FUSE-LABEL: func.func private @param_budget_boundary_fused_0(
// FUSE-SAME: %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>
// FUSE-LABEL: func.func @param_budget_boundary(
// FUSE: {{(func\.)?call}} @param_budget_boundary_fused_0(

// LIMIT-LABEL: func.func @param_budget_boundary(
// LIMIT-NOT: @param_budget_boundary_fused_0
// LIMIT: linalg.elemwise_binary
// LIMIT: linalg.elemwise_binary

// DISABLE-LABEL: func.func private @param_budget_boundary_fused_0(
// DISABLE-SAME: %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>
// DISABLE-LABEL: func.func @param_budget_boundary(
// DISABLE: {{(func\.)?call}} @param_budget_boundary_fused_0(
module {
  func.func @param_budget_boundary(%arg0: tensor<1xi16>, %arg1: tensor<1xi16>,
                                   %arg2: tensor<1xi16>, %arg3: tensor<1xi16>)
      -> tensor<1xi16> {
    %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<1xi16>, tensor<1xi16>) outs(%arg3 : tensor<1xi16>) -> tensor<1xi16>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%0, %arg2 : tensor<1xi16>, tensor<1xi16>) outs(%arg3 : tensor<1xi16>) -> tensor<1xi16>
    return %1 : tensor<1xi16>
  }
}

// -----

// SPLIT-LABEL: func.func private @param_count_over_limit_fused_0(
// SPLIT-SAME: %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>
// SPLIT-LABEL: func.func @param_count_over_limit(
// SPLIT: %[[FUSED:.*]] = {{(func\.)?call}} @param_count_over_limit_fused_0(
// SPLIT: %[[OUT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[FUSED]], %arg3 : tensor<1xi16>, tensor<1xi16>) outs(%arg6 : tensor<1xi16>) -> tensor<1xi16>
// SPLIT: return %[[OUT]] : tensor<1xi16>

// LIMIT-SPLIT-LABEL: func.func @param_count_over_limit(
// LIMIT-SPLIT-NOT: @param_count_over_limit_fused_0
// LIMIT-SPLIT: linalg.elemwise_binary
// LIMIT-SPLIT: linalg.elemwise_binary
// LIMIT-SPLIT: linalg.elemwise_binary

// DISABLE-SPLIT-LABEL: func.func private @param_count_over_limit_fused_0(
// DISABLE-SPLIT-SAME: %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>
// DISABLE-SPLIT-LABEL: func.func @param_count_over_limit(
// DISABLE-SPLIT: {{(func\.)?call}} @param_count_over_limit_fused_0(
func.func @param_count_over_limit(%arg0: tensor<1xi16>, %arg1: tensor<1xi16>,
                                  %arg2: tensor<1xi16>, %arg3: tensor<1xi16>,
                                  %arg4: tensor<1xi16>, %arg5: tensor<1xi16>,
                                  %arg6: tensor<1xi16>) -> tensor<1xi16> {
  %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<1xi16>, tensor<1xi16>) outs(%arg4 : tensor<1xi16>) -> tensor<1xi16>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%0, %arg2 : tensor<1xi16>, tensor<1xi16>) outs(%arg5 : tensor<1xi16>) -> tensor<1xi16>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %arg3 : tensor<1xi16>, tensor<1xi16>) outs(%arg6 : tensor<1xi16>) -> tensor<1xi16>
  return %2 : tensor<1xi16>
}

// -----
// FUSE-LABEL: func.func private @simple_kernel_fused_0(
// FUSE-LABEL: func.func private @simple_kernel_fused_1(
// FUSE-LABEL: func.func @simple_kernel(
// FUSE: %[[FUSED0:.*]] = {{(func\.)?call}} @simple_kernel_fused_0(
// FUSE: %[[FUSED1:.*]] = {{(func\.)?call}} @simple_kernel_fused_1(
// FUSE: %[[OUT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[FUSED1]], %arg5 : tensor<3x2xf16>, tensor<3x2xf16>) outs(%arg6 : tensor<3x2xf16>) -> tensor<3x2xf16>
// FUSE: return %[[OUT]] : tensor<3x2xf16>

module {
  func.func @simple_kernel(%arg0: tensor<3x2xf16>, %arg1: tensor<3x2xf16>,
                           %arg2: tensor<3x2xf16>, %arg3: tensor<3x2xf16>,
                           %arg4: tensor<3x2xf16>, %arg5: tensor<3x2xf16>,
                           %arg6: tensor<3x2xf16>) -> tensor<3x2xf16>
      attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<3x2xf16>, tensor<3x2xf16>) outs(%arg6 : tensor<3x2xf16>) -> tensor<3x2xf16>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%0, %arg2 : tensor<3x2xf16>, tensor<3x2xf16>) outs(%arg6 : tensor<3x2xf16>) -> tensor<3x2xf16>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %arg3 : tensor<3x2xf16>, tensor<3x2xf16>) outs(%arg6 : tensor<3x2xf16>) -> tensor<3x2xf16>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %arg4 : tensor<3x2xf16>, tensor<3x2xf16>) outs(%arg6 : tensor<3x2xf16>) -> tensor<3x2xf16>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %arg5 : tensor<3x2xf16>, tensor<3x2xf16>) outs(%arg6 : tensor<3x2xf16>) -> tensor<3x2xf16>
    return %4 : tensor<3x2xf16>
  }
}
