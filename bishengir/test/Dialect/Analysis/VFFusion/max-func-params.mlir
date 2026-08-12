// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="max-vf-params=4" --split-input-file %s | FileCheck %s --check-prefix=FUSE
// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="max-vf-params=3" --split-input-file %s | FileCheck %s --check-prefix=LIMIT
// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="max-vf-params=-1" --split-input-file %s | FileCheck %s --check-prefix=DISABLE
// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="max-vf-params=6" --split-input-file %s | FileCheck %s --check-prefix=SPLIT
// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="max-vf-params=4" --split-input-file %s | FileCheck %s --check-prefix=LIMIT-SPLIT
// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="max-vf-params=-1" --split-input-file %s | FileCheck %s --check-prefix=DISABLE-SPLIT

// FUSE-LABEL: func.func private @param_budget_boundary_fused_0(
// FUSE-SAME: %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>
// FUSE-LABEL: func.func @param_budget_boundary(
// FUSE-SAME: %[[ARG0:.*]]: tensor{{.*}}, %[[ARG1:.*]]: tensor{{.*}}, %[[ARG2:.*]]: tensor{{.*}}, %[[ARG3:.*]]: tensor{{.*}}
// FUSE: %[[FUSED0:.*]] = {{(func\.)?call}} @param_budget_boundary_fused_0(%[[ARG0]], %[[ARG1]], %[[ARG3]], %[[ARG2]])
// FUSE: return %[[FUSED0]] : tensor<1xi16>

// LIMIT-LABEL: func.func @param_budget_boundary(
// LIMIT-SAME: %[[ARG0:.*]]: tensor{{.*}}, %[[ARG1:.*]]: tensor{{.*}}, %[[ARG2:.*]]: tensor{{.*}}, %[[ARG3:.*]]: tensor{{.*}}
// LIMIT: linalg.elemwise_binary
// LIMIT: linalg.elemwise_binary
// LIMIT: return

// DISABLE-LABEL: func.func private @param_budget_boundary_fused_0(
// DISABLE-SAME: %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>
// DISABLE-LABEL: func.func @param_budget_boundary(
// DISABLE-SAME: %[[ARG0:.*]]: tensor{{.*}}, %[[ARG1:.*]]: tensor{{.*}}, %[[ARG2:.*]]: tensor{{.*}}, %[[ARG3:.*]]: tensor{{.*}}
// DISABLE: %[[FUSED0:.*]] = {{(func\.)?call}} @param_budget_boundary_fused_0(%[[ARG0]], %[[ARG1]], %[[ARG3]], %[[ARG2]])
// DISABLE: return %[[FUSED0]] : tensor<1xi16>
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
// SPLIT-SAME: %[[ARG0:.*]]: tensor{{.*}}, %[[ARG1:.*]]: tensor{{.*}}, %[[ARG2:.*]]: tensor{{.*}}, %[[ARG3:.*]]: tensor{{.*}}, %[[ARG4:.*]]: tensor{{.*}}, %[[ARG5:.*]]: tensor{{.*}}, %[[ARG6:.*]]: tensor{{.*}}
// SPLIT: %[[FUSED:.*]] = {{(func\.)?call}} @param_count_over_limit_fused_0(%[[ARG0]], %[[ARG1]], %[[ARG4]], %[[ARG2]], %[[ARG5]])
// SPLIT: %[[OUT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[FUSED]], %[[ARG3]] : tensor{{.*}}, tensor{{.*}}) outs(%[[ARG6]] : tensor{{.*}}) -> tensor{{.*}}
// SPLIT: return %[[OUT]] : tensor<1xi16>

// LIMIT-SPLIT-LABEL: func.func @param_count_over_limit(
// LIMIT-SPLIT-SAME: %[[ARG0:.*]]: tensor{{.*}}, %[[ARG1:.*]]: tensor{{.*}}, %[[ARG2:.*]]: tensor{{.*}}, %[[ARG3:.*]]: tensor{{.*}}, %[[ARG4:.*]]: tensor{{.*}}, %[[ARG5:.*]]: tensor{{.*}}, %[[ARG6:.*]]: tensor{{.*}}
// LIMIT-SPLIT: linalg.elemwise_binary
// LIMIT-SPLIT: linalg.elemwise_binary
// LIMIT-SPLIT: linalg.elemwise_binary
// LIMIT-SPLIT: return

// DISABLE-SPLIT-LABEL: func.func private @param_count_over_limit_fused_0(
// DISABLE-SPLIT-SAME: %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>, %{{.*}}: tensor<1xi16>
// DISABLE-SPLIT-LABEL: func.func @param_count_over_limit(
// DISABLE-SPLIT-SAME: %[[ARG0:.*]]: tensor{{.*}}, %[[ARG1:.*]]: tensor{{.*}}, %[[ARG2:.*]]: tensor{{.*}}, %[[ARG3:.*]]: tensor{{.*}}, %[[ARG4:.*]]: tensor{{.*}}, %[[ARG5:.*]]: tensor{{.*}}, %[[ARG6:.*]]: tensor{{.*}}
// DISABLE-SPLIT: %[[FUSED0:.*]] = {{(func\.)?call}} @param_count_over_limit_fused_0(%[[ARG0]], %[[ARG1]], %[[ARG4]], %[[ARG2]], %[[ARG5]], %[[ARG3]], %[[ARG6]])
// DISABLE-SPLIT: return %[[FUSED0]] : tensor<1xi16>
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
// FUSE-SAME: %{{.*}}: tensor{{.*}}, %{{.*}}: tensor{{.*}}, %{{.*}}: tensor{{.*}}
// FUSE-LABEL: func.func private @simple_kernel_fused_1(
// FUSE-SAME: %{{.*}}: tensor{{.*}}, %{{.*}}: tensor{{.*}}, %{{.*}}: tensor{{.*}}, %{{.*}}: tensor{{.*}}
// FUSE-LABEL: func.func @simple_kernel(
// FUSE-SAME: %[[ARG0:.*]]: tensor{{.*}}, %[[ARG1:.*]]: tensor{{.*}}, %[[ARG2:.*]]: tensor{{.*}}, %[[ARG3:.*]]: tensor{{.*}}, %[[ARG4:.*]]: tensor{{.*}}, %[[ARG5:.*]]: tensor{{.*}}, %[[ARG6:.*]]: tensor{{.*}}
// FUSE: %[[FUSED0:.*]] = {{(func\.)?call}} @simple_kernel_fused_0(%[[ARG0]], %[[ARG1]], %[[ARG6]], %[[ARG2]])
// FUSE: %[[FUSED1:.*]] = {{(func\.)?call}} @simple_kernel_fused_1(%[[FUSED0]], %[[ARG3]], %[[ARG6]], %[[ARG4]])
// FUSE: %[[OUT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[FUSED1]], %[[ARG5]] : tensor{{.*}}, tensor{{.*}}) outs(%[[ARG6]] : tensor{{.*}}) -> tensor{{.*}}
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

// -----
// FUSE-LABEL: func.func private @split_candidates_keep_parent_anchor_fused_0(
// FUSE-SAME: %{{.*}}: tensor{{.*}}, %{{.*}}: tensor{{.*}}, %{{.*}}: tensor{{.*}}, %{{.*}}: tensor{{.*}}
// FUSE-LABEL: func.func private @split_candidates_keep_parent_anchor_fused_1(
// FUSE-SAME: %{{.*}}: tensor{{.*}}, %{{.*}}: tensor{{.*}}, %{{.*}}: tensor{{.*}}, %{{.*}}: tensor{{.*}}
// FUSE-LABEL: func.func @split_candidates_keep_parent_anchor(
// FUSE-SAME: %[[ARG0:.*]]: tensor<1xi16>, %[[ARG1:.*]]: tensor<1xi16>, %[[ARG2:.*]]: tensor<1xi16>, %[[ARG3:.*]]: tensor<1xi16>, %[[ARG4:.*]]: tensor<1xi16>, %[[ARG5:.*]]: tensor<1xi16>
// FUSE: %[[ANCHOR:.*]] = arith.constant 0 : i32
// FUSE: %[[FIRST:.*]] = {{(func\.)?call}} @split_candidates_keep_parent_anchor_fused_0(%[[ARG0]], %[[ARG1]], %[[ARG5]], %[[ARG2]])
// FUSE: %[[SECOND:.*]] = {{(func\.)?call}} @split_candidates_keep_parent_anchor_fused_1(%[[FIRST]], %[[ARG3]], %[[ARG5]], %[[ARG4]])
// FUSE: return %[[SECOND]], %[[ANCHOR]] : tensor<1xi16>, i32
func.func @split_candidates_keep_parent_anchor(
    %arg0: tensor<1xi16>, %arg1: tensor<1xi16>,
    %arg2: tensor<1xi16>, %arg3: tensor<1xi16>,
    %arg4: tensor<1xi16>, %arg5: tensor<1xi16>)
    -> (tensor<1xi16>, i32) {
  %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<1xi16>, tensor<1xi16>) outs(%arg5 : tensor<1xi16>) -> tensor<1xi16>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%0, %arg2 : tensor<1xi16>, tensor<1xi16>) outs(%arg5 : tensor<1xi16>) -> tensor<1xi16>
  %c0_i32 = arith.constant 0 : i32
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %arg3 : tensor<1xi16>, tensor<1xi16>) outs(%arg5 : tensor<1xi16>) -> tensor<1xi16>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %arg4 : tensor<1xi16>, tensor<1xi16>) outs(%arg5 : tensor<1xi16>) -> tensor<1xi16>
  return %3, %c0_i32 : tensor<1xi16>, i32
}
