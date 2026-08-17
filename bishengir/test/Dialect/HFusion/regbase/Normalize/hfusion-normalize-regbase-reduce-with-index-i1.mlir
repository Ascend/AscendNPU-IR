// RUN: bishengir-opt --hfusion-normalize-ops="use-regbase=true" %s -split-input-file | FileCheck %s

// i1 (bool) argmax must be promoted to i32 so the reduce keeps a vectorized
// (vreduce) lowering instead of being scalarized by GenericUnroller into
// per-bit memref.load.

// CHECK-LABEL: func.func @argmax_i1
// CHECK-SAME: %[[arg0:.*]]: tensor<256xi1>
// CHECK: %[[cast:.*]] = hfusion.cast {{.*}} ins(%[[arg0]] : tensor<256xi1>) outs({{.*}} : tensor<256xi32>)
// CHECK: hfusion.reduce_with_index {{.*}}unsigned_src = true{{.*}} ins(%[[cast]], %{{.*}} : tensor<256xi32>, tensor<256xi32>)
// CHECK: return %{{.*}}#1 : tensor<i32>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2097152 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @argmax_i1(%arg0: tensor<256xi1>, %arg1: tensor<256xi32>) -> tensor<i32> {
    %0 = tensor.empty() : tensor<i1>
    %1 = tensor.empty() : tensor<i32>
    %reduced:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = true} <max>
                  ins(%arg0, %arg1 : tensor<256xi1>, tensor<256xi32>)
                  outs(%0, %1 : tensor<i1>, tensor<i32>)
                  dimensions = [0] -> tensor<i1>, tensor<i32>
    return %reduced#1 : tensor<i32>
  }
}

// -----

// i32 argmax is already i32 — no promote cast, reduce stays on i32.

// CHECK-LABEL: func.func @argmax_i32
// CHECK-NOT: hfusion.cast
// CHECK: hfusion.reduce_with_index {{.*}} ins(%{{.*}}, %{{.*}} : tensor<256xi32>, tensor<256xi32>)
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2097152 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @argmax_i32(%arg0: tensor<256xi32>, %arg1: tensor<256xi32>) -> tensor<i32> {
    %0 = tensor.empty() : tensor<i32>
    %1 = tensor.empty() : tensor<i32>
    %reduced:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <max>
                  ins(%arg0, %arg1 : tensor<256xi32>, tensor<256xi32>)
                  outs(%0, %1 : tensor<i32>, tensor<i32>)
                  dimensions = [0] -> tensor<i32>, tensor<i32>
    return %reduced#1 : tensor<i32>
  }
}

// -----

// Only i1 is promoted. i64 (and i8/i16) have a legal byte-granular scalar
// load, so the value pattern must leave them untouched — no truncating cast.

// CHECK-LABEL: func.func @argmax_i64
// CHECK-NOT: hfusion.cast
// CHECK: hfusion.reduce_with_index {{.*}} ins(%{{.*}}, %{{.*}} : tensor<256xi64>, tensor<256xi32>)
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2097152 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @argmax_i64(%arg0: tensor<256xi64>, %arg1: tensor<256xi32>) -> tensor<i32> {
    %0 = tensor.empty() : tensor<i64>
    %1 = tensor.empty() : tensor<i32>
    %reduced:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <max>
                  ins(%arg0, %arg1 : tensor<256xi64>, tensor<256xi32>)
                  outs(%0, %1 : tensor<i64>, tensor<i32>)
                  dimensions = [0] -> tensor<i64>, tensor<i32>
    return %reduced#1 : tensor<i32>
  }
}
