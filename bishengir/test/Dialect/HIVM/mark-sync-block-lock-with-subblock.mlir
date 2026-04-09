// RUN: bishengir-opt %s --hivm-mark-sync-block-lock-with-subblock -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @marks_top_level_lock_unlock
// CHECK: hivm.hir.sync_block_lock {hivm.sync_block_lock_with_subblock} lock_var(
// CHECK: hivm.hir.sync_block_unlock {hivm.sync_block_lock_with_subblock} lock_var(
// CHECK: return
module attributes {hacc.hivmc_version = #hacc.hivmc_version<"0.2.0">} {
  func.func @marks_top_level_lock_unlock() attributes {mix_mode = "mix", hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %lock = hivm.hir.create_sync_block_lock : memref<1xi64>
    hivm.hir.sync_block_lock lock_var(%lock : memref<1xi64>)
    hivm.hir.sync_block_unlock lock_var(%lock : memref<1xi64>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @skips_inside_limit_sub_block_id0_if
// CHECK: scf.if
// CHECK: hivm.hir.sync_block_lock lock_var(
// CHECK: hivm.hir.sync_block_unlock lock_var(
// CHECK: return
module attributes {hacc.hivmc_version = #hacc.hivmc_version<"0.2.0">} {
  func.func @skips_inside_limit_sub_block_id0_if() attributes {hivm.part_of_mix, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %lock = hivm.hir.create_sync_block_lock : memref<1xi64>
    %true = arith.constant true
    scf.if %true {
      hivm.hir.sync_block_lock lock_var(%lock : memref<1xi64>)
      hivm.hir.sync_block_unlock lock_var(%lock : memref<1xi64>)
    } {limit_sub_block_id0}
    return
  }
}

// -----

// CHECK-LABEL: func.func @skips_non_mix_module
// CHECK: hivm.hir.sync_block_lock lock_var(
// CHECK: return
module attributes {hacc.hivmc_version = #hacc.hivmc_version<"0.2.0">} {
  func.func @skips_non_mix_module() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %lock = hivm.hir.create_sync_block_lock : memref<1xi64>
    hivm.hir.sync_block_lock lock_var(%lock : memref<1xi64>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @skips_hivmc_below_0_2_0
// CHECK: hivm.hir.sync_block_lock lock_var(
// CHECK: return
module attributes {hacc.hivmc_version = #hacc.hivmc_version<"0.1.0">} {
  func.func @skips_hivmc_below_0_2_0() attributes {mix_mode = "mix", hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %lock = hivm.hir.create_sync_block_lock : memref<1xi64>
    hivm.hir.sync_block_lock lock_var(%lock : memref<1xi64>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @skips_ops_already_tagged
// CHECK: hivm.hir.sync_block_lock {hivm.sync_block_lock_with_subblock} lock_var(
// CHECK: return
module attributes {hacc.hivmc_version = #hacc.hivmc_version<"0.2.0">} {
  func.func @skips_ops_already_tagged() attributes {mix_mode = "mix", hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %lock = hivm.hir.create_sync_block_lock : memref<1xi64>
    hivm.hir.sync_block_lock {hivm.sync_block_lock_with_subblock} lock_var(%lock : memref<1xi64>)
    return
  }
}
