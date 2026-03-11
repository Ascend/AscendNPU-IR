// RUN: bishengir-opt %s -get-triton-metadata=triton-metadata-output=-- | FileCheck %s

module attributes {
  "ttg.num-warps" = 101 : i32,
  ttg.shared = 102 : i32,
  ttg.global_scratch_memory_alignment = 104 : i32,
  ttg.global_scratch_memory_size = 105 : i32,
  ttg.profile_scratch_memory_alignment = 106 : i32,
  ttg.profile_scratch_memory_size = 107 : i32
} {}

// CHECK: {
// CHECK-SAME: "global_scratch_align":104
// CHECK-SAME: "global_scratch_size":105
// CHECK-SAME: "num_warps":101
// CHECK-SAME: "profile_scratch_align":106
// CHECK-SAME: "profile_scratch_size":107
// CHECK-SAME: "shared":102
// CHECK-SAME: }
// CHECK: module
