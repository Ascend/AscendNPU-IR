// RUN: bishengir-opt %s | FileCheck %s
// RUN: bishengir-opt %s --cse | FileCheck %s

// 4095 is the last 12-bit region id.
// CHECK-LABEL: func.func @max_region_id
// CHECK: hivm.hir.proton_scope_boundary {region_id = 4095 : i32}
func.func @max_region_id() {
  hivm.hir.proton_scope_boundary {region_id = 4095 : i32}
  return
}

// Two identical boundaries have no results. Memory effects keep both of them.
// CHECK-LABEL: func.func @duplicate_boundaries_survive_cse
// CHECK-COUNT-2: hivm.hir.proton_scope_boundary {region_id = 0 : i32}
func.func @duplicate_boundaries_survive_cse() {
  hivm.hir.proton_scope_boundary {region_id = 0 : i32}
  hivm.hir.proton_scope_boundary {region_id = 0 : i32}
  return
}
