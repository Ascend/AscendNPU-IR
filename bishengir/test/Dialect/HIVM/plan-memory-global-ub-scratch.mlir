// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend960DT_968B7 -hivm-plan-memory=mem-plan-mode=global-work-space-plan | FileCheck %s
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend960DT_968B7 -hivm-plan-memory=mem-plan-mode=local-mem-plan -o /dev/null

// Global workspace planning must leave UB-only scratch to the local planner.
module attributes {hacc.target = #hacc.target<"Ascend960DT_968B7">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  // CHECK-LABEL: func.func @ub_scratch
  // CHECK: %[[TEMP:.*]] = memref.alloc() : memref<2xf32, #hivm.address_space<ub>>
  // CHECK: hivm.hir.vcumsum
  // CHECK-SAME: temp_buffer(%[[TEMP]] : memref<2xf32, #hivm.address_space<ub>>)
  func.func @ub_scratch(%input: memref<16xf32, #hivm.address_space<ub>>, %output: memref<16xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %scratch = memref.alloc() : memref<2xf32, #hivm.address_space<ub>>
    hivm.hir.vcumsum {hivm.vf_mode = #hivm.vf_mode<SIMT>} ins(%input : memref<16xf32, #hivm.address_space<ub>>) outs(%output : memref<16xf32, #hivm.address_space<ub>>) temp_buffer(%scratch : memref<2xf32, #hivm.address_space<ub>>) cum_dims = [0] reverse = false
    return
  }
}
