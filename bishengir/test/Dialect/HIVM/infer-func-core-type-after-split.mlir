// RUN: bishengir-opt %s -pass-pipeline='builtin.module(hivm-infer-func-core-type,hivm-split-mix-kernel,hivm-infer-func-core-type)' | FileCheck %s

// Core-type inference after splitting must preserve both MIX members.
// CHECK: hivm.module_core_type = #hivm.module_core_type<MIX>
// CHECK-LABEL: func.func @mixed_1_1_mix_aic(
// CHECK-SAME: hivm.core_ratio = #hivm.core_ratio<1, 1>
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK-SAME: hivm.part_of_mix
// CHECK: hivm.hir.matmul
// CHECK-LABEL: func.func @mixed_1_1_mix_aiv(
// CHECK-SAME: hivm.core_ratio = #hivm.core_ratio<1, 1>
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-SAME: hivm.part_of_mix
// CHECK: hivm.hir.vadd
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @mixed_1_1(%a: memref<16x16xf16>, %b: memref<16x16xf16>,
                       %c: memref<16x16xf16>, %x: memref<16xf16>,
                       %y: memref<16xf16>, %z: memref<16xf16>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                  hivm.core_ratio = #hivm.core_ratio<1, 1>} {
    hivm.hir.matmul ins(%a, %b : memref<16x16xf16>, memref<16x16xf16>)
                    outs(%c : memref<16x16xf16>)
    hivm.hir.vadd ins(%x, %y : memref<16xf16>, memref<16xf16>)
                  outs(%z : memref<16xf16>)
    return
  }
}
