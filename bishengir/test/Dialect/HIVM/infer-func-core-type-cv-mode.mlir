// RUN: split-file %s %t
// RUN: sed 's/TARGET_ATTR/hacc.target = #hacc.target<"Ascend910B1">/g' %t/valid.mlir | bishengir-opt -hivm-infer-func-core-type -split-input-file | FileCheck %s --check-prefixes=COMMON,GENERIC
// RUN: sed 's/TARGET_ATTR/hacc.target = #hacc.target<"Ascend950PR_9579">/g' %t/valid.mlir | bishengir-opt -hivm-infer-func-core-type -split-input-file | FileCheck %s --check-prefixes=COMMON,GENERIC
// RUN: sed 's/TARGET_ATTR/hacc.target = #hacc.target<"Ascend310B1">/g' %t/valid.mlir | bishengir-opt -hivm-infer-func-core-type -split-input-file | FileCheck %s --check-prefixes=COMMON,COUPLED
// RUN: sed 's/TARGET_ATTR//g' %t/valid.mlir | bishengir-opt -hivm-infer-func-core-type -split-input-file | FileCheck %s --check-prefixes=COMMON,GENERIC
// RUN: sed 's/TARGET_ATTR/hacc.target = #hacc.target<"Unknown">/g' %t/valid.mlir | bishengir-opt -hivm-infer-func-core-type -split-input-file | FileCheck %s --check-prefixes=COMMON,GENERIC
// RUN: sed 's/TARGET_ATTR/hacc.target = #hacc.target<"Ascend910B1">/g' %t/invalid.mlir | bishengir-opt -hivm-infer-func-core-type -verify-diagnostics
// RUN: sed 's/TARGET_ATTR/hacc.target = #hacc.target<"Ascend950PR_9579">/g' %t/invalid.mlir | bishengir-opt -hivm-infer-func-core-type -verify-diagnostics
// RUN: sed 's/TARGET_ATTR//g' %t/invalid.mlir | bishengir-opt -hivm-infer-func-core-type -verify-diagnostics
// RUN: sed 's/TARGET_ATTR/hacc.target = #hacc.target<"Unknown">/g' %t/invalid.mlir | bishengir-opt -hivm-infer-func-core-type -verify-diagnostics

// Characterize the existing consumer before changing its architecture query.
// C220 and C310 both use generic C/V inference despite their different data
// paths. Missing and unknown targets also retain generic inference. M300 keeps
// the coupled-mode convention: module=AIV; a direct MmadL1 marks its parent
// function AIC, even if that function also contains vector operations. Other
// unannotated functions are left alone; this is not ordinary AIV inference.

//--- valid.mlir

// GENERIC: module attributes {{.*}}hivm.module_core_type = #hivm.module_core_type<AIC>
// COUPLED: module attributes {{.*}}hivm.module_core_type = #hivm.module_core_type<AIV>
module attributes {TARGET_ATTR} {
  // COMMON-LABEL: func.func @cube_only(
  // COMMON-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
  func.func @cube_only(%lhs: memref<16x16xf16>,
                       %rhs: memref<16x16xf16>,
                       %dst: memref<16x16xf32>) {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    hivm.hir.mmadL1 ins(%lhs, %rhs, %true, %c16, %c16, %c16 :
        memref<16x16xf16>, memref<16x16xf16>, i1, index, index, index)
        outs(%dst : memref<16x16xf32>)
    // COMMON: return
    return
  }
}

// -----

// GENERIC: module attributes {{.*}}hivm.module_core_type = #hivm.module_core_type<AIV>
// COUPLED: module attributes {{.*}}hivm.module_core_type = #hivm.module_core_type<AIV>
module attributes {TARGET_ATTR} {
  // GENERIC-LABEL: func.func @vector_only(
  // GENERIC-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
  // COUPLED-LABEL: func.func @vector_only(
  // COUPLED-NOT: hivm.func_core_type
  func.func @vector_only(%lhs: memref<16x16xf32>,
                         %rhs: memref<16x16xf32>,
                         %dst: memref<16x16xf32>) {
    hivm.hir.vadd ins(%lhs, %rhs : memref<16x16xf32>, memref<16x16xf32>)
        outs(%dst : memref<16x16xf32>)
    // COMMON: return
    return
  }
}

// -----

// GENERIC: module attributes {{.*}}hivm.module_core_type = #hivm.module_core_type<MIX>
// COUPLED: module attributes {{.*}}hivm.module_core_type = #hivm.module_core_type<AIV>
module attributes {TARGET_ATTR} {
  // COMMON-LABEL: func.func @mixed_body(
  // GENERIC-SAME: hivm.func_core_type = #hivm.func_core_type<MIX>
  // COUPLED-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
  func.func @mixed_body(%lhs: memref<16x16xf16>,
                        %rhs: memref<16x16xf16>,
                        %acc: memref<16x16xf32>,
                        %vector_src: memref<16x16xf32>,
                        %vector_dst: memref<16x16xf32>) {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    hivm.hir.mmadL1 ins(%lhs, %rhs, %true, %c16, %c16, %c16 :
        memref<16x16xf16>, memref<16x16xf16>, i1, index, index, index)
        outs(%acc : memref<16x16xf32>)
    hivm.hir.vadd ins(%vector_src, %vector_src :
        memref<16x16xf32>, memref<16x16xf32>)
        outs(%vector_dst : memref<16x16xf32>)
    // COMMON: return
    return
  }
}

// -----

// Explicit annotations on functions without MmadL1 are preserved on all targets.
// GENERIC: module attributes {{.*}}hivm.module_core_type = #hivm.module_core_type<MIX>
// COUPLED: module attributes {{.*}}hivm.module_core_type = #hivm.module_core_type<AIV>
module attributes {TARGET_ATTR} {
  // COMMON-LABEL: func.func @explicit_aic(
  // COMMON-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
  func.func @explicit_aic() attributes {
      hivm.func_core_type = #hivm.func_core_type<AIC>} {
    // COMMON: return
    return
  }

  // COMMON-LABEL: func.func @explicit_mix(
  // COMMON-SAME: hivm.func_core_type = #hivm.func_core_type<MIX>
  func.func @explicit_mix() attributes {
      hivm.func_core_type = #hivm.func_core_type<MIX>} {
    // COMMON: return
    return
  }
}

// -----

// The legacy coupled path overwrites an explicit MIX annotation when the
// function directly contains MmadL1. Generic inference respects the annotation.
// GENERIC: module attributes {{.*}}hivm.module_core_type = #hivm.module_core_type<MIX>
// COUPLED: module attributes {{.*}}hivm.module_core_type = #hivm.module_core_type<AIV>
module attributes {TARGET_ATTR} {
  // COMMON-LABEL: func.func @explicit_mix_with_mmad(
  // GENERIC-SAME: hivm.func_core_type = #hivm.func_core_type<MIX>
  // COUPLED-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
  func.func @explicit_mix_with_mmad(%lhs: memref<16x16xf16>,
                                    %rhs: memref<16x16xf16>,
                                    %dst: memref<16x16xf32>) attributes {
      hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    hivm.hir.mmadL1 ins(%lhs, %rhs, %true, %c16, %c16, %c16 :
        memref<16x16xf16>, memref<16x16xf16>, i1, index, index, index)
        outs(%dst : memref<16x16xf32>)
    // COMMON: return
    return
  }
}

//--- invalid.mlir

// Generic inference must still diagnose incompatible explicit annotations;
// selecting a target with an on-chip data path must not bypass the check.
module attributes {TARGET_ATTR} {
  // expected-error@+1 {{hivm.core_ratio is only valid on a MIX function, but hivm.func_core_type is AIV}}
  func.func @invalid_ratio() attributes {
      hivm.core_ratio = #hivm.core_ratio<0, 1>,
      hivm.func_core_type = #hivm.func_core_type<AIV>} {
    return
  }
}
