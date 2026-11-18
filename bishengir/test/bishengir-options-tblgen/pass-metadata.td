// RUN: bishengir-options-tblgen -gen-bishengir-pass-metadata %s \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -I %p/../../../third-party/llvm-project/mlir/include \
// RUN:   -I %p/../../../third-party/llvm-project/llvm/include \
// RUN:   | FileCheck %s

include "bishengir/Dialect/HIVM/Transforms/HIVMPass.td"

def MixOnly : HIVMPass<"example-mix", "", [MixKernel], true>;

def AivOnly : HIVMPass<"example-aiv", "", [PureAivKernel]>;

def AicOnly : HIVMPass<"example-aic", "", [PureAicKernel]>;

def MixAic : HIVMPass<"example-mix-aic", "", [MixKernel, PureAicKernel]>;

// Default kernel set is mix, pure AIV, and pure AIC.
def AllKinds : HIVMPass<"example-all">;

// CHECK: #ifdef GEN_MIX_KERNEL_PASS_ARGS
// CHECK-NEXT: "example-all",
// CHECK-NEXT: "example-mix",
// CHECK-NEXT: "example-mix-aic",
// CHECK-NEXT: #endif

// CHECK: #ifdef GEN_PURE_AIV_KERNEL_PASS_ARGS
// CHECK-NEXT: "example-aiv",
// CHECK-NEXT: "example-all",
// CHECK-NEXT: #endif

// CHECK: #ifdef GEN_PURE_AIC_KERNEL_PASS_ARGS
// CHECK-NEXT: "example-aic",
// CHECK-NEXT: "example-all",
// CHECK-NEXT: "example-mix-aic",
// CHECK-NEXT: #endif

// CHECK: #ifdef GEN_CLASSIFY_KERNEL_PASS_ARGS
// CHECK-NEXT: "example-mix",
// CHECK-NEXT: #endif
