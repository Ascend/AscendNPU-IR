// RUN: not bishengir-options-tblgen -gen-bishengir-pass-metadata %s \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -I %p/../../../third-party/llvm-project/mlir/include \
// RUN:   -I %p/../../../third-party/llvm-project/llvm/include \
// RUN:   2>&1 | FileCheck %s

include "bishengir/Dialect/HIVM/Transforms/HIVMPass.td"

// CHECK: HIVMPass kernelKinds must not be empty
def EmptyKinds : HIVMPass<"example-empty-kinds", AnyOp, []>;
