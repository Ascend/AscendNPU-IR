// RUN: not --crash bishengir-opt %s --adapt-gpu-kernel 2>&1 | FileCheck %s
// REQUIRES: enable_triton_ir_compiless

// The typed architecture lookup returns kUnknown for an unrecognized device.
// SIMT target serialization must reject it instead of emitting "unknown".
// CHECK: LLVM ERROR: Unknown TargetDevice value
module attributes {hacc.target = #hacc.target<"not-a-target">} {
  llvm.func @unknown_target() attributes {nvvm.kernel = 1 : ui1} {
    llvm.return
  }
}
