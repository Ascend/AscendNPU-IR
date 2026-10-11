// The published build may fail later when A5 template bitcode is unavailable;
// this test only checks the pass order emitted before that stage.
// RUN: (bishengir-compile %s \
// RUN:   --target=Ascend950PR_9579 \
// RUN:   --enable-lir-compile=false \
// RUN:   --mlir-disable-threading \
// RUN:   --mlir-print-ir-before=loop-invariant-code-motion,hivm-cross-core-gss \
// RUN:   -o %t.ll 2>&1 || true) | FileCheck %s

// CHECK-NOT: IR Dump Before CrossCoreGSS (hivm-cross-core-gss)
// CHECK: IR Dump Before LoopInvariantCodeMotion (loop-invariant-code-motion)
// CHECK: IR Dump Before CrossCoreGSS (hivm-cross-core-gss)

module {
  func.func @test() {
    return
  }
}
