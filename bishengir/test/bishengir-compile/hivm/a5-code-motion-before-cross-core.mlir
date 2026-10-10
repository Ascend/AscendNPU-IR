// The published build may fail later when A5 template bitcode is unavailable;
// this test only checks the pass order emitted before that stage.
// RUN: (bishengir-compile %s \
// RUN:   --target=Ascend950PR_9579 \
// RUN:   --enable-lir-compile=false \
// RUN:   --mlir-disable-threading \
// RUN:   --mlir-print-ir-before=loop-invariant-code-motion,hivm-cross-core-gss \
// RUN:   -o %t.ll 2>&1 || true) | FileCheck %s --check-prefix=PIPELINED
// RUN: (bishengir-compile %s \
// RUN:   --target=Ascend950PR_9579 \
// RUN:   --enable-lir-compile=false \
// RUN:   --enable-preload=false \
// RUN:   --set-cv-pipeline-mode=off \
// RUN:   --mlir-disable-threading \
// RUN:   --mlir-print-ir-before=loop-invariant-code-motion,hivm-cross-core-gss \
// RUN:   -o %t.ll 2>&1 || true) | FileCheck %s --check-prefix=SSBUFFER

// PIPELINED-NOT: IR Dump Before CrossCoreGSS (hivm-cross-core-gss)
// PIPELINED: IR Dump Before LoopInvariantCodeMotion (loop-invariant-code-motion)
// PIPELINED: IR Dump Before CrossCoreGSS (hivm-cross-core-gss)

// SSBUFFER-NOT: IR Dump Before LoopInvariantCodeMotion (loop-invariant-code-motion)
// SSBUFFER: IR Dump Before CrossCoreGSS (hivm-cross-core-gss)
// SSBUFFER: IR Dump Before LoopInvariantCodeMotion (loop-invariant-code-motion)

module {
  func.func @test() {
    return
  }
}
