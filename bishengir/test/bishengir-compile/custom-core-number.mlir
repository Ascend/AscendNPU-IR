// REQUIRES: hivmc
// UNSUPPORTED: bishengir_published
//
// Regression test for the A5 / regbase compile driver: the top-level
// --custom-aic-number / --custom-aiv-number options must reach the
// hacc-append-device-spec pass inside the regbase pipeline. Previously they
// were forwarded only by the membase pipeline builder (see commit fa4738dd),
// so the regbase pipeline always kept the target's default core counts
// (e.g. VECTOR_CORE_COUNT 56 for Ascend950PR_9579) and the auto-blockify loop
// step was wrong.
//
// RUN: bishengir-compile --target=Ascend950PR_9579 \
// RUN:   --custom-aic-number=2 --custom-aiv-number=4 \
// RUN:   --mlir-print-ir-after=hacc-append-device-spec %s -o %t.o 2>&1 | FileCheck %s

// CHECK: "AI_CORE_COUNT", 2 : i32
// CHECK-SAME: "CUBE_CORE_COUNT", 2 : i32
// CHECK-SAME: "VECTOR_CORE_COUNT", 4 : i32

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test() {
    return
  }
}
