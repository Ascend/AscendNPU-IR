// REQUIRES: hivmc
// Test that --mlir-pass-pipeline-crash-reproducer flag is recognized and doesn't crash.
//
// RUN: bishengir-compile %s --mlir-pass-pipeline-crash-reproducer=%t 2>&1
// Success if we reach here (no crash from reproducer machinery)

func.func @test_func() {
  return
}
