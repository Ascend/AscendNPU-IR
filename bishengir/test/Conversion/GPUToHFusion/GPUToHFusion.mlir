// RUN: bishengir-opt -convert-gpu-to-hfusion %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_barrier
func.func @test_barrier() {
  gpu.barrier
  return
}
// CHECK: hfusion.barrier