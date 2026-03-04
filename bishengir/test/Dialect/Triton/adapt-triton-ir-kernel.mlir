// RUN: bishengir-opt --adapt-triton-ir-kernel %s | FileCheck %s

module {
  // CHECK: gpu.block = #gpu.block<x>
  // CHECK: gpu.block = #gpu.block<y>
  // CHECK: gpu.block = #gpu.block<z>
  tt.func public @test_public(%in_ptr0: !tt.ptr<i64> {tt.divisibility = 16 : i32}) {
    tt.return
  }
  // CHECK-NOT: gpu.block
  tt.func private @test_private(%in_ptr0: !tt.ptr<i64> {tt.divisibility = 16 : i32}) {
    tt.return
  }
  // CHECK-NOT: gpu.block
  tt.func private @abort()
}