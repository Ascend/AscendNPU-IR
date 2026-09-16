// Test that --save-temps=. produces .ll file in the current directory.
//
// UNSUPPORTED: bishengir_published
// RUN: hivmc-a5 -enable-lir-compile=false --save-temps=. --target=Ascend910B2 %s -o %t.o
// RUN: test -f save-temps.mlir.tmp.ll
// RUN: FileCheck --input-file=save-temps.mlir.tmp.ll %s

// CHECK: define {{.*}} @test_func()
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @test_func() {
    return
  }
}
