// Test that --save-temps=. produces module.hivm.opt.mlir in the current directory.
//
// RUN: bishengir-compile --save-temps=. %s -o %t.o
// RUN: test -f module.hivm.opt.mlir
// RUN: FileCheck --input-file=module.hivm.opt.mlir %s

// CHECK: module
// CHECK: func.func
func.func @test_func() {
  return
}