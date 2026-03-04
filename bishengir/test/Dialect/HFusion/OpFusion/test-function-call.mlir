// RUN: bishengir-opt -test-function-call -split-input-file -verify-diagnostics %s | FileCheck %s
func.func @test(%A: tensor<?x4096xf16>, %B: tensor<14336x4096xf16>, %C: tensor<?x14336xf16>,  %D: memref<12xi64>) -> () {
// CHECK: call @extern_callee({{.*}},{{.*}}, {{.*}}, {{.*}}) : (tensor<?x4096xf16>, tensor<14336x4096xf16>, tensor<?x14336xf16>, memref<12xi64>) -> ()
  return
}

func.func @callee(%A: tensor<?x4096xf16>, %B: tensor<14336x4096xf16>, %C: tensor<?x14336xf16>,  %D: memref<12xi64>) -> () {
  return 
}
