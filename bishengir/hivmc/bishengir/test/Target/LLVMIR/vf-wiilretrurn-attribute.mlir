// RUN: bishengir-translate --mlir-to-llvmir -split-input-file %s | FileCheck %s

// Test that hivm.vector_function attribute is translated to willreturn in LLVM IR

// CHECK: ; Function Attrs: noduplicate noinline willreturn
// CHECK-LABEL: define internal ptc_simdvf void @test_vf_willreturn()
llvm.func @test_vf_willreturn() attributes {hivm.vector_function} {
  llvm.return
}

// CHECK: attributes #{{.*}} = { noduplicate noinline willreturn }
