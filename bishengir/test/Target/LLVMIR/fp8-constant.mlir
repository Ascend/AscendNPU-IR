// RUN: bishengir-translate --mlir-to-llvmir -split-input-file %s | FileCheck %s

// Test that a zero constant of f8E4M3FN type translates to LLVM IR without
// crashing.  Previously, Constant::getNullValue() hit an unreachable because
// Float8E4M3TyID was missing from its type switch.
// CHECK-LABEL: define void @test_f8e4m3fn_zero()
// CHECK: ret void
llvm.func @test_f8e4m3fn_zero() {
  %0 = llvm.mlir.constant(0.000000e+00 : f8E4M3FN) : f8E4M3FN
  llvm.return
}

// -----

// Test that a bitcast of a zero i8 constant to f8E4M3FN translates without
// crashing.  This mirrors the pattern produced by ConvertTritonAscendGPUToLLVM
// and triggers constant folding inside the LLVM IR translator.
// CHECK-LABEL: define void @test_f8e4m3fn_bitcast_zero()
// CHECK: ret void
llvm.func @test_f8e4m3fn_bitcast_zero() {
  %0 = llvm.mlir.constant(0 : i8) : i8
  %1 = llvm.bitcast %0 : i8 to f8E4M3FN
  llvm.return
}
