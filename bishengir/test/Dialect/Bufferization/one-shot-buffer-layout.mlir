// RUN: bishengir-opt %s -one-shot-bufferize="allow-return-allocs-from-loops bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -split-input-file | FileCheck %s --check-prefix=CHECK
// RUN: bishengir-opt %s -one-shot-bufferize="allow-return-allocs-from-loops bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map unknown-type-conversion=identity-layout-map" -split-input-file | FileCheck %s --check-prefix=CHECK-IDENTITY

// CHECK: scf.yield {{.*}} : memref<256xf16, strided<[?], offset: ?>>
// CHECK-IDENTITY: scf.yield {{.*}} : memref<256xf16>
func.func @test_clone_yield_operands_alias_by_for_operands(%arg0: i32, %arg1 : tensor<256xf16>,
                                                              %arg2 : tensor<256xf16>) -> (tensor<256xf16>) {
  %cst_0 = arith.constant 0.000000e+00 : f16
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<256xf16>
  %1 = hivm.hir.vbrc ins(%cst_0 : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  %2 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %1) -> (tensor<256xf16>) {
    %3 = arith.cmpi eq, %arg3, %c1 : index
    %4 = scf.if %3 -> tensor<256xf16> {
      %5 = tensor.empty() : tensor<256xf16>
      %6 = hivm.hir.vadd ins(%arg4, %arg2 : tensor<256xf16>, tensor<256xf16>) outs(%5 : tensor<256xf16>) -> tensor<256xf16>
      scf.yield %6 : tensor<256xf16>
    } else {
      %5 = tensor.empty() : tensor<256xf16>
      %6 = hivm.hir.vadd ins(%arg4, %arg2 : tensor<256xf16>, tensor<256xf16>) outs(%5 : tensor<256xf16>) -> tensor<256xf16>
      hivm.hir.debug {debugtype = "print", hex = false, prefix = " %arg4 : ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %arg4 : tensor<256xf16>
      scf.yield %6 : tensor<256xf16>
    }
    hivm.hir.debug {debugtype = "print", hex = false, prefix = " %arg4 : ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %arg4 : tensor<256xf16>
    scf.yield %4 : tensor<256xf16>
  }
  return %2 : tensor<256xf16>
}