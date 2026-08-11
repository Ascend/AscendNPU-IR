// RUN: bishengir-opt -legalize-bool-for-simtvf %s -split-input-file -verify-diagnostics | FileCheck %s

// This case tests two parts:
// 1) bool tensor returned inside scope could be casted to i8 inner the scope, and the scope result
//    should be casted back to i1 outside the scope, to keep consistent with scope's original use
// 2) bool tensor used inside scope should be casted to i8 outside scope and casted
//    back to i1 inside scope to keep consitent with its original use in scope
// CHECK-LABEL: func.func @mask_test_kernel
// CHECK-NEXT:  %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:  %0 = bufferization.to_tensor %arg0 restrict writable : memref<128xi32>
// CHECK-NEXT:  %1 = bufferization.to_tensor %arg1 restrict writable : memref<128xi32>
// CHECK-NEXT:  %2 = tensor.empty() : tensor<128xi1>
// CHECK-NEXT:  %3 = hivm.hir.vcmp ins(%0, %c0_i32 : tensor<128xi32>, i32) outs(%2 : tensor<128xi1>) compare_mode = <ne> -> tensor<128xi1>
// CHECK-NEXT:  %4 = tensor.empty() : tensor<128xi8>
// CHECK-NEXT:  %5 = hivm.hir.vcast ins(%3 : tensor<128xi1>) outs(%4 : tensor<128xi8>) -> tensor<128xi8>
// CHECK-NEXT:  %6 = tensor.empty() : tensor<128xi1>
// CHECK-NEXT:  %7 = hivm.hir.vcmp ins(%1, %c0_i32 : tensor<128xi32>, i32) outs(%6 : tensor<128xi1>) compare_mode = <le> -> tensor<128xi1>
// CHECK-NEXT:  %8 = tensor.empty() : tensor<128xi8>
// CHECK-NEXT:  %9 = hivm.hir.vcast ins(%7 : tensor<128xi1>) outs(%8 : tensor<128xi8>) -> tensor<128xi8>
// CHECK-NEXT:  %10 = scope.scope : () -> tensor<128xi8>
// CHECK-NEXT:  %15 = tensor.empty() : tensor<128xi1>
// CHECK-NEXT:  %16 = tensor.empty() : tensor<128xi1>
// CHECK-NEXT:  %17 = hivm.hir.vcast ins(%5 : tensor<128xi8>) outs(%16 : tensor<128xi1>) -> tensor<128xi1>
// CHECK-NEXT:  %18 = tensor.empty() : tensor<128xi1>
// CHECK-NEXT:  %19 = hivm.hir.vcast ins(%9 : tensor<128xi8>) outs(%18 : tensor<128xi1>) -> tensor<128xi1>
// CHECK-NEXT:  %20 = hivm.hir.vand ins(%17, %19 : tensor<128xi1>, tensor<128xi1>) outs(%15 : tensor<128xi1>) -> tensor<128xi1>
// CHECK-NEXT:  %21 = arith.extui %20 : tensor<128xi1> to tensor<128xi8>
// CHECK-NEXT:  scope.return %21 : tensor<128xi8>
// CHECK:  %11 = tensor.empty() : tensor<128xi1>
// CHECK-NEXT:  %12 = hivm.hir.vcast ins(%10 : tensor<128xi8>) outs(%11 : tensor<128xi1>) -> tensor<128xi1>
// CHECK-NEXT:  %13 = tensor.empty() : tensor<128xi1>
// CHECK-NEXT:  %14 = hivm.hir.vand ins(%7, %12 : tensor<128xi1>, tensor<128xi1>) outs(%13 : tensor<128xi1>) -> tensor<128xi1>
module {
  func.func @mask_test_kernel(%arg0: memref<128xi32>, %arg1: memref<128xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<128xi32>
    %1 = bufferization.to_tensor %arg1 restrict writable : memref<128xi32>
    %2 = tensor.empty() : tensor<128xi1>
    %3 = hivm.hir.vcmp ins(%0, %c0_i32 : tensor<128xi32>, i32) outs(%2 : tensor<128xi1>) compare_mode = <ne> -> tensor<128xi1>
    %4 = tensor.empty() : tensor<128xi1>
    %5 = hivm.hir.vcmp ins(%1, %c0_i32 : tensor<128xi32>, i32) outs(%4 : tensor<128xi1>) compare_mode = <le> -> tensor<128xi1>
    %6 = scope.scope : () -> tensor<128xi1> {
      %7 = tensor.empty() : tensor<128xi1>
      %8 = hivm.hir.vand ins(%3, %5 : tensor<128xi1>, tensor<128xi1>) outs(%7 : tensor<128xi1>) -> tensor<128xi1>
      scope.return %8 : tensor<128xi1>
    } {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, no_inline, noinline, outline, vector_type = "simt"}
    %9 = tensor.empty() : tensor<128xi1>
    %10 = hivm.hir.vand ins(%5, %6 : tensor<128xi1>, tensor<128xi1>) outs(%9 : tensor<128xi1>) -> tensor<128xi1>
    return
  }
}
