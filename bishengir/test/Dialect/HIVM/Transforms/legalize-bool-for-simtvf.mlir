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
    } {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, no_inline, noinline, outline, vector_mode = "simt"}
    %9 = tensor.empty() : tensor<128xi1>
    %10 = hivm.hir.vand ins(%5, %6 : tensor<128xi1>, tensor<128xi1>) outs(%9 : tensor<128xi1>) -> tensor<128xi1>
    return
  }
}

// This case tests: non-bool tensor used inside scope is from op with multi results
// CHECK-LABEL: scope_non_bool_use_from_multi_results
module {
  func.func @scope_non_bool_use_from_multi_results(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: tensor<32xi32>, %arg4: tensor<32xi32>, %arg5: i32, %arg6: tensor<1x32xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c80_i32 = arith.constant 80 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tensor.empty() : tensor<32xi32>
    %1 = hivm.hir.varange offset[%c0] strides[%c1] outs(%0 : tensor<32xi32>) -> tensor<32xi32>
    %2 = arith.cmpi eq, %arg5, %c2_i32 : i32
    %3:5 = scf.if %2 -> (i32, i32, i32, tensor<32xi32>, tensor<32xi32>) {
      %5 = arith.addi %arg2, %c80_i32 : i32
      %6 = arith.divsi %5, %c2_i32 : i32
      %7 = arith.muli %6, %c4_i32 : i32
      %8 = tensor.empty() : tensor<32xi32>
      %9 = hivm.hir.vadd ins(%1, %6 : tensor<32xi32>, i32) outs(%8 : tensor<32xi32>) -> tensor<32xi32>
      %10 = tensor.empty() : tensor<32xi32>
      %11 = hivm.hir.vadd ins(%1, %7 : tensor<32xi32>, i32) outs(%10 : tensor<32xi32>) -> tensor<32xi32>
      scf.yield %5, %6, %7, %9, %11 : i32, i32, i32, tensor<32xi32>, tensor<32xi32>
    } else {
      scf.yield %arg0, %arg1, %arg2, %arg3, %arg4 : i32, i32, i32, tensor<32xi32>, tensor<32xi32>
    }
    %4:2 = scope.scope : () -> (tensor<32x32xi32>, tensor<32x32xi32>) {
      // CHECK: %expanded = tensor.expand_shape %3#3 {{\[\[0, 1\]\]}} output_shape [32, 1] : tensor<32xi32> into tensor<32x1xi32>
      %expanded = tensor.expand_shape %3#3 [[0, 1]] output_shape [32, 1] : tensor<32xi32> into tensor<32x1xi32>
      %5 = tensor.empty() : tensor<32x1xi32>
      %6 = hivm.hir.vmul ins(%expanded, %arg0 : tensor<32x1xi32>, i32) outs(%5 : tensor<32x1xi32>) -> tensor<32x1xi32>
      %7 = tensor.empty() : tensor<32x32xi32>
      %8 = hivm.hir.vadd ins(%6, %arg6 : tensor<32x1xi32>, tensor<1x32xi32>) outs(%7 : tensor<32x32xi32>) broadcast = [0, 1] -> tensor<32x32xi32>
      // CHECK: %expanded_0 = tensor.expand_shape %3#4 {{\[\[0, 1\]\]}} output_shape [1, 32] : tensor<32xi32> into tensor<1x32xi32>
      %expanded_0 = tensor.expand_shape %3#4 [[0, 1]] output_shape [1, 32] : tensor<32xi32> into tensor<1x32xi32>
      %9 = tensor.empty() : tensor<1x32xi32>
      %10 = hivm.hir.vmul ins(%expanded_0, %arg1 : tensor<1x32xi32>, i32) outs(%9 : tensor<1x32xi32>) -> tensor<1x32xi32>
      %11 = tensor.empty() : tensor<32x32xi32>
      %12 = hivm.hir.vadd ins(%10, %arg6 : tensor<1x32xi32>, tensor<1x32xi32>) outs(%11 : tensor<32x32xi32>) broadcast = [0, 1] -> tensor<32x32xi32>
      scope.return %8, %12 : tensor<32x32xi32>, tensor<32x32xi32>
    }  {hivm.vf_mode = #hivm.vf_mode<SIMT>}
    return
  }
}

// This case tests: bool tensor used inside scope is from op with multi results
// CHECK-LABEL: scope_bool_use_from_multi_results
module {
  func.func @scope_bool_use_from_multi_results(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: tensor<32xi1>, %arg4: tensor<32xi1>, %arg5: i32, %arg6: tensor<32x1xi1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c80_i32 = arith.constant 80 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tensor.empty() : tensor<32xi32>
    %1 = hivm.hir.varange offset[%c0] strides[%c1] outs(%0 : tensor<32xi32>) -> tensor<32xi32>
    %2 = arith.cmpi eq, %arg5, %c2_i32 : i32
    %3:5 = scf.if %2 -> (i32, i32, i32, tensor<32xi1>, tensor<32xi1>) {
      %5 = arith.addi %arg2, %c80_i32 : i32
      %6 = arith.divsi %5, %c2_i32 : i32
      %7 = arith.muli %6, %c4_i32 : i32
      %8 = tensor.empty() : tensor<32xi1>
      %9 = hivm.hir.vcmp ins(%1, %6 : tensor<32xi32>, i32) outs(%8 : tensor<32xi1>) compare_mode = <lt> -> tensor<32xi1>
      %10 = tensor.empty() : tensor<32xi1>
      %11 = hivm.hir.vcmp ins(%1, %7 : tensor<32xi32>, i32) outs(%10 : tensor<32xi1>) compare_mode = <ne> -> tensor<32xi1>
      scf.yield %5, %6, %7, %9, %11 : i32, i32, i32, tensor<32xi1>, tensor<32xi1>
    } else {
      scf.yield %arg0, %arg1, %arg2, %arg3, %arg4 : i32, i32, i32, tensor<32xi1>, tensor<32xi1>
    }
    // CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<32xi8>
    // CHECK: %[[I1_I8_CST0:.*]] = hivm.hir.vcast ins(%{{[0-9]+}}#4 : tensor<32xi1>) outs(%[[EMPTY0:.*]] : tensor<32xi8>) -> tensor<32xi8>
    // CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<32xi8>
    // CHECK: %[[I1_I8_CST1:.*]] = hivm.hir.vcast ins(%{{[0-9]+}}#3 : tensor<32xi1>) outs(%[[EMPTY1:.*]] : tensor<32xi8>) -> tensor<32xi8>
    // CHECK: %[[NEW_SCOPE:.*]]:2 = scope.scope : () -> (tensor<32x1xi8>, tensor<32x1xi8>) {
    %4:2 = scope.scope : () -> (tensor<32x1xi1>, tensor<32x1xi1>) {
      // CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<32xi1>
      // CHECK: %[[I8_I1_BACK0:.*]] = hivm.hir.vcast ins(%[[I1_I8_CST1:.*]] : tensor<32xi8>) outs(%[[EMPTY2:.*]] : tensor<32xi1>) -> tensor<32xi1>
      // CHECK: %expanded = tensor.expand_shape  %[[I8_I1_BACK0:.*]] {{\[\[0, 1\]\]}} output_shape [32, 1] : tensor<32xi1> into tensor<32x1xi1>
      %expanded = tensor.expand_shape %3#3 [[0, 1]] output_shape [32, 1] : tensor<32xi1> into tensor<32x1xi1>
      // CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<32xi1>
      // CHECK: %[[I8_I1_BACK1:.*]] = hivm.hir.vcast ins(%[[I1_I8_CST0:.*]] : tensor<32xi8>) outs(%[[EMPTY3:.*]] : tensor<32xi1>) -> tensor<32xi1>
      // CHECK: %expanded_0 = tensor.expand_shape  %[[I8_I1_BACK1:.*]] {{\[\[0, 1\]\]}} output_shape [32, 1] : tensor<32xi1> into tensor<32x1xi1>      
	    %expanded_0 = tensor.expand_shape %3#4 [[0, 1]] output_shape [32, 1] : tensor<32xi1> into tensor<32x1xi1>
      %5 = tensor.empty() : tensor<32x1xi1>
      %6 = hivm.hir.vcmp ins(%expanded, %expanded_0 : tensor<32x1xi1>, tensor<32x1xi1>) outs(%5 : tensor<32x1xi1>) compare_mode = <ne> -> tensor<32x1xi1>
      %7 = tensor.empty() : tensor<32x1xi1>
      %8 = hivm.hir.vcmp ins(%6, %arg6 : tensor<32x1xi1>, tensor<32x1xi1>) outs(%7 : tensor<32x1xi1>) compare_mode = <lt> -> tensor<32x1xi1>
      // CHECK: %[[I1_I8_RET0:.*]] = arith.extui %{{[0-9]+}} : tensor<32x1xi1> to tensor<32x1xi8>
      // CHECK: %[[I1_I8_RET1:.*]] = arith.extui %{{[0-9]+}} : tensor<32x1xi1> to tensor<32x1xi8>
      // CHECK: scope.return %[[I1_I8_RET0:.*]], %[[I1_I8_RET1:.*]] : tensor<32x1xi8>, tensor<32x1xi8>
      scope.return %6, %8 : tensor<32x1xi1>, tensor<32x1xi1>
    }  {hivm.vf_mode = #hivm.vf_mode<SIMT>}
    // CHECK: %[[EMPTY4:.*]] = tensor.empty() : tensor<32x1xi1>
    // CHECK: %[[I8_I1_OUT0:.*]] = hivm.hir.vcast ins(%[[NEW_SCOPE:.*]]#0 : tensor<32x1xi8>) outs(%[[EMPTY4:.*]] : tensor<32x1xi1>) -> tensor<32x1xi1>
    // CHECK: %[[EMPTY5:.*]] = tensor.empty() : tensor<32x1xi1>
    // CHECK: %[[I8_I1_OUT1:.*]] = hivm.hir.vcast ins(%[[NEW_SCOPE:.*]]#1 : tensor<32x1xi8>) outs(%[[EMPTY5:.*]] : tensor<32x1xi1>) -> tensor<32x1xi1>
    // CHECK: %{{[0-9]+}} = hivm.hir.vand ins(%[[I8_I1_OUT0:.*]], %arg6 : tensor<32x1xi1>, tensor<32x1xi1>) outs(%{{[0-9]+}} : tensor<32x1xi1>) -> tensor<32x1xi1>
    // CHECK: %{{[0-9]+}} = hivm.hir.vxor ins(%[[I8_I1_OUT1:.*]], %arg6 : tensor<32x1xi1>, tensor<32x1xi1>) outs(%{{[0-9]+}} : tensor<32x1xi1>) -> tensor<32x1xi1>    
    %9 = tensor.empty() : tensor<32x1xi1>
    %10 = hivm.hir.vand ins(%4#0, %arg6 : tensor<32x1xi1>, tensor<32x1xi1>) outs(%9 : tensor<32x1xi1>) -> tensor<32x1xi1>
    %11 = tensor.empty() : tensor<32x1xi1>
    %12 = hivm.hir.vxor ins(%4#1, %arg6 : tensor<32x1xi1>, tensor<32x1xi1>) outs(%11 : tensor<32x1xi1>) -> tensor<32x1xi1>
    return
  }
}
