// RUN: bishengir-opt -convert-hivm-to-tritongpu="allow-return-value=true" %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @vshr_unsigned_scalar_shift
func.func @vshr_unsigned_scalar_shift(%arg0: tensor<4xi32>, %shift: i32) -> tensor<4xi32> {
  %empty = tensor.empty() : tensor<4xi32>
  // CHECK: %[[SHIFT:.*]] = tt.splat %arg1 : i32 -> tensor<4xi32>
  // CHECK: %[[RET:.*]] = arith.shrui %arg0, %[[SHIFT]] : tensor<4xi32>
  %0 = hivm.hir.vshr ins(%arg0, %shift : tensor<4xi32>, i32)
      outs(%empty : tensor<4xi32>) round : false is_signed : false -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// CHECK-LABEL: func.func @vshr_signed_scalar_shift
func.func @vshr_signed_scalar_shift(%arg0: tensor<4xi32>, %shift: i32) -> tensor<4xi32> {
  %empty = tensor.empty() : tensor<4xi32>
  // CHECK: %[[SHIFT:.*]] = tt.splat %arg1 : i32 -> tensor<4xi32>
  // CHECK: %[[RET:.*]] = arith.shrsi %arg0, %[[SHIFT]] : tensor<4xi32>
  %0 = hivm.hir.vshr ins(%arg0, %shift : tensor<4xi32>, i32)
      outs(%empty : tensor<4xi32>) round : false is_signed : true -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// CHECK-LABEL: func.func @vshr_rounded_scalar_shift
func.func @vshr_rounded_scalar_shift(%arg0: tensor<4xi32>, %shift: i32) -> tensor<4xi32> {
  %empty = tensor.empty() : tensor<4xi32>
  // CHECK: %[[SHIFT:.*]] = tt.splat %arg1 : i32 -> tensor<4xi32>
  // CHECK: %[[RET:.*]] = arith.shrsi %arg0, %[[SHIFT]] : tensor<4xi32>
  %0 = hivm.hir.vshr ins(%arg0, %shift : tensor<4xi32>, i32)
      outs(%empty : tensor<4xi32>) round : true is_signed : true -> tensor<4xi32>
  return %0 : tensor<4xi32>
}
