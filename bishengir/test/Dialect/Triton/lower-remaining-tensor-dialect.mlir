// RUN: bishengir-opt -split-input-file -verify-diagnostics -lower-remaining-tensor-dialect %s | FileCheck %s

// CHECK-LABEL: @tensorEmptyf16
// CHECK-NOT: tensor.empty
// CHECK: %[[CST_0:.*]] = arith.constant dense<0.0{{.*}}> : tensor<4xf16>
// CHECK: tt.store %{{.*}}, %[[CST_0]] : !tt.ptr<tensor<4xf16>>
tt.func @tensorEmptyf16(%dst: !tt.ptr<f16>) {
  %0 = arith.constant 4 : i64
  %1 = arith.constant 1 : i64
  %2 = arith.constant 0 : i32

  %dstPtr = tt.make_tensor_ptr %dst, [%0], [%1], [%2] {order = array<i32: 0>} : !tt.ptr<tensor<4xf16>>
  %empty = tensor.empty() : tensor<4xf16>
  tt.store %dstPtr, %empty : !tt.ptr<tensor<4xf16>>
  tt.return
}

// -----

// CHECK-LABEL: @tensorEmptyNullptr
// CHECK-NOT: tensor.empty
// CHECK: %[[CST_0:.*]] = arith.constant 0 : i64
// CHECK: %[[NULLPTR:.*]] = tt.int_to_ptr %[[CST_0:.*]] : i64 -> !tt.ptr<i32>
// CHECK: %[[SPLAT_NULLPTR:.*]] = tt.splat %[[NULLPTR]] : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
// CHECK: tt.store %{{.*}}, %[[SPLAT_NULLPTR]] : !tt.ptr<tensor<4x!tt.ptr<i32>>>
tt.func @tensorEmptyNullptr(%dst: !tt.ptr<!tt.ptr<i32>>) {
  %0 = arith.constant 4 : i64
  %1 = arith.constant 1 : i64
  %2 = arith.constant 0 : i32

  %dstPtr = tt.make_tensor_ptr %dst, [%0], [%1], [%2] {order = array<i32: 0>} : !tt.ptr<tensor<4x!tt.ptr<i32>>>
  %empty = tensor.empty() : tensor<4x!tt.ptr<i32>>
  tt.store %dstPtr, %empty : !tt.ptr<tensor<4x!tt.ptr<i32>>>
  tt.return
}

// -----
// expected-error @+1 {{Unsupported tensor dialect operations found in the SIMT kernel}}
module {
  tt.func @unsupported(%dst: !tt.ptr<i32>) {
    %0 = arith.constant 4 : i64
    %1 = arith.constant 1 : i64
    %2 = arith.constant 0 : i32

    %dstPtr = tt.make_tensor_ptr %dst, [%0], [%1], [%2] {order = array<i32: 0>} : !tt.ptr<tensor<4xi32>>

    %tmp = tt.make_range {end = 3 : i32, start = 1 : i32} : tensor<2xi32>
    %extraDim = arith.constant 1 : index
    // expected-error @+1 {{tensor.pad is an unsupported tensor dialect operation}}
    %res = tensor.pad %tmp low[%extraDim] high[%extraDim] {
      ^bb0(%arg0: index):
      %val = arith.index_cast %arg0 : index to i32
      // expected-error @+1 {{tensor.yield is an unsupported tensor dialect operation}}
      tensor.yield %val : i32
    } : tensor<2xi32> to tensor<4xi32>

    tt.store %dstPtr, %res : !tt.ptr<tensor<4xi32>>
    tt.return
  }
}
