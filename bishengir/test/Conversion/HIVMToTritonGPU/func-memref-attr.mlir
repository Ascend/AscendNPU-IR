// RUN: bishengir-opt -convert-hivm-to-tritongpu %s -split-input-file -verify-diagnostics | FileCheck %s

// Test that memref_attr with value 1 causes tt.addptr to be generated for the offset
// CHECK-LABEL: tt.func @memref_with_offset_kernel(
// CHECK-SAME: %arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !tt.ptr<f32>, %arg6: !tt.ptr<f32>, %arg7: i64, %arg8: i64, %arg9: i64)
// CHECK: %[[PTR:.*]] = tt.addptr %arg0, %arg2 : !tt.ptr<f32>, i64
// CHECK: tt.return

module {
  func.func @memref_with_offset_kernel(%arg0: memref<?xf32, strided<[1], offset: ?>>, %arg1: memref<8xf32>) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>, memref_attr = array<i32: 1>} {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %reinterpret_cast = memref.subview %arg0[0] [8] [1] : memref<?xf32, strided<[1], offset: ?>> to memref<8xf32, strided<[1], offset: ?>>
    %0 = hivm.hir.local_load ins(%arg1 : memref<8xf32>) -> tensor<8xf32>
    hivm.hir.local_store ins(%reinterpret_cast : memref<8xf32, strided<[1], offset: ?>>, %0 : tensor<8xf32>)
    return
  }
}

// -----

// Test that memref_attr with value 0 does NOT generate tt.addptr
// CHECK-LABEL: tt.func @memref_without_offset_kernel(
// CHECK-SAME: %arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !tt.ptr<f32>, %arg6: !tt.ptr<f32>, %arg7: i64, %arg8: i64, %arg9: i64)
// CHECK-NOT: tt.addptr %arg0, %arg2
// CHECK: tt.return

module {
  func.func @memref_without_offset_kernel(%arg0: memref<?xf32>, %arg1: memref<8xf32>) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>, memref_attr = array<i32: 0>} {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1]>>
    %0 = hivm.hir.local_load ins(%arg1 : memref<8xf32>) -> tensor<8xf32>
    hivm.hir.local_store ins(%reinterpret_cast : memref<8xf32, strided<[1]>>, %0 : tensor<8xf32>)
    return
  }
}

// -----

// Test that memref_attr with value 1 and static offset [2] generates tt.addptr with constant offset
// CHECK-LABEL: tt.func @memref_with_static_offset_kernel(
// CHECK-SAME: %arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !tt.ptr<f32>, %arg6: !tt.ptr<f32>, %arg7: i64, %arg8: i64, %arg9: i64)
// CHECK: %[[CST:.*]] = arith.constant 2 : i64
// CHECK: %[[PTR:.*]] = tt.addptr %arg0, %[[CST]] : !tt.ptr<f32>, i64
// CHECK: tt.return

module {
  func.func @memref_with_static_offset_kernel(%arg0: memref<8xf32, strided<[1], offset: 2>>, %arg1: memref<8xf32>) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>, memref_attr = array<i32: 1>} {
    %0 = hivm.hir.local_load ins(%arg1 : memref<8xf32>) -> tensor<8xf32>
    hivm.hir.local_store ins(%arg0 : memref<8xf32, strided<[1], offset: 2>>, %0 : tensor<8xf32>)
    return
  }
}
