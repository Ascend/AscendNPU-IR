// RUN: bishengir-opt %s --convert-hivm-to-tritongpu -split-input-file | FileCheck %s

// CHECK-LABEL: tt.func @scalar_load_reinterpret_cast_dynamic_offset
// CHECK-SAME:  %{{[a-zA-Z0-9_]+}}: !tt.ptr<i32>, %[[ARG0:[a-zA-Z0-9_]+]]: !tt.ptr<i32>,
// CHECK:       %[[OFF:.*]] = arith.index_cast %{{.*}} : index to i64
// CHECK:       %[[PTR:.*]] = tt.addptr %[[ARG0]], %[[OFF]] : !tt.ptr<i32>, i64
// CHECK:       %{{.*}} = tt.load %[[PTR]] : !tt.ptr<i32>
// CHECK-NOT:   arith.addi %[[OFF]]
// CHECK-NOT:   memref.load
// CHECK-NOT:   memref.reinterpret_cast
// CHECK-NOT:   builtin.unrealized_conversion_cast
// CHECK:       tt.return

module {
  func.func @scalar_load_reinterpret_cast_dynamic_offset(%arg0: memref<?xi32>, %arg1: index, %arg2: memref<16xi32>) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %c0 = arith.constant 0 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %0 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %empty = tensor.empty() : tensor<16xi32>
    %1 = hivm.hir.vbrc ins(%0 : i32) outs(%empty : tensor<16xi32>) -> tensor<16xi32>
    hivm.hir.store ins(%1 : tensor<16xi32>) outs(%arg2 : memref<16xi32>)
    return
  }
}

// -----

// CHECK-LABEL: tt.func @scalar_load_static_offset_strided
// CHECK-SAME:  %{{[a-zA-Z0-9_]+}}: !tt.ptr<f32>, %[[ARG0:[a-zA-Z0-9_]+]]: !tt.ptr<f32>,
// CHECK-DAG:   %[[C14:.*]] = arith.constant 14 : i64
// CHECK:       %[[PTR:.*]] = tt.addptr %[[ARG0]], %[[C14]] : !tt.ptr<f32>, i64
// CHECK:       %{{.*}} = tt.load %[[PTR]] : !tt.ptr<f32>
// CHECK-NOT:   memref.load
// CHECK-NOT:   builtin.unrealized_conversion_cast
// CHECK:       tt.return

module {
  func.func @scalar_load_static_offset_strided(%arg0: memref<?xf32>, %arg1: memref<16xf32>) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %c2 = arith.constant 2 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [8], sizes: [4], strides: [3] : memref<?xf32> to memref<4xf32, strided<[3], offset: 8>>
    %0 = memref.load %reinterpret_cast[%c2] : memref<4xf32, strided<[3], offset: 8>>
    %empty = tensor.empty() : tensor<16xf32>
    %1 = hivm.hir.vbrc ins(%0 : f32) outs(%empty : tensor<16xf32>) -> tensor<16xf32>
    hivm.hir.store ins(%1 : tensor<16xf32>) outs(%arg1 : memref<16xf32>)
    return
  }
}

// -----

// CHECK-LABEL: tt.func @scalar_load_blockarg_dynamic_offset
// CHECK-SAME:  %{{[a-zA-Z0-9_]+}}: !tt.ptr<i64>, %[[BASE:[a-zA-Z0-9_]+]]: !tt.ptr<i64>, %[[DESCOFF:[a-zA-Z0-9_]+]]: i64,
// CHECK:       %[[PTR:.*]] = tt.addptr %[[BASE]], %[[DESCOFF]] : !tt.ptr<i64>, i64
// CHECK:       %{{.*}} = tt.load %[[PTR]] : !tt.ptr<i64>
// CHECK-NOT:   tt.addptr %[[PTR]]
// CHECK-NOT:   memref.load
// CHECK-NOT:   builtin.unrealized_conversion_cast
// CHECK:       tt.return

module {
  func.func @scalar_load_blockarg_dynamic_offset(%arg0: memref<1xi64, strided<[1], offset: ?>>, %arg1: memref<16xi64>) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %c0 = arith.constant 0 : index
    %0 = memref.load %arg0[%c0] : memref<1xi64, strided<[1], offset: ?>>
    %empty = tensor.empty() : tensor<16xi64>
    %1 = hivm.hir.vbrc ins(%0 : i64) outs(%empty : tensor<16xi64>) -> tensor<16xi64>
    hivm.hir.store ins(%1 : tensor<16xi64>) outs(%arg1 : memref<16xi64>)
    return
  }
}

// -----

// CHECK-LABEL: tt.func @scalar_store_reinterpret_cast_dynamic_offset
// CHECK-SAME:  %{{[a-zA-Z0-9_]+}}: !tt.ptr<i32>, %[[ARG0:[a-zA-Z0-9_]+]]: !tt.ptr<i32>,
// CHECK:       %[[OFF:.*]] = arith.index_cast %{{.*}} : index to i64
// CHECK:       %[[PTR:.*]] = tt.addptr %[[ARG0]], %[[OFF]] : !tt.ptr<i32>, i64
// CHECK:       tt.store %[[PTR]], %{{.*}} : !tt.ptr<i32>
// CHECK-NOT:   memref.store
// CHECK-NOT:   builtin.unrealized_conversion_cast
// CHECK:       tt.return

module {
  func.func @scalar_store_reinterpret_cast_dynamic_offset(%arg0: memref<?xi32>, %arg1: index, %arg2: i32) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %c0 = arith.constant 0 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    memref.store %arg2, %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>>
    return
  }
}
