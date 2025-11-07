// RUN: bishengir-opt %s -canonicalize-ext -cse -split-input-file | FileCheck %s

// -----
// CHECK-LABEL: @test_fold_copy
// CHECK-NOT: memref.copy
module {
  func.func @test_fold_copy(%arg0: memref<3072xf32>, %arg1: index, %arg2: index) -> memref<1x3072xf32> 
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c1 = arith.constant 1 : index
    %subview = memref.subview %arg0[%arg1] [%arg2] [1] : memref<3072xf32> to memref<?xf32, strided<[1], offset: ?>>
    %expand_shape = memref.expand_shape %subview [[0, 1]] output_shape [1, %arg2] : memref<?xf32, strided<[1], offset: ?>> into memref<1x?xf32, strided<[?, 1], offset: ?>>
    annotation.mark %expand_shape {} : memref<1x?xf32, strided<[?, 1], offset: ?>>
    %dim = memref.dim %expand_shape, %c1 : memref<1x?xf32, strided<[?, 1], offset: ?>>
    %alloc = memref.alloc(%dim) {alignment = 64 : i64} : memref<1x?xf32>
    memref.copy %expand_shape, %alloc : memref<1x?xf32, strided<[?, 1], offset: ?>> to memref<1x?xf32>
    %collapse_shape = memref.collapse_shape %alloc [[0, 1]] : memref<1x?xf32> into memref<?xf32>
    memref.copy %collapse_shape, %subview : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
    %expand_shape_0 = memref.expand_shape %arg0 [[0, 1]] output_shape [1, 3072] : memref<3072xf32> into memref<1x3072xf32>
    return %expand_shape_0 : memref<1x3072xf32>
  }
}

// -----

// CHECK-LABEL: @test_not_fold_copy_0
// CHECK: memref.copy
// CHECK: memref.copy
module {
  func.func @test_not_fold_copy_0(
    %arg0: memref<24x12xf32, strided<[?, ?], offset: ?>>, %arg1: memref<12x25xf32, strided<[?, ?], offset: ?>>, 
    %arg2: memref<24x25xf32, strided<[?, ?], offset: ?>>, 
    %arg3: index, %arg4: index, %arg5: index) -> memref<24x25xf32, strided<[?, ?], offset: ?>> {
    %cst = arith.constant 0.000000e+00 : f32
    %subview = memref.subview %arg0[%arg3, %arg5] [4, %arg5] [1, 1] : memref<24x12xf32, strided<[?, ?], offset: ?>> to memref<4x?xf32, strided<[?, ?], offset: ?>>
    %subview_0 = memref.subview %arg1[%arg5, %arg4] [%arg5, 5] [1, 1] : memref<12x25xf32, strided<[?, ?], offset: ?>> to memref<?x5xf32, strided<[?, ?], offset: ?>>
    %subview_1 = memref.subview %arg2[%arg3, %arg4] [4, 5] [1, 1] : memref<24x25xf32, strided<[?, ?], offset: ?>> to memref<4x5xf32, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc() : memref<4x7xf32, 3>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<4x7xf32, 3>)
    %alloc_3 = memref.alloc() : memref<7x5xf32, 3>
    linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<7x5xf32, 3>)
    %alloc_5 = memref.alloc() : memref<4x5xf32, 3>
    memref.copy %subview_1, %alloc_5 : memref<4x5xf32, strided<[?, ?], offset: ?>> to memref<4x5xf32, 3>
    linalg.matmul ins(%alloc, %alloc_3 : memref<4x7xf32, 3>, memref<7x5xf32, 3>) outs(%alloc_5 : memref<4x5xf32, 3>)
    memref.copy %alloc_5, %subview_1 : memref<4x5xf32, 3> to memref<4x5xf32, strided<[?, ?], offset: ?>>
    memref.dealloc %alloc : memref<4x7xf32, 3>
    memref.dealloc %alloc_3 : memref<7x5xf32, 3>
    memref.dealloc %alloc_5 : memref<4x5xf32, 3>
    return %arg2 : memref<24x25xf32, strided<[?, ?], offset: ?>>
  }
}

// -----

// CHECK-LABEL: @test_not_fold_copy_1
// CHECK: memref.copy
// CHECK: memref.copy
// CHECK: memref.copy
module {
  func.func @test_not_fold_copy_1(%arg0: memref<3072xf32>, %arg1: memref<1x?xf32>, %arg2: index) -> memref<1x?xf32, strided<[?, 1], offset: ?>> 
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c1 = arith.constant 1 : index
    %subview = memref.subview %arg0[%arg2] [%arg2] [1] : memref<3072xf32> to memref<?xf32, strided<[1], offset: ?>>
    %expand_shape = memref.expand_shape %subview [[0, 1]] output_shape [1, %arg2] : memref<?xf32, strided<[1], offset: ?>> into memref<1x?xf32, strided<[?, 1], offset: ?>>
    %dim = memref.dim %expand_shape, %c1 : memref<1x?xf32, strided<[?, 1], offset: ?>>
    %alloc = memref.alloc(%dim) {alignment = 64 : i64} : memref<1x?xf32>
    memref.copy %expand_shape, %alloc : memref<1x?xf32, strided<[?, 1], offset: ?>> to memref<1x?xf32>
    memref.copy %expand_shape, %arg1 : memref<1x?xf32, strided<[?, 1], offset: ?>> to memref<1x?xf32>
    %collapse_shape = memref.collapse_shape %alloc [[0, 1]] : memref<1x?xf32> into memref<?xf32>
    memref.copy %collapse_shape, %subview : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
    return %expand_shape : memref<1x?xf32, strided<[?, 1], offset: ?>>
  }
}

// -----

// CHECK-LABEL: func.func @test_reshape_trace
// CHECK-NOT: memref.copy
#map = affine_map<()[s0] -> (-s0 + 1)>
#map1 = affine_map<()[s0, s1] -> (s0 * 32 + s1 * 32)>
#map2 = affine_map<()[s0, s1] -> (s0 * -32 - s1 * 32 + 32, 32)>
module {
  func.func @test_reshape_trace(%arg0: memref<24x128xf32>, %arg1: memref<24x128xf32>, %arg2: memref<24x32xf32>, %arg3: memref<24x32xf32>, %arg4: memref<32x4xf32>) -> memref<32x4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c48 = arith.constant 48 : index
    %0 = scf.for %arg5 = %c0 to %c48 step %c1 iter_args(%arg6 = %arg4) -> (memref<32x4xf32>) {
      %1 = affine.apply #map()[%arg5]
      %2 = scf.for %arg7 = %c0 to %1 step %c48 iter_args(%arg8 = %arg6) -> (memref<32x4xf32>) {
        %3 = affine.apply #map1()[%arg5, %arg7]
        %4 = affine.min #map2()[%arg5, %arg7]
        %subview = memref.subview %arg8[%3, 0] [%4, 4] [1, 1] : memref<32x4xf32> to memref<?x4xf32, strided<[4, 1], offset: ?>>
        %expand_shape = memref.expand_shape %subview [[0, 1], [2]] output_shape [1, %4, 4] : memref<?x4xf32, strided<[4, 1], offset: ?>> into memref<1x?x4xf32, strided<[?, 4, 1], offset: ?>>
        %dim = memref.dim %expand_shape, %c1 : memref<1x?x4xf32, strided<[?, 4, 1], offset: ?>>
        %alloc = memref.alloc(%dim) {alignment = 64 : i64} : memref<1x?x4xf32>
        memref.copy %expand_shape, %alloc : memref<1x?x4xf32, strided<[?, 4, 1], offset: ?>> to memref<1x?x4xf32>
        %collapse_shape = memref.collapse_shape %alloc [[0, 1], [2]] : memref<1x?x4xf32> into memref<?x4xf32>
        memref.copy %collapse_shape, %subview : memref<?x4xf32> to memref<?x4xf32, strided<[4, 1], offset: ?>>
        scf.yield %arg8 : memref<32x4xf32>
      }
      scf.yield %2 : memref<32x4xf32>
    }
    return %0 : memref<32x4xf32>
  }
}