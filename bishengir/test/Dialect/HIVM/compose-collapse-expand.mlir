// RUN: bishengir-opt %s  --compose-collapse-expand -split-input-file | FileCheck %s

// CHECK: @compose_expand_of_collapse
// CHECK-NOT: collapse
// CHECK-NOT: expand
  func.func @compose_expand_of_collapse(%arg0: memref<2x320x?x?xf16>, %arg1: memref<2x320x?x?xf16>, %arg2: memref<2x320x?x?xf16>, %arg3: !llvm.ptr, %arg4: memref<2xi64>) -> memref<2x320x?x?xf16> {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dim = memref.dim %arg0, %c3 : memref<2x320x?x?xf16>
    %dim_0 = memref.dim %arg0, %c2 : memref<2x320x?x?xf16>
    %s0 = symbol.symbolic_int @s0 [], affine_map<() -> ()> {max_val = 1024 : i64, min_val = 128 : i64} : index
    %s1 = symbol.symbolic_int @s1 [], affine_map<() -> ()> {max_val = 512 : i64, min_val = 256 : i64} : index
    symbol.bind_symbolic_shape %arg0, [%s0, %s1], affine_map<()[s0, s1] -> (2, 320, s0, s1)> : memref<2x320x?x?xf16>
    symbol.bind_symbolic_shape %arg1, [%s0, %s1], affine_map<()[s0, s1] -> (2, 320, s0, s1)> : memref<2x320x?x?xf16>
    symbol.bind_symbolic_shape %arg2, [%s0, %s1], affine_map<()[s0, s1] -> (2, 320, s0, s1)> : memref<2x320x?x?xf16>
    %collapse_shape_2 = memref.collapse_shape %arg2 [[0, 1, 2, 3]] : memref<2x320x?x?xf16> into memref<?xf16>
    %expand_shape = memref.expand_shape %collapse_shape_2 [[0, 1, 2, 3]] output_shape [2, 320, %dim_0, %dim] : memref<?xf16> into memref<2x320x?x?xf16>
    return %expand_shape : memref<2x320x?x?xf16>
  }


