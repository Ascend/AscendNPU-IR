// RUN: bishengir-opt %s --canonicalize-ext -split-input-file | FileCheck %s

// CHECK-LABEL: func @reinterpret_constant_arg_folder_unranked_memref
func.func @reinterpret_constant_arg_folder_unranked_memref(%arg0 : memref<*xf16>) -> memref<?xf16, strided<[?], offset: ?>> {
  %offset = arith.constant 0 : index
  %size = arith.constant 1024 : index
  %stride = arith.constant 1 : index
  // CHECK: memref.reinterpret_cast %arg0 to offset: [0], sizes: [1024], strides: [1] : memref<*xf16> to memref<1024xf16, strided<[1]>>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%offset], sizes: [%size], strides: [%stride] : memref<*xf16> to memref<?xf16, strided<[?], offset: ?>>
  return %reinterpret_cast : memref<?xf16, strided<[?], offset: ?>>
}

// -----

// CHECK-LABEL: func @reinterpret_constant_arg_folder_memref
func.func @reinterpret_constant_arg_folder_memref(%arg0 : memref<?xf16>) -> memref<?xf16, strided<[?], offset: ?>> {
  %offset = arith.constant 0 : index
  %size = arith.constant 1024 : index
  %stride = arith.constant 1 : index
  // CHECK: memref.reinterpret_cast %arg0 to offset: [0], sizes: [1024], strides: [1] : memref<?xf16> to memref<1024xf16, strided<[1]>>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%offset], sizes: [%size], strides: [%stride] : memref<?xf16> to memref<?xf16, strided<[?], offset: ?>>
  return %reinterpret_cast : memref<?xf16, strided<[?], offset: ?>>
}