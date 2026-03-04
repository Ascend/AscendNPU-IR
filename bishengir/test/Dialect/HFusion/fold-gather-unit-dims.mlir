
// RUN: bishengir-opt --cse --canonicalize --hfusion-generalize --hfusion-fold-unit-dims --convert-linalg-to-hfusion -split-input-file %s | FileCheck %s --check-prefix=GATHER

// -----

// GATHER-LABEL: func.func @gather_3x1_1x1({{.*}}: memref<?xf16>, {{.*}}: memref<?xf16>, {{.*}}: memref<?xi64>)
// GATHER: hfusion.gather {operandSegmentSizes = array<i32: 2, 1>} ins({{.*}}, {{.*}} : tensor<3xf16>, tensor<1xi64>) outs({{.*}} : tensor<1xf16>) axis = 0 -> tensor<1xf16>
func.func @gather_3x1_1x1(%arg2: memref<?xf16>, %arg3: memref<?xf16>, %arg4: memref<?xi64>) {
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [3, 1], strides: [1, 1] : memref<?xf16> to memref<3x1xf16, strided<[1, 1]>>
  %alloc = memref.alloc() : memref<3x1xf16>
  memref.copy %reinterpret_cast, %alloc : memref<3x1xf16, strided<[1, 1]>> to memref<3x1xf16>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<3x1xf16>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1, 1], strides: [1, 1] : memref<?xi64> to memref<1x1xi64, strided<[1, 1]>>
  %alloc_1 = memref.alloc() : memref<1x1xi64>
  memref.copy %reinterpret_cast_0, %alloc_1 : memref<1x1xi64, strided<[1, 1]>> to memref<1x1xi64>
  %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<1x1xi64>
  %2 = tensor.empty() : tensor<1x1xf16>
  %3 = hfusion.gather {operandSegmentSizes = array<i32: 2, 1>} ins(%0, %1 : tensor<3x1xf16>, tensor<1x1xi64>) outs(%2 : tensor<1x1xf16>) axis = 0 -> tensor<1x1xf16>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1, 1], strides: [1, 1] : memref<?xf16> to memref<1x1xf16, strided<[1, 1]>>
  bufferization.materialize_in_destination %3 in writable %reinterpret_cast_2 : (tensor<1x1xf16>, memref<1x1xf16, strided<[1, 1]>>) -> ()
  return
}

// -----

// GATHER-LABEL: func.func @gather_1x8x8x4_1x8x4x4({{.*}}: memref<?xf16>, {{.*}}: memref<?xf16>, {{.*}}: memref<?xi64>)
// GATHER: hfusion.gather {operandSegmentSizes = array<i32: 2, 1>} ins({{.*}}, {{.*}} : tensor<8x8x4xf16>, tensor<8x4x4xi64>) outs({{.*}} : tensor<8x4x4xf16>) axis = 1 -> tensor<8x4x4xf16>
func.func @gather_1x8x8x4_1x8x4x4(%arg2: memref<?xf16>, %arg3: memref<?xf16>, %arg4: memref<?xi64>) {
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1, 8, 8, 4], strides: [256, 32, 4, 1] : memref<?xf16> to memref<1x8x8x4xf16, strided<[256, 32, 4, 1]>>
  %alloc = memref.alloc() : memref<1x8x8x4xf16>
  memref.copy %reinterpret_cast, %alloc : memref<1x8x8x4xf16, strided<[256, 32, 4, 1]>> to memref<1x8x8x4xf16>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<1x8x8x4xf16>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1, 8, 4, 4], strides: [128, 16, 4, 1] : memref<?xi64> to memref<1x8x4x4xi64, strided<[128, 16, 4, 1]>>
  %alloc_1 = memref.alloc() : memref<1x8x4x4xi64>
  memref.copy %reinterpret_cast_0, %alloc_1 : memref<1x8x4x4xi64, strided<[128, 16, 4, 1]>> to memref<1x8x4x4xi64>
  %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<1x8x4x4xi64>
  %2 = tensor.empty() : tensor<1x8x4x4xf16>
  %3 = hfusion.gather {operandSegmentSizes = array<i32: 2, 1>} ins(%0, %1 : tensor<1x8x8x4xf16>, tensor<1x8x4x4xi64>) outs(%2 : tensor<1x8x4x4xf16>) axis = 2 -> tensor<1x8x4x4xf16>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1, 8, 4, 4], strides: [128, 16, 4, 1] : memref<?xf16> to memref<1x8x4x4xf16, strided<[128, 16, 4, 1]>>
  bufferization.materialize_in_destination %3 in writable %reinterpret_cast_2 : (tensor<1x8x4x4xf16>, memref<1x8x4x4xf16, strided<[128, 16, 4, 1]>>) -> ()
  return
}
