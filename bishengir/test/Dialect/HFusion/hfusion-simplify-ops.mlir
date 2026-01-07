// RUN: bishengir-opt %s -hfusion-simplify-ops -split-input-file | FileCheck %s

// CHECK-LABEL: @unusedCast
// CHECK-SAME: (%[[arg0:.*]]: tensor<bf16>)
// CHECK-NOT: hfusion.cast
// CHECK: return %[[arg0]] : tensor<bf16>

func.func @unusedCast(%arg0 : tensor<bf16>) -> tensor<bf16> {
  %empty = tensor.empty() : tensor<f32>
  %0 = hfusion.cast ins(%arg0 : tensor<bf16>) outs(%empty : tensor<f32>) -> tensor<f32>
  return %arg0 : tensor<bf16>
}

// -----

// CHECK-LABEL: @sameTypes
// CHECK-SAME: (%[[arg0:.*]]: tensor<bf16>)
// CHECK: hfusion.cast
// CHECK-NOT: return %[[arg0]]

func.func @sameTypes(%arg0: tensor<bf16>) -> tensor<bf16> {
    %empty = tensor.empty() : tensor<bf16>
    %0 = hfusion.cast ins(%arg0 : tensor<bf16>) outs(%empty: tensor<bf16>) -> tensor<bf16>
    return %0 : tensor<bf16>
}

// -----

// CHECK-LABEL: @pair
// CHECK-SAME: (%[[arg0:.*]]: tensor<bf16>)
// CHECK-NOT: hfusion.cast
// CHECK: return %[[arg0]] : tensor<bf16>

func.func @pair(%arg0: tensor<bf16>) -> tensor<bf16> {
    %empty_f32 = tensor.empty() : tensor<f32>
    %empty_bf16 = tensor.empty() : tensor<bf16>
    %0 = hfusion.cast ins(%arg0 : tensor<bf16>) outs(%empty_f32 : tensor<f32>) -> tensor<f32>
    %1 = hfusion.cast ins(%0 : tensor<f32>) outs(%empty_bf16 : tensor<bf16>) -> tensor<bf16>
    return %1 : tensor<bf16>
}

// -----

// CHECK-LABEL: @symmetricChain
// CHECK-SAME: (%[[arg0:.*]]: tensor<bf16>)
// CHECK-4: hfusion.cast
// CHECK: return %[[arg0]] : tensor<bf16>

func.func @symmetricChain(%arg0: tensor<bf16>) -> tensor<bf16> {
    %empty_f32 = tensor.empty() : tensor<f32>
    %empty_i1 = tensor.empty() : tensor<i1>
    %empty_bf16 = tensor.empty() : tensor<bf16>
    %0 = hfusion.cast ins(%arg0 : tensor<bf16>) outs(%empty_f32 : tensor<f32>) -> tensor<f32>
    %1 = hfusion.cast ins(%0 : tensor<f32>) outs(%empty_i1 : tensor<i1>) -> tensor<i1>
    %2 = hfusion.cast ins(%1 : tensor<i1>) outs(%empty_f32 : tensor<f32>) -> tensor<f32>
    %3 = hfusion.cast ins(%2 : tensor<f32>) outs(%empty_bf16 : tensor<bf16>) -> tensor<bf16>
    return %3 : tensor<bf16>
}

// -----

// CHECK-LABEL: @asymmetricChain
// CHECK-3: hfusion.cast

func.func @asymmetricChain(%arg0: tensor<bf16>) -> tensor<bf16> {
    %empty_f32 = tensor.empty() : tensor<f32>
    %empty_i1 = tensor.empty() : tensor<i1>
    %empty_bf16 = tensor.empty() : tensor<bf16>
    %0 = hfusion.cast ins(%arg0 : tensor<bf16>) outs(%empty_f32 : tensor<f32>) -> tensor<f32>
    %1 = hfusion.cast ins(%0 : tensor<f32>) outs(%empty_i1 : tensor<i1>) -> tensor<i1>
    %2 = hfusion.cast ins(%1 : tensor<i1>) outs(%empty_bf16 : tensor<bf16>) -> tensor<bf16>
    return %2 : tensor<bf16>
}

// -----

// CHECK-LABEL: @unusedChain
// CHECK-SAME: (%[[arg0:.*]]: tensor<bf16>)
// CHECK-NOT: hfusion.cast
// CHECK: return %[[arg0]] : tensor<bf16>

func.func @unusedChain(%arg0: tensor<bf16>) -> tensor<bf16> {
    %empty_f32 = tensor.empty() : tensor<f32>
    %empty_bf16 = tensor.empty() : tensor<bf16>
    %0 = hfusion.cast ins(%arg0 : tensor<bf16>) outs(%empty_f32 : tensor<f32>) -> tensor<f32>
    %1 = hfusion.cast ins(%0 : tensor<f32>) outs(%empty_bf16 : tensor<bf16>) -> tensor<bf16>
    return %arg0 : tensor<bf16>
}

// -----

// CHECK-LABEL: @bifurcation
// CHECK-SAME: (%[[arg0:.*]]: tensor<bf16>)
// CHECK-4: hfusion.cast

func.func @bifurcation(%arg0: tensor<bf16>) -> tensor<bf16> {
    %empty_f32 = tensor.empty() : tensor<f32>
    %empty_i1 = tensor.empty() : tensor<i1>
    %empty_bf16 = tensor.empty() : tensor<bf16>
    %0 = hfusion.cast ins(%arg0 : tensor<bf16>) outs(%empty_f32 : tensor<f32>) -> tensor<f32>
    %1 = hfusion.cast ins(%0 : tensor<f32>) outs(%empty_i1 : tensor<i1>) -> tensor<i1>
    %2 = hfusion.cast ins(%1 : tensor<i1>) outs(%empty_bf16 : tensor<bf16>) -> tensor<bf16>
    %3 = hfusion.cast ins(%1 : tensor<i1>) outs(%empty_f32 : tensor<f32>) -> tensor<f32>
    %4 = hfusion.cast ins(%3 : tensor<f32>) outs(%empty_bf16 : tensor<bf16>) -> tensor<bf16>
    %add_res = linalg.elemwise_binary { add, fun = #linalg.binary_fn<add> } ins(%2, %4 : tensor<bf16>, tensor<bf16>) outs(%empty_bf16 : tensor<bf16>) -> tensor<bf16>
    return %add_res : tensor<bf16>
}

// -----

// CHECK-LABEL: @bifurcationMultiOuts
// CHECK-SAME: (%[[arg0:.*]]: tensor<bf16>)
// CHECK: %[[cast0:.*]] = hfusion.cast ins(%[[arg0]] : tensor<bf16>)
// CHECK: return %[[cast0]], %[[arg0]] : tensor<f32>, tensor<bf16>
func.func @bifurcationMultiOuts(%arg0: tensor<bf16>) -> (tensor<f32>, tensor<bf16>) {
    %empty_f32 = tensor.empty() : tensor<f32>
    %empty_bf16 = tensor.empty() : tensor<bf16>
    %0 = hfusion.cast ins(%arg0 : tensor<bf16>) outs(%empty_f32 : tensor<f32>) -> tensor<f32>
    %1 = hfusion.cast ins(%0 : tensor<f32>) outs(%empty_bf16 : tensor<bf16>) -> tensor<bf16>
    return %0, %1 : tensor<f32>, tensor<bf16>
}

// -----

// CHECK-LABEL: @unusedBifurcation
// CHECK-SAME: (%[[arg0:.*]]: tensor<bf16>)
// CHECK-NOT: hfusion.cast
// CHECK: %[[result:.*]] = linalg.elemwise_binary
// CHECK: %[[arg0]], %[[arg0]] : tensor<bf16>, tensor<bf16>
// CHECK-NOT: hfusion.cast
// CHECK: return %[[result]] : tensor<bf16>

func.func @unusedBifurcation(%arg0: tensor<bf16>) -> tensor<bf16> {
    %empty_f32 = tensor.empty() : tensor<f32>
    %empty_i1 = tensor.empty() : tensor<i1>
    %empty_bf16 = tensor.empty() : tensor<bf16>
    %0 = hfusion.cast ins(%arg0 : tensor<bf16>) outs(%empty_f32 : tensor<f32>) -> tensor<f32>
    %1 = hfusion.cast ins(%0 : tensor<f32>) outs(%empty_i1 : tensor<i1>) -> tensor<i1>
    %2 = hfusion.cast ins(%1 : tensor<i1>) outs(%empty_bf16 : tensor<bf16>) -> tensor<bf16>
    %3 = hfusion.cast ins(%0 : tensor<f32>) outs(%empty_bf16 : tensor<bf16>) -> tensor<bf16>
    %add_res = linalg.elemwise_binary { add, fun = #linalg.binary_fn<add> } ins(%arg0, %3 : tensor<bf16>, tensor<bf16>) outs(%empty_bf16 : tensor<bf16>) -> tensor<bf16>
    return %add_res : tensor<bf16>
}

// -----

// CHECK-LABEL: @liveSingleCast
// CHECK-SAME: (%[[arg0:.*]]: tensor<bf16>)
// CHECK: %[[liveCast:.*]] = hfusion.cast ins(%[[arg0]]
// CHECK: return %[[liveCast]] : tensor<f32>

func.func @liveSingleCast(%arg0: tensor<bf16>) -> tensor<f32> {
    %empty_f32 = tensor.empty() : tensor<f32>
    %0 = hfusion.cast ins(%arg0 : tensor<bf16>) outs(%empty_f32 : tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @liveChain
// CHECK-SAME: (%[[arg0:.*]]: tensor<bf16>)
// CHECK: %[[cast0:.*]] = hfusion.cast ins(%[[arg0]]
// CHECK: %[[cast1:.*]] = hfusion.cast ins(%[[cast0]]
// CHECK: return %[[cast1]] : tensor<f32>

func.func @liveChain(%arg0: tensor<bf16>) -> tensor<f32> {
    %empty_i1 = tensor.empty() : tensor<i1>
    %empty_f32 = tensor.empty() : tensor<f32>
    %0 = hfusion.cast ins(%arg0 : tensor<bf16>) outs(%empty_i1 :tensor<i1>) -> tensor<i1>
    %1 = hfusion.cast ins(%0 : tensor<i1>) outs(%empty_f32 : tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: @liveBifurcation
// CHECK-SAME: (%[[arg0:.*]]: tensor<bf16>)
// CHECK: %[[cast0:.*]] = hfusion.cast ins(%[[arg0]]
// CHECK-NOT: hfusion.cast
// CHECK: %[[result:.*]] = linalg.elemwise_binary
// CHECK: ins(%[[cast0]], %[[cast0]]
// CHECK: return %[[result]] : tensor<f32>

func.func @liveBifurcation(%arg0: tensor<bf16>) -> tensor<f32> {
    %empty_bf16 = tensor.empty() : tensor<bf16>
    %empty_f32 = tensor.empty() : tensor<f32>
    %0 = hfusion.cast ins(%arg0 : tensor<bf16>) outs(%empty_f32 : tensor<f32>) -> tensor<f32>
    %1 = hfusion.cast ins(%0 : tensor<f32>) outs(%empty_bf16 : tensor<bf16>) -> tensor<bf16>
    %2 = hfusion.cast ins(%1 : tensor<bf16>) outs(%empty_f32 : tensor<f32>) -> tensor<f32>
    %add_res = linalg.elemwise_binary { add, fun = #linalg.binary_fn<add> } ins(%0, %2 : tensor<f32>, tensor<f32>) outs(%empty_f32 : tensor<f32>) -> tensor<f32>
    return %add_res : tensor<f32>
}

// -----

// CHECK-LABEL: @floorAndCeilCastF32
// CHECK-SAME: (%[[arg0:.*]]: tensor<f32>)
// CHECK: %[[floor:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<floor>} ins(%[[arg0]]
// CHECK: %[[ceil:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<ceil>} ins(%[[arg0]]

func.func @floorAndCeilCastF32(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %empty_floor = tensor.empty() : tensor<f32>
    %empty_ceil = tensor.empty() : tensor<f32>
    %0 = hfusion.cast {round_mode = #hfusion.round_mode<floor>} ins(%arg0 : tensor<f32>) outs(%empty_floor : tensor<f32>) -> tensor<f32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<ceil>} ins(%arg0 : tensor<f32>) outs(%empty_ceil : tensor<f32>) -> tensor<f32>
    return %0, %1 : tensor<f32>, tensor<f32>
}

// -----

// CHECK-LABEL: @castBetweenSameType
// CHECK: hfusion.cast
// CHECK: hfusion.cast
// CHECK: hfusion.cast

func.func @castBetweenSameType(%arg0: tensor<f16>) -> tensor<f16>{
    %empty_0 = tensor.empty() : tensor<f32>
    %empty_1 = tensor.empty() : tensor<f32>
    %empty_2 = tensor.empty() : tensor<f16>
    %0 = hfusion.cast ins(%arg0 : tensor<f16>) outs(%empty_0 : tensor<f32>) -> tensor<f32>
    %1 = hfusion.cast ins(%0 : tensor<f32>) outs(%empty_1 : tensor<f32>) -> tensor<f32>
    %2 = hfusion.cast ins(%1 : tensor<f32>) outs(%empty_2 : tensor<f16>) -> tensor<f16>
    return %2 : tensor<f16>
}

// -----

// Check not to remove live transpose (memref)
// CHECK-LABEL: func @liveTransposeMemref
// CHECK: linalg.transpose
func.func @liveTransposeMemref(%arg0: memref<1x16xf32>, %arg1: memref<16x1xf32>) {
  linalg.transpose ins(%arg0 : memref<1x16xf32>) outs(%arg1 : memref<16x1xf32>) permutation = [1, 0]
  return
}

// -----

// Check not to remove live transpose (tensor)
// CHECK-LABEL: func @liveTransposeTensor
// CHECK: linalg.transpose
func.func @liveTransposeTensor(%arg0: tensor<1x16xf32>) -> tensor<16x1xf32> {
  %empty = tensor.empty() : tensor<16x1xf32>
  %0 = linalg.transpose ins(%arg0 : tensor<1x16xf32>) outs(%empty : tensor<16x1xf32>) permutation = [1, 0]
  return %0 : tensor<16x1xf32>
}

// -----

// CHECK-LABEL: func @unusedTranspose
// CHECK-NOT: linalg.transpose
func.func @unusedTranspose(%arg0 : tensor<1x16xf32>) -> tensor<1x16xf32> {
  %empty = tensor.empty() : tensor<16x1xf32>
  %0 = linalg.transpose ins(%arg0 : tensor<1x16xf32>) outs(%empty : tensor<16x1xf32>) permutation = [1, 0]
  return %arg0 : tensor<1x16xf32>
}

// -----

// CHECK-LABEL: func @TransposeChain0
// CHECK: (%[[arg0:.*]]: tensor<1x16xf32>)
// CHECK-NOT: linalg.transpose
// CHECK: return %[[arg0]]
func.func @TransposeChain0(%arg0: tensor<1x16xf32>) -> tensor<1x16xf32> {
  %empty = tensor.empty() : tensor<16x1xf32>
  %empty_transpose = tensor.empty() : tensor<1x16xf32>
  %0 = linalg.transpose ins(%arg0 : tensor<1x16xf32>) outs(%empty : tensor<16x1xf32>) permutation = [1, 0]
  %1 = linalg.transpose ins(%0 : tensor<16x1xf32>) outs(%empty_transpose : tensor<1x16xf32>) permutation = [1, 0]
  return %1 : tensor<1x16xf32>
}

// -----

// CHECK-LABEL: func @TransposeChain1
// CHECK: (%[[arg0:.*]]: tensor<1x16xf32>)
// CHECK: %[[trans:.*]] = linalg.transpose ins(%[[arg0]]
// CHECK: return %[[trans]]
func.func @TransposeChain1(%arg0: tensor<1x16xf32>) -> tensor<16x1xf32> {
  %empty = tensor.empty() : tensor<16x1xf32>
  %empty_transpose = tensor.empty() : tensor<1x16xf32>
  %0 = linalg.transpose ins(%arg0 : tensor<1x16xf32>) outs(%empty : tensor<16x1xf32>) permutation = [1, 0]
  %1 = linalg.transpose ins(%0 : tensor<16x1xf32>) outs(%empty_transpose : tensor<1x16xf32>) permutation = [1, 0]
  %2 = linalg.transpose ins(%1 : tensor<1x16xf32>) outs(%empty : tensor<16x1xf32>) permutation = [1, 0]
  return %2 : tensor<16x1xf32>
}

// -----

// CHECK-LABEL: func @TransposeChain2
// CHECK: (%[[arg0:.*]]: tensor<1x16xf32>)
// CHECK-NOT: linalg.transpose
// CHECK: return %[[arg0]]
func.func @TransposeChain2(%arg0: tensor<1x16xf32>) -> tensor<1x16xf32> {
  %empty = tensor.empty() : tensor<16x1xf32>
  %empty_transpose = tensor.empty() : tensor<1x16xf32>
  %0 = linalg.transpose ins(%arg0 : tensor<1x16xf32>) outs(%empty : tensor<16x1xf32>) permutation = [1, 0]
  %1 = linalg.transpose ins(%0 : tensor<16x1xf32>) outs(%empty_transpose : tensor<1x16xf32>) permutation = [1, 0]
  %2 = linalg.transpose ins(%1 : tensor<1x16xf32>) outs(%empty : tensor<16x1xf32>) permutation = [1, 0]
  %3 = linalg.transpose ins(%2 : tensor<16x1xf32>) outs(%empty_transpose : tensor<1x16xf32>) permutation = [1, 0]
  return %3 : tensor<1x16xf32>
}

// -----

// CHECK-LABEL: func @TransposeChain3
// CHECK: (%[[arg0:.*]]: tensor<1x16x8xf32>)
// CHECK-NOT: linalg.transpose
// CHECK: return %[[arg0]]
func.func @TransposeChain3(%arg0: tensor<1x16x8xf32>) -> tensor<1x16x8xf32> {
  %empty0 = tensor.empty() : tensor<16x1x8xf32>
  %empty1 = tensor.empty() : tensor<8x1x16xf32>
  %empty2 = tensor.empty() : tensor<1x8x16xf32>
  %empty3 = tensor.empty() : tensor<1x16x8xf32>
  %0 = linalg.transpose ins(%arg0 : tensor<1x16x8xf32>) outs(%empty0 : tensor<16x1x8xf32>) permutation = [1, 0, 2]
  %1 = linalg.transpose ins(%0 : tensor<16x1x8xf32>) outs(%empty1 : tensor<8x1x16xf32>) permutation = [2, 1, 0]
  %2 = linalg.transpose ins(%1 : tensor<8x1x16xf32>) outs(%empty2 : tensor<1x8x16xf32>) permutation = [1, 0, 2]
  %3 = linalg.transpose ins(%2 : tensor<1x8x16xf32>) outs(%empty3 : tensor<1x16x8xf32>) permutation = [0, 2, 1]
  return %3 : tensor<1x16x8xf32>
}

// -----
// CHECK-LABEL:   func.func @castInLoop(
// CHECK-SAME:                            %[[VAL_0:.*]]: tensor<1x1x12xbf16>,
// CHECK-SAME:                            %[[VAL_1:.*]]: tensor<1x1x12xbf16>) -> tensor<1x1x12xbf16> {
func.func @castInLoop(%arg0: tensor<1x1x12xbf16>, %arg1: tensor<1x1x12xbf16>) -> tensor<1x1x12xbf16> {
  // CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
  // CHECK:           %[[VAL_3:.*]] = arith.constant 4 : i32
  // CHECK:           %[[VAL_4:.*]] = tensor.empty
  // CHECK:           %[[VAL_5:.*]] = tensor.empty
  // CHECK:           %[[VAL_6:.*]] = hfusion.cast 
  // CHECK:           %[[VAL_7:.*]] = hfusion.cast 
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %0 = tensor.empty() : tensor<1x1x12xbf16>
  %1 = tensor.empty() : tensor<1x1x12xf32>
  %2 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<1x1x12xbf16>) outs(%1 : tensor<1x1x12xf32>) -> tensor<1x1x12xf32>
  %4 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%0 : tensor<1x1x12xbf16>) outs(%1 : tensor<1x1x12xf32>) -> tensor<1x1x12xf32>
  
  // CHECK:           %[[VAL_8:.*]] = tensor.empty() : tensor<1x1x12xf32>
  // CHECK:           %[[VAL_9:.*]] = hfusion.cast {{.*}} ins(%[[VAL_1]] : tensor<1x1x12xbf16>) outs(%[[VAL_8]] : tensor<1x1x12xf32>) -> tensor<1x1x12xf32>
  // CHECK:           %[[VAL_10:.*]] = scf.for %[[VAL_11:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_2]] iter_args(%[[VAL_12:.*]] = %[[VAL_9]]) -> (tensor<1x1x12xf32>)  : i32 {
  // CHECK:             %[[VAL_13:.*]] = linalg.elemwise_binary {{.*}} ins(%[[VAL_12]], %[[VAL_6]] : tensor<1x1x12xf32>, tensor<1x1x12xf32>) outs(%[[VAL_7]] : tensor<1x1x12xf32>) -> tensor<1x1x12xf32>
  // CHECK:             scf.yield %[[VAL_13]] : tensor<1x1x12xf32>
  // CHECK:           }
  // CHECK:           %[[VAL_14:.*]] = tensor.empty() : tensor<1x1x12xbf16>
  // CHECK:           %[[VAL_15:.*]] = hfusion.cast {{.*}} ins(%[[VAL_10]] : tensor<1x1x12xf32>) outs(%[[VAL_14]] : tensor<1x1x12xbf16>) -> tensor<1x1x12xbf16>
  // CHECK:           return %[[VAL_15]] : tensor<1x1x12xbf16>
  %7 = scf.for %arg2 = %c1_i32 to %c4_i32 step %c1_i32 iter_args(%arg3 = %arg1) -> (tensor<1x1x12xbf16>)  : i32 {
    %8 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg3 : tensor<1x1x12xbf16>) outs(%1 : tensor<1x1x12xf32>) -> tensor<1x1x12xf32>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%8, %2 : tensor<1x1x12xf32>, tensor<1x1x12xf32>) outs(%4 : tensor<1x1x12xf32>) -> tensor<1x1x12xf32>
    %10 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%9 : tensor<1x1x12xf32>) outs(%0 : tensor<1x1x12xbf16>) -> tensor<1x1x12xbf16>
    scf.yield %10 : tensor<1x1x12xbf16>
  }
  return %7 : tensor<1x1x12xbf16>
}

// -----
// CHECK-LABEL: func.func @triton_not_8d
func.func @triton_not_8d(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi64>, %arg3: memref<?xi64>) {
  %c-1_i64 = arith.constant -1 : i64
  %0 = tensor.empty() : tensor<2x2x2x2x2x2x2x2xi64>
  %1 = linalg.fill ins(%c-1_i64 : i64) outs(%0 : tensor<2x2x2x2x2x2x2x2xi64>) -> tensor<2x2x2x2x2x2x2x2xi64>
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [2, 2, 2, 2, 2, 2, 2, 2], strides: [128, 64, 32, 16, 8, 4, 2, 1] : memref<?xi64> to memref<2x2x2x2x2x2x2x2xi64, strided<[128, 64, 32, 16, 8, 4, 2, 1]>>
  %alloc = memref.alloc() : memref<2x2x2x2x2x2x2x2xi64>
  memref.copy %reinterpret_cast, %alloc : memref<2x2x2x2x2x2x2x2xi64, strided<[128, 64, 32, 16, 8, 4, 2, 1]>> to memref<2x2x2x2x2x2x2x2xi64>
  %2 = bufferization.to_tensor %alloc restrict writable : memref<2x2x2x2x2x2x2x2xi64>
  // CHECK:  %[[VAL_2:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
  // CHECK-NOT: hfusion.binary_fn<vxor>
  // CHECK-NOT: hfusion.binary_fn<vor>
  // CHECK-NOT: hfusion.binary_fn<vand>
  %3 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vxor>} ins(%2, %1 : tensor<2x2x2x2x2x2x2x2xi64>, tensor<2x2x2x2x2x2x2x2xi64>) outs(%0 : tensor<2x2x2x2x2x2x2x2xi64>) -> tensor<2x2x2x2x2x2x2x2xi64>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [2, 2, 2, 2, 2, 2, 2, 2], strides: [128, 64, 32, 16, 8, 4, 2, 1] : memref<?xi64> to memref<2x2x2x2x2x2x2x2xi64, strided<[128, 64, 32, 16, 8, 4, 2, 1]>>
  bufferization.materialize_in_destination %3 in writable %reinterpret_cast_0 : (tensor<2x2x2x2x2x2x2x2xi64>,  memref<2x2x2x2x2x2x2x2xi64, strided<[128, 64, 32, 16, 8, 4, 2, 1]>>) -> ()
  return
}