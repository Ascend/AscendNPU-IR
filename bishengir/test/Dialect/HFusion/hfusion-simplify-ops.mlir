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
    %0 = hfusion.cast {arith.fastmath = #arith.fastmath<contract>} ins(%arg0 : tensor<bf16>) outs(%empty_f32 : tensor<f32>) -> tensor<f32>
    %1 = hfusion.cast {arith.fastmath = #arith.fastmath<contract>} ins(%0 : tensor<f32>) outs(%empty_bf16 : tensor<bf16>) -> tensor<bf16>
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
    %0 = hfusion.cast {arith.fastmath = #arith.fastmath<contract>} ins(%arg0 : tensor<bf16>) outs(%empty_f32 : tensor<f32>) -> tensor<f32>
    %1 = hfusion.cast {arith.fastmath = #arith.fastmath<contract>} ins(%0 : tensor<f32>) outs(%empty_i1 : tensor<i1>) -> tensor<i1>
    %2 = hfusion.cast {arith.fastmath = #arith.fastmath<contract>} ins(%1 : tensor<i1>) outs(%empty_f32 : tensor<f32>) -> tensor<f32>
    %3 = hfusion.cast {arith.fastmath = #arith.fastmath<contract>} ins(%2 : tensor<f32>) outs(%empty_bf16 : tensor<bf16>) -> tensor<bf16>
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
    %0 = hfusion.cast {arith.fastmath = #arith.fastmath<contract>} ins(%arg0 : tensor<bf16>) outs(%empty_f32 : tensor<f32>) -> tensor<f32>
    %1 = hfusion.cast ins(%0 : tensor<f32>) outs(%empty_i1 : tensor<i1>) -> tensor<i1>
    %2 = hfusion.cast ins(%1 : tensor<i1>) outs(%empty_bf16 : tensor<bf16>) -> tensor<bf16>
    %3 = hfusion.cast ins(%1 : tensor<i1>) outs(%empty_f32 : tensor<f32>) -> tensor<f32>
    %4 = hfusion.cast {arith.fastmath = #arith.fastmath<contract>} ins(%3 : tensor<f32>) outs(%empty_bf16 : tensor<bf16>) -> tensor<bf16>
    %add_res = linalg.elemwise_binary { add, fun = #linalg.binary_fn<add> } ins(%2, %4 : tensor<bf16>, tensor<bf16>) outs(%empty_bf16 : tensor<bf16>) -> tensor<bf16>
    return %add_res : tensor<bf16>
}

// -----

// CHECK-LABEL: @bifurcationMultiOuts
// CHECK-SAME: (%[[arg0:.*]]: tensor<bf16>)
// CHECK: %[[cast0:.*]] = hfusion.cast {arith.fastmath = #arith.fastmath<contract>} ins(%[[arg0]] : tensor<bf16>)
// CHECK: return %[[cast0]], %[[arg0]] : tensor<f32>, tensor<bf16>
func.func @bifurcationMultiOuts(%arg0: tensor<bf16>) -> (tensor<f32>, tensor<bf16>) {
    %empty_f32 = tensor.empty() : tensor<f32>
    %empty_bf16 = tensor.empty() : tensor<bf16>
    %0 = hfusion.cast {arith.fastmath = #arith.fastmath<contract>} ins(%arg0 : tensor<bf16>) outs(%empty_f32 : tensor<f32>) -> tensor<f32>
    %1 = hfusion.cast {arith.fastmath = #arith.fastmath<contract>} ins(%0 : tensor<f32>) outs(%empty_bf16 : tensor<bf16>) -> tensor<bf16>
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
    %0 = hfusion.cast {arith.fastmath = #arith.fastmath<contract>} ins(%arg0 : tensor<bf16>) outs(%empty_f32 : tensor<f32>) -> tensor<f32>
    %1 = hfusion.cast ins(%0 : tensor<f32>) outs(%empty_i1 : tensor<i1>) -> tensor<i1>
    %2 = hfusion.cast ins(%1 : tensor<i1>) outs(%empty_bf16 : tensor<bf16>) -> tensor<bf16>
    %3 = hfusion.cast {arith.fastmath = #arith.fastmath<contract>} ins(%0 : tensor<f32>) outs(%empty_bf16 : tensor<bf16>) -> tensor<bf16>
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
// CHECK: %[[cast0:.*]] = hfusion.cast {arith.fastmath = #arith.fastmath<contract>} ins(%[[arg0]]
// CHECK-NOT: hfusion.cast
// CHECK: %[[result:.*]] = linalg.elemwise_binary
// CHECK: ins(%[[cast0]], %[[cast0]]
// CHECK: return %[[result]] : tensor<f32>

func.func @liveBifurcation(%arg0: tensor<bf16>) -> tensor<f32> {
    %empty_bf16 = tensor.empty() : tensor<bf16>
    %empty_f32 = tensor.empty() : tensor<f32>
    %0 = hfusion.cast {arith.fastmath = #arith.fastmath<contract>} ins(%arg0 : tensor<bf16>) outs(%empty_f32 : tensor<f32>) -> tensor<f32>
    %1 = hfusion.cast {arith.fastmath = #arith.fastmath<contract>} ins(%0 : tensor<f32>) outs(%empty_bf16 : tensor<bf16>) -> tensor<bf16>
    %2 = hfusion.cast {arith.fastmath = #arith.fastmath<contract>} ins(%1 : tensor<bf16>) outs(%empty_f32 : tensor<f32>) -> tensor<f32>
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

// CHECK-LABEL: @reserve_cast
// CHECK: hfusion.cast
// CHECK: %[[VAL:.*]] = hfusion.cast
// CHECK-SAME: -> tensor<f32>
// CHECK: return %[[VAL]] : tensor<f32>

func.func @reserve_cast(%arg0: tensor<f32>) -> tensor<f32> {
    %empty_f32 = tensor.empty() : tensor<f32>
    %empty_bf16 = tensor.empty() : tensor<bf16>
    %0 = hfusion.cast ins(%arg0 : tensor<f32>) outs(%empty_bf16 : tensor<bf16>) -> tensor<bf16>
    %1 = hfusion.cast ins(%0 : tensor<bf16>) outs(%empty_f32 : tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: @still_eliminate_cast
// CHECK-SAME: (%[[arg0:.*]]: tensor<f32>)
// CHECK-NOT: hfusion.cast
// CHECK: return %[[arg0]] : tensor<f32>

func.func @still_eliminate_cast(%arg0: tensor<f32>) -> tensor<f32> {
    %empty_f32 = tensor.empty() : tensor<f32>
    %empty_bf16 = tensor.empty() : tensor<bf16>
    %0 = hfusion.cast {arith.fastmath = #arith.fastmath<contract>} ins(%arg0 : tensor<f32>) outs(%empty_bf16 : tensor<bf16>) -> tensor<bf16>
    %1 = hfusion.cast ins(%0 : tensor<bf16>) outs(%empty_f32 : tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: func @castInLoopWithInnerUse
// CHECK: scf.for
// CHECK: hfusion.cast
// CHECK: tensor.extract_slice
// CHECK: scf.yield
func.func @castInLoopWithInnerUse(%arg0: tensor<16x32xf16>, %arg1: tensor<16x32xf16>) -> (tensor<16x32xf16>, tensor<1x32xf16>) {
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %empty_f32 = tensor.empty() : tensor<16x32xf32>
  %empty_f16 = tensor.empty() : tensor<16x32xf16>
  %init_slice = tensor.empty() : tensor<1x32xf16>
  %0 = hfusion.cast ins(%arg0 : tensor<16x32xf16>) outs(%empty_f32 : tensor<16x32xf32>) -> tensor<16x32xf32>
  %1:2 = scf.for %i = %c1_i32 to %c4_i32 step %c1_i32 iter_args(%arg = %arg1, %arg_slice = %init_slice) -> (tensor<16x32xf16>, tensor<1x32xf16>) : i32 {
    %2 = hfusion.cast ins(%arg : tensor<16x32xf16>) outs(%empty_f32 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%2, %0 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%empty_f32 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %4 = hfusion.cast ins(%3 : tensor<16x32xf32>) outs(%empty_f16 : tensor<16x32xf16>) -> tensor<16x32xf16>
    %extracted_slice = tensor.extract_slice %4[0, 0] [1, 32] [1, 1] : tensor<16x32xf16> to tensor<1x32xf16>
    scf.yield %4, %extracted_slice : tensor<16x32xf16>, tensor<1x32xf16>
  }
  return %1#0, %1#1 : tensor<16x32xf16>, tensor<1x32xf16>
}

// -----
// CHECK-LABEL: func.func @collapse_chain
// CHECK:         %[[C:.*]] = arith.constant dense<4> : tensor<32xi32>
// CHECK:         %[[M:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %[[C]]
// CHECK:         linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[M]], %arg1
// CHECK-NOT:     binary_fn<add>
func.func @collapse_chain(%inv: tensor<32xi32>, %b: tensor<32xi32>, %e: tensor<32xi32>) -> tensor<32xi32> {
  %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%inv, %b : tensor<32xi32>, tensor<32xi32>) outs(%e : tensor<32xi32>) -> tensor<32xi32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%inv, %0 : tensor<32xi32>, tensor<32xi32>) outs(%e : tensor<32xi32>) -> tensor<32xi32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%inv, %1 : tensor<32xi32>, tensor<32xi32>) outs(%e : tensor<32xi32>) -> tensor<32xi32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%inv, %2 : tensor<32xi32>, tensor<32xi32>) outs(%e : tensor<32xi32>) -> tensor<32xi32>
  return %3 : tensor<32xi32>
}

// -----
// CHECK-LABEL: func.func @collapse_multi_operand
// CHECK:         arith.constant dense<2>
// CHECK-COUNT-2: binary_fn<mul>
// CHECK:         binary_fn<add>
// CHECK-NOT:     binary_fn<add>
// CHECK:         return
func.func @collapse_multi_operand(%a: tensor<8xi32>, %b: tensor<8xi32>, %e: tensor<8xi32>) -> tensor<8xi32> {
  %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%a, %b : tensor<8xi32>, tensor<8xi32>) outs(%e : tensor<8xi32>) -> tensor<8xi32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%0, %a : tensor<8xi32>, tensor<8xi32>) outs(%e : tensor<8xi32>) -> tensor<8xi32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %b : tensor<8xi32>, tensor<8xi32>) outs(%e : tensor<8xi32>) -> tensor<8xi32>
  return %2 : tensor<8xi32>
}

// -----
// CHECK-LABEL: func.func @collapse_self_add
// CHECK:         %[[C:.*]] = arith.constant dense<2> : tensor<4xi32>
// CHECK:         %[[M:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %[[C]]
// CHECK:         return %[[M]]
// CHECK-NOT:     binary_fn<add>
func.func @collapse_self_add(%a: tensor<4xi32>, %e: tensor<4xi32>) -> tensor<4xi32> {
  %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%a, %a : tensor<4xi32>, tensor<4xi32>) outs(%e : tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----
// CHECK-LABEL: func.func @collapse_multiuse
// CHECK:         %[[C:.*]] = arith.constant dense<2> : tensor<4xi32>
// CHECK:         %[[K:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1
// CHECK:         %[[M:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %[[C]]
// CHECK:         %[[R:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[M]], %[[K]]
// CHECK:         return %[[R]], %[[K]]
func.func @collapse_multiuse(%inv: tensor<4xi32>, %b: tensor<4xi32>, %e: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>) {
  %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%inv, %b : tensor<4xi32>, tensor<4xi32>) outs(%e : tensor<4xi32>) -> tensor<4xi32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%inv, %0 : tensor<4xi32>, tensor<4xi32>) outs(%e : tensor<4xi32>) -> tensor<4xi32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%inv, %1 : tensor<4xi32>, tensor<4xi32>) outs(%e : tensor<4xi32>) -> tensor<4xi32>
  return %2, %0 : tensor<4xi32>, tensor<4xi32>
}

// -----
// CHECK-LABEL: func.func @no_collapse_float
// CHECK-COUNT-2: binary_fn<add>
// CHECK-NOT:     binary_fn<mul>
func.func @no_collapse_float(%inv: tensor<4xf32>, %b: tensor<4xf32>, %e: tensor<4xf32>) -> tensor<4xf32> {
  %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%inv, %b : tensor<4xf32>, tensor<4xf32>) outs(%e : tensor<4xf32>) -> tensor<4xf32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%inv, %0 : tensor<4xf32>, tensor<4xf32>) outs(%e : tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @no_collapse_distinct
// CHECK-COUNT-2: binary_fn<add>
// CHECK-NOT:     binary_fn<mul>
func.func @no_collapse_distinct(%a: tensor<4xi32>, %b: tensor<4xi32>, %c: tensor<4xi32>, %e: tensor<4xi32>) -> tensor<4xi32> {
  %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%a, %b : tensor<4xi32>, tensor<4xi32>) outs(%e : tensor<4xi32>) -> tensor<4xi32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%0, %c : tensor<4xi32>, tensor<4xi32>) outs(%e : tensor<4xi32>) -> tensor<4xi32>
  return %1 : tensor<4xi32>
}

// -----
// Test: hoist negation from matmul result to matmul LHS input
// CHECK-LABEL: func.func @hoist_negation_matmul
// CHECK-SAME: (%[[A:.*]]: tensor<4x8xf32>, %[[B:.*]]: tensor<8x4xf32>)
// CHECK: %[[NEG_ONE:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[EMPTY_A:.*]] = tensor.empty() : tensor<4x8xf32>
// CHECK: %[[NEG_A:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[A]], %[[NEG_ONE]] : tensor<4x8xf32>, f32) outs(%[[EMPTY_A]] : tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK: %[[RES:.*]] = linalg.matmul ins(%[[NEG_A]], %[[B]] : tensor<4x8xf32>, tensor<8x4xf32>)
// CHECK-NOT: binary_fn<mul>
// CHECK: return %[[RES]] : tensor<4x4xf32>
func.func @hoist_negation_matmul(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>) -> tensor<4x4xf32> {
  %init = tensor.empty() : tensor<4x4xf32>
  %cst_zero = arith.constant 0.000000e+00 : f32
  %fill = linalg.fill ins(%cst_zero : f32) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %matmul = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x4xf32>) outs(%fill : tensor<4x4xf32>) -> tensor<4x4xf32>
  %cst_neg1 = arith.constant -1.000000e+00 : f32
  %out = tensor.empty() : tensor<4x4xf32>
  %neg = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%matmul, %cst_neg1 : tensor<4x4xf32>, f32) outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %neg : tensor<4x4xf32>
}

// -----
// Test: hoist negation from batch_matmul result
// CHECK-LABEL: func.func @hoist_negation_batch_matmul
// CHECK: %[[NEG_ONE:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[NEG_A:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: %[[RES:.*]] = linalg.batch_matmul ins(%[[NEG_A]],
// CHECK-NOT: binary_fn<mul>
// CHECK: return %[[RES]]
func.func @hoist_negation_batch_matmul(%a: tensor<2x4x8xf32>, %b: tensor<2x8x4xf32>) -> tensor<2x4x4xf32> {
  %init = tensor.empty() : tensor<2x4x4xf32>
  %cst_zero = arith.constant 0.000000e+00 : f32
  %fill = linalg.fill ins(%cst_zero : f32) outs(%init : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  %matmul = linalg.batch_matmul ins(%a, %b : tensor<2x4x8xf32>, tensor<2x8x4xf32>) outs(%fill : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  %cst_neg1 = arith.constant -1.000000e+00 : f32
  %out = tensor.empty() : tensor<2x4x4xf32>
  %neg = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%matmul, %cst_neg1 : tensor<2x4x4xf32>, f32) outs(%out : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  return %neg : tensor<2x4x4xf32>
}

// -----
// Test: hoist negation from two consecutive matmuls (second matmul uses first matmul result)
// CHECK-LABEL: func.func @hoist_negation_two_matmuls
// CHECK-SAME: (%[[A:.*]]: tensor<4x8xf32>, %[[B:.*]]: tensor<8x4xf32>, %[[C:.*]]: tensor<4x4xf32>)
// CHECK-DAG: %[[NEG_ONE:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[EMPTY_A:.*]] = tensor.empty() : tensor<4x8xf32>
// CHECK: %[[NEG_A:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[A]], %[[NEG_ONE]] : tensor<4x8xf32>, f32) outs(%[[EMPTY_A]] : tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK: %[[EMPTY_NEG_A:.*]] = tensor.empty() : tensor<4x8xf32>
// CHECK: %[[NEG_NEG_A:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[NEG_A]], %[[NEG_ONE]] : tensor<4x8xf32>, f32) outs(%[[EMPTY_NEG_A]] : tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK: %[[MATMUL1:.*]] = linalg.matmul ins(%[[NEG_NEG_A]], %[[B]] : tensor<4x8xf32>, tensor<8x4xf32>)
// CHECK: %[[MATMUL2:.*]] = linalg.matmul ins(%[[MATMUL1]], %[[C]] : tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-NOT: binary_fn<mul>
// CHECK: return %[[MATMUL2]]
func.func @hoist_negation_two_matmuls(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>, %c: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %init1 = tensor.empty() : tensor<4x4xf32>
  %cst_zero = arith.constant 0.000000e+00 : f32
  %fill1 = linalg.fill ins(%cst_zero : f32) outs(%init1 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %matmul1 = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x4xf32>) outs(%fill1 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %cst_neg1 = arith.constant -1.000000e+00 : f32
  %out1 = tensor.empty() : tensor<4x4xf32>
  %neg1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%matmul1, %cst_neg1 : tensor<4x4xf32>, f32) outs(%out1 : tensor<4x4xf32>) -> tensor<4x4xf32>
  
  %init2 = tensor.empty() : tensor<4x4xf32>
  %fill2 = linalg.fill ins(%cst_zero : f32) outs(%init2 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %matmul2 = linalg.matmul ins(%neg1, %c : tensor<4x4xf32>, tensor<4x4xf32>) outs(%fill2 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %out2 = tensor.empty() : tensor<4x4xf32>
  %neg2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%matmul2, %cst_neg1 : tensor<4x4xf32>, f32) outs(%out2 : tensor<4x4xf32>) -> tensor<4x4xf32>
  
  return %neg2 : tensor<4x4xf32>
}

// -----
// Test: negation NOT hoisted when matmul result has multiple uses
// CHECK-LABEL: func.func @no_hoist_multi_use
// CHECK: linalg.matmul
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
func.func @no_hoist_multi_use(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>) -> tensor<4x4xf32> {
  %init = tensor.empty() : tensor<4x4xf32>
  %cst_zero = arith.constant 0.000000e+00 : f32
  %fill = linalg.fill ins(%cst_zero : f32) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %matmul = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x4xf32>) outs(%fill : tensor<4x4xf32>) -> tensor<4x4xf32>
  %cst_neg1 = arith.constant -1.000000e+00 : f32
  %out = tensor.empty() : tensor<4x4xf32>
  %neg = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%matmul, %cst_neg1 : tensor<4x4xf32>, f32) outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  %out2 = tensor.empty() : tensor<4x4xf32>
  %add = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%neg, %matmul : tensor<4x4xf32>, tensor<4x4xf32>) outs(%out2 : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %add : tensor<4x4xf32>
}

// -----
// Test: hoist negation using sub(0, a) from matmul result to matmul LHS input
// CHECK-LABEL: func.func @hoist_sub_zero_negation_matmul
// CHECK-SAME: (%[[A:.*]]: tensor<4x8xf32>, %[[B:.*]]: tensor<8x4xf32>)
// CHECK: %[[NEG_ONE:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[EMPTY_A:.*]] = tensor.empty() : tensor<4x8xf32>
// CHECK: %[[NEG_A:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[A]], %[[NEG_ONE]] : tensor<4x8xf32>, f32) outs(%[[EMPTY_A]] : tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK: %[[RES:.*]] = linalg.matmul ins(%[[NEG_A]], %[[B]] : tensor<4x8xf32>, tensor<8x4xf32>)
// CHECK-NOT: binary_fn<sub>
// CHECK: return %[[RES]] : tensor<4x4xf32>
func.func @hoist_sub_zero_negation_matmul(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>) -> tensor<4x4xf32> {
  %init = tensor.empty() : tensor<4x4xf32>
  %cst_zero = arith.constant 0.000000e+00 : f32
  %fill = linalg.fill ins(%cst_zero : f32) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %matmul = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x4xf32>) outs(%fill : tensor<4x4xf32>) -> tensor<4x4xf32>
  %out = tensor.empty() : tensor<4x4xf32>
  %neg = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%cst_zero, %matmul : f32, tensor<4x4xf32>) outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %neg : tensor<4x4xf32>
}

// -----
// Test: hoist negation using sub(0, a) with dense constant zero tensor
// CHECK-LABEL: func.func @hoist_sub_dense_zero_negation_matmul
// CHECK-SAME: (%[[A:.*]]: tensor<4x8xf32>, %[[B:.*]]: tensor<8x4xf32>)
// CHECK: %[[NEG_ONE:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[EMPTY_A:.*]] = tensor.empty() : tensor<4x8xf32>
// CHECK: %[[NEG_A:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[A]], %[[NEG_ONE]] : tensor<4x8xf32>, f32) outs(%[[EMPTY_A]] : tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK: %[[RES:.*]] = linalg.matmul ins(%[[NEG_A]], %[[B]] : tensor<4x8xf32>, tensor<8x4xf32>)
// CHECK-NOT: binary_fn<sub>
// CHECK: return %[[RES]] : tensor<4x4xf32>
func.func @hoist_sub_dense_zero_negation_matmul(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>) -> tensor<4x4xf32> {
  %init = tensor.empty() : tensor<4x4xf32>
  %cst_zero = arith.constant 0.000000e+00 : f32
  %cst_zero_tensor = arith.constant dense<0.000000e+00> : tensor<4x4xf32>
  %fill = linalg.fill ins(%cst_zero : f32) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %matmul = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x4xf32>) outs(%fill : tensor<4x4xf32>) -> tensor<4x4xf32>
  %out = tensor.empty() : tensor<4x4xf32>
  %neg = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%cst_zero_tensor, %matmul : tensor<4x4xf32>, tensor<4x4xf32>) outs(%out : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %neg : tensor<4x4xf32>
}

// -----
// Test: sub(0, matmul) NOT hoisted when matmul result has multiple uses
// CHECK-LABEL: func.func @no_hoist_sub_zero_multi_use_matmul
// CHECK: linalg.matmul
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
func.func @no_hoist_sub_zero_multi_use_matmul(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>) -> tensor<4x4xf32> {
  %init = tensor.empty() : tensor<4x4xf32>
  %cst_zero = arith.constant 0.000000e+00 : f32
  %fill = linalg.fill ins(%cst_zero : f32) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %matmul = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x4xf32>) outs(%fill : tensor<4x4xf32>) -> tensor<4x4xf32>
  
  %out1 = tensor.empty() : tensor<4x4xf32>
  %neg = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%cst_zero, %matmul : f32, tensor<4x4xf32>) outs(%out1 : tensor<4x4xf32>) -> tensor<4x4xf32>
  
  %out2 = tensor.empty() : tensor<4x4xf32>
  %add = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%neg, %matmul : tensor<4x4xf32>, tensor<4x4xf32>) outs(%out2 : tensor<4x4xf32>) -> tensor<4x4xf32>
  
  return %add : tensor<4x4xf32>
}

// -----
// Test: hoist negation using sub(0, a) through two consecutive matmuls
// CHECK-LABEL: func.func @hoist_sub_zero_negation_two_matmuls
// CHECK-SAME: (%[[A:.*]]: tensor<4x8xf32>, %[[B:.*]]: tensor<8x4xf32>, %[[C:.*]]: tensor<4x4xf32>)
// CHECK-DAG: %[[NEG_ONE:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[EMPTY_A:.*]] = tensor.empty() : tensor<4x8xf32>
// CHECK: %[[NEG_A:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[A]], %[[NEG_ONE]] : tensor<4x8xf32>, f32) outs(%[[EMPTY_A]] : tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK: %[[EMPTY_NEG_A:.*]] = tensor.empty() : tensor<4x8xf32>
// CHECK: %[[NEG_NEG_A:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[NEG_A]], %[[NEG_ONE]] : tensor<4x8xf32>, f32) outs(%[[EMPTY_NEG_A]] : tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK: %[[MATMUL1:.*]] = linalg.matmul ins(%[[NEG_NEG_A]], %[[B]] : tensor<4x8xf32>, tensor<8x4xf32>)
// CHECK: %[[MATMUL2:.*]] = linalg.matmul ins(%[[MATMUL1]], %[[C]] : tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-NOT: binary_fn<sub>
// CHECK: return %[[MATMUL2]]
func.func @hoist_sub_zero_negation_two_matmuls(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>, %c: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %init1 = tensor.empty() : tensor<4x4xf32>
  %cst_zero = arith.constant 0.000000e+00 : f32
  %fill1 = linalg.fill ins(%cst_zero : f32) outs(%init1 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %matmul1 = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x4xf32>) outs(%fill1 : tensor<4x4xf32>) -> tensor<4x4xf32>

  %out1 = tensor.empty() : tensor<4x4xf32>
  %neg1 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%cst_zero, %matmul1 : f32, tensor<4x4xf32>) outs(%out1 : tensor<4x4xf32>) -> tensor<4x4xf32>

  %init2 = tensor.empty() : tensor<4x4xf32>
  %fill2 = linalg.fill ins(%cst_zero : f32) outs(%init2 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %matmul2 = linalg.matmul ins(%neg1, %c : tensor<4x4xf32>, tensor<4x4xf32>) outs(%fill2 : tensor<4x4xf32>) -> tensor<4x4xf32>

  %out2 = tensor.empty() : tensor<4x4xf32>
  %neg2 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%cst_zero, %matmul2 : f32, tensor<4x4xf32>) outs(%out2 : tensor<4x4xf32>) -> tensor<4x4xf32>

  return %neg2 : tensor<4x4xf32>
}