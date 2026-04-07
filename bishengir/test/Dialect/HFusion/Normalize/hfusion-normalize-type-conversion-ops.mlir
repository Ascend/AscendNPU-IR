// RUN: bishengir-opt --hfusion-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @test_NormalizeToTargetType_broadcast_i1
// CHECK: %[[CAST16:.*]] = hfusion.cast {{.*}} ins({{.*}} : tensor<8xi1>) outs({{.*}} : tensor<8xf16>) -> tensor<8xf16>
// CHECK: %[[BROADCAST16:.*]] = linalg.broadcast ins(%[[CAST16]] : tensor<8xf16>) outs({{.*}} : tensor<8x16xf16>) dimensions = [1]
// CHECK: %[[VEQ:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[BROADCAST16]], {{.*}} : tensor<8x16xf16>, f16) outs({{.*}} : tensor<8x16xi1>) -> tensor<8x16xi1>
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[VEQ]] : tensor<8x16xi1>) outs(%{{.*}} : tensor<8x16xi1>) -> tensor<8x16xi1>
func.func @test_NormalizeToTargetType_broadcast_i1(%arg0: tensor<8xi1>, %arg1: tensor<8x16xi1>) -> tensor<8x16xi1> {
  %0 = tensor.empty() : tensor<8x16xi1>
  %1 = linalg.broadcast
    ins(%arg0 : tensor<8xi1>)
    outs(%0 : tensor<8x16xi1>)
    dimensions = [1]
  return %1 : tensor<8x16xi1>
}

// -----

// CHECK-LABEL: @test_NormalizeToTargetType_broadcast_i8
// CHECK: %[[CAST16:.*]] = hfusion.cast {{.*}} ins({{.*}} : tensor<8xi8>) outs({{.*}} : tensor<8xf16>) -> tensor<8xf16>
// CHECK: %[[BROADCAST16:.*]] = linalg.broadcast ins(%[[CAST16]] : tensor<8xf16>) outs({{.*}} : tensor<8x16xf16>) dimensions = [1]
// CHECK: hfusion.cast {{.*}} ins(%[[BROADCAST16]] : tensor<8x16xf16>) outs({{.*}} : tensor<8x16xi8>) -> tensor<8x16xi8>
func.func @test_NormalizeToTargetType_broadcast_i8(%arg0: tensor<8xi8>, %arg1: tensor<8x16xi8>) -> tensor<8x16xi8> {
  %0 = tensor.empty() : tensor<8x16xi8>
  %1 = linalg.broadcast
    ins(%arg0 : tensor<8xi8>)
    outs(%0 : tensor<8x16xi8>)
    dimensions = [1]
  return %1 : tensor<8x16xi8>
}

// -----

// CHECK-LABEL: @test_NormalizeToTargetType_concat_i1
// CHECK: %[[IN0:.*]] = hfusion.cast {{.*}} ins(%arg0 : tensor<2048xi1>) outs({{.*}} : tensor<2048xf16>) -> tensor<2048xf16>
// CHECK: %[[IN1:.*]] = hfusion.cast {{.*}} ins(%arg1 : tensor<2048xi1>) outs({{.*}} : tensor<2048xf16>) -> tensor<2048xf16>
// CHECK: %[[CONCAT:.*]] = tensor.concat dim(0) %[[IN0]], %[[IN1]] : (tensor<2048xf16>, tensor<2048xf16>) -> tensor<4096xf16>
// CHECK: %[[VEQ:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[CONCAT]], {{.*}} : tensor<4096xf16>, f16) outs({{.*}} : tensor<4096xi1>) -> tensor<4096xi1>
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[VEQ]] : tensor<4096xi1>) outs(%{{.*}} : tensor<4096xi1>) -> tensor<4096xi1>
func.func @test_NormalizeToTargetType_concat_i1(%arg0: tensor<2048xi1>, %arg1: tensor<2048xi1>) -> tensor<4096xi1> {
  %0 = tensor.concat dim(0) %arg0, %arg1 : (tensor<2048xi1>, tensor<2048xi1>) -> tensor<4096xi1>
  return %0 : tensor<4096xi1>
}

// -----

// CHECK-LABEL: @test_NormalizeToTargetType_compare_i1
// CHECK: %[[LHS:.*]] = hfusion.cast {{.*}} ins(%arg0 : tensor<32xi1>) outs({{.*}} : tensor<32xf16>) -> tensor<32xf16>
// CHECK: %[[RHS:.*]] = hfusion.cast {{.*}} ins(%arg1 : tensor<32xi1>) outs({{.*}} : tensor<32xf16>) -> tensor<32xf16>
// CHECK: %[[VEQ:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[LHS]], %[[RHS]] : tensor<32xf16>, tensor<32xf16>) outs({{.*}} : tensor<32xi1>) -> tensor<32xi1>
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[VEQ]] : tensor<32xi1>) outs({{.*}} : tensor<32xi1>) -> tensor<32xi1>
func.func @test_NormalizeToTargetType_compare_i1(%arg0: tensor<32xi1>, %arg1: tensor<32xi1>) -> tensor<32xi1> {
  %0 = tensor.empty() : tensor<32xi1>
  %1 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
    ins(%arg0, %arg1 : tensor<32xi1>, tensor<32xi1>)
    outs(%0 : tensor<32xi1>) -> tensor<32xi1>
  return %1 : tensor<32xi1>
}

// -----

// CHECK-LABEL: @test_NormalizeToTargetType_transpose_i1
// CHECK: %[[SRC:.*]] = hfusion.cast {{.*}} ins(%{{.*}} : tensor<32x8xi1>) outs({{.*}} : tensor<32x8xf16>) -> tensor<32x8xf16>
// CHECK: %[[INIT:.*]] = hfusion.cast {{.*}} ins(%{{.*}} : tensor<8x32xi1>) outs({{.*}} : tensor<8x32xf16>) -> tensor<8x32xf16>
// CHECK: %[[TRANSPOSE:.*]] = linalg.transpose ins(%[[SRC]] : tensor<32x8xf16>) outs(%[[INIT]] : tensor<8x32xf16>) permutation = [1, 0]
// CHECK: %[[VEQ:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[TRANSPOSE]], {{.*}} : tensor<8x32xf16>, f16) outs({{.*}} : tensor<8x32xi1>) -> tensor<8x32xi1>
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[VEQ]] : tensor<8x32xi1>) outs(%{{.*}} : tensor<8x32xi1>) -> tensor<8x32xi1>
func.func @test_NormalizeToTargetType_transpose_i1() -> tensor<8x32xi1> {
  %src = tensor.empty() : tensor<32x8xi1>
  %dst = tensor.empty() : tensor<8x32xi1>
  %transposed = linalg.transpose ins(%src : tensor<32x8xi1>) outs(%dst : tensor<8x32xi1>) permutation = [1, 0]
  return %transposed : tensor<8x32xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_interleave_i1
// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<4x2x64xi1>
// CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<4x2x64xi1>
// CHECK: %[[VAL_3:.*]] = tensor.empty() : tensor<4x2x64xf16>
// CHECK: %[[VAL_4:.*]] = hfusion.cast {{.*}} ins(%[[VAL_0]] : tensor<4x2x64xi1>) outs(%[[VAL_3:.*]] : tensor<4x2x64xf16>) -> tensor<4x2x64xf16>
// CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<4x2x64xf16>
// CHECK: %[[VAL_6:.*]] = hfusion.cast {{.*}} ins(%[[VAL_1]] : tensor<4x2x64xi1>) outs(%[[VAL_5:.*]] : tensor<4x2x64xf16>) -> tensor<4x2x64xf16>
// CHECK: %[[VAL_7:.*]] = hfusion.interleave %[[VAL_4:.*]], %[[VAL_6:.*]] : tensor<4x2x64xf16>, tensor<4x2x64xf16> -> tensor<4x2x128xf16>
// CHECK: %[[VAL_8:.*]] = tensor.empty() : tensor<4x2x128xi1>
// CHECK: %[[VAL_10:.*]] = tensor.empty() : tensor<4x2x128xi1>
// CHECK: %[[VAL_9:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[VAL_7:.*]], %{{.*}} : tensor<4x2x128xf16>, f16) outs(%[[VAL_10:.*]] : tensor<4x2x128xi1>) -> tensor<4x2x128xi1>
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[VAL_9]] : tensor<4x2x128xi1>) outs(%[[VAL_8]] : tensor<4x2x128xi1>) -> tensor<4x2x128xi1>
func.func @test_NormalizeToTargetType_interleave_i1() -> tensor<4x2x128xi1> {
  %0 = tensor.empty() : tensor<4x2x64xi1>
  %1 = tensor.empty() : tensor<4x2x64xi1>
  %2 = hfusion.interleave %0, %1 : tensor<4x2x64xi1>, tensor<4x2x64xi1> -> tensor<4x2x128xi1>
  return %2 : tensor<4x2x128xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_interleave_i8
// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<4x2x64xi8>
// CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<4x2x64xi8>
// CHECK: %[[VAL_3:.*]] = tensor.empty() : tensor<4x2x64xf16>
// CHECK: %[[VAL_4:.*]] = hfusion.cast {{.*}} ins(%[[VAL_0]] : tensor<4x2x64xi8>) outs(%[[VAL_3:.*]] : tensor<4x2x64xf16>) -> tensor<4x2x64xf16>
// CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<4x2x64xf16>
// CHECK: %[[VAL_6:.*]] = hfusion.cast {{.*}} ins(%[[VAL_1]] : tensor<4x2x64xi8>) outs(%[[VAL_5:.*]] : tensor<4x2x64xf16>) -> tensor<4x2x64xf16>
// CHECK: %[[VAL_7:.*]] = hfusion.interleave %[[VAL_4:.*]], %[[VAL_6:.*]] : tensor<4x2x64xf16>, tensor<4x2x64xf16> -> tensor<4x2x128xf16>
// CHECK: %[[VAL_8:.*]] = tensor.empty() : tensor<4x2x128xi8>
// CHECK: %[[VAL_9:.*]] = hfusion.cast {{.*}} ins(%[[VAL_7:.*]] : tensor<4x2x128xf16>) outs(%[[VAL_8:.*]] : tensor<4x2x128xi8>) -> tensor<4x2x128xi8>
func.func @test_NormalizeToTargetType_interleave_i8() -> tensor<4x2x128xi8> {
  %0 = tensor.empty() : tensor<4x2x64xi8>
  %1 = tensor.empty() : tensor<4x2x64xi8>
  %2 = hfusion.interleave %0, %1 : tensor<4x2x64xi8>, tensor<4x2x64xi8> -> tensor<4x2x128xi8>
  return %2 : tensor<4x2x128xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeToTargetType_interleave_f16
// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<4x2x64xf16>
// CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<4x2x64xf16>
// CHECK: %[[VAL_2:.*]] = hfusion.interleave %[[VAL_0]], %[[VAL_1]] : tensor<4x2x64xf16>, tensor<4x2x64xf16> -> tensor<4x2x128xf16>
func.func @test_NormalizeToTargetType_interleave_f16() -> tensor<4x2x128xf16> {
  %0 = tensor.empty() : tensor<4x2x64xf16>
  %1 = tensor.empty() : tensor<4x2x64xf16>
  %2 = hfusion.interleave %0, %1 : tensor<4x2x64xf16>, tensor<4x2x64xf16> -> tensor<4x2x128xf16>
  return %2 : tensor<4x2x128xf16>
}

// -----

// CHECK-LABEL: @test_NormalizeToTargetType_normalize_interleave_i1
// CHECK:  %[[res_f16:.*]] = hfusion.interleave %[[arg0_f16:.*]], %[[arg1_f16:.*]] : tensor<4x2x32xf16>, tensor<4x2x32xf16> -> tensor<4x2x64xf16>
func.func @test_NormalizeToTargetType_normalize_interleave_i1(%arg0 : tensor<4x2x32xi1>, %arg1 : tensor<4x2x32xi1>) -> tensor<4x2x64xi1> {
  %0 = tensor.empty() : tensor<4x2x64xi1>
  %1 = hfusion.interleave %arg0, %arg1 : tensor<4x2x32xi1>, tensor<4x2x32xi1> -> tensor<4x2x64xi1>
  return %1 : tensor<4x2x64xi1>
}

// -----

// CHECK-LABEL: @test_NormalizeToTargetType_normalize_interleave_i8
// CHECK: %[[res_f16:.*]] = hfusion.interleave %[[arg0_f16:.*]], %[[arg1_f16:.*]] : tensor<4x2x32xf16>, tensor<4x2x32xf16> -> tensor<4x2x64xf16>
func.func @test_NormalizeToTargetType_normalize_interleave_i8(%arg0 : tensor<4x2x32xi8>, %arg1 : tensor<4x2x32xi8>) -> tensor<4x2x64xi8> {
  %0 = tensor.empty() : tensor<4x2x64xi8>
  %1 = hfusion.interleave %arg0, %arg1 : tensor<4x2x32xi8>, tensor<4x2x32xi8> -> tensor<4x2x64xi8>
  return %1 : tensor<4x2x64xi8>
}

// -----

// CHECK-LABEL: @test_NormalizeToTargetType_normalize_deinterleave_i8
// CHECK: %[[res_f16:.*]] = hfusion.deinterleave %[[cast_f16:.*]] channel<1> : tensor<4x2x128xf16> -> tensor<4x2x64xf16>
func.func @test_NormalizeToTargetType_normalize_deinterleave_i8() -> tensor<4x2x64xi8> {
  %0 = tensor.empty() : tensor<4x2x128xi8>
  %1 = hfusion.deinterleave %0 channel<1> : tensor<4x2x128xi8> -> tensor<4x2x64xi8>
  return %1 : tensor<4x2x64xi8>
}

// -----

// CHECK-LABEL: @test_NormalizeToTargetType_reduce_with_index_i1_return_value
// CHECK %[[reduce:.*]] = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <min>  ins(%[[arg0:.*]], %[[arg1:.*] : tensor<32x32x128xf16>
func.func @test_NormalizeToTargetType_reduce_with_index_i1_return_value(%arg0: tensor<32x32x128xi1>, %arg1: tensor<32x32x128xi32>) -> tensor<32x128xi1> {
  %0 = tensor.empty() : tensor<32x128xi1>
  %1 = tensor.empty() : tensor<32x128xi32>
  %reduced:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <min>
                ins(%arg0, %arg1 : tensor<32x32x128xi1>, tensor<32x32x128xi32>)
                outs(%0, %1 : tensor<32x128xi1>, tensor<32x128xi32>)
                dimensions = [1] -> tensor<32x128xi1>, tensor<32x128xi32>
  return %reduced#0 : tensor<32x128xi1>
}

// -----

// CHECK-LABEL: @test_NormalizeToTargetType_reduce_with_index_i1_return_index
// CHECK %[[reduce:.*]] = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <min>  ins(%[[arg0:.*]], %[[arg1:.*] : tensor<32x32x128xf16>
func.func @test_NormalizeToTargetType_reduce_with_index_i1_return_index(%arg0: tensor<32x32x128xi1>, %arg1: tensor<32x32x128xi32>) -> tensor<32x128xi32> {
  %0 = tensor.empty() : tensor<32x128xi1>
  %1 = tensor.empty() : tensor<32x128xi32>
  %reduced:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <min>
                ins(%arg0, %arg1 : tensor<32x32x128xi1>, tensor<32x32x128xi32>)
                outs(%0, %1 : tensor<32x128xi1>, tensor<32x128xi32>)
                dimensions = [1] -> tensor<32x128xi1>, tensor<32x128xi32>
  return %reduced#1 : tensor<32x128xi32>
}

// -----

// CHECK-LABEL: @test_NormalizeToTargetType_reduce_i1_addi
// CHECK: %[[reduce:.*]] = linalg.reduce { arith.ori }
func.func @test_NormalizeToTargetType_reduce_i1_addi(%arg0: tensor<16x32x64xi1>) -> tensor<16x64xi1> {
  %0 = tensor.empty() : tensor<16x64xi1>
  %reduce = linalg.reduce { arith.addi } ins(%arg0 : tensor<16x32x64xi1>) outs(%0 : tensor<16x64xi1>) dimensions = [1]
  return %reduce : tensor<16x64xi1>
}

// -----

// CHECK-LABEL: @test_NormalizeToTargetType_reduce_i1_muli
// CHECK: %[[reduce:.*]] = linalg.reduce { arith.andi }
func.func @test_NormalizeToTargetType_reduce_i1_muli(%arg0: tensor<16x32x64xi1>) -> tensor<16x64xi1> {
  %0 = tensor.empty() : tensor<16x64xi1>
  %reduce = linalg.reduce { arith.muli } ins(%arg0 : tensor<16x32x64xi1>) outs(%0 : tensor<16x64xi1>) dimensions = [1]
  return %reduce : tensor<16x64xi1>
}

// -----

// CHECK-LABEL: @test_NormalizeToTargetType_reduce_i1_maxi
// CHECK: %[[reduce:.*]] = linalg.reduce { arith.ori }
// CHECK: %[[reduce:.*]] = linalg.reduce { arith.ori }
func.func @test_NormalizeToTargetType_reduce_i1_maxi(%arg0: tensor<16x32x64xi1>) -> (tensor<16x64xi1>, tensor<16x64xi1>) {
  %0 = tensor.empty() : tensor<16x64xi1>
  %reduce = linalg.reduce { arith.maxui } ins(%arg0 : tensor<16x32x64xi1>) outs(%0 : tensor<16x64xi1>) dimensions = [1]
  %reduce_0 = linalg.reduce { arith.maxsi } ins(%arg0 : tensor<16x32x64xi1>) outs(%0 : tensor<16x64xi1>) dimensions = [1]
  return %reduce, %reduce_0 : tensor<16x64xi1>, tensor<16x64xi1>
}

// -----

// CHECK-LABEL: @test_NormalizeToTargetType_reduce_i1_mini
// CHECK: %[[reduce:.*]] = linalg.reduce { arith.andi }
// CHECK: %[[reduce:.*]] = linalg.reduce { arith.andi }
func.func @test_NormalizeToTargetType_reduce_i1_mini(%arg0: tensor<16x32x64xi1>) -> (tensor<16x64xi1>, tensor<16x64xi1>) {
  %0 = tensor.empty() : tensor<16x64xi1>
  %reduce = linalg.reduce { arith.minui } ins(%arg0 : tensor<16x32x64xi1>) outs(%0 : tensor<16x64xi1>) dimensions = [1]
  %reduce_0 = linalg.reduce { arith.minsi } ins(%arg0 : tensor<16x32x64xi1>) outs(%0 : tensor<16x64xi1>) dimensions = [1]
  return %reduce, %reduce_0 : tensor<16x64xi1>, tensor<16x64xi1>
}

// -----

// CHECK-LABEL: @test_NormalizeToTargetType_i8_elemwise_binary
// CHECK-SAME: %[[arg0:.*]]: tensor<16xi8>, %[[arg1:.*]]: tensor<16xi8>
// CHECK-DAG: %[[cast0:.*]] = hfusion.cast {{.*}} ins(%[[arg0]] : tensor<16xi8>)
// CHECK-DAG: %[[cast1:.*]] = hfusion.cast {{.*}} ins(%[[arg1]] : tensor<16xi8>)
// CHECK-DAG: %[[cast2:.*]] = hfusion.cast
// CHECK-NOT: tensor<16xf32>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[cast0]], %[[cast1]] : tensor<16xf16>, tensor<16xf16>) outs(%[[cast2]] : tensor<16xf16>)
// CHECK: %[[cast3:.*]] = hfusion.cast
// return %[[cast3]]
func.func @test_NormalizeToTargetType_i8_elemwise_binary(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi8> {
  %dst = tensor.empty() : tensor<16xi8>
  %res = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
        ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>)
        outs(%dst : tensor<16xi8>) -> tensor<16xi8>
  return %res : tensor<16xi8>
}

// -----

// CHECK-LABEL: @test_NormalizeToTargetType_i8_elemwise_binary
// CHECK-SAME: %[[arg0:.*]]: tensor<16xi8>, %[[arg1:.*]]: tensor<16xi8>
// CHECK: %[[arg2:.*]] = tensor.empty()
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[arg0]], %[[arg1]] : tensor<16xi8>, tensor<16xi8>) outs(%[[arg2]] : tensor<16xi8>)
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @test_NormalizeToTargetType_i8_elemwise_binary(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi8> {
    %dst = tensor.empty() : tensor<16xi8>
    %res = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
          ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>)
          outs(%dst : tensor<16xi8>) -> tensor<16xi8>
    return %res : tensor<16xi8>
  }
}

// -----

// CHECK-LABEL: @test_NormalizeToTargetType_triton_maximum
// CHECK-DAG: %[[cast0:.*]] = hfusion.cast {{.*}}
// CHECK-DAG: %[[cast1:.*]] = hfusion.cast {{.*}}
// CHECK-DAG: %[[cast2:.*]] = hfusion.cast {{.*}}
// CHECK: %[[result:.*]] = hfusion.elemwise_binary {{.*}}
// CHECK: %[[cast3:.*]] = hfusion.cast {{.*}}
func.func @test_NormalizeToTargetType_triton_maximum(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: index, %arg4: index) {
  %c64 = arith.constant 64 : index
  %c64_i32 = arith.constant 64 : i32
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  scf.for %arg10 = %c0_i32 to %c128_i32 step %c64_i32  : i32 {
    %alloc = memref.alloc() : memref<64xi8>
    %8 = bufferization.to_tensor %alloc restrict writable : memref<64xi8>
    %alloc_2 = memref.alloc() : memref<64xi8>
    %9 = bufferization.to_tensor %alloc_2 restrict writable : memref<64xi8>
    %10 = tensor.empty() : tensor<64xi8>
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%8, %9 : tensor<64xi8>, tensor<64xi8>) outs(%10 : tensor<64xi8>) -> tensor<64xi8>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%arg4], sizes: [64], strides: [1] : memref<?xi8> to memref<64xi8, strided<[1], offset: ?>>
    %extracted_slice = tensor.extract_slice %11[0] [%arg3] [1] : tensor<64xi8> to tensor<?xi8>
    %subview_6 = memref.subview %reinterpret_cast_5[0] [%arg3] [1] : memref<64xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<?xi8>, memref<?xi8, strided<[1], offset: ?>>) -> ()
  }
  return
}

// -----

// CHECK-LABEL: @test_NormalizeToTargetType_i8_elemwise_unary_relu
// CHECK: hfusion.cast {{.*}}
// CHECK: hfusion.cast {{.*}}
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>} {{.*}}
// CHECK: hfusion.cast {{.*}}
func.func @test_NormalizeToTargetType_i8_elemwise_unary_relu(%arg0: tensor<16xi8>) -> tensor<16xi8> {
  %dst = tensor.empty() : tensor<16xi8>
  %res = hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>}
          ins(%arg0 : tensor<16xi8>)
          outs(%dst : tensor<16xi8>) -> tensor<16xi8>
  return %res : tensor<16xi8>
}

// -----

// CHECK-LABEL: @test_NormalizeToTargetType_i8_reduce
// CHECK: arith.addf {{.*}} f32
// CHECK: linalg.reduce { arith.mulf } ins({{.*}} : tensor<32x16xf16>) outs({{.*}} : tensor<32xf16>) dimensions = [1]
func.func @test_NormalizeToTargetType_i8_reduce(%arg0: tensor<32x16x8xi8>, %arg1: tensor<32x16xi8>) -> (tensor<32xi8>, tensor<32xi8>) {
  %dst0 = tensor.empty() : tensor<32xi8>
  %reduced0 = linalg.reduce ins(%arg0 : tensor<32x16x8xi8>) outs(%dst0 : tensor<32xi8>) dimensions = [1, 2]
    (%in: i8, %init: i8) {
      %1 = arith.addi %in, %init : i8
      linalg.yield %1 : i8
    }
  %dst1 = tensor.empty() : tensor<32xi8>
  %reduced1 = linalg.reduce {arith.muli} ins(%arg1 : tensor<32x16xi8>) outs(%dst1: tensor<32xi8>) dimensions = [1]
  return %reduced0, %reduced1 : tensor<32xi8>, tensor<32xi8>
}

// -----

// CHECK-LABEL: @test_NormalizeToTargetType_i8_reduce_region
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: arith.maximumf
// CHECK: arith.minimumf
func.func @test_NormalizeToTargetType_i8_reduce_region(%arg0: tensor<32x16x8xi8>, %arg1: tensor<32x16xi8>) -> tensor<32xi8> {
  %dst0 = tensor.empty() : tensor<32xi8>
  %reduced0 = linalg.reduce ins(%arg0 : tensor<32x16x8xi8>) outs(%dst0 : tensor<32xi8>) dimensions = [1, 2]
    (%in: i8, %init: i8) {
      %1 = arith.addi %in, %init : i8
      %2 = arith.muli %1, %in : i8
      %3 = arith.subi %2, %init : i8
      %4 = arith.maxsi %2, %3 : i8
      %5 = arith.minsi %1, %4 : i8
      linalg.yield %5 : i8
    }
  return %reduced0 : tensor<32xi8>
}

// -----

// CHECK-LABEL: @test_NormalizeToTargetType_i8_reduce_with_index
// CHECK-DAG: %[[cast0:.*]] = hfusion.cast
// CHECK-DAG: %[[cast1:.*]] = hfusion.cast
// CHECK-DAG: %[[cast2:.*]] = hfusion.cast
// CHECK-DAG: %[[cast3:.*]] = hfusion.cast
// CHECK: linalg.reduce {{.*}}
// CHECK: arith.addf
// CHECK: arith.cmpf olt
// CHECK: arith.cmpf oge
// CHECK: hfusion.cast {{.*}}
// CHECK: hfusion.cast {{.*}}
func.func @test_NormalizeToTargetType_i8_reduce_with_index(%arg0: tensor<32x16x8xi8>) -> (tensor<32xi8>, tensor<32xi32>) {
  %0 = tensor.empty() : tensor<32x2xi8>
  %1 = tensor.empty() : tensor<32x2xi32>
  %2 = tensor.empty() : tensor<32xi8>
  %3 = tensor.empty() : tensor<32xi32>
  %reduced:2 = linalg.reduce ins(%0, %1 : tensor<32x2xi8>, tensor<32x2xi32>) outs(%2, %3 : tensor<32xi8>, tensor<32xi32>) dimensions = [1]
    (%in: i8, %in_0: i32, %init: i8, %init_1: i32) {
      %4 = arith.addi %in, %init : i8
      %5 = arith.cmpi slt, %in, %4 : i8
      %6 = arith.select %5, %in, %init : i8
      %7 = arith.cmpi sge, %6, %in : i8
      %8 = arith.select %7, %in, %init : i8
      %9 = arith.select %7, %in_0, %init_1 : i32
      linalg.yield %8, %9 : i8, i32
    }
  return %reduced#0, %reduced#1 : tensor<32xi8>, tensor<32xi32>
}

// -----

// CHECK-LABEL: @test_NormalizeToTargetType_reduce_max_with_index
// CHECK: hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <max> ins(%[[input0:.*]], %[[input1:.*]] : tensor<4x64xf16>, tensor<4x64xi32>) outs(%[[init0:.*]], %[[init1:.*]] : tensor<4xf16>, tensor<4xi32>) dimensions = [1] -> tensor<4xf16>, tensor<4xi32>
func.func @test_NormalizeToTargetType_reduce_max_with_index(%arg0: tensor<4x64xi8>, %arg1: tensor<4x64xi32>) -> (tensor<4xi8>, tensor<4xi32>) {
  %0 = tensor.empty() : tensor<4xi8>
  %1 = tensor.empty() : tensor<4xi32>
  %2:2 = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <max> ins(%arg0, %arg1 : tensor<4x64xi8>, tensor<4x64xi32>) outs(%0, %1 : tensor<4xi8>, tensor<4xi32>) dimensions = [1] -> tensor<4xi8>, tensor<4xi32>
  return %2#0, %2#1 : tensor<4xi8>, tensor<4xi32>
}

// -----

// CHECK-LABEL: @test_NormalizeToTargetType_gather_b8
// CHECK: hfusion.gather {operandSegmentSizes = array<i32: 2, 1>} ins(%[[INPUT0:.*]], %arg1 : tensor<4x64xf16>, tensor<4x32xi32>) outs(%[[OUTPUT0:.*]] : tensor<4x32xf16>) axis = 1 -> tensor<4x32xf16>
func.func @test_NormalizeToTargetType_gather_b8(%arg0: tensor<4x64xi8>, %arg1: tensor<4x32xi32>) -> tensor<4x32xi8> {
  %0 = tensor.empty() : tensor<4x32xi8>
  %1 = hfusion.gather ins(%arg0, %arg1 : tensor<4x64xi8>, tensor<4x32xi32>) outs(%0 : tensor<4x32xi8>) axis = 1 -> tensor<4x32xi8>
  return %1 : tensor<4x32xi8>
}

// -----

// CHECK-LABEL: @test_NormalizeToTargetType_reduce_i8_andi
// CHECK: linalg.reduce { arith.andi } ins({{.*}} : tensor<16x32x64xi8>)
func.func @test_NormalizeToTargetType_reduce_i8_andi(%arg0: tensor<16x32x64xi8>) -> tensor<16x32xi8> {
  %0 = tensor.empty() : tensor<16x32xi8>
  %reduce = linalg.reduce { arith.andi } ins(%arg0 : tensor<16x32x64xi8>) outs(%0 : tensor<16x32xi8>) dimensions = [2]
  return %reduce : tensor<16x32xi8>
}

// -----

// CHECK-LABEL: @test_NormalizeToTargetType_reduce_i8_ori
// CHECK: linalg.reduce { arith.ori } ins({{.*}} : tensor<16x32x64xi8>)
func.func @test_NormalizeToTargetType_reduce_i8_ori(%arg0: tensor<16x32x64xi8>) -> tensor<16x32xi8> {
  %0 = tensor.empty() : tensor<16x32xi8>
  %reduce = linalg.reduce { arith.ori } ins(%arg0 : tensor<16x32x64xi8>) outs(%0 : tensor<16x32xi8>) dimensions = [2]
  return %reduce : tensor<16x32xi8>
}

// -----

// CHECK-LABEL: @test_NormalizeToTargetType_reduce_i8_xori
// CHECK: linalg.reduce { arith.xori } ins({{.*}} : tensor<16x32x64xi8>)
func.func @test_NormalizeToTargetType_reduce_i8_xori(%arg0: tensor<16x32x64xi8>) -> tensor<16x32xi8> {
  %0 = tensor.empty() : tensor<16x32xi8>
  %reduce = linalg.reduce { arith.xori } ins(%arg0 : tensor<16x32x64xi8>) outs(%0 : tensor<16x32xi8>) dimensions = [2]
  return %reduce : tensor<16x32xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeF16ToF32Type_normalize_i8_to_f32
// CHECK: %[[cast0:.*]] = hfusion.cast {{.*}} ins({{.*}} : tensor<1xi8>) outs({{.*}} : tensor<1xf16>)
// CHECK: %[[cast1:.*]] = hfusion.cast {{.*}} ins(%[[cast0]] : tensor<1xf16>) outs({{.*}} : tensor<1xf32>)
// CHECK: %[[cast2:.*]] = hfusion.cast {{.*}} ins({{.*}} : tensor<1xi8>) outs({{.*}} : tensor<1xf16>)
// CHECK: %[[cast3:.*]] = hfusion.cast {{.*}} ins(%[[cast2]] : tensor<1xf16>) outs({{.*}} : tensor<1xf32>)
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[cast1]], %[[cast3]] : tensor<1xf32>, tensor<1xf32>)
func.func @test_NormalizeF16ToF32Type_normalize_i8_to_f32(%arg0: memref<?xi8>, %arg1: tensor<1xi8>, %arg2: tensor<1xi8>) {
  %0 = tensor.empty() : tensor<1xf32>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg1 : tensor<1xi8>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
  %2 = tensor.empty() : tensor<1xf32>
  %3 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg2 : tensor<1xi8>) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
  %4 = tensor.empty() : tensor<1xf32>
  %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%1, %3 : tensor<1xf32>, tensor<1xf32>) outs(%4 : tensor<1xf32>) -> tensor<1xf32>
  %6 = tensor.empty() : tensor<1xi8>
  %7 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%5 : tensor<1xf32>) outs(%6 : tensor<1xi8>) -> tensor<1xi8>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1], offset: ?>>
  bufferization.materialize_in_destination %7 in writable %reinterpret_cast : (tensor<1xi8>, memref<1xi8, strided<[1], offset: ?>>) -> ()
  return
}

// -----

// CHECK-LABEL: @test_NormalizeF16ToF32Type_linalg_log_f16
// CHECK: %[[CAST_IN:.*]] = hfusion.cast {{.*}} ins(%arg0 : tensor<16xf16>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[LOG:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[CAST_IN]] : tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: hfusion.cast {{.*}} ins(%[[LOG]] : tensor<16xf32>) outs({{.*}} : tensor<16xf16>) -> tensor<16xf16>
func.func @test_NormalizeF16ToF32Type_linalg_log_f16(%arg0: tensor<16xf16>) -> tensor<16xf16> {
  %0 = tensor.empty() : tensor<16xf16>
  %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>}
    ins(%arg0 : tensor<16xf16>)
    outs(%0 : tensor<16xf16>) -> tensor<16xf16>
  return %1 : tensor<16xf16>
}

// -----

// CHECK-LABEL: @test_NormalizeF16ToF32Type_hfusion_powf_f16
// CHECK: %[[CAST_LHS:.*]] = hfusion.cast {{.*}} ins(%arg0 : tensor<16xf16>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[CAST_RHS:.*]] = hfusion.cast {{.*}} ins(%arg1 : tensor<16xf16>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins({{.*}} : tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins({{.*}} : tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<16xf32>) outs({{.*}} : tensor<16xf16>) -> tensor<16xf16>
func.func @test_NormalizeF16ToF32Type_hfusion_powf_f16(%arg0: tensor<16xf16>, %arg1: tensor<16xf16>) -> tensor<16xf16> {
  %0 = tensor.empty() : tensor<16xf16>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>}
    ins(%arg0, %arg1 : tensor<16xf16>, tensor<16xf16>)
    outs(%0 : tensor<16xf16>) -> tensor<16xf16>
  return %1 : tensor<16xf16>
}

// -----

// CHECK-LABEL: @test_NormalizeF16ToF32Type_hfusion_rsqrt_f16
// CHECK: %[[CAST_IN:.*]] = hfusion.cast {{.*}} ins(%arg0 : tensor<16xf16>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[SQRT:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[CAST_IN]] : tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[REC:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%[[SQRT]] : tensor<16xf32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: hfusion.cast {{.*}} ins(%[[REC]] : tensor<16xf32>) outs({{.*}} : tensor<16xf16>) -> tensor<16xf16>
func.func @test_NormalizeF16ToF32Type_hfusion_rsqrt_f16(%arg0: tensor<16xf16>) -> tensor<16xf16> {
  %0 = tensor.empty() : tensor<16xf16>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>}
    ins(%arg0 : tensor<16xf16>)
    outs(%0 : tensor<16xf16>) -> tensor<16xf16>
  return %1 : tensor<16xf16>
}

// -----

// CHECK-LABEL: @test_NormalizeCumOpF16ToF32Type_cumsum_f16
// CHECK: %[[INPUT_F32:.*]] = hfusion.cast {{.*}} ins(%arg0 : tensor<4x64xf16>) outs({{.*}} : tensor<4x64xf32>) -> tensor<4x64xf32>
// CHECK: %[[CUMSUM:.*]] = hfusion.cumsum %[[INPUT_F32]] : tensor<4x64xf32> cum_dims = [0] reverse = true -> tensor<4x32xf32>
// CHECK: hfusion.cast {{.*}} ins(%[[CUMSUM]] : tensor<4x32xf32>) outs({{.*}} : tensor<4x32xf16>) -> tensor<4x32xf16>
func.func @test_NormalizeCumOpF16ToF32Type_cumsum_f16(%arg0: tensor<4x64xf16>, %reverse: tensor<?xi1>) -> tensor<4x32xf16> {
  %0 = tensor.empty() : tensor<4x32xf16>
  %1 = hfusion.cumsum %arg0 : tensor<4x64xf16> cum_dims = [0] reverse = true -> tensor<4x32xf16>
  return %1 : tensor<4x32xf16>
}

// -----

// CHECK-LABEL: @test_NormalizeCumOpF16ToF32Type_cumprod_f16
// CHECK: %[[INPUT_F32:.*]] = hfusion.cast {{.*}} ins(%arg0 : tensor<4x64xf16>) outs({{.*}} : tensor<4x64xf32>) -> tensor<4x64xf32>
// CHECK: %[[CUMPROD:.*]] = hfusion.cumprod %[[INPUT_F32]] : tensor<4x64xf32> cum_dims = [1] reverse = true -> tensor<4x32xf32>
// CHECK: hfusion.cast {{.*}} ins(%[[CUMPROD]] : tensor<4x32xf32>) outs({{.*}} : tensor<4x32xf16>) -> tensor<4x32xf16>
func.func @test_NormalizeCumOpF16ToF32Type_cumprod_f16(%arg0: tensor<4x64xf16>, %reverse: tensor<?xi1>) -> tensor<4x32xf16> {
  %0 = tensor.empty() : tensor<4x32xf16>
  %1 = hfusion.cumprod %arg0 : tensor<4x64xf16> cum_dims = [1] reverse = true -> tensor<4x32xf16>
  return %1 : tensor<4x32xf16>
}

// CHECK-LABEL: func.func @test_ReduceNormalize910_95_linalg_reduce_with_index
// CHECK-NOT: i16
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_ReduceNormalize910_95_linalg_reduce_with_index(%0: tensor<3x1x1x2x600xi8>, %1: tensor<3x1x1x2x600xi32>) -> tensor<3x1x1x2xi32> {
    %2 = tensor.empty() : tensor<3x1x1x2xi8>
    %3 = tensor.empty() : tensor<3x1x1x2xi32>
    %reduced:2 = linalg.reduce ins(%0, %1 : tensor<3x1x1x2x600xi8>, tensor<3x1x1x2x600xi32>) outs(%2, %3 : tensor<3x1x1x2xi8>, tensor<3x1x1x2xi32>) dimensions = [4]  {reduce_mode = "max_with_index", tie_break_left = "true"}
        (%in: i8, %in_6: i32, %init: i8, %init_7: i32) {
          %8 = arith.cmpi sgt, %in, %init : i8
          %9 = arith.cmpi eq, %in, %init : i8
          %10 = arith.cmpi slt, %in_6, %init_7 : i32
          %11 = arith.andi %9, %10 : i1
          %12 = arith.ori %8, %11 : i1
          %13 = arith.select %12, %in, %init : i8
          %14 = arith.select %12, %in_6, %init_7 : i32
          linalg.yield %13, %14 : i8, i32
        }
    return %reduced#1 : tensor<3x1x1x2xi32>
  }
}

// -----

// CHECK-LABEL: func.func @test_ReduceNormalize910_95_triton_sum_dim0_dim2
// CHECK: %reduced = linalg.reduce ins({{.*}} : tensor<5x8x7xi16>) outs({{.*}} : tensor<8xi16>) dimensions = [0, 2]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_ReduceNormalize910_95_triton_sum_dim0_dim2(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xi8>) {
    %c0_i8 = arith.constant 0 : i8
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [5, 8, 7], strides: [56, 7, 1] : memref<?xi8> to memref<5x8x7xi8, strided<[56, 7, 1]>>
    %alloc = memref.alloc() : memref<5x8x7xi8>
    memref.copy %reinterpret_cast, %alloc : memref<5x8x7xi8, strided<[56, 7, 1]>> to memref<5x8x7xi8>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<5x8x7xi8>
    %1 = tensor.empty() : tensor<8xi8>
    %2 = linalg.fill ins(%c0_i8 : i8) outs(%1 : tensor<8xi8>) -> tensor<8xi8>
    %reduced = linalg.reduce ins(%0 : tensor<5x8x7xi8>) outs(%2 : tensor<8xi8>) dimensions = [0, 2] 
      (%in: i8, %init: i8) {
        %3 = arith.addi %in, %init : i8
        linalg.yield %3 : i8
      }
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8], strides: [1] : memref<?xi8> to memref<8xi8, strided<[1]>>
    bufferization.materialize_in_destination %reduced in writable %reinterpret_cast_0 : (tensor<8xi8>, memref<8xi8, strided<[1]>>) -> ()
    return
  }
}

// -----

// CHECK-LABEL: @test_NormalizeCumOpI8ToTargetType_cumsum_b8
// CHECK: hfusion.cumsum %[[INPUT0:.*]] : tensor<4x64xf32> cum_dims = [0] reverse = true -> tensor<4x32xf32>
func.func @test_NormalizeCumOpI8ToTargetType_cumsum_b8(%arg0: tensor<4x64xi8>, %reverse: tensor<?xi1>) -> tensor<4x32xi8> {
  %0 = tensor.empty() : tensor<4x32xi8>
  %1 = hfusion.cumsum %arg0 : tensor<4x64xi8> cum_dims = [0] reverse = true -> tensor<4x32xi8>
  return %1 : tensor<4x32xi8>
}

// -----

// CHECK-LABEL: @test_NormalizeCumOpI8ToTargetType_cumprod_b8
// CHECK: hfusion.cumprod %[[INPUT0:.*]] : tensor<4x64xf32> cum_dims = [1] reverse = true -> tensor<4x32xf32>
func.func @test_NormalizeCumOpI8ToTargetType_cumprod_b8(%arg0: tensor<4x64xi8>, %reverse: tensor<?xi1>) -> tensor<4x32xi8> {
  %0 = tensor.empty() : tensor<4x32xi8>
  %1 = hfusion.cumprod %arg0 : tensor<4x64xi8> cum_dims = [1] reverse = true -> tensor<4x32xi8>
  return %1 : tensor<4x32xi8>
}

// -----

// CHECK-LABEL: @test_ReduceI1AddToSelectMaxCompare_reduce_i1_add_to_select_max_compare
// CHECK: hfusion.select
// CHECK: linalg.reduce
// CHECK: arith.maxsi
// CHECK: hfusion.compare
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_ReduceI1AddToSelectMaxCompare_reduce_i1_add_to_select_max_compare(%arg0: tensor<8xi1> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}) -> tensor<i1> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %cst_0 = arith.constant false
    %0 = bufferization.alloc_tensor() : tensor<i1>
    %1 = linalg.fill ins(%cst_0 : i1) outs(%0 : tensor<i1>) -> tensor<i1>
    %reduced = linalg.reduce ins(%arg0 : tensor<8xi1>) outs(%1 : tensor<i1>) dimensions = [0]
        (%in: i1, %init: i1) {
          %2 = arith.addi %in, %init : i1
          linalg.yield %2 : i1
        }
    return %reduced : tensor<i1>
  }
}
