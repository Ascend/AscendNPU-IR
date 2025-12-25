// RUN: bishengir-opt --hfusion-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @test_normalize_i8_elemwise_binary
// CHECK-SAME: %[[arg0:.*]]: tensor<16xi8>, %[[arg1:.*]]: tensor<16xi8>
// CHECK-DAG: %[[cast0:.*]] = hfusion.cast {{.*}} ins(%[[arg0]] : tensor<16xi8>)
// CHECK-DAG: %[[cast1:.*]] = hfusion.cast
// CHECK-DAG: %[[cast2:.*]] = hfusion.cast {{.*}} ins(%[[arg1]] : tensor<16xi8>)
// CHECK-DAG: %[[cast3:.*]] = hfusion.cast
// CHECK-DAG: %[[cast4:.*]] = hfusion.cast
// CHECK-DAG: %[[cast5:.*]] = hfusion.cast
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[cast1]], %[[cast3]] : tensor<16xf32>, tensor<16xf32>) outs(%[[cast5]] : tensor<16xf32>)
// CHECK: %[[cast6:.*]] = hfusion.cast
// return %[[cast6]]
func.func @test_normalize_i8_elemwise_binary(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi8> {
  %dst = tensor.empty() : tensor<16xi8>
  %res = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
        ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>)
        outs(%dst : tensor<16xi8>) -> tensor<16xi8>
  return %res : tensor<16xi8>
}

// -----

// CHECK-LABEL: @test_normalize_triton_maximum
// CHECK-DAG: %[[cast0:.*]] = hfusion.cast
// CHECK-DAG: %[[cast1:.*]] = hfusion.cast
// CHECK-DAG: %[[cast2:.*]] = hfusion.cast
// CHECK: %[[result:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%[[cast0]], %[[cast1]] : tensor<64xf16>, tensor<64xf16>) outs(%[[cast2]] : tensor<64xf16>)
// CHECK: %[[result1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: %[[cast3:.*]] = hfusion.cast
// CHECK: %[[result2:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: %[[cast4:.*]] = hfusion.cast
// CHECK: %[[result3:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: %[[result4:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
// CHECK: %[[result5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: %[[cast5:.*]] = hfusion.cast
// CHECK: ins(%24 : tensor<64xf16>) outs(%25 : tensor<64xi8>) -> tensor<64xi8>
func.func @test_normalize_triton_maximum(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: index, %arg4: index) {
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
// CHECK-LABEL: @test_normalize_triton_where_hfusion_compare_select
// CHECK: %[[cast0:.*]] = hfusion.cast
// CHECK: %[[fill0:.*]] = linalg.fill
// CHECK: %[[compare_tmp:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[cast0]], %[[fill0]] : tensor<8x8x4xf16>, tensor<8x8x4xf16>)
// CHECK: %[[compare:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[compare_tmp]] : tensor<8x8x4xi1>)
// CHECK: %[[select:.*]] =  hfusion.select ins(%[[compare]], {{.*}}, {{.*}} : tensor<8x8x4xi1>, tensor<8x8x4xf16>, tensor<8x8x4xf16>)
// CHECK: %[[cast1:.*]] = hfusion.cast
func.func @test_normalize_triton_where_hfusion_compare_select(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xi8>, %arg4: memref<?xi8>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {global_kernel = "local"} {
  %c0_i8 = arith.constant 0 : i8
  %0 = tensor.empty() : tensor<8x8x4xi8>
  %1 = linalg.fill ins(%c0_i8 : i8) outs(%0 : tensor<8x8x4xi8>) -> tensor<8x8x4xi8>
  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [8, 8, 4], strides: [32, 4, 1] : memref<?xi8> to memref<8x8x4xi8, strided<[32, 4, 1]>>
  %alloc = memref.alloc() : memref<8x8x4xi8>
  memref.copy %reinterpret_cast, %alloc : memref<8x8x4xi8, strided<[32, 4, 1]>> to memref<8x8x4xi8>
  %2 = bufferization.to_tensor %alloc restrict writable : memref<8x8x4xi8>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [8, 8, 4], strides: [32, 4, 1] : memref<?xi8> to memref<8x8x4xi8, strided<[32, 4, 1]>>
  %alloc_1 = memref.alloc() : memref<8x8x4xi8>
  memref.copy %reinterpret_cast_0, %alloc_1 : memref<8x8x4xi8, strided<[32, 4, 1]>> to memref<8x8x4xi8>
  %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<8x8x4xi8>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8, 8, 4], strides: [32, 4, 1] : memref<?xi8> to memref<8x8x4xi8, strided<[32, 4, 1]>>
  %alloc_3 = memref.alloc() : memref<8x8x4xi8>
  memref.copy %reinterpret_cast_2, %alloc_3 : memref<8x8x4xi8, strided<[32, 4, 1]>> to memref<8x8x4xi8>
  %4 = bufferization.to_tensor %alloc_3 restrict writable : memref<8x8x4xi8>
  %5 = tensor.empty() : tensor<8x8x4xi1>
  %6 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%4, %1 : tensor<8x8x4xi8>, tensor<8x8x4xi8>) outs(%5 : tensor<8x8x4xi1>) -> tensor<8x8x4xi1>
  %7 = hfusion.select ins(%6, %2, %3 : tensor<8x8x4xi1>, tensor<8x8x4xi8>, tensor<8x8x4xi8>) outs(%0 : tensor<8x8x4xi8>) -> tensor<8x8x4xi8>
  %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8, 8, 4], strides: [32, 4, 1] : memref<?xi8> to memref<8x8x4xi8, strided<[32, 4, 1]>>
  %cast = memref.cast %reinterpret_cast_4 : memref<8x8x4xi8, strided<[32, 4, 1]>> to memref<8x8x4xi8, strided<[?, ?, ?], offset: ?>>
  bufferization.materialize_in_destination %7 in writable %cast : (tensor<8x8x4xi8>, memref<8x8x4xi8, strided<[?, ?, ?], offset: ?>>) -> ()
  return
}

// -----

// CHECK-LABEL: @test_normalize_i8_elemwise_unary_relu
// CHECK:  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>}
// CHECK:  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>}
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK:  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<trunc>}
func.func @test_normalize_i8_elemwise_unary_relu(%arg0: tensor<16xi8>) -> tensor<16xi8> {
  %dst = tensor.empty() : tensor<16xi8>
  %res = hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>}
          ins(%arg0 : tensor<16xi8>)
          outs(%dst : tensor<16xi8>) -> tensor<16xi8>
  return %res : tensor<16xi8>
}

// -----

// CHECK-LABEL: @test_normalize_i8_elemwise_unary_rec
// CHECK: %[[cast0:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>}
// CHECK: %[[cast1:.*]] =  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>}
// CHECK: %[[unary_rec:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%[[cast1]] : tensor<16xf32>) outs({{.*}} : tensor<16xf32>)
// CHECK:  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} 
// CHECK:  hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>}
func.func @test_normalize_i8_elemwise_unary_rec(%arg0: tensor<16xi8>) -> tensor<16xi8> {
  %dst = tensor.empty() : tensor<16xi8>
  %res = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>}
          ins(%arg0 : tensor<16xi8>)
          outs(%dst : tensor<16xi8>) -> tensor<16xi8>
  return %res : tensor<16xi8>
}

// -----

// CHECK-LABEL: @test_normalize_i8_reduce
// CHECK: arith.addf {{.*}} f32
// CHECK: linalg.reduce { arith.mulf } ins({{.*}} : tensor<32x16xf16>) outs({{.*}} : tensor<32xf16>) dimensions = [1]
func.func @test_normalize_i8_reduce(%arg0: tensor<32x16x8xi8>, %arg1: tensor<32x16xi8>) -> (tensor<32xi8>, tensor<32xi8>) {
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

// CHECK-LABEL: @test_normalize_i8_reduce_region
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: arith.maximumf
// CHECK: arith.minimumf
func.func @test_normalize_i8_reduce_region(%arg0: tensor<32x16x8xi8>, %arg1: tensor<32x16xi8>) -> tensor<32xi8> {
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

// CHECK-LABEL: @test_normalize_i8_reduce_with_index
// CHECK-DAG: %[[cast0:.*]] = hfusion.cast
// CHECK-DAG: %[[cast1:.*]] = hfusion.cast
// CHECK-DAG: %[[cast2:.*]] = hfusion.cast
// CHECK-DAG: %[[cast3:.*]] = hfusion.cast
// CHECK: linalg.reduce ins(%[[cast1]], {{.*}} : tensor<32x2xf32>, tensor<32x2xi32>) outs(%[[cast3]], {{.*}} : tensor<32xf32>, tensor<32xi32>)
// CHECK: arith.addf
// CHECK: arith.cmpf olt
// CHECK: arith.cmpf oge
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<32xf32>) outs({{.*}} : tensor<32xi32>)
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<32xi32>) outs({{.*}} : tensor<32xi8>)
func.func @test_normalize_i8_reduce_with_index(%arg0: tensor<32x16x8xi8>) -> (tensor<32xi8>, tensor<32xi32>) {
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

// CHECK-LABEL: @test_normalize_i8_elemwise_mod
// CHECK: hfusion.cast {{.*}} ins(%arg0 : tensor<64xi8>) outs({{.*}} : tensor<64xf16>) -> tensor<64xf16>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<64xf16>) outs({{.*}} : tensor<64xf32>) -> tensor<64xf32>
// CHECK: hfusion.cast {{.*}} ins(%arg1 : tensor<64xi8>) outs({{.*}} : tensor<64xf16>) -> tensor<64xf16>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<64xf16>) outs({{.*}} : tensor<64xf32>) -> tensor<64xf32>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins({{.*}}, {{.*}} : tensor<64xf32>, tensor<64xf32>) outs({{.*}} : tensor<64xf32>) -> tensor<64xf32>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<64xf32>) outs({{.*}} : tensor<64xf32>) -> tensor<64xf32>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins({{.*}}, {{.*}} : tensor<64xf32>, tensor<64xf32>) outs({{.*}} : tensor<64xf32>) -> tensor<64xf32>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins({{.*}}, {{.*}} : tensor<64xf32>, tensor<64xf32>) outs({{.*}} : tensor<64xf32>) -> tensor<64xf32>
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins({{.*}}, {{.*}} : tensor<64xf32>, f32) outs({{.*}} : tensor<64xi1>) -> tensor<64xi1>
// CHECK: hfusion.select ins({{.*}}, {{.*}}, {{.*}} : tensor<64xi1>, tensor<64xf32>, tensor<64xf32>) outs({{.*}} : tensor<64xf32>) -> tensor<64xf32>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<64xf32>) outs({{.*}} : tensor<64xf16>) -> tensor<64xf16>
// CHECK: hfusion.cast {{.*}} ins({{.*}} : tensor<64xf16>) outs({{.*}} : tensor<64xi8>) -> tensor<64xi8>
func.func @test_normalize_i8_elemwise_mod(%arg0: tensor<64xi8>, %arg1: tensor<64xi8>) -> tensor<64xi8> {
  %0 = tensor.empty() : tensor<64xi8>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
                                ins(%arg0, %arg1 : tensor<64xi8>, tensor<64xi8>)
                                outs(%0 : tensor<64xi8>) -> tensor<64xi8>
  return %1 : tensor<64xi8>
}

// -----

// CHECK-LABEL: @test_reduce_max_with_index
// CHECK: hfusion.reduce_with_index {already_denaned, tie_break_left = true} <max> ins(%[[input0:.*]], %[[input1:.*]] : tensor<4x64xf16>, tensor<4x64xi32>) outs(%[[init0:.*]], %[[init1:.*]] : tensor<4xf16>, tensor<4xi32>) dimensions = [1] -> tensor<4xf16>, tensor<4xi32>
module {
  func.func @test_reduce_max_with_index(%arg0: tensor<4x64xi8>, %arg1: tensor<4x64xi32>) -> (tensor<4xi8>, tensor<4xi32>) {
    %0 = tensor.empty() : tensor<4xi8>
    %1 = tensor.empty() : tensor<4xi32>
    %2:2 = hfusion.reduce_with_index {tie_break_left = true} <max> ins(%arg0, %arg1 : tensor<4x64xi8>, tensor<4x64xi32>) outs(%0, %1 : tensor<4xi8>, tensor<4xi32>) dimensions = [1] -> tensor<4xi8>, tensor<4xi32>
    return %2#0, %2#1 : tensor<4xi8>, tensor<4xi32>
  }
}

// -----

// CHECK-LABEL: @test_gather_b8
// CHECK: hfusion.gather {operandSegmentSizes = array<i32: 2, 1>} ins(%[[INPUT0:.*]], %arg1 : tensor<4x64xf16>, tensor<4x32xi32>) outs(%[[OUTPUT0:.*]] : tensor<4x32xf16>) axis = 1 -> tensor<4x32xf16>
module {
  func.func @test_gather_b8(%arg0: tensor<4x64xi8>, %arg1: tensor<4x32xi32>) -> tensor<4x32xi8> {
    %0 = tensor.empty() : tensor<4x32xi8>
    %1 = hfusion.gather ins(%arg0, %arg1 : tensor<4x64xi8>, tensor<4x32xi32>) outs(%0 : tensor<4x32xi8>) axis = 1 -> tensor<4x32xi8>
    return %1 : tensor<4x32xi8>
  }
}

// -----

// CHECK-LABEL: @test_cumsum_b8
// CHECK: hfusion.cumsum %[[INPUT0:.*]] : tensor<4x64xf32> cum_dims = [0] reverse = false -> tensor<4x32xf32>
module {
  func.func @test_cumsum_b8(%arg0: tensor<4x64xi8>) -> tensor<4x32xi8> {
    %0 = tensor.empty() : tensor<4x32xi8>
    %1 = hfusion.cumsum %arg0 : tensor<4x64xi8> cum_dims = [0] reverse = false -> tensor<4x32xi8>
    return %1 : tensor<4x32xi8>
  }
}

// -----

// CHECK-LABEL: @test_cumprod_b8
// CHECK: hfusion.cumprod %[[INPUT0:.*]] : tensor<4x64xf32> cum_dims = [1] reverse = false -> tensor<4x32xf32>
module {
  func.func @test_cumprod_b8(%arg0: tensor<4x64xi8>) -> tensor<4x32xi8> {
    %0 = tensor.empty() : tensor<4x32xi8>
    %1 = hfusion.cumprod %arg0 : tensor<4x64xi8> cum_dims = [1] reverse = false -> tensor<4x32xi8>
    return %1 : tensor<4x32xi8>
  }
}

// -----

// CHECK-LABEL: @test_reduce_i8_andi
// CHECK: linalg.reduce { arith.andi } ins({{.*}} : tensor<16x32x64xi8>)
func.func @test_reduce_i8_andi(%arg0: tensor<16x32x64xi8>) -> tensor<16x32xi8> {
  %0 = tensor.empty() : tensor<16x32xi8>
  %reduce = linalg.reduce { arith.andi } ins(%arg0 : tensor<16x32x64xi8>) outs(%0 : tensor<16x32xi8>) dimensions = [2]
  return %reduce : tensor<16x32xi8>
}

// -----

// CHECK-LABEL: @test_reduce_i8_ori
// CHECK: linalg.reduce { arith.ori } ins({{.*}} : tensor<16x32x64xi8>)
func.func @test_reduce_i8_ori(%arg0: tensor<16x32x64xi8>) -> tensor<16x32xi8> {
  %0 = tensor.empty() : tensor<16x32xi8>
  %reduce = linalg.reduce { arith.ori } ins(%arg0 : tensor<16x32x64xi8>) outs(%0 : tensor<16x32xi8>) dimensions = [2]
  return %reduce : tensor<16x32xi8>
}

// -----

// CHECK-LABEL: @test_reduce_i8_xori
// CHECK: linalg.reduce { arith.xori } ins({{.*}} : tensor<16x32x64xi8>)
func.func @test_reduce_i8_xori(%arg0: tensor<16x32x64xi8>) -> tensor<16x32xi8> {
  %0 = tensor.empty() : tensor<16x32xi8>
  %reduce = linalg.reduce { arith.xori } ins(%arg0 : tensor<16x32x64xi8>) outs(%0 : tensor<16x32xi8>) dimensions = [2]
  return %reduce : tensor<16x32xi8>
}
