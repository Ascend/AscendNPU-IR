// RUN: bishengir-opt --hfusion-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_NormalizeXor_xori
// CHECK-SAME: (%[[arg0:.*]]: tensor<512xi16>, %[[arg1:.*]]: tensor<512xi16>)
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<512xi16>
// CHECK: %[[vor:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%[[arg0]], %[[arg1]] : tensor<512xi16>, tensor<512xi16>) outs(%[[empty]] : tensor<512xi16>) -> tensor<512xi16>
// CHECK: %[[empty1:.*]] = tensor.empty() : tensor<512xi16>
// CHECK: %[[vand:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[arg0]], %[[arg1]] : tensor<512xi16>, tensor<512xi16>) outs(%[[empty1]] : tensor<512xi16>) -> tensor<512xi16>
// CHECK: %[[vnot:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[vand]] : tensor<512xi16>) outs(%[[vand]] : tensor<512xi16>) -> tensor<512xi16>
// CHECK: %[[empty2:.*]] = tensor.empty() : tensor<512xi16>
// CHECK: %[[xor:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[vnot]], %[[vor]] : tensor<512xi16>, tensor<512xi16>) outs(%[[empty2]] : tensor<512xi16>) -> tensor<512xi16>
// CHECK: return %[[xor]] : tensor<512xi16>
func.func @test_NormalizeXor_xori(%arg0: tensor<512xi16>, %arg1: tensor<512xi16>) -> tensor<512xi16> {
    %0 = tensor.empty() : tensor<512xi16>
    %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vxor>} ins(%arg0, %arg1 : tensor<512xi16>, tensor<512xi16>) outs(%0 : tensor<512xi16>) -> tensor<512xi16>
    return %1 : tensor<512xi16>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeShiftI8ToI16_i8_shrsi
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<32xf16>
// CHECK: %[[CAST0:.*]] = hfusion.cast {{.*}} ins(%[[ARG0:.*]] : tensor<32xi8>) outs(%[[EMPTY0]] : tensor<32xf16>) -> tensor<32xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<32xi16>
// CHECK: %[[CAST1:.*]] = hfusion.cast {{.*}} ins(%[[CAST0]] : tensor<32xf16>) outs(%[[EMPTY1]] : tensor<32xi16>) -> tensor<32xi16>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<32xf16>
// CHECK: %[[CAST2:.*]] = hfusion.cast {{.*}} ins(%[[ARG1:.*]] : tensor<32xi8>) outs(%[[EMPTY2]] : tensor<32xf16>) -> tensor<32xf16>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<32xi16>
// CHECK: %[[CAST3:.*]] = hfusion.cast {{.*}} ins(%[[CAST2]] : tensor<32xf16>) outs(%[[EMPTY3]] : tensor<32xi16>) -> tensor<32xi16>
// CHECK: %[[EMPTY4:.*]] = tensor.empty() : tensor<32xi16>
// CHECK: %[[SHIFT:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%[[CAST1]], %[[CAST3]] : tensor<32xi16>, tensor<32xi16>) outs(%[[EMPTY4]] : tensor<32xi16>) -> tensor<32xi16>
// CHECK: %[[EMPTY5:.*]] = tensor.empty() : tensor<32xi8>
// CHECK: %[[RES:.*]] = hfusion.cast {{.*}} ins(%[[SHIFT]] : tensor<32xi16>) outs(%[[EMPTY5]] : tensor<32xi8>) -> tensor<32xi8>
// CHECK: return %[[RES]] : tensor<32xi8>
func.func @test_NormalizeShiftI8ToI16_i8_shrsi(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi8> {
  %0 = tensor.empty() : tensor<32xi8>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%arg0, %arg1 : tensor<32xi8>, tensor<32xi8>) outs(%0 : tensor<32xi8>) -> tensor<32xi8>
  return %1 : tensor<32xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeShiftI8ToI16_i8_shrui
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<32xf16>
// CHECK: %[[CAST0:.*]] = hfusion.cast {{.*}} ins(%[[ARG0:.*]] : tensor<32xi8>) outs(%[[EMPTY0]] : tensor<32xf16>) -> tensor<32xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<32xi16>
// CHECK: %[[CAST1:.*]] = hfusion.cast {{.*}} ins(%[[CAST0]] : tensor<32xf16>) outs(%[[EMPTY1]] : tensor<32xi16>) -> tensor<32xi16>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<32xf16>
// CHECK: %[[CAST2:.*]] = hfusion.cast {{.*}} ins(%[[ARG1:.*]] : tensor<32xi8>) outs(%[[EMPTY2]] : tensor<32xf16>) -> tensor<32xf16>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<32xi16>
// CHECK: %[[CAST3:.*]] = hfusion.cast {{.*}} ins(%[[CAST2]] : tensor<32xf16>) outs(%[[EMPTY3]] : tensor<32xi16>) -> tensor<32xi16>
// CHECK: %[[EMPTY4:.*]] = tensor.empty() : tensor<32xi16>
// CHECK: %[[SHIFT:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%[[CAST1]], %[[CAST3]] : tensor<32xi16>, tensor<32xi16>) outs(%[[EMPTY4]] : tensor<32xi16>) -> tensor<32xi16>
// CHECK: %[[EMPTY5:.*]] = tensor.empty() : tensor<32xi8>
// CHECK: %[[RES:.*]] = hfusion.cast {{.*}} ins(%[[SHIFT]] : tensor<32xi16>) outs(%[[EMPTY5]] : tensor<32xi8>) -> tensor<32xi8>
// CHECK: return %[[RES]] : tensor<32xi8>
func.func @test_NormalizeShiftI8ToI16_i8_shrui(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi8> {
  %0 = tensor.empty() : tensor<32xi8>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%arg0, %arg1 : tensor<32xi8>, tensor<32xi8>) outs(%0 : tensor<32xi8>) -> tensor<32xi8>
  return %1 : tensor<32xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeShiftI8ToI16_i8_shift
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<200xf16>
// CHECK: %[[CAST0:.*]] = hfusion.cast {{.*}} ins(%[[ARG0:.*]] : tensor<200xi8>) outs(%[[EMPTY0:.*]] : tensor<200xf16>) -> tensor<200xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<200xi16>
// CHECK: %[[CAST1:.*]] = hfusion.cast {{.*}} ins(%[[CAST0:.*]] : tensor<200xf16>) outs(%[[EMPTY1:.*]]: tensor<200xi16>) -> tensor<200xi16>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<200xf16>
// CHECK: %[[CAST2:.*]] = hfusion.cast {{.*}} ins(%[[ARG1:.*]] : tensor<200xi8>) outs(%[[EMPTY2:.*]] : tensor<200xf16>) -> tensor<200xf16>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<200xi16>
// CHECK: %[[CAST3:.*]] = hfusion.cast {{.*}} ins(%[[CAST2:.*]] : tensor<200xf16>) outs(%[[EMPTY3:.*]] : tensor<200xi16>) -> tensor<200xi16>
// CHECK: %[[EMPTY4:.*]] = tensor.empty() : tensor<200xi16>
// CHECK: %[[SHLI:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shli>} ins(%[[CAST1:.*]], %[[CAST3:.*]] : tensor<200xi16>, tensor<200xi16>) outs(%[[EMPTY4:.*]] : tensor<200xi16>) -> tensor<200xi16>
// CHECK: %[[EMPTY6:.*]] = tensor.empty() : tensor<200xi8>
// CHECK: %[[RES:.*]] = hfusion.cast {{.*}} ins(%[[EMPTY4:.*]] : tensor<200xi16>) outs(%[[EMPTY6:.*]] : tensor<200xi8>) -> tensor<200xi8>
func.func @test_NormalizeShiftI8ToI16_i8_shift(%arg0: tensor<200xi8>, %arg1: tensor<200xi8>) -> tensor<200xi8>{
  %0 = tensor.empty() : tensor<200xi8>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shli>} ins(%arg0, %arg1: tensor<200xi8>, tensor<200xi8>) outs(%0 : tensor<200xi8>) -> tensor<200xi8>
  return %1 : tensor<200xi8>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCmpToCast_hfusion_compare_neq_ops
// CHECK-SAME: (%[[arg0:.*]]: tensor<1024xi64>, %[[arg1:.*]]: tensor<1024xi1>)
// CHECK: %[[arg2:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[arg3:.*]] = hfusion.cast {{.*}} ins(%[[arg0]] : tensor<1024xi64>) outs(%[[arg2]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[arg4:.*]] = tensor.empty() : tensor<1024xi1>
// CHECK: %[[arg6:.*]] = tensor.empty() : tensor<1024xi1>
// CHECK: %[[arg5:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[arg3]], %[[cst:.*]] : tensor<1024xf32>, f32) outs(%[[arg6]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: %[[arg7:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[arg5]] : tensor<1024xi1>) outs(%[[arg4]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: return %[[arg7]]
func.func @test_NormalizeCmpToCast_hfusion_compare_neq_ops(
  %src1 : tensor<1024xi64>,  %dst : tensor<1024xi1>) ->  tensor<1024xi1> {
  %c0_i64 = arith.constant 0 : i64
  %0 = tensor.empty() : tensor<1024xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1024xi64>) -> tensor<1024xi64>
  %ret = hfusion.compare {compare_fn  = #hfusion.compare_fn<vne>}
    ins(%src1, %1 : tensor<1024xi64>, tensor<1024xi64>)
    outs(%dst : tensor<1024xi1>)
    -> tensor<1024xi1>
  return %ret : tensor<1024xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCmpToCast_hfusion_compare_neq_ops_left_zero
// CHECK-SAME: (%[[arg0:.*]]: tensor<1024xi64>, %[[arg1:.*]]: tensor<1024xi1>)
// CHECK: %[[arg2:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[arg3:.*]] = hfusion.cast {{.*}} ins(%[[arg0]] : tensor<1024xi64>) outs(%[[arg2]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NOT: compare_fn = #hfusion.compare_fn<vne>
// CHECK: return
func.func @test_NormalizeCmpToCast_hfusion_compare_neq_ops_left_zero(
  %src1 : tensor<1024xi64>,  %dst : tensor<1024xi1>) ->  tensor<1024xi1> {
  %c0_i64 = arith.constant 0 : i64
  %0 = tensor.empty() : tensor<1024xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1024xi64>) -> tensor<1024xi64>
  %ret = hfusion.compare {compare_fn  = #hfusion.compare_fn<vne>}
    ins(%1, %src1 : tensor<1024xi64>, tensor<1024xi64>)
    outs(%dst : tensor<1024xi1>)
    -> tensor<1024xi1>
  return %ret : tensor<1024xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCmpVne_normalize_compare_neq_to_Not_eq
// CHECK-SAME: (%[[arg0:.*]]: tensor<1024xi64>, %[[arg1:.*]]: tensor<1024xi64>, %[[arg2:.*]]: tensor<1024xi1>)
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<1024xi1>
// CHECK: %[[veq:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[arg0]], %[[arg1]] : tensor<1024xi64>, tensor<1024xi64>) outs(%[[empty]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: %[[notOp:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[veq]] : tensor<1024xi1>) outs(%[[arg2]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: return %[[notOp]]
func.func @test_NormalizeCmpVne_normalize_compare_neq_to_Not_eq(
  %src1 : tensor<1024xi64>, %src2 : tensor<1024xi64>,  %dst : tensor<1024xi1>) ->  tensor<1024xi1> {
  %ret = hfusion.compare {compare_fn  = #hfusion.compare_fn<vne>}
    ins(%src1, %src2 : tensor<1024xi64>, tensor<1024xi64>)
    outs(%dst : tensor<1024xi1>)
    -> tensor<1024xi1>
  return %ret : tensor<1024xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCmpVne_normalize_f32_compare_neq_to_Not_eq
// CHECK-SAME: (%[[arg0:.*]]: tensor<128xf32>, %[[arg1:.*]]: tensor<128xf32>, %[[arg2:.*]]: tensor<128xi1>)
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<128xi1>
// CHECK: %[[veq:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[arg0]], %[[arg1]] : tensor<128xf32>, tensor<128xf32>) outs(%[[empty]] : tensor<128xi1>) -> tensor<128xi1>
// CHECK: %[[notOp:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[veq]] : tensor<128xi1>) outs(%[[arg2]] : tensor<128xi1>) -> tensor<128xi1>
// CHECK: return %[[notOp]]
func.func @test_NormalizeCmpVne_normalize_f32_compare_neq_to_Not_eq(
  %src1 : tensor<128xf32>, %src2 : tensor<128xf32>,  %dst : tensor<128xi1>) ->  tensor<128xi1> {
  %ret = hfusion.compare {compare_fn  = #hfusion.compare_fn<vne>}
    ins(%src1, %src2 : tensor<128xf32>, tensor<128xf32>)
    outs(%dst : tensor<128xi1>)
    -> tensor<128xi1>
  return %ret : tensor<128xi1>
}

// -----

// CHECK-LABEL: @test_NormalizeCmpVne_triton_where_hfusion_compare_select
// CHECK: %[[cast0:.*]] = hfusion.cast
// CHECK: %[[fill0:.*]] = linalg.fill
// CHECK: %[[compare_tmp:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[cast0]], %[[fill0]] : tensor<8x8x4xf16>, tensor<8x8x4xf16>)
// CHECK: %[[compare:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[compare_tmp]] : tensor<8x8x4xi1>)
// CHECK: %[[select:.*]] =  hfusion.select ins(%[[compare]], {{.*}}, {{.*}} : tensor<8x8x4xi1>, tensor<8x8x4xf16>, tensor<8x8x4xf16>)
// CHECK: %[[cast1:.*]] = hfusion.cast
func.func @test_NormalizeCmpVne_triton_where_hfusion_compare_select(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xi8>, %arg4: memref<?xi8>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {global_kernel = "local"} {
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

// CHECK-LABEL: func.func @test_NormalizeI8I32Cmp_normalize_i8_hfusion_compare
// CHECK: %[[CAST0:.*]] = hfusion.cast {{.*}} ins({{.*}} : tensor<16xi8>)
// CHECK: %[[CAST1:.*]] = hfusion.cast {{.*}} ins({{.*}} : tensor<16xi8>)
// CHECK: %[[Veq:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[CAST0]], %[[CAST1]] : tensor<16xf16>, tensor<16xf16>)
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[Veq]] : tensor<16xi1>)
func.func @test_NormalizeI8I32Cmp_normalize_i8_hfusion_compare(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi1> {
  %dst1 = tensor.empty() : tensor<16xi1>
  %dst2 = tensor.empty() : tensor<16xi1>
  %res1 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
    ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>)
    outs(%dst1 : tensor<16xi1>) -> tensor<16xi1>
  return %res1 : tensor<16xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeI8I32Cmp_normalize_i32_hfusion_compare
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[ARG0:.*]], %[[ARG0:.*]] : tensor<16xi32>, tensor<16xi32>)
// CHECK: %[[Veq:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[ARG1:.*]], %[[ARG1:.*]] : tensor<16xi32>, tensor<16xi32>)
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[Veq]] : tensor<16xi1>)
func.func @test_NormalizeI8I32Cmp_normalize_i32_hfusion_compare(%arg0: tensor<16xi32>, %arg1: tensor<16xi32>) -> (tensor<16xi1>, tensor<16xi1>) {
  %dst1 = tensor.empty() : tensor<16xi1>
  %dst2 = tensor.empty() : tensor<16xi1>
  %res1 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
    ins(%arg0, %arg1 : tensor<16xi32>, tensor<16xi32>)
    outs(%dst1 : tensor<16xi1>) -> tensor<16xi1>
  %res2 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
    ins(%arg0, %arg0 : tensor<16xi32>, tensor<16xi32>)
    outs(%dst2 : tensor<16xi1>) -> tensor<16xi1>
  return %res1, %res2 : tensor<16xi1>, tensor<16xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeI8I32Cmp_normalize_i32_hfusion_compare_vlt
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<16xi64>
// CHECK: %[[CAST0:.*]] = hfusion.cast {{.*}} ins(%[[ARG0:.*]] : tensor<16xi32>) outs(%[[EMPTY0]] : tensor<16xi64>) -> tensor<16xi64>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<16xi64>
// CHECK: %[[CAST1:.*]] = hfusion.cast {{.*}} ins(%[[ARG1:.*]] : tensor<16xi32>) outs(%[[EMPTY1]] : tensor<16xi64>) -> tensor<16xi64>
// CHECK: %[[CMP:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>} ins(%[[CAST0]], %[[CAST1]] : tensor<16xi64>, tensor<16xi64>) outs(%[[DST:.*]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: return %[[CMP]] : tensor<16xi1>
func.func @test_NormalizeI8I32Cmp_normalize_i32_hfusion_compare_vlt(%arg0: tensor<16xi32>, %arg1: tensor<16xi32>) -> tensor<16xi1> {
  %dst = tensor.empty() : tensor<16xi1>
  %res = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}
    ins(%arg0, %arg1 : tensor<16xi32>, tensor<16xi32>)
    outs(%dst : tensor<16xi1>) -> tensor<16xi1>
  return %res : tensor<16xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeI8I32Cmp_normalize_i32_hfusion_compare_dynamic(
// CHECK: hfusion.compare {fun = #hfusion.compare_fn<veq>} ins(%[[ARG0:.*]], %[[ARG1:.*]] : tensor<?x?xi32>, tensor<?x?xi32>)
func.func @test_NormalizeI8I32Cmp_normalize_i32_hfusion_compare_dynamic(%arg0: tensor<?x?xi32>) -> (tensor<?x?xi32>, tensor<?x?xi1>) attributes {OperatorType = "Default", compute_capability = "", frontend_symbol = {input_0 = ["s93", "s94"], output_0 = ["s93", "s94"], output_1 = ["s93", "s94"]}, hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32319_i32 = arith.constant 32319 : i32
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xi32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xi32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xi32>
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?x?xi32>) -> tensor<?x?xi32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%arg0, %1 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%0 : tensor<?x?xi32>) -> tensor<?x?xi32>
  %3 = tensor.empty(%dim, %dim_0) : tensor<?x?xi32>
  %4 = linalg.fill ins(%c32319_i32 : i32) outs(%3 : tensor<?x?xi32>) -> tensor<?x?xi32>
  %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>} ins(%2, %4 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%3 : tensor<?x?xi32>) -> tensor<?x?xi32>
  %6 = tensor.empty(%dim, %dim_0) : tensor<?x?xi1>
  %7 = hfusion.compare {fun = #hfusion.compare_fn<veq>} ins(%arg0, %5 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%6 : tensor<?x?xi1>) -> tensor<?x?xi1>
  return %5, %7 : tensor<?x?xi32>, tensor<?x?xi1>
}

// -----

// CHECK-LABEL: @test_NormalizeI8I32Cmp_select_i1_to_i16_compare
// CHECK: hfusion.select
// CHECK: hfusion.select
// CHECK: hfusion.select
// CHECK: hfusion.compare
module attributes {hacc.target = #hacc.target<"Ascend950PR_957c">} {
  func.func @test_NormalizeI8I32Cmp_select_i1_to_i16_compare(
    %arg0: tensor<8xi1> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
    %arg1: tensor<8xi1>,
    %arg2: tensor<8xi1>) -> tensor<8xi1> {
    %out = tensor.empty() : tensor<8xi1>
    %0 = hfusion.select
        ins(%arg0, %arg1, %arg2 : tensor<8xi1>, tensor<8xi1>, tensor<8xi1>)
        outs(%out : tensor<8xi1>) -> tensor<8xi1>
    return %0 : tensor<8xi1>
  }
}

// -----

// CHECK-LABEL: @test_NormalizeCmpOp_compare_i1
// CHECK: %[[ret:.*]] = hfusion.compare  {compare_fn = #hfusion.compare_fn<vlt>} ins(%[[in1:.*]], %[[in2:.*]] : tensor<16x32xf16>, tensor<16x32xf16>)
func.func @test_NormalizeCmpOp_compare_i1(%arg0: tensor<16x32xi1>,%arg1: tensor<16x32xi1>,  %dst : tensor<16x32xi1>) -> (tensor<16x32xi1>) {
  %ret = hfusion.compare {compare_fn  = #hfusion.compare_fn<vlt>}
    ins(%arg0, %arg1 : tensor<16x32xi1>, tensor<16x32xi1>)
    outs(%dst : tensor<16x32xi1>)
    -> tensor<16x32xi1>
  return %ret : tensor<16x32xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeCmpOp_remove_overflow_annotation
// CHECK: %[[RET:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>} ins(%[[IN1:.*]], %[[IN2:.*]] : tensor<8xf32>, tensor<8xf32>) outs(%[[DST:.*]] : tensor<8xi1>) -> tensor<8xi1>
// CHECK-NOT: annotation.mark
// CHECK: return %[[RET]] : tensor<8xi1>
func.func @test_NormalizeCmpOp_remove_overflow_annotation(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %dst: tensor<8xi1>) -> tensor<8xi1> {
  %ret = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}
    ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%dst : tensor<8xi1>)
    -> tensor<8xi1>
  annotation.mark %ret {overflow_mode = "trunc"} : tensor<8xi1>
  return %ret : tensor<8xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeIsInf_isinf
// CHECK: %[[CST0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[NEGONE:.*]] = arith.constant -1 : i32
// CHECK: %[[POSONE:.*]] = arith.constant 1 : i32
// CHECK: %[[NEGINF:.*]] = arith.constant -2139095040 : i32
// CHECK: %[[MASKVAL:.*]] = arith.constant 2147483647 : i32
// CHECK: %[[INPUT:.*]] = tensor.empty() : tensor<8192xf32>
// CHECK: %[[MASKRES:.*]] = tensor.empty() : tensor<8192xi32>
// CHECK: %[[VDUPOP:.*]] = linalg.fill ins(%[[MASKVAL]] : i32) outs(%[[MASKRES]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[BITCASTEMPTY:.*]] = tensor.empty() : tensor<8192xi32>
// CHECK: %[[BITCASTINPUT:.*]] = hfusion.bitcast ins(%[[INPUT]] : tensor<8192xf32>) outs(%[[BITCASTOUT:.*]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[VANDOP:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[BITCASTINPUT]], %[[VDUPOP]] : tensor<8192xi32>, tensor<8192xi32>) outs(%[[VANDOUTPUT:.*]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[VADDRES:.*]] = tensor.empty() : tensor<8192xi32>
// CHECK: %[[VADDOP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VANDOP]], %[[NEGINF]] : tensor<8192xi32>, i32) outs(%[[VADDRES]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[BITCASTADD:.*]] = hfusion.bitcast ins(%[[VADDOP]] : tensor<8192xi32>) outs(%[[BITCASTOUT:.*]] : tensor<8192xf32>) -> tensor<8192xf32>
// CHECK: %[[VABSOP:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[BITCASTADD]] : tensor<8192xf32>) outs(%[[BITCASTOUTPUT:.*]] : tensor<8192xf32>) -> tensor<8192xf32>
// CHECK: %[[BITCASTABS:.*]] = hfusion.bitcast ins(%[[VABSOP]] : tensor<8192xf32>) outs(%[[BITCASTOUT:.*]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[VMINOP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>} ins(%[[BITCASTABS]], %[[POSONE]] : tensor<8192xi32>, i32) outs(%[[BITCASTOUTPUT:.*]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[VMULOP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VMINOP]], %[[NEGONE]] : tensor<8192xi32>, i32) outs(%[[VMINOP]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[VADDOP1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VMULOP]], %[[POSONE]] : tensor<8192xi32>, i32) outs(%[[VMULOP]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[TMPF:.*]] = tensor.empty() : tensor<8192xf32>
// CHECK: %[[CASTF:.*]] = hfusion.cast {{.*}} ins(%[[VADDOP1]] : tensor<8192xi32>) outs(%[[TMPF]] : tensor<8192xf32>) -> tensor<8192xf32>
// CHECK: %[[OUT1:.*]] = tensor.empty() : tensor<8192xi1>
// CHECK: %[[OUT2:.*]] = tensor.empty() : tensor<8192xi1>
// CHECK: %[[RES:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[CASTF]], %[[CST0]] : tensor<8192xf32>, f32) outs(%[[OUT2]] : tensor<8192xi1>) -> tensor<8192xi1>
// CHECK: %[[RES2:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[RES]] : tensor<8192xi1>) outs(%[[OUT1]] : tensor<8192xi1>) -> tensor<8192xi1>
func.func @test_NormalizeIsInf_isinf() -> tensor<8192xi1> {
  %0 = tensor.empty() : tensor<8192xf32>
  %2 = hfusion.isinf %0 : tensor<8192xf32> -> tensor<8192xi1>
  return %2 : tensor<8192xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeIsInf_isinf_f16
// CHECK: %[[CST0:.*]] = arith.constant 0.000000e+00 : f16
// CHECK: %[[NEGONE:.*]] = arith.constant -1 : i16
// CHECK: %[[POSONE:.*]] = arith.constant 1 : i16
// CHECK: %[[NEGINF:.*]] = arith.constant -31744 : i16
// CHECK: %[[MASKVAL:.*]] = arith.constant 32767 : i16
// CHECK: %[[INPUT:.*]] = tensor.empty() : tensor<256xf16>
// CHECK: %[[VDUPOP:.*]] = linalg.fill ins(%[[MASKVAL]] : i16) outs(%[[MASKRES:.*]] : tensor<256xi16>) -> tensor<256xi16>
// CHECK: %[[BITCASTINPUT:.*]] = hfusion.bitcast ins(%[[INPUT]] : tensor<256xf16>) outs(%[[BITCASTOUT:.*]] : tensor<256xi16>) -> tensor<256xi16>
// CHECK: %[[VANDOP:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[BITCASTINPUT]], %[[VDUPOP]] : tensor<256xi16>, tensor<256xi16>) outs(%[[VANDOUTPUT:.*]] : tensor<256xi16>) -> tensor<256xi16>
// CHECK: %[[VADDOP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VANDOP]], %[[NEGINF]] : tensor<256xi16>, i16) outs(%[[VADDRES:.*]] : tensor<256xi16>) -> tensor<256xi16>
// CHECK: %[[BITCASTADD:.*]] = hfusion.bitcast ins(%[[VADDOP]] : tensor<256xi16>) outs(%[[BITCASTADDOUT:.*]] : tensor<256xf16>) -> tensor<256xf16>
// CHECK: %[[VABSOP:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[BITCASTADD]] : tensor<256xf16>) outs(%[[ABSOUT:.*]] : tensor<256xf16>) -> tensor<256xf16>
// CHECK: %[[BITCASTABS:.*]] = hfusion.bitcast ins(%[[VABSOP]] : tensor<256xf16>) outs(%[[BITCASTABSOUT:.*]] : tensor<256xi16>) -> tensor<256xi16>
// CHECK: %[[VMINOP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>} ins(%[[BITCASTABS]], %[[POSONE]] : tensor<256xi16>, i16) outs(%[[MINOUT:.*]] : tensor<256xi16>) -> tensor<256xi16>
// CHECK: %[[VMULOP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VMINOP]], %[[NEGONE]] : tensor<256xi16>, i16) outs(%{{.*}} : tensor<256xi16>) -> tensor<256xi16>
// CHECK: %[[VADDOP1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VMULOP]], %[[POSONE]] : tensor<256xi16>, i16) outs(%{{.*}} : tensor<256xi16>) -> tensor<256xi16>
// CHECK: %[[CASTF:.*]] = hfusion.cast {{.*}} ins(%[[VADDOP1]] : tensor<256xi16>) outs(%[[TMPF:.*]] : tensor<256xf16>) -> tensor<256xf16>
// CHECK: %[[RES:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[CASTF]], %[[CST0]] : tensor<256xf16>, f16) outs(%[[OUT2:.*]] : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[RES2:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[RES]] : tensor<256xi1>) outs(%[[OUT1:.*]] : tensor<256xi1>) -> tensor<256xi1>
func.func @test_NormalizeIsInf_isinf_f16() -> tensor<256xi1> {
  %0 = tensor.empty() : tensor<256xf16>
  %1 = hfusion.isinf %0 : tensor<256xf16> -> tensor<256xi1>
  return %1 : tensor<256xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeIsInf_isinf_bf16
// CHECK: %[[NEGINF:.*]] = arith.constant -31744 : i16
// CHECK: %[[MASKVAL:.*]] = arith.constant 32767 : i16
// CHECK: %[[INPUT:.*]] = tensor.empty() : tensor<128xbf16>
// CHECK: %[[VDUPOP:.*]] = linalg.fill ins(%[[MASKVAL]] : i16) outs(%[[MASKRES:.*]] : tensor<128xi16>) -> tensor<128xi16>
// CHECK: %[[BITCASTINPUT:.*]] = hfusion.bitcast ins(%[[INPUT]] : tensor<128xbf16>) outs(%[[BITCASTOUT:.*]] : tensor<128xi16>) -> tensor<128xi16>
// CHECK: %[[VADDOP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%{{.*}}, %[[NEGINF]] : tensor<128xi16>, i16) outs(%[[VADDRES:.*]] : tensor<128xi16>) -> tensor<128xi16>
// CHECK: %[[BITCASTADD:.*]] = hfusion.bitcast ins(%[[VADDOP]] : tensor<128xi16>) outs(%[[BITCASTADDOUT:.*]] : tensor<128xbf16>) -> tensor<128xbf16>
// CHECK: %[[VABSOP:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[BITCASTADD]] : tensor<128xbf16>) outs(%[[ABSOUT:.*]] : tensor<128xbf16>) -> tensor<128xbf16>
// CHECK: return %[[RES:.*]] : tensor<128xi1>
func.func @test_NormalizeIsInf_isinf_bf16() -> tensor<128xi1> {
  %0 = tensor.empty() : tensor<128xbf16>
  %1 = hfusion.isinf %0 : tensor<128xbf16> -> tensor<128xi1>
  return %1 : tensor<128xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeIsNan_isnan
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK: %[[POSONE:.*]] = arith.constant 1 : i32
// CHECK: %[[NEGINF:.*]] = arith.constant -2139095040 : i32
// CHECK: %[[MASKVAL:.*]] = arith.constant 2147483647 : i32
// CHECK: %[[INPUT:.*]] = tensor.empty() : tensor<8192xf32>
// CHECK: %[[MASKRES:.*]] = tensor.empty() : tensor<8192xi32>
// CHECK: %[[VDUPOP:.*]] = linalg.fill ins(%[[MASKVAL]] : i32) outs(%[[MASKRES]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[BITCASTEMPTY:.*]] = tensor.empty() : tensor<8192xi32>
// CHECK: %[[BITCASTINPUT:.*]] = hfusion.bitcast ins(%[[INPUT]] : tensor<8192xf32>) outs(%[[BITCASTOUT:.*]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[VANDOP:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[BITCASTINPUT]], %[[VDUPOP]] : tensor<8192xi32>, tensor<8192xi32>) outs(%[[VANDOUTPUT:.*]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[VADDOP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VANDOP]], %[[NEGINF]] : tensor<8192xi32>, i32) outs(%[[VADDRES:.*]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[VMINOP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>} ins(%[[VADDOP]], %[[POSONE]] : tensor<8192xi32>, i32) outs(%[[VADDOP]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[VMAXOP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%[[VMINOP]], %[[ZERO]] : tensor<8192xi32>, i32) outs(%[[VMINOP]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[TMPF:.*]] = tensor.empty() : tensor<8192xf32>
// CHECK: %[[CASTF:.*]] = hfusion.cast {{.*}} ins(%[[VMAXOP]] : tensor<8192xi32>) outs(%[[TMPF]] : tensor<8192xf32>) -> tensor<8192xf32>
// CHECK: %[[OUT1:.*]] = tensor.empty() : tensor<8192xi1>
// CHECK: %[[OUT2:.*]] = tensor.empty() : tensor<8192xi1>
// CHECK: %[[RES:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[CASTF]], %[[CST0:.*]] : tensor<8192xf32>, f32) outs(%[[OUT2]] : tensor<8192xi1>) -> tensor<8192xi1>
// CHECK: %[[RES2:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[RES]] : tensor<8192xi1>) outs(%[[OUT1]] : tensor<8192xi1>) -> tensor<8192xi1>
func.func @test_NormalizeIsNan_isnan() -> tensor<8192xi1> {
  %0 = tensor.empty() : tensor<8192xf32>
  %2 = hfusion.isnan %0 : tensor<8192xf32> -> tensor<8192xi1>
  return %2 : tensor<8192xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeIsNan_isnan_f16
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i16
// CHECK: %[[POSONE:.*]] = arith.constant 1 : i16
// CHECK: %[[NEGINF:.*]] = arith.constant -31744 : i16
// CHECK: %[[MASKVAL:.*]] = arith.constant 32767 : i16
// CHECK: %[[INPUT:.*]] = tensor.empty() : tensor<256xf16>
// CHECK: %[[VDUPOP:.*]] = linalg.fill ins(%[[MASKVAL]] : i16) outs(%[[MASKRES:.*]] : tensor<256xi16>) -> tensor<256xi16>
// CHECK: %[[BITCASTINPUT:.*]] = hfusion.bitcast ins(%[[INPUT]] : tensor<256xf16>) outs(%[[BITCASTOUT:.*]] : tensor<256xi16>) -> tensor<256xi16>
// CHECK: %[[VANDOP:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[BITCASTINPUT]], %[[VDUPOP]] : tensor<256xi16>, tensor<256xi16>) outs(%[[VANDOUTPUT:.*]] : tensor<256xi16>) -> tensor<256xi16>
// CHECK: %[[VADDOP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VANDOP]], %[[NEGINF]] : tensor<256xi16>, i16) outs(%[[VADDRES:.*]] : tensor<256xi16>) -> tensor<256xi16>
// CHECK: %[[VMINOP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>} ins(%[[VADDOP]], %[[POSONE]] : tensor<256xi16>, i16) outs(%[[MINOUT:.*]] : tensor<256xi16>) -> tensor<256xi16>
// CHECK: %[[VMAXOP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%[[VMINOP]], %[[ZERO]] : tensor<256xi16>, i16) outs(%[[MAXOUT:.*]] : tensor<256xi16>) -> tensor<256xi16>
// CHECK: %[[CASTF:.*]] = hfusion.cast {{.*}} ins(%[[VMAXOP]] : tensor<256xi16>) outs(%[[TMPF:.*]] : tensor<256xf16>) -> tensor<256xf16>
// CHECK: %[[RES:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[CASTF]], %[[CST0:.*]] : tensor<256xf16>, f16) outs(%[[OUT2:.*]] : tensor<256xi1>) -> tensor<256xi1>
// CHECK: %[[RES2:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[RES]] : tensor<256xi1>) outs(%[[OUT1:.*]] : tensor<256xi1>) -> tensor<256xi1>
func.func @test_NormalizeIsNan_isnan_f16() -> tensor<256xi1> {
  %0 = tensor.empty() : tensor<256xf16>
  %1 = hfusion.isnan %0 : tensor<256xf16> -> tensor<256xi1>
  return %1 : tensor<256xi1>
}

// -----

// CHECK-LABEL: func.func @test_NormalizeIsNan_isnan_bf16
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i16
// CHECK: %[[POSONE:.*]] = arith.constant 1 : i16
// CHECK: %[[NEGINF:.*]] = arith.constant -31744 : i16
// CHECK: %[[MASKVAL:.*]] = arith.constant 32767 : i16
// CHECK: %[[INPUT:.*]] = tensor.empty() : tensor<128xbf16>
// CHECK: %[[VDUPOP:.*]] = linalg.fill ins(%[[MASKVAL]] : i16) outs(%[[MASKRES:.*]] : tensor<128xi16>) -> tensor<128xi16>
// CHECK: %[[BITCASTINPUT:.*]] = hfusion.bitcast ins(%[[INPUT]] : tensor<128xbf16>) outs(%[[BITCASTOUT:.*]] : tensor<128xi16>) -> tensor<128xi16>
// CHECK: %[[VADDOP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%{{.*}}, %[[NEGINF]] : tensor<128xi16>, i16) outs(%[[VADDRES:.*]] : tensor<128xi16>) -> tensor<128xi16>
// CHECK: %[[VMINOP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>} ins(%[[VADDOP]], %[[POSONE]] : tensor<128xi16>, i16) outs(%[[MINOUT:.*]] : tensor<128xi16>) -> tensor<128xi16>
// CHECK: %[[VMAXOP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%[[VMINOP]], %[[ZERO]] : tensor<128xi16>, i16) outs(%[[MAXOUT:.*]] : tensor<128xi16>) -> tensor<128xi16>
// CHECK: return %[[RES:.*]] : tensor<128xi1>
func.func @test_NormalizeIsNan_isnan_bf16() -> tensor<128xi1> {
  %0 = tensor.empty() : tensor<128xbf16>
  %1 = hfusion.isnan %0 : tensor<128xbf16> -> tensor<128xi1>
  return %1 : tensor<128xi1>
}
