// RUN: bishengir-opt --hivm-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: memref.global "private" constant @tbl : memref<320xi32, #hivm.address_space<gm>>
// CHECK-LABEL: func.func @test_HighPrecisionNormalizeSin_hivm_vsin_f32(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-DAG: %[[QNAN:.*]] = arith.constant 0x7FC00000 : f32
// CHECK-DAG: %[[PI:.*]] = arith.constant 3.14159274 : f32
// CHECK: %[[TBL:.*]] = memref.get_global @tbl : memref<320xi32, #hivm.address_space<gm>>
// CHECK: %[[TO_TENSOR:.*]] = bufferization.to_tensor %{{.*}} restrict writable : memref<320xi32>
// CHECK: %[[BITCAST:.*]] = hivm.hir.bitcast %[[ARG0]] : tensor<4xf32> -> tensor<4xi32>
// CHECK: hivm.hir.vgather ins(%[[TO_TENSOR]] : tensor<320xi32>) indices(%{{.*}} : tensor<4xi32>) outs(%{{.*}} : tensor<4xi32>) -> tensor<4xi32>
// CHECK: hivm.hir.vgather ins(%[[TO_TENSOR]] : tensor<320xi32>) indices(%{{.*}} : tensor<4xi32>) outs(%{{.*}} : tensor<4xi32>) -> tensor<4xi32>
// XOR decomposed: vor + vand + vnot + vand
// CHECK: hivm.hir.vor ins(%{{.*}}, %{{.*}} : tensor<4xi32>, tensor<4xi32>) outs(%{{.*}} : tensor<4xi32>) -> tensor<4xi32>
// CHECK: hivm.hir.vand ins(%{{.*}}, %{{.*}} : tensor<4xi32>, tensor<4xi32>) outs(%{{.*}} : tensor<4xi32>) -> tensor<4xi32>
// CHECK: hivm.hir.vnot ins(%{{.*}} : tensor<4xi32>) outs(%{{.*}} : tensor<4xi32>) -> tensor<4xi32>
// CHECK: hivm.hir.vand ins(%{{.*}}, %{{.*}} : tensor<4xi32>, tensor<4xi32>) outs(%{{.*}} : tensor<4xi32>) -> tensor<4xi32>
// CHECK: %[[SIGN:.*]] = hivm.hir.vsel ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[REDUCED:.*]] = hivm.hir.vmul ins(%{{.*}}, %[[PI]] : tensor<4xf32>, f32) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[PI_MINUS:.*]] = hivm.hir.vsub ins(%[[PI]], %[[REDUCED]] : f32, tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: hivm.hir.vmin ins(%[[PI_MINUS]], %[[REDUCED]] : tensor<4xf32>, tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[SIGNED:.*]] = hivm.hir.vmul ins(%{{.*}}, %[[SIGN]] : tensor<4xf32>, tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[QNAN_T:.*]] = hivm.hir.vbrc ins(%[[QNAN]] : f32) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[RES:.*]] = hivm.hir.vsel ins(%{{.*}}, %[[QNAN_T]], %[[SIGNED]] : tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: return %[[RES]] : tensor<4xf32>
// CHECK-NOT: hivm.hir.vsin
func.func @test_HighPrecisionNormalizeSin_hivm_vsin_f32(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = tensor.empty() : tensor<4xf32>
  %1 = hivm.hir.vsin ins(%arg0 : tensor<4xf32>) outs(%0 : tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// CHECK: memref.global "private" constant @tbl : memref<320xi32, #hivm.address_space<gm>>
// CHECK-LABEL: func.func @test_HighPrecisionNormalizeSin_hivm_vsin_f16(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4xf16>) -> tensor<4xf16> {
// CHECK: %[[EMPTY_F32:.*]] = tensor.empty() : tensor<4xf32>
// CHECK: %[[CAST_IN:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<4xf16>) outs(%[[EMPTY_F32]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[TBL:.*]] = memref.get_global @tbl : memref<320xi32, #hivm.address_space<gm>>
// CHECK: %[[BITCAST:.*]] = hivm.hir.bitcast %[[CAST_IN]] : tensor<4xf32> -> tensor<4xi32>
// CHECK: hivm.hir.vgather ins(%{{.*}} : tensor<320xi32>) indices(%{{.*}} : tensor<4xi32>) outs(%{{.*}} : tensor<4xi32>) -> tensor<4xi32>
// XOR decomposed: vor + vand + vnot + vand
// CHECK: hivm.hir.vor ins(%{{.*}}, %{{.*}} : tensor<4xi32>, tensor<4xi32>) outs(%{{.*}} : tensor<4xi32>) -> tensor<4xi32>
// CHECK: hivm.hir.vand ins(%{{.*}}, %{{.*}} : tensor<4xi32>, tensor<4xi32>) outs(%{{.*}} : tensor<4xi32>) -> tensor<4xi32>
// CHECK: hivm.hir.vnot ins(%{{.*}} : tensor<4xi32>) outs(%{{.*}} : tensor<4xi32>) -> tensor<4xi32>
// CHECK: hivm.hir.vand ins(%{{.*}}, %{{.*}} : tensor<4xi32>, tensor<4xi32>) outs(%{{.*}} : tensor<4xi32>) -> tensor<4xi32>
// CHECK: %[[EMPTY_F16:.*]] = tensor.empty() : tensor<4xf16>
// CHECK-NEXT: %[[CAST_OUT:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<4xf32>) outs(%[[EMPTY_F16]] : tensor<4xf16>) round_mode = <round> -> tensor<4xf16>
// CHECK-NEXT: return %[[CAST_OUT]] : tensor<4xf16>
// CHECK-NOT: hivm.hir.vsin
func.func @test_HighPrecisionNormalizeSin_hivm_vsin_f16(%arg0 : tensor<4xf16>) -> tensor<4xf16> {
  %0 = tensor.empty() : tensor<4xf16>
  %1 = hivm.hir.vsin ins(%arg0 : tensor<4xf16>) outs(%0 : tensor<4xf16>) -> tensor<4xf16>
  return %1 : tensor<4xf16>
}

// -----

// CHECK: memref.global "private" constant @tbl : memref<320xi32, #hivm.address_space<gm>>
// CHECK-LABEL: func.func @test_HighPrecisionNormalizeCos_hivm_vcos_f32(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-DAG: %[[QNAN:.*]] = arith.constant 0x7FC00000 : f32
// CHECK-DAG: %[[PIO2:.*]] = arith.constant 1.57079637 : f32
// CHECK-DAG: %[[PI:.*]] = arith.constant 3.14159274 : f32
// CHECK: %[[TBL:.*]] = memref.get_global @tbl : memref<320xi32, #hivm.address_space<gm>>
// CHECK: %[[TO_TENSOR:.*]] = bufferization.to_tensor %{{.*}} restrict writable : memref<320xi32>
// CHECK: %[[BITCAST:.*]] = hivm.hir.bitcast %[[ARG0]] : tensor<4xf32> -> tensor<4xi32>
// CHECK: hivm.hir.vgather ins(%[[TO_TENSOR]] : tensor<320xi32>) indices(%{{.*}} : tensor<4xi32>) outs(%{{.*}} : tensor<4xi32>) -> tensor<4xi32>
// CHECK: hivm.hir.vgather ins(%[[TO_TENSOR]] : tensor<320xi32>) indices(%{{.*}} : tensor<4xi32>) outs(%{{.*}} : tensor<4xi32>) -> tensor<4xi32>
// CHECK: %[[SIGN:.*]] = hivm.hir.vsel ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[REDUCED:.*]] = hivm.hir.vmul ins(%{{.*}}, %[[PI]] : tensor<4xf32>, f32) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[SHIFT:.*]] = hivm.hir.vsub ins(%[[PIO2]], %[[REDUCED]] : f32, tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[SIGNED:.*]] = hivm.hir.vmul ins(%{{.*}}, %[[SIGN]] : tensor<4xf32>, tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[QNAN_T:.*]] = hivm.hir.vbrc ins(%[[QNAN]] : f32) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[RES:.*]] = hivm.hir.vsel ins(%{{.*}}, %[[QNAN_T]], %[[SIGNED]] : tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: return %[[RES]] : tensor<4xf32>
// CHECK-NOT: hivm.hir.vcos
func.func @test_HighPrecisionNormalizeCos_hivm_vcos_f32(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = tensor.empty() : tensor<4xf32>
  %1 = hivm.hir.vcos ins(%arg0 : tensor<4xf32>) outs(%0 : tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// CHECK: memref.global "private" constant @tbl : memref<320xi32, #hivm.address_space<gm>>
// CHECK-LABEL: func.func @test_HighPrecisionNormalizeCos_hivm_vcos_f16(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4xf16>) -> tensor<4xf16> {
// CHECK: %[[EMPTY_F32:.*]] = tensor.empty() : tensor<4xf32>
// CHECK: %[[CAST_IN:.*]] = hivm.hir.vcast ins(%[[ARG0]] : tensor<4xf16>) outs(%[[EMPTY_F32]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[TBL:.*]] = memref.get_global @tbl : memref<320xi32, #hivm.address_space<gm>>
// CHECK: %[[BITCAST:.*]] = hivm.hir.bitcast %[[CAST_IN]] : tensor<4xf32> -> tensor<4xi32>
// CHECK: hivm.hir.vgather ins(%{{.*}} : tensor<320xi32>) indices(%{{.*}} : tensor<4xi32>) outs(%{{.*}} : tensor<4xi32>) -> tensor<4xi32>
// CHECK: %[[EMPTY_F16:.*]] = tensor.empty() : tensor<4xf16>
// CHECK-NEXT: %[[CAST_OUT:.*]] = hivm.hir.vcast ins(%{{.*}} : tensor<4xf32>) outs(%[[EMPTY_F16]] : tensor<4xf16>) round_mode = <round> -> tensor<4xf16>
// CHECK-NEXT: return %[[CAST_OUT]] : tensor<4xf16>
// CHECK-NOT: hivm.hir.vcos
func.func @test_HighPrecisionNormalizeCos_hivm_vcos_f16(%arg0 : tensor<4xf16>) -> tensor<4xf16> {
  %0 = tensor.empty() : tensor<4xf16>
  %1 = hivm.hir.vcos ins(%arg0 : tensor<4xf16>) outs(%0 : tensor<4xf16>) -> tensor<4xf16>
  return %1 : tensor<4xf16>
}
