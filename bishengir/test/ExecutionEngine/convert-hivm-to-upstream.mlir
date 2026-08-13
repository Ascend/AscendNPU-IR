// REQUIRES: execution-engine
// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_9579 --execution-engine-convert-hivm-to-upstream="convert-to-named-op=true" %s --split-input-file | FileCheck %s --check-prefixes=COMMON,CHECK-TRUE
// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_9579 --execution-engine-convert-hivm-to-upstream="convert-to-named-op=false" %s --split-input-file | FileCheck %s --check-prefixes=COMMON,CHECK-FALSE
// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_9579 --execution-engine-convert-hivm-to-upstream="convert-to-named-op=true" --hfusion-generalize %s --split-input-file | FileCheck %s --check-prefix=GENERALIZE

// -----

func.func @tensor_direct_linalg_lowering(%a: tensor<1x?x10xf32>, %b: tensor<?x5x10xf32>, %c: tensor<5x?x10xf32>, %d: tensor<5x?x10xf32>) -> (tensor<5x?x10xf32>, tensor<5x?x10xf32>) {

    // CHECK-TRUE: linalg.abs
    // CHECK-FALSE: linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
    %0 = hivm.hir.vabs ins(%a: tensor<1x?x10xf32>) outs(%c: tensor<5x?x10xf32>) broadcast = [0] -> tensor<5x?x10xf32>

    // CHECK-TRUE: linalg.add
    // CHECK-FALSE: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %d, %c0 : tensor<5x?x10xf32>
    %empty0 = tensor.empty(%dim) : tensor<5x?x10xf32>
    %1 = hivm.hir.vadd ins(%b, %b: tensor<?x5x10xf32>, tensor<?x5x10xf32>) outs(%empty0: tensor<5x?x10xf32>) transpose = [1, 0, 2] -> tensor<5x?x10xf32>

    // CHECK-TRUE: linalg.sub
    // CHECK-FALSE: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
    %2 = hivm.hir.vsub ins(%0, %1: tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK-TRUE: linalg.mul
    // CHECK-FALSE: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
    %3 = hivm.hir.vmul ins(%1, %2: tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK-TRUE: linalg.div
    // CHECK-FALSE: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
    %4 = hivm.hir.vdiv ins(%2, %3: tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // COMMON: linalg.max
    %5 = hivm.hir.vmax ins(%3, %4: tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // COMMON: linalg.min
    %6 = hivm.hir.vmin ins(%4, %5: tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK-TRUE: linalg.exp
    // CHECK-FALSE: linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
    %7 = hivm.hir.vexp ins(%6: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK-TRUE: linalg.log
    // CHECK-FALSE: linalg.elemwise_unary {fun = #linalg.unary_fn<log>}
    %8 = hivm.hir.vln ins(%7: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK-TRUE: linalg.rsqrt
    // CHECK-FALSE: linalg.elemwise_unary {fun = #linalg.unary_fn<rsqrt>}
    %9 = hivm.hir.vrsqrt ins(%8: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK-TRUE: linalg.sqrt
    // CHECK-FALSE: linalg.elemwise_unary {fun = #linalg.unary_fn<sqrt>}
    %10 = hivm.hir.vsqrt ins(%9: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK-TRUE: linalg.tanh
    // CHECK-FALSE: linalg.elemwise_unary {fun = #linalg.unary_fn<tanh>}
    %11 = hivm.hir.vtanh ins(%10: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK-TRUE: linalg.reciprocal
    // CHECK-FALSE: linalg.elemwise_unary {fun = #linalg.unary_fn<reciprocal>}
    %empty12 = tensor.empty(%dim) : tensor<5x?x10xf32>
    %12 = hivm.hir.vrec ins(%11: tensor<5x?x10xf32>) outs(%empty12: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK-TRUE: linalg.select
    // CHECK-FALSE: hfusion.select
    %13 = arith.constant true
    %empty14 = tensor.empty(%dim) : tensor<5x?x10xf32>
    %14 = hivm.hir.vsel ins(%13, %12, %c: i1, tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%empty14: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK-TRUE: linalg.erf
    // CHECK-FALSE: linalg.elemwise_unary {fun = #linalg.unary_fn<erf>}
    %15 = hivm.hir.verf ins(%14: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>
 
    // COMMON: hivm.hir.store
    %16 = hivm.hir.store ins(%15: tensor<5x?x10xf32>) outs(%d: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // COMMON: linalg.transpose
    %17 = hivm.hir.vtranspose ins(%b: tensor<?x5x10xf32>) outs(%16: tensor<5x?x10xf32>) permutation = [1, 0, 2] -> tensor<5x?x10xf32>

    return %16, %17: tensor<5x?x10xf32>, tensor<5x?x10xf32>
}

// -----

func.func @memref_direct_linalg_lowering(%a: memref<1x?x10xf32>, %b: memref<?x5x10xf32>, %c: memref<5x?x10xf32>, %d: memref<5x?x10xf32>, %e: memref<1x?x10xi8>, %f: memref<5x?x10xi8>) {

    // CHECK-TRUE: linalg.abs
    // CHECK-FALSE: linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
    hivm.hir.vabs ins(%a: memref<1x?x10xf32>) outs(%c: memref<5x?x10xf32>) broadcast = [0]

    // CHECK-TRUE: linalg.add
    // CHECK-FALSE: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
    hivm.hir.vadd ins(%b, %b: memref<?x5x10xf32>, memref<?x5x10xf32>) outs(%c: memref<5x?x10xf32>) transpose = [1, 0, 2]

    // CHECK-TRUE: linalg.sub
    // CHECK-FALSE: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
    hivm.hir.vsub ins(%c, %d: memref<5x?x10xf32>, memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK-TRUE: linalg.mul
    // CHECK-FALSE: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
    hivm.hir.vmul ins(%c, %d: memref<5x?x10xf32>, memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK-TRUE: linalg.div
    // CHECK-FALSE: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
    hivm.hir.vdiv ins(%c, %d: memref<5x?x10xf32>, memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // COMMON: linalg.max
    hivm.hir.vmax ins(%c, %d: memref<5x?x10xf32>, memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // COMMON: linalg.min
    hivm.hir.vmin ins(%c, %d: memref<5x?x10xf32>, memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK-TRUE: linalg.exp
    // CHECK-FALSE: linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
    hivm.hir.vexp ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK-TRUE: linalg.log
    // CHECK-FALSE: linalg.elemwise_unary {fun = #linalg.unary_fn<log>}
    hivm.hir.vln ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK-TRUE: linalg.rsqrt
    // CHECK-FALSE: linalg.elemwise_unary {fun = #linalg.unary_fn<rsqrt>}
    hivm.hir.vrsqrt ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK-TRUE: linalg.sqrt
    // CHECK-FALSE: linalg.elemwise_unary {fun = #linalg.unary_fn<sqrt>}
    hivm.hir.vsqrt ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK-TRUE: linalg.tanh
    // CHECK-FALSE: linalg.elemwise_unary {fun = #linalg.unary_fn<tanh>}
    hivm.hir.vtanh ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK-TRUE: linalg.reciprocal
    // CHECK-FALSE: linalg.elemwise_unary {fun = #linalg.unary_fn<reciprocal>}
    hivm.hir.vrec ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK-TRUE: linalg.erf
    // CHECK-FALSE: linalg.elemwise_unary {fun = #linalg.unary_fn<erf>}
    hivm.hir.verf ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    hivm.hir.store ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // COMMON: linalg.transpose
    hivm.hir.vtranspose ins(%b: memref<?x5x10xf32>) outs(%c: memref<5x?x10xf32>) permutation = [1, 0, 2]

    func.return
}

// -----

// COMMON-LABEL: func.func @atomic_cas_bf16_to_upstream
// COMMON: hfusion.atomic_cas
func.func @atomic_cas_bf16_to_upstream(%src0: tensor<1xbf16>, %src1: tensor<1xbf16>, %dst: tensor<1xbf16>) -> tensor<1xbf16> {
  %0 = hivm.hir.atomic_cas ins(%src0, %src1 : tensor<1xbf16>, tensor<1xbf16>) outs(%dst : tensor<1xbf16>) -> tensor<1xbf16>
  return %0 : tensor<1xbf16>
}

// -----

// COMMON-LABEL: func.func @atomic_cas_fp8_e4m3_to_upstream
// COMMON: hfusion.atomic_cas
func.func @atomic_cas_fp8_e4m3_to_upstream(%src0: tensor<1xf8E4M3FN>, %src1: tensor<1xf8E4M3FN>, %dst: tensor<1xf8E4M3FN>) -> tensor<1xf8E4M3FN> {
  %0 = hivm.hir.atomic_cas ins(%src0, %src1 : tensor<1xf8E4M3FN>, tensor<1xf8E4M3FN>) outs(%dst : tensor<1xf8E4M3FN>) -> tensor<1xf8E4M3FN>
  return %0 : tensor<1xf8E4M3FN>
}

// -----

// COMMON-LABEL: func.func @atomic_cas_fp8_e5m2_to_upstream
// COMMON: hfusion.atomic_cas
func.func @atomic_cas_fp8_e5m2_to_upstream(%src0: tensor<1xf8E5M2>, %src1: tensor<1xf8E5M2>, %dst: tensor<1xf8E5M2>) -> tensor<1xf8E5M2> {
  %0 = hivm.hir.atomic_cas ins(%src0, %src1 : tensor<1xf8E5M2>, tensor<1xf8E5M2>) outs(%dst : tensor<1xf8E5M2>) -> tensor<1xf8E5M2>
  return %0 : tensor<1xf8E5M2>
}


// -----

func.func @elemwise_lowering(%a: tensor<?x5x10xf32>, %aT: tensor<5x?x10xf32>, %b: memref<5x1x10xi32>, %bB: memref<5x?x10xi32>) -> tensor<5x?x10xf32> {

    // COMMON: hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>}
    %0 = hivm.hir.vrelu ins(%a: tensor<?x5x10xf32>) outs(%aT: tensor<5x?x10xf32>) transpose = [1, 0, 2] -> tensor<5x?x10xf32>

    // COMMON: hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>}
    hivm.hir.vrelu ins(%b: memref<5x1x10xi32>) outs(%bB: memref<5x?x10xi32>) broadcast = [1]

    // COMMON: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
    hivm.hir.vnot ins(%b: memref<5x1x10xi32>) outs(%bB: memref<5x?x10xi32>) broadcast = [1]

    func.return %0: tensor<5x?x10xf32>
}

// -----

// COMMON-LABEL: func.func @bitwise_like_lowering
func.func @bitwise_like_lowering(%a: tensor<?x5x10xf32>, %aT: tensor<5x?x10xf32>, %b: memref<5x1x10xi32>, %bB: memref<5x?x10xi32>) -> tensor<5x?x10xf32>
{
    // COMMON-NOT: linalg.map
    // COMMON: linalg.generic {{.*}}
    // COMMON: arith.bitcast
    // COMMON: arith.andi
    // COMMON: arith.bitcast
    %0 = hivm.hir.vand ins(%a, %a: tensor<?x5x10xf32>, tensor<?x5x10xf32>) outs(%aT: tensor<5x?x10xf32>) transpose = [1, 0, 2] -> tensor<5x?x10xf32>

    // COMMON: linalg.generic {{.*}}
    // COMMON: arith.andi
    hivm.hir.vand ins(%b, %b: memref<5x1x10xi32>, memref<5x1x10xi32>) outs(%bB: memref<5x?x10xi32>) broadcast = [1]

    // COMMON: linalg.generic {{.*}}
    // COMMON: arith.bitcast
    // COMMON: arith.ori
    // COMMON: arith.bitcast
    %1 = hivm.hir.vor ins(%a, %a: tensor<?x5x10xf32>, tensor<?x5x10xf32>) outs(%0: tensor<5x?x10xf32>) transpose = [1, 0, 2] -> tensor<5x?x10xf32>

    // COMMON: linalg.generic {{.*}}
    // COMMON: arith.ori
    hivm.hir.vor ins(%b, %b: memref<5x1x10xi32>, memref<5x1x10xi32>) outs(%bB: memref<5x?x10xi32>) broadcast = [1]

    // COMMON: hfusion.elemwise_binary {fun = #hfusion.binary_fn<vxor>}
    hivm.hir.vxor ins(%bB, %bB: memref<5x?x10xi32>, memref<5x?x10xi32>) outs(%bB: memref<5x?x10xi32>)

    // COMMON: hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>}
    hivm.hir.vshr {is_signed = true} ins(%bB, %bB: memref<5x?x10xi32>, memref<5x?x10xi32>) outs(%bB: memref<5x?x10xi32>)

    // COMMON: hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>}
    hivm.hir.vshr {is_signed = false} ins(%bB, %bB: memref<5x?x10xi32>, memref<5x?x10xi32>) outs(%bB: memref<5x?x10xi32>)

    // COMMON-NOT: linalg.map
    func.return %1: tensor<5x?x10xf32>
}

// -----

// COMMON-LABEL: func.func @scalar_vbrc_add_mul_lowering(
// COMMON: %[[SCALAR:.*]] = arith.constant 2.000000e+00 : f32
// COMMON-NOT: linalg.fill
// COMMON-NOT: linalg.add
// COMMON-NOT: linalg.mul
// COMMON: %[[ADD:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%{{.*}}, %[[SCALAR]] : tensor<64xf32>, f32)
// COMMON-NOT: linalg.fill
// COMMON: %[[MUL:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ADD]], %[[SCALAR]] : tensor<64xf32>, f32)
// COMMON-NOT: linalg.fill
// COMMON: return %[[MUL]]
func.func @scalar_vbrc_add_mul_lowering(%arg0: tensor<64xf32>) -> tensor<64xf32> {
  %scalar = arith.constant 2.000000e+00 : f32
  %brc_empty = tensor.empty() : tensor<64xf32>
  %brc = hivm.hir.vbrc ins(%scalar : f32) outs(%brc_empty : tensor<64xf32>) -> tensor<64xf32>
  %add_empty = tensor.empty() : tensor<64xf32>
  %add = hivm.hir.vadd ins(%arg0, %brc : tensor<64xf32>, tensor<64xf32>) outs(%add_empty : tensor<64xf32>) -> tensor<64xf32>
  %mul_empty = tensor.empty() : tensor<64xf32>
  %mul = hivm.hir.vmul ins(%add, %brc : tensor<64xf32>, tensor<64xf32>) outs(%mul_empty : tensor<64xf32>) -> tensor<64xf32>
  return %mul : tensor<64xf32>
}

// -----

// COMMON-LABEL: func.func @scalar_vbrc_vand_lowering(
// COMMON: %[[MASK:.*]] = arith.constant 2147483647 : i32
// COMMON-NOT: linalg.fill
// COMMON: %[[VAND:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%{{.*}}, %[[MASK]] : tensor<64xi32>, i32)
// COMMON-NOT: linalg.fill
// COMMON: return %[[VAND]]
func.func @scalar_vbrc_vand_lowering(%arg0: tensor<64xi32>) -> tensor<64xi32> {
  %mask = arith.constant 2147483647 : i32
  %brc_empty = tensor.empty() : tensor<64xi32>
  %brc = hivm.hir.vbrc ins(%mask : i32) outs(%brc_empty : tensor<64xi32>) -> tensor<64xi32>
  %result_empty = tensor.empty() : tensor<64xi32>
  %result = hivm.hir.vand ins(%arg0, %brc : tensor<64xi32>, tensor<64xi32>) outs(%result_empty : tensor<64xi32>) -> tensor<64xi32>
  return %result : tensor<64xi32>
}

// -----

// COMMON-LABEL: func.func @scalar_vbrc_vor_lowering(
// COMMON: %[[MASK:.*]] = arith.constant 2147483647 : i32
// COMMON-NOT: linalg.fill
// COMMON-NOT: linalg.map
// COMMON: %[[VOR:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%{{.*}}, %[[MASK]] : tensor<64xi32>, i32)
// COMMON-NOT: linalg.fill
// COMMON-NOT: linalg.map
// COMMON: return %[[VOR]]
func.func @scalar_vbrc_vor_lowering(%arg0: tensor<64xi32>) -> tensor<64xi32> {
  %mask = arith.constant 2147483647 : i32
  %brc_empty = tensor.empty() : tensor<64xi32>
  %brc = hivm.hir.vbrc ins(%mask : i32) outs(%brc_empty : tensor<64xi32>) -> tensor<64xi32>
  %result_empty = tensor.empty() : tensor<64xi32>
  %result = hivm.hir.vor ins(%arg0, %brc : tensor<64xi32>, tensor<64xi32>) outs(%result_empty : tensor<64xi32>) -> tensor<64xi32>
  return %result : tensor<64xi32>
}

// -----

// COMMON-LABEL: func.func @integer_bitwise_same_shape_lowering
// COMMON-NOT: linalg.map
// COMMON-NOT: linalg.generic
// COMMON: %[[VAND:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
// COMMON: %[[VOR:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>}
// COMMON: %[[VXOR:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vxor>}
// COMMON-NOT: linalg.map
// COMMON-NOT: linalg.generic
// COMMON: return %[[VXOR]]
func.func @integer_bitwise_same_shape_lowering(%lhs: tensor<32xi32>, %rhs: tensor<32xi32>, %dst: tensor<32xi32>) -> tensor<32xi32> {
  %0 = hivm.hir.vand ins(%lhs, %rhs : tensor<32xi32>, tensor<32xi32>) outs(%dst : tensor<32xi32>) -> tensor<32xi32>
  %1 = hivm.hir.vor ins(%0, %rhs : tensor<32xi32>, tensor<32xi32>) outs(%dst : tensor<32xi32>) -> tensor<32xi32>
  %2 = hivm.hir.vxor ins(%1, %rhs : tensor<32xi32>, tensor<32xi32>) outs(%dst : tensor<32xi32>) -> tensor<32xi32>
  return %2 : tensor<32xi32>
}

// -----

// COMMON-LABEL: func.func @unsigned_vand_same_shape_lowering
// COMMON-NOT: arith.andi
// COMMON: %[[VAND:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
// COMMON: return %[[VAND]]
func.func @unsigned_vand_same_shape_lowering(%lhs: tensor<32xui32>, %rhs: tensor<32xui32>, %dst: tensor<32xui32>) -> tensor<32xui32> {
  %0 = hivm.hir.vand ins(%lhs, %rhs : tensor<32xui32>, tensor<32xui32>) outs(%dst : tensor<32xui32>) -> tensor<32xui32>
  return %0 : tensor<32xui32>
}

// -----

// COMMON-LABEL: func.func @fp_bitwise_same_shape_lowering
// COMMON-NOT: linalg.map
// COMMON: %[[VAND:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
// COMMON: %[[VOR:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>}
// COMMON-NOT: linalg.map
// COMMON: return %[[VOR]]
// GENERALIZE-LABEL: func.func @fp_bitwise_same_shape_lowering
// GENERALIZE-NOT: linalg.map
// GENERALIZE: linalg.generic
// GENERALIZE: arith.andi
// GENERALIZE: linalg.generic
// GENERALIZE: arith.ori
// GENERALIZE-NOT: hfusion.elemwise_binary
// GENERALIZE: return
func.func @fp_bitwise_same_shape_lowering(%lhs: tensor<32xf32>, %rhs: tensor<32xf32>, %dst: tensor<32xf32>) -> tensor<32xf32> {
  %0 = hivm.hir.vand ins(%lhs, %rhs : tensor<32xf32>, tensor<32xf32>) outs(%dst : tensor<32xf32>) -> tensor<32xf32>
  %1 = hivm.hir.vor ins(%0, %rhs : tensor<32xf32>, tensor<32xf32>) outs(%dst : tensor<32xf32>) -> tensor<32xf32>
  return %1 : tensor<32xf32>
}

// -----

// COMMON-LABEL: func.func @fp16_vand_vv_broadcast_lowering
// COMMON-NOT: linalg.map
// COMMON: linalg.generic
// COMMON: arith.bitcast {{.*}} : f16 to i16
// COMMON: arith.bitcast {{.*}} : f16 to i16
// COMMON: arith.andi
// COMMON: arith.bitcast {{.*}} : i16 to f16
// COMMON-NOT: linalg.map
func.func @fp16_vand_vv_broadcast_lowering(%lhs: tensor<2x1x4xf16>, %rhs: tensor<1x3x4xf16>, %dst: tensor<2x3x4xf16>) -> tensor<2x3x4xf16> {
  %0 = hivm.hir.vand ins(%lhs, %rhs : tensor<2x1x4xf16>, tensor<1x3x4xf16>) outs(%dst : tensor<2x3x4xf16>) broadcast = [0, 1] -> tensor<2x3x4xf16>
  return %0 : tensor<2x3x4xf16>
}

// -----

// COMMON-LABEL: func.func @bf16_vor_vv_broadcast_lowering
// COMMON-NOT: linalg.map
// COMMON: linalg.generic
// COMMON: arith.bitcast {{.*}} : bf16 to i16
// COMMON: arith.bitcast {{.*}} : bf16 to i16
// COMMON: arith.ori
// COMMON: arith.bitcast {{.*}} : i16 to bf16
// COMMON-NOT: linalg.map
func.func @bf16_vor_vv_broadcast_lowering(%lhs: tensor<2x1x4xbf16>, %rhs: tensor<1x3x4xbf16>, %dst: tensor<2x3x4xbf16>) -> tensor<2x3x4xbf16> {
  %0 = hivm.hir.vor ins(%lhs, %rhs : tensor<2x1x4xbf16>, tensor<1x3x4xbf16>) outs(%dst : tensor<2x3x4xbf16>) broadcast = [0, 1] -> tensor<2x3x4xbf16>
  return %0 : tensor<2x3x4xbf16>
}

// -----

func.func @cumulative_like_lowering(%a: tensor<5x?x10xf32>, %b: memref<5x?x10xi32>) -> tensor<5x?x10xf32> {

    // CHECK-FALSE: linalg.generic	 
    // CHECK-FALSE-SAME:  outs({{.*}}: tensor<5x?x10xf32>, tensor<5x1x1xf32>) 
    // CHECK-FALSE-NEXT:  ^bb0(%[[in:.*]]: f32, %{{.*}}: f32, %[[out:.*]]: f32) 
    // CHECK-FALSE-NEXT:      %[[res:.*]] = arith.mulf 
    // CHECK-FALSE-DAG-SAME:      %[[in]] 
    // CHECK-FALSE-DAG-SAME:      %[[out]] 
    // CHECK-FALSE-NEXT:      linalg.yield %[[res]], %[[res]] 
    // CHECK-TRUE: %0 = hfusion.cumprod %arg0 : tensor<5x?x10xf32> cum_dims = [0] reverse = false -> tensor<5x?x10xf32>
    %0 = hivm.hir.vcumprod ins(%a: tensor<5x?x10xf32>) outs(%a: tensor<5x?x10xf32>) cum_dims = [0] reverse = false -> tensor<5x?x10xf32>	 

    // CHECK-FALSE: linalg.generic	 
    // CHECK-FALSE-SAME:  outs({{.*}}: memref<5x?x10xi32>, memref<5x?x1xi32>)	 
    // CHECK-FALSE-NEXT:  ^bb0(%[[in:.*]]: i32, %{{.*}}: i32, %[[out:.*]]: i32) 
    // CHECK-FALSE-NEXT:      %[[res:.*]] = arith.muli 
    // CHECK-FALSE-DAG-SAME:      %[[in]] 
    // CHECK-FALSE-DAG-SAME:      %[[out]] 
    // CHECK-FALSE-NEXT:      linalg.yield %[[res]], %[[res]]
    // CHECK-TRUE-NOT: hfusion.cumprod %b : memref<5x?x10xi32> cum_dims = [1] reverse = false -> memref<5x?x10xi32>
    hivm.hir.vcumprod ins(%b: memref<5x?x10xi32>) outs(%b: memref<5x?x10xi32>) cum_dims = [1] reverse = false	 

    // CHECK-FALSE: linalg.generic	 
    // CHECK-FALSE-SAME:  outs({{.*}}: memref<5x?x10xi32>, memref<5x?x1xi32>) 
    // CHECK-FALSE-NEXT:  ^bb0(%[[in:.*]]: i32, %{{.*}}: i32, %[[out:.*]]: i32) 
    // CHECK-FALSE-NEXT:      %[[res:.*]] = arith.addi 
    // CHECK-FALSE-DAG-SAME:      %[[in]] 
    // CHECK-FALSE-DAG-SAME:      %[[out]] 
    // CHECK-FALSE-NEXT:      linalg.yield %[[res]], %[[res]]
    // CHECK-TRUE-NOT: hfusion.cumsum %b : memref<5x?x10xi32> cum_dims = [1] reverse = false -> memref<5x?x10xi32>
    hivm.hir.vcumsum ins(%b: memref<5x?x10xi32>) outs(%b: memref<5x?x10xi32>) cum_dims = [1] reverse = false

    func.return %0: tensor<5x?x10xf32>
}

// -----

func.func @arange_lowering(%a: tensor<5x?x10xi64>, %b: memref<5x?x10xi32>) -> tensor<5x?x10xi64> {

    // COMMON: %[[C0:.*]] = arith.constant 0
    %c0 = arith.constant 0: index
    // COMMON: %[[C1:.*]] = arith.constant 1
    %c1 = arith.constant 1: index
    // COMMON: %[[C2:.*]] = arith.constant 2
    %c2 = arith.constant 2: index
    // COMMON: %[[C3:.*]] = arith.constant 3
    %c3 = arith.constant 3: index

    // COMMON: hfusion.arange
    // COMMON-SAME:  strides[%[[C0:.*]], %[[C3:.*]], %[[C2:.*]]]
    %0 = hivm.hir.varange offset[] strides[%c0, %c3, %c2] outs(%a: tensor<5x?x10xi64>) -> tensor<5x?x10xi64>

    // COMMON: hfusion.arange offset[%[[C3]]] strides[%[[C1]], %[[C1]], %[[C1]]]
    hivm.hir.varange offset[%c3] strides[%c1, %c1, %c1] outs(%b: memref<5x?x10xi32>)

    func.return %0: tensor<5x?x10xi64>
}

// -----

// COMMON-LABEL: @concat_lowering
// COMMON-SAME:      %[[a:[^:]*]]: {{[^,]*}}, 
// COMMON-SAME:      %[[b:[^:]*]]: {{[^,]*}}, 
// COMMON-SAME:      %[[c:[^:]*]]: {{[^,]*}}, 
// COMMON-SAME:      %[[d:[^:]*]]: {{[^,]*}}, 
// COMMON-SAME:      %[[e:[^:]*]]: {{[^,]*}}, 
// COMMON-SAME:      %[[f:[^:]*]]: {{[^,]*}}
func.func @concat_lowering(%a: tensor<5x?x10xf32>, %b: tensor<?x?x10xf32>, %c: tensor<?x?x10xf32>, %d: memref<5x?x10xi32>, %e: memref<?x?x10xi32>, %f: memref<?x?x10xi32>) -> tensor<?x?x10xf32> {

    // COMMON: tensor.concat
    %0 = hivm.hir.vconcat dim(0) ins(%a, %b: tensor<5x?x10xf32>, tensor<?x?x10xf32>) outs(%c: tensor<?x?x10xf32>) -> tensor<?x?x10xf32>

    // COMMON-DAG: %[[tensorD:.*]] = bufferization.to_tensor %[[d]]
    // COMMON-DAG: %[[tensorE:.*]] = bufferization.to_tensor %[[e]]
    // COMMON:   %[[concat:.*]] = tensor.concat dim(0)
    // COMMON-DAG-SAME:                      %[[tensorD]]
    // COMMON-DAG-SAME:                      %[[tensorE]]
    // COMMON:   bufferization.materialize_in_destination %[[concat]]
    // COMMON-SAME:                                          %[[f]]
    hivm.hir.vconcat dim(0) ins(%d, %e: memref<5x?x10xi32>, memref<?x?x10xi32>) outs(%f: memref<?x?x10xi32>)

    func.return %0: tensor<?x?x10xf32>
}

// -----
func.func @vcat_lowering() {
    %f16 = memref.alloc() : memref<2x16xf16>
    %f32 = memref.alloc() : memref<2x16xf32>
    %s16 = memref.alloc() : memref<2x16xi16>
    %s32 = memref.alloc() : memref<2x16xi32>
    %s64 = memref.alloc() : memref<2x16xi64>
    %bf16 = memref.alloc() : memref<2x16xbf16>
    %u8 = memref.alloc() : memref<2x16xui8>
    // COMMON: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>}
    hivm.hir.vcast ins(%bf16 : memref<2x16xbf16>) outs(%f32 : memref<2x16xf32>)
                   round_mode = #hivm.round_mode<rint>
    // COMMON: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<round>}
    hivm.hir.vcast ins(%bf16 : memref<2x16xbf16>) outs(%s32 : memref<2x16xi32>)
                   round_mode = #hivm.round_mode<round>
    // COMMON: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<ceil>}
    hivm.hir.vcast ins(%bf16 : memref<2x16xbf16>) outs(%s32 : memref<2x16xi32>)
                   round_mode = #hivm.round_mode<ceil>
    // COMMON: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<floor>}
    hivm.hir.vcast ins(%bf16 : memref<2x16xbf16>) outs(%s32 : memref<2x16xi32>)
                   round_mode = #hivm.round_mode<floor>
    return
}

// -----

// COMMON-LABEL: func.func @vmp_lowering(
// COMMON: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
// COMMON: hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
// COMMON: hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}
// COMMON: hfusion.compare {compare_fn = #hfusion.compare_fn<vle>}
// COMMON: hfusion.compare {compare_fn = #hfusion.compare_fn<vgt>}
// COMMON: hfusion.compare {compare_fn = #hfusion.compare_fn<vge>}
func.func @vmp_lowering(
    %a_f32: tensor<4xf32>,
    %b_f32: tensor<4xf32>,
    %sink: memref<4xi1>) {
  %init = tensor.empty() : tensor<4xi1>
  %eq = hivm.hir.vcmp
          ins(%a_f32, %b_f32 : tensor<4xf32>, tensor<4xf32>)
          outs(%init : tensor<4xi1>)
          compare_mode = #hivm.compare_mode<eq> -> tensor<4xi1>
  %ne = hivm.hir.vcmp
          ins(%a_f32, %b_f32 : tensor<4xf32>, tensor<4xf32>)
          outs(%init : tensor<4xi1>)
          compare_mode = #hivm.compare_mode<ne> -> tensor<4xi1>
  %lt = hivm.hir.vcmp
          ins(%a_f32, %b_f32 : tensor<4xf32>, tensor<4xf32>)
          outs(%init : tensor<4xi1>)
          compare_mode = #hivm.compare_mode<lt> -> tensor<4xi1>
  %le = hivm.hir.vcmp
          ins(%a_f32, %b_f32 : tensor<4xf32>, tensor<4xf32>)
          outs(%init : tensor<4xi1>)
          compare_mode = #hivm.compare_mode<le> -> tensor<4xi1>
  %gt = hivm.hir.vcmp
          ins(%a_f32, %b_f32 : tensor<4xf32>, tensor<4xf32>)
          outs(%init : tensor<4xi1>)
          compare_mode = #hivm.compare_mode<gt> -> tensor<4xi1>
  %ge = hivm.hir.vcmp
          ins(%a_f32, %b_f32 : tensor<4xf32>, tensor<4xf32>)
          outs(%init : tensor<4xi1>)
          compare_mode = #hivm.compare_mode<ge> -> tensor<4xi1>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %eq_0 = tensor.extract %eq[%c0] : tensor<4xi1>
  memref.store %eq_0, %sink[%c0] : memref<4xi1>
  %ne_1 = tensor.extract %ne[%c1] : tensor<4xi1>
  memref.store %ne_1, %sink[%c1] : memref<4xi1>
  %lt_2 = tensor.extract %lt[%c2] : tensor<4xi1>
  memref.store %lt_2, %sink[%c2] : memref<4xi1>
  %le_3 = tensor.extract %le[%c3] : tensor<4xi1>
  memref.store %le_3, %sink[%c3] : memref<4xi1>
  %gt_0 = tensor.extract %gt[%c0] : tensor<4xi1>
  %ge_1 = tensor.extract %ge[%c1] : tensor<4xi1>
  memref.store %gt_0, %sink[%c0] : memref<4xi1>
  memref.store %ge_1, %sink[%c1] : memref<4xi1>
  return
}

// -----
func.func @vsub_inline_OTF_broadcast(%arg0: tensor<64xf32>) -> tensor<64x64xf32> {
  %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
  %expanded_0 = tensor.expand_shape %arg0 [[0, 1]] output_shape [1, 64] : tensor<64xf32> into tensor<1x64xf32>
  %0 = tensor.empty() : tensor<64x64xf32>
  // CHECK-TRUE: linalg.sub ins({{.*}}, {{.*}} : tensor<64x64xf32>, tensor<64x64xf32>) outs({{.*}} : tensor<64x64xf32>) -> tensor<64x64xf32>
  // CHECK-FALSE: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
  %1 = hivm.hir.vsub ins(%expanded, %expanded_0 : tensor<64x1xf32>, tensor<1x64xf32>) outs(%0 : tensor<64x64xf32>) broadcast = [0, 1] -> tensor<64x64xf32>
  return %1 : tensor<64x64xf32>
}

// -----
func.func @vmod(%arg0: tensor<32xi64>, %arg1: i64) -> tensor<32xi64> {
  %0 = tensor.empty() : tensor<32xi64>
 // COMMON: hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
  %1 = hivm.hir.vmod ins(%arg0, %arg1 : tensor<32xi64>, i64) outs(%0 : tensor<32xi64>) -> tensor<32xi64>
  return %1 : tensor<32xi64>
}

// -----
func.func @vmodui(%arg0: tensor<32xi64>, %arg1: i64) -> tensor<32xi64> {
  %0 = tensor.empty() : tensor<32xi64>
 // COMMON: hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>}
  %1 = hivm.hir.vmodui ins(%arg0, %arg1 : tensor<32xi64>, i64) outs(%0 : tensor<32xi64>) -> tensor<32xi64>
  return %1 : tensor<32xi64>
}

// -----
// COMMON-LABEL: @vcumsum
func.func @vcumsum(%arg0: tensor<64x16xf32>) -> tensor<64x16xf32> {
  %0 = tensor.empty() : tensor<64x16xf32>
  // CHECK-TRUE: %{{.*}} = hfusion.cumsum %arg0 : tensor<64x16xf32> cum_dims = [0] reverse = false -> tensor<64x16xf32>
  %1 = hivm.hir.vcumsum ins(%arg0 : tensor<64x16xf32>) outs(%0 : tensor<64x16xf32>) cum_dims = [0] reverse = false -> tensor<64x16xf32>
  return %1 : tensor<64x16xf32>
}

// -----
// COMMON-LABEL: @vcumprod
func.func @vcumprod(%arg0: tensor<64x16xf32>) -> tensor<64x16xf32> {
  %0 = tensor.empty() : tensor<64x16xf32>
  // CHECK-TRUE: %{{.*}} = hfusion.cumprod %arg0 : tensor<64x16xf32> cum_dims = [0] reverse = false -> tensor<64x16xf32>
  %1 = hivm.hir.vcumprod ins(%arg0 : tensor<64x16xf32>) outs(%0 : tensor<64x16xf32>) cum_dims = [0] reverse = false -> tensor<64x16xf32>
  return %1 : tensor<64x16xf32>
}

// -----

// COMMON-LABEL: @vshl_lowering
func.func @vshl_lowering(%a: tensor<64xi32>, %c: i32, %d: tensor<64xi32>) -> tensor<64xi32> {
    // CHECK-TRUE: hfusion.elemwise_binary {fun = #hfusion.binary_fn<shli>}
    // CHECK-TRUE-NOT: linalg.map
    // CHECK-FALSE: linalg.generic
    // CHECK-FALSE: arith.shli
    %0 = hivm.hir.vshl ins(%a, %c : tensor<64xi32>, i32) outs(%d : tensor<64xi32>) -> tensor<64xi32>
    return %0 : tensor<64xi32>
}

// -----

// COMMON-LABEL: @vreduce_min_f32
func.func @vreduce_min_f32(%arg0: tensor<4x8xf32>) -> tensor<4x1xf32> {
  %init = tensor.empty() : tensor<4x1xf32>
  // COMMON: linalg.reduce
  // COMMON: arith.minimumf
  %0 = hivm.hir.vreduce <min> ins(%arg0 : tensor<4x8xf32>) outs(%init : tensor<4x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<4x1xf32>
  return %0 : tensor<4x1xf32>
}

// -----

// COMMON-LABEL: @vreduce_max_f32
func.func @vreduce_max_f32(%arg0: tensor<4x8xf32>) -> tensor<4x1xf32> {
  %init = tensor.empty() : tensor<4x1xf32>
  // COMMON: linalg.reduce
  // COMMON: arith.maximumf
  %0 = hivm.hir.vreduce <max> ins(%arg0 : tensor<4x8xf32>) outs(%init : tensor<4x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<4x1xf32>
  return %0 : tensor<4x1xf32>
}

// -----

// COMMON-LABEL: @vreduce_min_si32
func.func @vreduce_min_si32(%arg0: tensor<4x8xi32>) -> tensor<4x1xi32> {
  %init = tensor.empty() : tensor<4x1xi32>
  // COMMON: linalg.reduce
  // COMMON: arith.minsi
  %0 = hivm.hir.vreduce <min> ins(%arg0 : tensor<4x8xi32>) outs(%init : tensor<4x1xi32>) unsigned_src = false reduce_dims = [1] -> tensor<4x1xi32>
  return %0 : tensor<4x1xi32>
}

// -----

// COMMON-LABEL: @vreduce_min_ui32
func.func @vreduce_min_ui32(%arg0: tensor<4x8xi32>) -> tensor<4x1xi32> {
  %init = tensor.empty() : tensor<4x1xi32>
  // COMMON: linalg.reduce
  // COMMON: arith.minui
  %0 = hivm.hir.vreduce <min> ins(%arg0 : tensor<4x8xi32>) outs(%init : tensor<4x1xi32>) unsigned_src = true reduce_dims = [1] -> tensor<4x1xi32>
  return %0 : tensor<4x1xi32>
}

// -----

// COMMON-LABEL: @vdiv_lowering
func.func @vdiv_lowering(%a: tensor<64xi32>, %c: i32, %d: tensor<64xi32>) -> tensor<64xi32> {
   // CHECK-TRUE: linalg.div
   // CHECK-FALSE: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
   %0 = hivm.hir.vdiv ins(%a, %c : tensor<64xi32>, i32) outs(%d : tensor<64xi32>) -> tensor<64xi32>
   return %0 : tensor<64xi32>
}

// -----

// COMMON-LABEL: @vdiv_unsigned_lowering
func.func @vdiv_unsigned_lowering(%a: tensor<64xi32>, %c: i32, %d: tensor<64xi32>) -> tensor<64xi32> {
   // CHECK-TRUE: linalg.div_unsigned
   // CHECK-FALSE: linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>}
   %0 = hivm.hir.vdiv ins(%a, %c : tensor<64xi32>, i32) outs(%d : tensor<64xi32>) isSigned = false -> tensor<64xi32>
   return %0 : tensor<64xi32>
}

// -----

// COMMON-LABEL: @vdiv_float_lowering
func.func @vdiv_float_lowering(%a: tensor<64xf32>, %c: f32, %d: tensor<64xf32>) -> tensor<64xf32> {
   // CHECK-TRUE: linalg.div
   // CHECK-FALSE: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
   %0 = hivm.hir.vdiv ins(%a, %c : tensor<64xf32>, f32) outs(%d : tensor<64xf32>) -> tensor<64xf32>
   return %0 : tensor<64xf32>
}

// -----

// COMMON-LABEL: @vdiv_hp_float_lowering
func.func @vdiv_hp_float_lowering(%a: tensor<64xf32>, %b: tensor<64xf32>, %d: tensor<64xf32>) -> tensor<64xf32> {
   // COMMON: hfusion.elemwise_binary {fun = #hfusion.binary_fn<divfhp>}
   %0 = hivm.hir.vdiv ins(%a, %b : tensor<64xf32>, tensor<64xf32>) outs(%d : tensor<64xf32>) isHP = true -> tensor<64xf32>
   return %0 : tensor<64xf32>
}
