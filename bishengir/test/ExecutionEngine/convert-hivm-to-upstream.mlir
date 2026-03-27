// RUN: bishengir-opt --execution-engine-convert-hivm-to-upstream %s --split-input-file | FileCheck %s

// -----
 
func.func @tensor_direct_linalg_lowering(%a: tensor<1x?x10xf32>, %b: tensor<?x5x10xf32>, %c: tensor<5x?x10xf32>, %d: tensor<5x?x10xf32>) -> (tensor<5x?x10xf32>, tensor<5x?x10xf32>) {
 
    // CHECK: linalg.abs
    %0 = hivm.hir.vabs ins(%a: tensor<1x?x10xf32>) outs(%c: tensor<5x?x10xf32>) broadcast = [0] -> tensor<5x?x10xf32>
 
    // CHECK: linalg.add
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %d, %c0 : tensor<5x?x10xf32>
    %empty0 = tensor.empty(%dim) : tensor<5x?x10xf32>
    %1 = hivm.hir.vadd ins(%b, %b: tensor<?x5x10xf32>, tensor<?x5x10xf32>) outs(%empty0: tensor<5x?x10xf32>) transpose = [1, 0, 2] -> tensor<5x?x10xf32>
 
    // CHECK: linalg.sub
    %2 = hivm.hir.vsub ins(%0, %1: tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>
 
    // CHECK: linalg.mul
    %3 = hivm.hir.vmul ins(%1, %2: tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>
 
    // CHECK: linalg.div
    %4 = hivm.hir.vdiv ins(%2, %3: tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>
 
    // CHECK: linalg.max
    %5 = hivm.hir.vmax ins(%3, %4: tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>
 
    // CHECK: linalg.min
    %6 = hivm.hir.vmin ins(%4, %5: tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>
 
    // CHECK: linalg.exp
    %7 = hivm.hir.vexp ins(%6: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>
 
    // CHECK: linalg.log
    %8 = hivm.hir.vln ins(%7: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>
 
    // CHECK: linalg.rsqrt
    %9 = hivm.hir.vrsqrt ins(%8: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>
 
    // CHECK: linalg.sqrt
    %10 = hivm.hir.vsqrt ins(%9: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>
 
    // CHECK: linalg.tanh
    %11 = hivm.hir.vtanh ins(%10: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>
 
    // CHECK: linalg.reciprocal
    %empty12 = tensor.empty(%dim) : tensor<5x?x10xf32>   
    %12 = hivm.hir.vrec ins(%11: tensor<5x?x10xf32>) outs(%empty12: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>
 
    // CHECK: linalg.select
    %13 = arith.constant true
    %empty14 = tensor.empty(%dim) : tensor<5x?x10xf32>
    %14 = hivm.hir.vsel ins(%13, %12, %c: i1, tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%empty14: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>
 
    // CHECK: linalg.erf
    %15 = hivm.hir.verf ins(%14: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    %16 = hivm.hir.store ins(%15: tensor<5x?x10xf32>) outs(%d: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>
 
    // CHECK: linalg.transpose
    %17 = hivm.hir.vtranspose ins(%b: tensor<?x5x10xf32>) outs(%16: tensor<5x?x10xf32>) permutation = [1, 0, 2] -> tensor<5x?x10xf32>
 
    return %16, %17: tensor<5x?x10xf32>, tensor<5x?x10xf32>
}

// -----
 
func.func @memref_direct_linalg_lowering(%a: memref<1x?x10xf32>, %b: memref<?x5x10xf32>, %c: memref<5x?x10xf32>, %d: memref<5x?x10xf32>, %e: memref<1x?x10xi8>, %f: memref<5x?x10xi8>) {
 
    // CHECK: linalg.abs
    hivm.hir.vabs ins(%a: memref<1x?x10xf32>) outs(%c: memref<5x?x10xf32>) broadcast = [0]
 
    // CHECK: linalg.add
    hivm.hir.vadd ins(%b, %b: memref<?x5x10xf32>, memref<?x5x10xf32>) outs(%c: memref<5x?x10xf32>) transpose = [1, 0, 2]
 
    // CHECK: linalg.sub
    hivm.hir.vsub ins(%c, %d: memref<5x?x10xf32>, memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)
 
    // CHECK: linalg.mul
    hivm.hir.vmul ins(%c, %d: memref<5x?x10xf32>, memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)
 
    // CHECK: linalg.div
    hivm.hir.vdiv ins(%c, %d: memref<5x?x10xf32>, memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)
 
    // CHECK: linalg.max
    hivm.hir.vmax ins(%c, %d: memref<5x?x10xf32>, memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)
 
    // CHECK: linalg.min
    hivm.hir.vmin ins(%c, %d: memref<5x?x10xf32>, memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)
 
    // CHECK: linalg.exp
    hivm.hir.vexp ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)
 
    // CHECK: linalg.log
    hivm.hir.vln ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)
 
    // CHECK: linalg.rsqrt
    hivm.hir.vrsqrt ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)
 
    // CHECK: linalg.sqrt
    hivm.hir.vsqrt ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)
 
    // CHECK: linalg.tanh
    hivm.hir.vtanh ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)
 
    // CHECK: linalg.reciprocal
    hivm.hir.vrec ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)
 
    // CHECK: linalg.erf
    hivm.hir.verf ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)
 
    hivm.hir.store ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)
 
    // CHECK: linalg.transpose
    hivm.hir.vtranspose ins(%b: memref<?x5x10xf32>) outs(%c: memref<5x?x10xf32>) permutation = [1, 0, 2]
 
    func.return
}
 
// -----
 
func.func @elemwise_lowering(%a: tensor<?x5x10xf32>, %aT: tensor<5x?x10xf32>, %b: memref<5x1x10xi32>, %bB: memref<5x?x10xi32>) -> tensor<5x?x10xf32> {
 
    // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>}
    %0 = hivm.hir.vrelu ins(%a: tensor<?x5x10xf32>) outs(%aT: tensor<5x?x10xf32>) transpose = [1, 0, 2] -> tensor<5x?x10xf32>
 
    // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>}
    hivm.hir.vrelu ins(%b: memref<5x1x10xi32>) outs(%bB: memref<5x?x10xi32>) broadcast = [1]
 
    // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
    hivm.hir.vnot ins(%b: memref<5x1x10xi32>) outs(%bB: memref<5x?x10xi32>) broadcast = [1]
 
    func.return %0: tensor<5x?x10xf32>
}
 
// -----
 
func.func @bitwise_like_lowering(%a: tensor<?x5x10xf32>, %aT: tensor<5x?x10xf32>, %b: memref<5x1x10xi32>, %bB: memref<5x?x10xi32>) -> tensor<5x?x10xf32> 
{
 
    // CHECK: linalg.map {{.*}}
    // CHECK-2: arith.bitcast
    // CHECK:   arith.andi
    // CHECK:   arith.bitcast
    %0 = hivm.hir.vand ins(%a, %a: tensor<?x5x10xf32>, tensor<?x5x10xf32>) outs(%aT: tensor<5x?x10xf32>) transpose = [1, 0, 2] -> tensor<5x?x10xf32>
 
    // CHECK: linalg.map
    // CHECK-SAME:  arith.andi
    hivm.hir.vand ins(%b, %b: memref<5x1x10xi32>, memref<5x1x10xi32>) outs(%bB: memref<5x?x10xi32>) broadcast = [1]
 
    // CHECK: linalg.map {{.*}}
    // CHECK-2: arith.bitcast
    // CHECK:   arith.ori
    // CHECK:   arith.bitcast
    %1 = hivm.hir.vor ins(%a, %a: tensor<?x5x10xf32>, tensor<?x5x10xf32>) outs(%0: tensor<5x?x10xf32>) transpose = [1, 0, 2] -> tensor<5x?x10xf32>
 
    // CHECK: linalg.map
    // CHECK-SAME:  arith.ori
    hivm.hir.vor ins(%b, %b: memref<5x1x10xi32>, memref<5x1x10xi32>) outs(%bB: memref<5x?x10xi32>) broadcast = [1]
 
    // CHECK: linalg.map
    // CHECK-SAME:  arith.xori
    hivm.hir.vxor ins(%bB, %bB: memref<5x?x10xi32>, memref<5x?x10xi32>) outs(%bB: memref<5x?x10xi32>)
 
    func.return %1: tensor<5x?x10xf32>
}

// -----
 
func.func @cumulative_like_lowering(%a: tensor<5x?x10xf32>, %b: memref<5x?x10xi32>) -> tensor<5x?x10xf32> {
 
    // CHECK: linalg.generic
    // CHECK-SAME:  outs({{.*}}: tensor<5x?x10xf32>, tensor<5x1x1xf32>)
    // CHECK-NEXT:  ^bb0(%[[in:.*]]: f32, %{{.*}}: f32, %[[out:.*]]: f32)
    // CHECK-NEXT:      %[[res:.*]] = arith.mulf
    // CHECK-DAG-SAME:      %[[in]]
    // CHECK-DAG-SAME:      %[[out]]
    // CHECK-NEXT:      linalg.yield %[[res]], %[[res]]
    %0 = hivm.hir.vcumprod ins(%a: tensor<5x?x10xf32>) outs(%a: tensor<5x?x10xf32>) cum_dims = [0] reverse = false -> tensor<5x?x10xf32>
 
    // CHECK: linalg.generic
    // CHECK-SAME:  outs({{.*}}: memref<5x?x10xi32>, memref<5x?x1xi32>)
    // CHECK-NEXT:  ^bb0(%[[in:.*]]: i32, %{{.*}}: i32, %[[out:.*]]: i32)
    // CHECK-NEXT:      %[[res:.*]] = arith.muli
    // CHECK-DAG-SAME:      %[[in]]
    // CHECK-DAG-SAME:      %[[out]]
    // CHECK-NEXT:      linalg.yield %[[res]], %[[res]]
    hivm.hir.vcumprod ins(%b: memref<5x?x10xi32>) outs(%b: memref<5x?x10xi32>) cum_dims = [1] reverse = false
 
    // CHECK: linalg.generic
    // CHECK-SAME:  outs({{.*}}: memref<5x?x10xi32>, memref<5x?x1xi32>)
    // CHECK-NEXT:  ^bb0(%[[in:.*]]: i32, %{{.*}}: i32, %[[out:.*]]: i32)
    // CHECK-NEXT:      %[[res:.*]] = arith.addi
    // CHECK-DAG-SAME:      %[[in]]
    // CHECK-DAG-SAME:      %[[out]]
    // CHECK-NEXT:      linalg.yield %[[res]], %[[res]]
    hivm.hir.vcumsum ins(%b: memref<5x?x10xi32>) outs(%b: memref<5x?x10xi32>) cum_dims = [1] reverse = false
 
    func.return %0: tensor<5x?x10xf32>
}
 
// -----
 
func.func @arange_lowering(%a: tensor<5x?x10xi64>, %b: memref<5x?x10xi32>) -> tensor<5x?x10xi64> {
 
    // CHECK: %[[C0:.*]] = arith.constant 0
    %c0 = arith.constant 0: index
    // CHECK: %[[C1:.*]] = arith.constant 1
    %c1 = arith.constant 1: index
    // CHECK: %[[C2:.*]] = arith.constant 2
    %c2 = arith.constant 2: index
    // CHECK: %[[C3:.*]] = arith.constant 3
    %c3 = arith.constant 3: index
 
    // CHECK: hfusion.arange
    // CHECK-SAME:  strides[%[[C0]], %[[C3]], %[[C2]]]
    %0 = hivm.hir.varange offset[] strides[%c0, %c3, %c2] outs(%a: tensor<5x?x10xi64>) -> tensor<5x?x10xi64>
 
    // CHECK: hfusion.arange offset[%[[C3]]] strides[%[[C1]], %[[C1]], %[[C1]]]
    hivm.hir.varange offset[%c3] strides[%c1, %c1, %c1] outs(%b: memref<5x?x10xi32>)
 
    func.return %0: tensor<5x?x10xi64>
}
 
// -----
 
// CHECK-LABEL: @concat_lowering
// CHECK-SAME:      %[[a:[^:]*]]: {{[^,]*}}, 
// CHECK-SAME:      %[[b:[^:]*]]: {{[^,]*}}, 
// CHECK-SAME:      %[[c:[^:]*]]: {{[^,]*}}, 
// CHECK-SAME:      %[[d:[^:]*]]: {{[^,]*}}, 
// CHECK-SAME:      %[[e:[^:]*]]: {{[^,]*}}, 
// CHECK-SAME:      %[[f:[^:]*]]: {{[^,]*}}
func.func @concat_lowering(%a: tensor<5x?x10xf32>, %b: tensor<?x?x10xf32>, %c: tensor<?x?x10xf32>, %d: memref<5x?x10xi32>, %e: memref<?x?x10xi32>, %f: memref<?x?x10xi32>) -> tensor<?x?x10xf32> {
 
    // CHECK: tensor.concat
    %0 = hivm.hir.vconcat dim(0) ins(%a, %b: tensor<5x?x10xf32>, tensor<?x?x10xf32>) outs(%c: tensor<?x?x10xf32>) -> tensor<?x?x10xf32>
 
    // CHECK-DAG: %[[tensorD:.*]] = bufferization.to_tensor %[[d]]
    // CHECK-DAG: %[[tensorE:.*]] = bufferization.to_tensor %[[e]]
    // CHECK:   %[[concat:.*]] = tensor.concat dim(0)
    // CHECK-DAG-SAME:                      %[[tensorD]]
    // CHECK-DAG-SAME:                      %[[tensorE]]
    // CHECK:   bufferization.materialize_in_destination %[[concat]]
    // CHECK-SAME:                                          %[[f]]
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
    // CHECK: hfusion.cast {round_mode = #hfusion.round_mode<rint>}
    hivm.hir.vcast ins(%bf16 : memref<2x16xbf16>) outs(%f32 : memref<2x16xf32>)
                   round_mode = #hivm.round_mode<rint>
    // CHECK: hfusion.cast {round_mode = #hfusion.round_mode<round>}
    hivm.hir.vcast ins(%bf16 : memref<2x16xbf16>) outs(%s32 : memref<2x16xi32>)
                   round_mode = #hivm.round_mode<round>
    // CHECK: hfusion.cast {round_mode = #hfusion.round_mode<ceil>}
    hivm.hir.vcast ins(%bf16 : memref<2x16xbf16>) outs(%s32 : memref<2x16xi32>)
                   round_mode = #hivm.round_mode<ceil>
    // CHECK: hfusion.cast {round_mode = #hfusion.round_mode<floor>}
    hivm.hir.vcast ins(%bf16 : memref<2x16xbf16>) outs(%s32 : memref<2x16xi32>)
                   round_mode = #hivm.round_mode<floor>
    return
}

// -----
 
// CHECK-LABEL: func.func @vmp_lowering(
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vle>}
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vgt>}
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vge>}
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
  // CHECK: linalg.sub ins({{.*}}, {{.*}} : tensor<64x64xf32>, tensor<64x64xf32>) outs({{.*}} : tensor<64x64xf32>) -> tensor<64x64xf32>
  %1 = hivm.hir.vsub ins(%expanded, %expanded_0 : tensor<64x1xf32>, tensor<1x64xf32>) outs(%0 : tensor<64x64xf32>) broadcast = [0, 1] -> tensor<64x64xf32>
  return %1 : tensor<64x64xf32>
}

// -----
func.func @vmod(%arg0: tensor<32xi64>, %arg1: i64) -> tensor<32xi64> {
  %0 = tensor.empty() : tensor<32xi64>
 // CHECK: %{{.*}} = arith.remsi %{{.*}}, %{{.*}} : i64
  %1 = hivm.hir.vmod ins(%arg0, %arg1 : tensor<32xi64>, i64) outs(%0 : tensor<32xi64>) -> tensor<32xi64>
  return %1 : tensor<32xi64>
}

// -----
func.func @vmodui(%arg0: tensor<32xi64>, %arg1: i64) -> tensor<32xi64> {
  %0 = tensor.empty() : tensor<32xi64>
 // CHECK: %{{.*}} = arith.remui %{{.*}}, %{{.*}} : i64
  %1 = hivm.hir.vmodui ins(%arg0, %arg1 : tensor<32xi64>, i64) outs(%0 : tensor<32xi64>) -> tensor<32xi64>
  return %1 : tensor<32xi64>
}

// -----
func.func @triton_dot_max_2D_acc_None_mix_aiv(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?xf32>, %arg6: memref<?xi32>, %arg7: memref<?xf32>, %arg8: memref<?xi32>, %arg9: i32, %arg10: i32, %arg11: i32) {
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c-2139095040_i32 = arith.constant -2139095040 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %cst = arith.constant 0xFF800000 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  hivm.hir.set_ctrl false at ctrl[60]
  hivm.hir.set_ctrl true at ctrl[48]
  %0 = arith.muli %arg9, %arg10 : i32
  %1 = arith.muli %0, %arg11 : i32
  annotation.mark %1 {logical_block_num} : i32
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [23, 77], strides: [77, 1] : memref<?xf32> to memref<23x77xf32, strided<[77, 1]>>
  %alloc = memref.alloc() : memref<23x77xf32>
  hivm.hir.load ins(%reinterpret_cast : memref<23x77xf32, strided<[77, 1]>>) outs(%alloc : memref<23x77xf32>) eviction_policy = <EvictFirst>
  %2 = bufferization.to_tensor %alloc restrict writable : memref<23x77xf32>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [77, 32], strides: [32, 1] : memref<?xf32> to memref<77x32xf32, strided<[32, 1]>>
  %alloc_2 = memref.alloc() : memref<77x32xf32>
  hivm.hir.load ins(%reinterpret_cast_1 : memref<77x32xf32, strided<[32, 1]>>) outs(%alloc_2 : memref<77x32xf32>) eviction_policy = <EvictFirst>
  %3 = bufferization.to_tensor %alloc_2 restrict writable : memref<77x32xf32>
  %4 = hivm.hir.bitcast %2 : tensor<23x77xf32> -> tensor<23x77xi32>
  %5 = tensor.empty() : tensor<23x77xi32>
  %6 = hivm.hir.vbrc ins(%c2147483647_i32 : i32) outs(%5 : tensor<23x77xi32>) -> tensor<23x77xi32>
  %7 = tensor.empty() : tensor<23x77xi32>
  %8 = hivm.hir.vand ins(%4, %6 : tensor<23x77xi32>, tensor<23x77xi32>) outs(%7 : tensor<23x77xi32>) -> tensor<23x77xi32>
  %9 = tensor.empty() : tensor<23x77xi32>
  %10 = hivm.hir.vadd ins(%8, %c-2139095040_i32 : tensor<23x77xi32>, i32) outs(%9 : tensor<23x77xi32>) -> tensor<23x77xi32>
  %11 = hivm.hir.vmin ins(%10, %c1_i32 : tensor<23x77xi32>, i32) outs(%10 : tensor<23x77xi32>) -> tensor<23x77xi32>
  %12 = hivm.hir.vmax ins(%11, %c0_i32 : tensor<23x77xi32>, i32) outs(%11 : tensor<23x77xi32>) -> tensor<23x77xi32>
  %13 = tensor.empty() : tensor<23x77xf32>
  %14 = hivm.hir.vcast ins(%12 : tensor<23x77xi32>) outs(%13 : tensor<23x77xf32>) -> tensor<23x77xf32>
  %15 = tensor.empty() : tensor<23x77xi1>
  %16 = hivm.hir.vcmp ins(%14, %cst_0 : tensor<23x77xf32>, f32) outs(%15 : tensor<23x77xi1>) compare_mode = <ne> -> tensor<23x77xi1>
  %17 = tensor.empty() : tensor<23x77xf32>
  %18 = hivm.hir.vsel ins(%16, %cst, %2 : tensor<23x77xi1>, f32, tensor<23x77xf32>) outs(%17 : tensor<23x77xf32>) -> tensor<23x77xf32>
  %19 = tensor.empty() : tensor<23x1xf32>
  %20 = tensor.empty() : tensor<23x1xi32>
  %21:2 = hivm.hir.vreduce <max_with_index> ins(%18 : tensor<23x77xf32>) outs(%19, %20 : tensor<23x1xf32>, tensor<23x1xi32>) unsigned_src = false tie_break_left = true reduce_dims = [1] -> tensor<23x1xf32>, tensor<23x1xi32>
  %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [23, 1], strides: [1, 1] : memref<?xf32> to memref<23x1xf32, strided<[1, 1]>>
  hivm.hir.store ins(%21#0 : tensor<23x1xf32>) outs(%reinterpret_cast_3 : memref<23x1xf32, strided<[1, 1]>>)
  %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [23, 1], strides: [1, 1] : memref<?xi32> to memref<23x1xi32, strided<[1, 1]>>
  hivm.hir.store ins(%21#1 : tensor<23x1xi32>) outs(%reinterpret_cast_4 : memref<23x1xi32, strided<[1, 1]>>)
  %22 = hivm.hir.bitcast %3 : tensor<77x32xf32> -> tensor<77x32xi32>
  %23 = tensor.empty() : tensor<77x32xi32>
  %24 = hivm.hir.vbrc ins(%c2147483647_i32 : i32) outs(%23 : tensor<77x32xi32>) -> tensor<77x32xi32>
  %25 = tensor.empty() : tensor<77x32xi32>
  %26 = hivm.hir.vand ins(%22, %24 : tensor<77x32xi32>, tensor<77x32xi32>) outs(%25 : tensor<77x32xi32>) -> tensor<77x32xi32>
  %27 = tensor.empty() : tensor<77x32xi32>
  %28 = hivm.hir.vadd ins(%26, %c-2139095040_i32 : tensor<77x32xi32>, i32) outs(%27 : tensor<77x32xi32>) -> tensor<77x32xi32>
  %29 = hivm.hir.vmin ins(%28, %c1_i32 : tensor<77x32xi32>, i32) outs(%28 : tensor<77x32xi32>) -> tensor<77x32xi32>
  %30 = hivm.hir.vmax ins(%29, %c0_i32 : tensor<77x32xi32>, i32) outs(%29 : tensor<77x32xi32>) -> tensor<77x32xi32>
  %31 = tensor.empty() : tensor<77x32xf32>
  %32 = hivm.hir.vcast ins(%30 : tensor<77x32xi32>) outs(%31 : tensor<77x32xf32>) -> tensor<77x32xf32>
  %33 = tensor.empty() : tensor<77x32xi1>
  %34 = hivm.hir.vcmp ins(%32, %cst_0 : tensor<77x32xf32>, f32) outs(%33 : tensor<77x32xi1>) compare_mode = <ne> -> tensor<77x32xi1>
  %35 = tensor.empty() : tensor<77x32xf32>
  %36 = hivm.hir.vsel ins(%34, %cst, %3 : tensor<77x32xi1>, f32, tensor<77x32xf32>) outs(%35 : tensor<77x32xf32>) -> tensor<77x32xf32>
  %37 = tensor.empty() : tensor<77x1xf32>
  %38 = tensor.empty() : tensor<77x1xi32>
  // CHECK: %{{.*}} = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <max>
  %39:2 = hivm.hir.vreduce <max_with_index> ins(%36 : tensor<77x32xf32>) outs(%37, %38 : tensor<77x1xf32>, tensor<77x1xi32>) unsigned_src = false tie_break_left = true reduce_dims = [1] -> tensor<77x1xf32>, tensor<77x1xi32>
  %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [77, 1], strides: [1, 1] : memref<?xf32> to memref<77x1xf32, strided<[1, 1]>>
  hivm.hir.store ins(%39#0 : tensor<77x1xf32>) outs(%reinterpret_cast_5 : memref<77x1xf32, strided<[1, 1]>>)
  %reinterpret_cast_6 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [77, 1], strides: [1, 1] : memref<?xi32> to memref<77x1xi32, strided<[1, 1]>>
  hivm.hir.store ins(%39#1 : tensor<77x1xi32>) outs(%reinterpret_cast_6 : memref<77x1xi32, strided<[1, 1]>>)
  return
}

// -----
// CHECK-LABEL: @vinterleave_tensor
func.func @vinterleave_tensor(%a: tensor<2x16xf32>, %b: tensor<2x16xf32>, %c: tensor<2x32xf32>) -> tensor<2x32xf32> {
  // CHECK: %{{.*}} = hfusion.interleave %{{.*}}, %{{.*}} : tensor<2x16xf32>, tensor<2x16xf32> -> tensor<2x32xf32>
  %0 = hivm.hir.vinterleave ins(%a, %b : tensor<2x16xf32>, tensor<2x16xf32>) outs(%c : tensor<2x32xf32>) interleave_channel_nums = 2 -> tensor<2x32xf32>
  return %0 : tensor<2x32xf32>
}

// -----
// CHECK-LABEL: @vdeinterleave_tensor
func.func @vdeinterleave_tensor(%arg0: tensor<32xf32>) -> tensor<16xf32> {
  %0 = tensor.empty() : tensor<16xf32>
  // CHECK: hfusion.deinterleave %{{.*}} channel<0>
  %1 = hivm.hir.vdeinterleave ins(%arg0 : tensor<32xf32>) outs(%0 : tensor<16xf32>) index_mode = <CHANNEL_0> -> tensor<16xf32>
  return %1 : tensor<16xf32>
}