// REQUIRES: execution-engine
// RUN: bishengir-opt --execution-engine-convert-hivm-to-upstream %s --split-input-file | FileCheck %s

// -----
 
func.func @tensor_direct_linalg_lowering(%a: tensor<1x?x10xf32>, %b: tensor<?x5x10xf32>, %c: tensor<5x?x10xf32>, %d: tensor<5x?x10xf32>) -> (tensor<5x?x10xf32>, tensor<5x?x10xf32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
 
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
 
    // CHECK: linalg.copy
    %16 = hivm.hir.store ins(%15: tensor<5x?x10xf32>) outs(%d: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>
 
    // CHECK: linalg.transpose
    %17 = hivm.hir.vtranspose ins(%b: tensor<?x5x10xf32>) outs(%16: tensor<5x?x10xf32>) permutation = [1, 0, 2] -> tensor<5x?x10xf32>
 
    return %16, %17: tensor<5x?x10xf32>, tensor<5x?x10xf32>
}

// -----
 
func.func @memref_direct_linalg_lowering(%a: memref<1x?x10xf32>, %b: memref<?x5x10xf32>, %c: memref<5x?x10xf32>, %d: memref<5x?x10xf32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
 
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
 
    // CHECK: linalg.copy
    hivm.hir.store ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)
 
    // CHECK: linalg.transpose
    hivm.hir.vtranspose ins(%b: memref<?x5x10xf32>) outs(%c: memref<5x?x10xf32>) permutation = [1, 0, 2]
 
    func.return
}
 
// -----
 
func.func @elemwise_lowering(%a: tensor<?x5x10xf32>, %aT: tensor<5x?x10xf32>, %b: memref<5x1x10xi32>, %bB: memref<5x?x10xi32>) -> tensor<5x?x10xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
 
    // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>}
    %0 = hivm.hir.vrelu ins(%a: tensor<?x5x10xf32>) outs(%aT: tensor<5x?x10xf32>) transpose = [1, 0, 2] -> tensor<5x?x10xf32>
 
    // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>}
    hivm.hir.vrelu ins(%b: memref<5x1x10xi32>) outs(%bB: memref<5x?x10xi32>) broadcast = [1]
 
    // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
    hivm.hir.vnot ins(%b: memref<5x1x10xi32>) outs(%bB: memref<5x?x10xi32>) broadcast = [1]
 
    func.return %0: tensor<5x?x10xf32>
}
 
// -----
 
func.func @bitwise_like_lowering(%a: tensor<?x5x10xf32>, %aT: tensor<5x?x10xf32>, %b: memref<5x1x10xi32>, %bB: memref<5x?x10xi32>) -> tensor<5x?x10xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
 
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

// CHECK-LABEL: func.func @shift_like_lowering
func.func @shift_like_lowering(%a: tensor<5x10xi32>, %b: tensor<5x10xi32>, %dst: tensor<5x10xi32>) -> (tensor<5x10xi32>, tensor<5x10xi32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {

    // CHECK: linalg.map
    // CHECK-SAME:  arith.shli
    %0 = hivm.hir.vshl ins(%a, %b: tensor<5x10xi32>, tensor<5x10xi32>) outs(%dst: tensor<5x10xi32>) -> tensor<5x10xi32>

    // CHECK: linalg.map
    // CHECK-SAME:  arith.shrsi
    %1 = hivm.hir.vshr ins(%a, %b: tensor<5x10xi32>, tensor<5x10xi32>) outs(%dst: tensor<5x10xi32>) -> tensor<5x10xi32>

    func.return %0, %1: tensor<5x10xi32>, tensor<5x10xi32>
}

// -----
 
func.func @cumulative_like_lowering(%a: tensor<5x?x10xf32>, %b: memref<5x?x10xi32>) -> tensor<5x?x10xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
 
    // CHECK: linalg.generic
    // CHECK-SAME:  outs({{.*}}: tensor<5x?x10xf32>, tensor<5x1x1xf32>)
    // CHECK-NEXT:  ^bb0(%[[in:.*]]: f32, %{{.*}}: f32, %[[out:.*]]: f32)
    // CHECK-NEXT:      %[[res:.*]] = arith.mulf
    // CHECK-DAG-SAME:      %[[in]]
    // CHECK-DAG-SAME:      %[[out]]
    // CHECK-NEXT:      linalg.yield %[[res]], %[[res]]
    %0 = hivm.hir.vcumprod ins(%a: tensor<5x?x10xf32>) outs(%a: tensor<5x?x10xf32>) cum_dims = [0] -> tensor<5x?x10xf32>
 
    // CHECK: linalg.generic
    // CHECK-SAME:  outs({{.*}}: memref<5x?x10xi32>, memref<5x?x1xi32>)
    // CHECK-NEXT:  ^bb0(%[[in:.*]]: i32, %{{.*}}: i32, %[[out:.*]]: i32)
    // CHECK-NEXT:      %[[res:.*]] = arith.muli
    // CHECK-DAG-SAME:      %[[in]]
    // CHECK-DAG-SAME:      %[[out]]
    // CHECK-NEXT:      linalg.yield %[[res]], %[[res]]
    hivm.hir.vcumprod ins(%b: memref<5x?x10xi32>) outs(%b: memref<5x?x10xi32>) cum_dims = [1]
 
    // CHECK: linalg.generic
    // CHECK-SAME:  outs({{.*}}: memref<5x?x10xi32>, memref<5x?x1xi32>)
    // CHECK-NEXT:  ^bb0(%[[in:.*]]: i32, %{{.*}}: i32, %[[out:.*]]: i32)
    // CHECK-NEXT:      %[[res:.*]] = arith.addi
    // CHECK-DAG-SAME:      %[[in]]
    // CHECK-DAG-SAME:      %[[out]]
    // CHECK-NEXT:      linalg.yield %[[res]], %[[res]]
    hivm.hir.vcumsum ins(%b: memref<5x?x10xi32>) outs(%b: memref<5x?x10xi32>) cum_dims = [1]
 
    func.return %0: tensor<5x?x10xf32>
}
 
// -----
 
func.func @arange_lowering(%a: tensor<5x?x10xi64>, %b: memref<5x?x10xi32>) -> tensor<5x?x10xi64> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
 
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
func.func @concat_lowering(%a: tensor<5x?x10xf32>, %b: tensor<?x?x10xf32>, %c: tensor<?x?x10xf32>, %d: memref<5x?x10xi32>, %e: memref<?x?x10xi32>, %f: memref<?x?x10xi32>) -> tensor<?x?x10xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
 
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