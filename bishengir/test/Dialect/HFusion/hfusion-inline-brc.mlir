// RUN: bishengir-opt --hfusion-inline-brc -split-input-file %s | FileCheck %s

// -----

// CHECK-LABEL: func.func @inline_brc_to_elemwise_binary_func(
// CHECK-SAME: %[[arg0:.*]]: tensor<32xf16>
// CHECK-SAME: %[[arg1:.*]]: tensor<f16>
// CHECK: %[[cst:.*]] = arith.constant
// CHECK: %[[extracted:.*]] = tensor.extract {{.*}} tensor<f16>
// CHECK: linalg.binary_fn<div>} ins({{.*}}, %[[extracted]] : tensor<32xf16>, f16)
// CHECK-NEXT: linalg.binary_fn<sub>} ins(%[[cst]], {{.*}} : f16, tensor<32xf16>)
// CHECK-NEXT: linalg.binary_fn<add>} ins({{.*}}, %[[extracted]] : tensor<32xf16>, f16)
// CHECK-NEXT: linalg.binary_fn<mul>} ins(%[[cst]], {{.*}} : f16, tensor<32xf16>)
func.func @inline_brc_to_elemwise_binary_func(%arg0: tensor<32xf16>, %arg1: tensor<f16>) -> tensor<32xf16> {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = tensor.empty() : tensor<32xf16>
    %broadcasted = linalg.broadcast ins(%arg1 : tensor<f16>) outs(%0 : tensor<32xf16>) dimensions = [0] 
    %filled = linalg.fill ins(%cst : f16) outs(%0 : tensor<32xf16>) -> tensor<32xf16>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %broadcasted : tensor<32xf16>, tensor<32xf16>) outs(%0 : tensor<32xf16>) -> tensor<32xf16>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%filled, %1 : tensor<32xf16>, tensor<32xf16>) outs(%0 : tensor<32xf16>) -> tensor<32xf16>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%2, %broadcasted : tensor<32xf16>, tensor<32xf16>) outs(%0 : tensor<32xf16>) -> tensor<32xf16>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%filled, %3 : tensor<32xf16>, tensor<32xf16>) outs(%0 : tensor<32xf16>) -> tensor<32xf16>
    return %4 : tensor<32xf16>
}

// -----

// CHECK-LABEL: func.func @inline_brc_with_different_ranks(
// CHECK: tensor.extract {{.*}} tensor<1xf16>
// CHECK: linalg.elemwise_binary {{.*}} f16, tensor<8x4x1xf16>
func.func @inline_brc_with_different_ranks(%arg0: tensor<8x4x1xf16>, %arg1: tensor<1xf16>) -> tensor<8x4x1xf16> {
    %0 = tensor.empty() : tensor<8x4x1xf16>
    %broadcasted = linalg.broadcast ins(%arg1 : tensor<1xf16>) outs(%0 : tensor<8x4x1xf16>) dimensions = [0, 1] 
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} 
                                ins(%broadcasted, %arg0 : tensor<8x4x1xf16>, tensor<8x4x1xf16>) 
                                outs(%0 : tensor<8x4x1xf16>) -> tensor<8x4x1xf16>
    return %1 : tensor<8x4x1xf16>
}

// CHECK-LABEL: func.func @inline_fill_with_different_ranks(
// CHECK-NOT: tensor.extract
// CHECK: linalg.elemwise_binary {{.*}} f16, tensor<8x4x1xf16>
func.func @inline_fill_with_different_ranks(%arg0: tensor<8x4x1xf16>) -> tensor<8x4x1xf16> {
    %0 = tensor.empty() : tensor<8x4x1xf16>
    %cst = arith.constant 1.000000e+00 : f16
    %filled = linalg.fill ins(%cst : f16) outs(%0 : tensor<8x4x1xf16>) -> tensor<8x4x1xf16>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} 
                                ins(%filled, %arg0 : tensor<8x4x1xf16>, tensor<8x4x1xf16>) 
                                outs(%0 : tensor<8x4x1xf16>) -> tensor<8x4x1xf16>
    return %1 : tensor<8x4x1xf16>
}

// -----

// CHECK-LABEL: func.func @inline_brc_with_same_operands(
func.func @inline_brc_with_same_operands(%arg0: tensor<24x128x1x1xbf16>, %arg1: tensor<24x128x1x1xbf16>, 
                                         %arg2: tensor<bf16>) -> tensor<24x128x1x1xbf16> {
    // CHECK: %[[CST:.*]] = arith.constant
    %cst = arith.constant 1.000000e+00 : bf16
    %0 = tensor.empty() : tensor<24x128x1x1xbf16>
    %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<24x128x1x1xbf16>) -> tensor<24x128x1x1xbf16>
    // CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[CST]], %[[CST]] : bf16, bf16)
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %1 : tensor<24x128x1x1xbf16>, tensor<24x128x1x1xbf16>) 
                                                               outs(%0 : tensor<24x128x1x1xbf16>) -> tensor<24x128x1x1xbf16>
    %3 = tensor.empty() : tensor<24x128x1x1xbf16>
    // CHECK: %[[EXTRACTED:.*]] = tensor.extract
    %4 = linalg.broadcast ins(%arg2 : tensor<bf16>) outs(%3 : tensor<24x128x1x1xbf16>) dimensions = [0, 1, 2, 3]
    // CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[EXTRACTED]], %[[EXTRACTED]] : bf16, bf16)
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%4, %4 : tensor<24x128x1x1xbf16>, tensor<24x128x1x1xbf16>) 
                                                               outs(%3 : tensor<24x128x1x1xbf16>) -> tensor<24x128x1x1xbf16>
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%2, %5 : tensor<24x128x1x1xbf16>, tensor<24x128x1x1xbf16>) 
                                                               outs(%arg1 : tensor<24x128x1x1xbf16>) -> tensor<24x128x1x1xbf16>
    return %6 : tensor<24x128x1x1xbf16>
}

// -----

// CHECK-LABEL: func.func @inline_brc_with_vector_operands_diff_source(
func.func @inline_brc_with_vector_operands_diff_source(%arg0: tensor<24x128x1x1xbf16>, %arg1: tensor<24x128x1x1xbf16>, 
                                                       %arg2: tensor<bf16>, %arg3: tensor<bf16>) -> tensor<24x128x1x1xbf16> {
    // CHECK: %[[CST1:.*]] = arith.constant 1.000000e+00 : bf16
    // CHECK: %[[CST2:.*]] = arith.constant 2.000000e+00 : bf16
    %cst1 = arith.constant 1.000000e+00 : bf16
    %cst2 = arith.constant 2.000000e+00 : bf16
    %0 = tensor.empty() : tensor<24x128x1x1xbf16>
    %1 = tensor.empty() : tensor<24x128x1x1xbf16>

    %fill1 = linalg.fill ins(%cst1 : bf16) outs(%0 : tensor<24x128x1x1xbf16>) -> tensor<24x128x1x1xbf16>
    %fill2 = linalg.fill ins(%cst2 : bf16) outs(%1 : tensor<24x128x1x1xbf16>) -> tensor<24x128x1x1xbf16>

    // CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[CST1]], %[[CST2]] : bf16, bf16
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%fill1, %fill2 : tensor<24x128x1x1xbf16>, tensor<24x128x1x1xbf16>) 
                                                               outs(%0 : tensor<24x128x1x1xbf16>) -> tensor<24x128x1x1xbf16>
    %3 = tensor.empty() : tensor<24x128x1x1xbf16>
    %4 = tensor.empty() : tensor<24x128x1x1xbf16>
    // CHECK: %[[EXTRACT1:.*]] = tensor.extract
    // CHECK: %[[EXTRACT2:.*]] = tensor.extract
    %5 = linalg.broadcast ins(%arg2 : tensor<bf16>) outs(%3 : tensor<24x128x1x1xbf16>) dimensions = [0, 1, 2, 3]
    %6 = linalg.broadcast ins(%arg3 : tensor<bf16>) outs(%4 : tensor<24x128x1x1xbf16>) dimensions = [0, 1, 2, 3]
    // CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[EXTRACT1]], %[[EXTRACT2]] : bf16, bf16
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%5, %6 : tensor<24x128x1x1xbf16>, tensor<24x128x1x1xbf16>) 
                                                               outs(%3 : tensor<24x128x1x1xbf16>) -> tensor<24x128x1x1xbf16>

    %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%2, %7 : tensor<24x128x1x1xbf16>, tensor<24x128x1x1xbf16>) 
                                                               outs(%arg1 : tensor<24x128x1x1xbf16>) -> tensor<24x128x1x1xbf16>
    return %8 : tensor<24x128x1x1xbf16>
}

// CHECK-LABEL: func.func @inline_brc_to_cast_binaryOp_users
// CHECK: %[[cst:.*]] = arith.constant 1.000000e+00 : f16
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[cst]]
func.func @inline_brc_to_cast_binaryOp_users(%arg0: tensor<32xf16>, %arg1: tensor<f32>) -> tensor<32xf16> {
    %cst = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<32xf32>
    %1 = tensor.empty() : tensor<32xf16>
    %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%2 : tensor<32xf32>) outs(%1 : tensor<32xf16>) -> tensor<32xf16>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %arg0 : tensor<32xf16>, tensor<32xf16>) outs(%1 : tensor<32xf16>) -> tensor<32xf16>
    return %4 : tensor<32xf16>
}

// -----

// CHECK-LABEL: func.func @inline_brc_to_hfusion_binary_func_int
// CHECK:  %[[empty0:.*]] = tensor.empty() : tensor<32xi32>
// CHECK:  %[[shli:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shli>} ins(%[[arg0:.*]], %[[arg1:.*]] : tensor<32xi32>, tensor<i32>) outs(%[[empty0:.*]] : tensor<32xi32>) -> tensor<32xi32>
// CHECK:  %[[empty1:.*]] = tensor.empty() : tensor<32xi32>
// CHECK:  %[[shrui:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%[[shli:.*]], %[[arg1:.*]] : tensor<32xi32>, tensor<i32>) outs(%[[empty1:.*]] : tensor<32xi32>) -> tensor<32xi32>
func.func @inline_brc_to_hfusion_binary_func_int(%arg0: tensor<32xi32>, %arg1: tensor<i32>) -> tensor<32xi32> {
    %0 = tensor.empty() : tensor<32xi32>
    %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shli>} ins(%arg0, %arg1 : tensor<32xi32>, tensor<i32>) outs(%0 : tensor<32xi32>) -> tensor<32xi32>
    %2 = tensor.empty() : tensor<32xi32>
    %3 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins(%1, %arg1 : tensor<32xi32>, tensor<i32>) outs(%2 : tensor<32xi32>) -> tensor<32xi32>
    return %3 : tensor<32xi32>
}

// -----

// CHECK-LABEL: func.func @inline_fill_to_hfusion_vand_i32(
// CHECK: %[[MASK:.*]] = arith.constant 2147483647 : i32
// CHECK-NOT: linalg.fill
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins({{.*}}, %[[MASK]] : tensor<64x64xi32>, i32)
func.func @inline_fill_to_hfusion_vand_i32(%arg0: tensor<64x64xi32>) -> tensor<64x64xi32> {
    %empty = tensor.empty() : tensor<64x64xi32>
    %mask = arith.constant 2147483647 : i32
    %filled = linalg.fill ins(%mask : i32) outs(%empty : tensor<64x64xi32>) -> tensor<64x64xi32>
    %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%arg0, %filled : tensor<64x64xi32>, tensor<64x64xi32>) outs(%empty : tensor<64x64xi32>) -> tensor<64x64xi32>
    return %result : tensor<64x64xi32>
}

// -----

// CHECK-LABEL: func.func @inline_fill_to_linalg_add_mul(
// CHECK: %[[SCALAR:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-NOT: linalg.fill
// CHECK-NOT: linalg.add
// CHECK-NOT: linalg.mul
// CHECK: %[[ADD:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins({{.*}}, %[[SCALAR]] : tensor<64xf32>, f32)
// CHECK-NOT: linalg.fill
// CHECK: %[[MUL:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ADD]], %[[SCALAR]] : tensor<64xf32>, f32)
// CHECK-NOT: linalg.fill
// CHECK: return %[[MUL]]
func.func @inline_fill_to_linalg_add_mul(%arg0: tensor<64xf32>) -> tensor<64xf32> {
  %scalar = arith.constant 2.000000e+00 : f32
  %fill_empty = tensor.empty() : tensor<64xf32>
  %filled = linalg.fill ins(%scalar : f32) outs(%fill_empty : tensor<64xf32>) -> tensor<64xf32>
  %add_empty = tensor.empty() : tensor<64xf32>
  %add = linalg.add ins(%arg0, %filled : tensor<64xf32>, tensor<64xf32>) outs(%add_empty : tensor<64xf32>) -> tensor<64xf32>
  %mul_empty = tensor.empty() : tensor<64xf32>
  %mul = linalg.mul ins(%add, %filled : tensor<64xf32>, tensor<64xf32>) outs(%mul_empty : tensor<64xf32>) -> tensor<64xf32>
  return %mul : tensor<64xf32>
}

// -----

// CHECK-LABEL: func.func @preserve_memref_broadcast(
// CHECK: linalg.broadcast ins(%{{.*}} : memref<f32>) outs(%{{.*}} : memref<64xf32>) dimensions = [0]
func.func @preserve_memref_broadcast(%arg0: memref<f32>,
                                     %arg1: memref<64xf32>) {
  linalg.broadcast ins(%arg0 : memref<f32>)
                   outs(%arg1 : memref<64xf32>) dimensions = [0]
  return
}

// -----

// CHECK-LABEL: func.func @preserve_reduction_init_fill(
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[FILLED:.*]] = linalg.fill ins(%[[ZERO]] : f32)
// CHECK: %[[REDUCED:.*]] = linalg.reduce {{.*}} ins({{.*}} : tensor<4x8xf32>) outs(%[[FILLED]] : tensor<4xf32>) dimensions = [1]
// CHECK: return %[[REDUCED]]
func.func @preserve_reduction_init_fill(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %init_empty = tensor.empty() : tensor<4xf32>
  %init = linalg.fill ins(%zero : f32) outs(%init_empty : tensor<4xf32>) -> tensor<4xf32>
  %reduced = linalg.reduce ins(%arg0 : tensor<4x8xf32>) outs(%init : tensor<4xf32>) dimensions = [1]
    (%in: f32, %acc: f32) {
      %sum = arith.addf %acc, %in : f32
      linalg.yield %sum : f32
    }
  return %reduced : tensor<4xf32>
}

// -----

// CHECK-LABEL: func.func @inline_brc_with_splat_dense
// CHECK-DAG: %[[cst_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[cst_1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins({{.*}}, %[[cst_0]]
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins({{.*}}, %[[cst_1]]
func.func @inline_brc_with_splat_dense(%arg0: tensor<1x4x2047x2047xf32>) -> tensor<1x4x2047x2047xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x2047x2047xf32>
  %cst_1 = arith.constant dense<1.000000e+00> : tensor<f32>
  %1 = tensor.empty() : tensor<1x4x2047x2047xf32>
  %broadcasted = linalg.broadcast ins(%cst_0 : tensor<1x2047x2047xf32>) outs(%1 : tensor<1x4x2047x2047xf32>) dimensions = [1] 
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %broadcasted : tensor<1x4x2047x2047xf32>, tensor<1x4x2047x2047xf32>) 
                                                             outs(%1 : tensor<1x4x2047x2047xf32>) -> tensor<1x4x2047x2047xf32>
  %3 = tensor.empty() : tensor<1x4x2047x2047xf32>
  %4 = linalg.broadcast ins(%cst_1 : tensor<f32>) outs(%3 : tensor<1x4x2047x2047xf32>) dimensions = [0, 1, 2, 3] 
  %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %4 : tensor<1x4x2047x2047xf32>, tensor<1x4x2047x2047xf32>) 
                                                             outs(%3 : tensor<1x4x2047x2047xf32>) -> tensor<1x4x2047x2047xf32>
  return %5 : tensor<1x4x2047x2047xf32>
}

// -----

// CHECK-LABEL: func.func @inline_brc_skip_sel_mask
// CHECK: linalg.fill
func.func @inline_brc_skip_sel_mask() -> tensor<64xf32> {
    %cst = arith.constant 1 : i1
    %0 = tensor.empty() : tensor<64xi1>
    %1 = tensor.empty() : tensor<64xf32>
    %2 = tensor.empty() : tensor<64xf32>
    %3 = linalg.fill ins(%cst : i1) outs(%0 : tensor<64xi1>) -> tensor<64xi1>
    %4 = tensor.empty() : tensor<64xf32>
    %5 = hfusion.select ins(%3, %1, %2 : tensor<64xi1>, tensor<64xf32>, tensor<64xf32>) outs(%4 : tensor<64xf32>) -> tensor<64xf32>
    return %5 : tensor<64xf32>
}
