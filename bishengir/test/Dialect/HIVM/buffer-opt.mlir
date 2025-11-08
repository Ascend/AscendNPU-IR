// RUN: bishengir-opt %s -hivm-opt-func-output -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func.func @testA_0(
func.func @testA_0(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>) -> (memref<?x?xf32>, memref<?x?xf32>) attributes {fusion_kind = #hfusion.fusion_kind<MIX_CV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
  %dim_0 = memref.dim %arg0, %c1 : memref<?x?xf32>
  %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
  %alloc_1 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
  %alloc_2 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
  linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%arg1 : memref<?x?xf32>) outs(%alloc : memref<?x?xf32>)
  linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%arg1, %alloc : memref<?x?xf32>, memref<?x?xf32>) outs(%alloc_1 : memref<?x?xf32>)
  linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%alloc_1 : memref<?x?xf32>) outs(%alloc_2 : memref<?x?xf32>)
  linalg.matmul ins(%arg1, %alloc_2 : memref<?x?xf32>, memref<?x?xf32>) outs(%arg2 : memref<?x?xf32>)
  linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%alloc_2 : memref<?x?xf32>) outs(%arg3 : memref<?x?xf32>)
// CHECK: return
// CHECK-NOT: memref
  return %arg2, %arg3 : memref<?x?xf32>, memref<?x?xf32>
}
// CHECK-LABEL: func.func @testA(
func.func @testA(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) -> (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?x?xf32>, memref<?x?xf32>, memref<?x?x?x?xf32>, memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
  %dim_0 = memref.dim %arg0, %c1 : memref<?x?xf32>
  %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
  %alloc_1 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
// CHECK: call @testA_0(
// CHECK-SAME: -> ()
  %0:2 = call @testA_0(%arg0, %arg2, %alloc, %alloc_1) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> (memref<?x?xf32>, memref<?x?xf32>)
  %alloc_2 = memref.alloc(%dim, %dim, %dim_0) {alignment = 64 : i64} : memref<?x?x?xf32>
  linalg.broadcast ins(%arg2 : memref<?x?xf32>) outs(%alloc_2 : memref<?x?x?xf32>) dimensions = [0]
  %alloc_3 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
  linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%arg0 : memref<?x?xf32>) outs(%alloc_3 : memref<?x?xf32>)
  %alloc_4 = memref.alloc(%dim, %dim, %dim_0, %dim) {alignment = 64 : i64} : memref<?x?x?x?xf32>
  linalg.broadcast ins(%alloc_2 : memref<?x?x?xf32>) outs(%alloc_4 : memref<?x?x?x?xf32>) dimensions = [3]
  %alloc_5 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
  linalg.transpose ins(%alloc_3 : memref<?x?xf32>) outs(%alloc_5 : memref<?x?xf32>) permutation = [0, 1]
  return %arg1, %0#0, %0#1, %alloc_2, %alloc_3, %alloc_4, %alloc_5 : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?x?xf32>, memref<?x?xf32>, memref<?x?x?x?xf32>, memref<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @testB_0(
func.func @testB_0(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32>, %arg5: memref<?x?xf32>) -> (memref<?x?xf32>, memref<?x?xf32>) attributes {fusion_kind = #hfusion.fusion_kind<MIX_CV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
  %dim_0 = memref.dim %arg0, %c1 : memref<?x?xf32>
  %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
  linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg1 : memref<?x?xf32>) outs(%alloc : memref<?x?xf32>)
  linalg.matmul ins(%alloc, %arg2 : memref<?x?xf32>, memref<?x?xf32>) outs(%arg3 : memref<?x?xf32>)
  linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>, min_signed} ins(%arg4, %alloc : memref<?x?xf32>, memref<?x?xf32>) outs(%arg5 : memref<?x?xf32>)
// CHECK: return
// CHECK-NOT: memref
  return %arg3, %arg5 : memref<?x?xf32>, memref<?x?xf32>
}
// CHECK-LABEL: func.func @testB(
func.func @testB(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) -> (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?x?xf32>, memref<?x?xf32>, memref<?x?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
  %dim_0 = memref.dim %arg0, %c1 : memref<?x?xf32>
  %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
  linalg.transpose ins(%arg2 : memref<?x?xf32>) outs(%alloc : memref<?x?xf32>) permutation = [0, 1]
  %alloc_1 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
  linalg.transpose ins(%arg2 : memref<?x?xf32>) outs(%alloc_1 : memref<?x?xf32>) permutation = [0, 1]
  %alloc_2 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
  linalg.matmul ins(%alloc_1, %alloc_1 : memref<?x?xf32>, memref<?x?xf32>) outs(%alloc_2 : memref<?x?xf32>)
  %alloc_3 = memref.alloc(%dim, %dim_0, %dim) {alignment = 64 : i64} : memref<?x?x?xf32>
  linalg.broadcast ins(%arg2 : memref<?x?xf32>) outs(%alloc_3 : memref<?x?x?xf32>) dimensions = [2]
  %alloc_4 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
  linalg.transpose ins(%alloc_2 : memref<?x?xf32>) outs(%alloc_4 : memref<?x?xf32>) permutation = [0, 1]
  %alloc_5 = memref.alloc(%dim, %dim_0, %dim) {alignment = 64 : i64} : memref<?x?x?xf32>
  linalg.broadcast ins(%alloc_2 : memref<?x?xf32>) outs(%alloc_5 : memref<?x?x?xf32>) dimensions = [2]
  %alloc_6 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
  %alloc_7 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
// CHECK: call @testB_0(
// CHECK-SAME: -> ()
  %0:2 = call @testB_0(%arg0, %arg2, %alloc_2, %alloc_6, %alloc_1, %alloc_7) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> (memref<?x?xf32>, memref<?x?xf32>)
  return %arg0, %arg1, %alloc, %alloc_3, %alloc_4, %alloc_5, %0#0, %0#1 : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?x?xf32>, memref<?x?xf32>, memref<?x?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @testC_0(
func.func @testC_0(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32>) -> (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) attributes {fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
  linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%arg0, %arg0 : memref<?x?xf32>, memref<?x?xf32>) outs(%arg1 : memref<?x?xf32>)
  linalg.elemwise_binary {fun = #linalg.binary_fn<mul>, mul} ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>) outs(%arg2 : memref<?x?xf32>)
  linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg2 : memref<?x?xf32>) outs(%arg3 : memref<?x?xf32>)
  linalg.elemwise_binary {fun = #linalg.binary_fn<sub>, sub} ins(%arg0, %arg3 : memref<?x?xf32>, memref<?x?xf32>) outs(%arg4 : memref<?x?xf32>)
// CHECK: return
// CHECK-NOT: memref
  return %arg1, %arg2, %arg3, %arg4 : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
}

// CHECK-LABEL: func.func @testC(
func.func @testC(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) -> (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
  %dim_0 = memref.dim %arg0, %c1 : memref<?x?xf32>
  %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
  %alloc_1 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
  %alloc_2 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
  %alloc_3 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
  %alloc_4 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
// CHECK: call @testC_0(
// CHECK-SAME: -> ()
  %0:4 = call @testC_0(%arg2, %alloc, %alloc_1, %alloc_2, %alloc_4) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>)
  linalg.matmul ins(%0#2, %0#1 : memref<?x?xf32>, memref<?x?xf32>) outs(%alloc_3 : memref<?x?xf32>)
  return %arg0, %arg1, %0#0, %alloc_3, %0#3 : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @testD_0(
func.func @testD_0(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg5: index, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32>) -> (memref<?x?xf32>, memref<?x?xf32>, index, memref<?x?xf32>, memref<?x?xf32>) attributes {fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
  linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%arg0, %arg0 : memref<?x?xf32>, memref<?x?xf32>) outs(%arg1 : memref<?x?xf32>)
  linalg.elemwise_binary {fun = #linalg.binary_fn<mul>, mul} ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>) outs(%arg2 : memref<?x?xf32>)
  linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg2 : memref<?x?xf32>) outs(%arg3 : memref<?x?xf32>)
  linalg.elemwise_binary {fun = #linalg.binary_fn<sub>, sub} ins(%arg0, %arg3 : memref<?x?xf32>, memref<?x?xf32>) outs(%arg4 : memref<?x?xf32>)
// CHECK: return
// CHECK-NOT: memref
  return %arg1, %arg2, %arg5, %arg3, %arg4 : memref<?x?xf32>, memref<?x?xf32>, index, memref<?x?xf32>, memref<?x?xf32>
}

// CHECK-LABEL: func.func @testD(
func.func @testD(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) -> (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
  %dim_0 = memref.dim %arg0, %c1 : memref<?x?xf32>
  %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
  %alloc_1 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
  %alloc_2 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
  %alloc_3 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
  %alloc_4 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
// CHECK: call @testD_0(
// CHECK-SAME: -> index
  %0:5 = call @testD_0(%arg2, %alloc, %c1, %alloc_1, %alloc_2, %alloc_4) : (memref<?x?xf32>, memref<?x?xf32>, index, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> (memref<?x?xf32>, memref<?x?xf32>, index, memref<?x?xf32>, memref<?x?xf32>)
  linalg.matmul ins(%0#4, %0#1 : memref<?x?xf32>, memref<?x?xf32>) outs(%alloc_3 : memref<?x?xf32>)
  return %arg0, %arg1, %0#0, %alloc_3, %0#3, %0#2 : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index
}

// CHECK-LABEL: func.func @testD_caller2(
func.func @testD_caller2(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) -> (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, memref<?x?xf32>, memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
  %dim_0 = memref.dim %arg0, %c1 : memref<?x?xf32>
  %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
  %alloc_1 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
  %alloc_2 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
  %alloc_3 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
  %alloc_4 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
// CHECK: call @testD_0(
// CHECK-SAME: -> index
  %0:5 = call @testD_0(%arg2, %alloc, %c1, %alloc_1, %alloc_2, %alloc_4) : (memref<?x?xf32>, memref<?x?xf32>, index, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> (memref<?x?xf32>, memref<?x?xf32>, index, memref<?x?xf32>, memref<?x?xf32>)
  %2:5 = call @testD_0(%arg1, %alloc, %c1, %alloc_1, %alloc_2, %alloc_4) : (memref<?x?xf32>, memref<?x?xf32>, index, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> (memref<?x?xf32>, memref<?x?xf32>, index, memref<?x?xf32>, memref<?x?xf32>)
  linalg.matmul ins(%0#4, %0#1 : memref<?x?xf32>, memref<?x?xf32>) outs(%alloc_3 : memref<?x?xf32>)
  return %arg0, %arg1, %0#0, %alloc_3, %0#3, %0#2, %2#3, %2#1 : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, memref<?x?xf32>, memref<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @testE_0(
func.func @testE_0(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg5: index, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32>) -> (memref<?x?xf32>, memref<?x?xf32>, index, memref<?x?xf32>, memref<?x?xf32>) attributes {fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
  linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%arg0, %arg0 : memref<?x?xf32>, memref<?x?xf32>) outs(%arg1 : memref<?x?xf32>)
  linalg.elemwise_binary {fun = #linalg.binary_fn<mul>, mul} ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>) outs(%arg2 : memref<?x?xf32>)
  linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg2 : memref<?x?xf32>) outs(%arg3 : memref<?x?xf32>)
  linalg.elemwise_binary {fun = #linalg.binary_fn<sub>, sub} ins(%arg0, %arg3 : memref<?x?xf32>, memref<?x?xf32>) outs(%arg4 : memref<?x?xf32>)
// CHECK: return
// CHECK-NOT: memref
  return %arg1, %arg2, %arg5, %arg3, %arg4 : memref<?x?xf32>, memref<?x?xf32>, index, memref<?x?xf32>, memref<?x?xf32>
}

// CHECK-LABEL: func.func @testE(
func.func @testE(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) -> (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
  %dim_0 = memref.dim %arg0, %c1 : memref<?x?xf32>
  %alloc_3 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
// CHECK-NOT: call @testE_0(
  linalg.matmul ins(%arg1, %arg2 : memref<?x?xf32>, memref<?x?xf32>) outs(%alloc_3 : memref<?x?xf32>)
// CHECK: return
// CHECK-SAME: memref<?x?xf32> 
// CHECK-NOT: ,
  return %arg0, %alloc_3, %arg1 : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
}
