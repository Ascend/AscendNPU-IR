// RUN: bishengir-opt %s -split-input-file \
// RUN:   -pass-pipeline="builtin.module(  \
// RUN:     canonicalize-ext,                  \
// RUN:     func.func(hfusion-outline-single-op))" | FileCheck %s

// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="ANY_PBR" --hfusion-fuse-ops -split-input-file %s | FileCheck %s --check-prefix=ANYPBR

// CHECK-LABEL: func.func @host_first_reduce_single_outlined_0_0
// CHECK-SAME: attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>}
// CHECK: linalg.reduce
// CHECK: return
// CHECK-LABEL: func.func @host_first_reduce
// CHECK-SAME: attributes {hacc.function_kind = #hacc.function_kind<HOST>}
// CHECK: call @host_first_reduce_single_outlined_0_0
// CHECK: return
func.func @host_first_reduce(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> 
attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %reduced = linalg.reduce { arith.addf } 
    ins(%arg0 : tensor<?x?xf32>) outs(%arg1 : tensor<?xf32>) dimensions = [0]
  return %reduced : tensor<?xf32>
}

// -----
// CHECK-LABEL: func.func @host_multi_ops_single_outlined_0
// CHECK-SAME: attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>}
// CHECK: linalg.elemwise_binary
// CHECK: return
// CHECK-LABEL: func.func @host_multi_ops_single_outlined_1
// CHECK-SAME: attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>}
// CHECK: linalg.reduce
// CHECK: return
// CHECK-LABEL: func.func @host_multi_ops
// CHECK-SAME: attributes {hacc.function_kind = #hacc.function_kind<HOST>}
// CHECK: call @host_multi_ops_single_outlined_0
// CHECK: call @host_multi_ops_single_outlined_1
// CHECK: return
func.func @host_multi_ops(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>, 
                          %0 : tensor<?x?xf32>, %3 : tensor<?xf32>) -> tensor<?xf32> 
attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} 
    ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = linalg.matmul ins(%arg2, %arg3: tensor<?x?xf32>, tensor<?x?xf32>) 
    outs(%arg2: tensor<?x?xf32>) -> tensor<?x?xf32>
  %reduced = linalg.reduce { arith.addf } 
    ins(%1 : tensor<?x?xf32>) outs(%3 : tensor<?xf32>) dimensions = [0]
  return %reduced : tensor<?xf32>
}

// -----

// ANYPBR-LABEL: func.func @test_fuse_reduces_0(
// ANYPBR: reduce
// ANYPBR: elemwise_unary
// ANYPBR: elemwise_unary
// ANYPBR: broadcast
// ANYPBR: elemwise_unary
// ANYPBR: elemwise_unary
// ANYPBR: reduce
// ANYPBR: broadcast
// ANYPBR: elemwise_unary
// ANYPBR: elemwise_unary
// ANYPBR-LABEL: func.func @test_fuse_reduces(
// ANYPBR: reduce
func.func @test_fuse_reduces(%arg0: tensor<?x?x?xf32>) -> (tensor<?x?xf32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 18 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %4 = linalg.reduce {arith.addf} ins(%arg0 : tensor<?x?x?xf32>) outs(%3: tensor<?x?xf32>) dimensions = [0]
  %5 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %6 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%4 : tensor<?x?xf32>) outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %7 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %8 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%6 : tensor<?x?xf32>) outs(%7 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %9 = tensor.empty(%0, %1, %c3) : tensor<?x?x?xf32>
  %10 = linalg.broadcast ins(%8 : tensor<?x?xf32>) outs(%9: tensor<?x?x?xf32>) dimensions = [2]
  %11 = tensor.empty(%0, %1, %c3) : tensor<?x?x?xf32>
  %12 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%10 : tensor<?x?x?xf32>) outs(%11 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %13 = tensor.empty(%0, %1, %c3) : tensor<?x?x?xf32>
  %14 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%12 : tensor<?x?x?xf32>) outs(%13 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %15 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %16 = linalg.reduce {arith.addf} ins(%14 : tensor<?x?x?xf32>) outs(%15: tensor<?x?xf32>) dimensions = [0]
  %17 = tensor.empty(%0, %1, %c3) : tensor<?x?x?xf32>
  %18 = linalg.broadcast ins(%16 : tensor<?x?xf32>) outs(%17: tensor<?x?x?xf32>) dimensions = [2]
  %19 = tensor.empty(%0, %1, %c3) : tensor<?x?x?xf32>
  %20 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%18 : tensor<?x?x?xf32>) outs(%19 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %21 = tensor.empty(%0, %1, %c3) : tensor<?x?x?xf32>
  %22 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%20 : tensor<?x?x?xf32>) outs(%21 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %23 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %24 = linalg.reduce {arith.addf} ins(%22 : tensor<?x?x?xf32>) outs(%23: tensor<?x?xf32>) dimensions = [1]
  return %24 : tensor<?x?xf32>
}
