// RUN: bishengir-opt  --test-assign-fusion-kind --fusion-kind="MIX_CV" -hfusion-fuse-ops="max-horizontal-fusion-size=-1" -split-input-file %s | FileCheck %s --check-prefix=MIX-CV

// MIX-CV-LABEL: func.func @testChain_0(
// MIX-CV-SAME: MIX_CV
// MIX-CV: matmul
// MIX-CV: elemwise_unary
// MIX-CV: return
func.func @testChain(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> (tensor<8x8xf32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %0 = tensor.empty () : tensor<8x8xf32>
  %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>} ins(%arg0 : tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  %2 = tensor.empty () : tensor<8x8xf32>
  %3 = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>} ins(%1 : tensor<8x8xf32>) outs(%2 : tensor<8x8xf32>) -> tensor<8x8xf32>
  %4 = tensor.empty () : tensor<8x8xf32>
  %5 = linalg.matmul ins(%3, %3 : tensor<8x8xf32>, tensor<8x8xf32>) outs(%4 : tensor<8x8xf32>) -> tensor<8x8xf32>
  %6 = tensor.empty () : tensor<8x8xf32>
  %7 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%5 : tensor<8x8xf32>) outs(%6 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %7 : tensor<8x8xf32>
}

// -----

// MIX-CV-LABEL: func.func @testChain2_0(
// MIX-CV-SAME: MIX_CV
// MIX-CV: matmul
// MIX-CV: elemwise_unary
// MIX-CV: elemwise_unary
// MIX-CV: elemwise_unary
// MIX-CV: elemwise_unary
// MIX-CV: return
// MIX-CV-LABEL: func.func @testChain2(
// MIX-CV: transpose
// MIX-CV: matmul
// MIX-CV: call
// MIX-CV: return
func.func @testChain2(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> (tensor<8x8xf32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %0 = tensor.empty () : tensor<8x8xf32>
  %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%arg0 : tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  %2 = tensor.empty () : tensor<8x8xf32>
  %3 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%1 : tensor<8x8xf32>) outs(%2 : tensor<8x8xf32>) -> tensor<8x8xf32>
  %4 = tensor.empty () : tensor<8x8xf32>
  %5 = linalg.matmul ins(%3, %3 : tensor<8x8xf32>, tensor<8x8xf32>) outs(%4 : tensor<8x8xf32>) -> tensor<8x8xf32>
  %6 = tensor.empty () : tensor<8x8xf32>
  %7 = linalg.transpose ins(%5 : tensor<8x8xf32>) outs(%6 : tensor<8x8xf32>) permutation = [0, 1]
  %8 = tensor.empty () : tensor<8x8xf32>
  %9 = linalg.matmul ins(%7, %7 : tensor<8x8xf32>, tensor<8x8xf32>) outs(%8 : tensor<8x8xf32>) -> tensor<8x8xf32>
  %10 = tensor.empty () : tensor<8x8xf32>
  %11 = linalg.matmul ins(%9, %9 : tensor<8x8xf32>, tensor<8x8xf32>) outs(%10 : tensor<8x8xf32>) -> tensor<8x8xf32>
  %12 = tensor.empty () : tensor<8x8xf32>
  %13 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%11 : tensor<8x8xf32>) outs(%12 : tensor<8x8xf32>) -> tensor<8x8xf32>
  %14 = tensor.empty () : tensor<8x8xf32>
  %15 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%13 : tensor<8x8xf32>) outs(%14 : tensor<8x8xf32>) -> tensor<8x8xf32>
  %16 = tensor.empty () : tensor<8x8xf32>
  %17 = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>} ins(%15 : tensor<8x8xf32>) outs(%16 : tensor<8x8xf32>) -> tensor<8x8xf32>
  %18 = tensor.empty () : tensor<8x8xf32>
  %19 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%17 : tensor<8x8xf32>) outs(%18 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %19 : tensor<8x8xf32>
}


// -----

// MIX-CV-LABEL: func.func @testComplex_0(
// MIX-CV-SAME: MIX_CV
// MIX-CV: elemwise_unary
// MIX-CV: unary
// MIX-CV: matmul
// MIX-CV: matmul
// MIX-CV: unary
// MIX-CV: unary
// MIX-CV: unary
// MIX-CV: unary
// MIX-CV: unary
// MIX-CV: return
// MIX-CV-LABEL: func.func @testComplex_1(
// MIX-CV-SAME: MIX_CV
// MIX-CV: matmul
// MIX-CV: elemwise_binary
// MIX-CV: matmul
// MIX-CV: matmul
// MIX-CV: elemwise_binary
// MIX-CV: return
// MIX-CV-LABEL: func.func @testComplex_2(
// MIX-CV-SAME: MIX_CV
// MIX-CV: matmul
// MIX-CV: matmul
// MIX-CV: unary
// MIX-CV: return
// MIX-CV-NOT: testComplex_3(
// MIX-CV-LABEL: func.func @testComplex(
// MIX-CV: reduce
// MIX-CV: broadcast
// MIX-CV: broadcast
// MIX-CV: transpose
// MIX-CV: transpose
// MIX-CV: return

func.func @testComplex(%arg0: tensor<7x7xf32>, %arg1: tensor<7x7xf32>, %arg2: tensor<7x7xf32>) -> (tensor<7xf32>, tensor<?x7x7xf32>, tensor<7x7xf32>, tensor<?x7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<?x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<?x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<7x7xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<7x7xf32>
  %2 = tensor.empty () : tensor<7xf32>
  %3 = linalg.reduce {arith.addf} ins(%arg0 : tensor<7x7xf32>) outs(%2: tensor<7xf32>) dimensions = [1]
  %4 = tensor.empty () : tensor<7x7xf32>
  %5 = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>} ins(%arg1 : tensor<7x7xf32>) outs(%4 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %6 = tensor.empty (%0) : tensor<?x7x7xf32>
  %7 = linalg.broadcast ins(%arg1 : tensor<7x7xf32>) outs(%6: tensor<?x7x7xf32>) dimensions = [0]
  %8 = tensor.empty () : tensor<7x7xf32>
  %9 = linalg.elemwise_binary {min_signed, fun = #linalg.binary_fn<min_signed>} ins(%arg1, %arg2 : tensor<7x7xf32>, tensor<7x7xf32>) outs(%8 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %10 = tensor.empty (%0) : tensor<?x7xf32>
  %11 = linalg.broadcast ins(%3 : tensor<7xf32>) outs(%10: tensor<?x7xf32>) dimensions = [0]
  %12 = tensor.empty () : tensor<7x7xf32>
  %13 = linalg.transpose ins(%arg2 : tensor<7x7xf32>) outs(%12 : tensor<7x7xf32>) permutation = [0, 1]
  %14 = tensor.empty (%0) : tensor<?x7x7xf32>
  %15 = linalg.broadcast ins(%9 : tensor<7x7xf32>) outs(%14: tensor<?x7x7xf32>) dimensions = [0]
  %16 = tensor.empty () : tensor<7x7xf32>
  %17 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%5 : tensor<7x7xf32>) outs(%16 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %18 = tensor.empty () : tensor<7x7xf32>
  %19 = linalg.transpose ins(%arg1 : tensor<7x7xf32>) outs(%18 : tensor<7x7xf32>) permutation = [0, 1]
  %20 = tensor.empty () : tensor<7x7xf32>
  %21 = linalg.matmul ins(%arg0, %arg0 : tensor<7x7xf32>, tensor<7x7xf32>) outs(%20 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %22 = tensor.empty () : tensor<7x7xf32>
  %23 = linalg.matmul ins(%13, %arg0 : tensor<7x7xf32>, tensor<7x7xf32>) outs(%22 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %24 = tensor.empty () : tensor<7x7xf32>
  %25 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%13 : tensor<7x7xf32>) outs(%24 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %26 = tensor.empty () : tensor<7x7xf32>
  %27 = linalg.matmul ins(%arg2, %5 : tensor<7x7xf32>, tensor<7x7xf32>) outs(%26 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %28 = tensor.empty () : tensor<7x7xf32>
  %29 = linalg.elemwise_binary {max_signed, fun = #linalg.binary_fn<max_signed>} ins(%27, %19 : tensor<7x7xf32>, tensor<7x7xf32>) outs(%28 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %30 = tensor.empty (%0) : tensor<?x7xf32>
  %31 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%11 : tensor<?x7xf32>) outs(%30 : tensor<?x7xf32>) -> tensor<?x7xf32>
  %32 = tensor.empty () : tensor<7x7xf32>
  %33 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%25 : tensor<7x7xf32>) outs(%32 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %34 = tensor.empty () : tensor<7x7xf32>
  %35 = linalg.matmul ins(%arg0, %21 : tensor<7x7xf32>, tensor<7x7xf32>) outs(%34 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %36 = tensor.empty () : tensor<7x7xf32>
  %37 = linalg.matmul ins(%17, %23 : tensor<7x7xf32>, tensor<7x7xf32>) outs(%36 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %38 = tensor.empty (%0) : tensor<?x7xf32>
  %39 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%31 : tensor<?x7xf32>) outs(%38 : tensor<?x7xf32>) -> tensor<?x7xf32>
  %40 = tensor.empty () : tensor<7x7xf32>
  %41 = linalg.elemwise_binary {mul, fun = #linalg.binary_fn<mul>} ins(%35, %arg1 : tensor<7x7xf32>, tensor<7x7xf32>) outs(%40 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %42 = tensor.empty () : tensor<7x7xf32>
  %43 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%23 : tensor<7x7xf32>) outs(%42 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %44 = tensor.empty () : tensor<7x7xf32>
  %45 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>} ins(%23 : tensor<7x7xf32>) outs(%44 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %46 = tensor.empty () : tensor<7x7xf32>
  %47 = linalg.matmul ins(%27, %13 : tensor<7x7xf32>, tensor<7x7xf32>) outs(%46 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %48 = tensor.empty () : tensor<7x7xf32>
  %49 = linalg.elemwise_binary {mul, fun = #linalg.binary_fn<mul>} ins(%13, %21 : tensor<7x7xf32>, tensor<7x7xf32>) outs(%48 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %50 = tensor.empty () : tensor<7x7xf32>
  %51 = linalg.transpose ins(%25 : tensor<7x7xf32>) outs(%50 : tensor<7x7xf32>) permutation = [0, 1]
  %52 = tensor.empty () : tensor<7x7xf32>
  %53 = linalg.matmul ins(%43, %33 : tensor<7x7xf32>, tensor<7x7xf32>) outs(%52 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %54 = tensor.empty (%0) : tensor<?x7xf32>
  %55 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%31 : tensor<?x7xf32>) outs(%54 : tensor<?x7xf32>) -> tensor<?x7xf32>
  %56 = tensor.empty () : tensor<7x7xf32>
  %57 = linalg.matmul ins(%arg1, %41 : tensor<7x7xf32>, tensor<7x7xf32>) outs(%56 : tensor<7x7xf32>) -> tensor<7x7xf32>
  %58 = tensor.empty () : tensor<7x7xf32>
  %59 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%47 : tensor<7x7xf32>) outs(%58 : tensor<7x7xf32>) -> tensor<7x7xf32>
  return %3, %7, %9, %15, %25, %29, %37, %39, %45, %49, %51, %53, %55, %57, %59 : tensor<7xf32>, tensor<?x7x7xf32>, tensor<7x7xf32>, tensor<?x7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<?x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>, tensor<?x7xf32>, tensor<7x7xf32>, tensor<7x7xf32>
}