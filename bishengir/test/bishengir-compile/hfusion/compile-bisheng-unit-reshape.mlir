// XFAIL: *
// RUN: bishengir-compile -enable-lir-compile=false -enable-hfusion-compile=true -block-dim=20 %s | FileCheck %s

// CHECK-LABEL: LLVMDialectModule
module {
  func.func @mlir_fused__to_copy_npu_dynamic_quant_118(%arg0: tensor<1x?x32x128xf16>, %arg1: i64, %arg2: i64) -> (tensor<?x4096xbf16>, tensor<1x?x4096xbf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c-1 = arith.constant -1 : index
    %c4096_i64 = arith.constant 4096 : i64
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<1x?x32x128xf16> into tensor<?x4096xf16>
    %dim = tensor.dim %arg0, %c1 : tensor<1x?x32x128xf16>
    %0 = arith.index_cast %dim : index to i64
    %1 = arith.cmpi slt, %0, %c0_i64 : i64
    %2 = arith.select %1, %0, %c0_i64 : i64
    %3 = arith.index_cast %2 : i64 to index
    %4 = arith.index_cast %arg2 : i64 to index
    %5 = arith.cmpi slt, %4, %c0 : index
    %6 = arith.addi %4, %dim : index
    %7 = arith.select %5, %6, %4 : index
    %8 = arith.cmpi slt, %7, %c0 : index
    %9 = arith.select %8, %c-1, %7 : index
    %10 = arith.cmpi sgt, %9, %dim : index
    %11 = arith.select %10, %dim, %9 : index
    %12 = arith.subi %11, %3 : index
    %13 = arith.cmpi slt, %12, %c0 : index
    %14 = arith.select %13, %c0, %12 : index
    %extracted_slice = tensor.extract_slice %collapsed[%3, 0] [%14, 4096] [1, 1] : tensor<?x4096xf16> to tensor<?x4096xf16>
    %expanded = tensor.expand_shape %extracted_slice [[0, 1], [2]] output_shape [1, %14, 4096] : tensor<?x4096xf16> into tensor<1x?x4096xf16>
    %15 = arith.index_cast %14 : index to i64
    %16 = arith.muli %15, %c4096_i64 : i64
    %17 = arith.divui %16, %arg2 : i64
    %from_elements = tensor.from_elements %arg2, %17 : tensor<2xi64>
    %reshape = tensor.reshape %expanded(%from_elements) : (tensor<1x?x4096xf16>, tensor<2xi64>) -> tensor<?x4096xf16>
    %dim_0 = tensor.dim %reshape, %c0 : tensor<?x4096xf16>
    %18 = tensor.empty(%dim_0) : tensor<?x4096xf32>
    %19 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%reshape : tensor<?x4096xf16>) outs(%18 : tensor<?x4096xf32>) -> tensor<?x4096xf32>
    %20 = tensor.empty(%dim_0) : tensor<?x4096xbf16>
    %21 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%19 : tensor<?x4096xf32>) outs(%20 : tensor<?x4096xbf16>) -> tensor<?x4096xbf16>
    %dim_1 = tensor.dim %expanded, %c1 : tensor<1x?x4096xf16>
    %22 = tensor.empty(%dim_1) : tensor<1x?x4096xf32>
    %23 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%expanded : tensor<1x?x4096xf16>) outs(%22 : tensor<1x?x4096xf32>) -> tensor<1x?x4096xf32>
    %24 = tensor.empty(%dim_1) : tensor<1x?x4096xbf16>
    %25 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%23 : tensor<1x?x4096xf32>) outs(%24 : tensor<1x?x4096xbf16>) -> tensor<1x?x4096xbf16>
    return %21, %25 : tensor<?x4096xbf16>, tensor<1x?x4096xbf16>
  }
}