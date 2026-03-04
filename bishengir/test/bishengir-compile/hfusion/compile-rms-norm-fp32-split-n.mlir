// RUN: bishengir-compile -enable-lir-compile=false -enable-hfusion-compile  -block-dim=40 %s | FileCheck %s

// CHECK: LLVMDialectModule
#map = affine_map<(d0) -> (d0)>
module {
func.func @rms_norm_f32(%x: tensor<256x128xf32>, %gamma: tensor<128xf32>, %out0: tensor<256x128xf32>, %out1: tensor<256xf32>) -> (tensor<256x128xf32>, tensor<256xf32>)
   attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %dim0 = tensor.dim %x, %c0 : tensor<256x128xf32>
      %dim1 = tensor.dim %x, %c1 : tensor<256x128xf32>

      %epsilon = arith.constant 1.000000e-01 : f32

      %empty0 = tensor.empty() : tensor<256x128xf32>
      %squrare = linalg.elemwise_binary {__1__, fun = #linalg.binary_fn<mul>} ins(%x, %x : tensor<256x128xf32>, tensor<256x128xf32>) outs(%empty0 : tensor<256x128xf32>) -> tensor<256x128xf32>

      %zero = arith.constant 0.0 : f32
      %empty1 = tensor.empty() : tensor<256xf32>
      %filled = linalg.fill ins(%zero : f32) outs(%empty1 : tensor<256xf32>) -> tensor<256xf32>
      %sum0 = linalg.reduce {arith.addf} ins(%squrare : tensor<256x128xf32>) outs(%filled : tensor<256xf32>) dimensions = [1] {__2__}

      %empty2 = tensor.empty() : tensor<256xf32>
      %c1f = arith.constant 1.0 : f32
      %dim1ui = arith.index_castui %dim1 : index to i32
      %dim1f = arith.uitofp %dim1ui : i32 to f32
      %avg_factor = arith.divf %c1f, %dim1f : f32
      %mean = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>, __3__} ins(%sum0, %avg_factor : tensor<256xf32>, f32) outs(%empty2 : tensor<256xf32>) -> tensor<256xf32>

      %empty3 = tensor.empty() : tensor<256xf32>
      %sum1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>, __4__} ins(%mean, %epsilon : tensor<256xf32>, f32) outs(%empty3 : tensor<256xf32>) -> tensor<256xf32>

      %empty4 = tensor.empty() : tensor<256xf32>
      %sqrt = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>, __5__} ins(%sum1 : tensor<256xf32>) outs(%empty4 : tensor<256xf32>) -> tensor<256xf32>

      %rsqrt = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>, __6__} ins(%sqrt : tensor<256xf32>) outs(%out1 : tensor<256xf32>) -> tensor<256xf32>

      %empty6 = tensor.empty() : tensor<256 x128xf32>
      %brc_rsqrt = linalg.broadcast ins(%rsqrt : tensor<256xf32>) outs(%empty6 : tensor<256x128xf32>) dimensions = [1] {__7__}

      %empty7 = tensor.empty() : tensor<256x128xf32>
      %mul0 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>, __8__} ins(%brc_rsqrt, %x : tensor<256x128xf32>, tensor<256x128xf32>) outs(%empty7 : tensor<256x128xf32>) -> tensor<256x128xf32>

      %empty8 = tensor.empty() : tensor<256x128xf32>
      %brc_gamma = linalg.broadcast ins(%gamma : tensor<128xf32>) outs(%empty8 : tensor<256x128xf32>) dimensions = [0] {__9__}

      %mul1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>, __10__} ins(%brc_gamma, %mul0 : tensor<256x128xf32>, tensor<256x128xf32>) outs(%out0 : tensor<256x128xf32>) -> tensor<256x128xf32>

      return %mul1, %rsqrt : tensor<256x128xf32>, tensor<256xf32>
   }
}
