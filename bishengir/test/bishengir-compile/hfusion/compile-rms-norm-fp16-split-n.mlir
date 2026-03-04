// RUN: bishengir-compile -enable-lir-compile=false -enable-hfusion-compile=true -block-dim=40 %s | FileCheck %s

// CHECK: LLVMDialectModule
#map = affine_map<(d0) -> (d0)>
module attributes {transform.with_named_sequence} {
func.func @rms_norm_f16(%x: tensor<400x4096xf16>, %gamma: tensor<4096xf16>, %out0: tensor<400x4096xf16>, %out1: tensor<400xf16>)
        -> (tensor<400x4096xf16>, tensor<400xf16>)
   attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index

      // cast x
      %x_f32 = tensor.empty() : tensor<400x4096xf32>
      %x1 = hfusion.cast {mode = #hfusion.round_mode<rint>}
        ins(%x : tensor<400x4096xf16>)
        outs(%x_f32 : tensor<400x4096xf32>) -> tensor<400x4096xf32>

      // cast gamma
      %gamma_f32 = tensor.empty() : tensor<4096xf32>
      %gamma1 = hfusion.cast {mode = #hfusion.round_mode<rint>}
        ins(%gamma : tensor<4096xf16>)
        outs(%gamma_f32 : tensor<4096xf32>) -> tensor<4096xf32>

      %dim0 = tensor.dim %x1, %c0 : tensor<400x4096xf32>
      %dim1 = tensor.dim %x1, %c1 : tensor<400x4096xf32>

      %epsilon = arith.constant 1.000000e-01 : f32

      %empty0 = tensor.empty() : tensor<400x4096xf32>
      %squrare = linalg.elemwise_binary {__1__, fun = #linalg.binary_fn<mul>} ins(%x1, %x1 : tensor<400x4096xf32>, tensor<400x4096xf32>) outs(%empty0 : tensor<400x4096xf32>) -> tensor<400x4096xf32>

      %zero = arith.constant 0.0 : f32
      %empty1 = tensor.empty() : tensor<400xf32>
      %filled = linalg.fill ins(%zero : f32) outs(%empty1 : tensor<400xf32>) -> tensor<400xf32>
      %sum0 = linalg.reduce {arith.addf} ins(%squrare : tensor<400x4096xf32>) outs(%filled : tensor<400xf32>) dimensions = [1] {__2__}

      %empty2 = tensor.empty() : tensor<400xf32>
      %c1f = arith.constant 1.0 : f32
      %dim1ui = arith.index_castui %dim1 : index to i32
      %dim1f = arith.uitofp %dim1ui : i32 to f32
      %avg_factor = arith.divf %c1f, %dim1f : f32
      %mean = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>, __3__} ins(%sum0, %avg_factor : tensor<400xf32>, f32) outs(%empty2 : tensor<400xf32>) -> tensor<400xf32>

      %empty3 = tensor.empty() : tensor<400xf32>
      %sum1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>, __4__} ins(%mean, %epsilon : tensor<400xf32>, f32) outs(%empty3 : tensor<400xf32>) -> tensor<400xf32>

      %empty4 = tensor.empty() : tensor<400xf32>
      %sqrt = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>, __5__} ins(%sum1 : tensor<400xf32>) outs(%empty4 : tensor<400xf32>) -> tensor<400xf32>

      %empty41 = tensor.empty() : tensor<400xf32>
      %rsqrt = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>, __6__} ins(%sqrt : tensor<400xf32>) outs(%empty41 : tensor<400xf32>) -> tensor<400xf32>

      %empty6 = tensor.empty() : tensor<400x4096xf32>
      %brc_rsqrt = linalg.broadcast ins(%rsqrt : tensor<400xf32>) outs(%empty6 : tensor<400x4096xf32>) dimensions = [1] {__7__}

      %empty7 = tensor.empty() : tensor<400x4096xf32>
      %mul0 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>, __8__} ins(%brc_rsqrt, %x1 : tensor<400x4096xf32>, tensor<400x4096xf32>) outs(%empty7 : tensor<400x4096xf32>) -> tensor<400x4096xf32>

      %empty8 = tensor.empty() : tensor<400x4096xf32>
      %brc_gamma = linalg.broadcast ins(%gamma1 : tensor<4096xf32>) outs(%empty8 : tensor<400x4096xf32>) dimensions = [0] {__9__}

      %empty9 = tensor.empty() : tensor<400x4096xf32>
      %mul1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>, __10__} ins(%brc_gamma, %mul0 : tensor<400x4096xf32>, tensor<400x4096xf32>) outs(%empty9 : tensor<400x4096xf32>) -> tensor<400x4096xf32>

      // cast mul1
      %mul1_out = hfusion.cast {mode = #hfusion.round_mode<rint>}
        ins(%mul1 : tensor<400x4096xf32>)
        outs(%out0 : tensor<400x4096xf16>) -> tensor<400x4096xf16>

      // cast rsqrt
      %rsqrt_out = hfusion.cast {mode = #hfusion.round_mode<rint>}
        ins(%rsqrt : tensor<400xf32>)
        outs(%out1 : tensor<400xf16>) -> tensor<400xf16>

      return %mul1_out, %rsqrt_out : tensor<400x4096xf16>, tensor<400xf16>
   }
}
