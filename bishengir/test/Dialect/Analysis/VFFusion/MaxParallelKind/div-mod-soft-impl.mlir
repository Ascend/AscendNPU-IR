// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="fusion-mode=max-parallel" --split-input-file %s | FileCheck %s

// `shouldSkipFusionPair` keeps a soft `linalg.elemwise_binary<div>` (non-f16/f32)
// out of the VF of an `hfusion.elemwise_binary<mod>`. div feeds mod (blocked)
// and mul (fused); mod feeds add (fused). mul is emitted right after div so
// fusing div+mul does not strand mod (a div user) between them
// (hasInvalidDependencyIfFused). div and mod land in two separate outlined
// VFs: fused_0 holds div (+ mul), fused_1 holds mod (+ add).

// CHECK-LABEL: func.func private @div_mod_skip_i32_fused_0(
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK-NOT: hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
// CHECK-LABEL: func.func private @div_mod_skip_i32_fused_1(
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
// CHECK-NOT: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK-LABEL: func.func @div_mod_skip_i32(
// CHECK: {{(func\.)?call}} @div_mod_skip_i32_fused_
// CHECK: {{(func\.)?call}} @div_mod_skip_i32_fused_
// CHECK-NOT: {{(func\.)?call}} @div_mod_skip_i32_fused_
func.func @div_mod_skip_i32(%arg0: tensor<4xi32>, %arg1: i32, %arg2: i32)
    -> (tensor<4xi32>, tensor<4xi32>) {
  %c512 = arith.constant 512 : i32
  %empty = tensor.empty() : tensor<4xi32>
  %div = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %c512 : tensor<4xi32>, i32) outs(%empty : tensor<4xi32>) -> tensor<4xi32>
  %mul = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%div, %arg1 : tensor<4xi32>, i32) outs(%empty : tensor<4xi32>) -> tensor<4xi32>
  %mod = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%div, %c512 : tensor<4xi32>, i32) outs(%empty : tensor<4xi32>) -> tensor<4xi32>
  %add = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%mod, %arg2 : tensor<4xi32>, i32) outs(%empty : tensor<4xi32>) -> tensor<4xi32>
  return %mul, %add : tensor<4xi32>, tensor<4xi32>
}

// -----

// Same topology but f32: hardware float div is safe to keep with mod, so
// `shouldSkipFusionPair` does not apply and all four ops fuse into a single VF
// that contains both `binary_fn<div>` and `binary_fn<mod>`.

// CHECK-LABEL: func.func private @div_mod_fuse_f32_fused_0(
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
// CHECK-LABEL: func.func @div_mod_fuse_f32(
func.func @div_mod_fuse_f32(%arg0: tensor<4xf32>, %arg1: f32, %arg2: f32)
    -> (tensor<4xf32>, tensor<4xf32>) {
  %c512 = arith.constant 5.120000e+02 : f32
  %empty = tensor.empty() : tensor<4xf32>
  %div = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %c512 : tensor<4xf32>, f32) outs(%empty : tensor<4xf32>) -> tensor<4xf32>
  %mul = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%div, %arg1 : tensor<4xf32>, f32) outs(%empty : tensor<4xf32>) -> tensor<4xf32>
  %mod = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%div, %c512 : tensor<4xf32>, f32) outs(%empty : tensor<4xf32>) -> tensor<4xf32>
  %add = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%mod, %arg2 : tensor<4xf32>, f32) outs(%empty : tensor<4xf32>) -> tensor<4xf32>
  return %mul, %add : tensor<4xf32>, tensor<4xf32>
}
