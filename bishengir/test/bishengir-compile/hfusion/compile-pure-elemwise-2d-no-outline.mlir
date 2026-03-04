// RUN: bishengir-compile -enable-lir-compile=false -enable-hfusion-compile=true  -block-dim=20 %s | FileCheck %s

// CHECK: LLVMDialectModule
// CHECK-DAG: define dso_local void @add_mul_2d
func.func @add_mul_2d(%arg0: tensor<1024x1024xf32>, %arg1: tensor<1024x1024xf32>, %arg2: tensor<1024x1024xf32>, %arg3: tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
attributes {hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %1 = tensor.empty() : tensor<1024x1024xf32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%1 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>, sub} ins(%2, %arg2 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%arg3: tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  return %3 : tensor<1024x1024xf32>
}
