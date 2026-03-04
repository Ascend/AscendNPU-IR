// RUN: bishengir-compile -enable-lir-compile=false -enable-hfusion-compile=true -enable-hivm-inject-barrier-all-sync -block-dim=20 %s | FileCheck %s
// RUN: bishengir-compile -enable-lir-compile=false -enable-hfusion-compile=true -block-dim=20 %s | FileCheck %s

// CHECK: LLVMDialectModule
// CHECK-DAG: _mlir_ciface_vmul_1d_float
// CHECK-DAG: _mlir_ciface_vadd_1d_float
// CHECK-DAG: define dso_local void @add_mul_2d_0_0
func.func @add_mul_2d(%arg0: tensor<1024x1024xf32>, %arg1: tensor<1024x1024xf32>, %arg2: tensor<1024x1024xf32>, %arg3: tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
  %1 = tensor.empty() : tensor<1024x1024xf32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%1 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>, sub} ins(%2, %arg2 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%arg3: tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  return %3 : tensor<1024x1024xf32>
}
