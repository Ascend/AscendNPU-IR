// RUN: bishengir-opt %s -hfusion-fuse-ops="multi-kernel=true" -split-input-file | FileCheck %s
 
module {
  // CHECK-LABEL: mesh.mesh @mesh0
  mesh.mesh @mesh0(shape = 4)
  // CHECK-NEXT: func.func @mix_c2_demo_static_multi_MIX_C2_0({{.*}} attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<MIX_C2>}
  // CHECK-NEXT: tensor.empty
  // CHECK-NEXT: linalg.matmul
  // CHECK-NEXT: mesh.all_reduce
  // CHECK-LABEL: func.func @mix_c2_demo_static(
  func.func @mix_c2_demo_static(%arg0: tensor<128x4096xf32>, %arg1: tensor<4096x4096xf32>) -> tensor<128x4096xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    // CHECK-NEXT: call @mix_c2_demo_static_multi_MIX_C2_0(
    // CHECK-NEXT: return
    %0 = tensor.empty() : tensor<128x4096xf32>
    %1 = linalg.matmul ins(%arg0, %arg1 : tensor<128x4096xf32>, tensor<4096x4096xf32>) outs(%0 : tensor<128x4096xf32>) -> tensor<128x4096xf32>
    %all_reduce = mesh.all_reduce %1 on @mesh0 mesh_axes = [0] : tensor<128x4096xf32> -> tensor<128x4096xf32>
    return %all_reduce : tensor<128x4096xf32>
  }
}
 
// -----
 
module {
  // CHECK-LABEL: mesh.mesh @mesh0
  mesh.mesh @mesh0(shape = 2)
  // CHECK-NEXT: func.func @mix_c2_demo_dynamic_multi_MIX_C2_0({{.*}} attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<MIX_C2>}
  // CHECK-NEXT: arith.constant 0
  // CHECK-NEXT: tensor.dim
  // CHECK-NEXT: tensor.empty
  // CHECK-NEXT: linalg.matmul_transpose_b
  // CHECK-NEXT: mesh.all_reduce
  // CHECK-LABEL: func.func @mix_c2_demo_dynamic(
  func.func @mix_c2_demo_dynamic(%arg0: tensor<?xi64>, %arg1: tensor<?x2048xf16>, %arg2: tensor<4096x2048xf16>) -> tensor<?x4096xf16> attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    // CHECK-NEXT: call @mix_c2_demo_dynamic_multi_MIX_C2_0(
    // CHECK-NEXT: return
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg1, %c0 : tensor<?x2048xf16>
    %0 = tensor.empty(%dim) : tensor<?x4096xf16>
    %1 = linalg.matmul_transpose_b ins(%arg1, %arg2 : tensor<?x2048xf16>, tensor<4096x2048xf16>) outs(%0 : tensor<?x4096xf16>) -> tensor<?x4096xf16>
    %all_reduce = mesh.all_reduce %1 on @mesh0 mesh_axes = [0] : tensor<?x4096xf16> -> tensor<?x4096xf16>
    return %all_reduce : tensor<?x4096xf16>
  }
}