// RUN: bishengir-opt -convert-hfusion-to-hivm %s | FileCheck %s
// RUN: bishengir-opt -convert-to-hivm-pipeline %s | FileCheck %s

// CHECK-LABEL: triton_conv1d_2d_kernel
func.func @triton_conv1d_2d_kernel(%arg0: tensor<32x128xf16>, %arg1: tensor<32x32x5xf16>, %arg2: tensor<32xf16>, %arg3: tensor<32x126xf16>) -> tensor<32x126xf16> {
  // CHECK: %{{.*}} = hivm.hir.Conv1dL1 {groups = 1 : i32, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x128xf16>, tensor<32x32x5xf16>, i1, tensor<32xf16>) outs(%{{.*}} : tensor<32x126xf16>) -> tensor<32x126xf16>
  %0 = hfusion.conv1d {dilation = 1 : i32, groups = 1 : i32, padding = 1 : i32, stride = 1 : i32} ins(%arg0, %arg1, %arg2 : tensor<32x128xf16>, tensor<32x32x5xf16>, tensor<32xf16>) outs(%arg3 : tensor<32x126xf16>) -> tensor<32x126xf16>
  return %0 : tensor<32x126xf16>
}

// CHECK-LABEL: triton_conv1d_3d_kernel
func.func @triton_conv1d_3d_kernel(%arg0: tensor<2x32x128xf16>, %arg1: tensor<32x32x5xf16>, %arg2: tensor<32xf16>, %arg3: tensor<2x32x126xf16>) -> tensor<2x32x126xf16> {
  // CHECK: %{{.*}} = hivm.hir.Conv1dL1 {groups = 1 : i32, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<2x32x128xf16>, tensor<32x32x5xf16>, i1, tensor<32xf16>) outs(%{{.*}} : tensor<2x32x126xf16>) -> tensor<2x32x126xf16>
  %0 = hfusion.conv1d {dilation = 1 : i32, groups = 1 : i32, padding = 1 : i32, stride = 1 : i32} ins(%arg0, %arg1, %arg2 : tensor<2x32x128xf16>, tensor<32x32x5xf16>, tensor<32xf16>) outs(%arg3 : tensor<2x32x126xf16>) -> tensor<2x32x126xf16>
  return %0 : tensor<2x32x126xf16>
}

// CHECK-LABEL: triton_conv2d_3d_kernel
func.func @triton_conv2d_3d_kernel(%arg0: tensor<32x128x128xf16>, %arg1: tensor<32x32x5x5xf16>, %arg2: tensor<32xf16>, %arg3: tensor<32x126x126xf16>) -> tensor<32x126x126xf16> {
  //CHECK: %{{.*}} = hivm.hir.Conv2dL1 {groups = 1 : i32, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x128x128xf16>, tensor<32x32x5x5xf16>, i1, tensor<32xf16>) outs(%{{.*}} : tensor<32x126x126xf16>) -> tensor<32x126x126xf16>
  %0 = hfusion.conv2d {dilation = 1 : i32, groups = 1 : i32, padding = 1 : i32, stride = 1 : i32} ins(%arg0, %arg1, %arg2 : tensor<32x128x128xf16>, tensor<32x32x5x5xf16>, tensor<32xf16>) outs(%arg3 : tensor<32x126x126xf16>) -> tensor<32x126x126xf16>
  return %0 : tensor<32x126x126xf16>
}

// CHECK-LABEL: triton_conv2d_4d_kernel
func.func @triton_conv2d_4d_kernel(%arg0: tensor<2x32x128x128xf16>, %arg1: tensor<32x32x5x5xf16>, %arg2: tensor<32xf16>, %arg3: tensor<2x32x126x126xf16>) -> tensor<2x32x126x126xf16> {
  //CHECK: %{{.*}} = hivm.hir.Conv2dL1 {groups = 1 : i32, padding = 1 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<2x32x128x128xf16>, tensor<32x32x5x5xf16>, i1, tensor<32xf16>) outs(%{{.*}} : tensor<2x32x126x126xf16>) -> tensor<2x32x126x126xf16>
  %0 = hfusion.conv2d {dilation = 1 : i32, groups = 1 : i32, padding = 1 : i32, stride = 1 : i32} ins(%arg0, %arg1, %arg2 : tensor<2x32x128x128xf16>, tensor<32x32x5x5xf16>, tensor<32xf16>) outs(%arg3 : tensor<2x32x126x126xf16>) -> tensor<2x32x126x126xf16>
  return %0 : tensor<2x32x126x126xf16>
}