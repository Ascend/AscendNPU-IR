// RUN: bishengir-opt %s | FileCheck %s
// RUN: bishengir-opt %s -hfusion-flatten-ops="flatten-mode=tidy register-based=true" | FileCheck %s --check-prefix=FLATTEN
// RUN: bishengir-opt %s -hfusion-flatten-ops="flatten-mode=tidy" | FileCheck %s --check-prefix=FLATTEN

func.func @view(%address: i64, %size: index, %stride: index, %i: index) -> tensor<4xf32> {
  %base = tv.make_tensor_view %address, sizes = [%size], strides = [%stride] : i64 -> !tv.tensor_view<?xf32, strides=[?]>
  %view = tv.make_partition_view %base : !tv.tensor_view<?xf32, strides=[?]> -> !tv.tensor_view<?xf32, strides=[?], #tv.partition_view<tile=[4], dim_map=[0], padding_value=zero>>
  %result_init = bufferization.alloc_tensor() {memory_space = #hivm.address_space<ub>} : tensor<4xf32>
  %result = tv.view_load %view[%i] outs(%result_init : tensor<4xf32>) : !tv.tensor_view<?xf32, strides=[?], #tv.partition_view<tile=[4], dim_map=[0], padding_value=zero>>, index -> tensor<4xf32>
  return %result : tensor<4xf32>
}
// CHECK: tv.make_tensor_view
// CHECK: tv.make_partition_view
// CHECK: tv.view_load

// FLATTEN-LABEL: func.func @partition_axes(
// FLATTEN: %[[INIT:.*]] = tensor.empty() : tensor<1x4xf32>
// FLATTEN: tv.view_load {{.*}} outs(%[[INIT]] : tensor<1x4xf32>)
// FLATTEN-SAME: -> tensor<1x4xf32>
func.func @partition_axes(%view: !tv.tensor_view<?x?xf32, strides=[?,1], #tv.partition_view<tile=[1,4], dim_map=[0,1], padding_value=zero>>, %i: index, %j: index) -> tensor<1x4xf32> {
  %init = tensor.empty() : tensor<1x4xf32>
  %value = tv.view_load %view[%i, %j] outs(%init : tensor<1x4xf32>) : !tv.tensor_view<?x?xf32, strides=[?,1], #tv.partition_view<tile=[1,4], dim_map=[0,1], padding_value=zero>>, index, index -> tensor<1x4xf32>
  return %value : tensor<1x4xf32>
}
