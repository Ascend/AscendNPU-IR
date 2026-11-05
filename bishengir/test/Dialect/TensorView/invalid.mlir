// RUN: bishengir-opt %s -split-input-file -verify-diagnostics

func.func @stride_mismatch(%address: i64, %size: index, %stride: index) {
  // expected-error @+1 {{static strides must match the stride operands}}
  %base = tv.make_tensor_view %address, sizes = [%size], strides = [%stride] : i64 -> !tv.tensor_view<?xf32, strides=[1]>
  return
}

// -----

func.func @negative_extent(%address: i64) {
  %c1 = arith.constant 1 : index
  %cm1 = arith.constant -1 : index
  // expected-error @+1 {{logical sizes must be nonnegative}}
  %base = tv.make_tensor_view %address, sizes = [%cm1], strides = [%c1] : i64 -> !tv.tensor_view<?xf32, strides=[1]>
  return
}

// -----

func.func @permuted_dimensions(%address: i64, %size: index, %stride: index) {
  %base = tv.make_tensor_view %address, sizes = [%size, %size], strides = [%stride, %stride] : i64 -> !tv.tensor_view<?x?xf32, strides=[?,?]>
  // expected-error @+1 {{only identity dim_map is supported}}
  %view = tv.make_partition_view %base : !tv.tensor_view<?x?xf32, strides=[?,?]> -> !tv.tensor_view<?x?xf32, strides=[?,?], #tv.partition_view<tile=[2,4], dim_map=[1,0], padding_value=zero>>
  return
}

// -----

func.func @integer_padding(%address: i64, %size: index, %stride: index) {
  %base = tv.make_tensor_view %address, sizes = [%size], strides = [%stride] : i64 -> !tv.tensor_view<?xi32, strides=[?]>
  // expected-error @+1 {{nan and infinity padding require a floating-point element type}}
  %view = tv.make_partition_view %base : !tv.tensor_view<?xi32, strides=[?]> -> !tv.tensor_view<?xi32, strides=[?], #tv.partition_view<tile=[4], dim_map=[0], padding_value=nan>>
  return
}

// -----

func.func @mixed_view_indices(%view: !tv.tensor_view<?x?xf32, strides=[?,1], #tv.gather_scatter_view<tile=[2,4], sparse_dim=[0], padding_value=zero>>, %rows: tensor<2xi32>, %j: index, %dst: memref<2x4xf32>) {
  // expected-error @+1 {{sparse coordinates must be rank-one index tensors or buffers}}
  tv.view_load %view[%rows, %j] outs(%dst : memref<2x4xf32>) : !tv.tensor_view<?x?xf32, strides=[?,1], #tv.gather_scatter_view<tile=[2,4], sparse_dim=[0], padding_value=zero>>, tensor<2xi32>, index
  return
}

// -----

func.func @view_index_length(%view: !tv.tensor_view<?x?xf32, strides=[?,1], #tv.gather_scatter_view<tile=[2,4], sparse_dim=[0], padding_value=zero>>, %rows: tensor<3xi32>, %j: index) {
  %init = tensor.empty() : tensor<2x4xf32>
  // expected-error @+1 {{sparse index length must match its tile dimension}}
  %value = tv.view_load %view[%rows, %j] outs(%init : tensor<2x4xf32>) : !tv.tensor_view<?x?xf32, strides=[?,1], #tv.gather_scatter_view<tile=[2,4], sparse_dim=[0], padding_value=zero>>, tensor<3xi32>, index -> tensor<2x4xf32>
  return
}

// -----

func.func @removed_pointer_load() {
  // expected-error @+1 {{unregistered operation 'tv.ptr_load'}}
  "tv.ptr_load"() : () -> ()
  return
}

// -----

func.func @removed_pointer_store() {
  // expected-error @+1 {{unregistered operation 'tv.ptr_store'}}
  "tv.ptr_store"() : () -> ()
  return
}
