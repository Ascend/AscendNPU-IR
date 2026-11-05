// RUN: bishengir-opt %s -hivm-infer-func-core-type | FileCheck %s --check-prefix=CORE
// RUN: bishengir-opt %s -tensor-copy-insertion="bufferize-function-boundaries" -one-shot-bufferize="bufferize-function-boundaries" | FileCheck %s --check-prefix=BUFFER
// RUN: bishengir-opt %s -tensor-copy-insertion="bufferize-function-boundaries" -one-shot-bufferize="bufferize-function-boundaries" -convert-tensor-view-to-hivm | FileCheck %s --check-prefix=HIVM --implicit-check-not=tv. --implicit-check-not=hivm.hir.vbrc

#partition = #tv.partition_view<tile=[1,4], dim_map=[0,1], padding_value=zero>
#strided = #tv.strided_view<tile=[4], dim_map=[0], traversal_strides=[2], padding_value="-inf">
#gather = #tv.gather_scatter_view<tile=[2,4], sparse_dim=[0], padding_value=nan>
#cartesian = #tv.gather_scatter_view<tile=[2,3], sparse_dim=[0,1], padding_value=inf>

// CORE-LABEL: func.func @partition(
// CORE-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// BUFFER-LABEL: func.func @partition
// BUFFER: %[[DATA:.*]] = memref.alloc() {{.*}} : memref<1x4xf32, #hivm.address_space<ub>>
// BUFFER: tv.view_load {{.*}} outs(%[[DATA]] : memref<1x4xf32, #hivm.address_space<ub>>)
// BUFFER: tv.view_store {{.*}}, %[[DATA]]
// HIVM-LABEL: func.func @partition
// HIVM: %[[DATA:.*]] = memref.alloc() {{.*}} : memref<1x4xf32, #hivm.address_space<ub>>
// HIVM: %[[STAGING:.*]] = memref.alloc() {alignment = 32 : i64} : memref<1x8xf32, #hivm.address_space<ub>>
// HIVM: memref.store {{.*}}, %[[DATA]]
// HIVM: scf.if
// HIVM: memref.reinterpret_cast
// HIVM-SAME: memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
// HIVM: hivm.hir.load
// HIVM: memref.load %[[STAGING]]
// HIVM: memref.store {{.*}}, %[[DATA]]
// HIVM: hivm.hir.store
func.func @partition(%src: i64, %dst: i64, %rows: index, %cols: index, %stride: index, %i: index, %j: index) {
  %c1 = arith.constant 1 : index
  %a = tv.make_tensor_view %src, sizes = [%rows, %cols], strides = [%stride, %c1] : i64 -> !tv.tensor_view<?x?xf32, strides=[?,1]>
  %b = tv.make_tensor_view %dst, sizes = [%rows, %cols], strides = [%stride, %c1] : i64 -> !tv.tensor_view<?x?xf32, strides=[?,1]>
  %av = tv.make_partition_view %a : !tv.tensor_view<?x?xf32, strides=[?,1]> -> !tv.tensor_view<?x?xf32, strides=[?,1], #partition>
  %bv = tv.make_partition_view %b : !tv.tensor_view<?x?xf32, strides=[?,1]> -> !tv.tensor_view<?x?xf32, strides=[?,1], #partition>
  %value_init = bufferization.alloc_tensor() {memory_space = #hivm.address_space<ub>} : tensor<1x4xf32>
  %value = tv.view_load %av[%i, %j] outs(%value_init : tensor<1x4xf32>) : !tv.tensor_view<?x?xf32, strides=[?,1], #partition>, index, index -> tensor<1x4xf32>
  tv.view_store %bv[%i, %j], %value : !tv.tensor_view<?x?xf32, strides=[?,1], #partition>, tensor<1x4xf32>, index, index
  return
}

// CORE-LABEL: func.func @strided(
// CORE-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// BUFFER-LABEL: func.func @strided
// BUFFER: tv.view_load
// BUFFER: tv.view_store
// HIVM-LABEL: func.func @strided
// HIVM: arith.constant 0xFF800000 : f32
// HIVM: arith.muli {{.*}} : index
// HIVM: hivm.hir.load
// HIVM: hivm.hir.store
func.func @strided(%src: i64, %dst: i64, %size: index, %stride: index, %i: index) {
  %a = tv.make_tensor_view %src, sizes = [%size], strides = [%stride] : i64 -> !tv.tensor_view<?xf32, strides=[?]>
  %b = tv.make_tensor_view %dst, sizes = [%size], strides = [%stride] : i64 -> !tv.tensor_view<?xf32, strides=[?]>
  %av = tv.make_strided_view %a : !tv.tensor_view<?xf32, strides=[?]> -> !tv.tensor_view<?xf32, strides=[?], #strided>
  %bv = tv.make_strided_view %b : !tv.tensor_view<?xf32, strides=[?]> -> !tv.tensor_view<?xf32, strides=[?], #strided>
  %value_init = bufferization.alloc_tensor() {memory_space = #hivm.address_space<ub>} : tensor<4xf32>
  %value = tv.view_load %av[%i] outs(%value_init : tensor<4xf32>) : !tv.tensor_view<?xf32, strides=[?], #strided>, index -> tensor<4xf32>
  tv.view_store %bv[%i], %value : !tv.tensor_view<?xf32, strides=[?], #strided>, tensor<4xf32>, index
  return
}

// CORE-LABEL: func.func @gather(
// CORE-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// BUFFER-LABEL: func.func @gather
// BUFFER: tv.view_load
// BUFFER: tv.view_store
// HIVM-LABEL: func.func @gather
// HIVM: scf.for
// HIVM: memref.load
// HIVM: arith.index_cast
// HIVM: scf.if
// HIVM: hivm.hir.load
// HIVM: scf.for
// HIVM: hivm.hir.store
func.func @gather(%src: i64, %dst: i64, %rows: index, %cols: index, %stride: index, %indices: tensor<2xi32>, %j: index) {
  %c1 = arith.constant 1 : index
  %a = tv.make_tensor_view %src, sizes = [%rows, %cols], strides = [%stride, %c1] : i64 -> !tv.tensor_view<?x?xf32, strides=[?,1]>
  %b = tv.make_tensor_view %dst, sizes = [%rows, %cols], strides = [%stride, %c1] : i64 -> !tv.tensor_view<?x?xf32, strides=[?,1]>
  %av = tv.make_gather_scatter_view %a : !tv.tensor_view<?x?xf32, strides=[?,1]> -> !tv.tensor_view<?x?xf32, strides=[?,1], #gather>
  %bv = tv.make_gather_scatter_view %b : !tv.tensor_view<?x?xf32, strides=[?,1]> -> !tv.tensor_view<?x?xf32, strides=[?,1], #gather>
  %value_init = bufferization.alloc_tensor() {memory_space = #hivm.address_space<ub>} : tensor<2x4xf32>
  %value = tv.view_load %av[%indices, %j] outs(%value_init : tensor<2x4xf32>) : !tv.tensor_view<?x?xf32, strides=[?,1], #gather>, tensor<2xi32>, index -> tensor<2x4xf32>
  tv.view_store %bv[%indices, %j], %value : !tv.tensor_view<?x?xf32, strides=[?,1], #gather>, tensor<2x4xf32>, tensor<2xi32>, index
  return
}

// CORE-LABEL: func.func @cartesian(
// CORE-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// BUFFER-LABEL: func.func @cartesian
// BUFFER: tv.view_load
// BUFFER: tv.view_store
// HIVM-LABEL: func.func @cartesian
// HIVM: scf.for
// HIVM: scf.for
// HIVM: memref.load
// HIVM: memref.load
// HIVM: hivm.hir.load
// HIVM: scf.for
// HIVM: scf.for
// HIVM: hivm.hir.store
func.func @cartesian(%src: i64, %dst: i64, %rows: index, %cols: index, %stride: index, %ri: tensor<2xi64>, %ci: tensor<3xi32>) {
  %c1 = arith.constant 1 : index
  %a = tv.make_tensor_view %src, sizes = [%rows, %cols], strides = [%stride, %c1] : i64 -> !tv.tensor_view<?x?xf32, strides=[?,1]>
  %b = tv.make_tensor_view %dst, sizes = [%rows, %cols], strides = [%stride, %c1] : i64 -> !tv.tensor_view<?x?xf32, strides=[?,1]>
  %av = tv.make_gather_scatter_view %a : !tv.tensor_view<?x?xf32, strides=[?,1]> -> !tv.tensor_view<?x?xf32, strides=[?,1], #cartesian>
  %bv = tv.make_gather_scatter_view %b : !tv.tensor_view<?x?xf32, strides=[?,1]> -> !tv.tensor_view<?x?xf32, strides=[?,1], #cartesian>
  %value_init = bufferization.alloc_tensor() {memory_space = #hivm.address_space<ub>} : tensor<2x3xf32>
  %value = tv.view_load %av[%ri, %ci] outs(%value_init : tensor<2x3xf32>) : !tv.tensor_view<?x?xf32, strides=[?,1], #cartesian>, tensor<2xi64>, tensor<3xi32> -> tensor<2x3xf32>
  tv.view_store %bv[%ri, %ci], %value : !tv.tensor_view<?x?xf32, strides=[?,1], #cartesian>, tensor<2x3xf32>, tensor<2xi64>, tensor<3xi32>
  return
}

// CORE-LABEL: func.func @buffer_views(
// CORE-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// BUFFER-LABEL: func.func @buffer_views
// BUFFER: tv.view_load
// BUFFER: tv.view_store
// HIVM-LABEL: func.func @buffer_views
// HIVM: %[[OFFSET:.*]] = arith.constant 7 : index
// HIVM: %[[EXTENT:.*]] = arith.constant 4 : index
// HIVM: memref.reinterpret_cast {{.*}} to offset: [%[[OFFSET]]], sizes: [%[[EXTENT]]], strides: [1]
// HIVM: hivm.hir.load
// HIVM: hivm.hir.store
func.func @buffer_views(%gm: memref<16xf32, #hivm.address_space<gm>>, %ub: memref<16xf32, #hivm.address_space<ub>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %source = memref.subview %gm[7] [4] [1] : memref<16xf32, #hivm.address_space<gm>> to memref<4xf32, strided<[1], offset: 7>, #hivm.address_space<gm>>
  %local = memref.subview %ub[3] [4] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<4xf32, strided<[1], offset: 3>, #hivm.address_space<ub>>
  %base = tv.make_tensor_view %source, sizes = [%c4], strides = [%c1] : memref<4xf32, strided<[1], offset: 7>, #hivm.address_space<gm>> -> !tv.tensor_view<4xf32, strides=[1]>
  %view = tv.make_partition_view %base : !tv.tensor_view<4xf32, strides=[1]> -> !tv.tensor_view<4xf32, strides=[1], #tv.partition_view<tile=[4], dim_map=[0], padding_value=zero>>
  tv.view_load %view[%c0] outs(%local : memref<4xf32, strided<[1], offset: 3>, #hivm.address_space<ub>>) : !tv.tensor_view<4xf32, strides=[1], #tv.partition_view<tile=[4], dim_map=[0], padding_value=zero>>, index
  tv.view_store %view[%c0], %local : !tv.tensor_view<4xf32, strides=[1], #tv.partition_view<tile=[4], dim_map=[0], padding_value=zero>>, memref<4xf32, strided<[1], offset: 3>, #hivm.address_space<ub>>, index
  return
}

// CORE-LABEL: func.func @load_destination(
// CORE-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// BUFFER-LABEL: func.func @load_destination(
// BUFFER-SAME: %[[INIT:[a-zA-Z0-9_]+]]: memref<4xf32,
// BUFFER: %[[NEW:.*]] = memref.alloc
// BUFFER-NOT: memref.copy
// BUFFER: tv.view_load {{.*}} outs(%[[NEW]]
// BUFFER-NOT: memref.copy
// BUFFER: return %[[NEW]], %[[INIT]]
// HIVM-LABEL: func.func @load_destination
// HIVM: hivm.hir.load
func.func @load_destination(%address: i64, %init: tensor<4xf32>, %i: index) -> (tensor<4xf32>, tensor<4xf32>) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %base = tv.make_tensor_view %address, sizes = [%c4], strides = [%c1] : i64 -> !tv.tensor_view<4xf32, strides=[1]>
  %view = tv.make_partition_view %base : !tv.tensor_view<4xf32, strides=[1]> -> !tv.tensor_view<4xf32, strides=[1], #tv.partition_view<tile=[4], dim_map=[0], padding_value=zero>>
  %value = tv.view_load %view[%i] outs(%init : tensor<4xf32>) : !tv.tensor_view<4xf32, strides=[1], #tv.partition_view<tile=[4], dim_map=[0], padding_value=zero>>, index -> tensor<4xf32>
  return %value, %init : tensor<4xf32>, tensor<4xf32>
}
