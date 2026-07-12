// RUN: bishengir-opt --hivm-simt-vf-sub-tiling=max-tile-size=4 %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @tile_local_store(
// CHECK: scf.for
// CHECK: %[[OFF:.*]] = affine.apply
// CHECK-DAG: %[[BUFFER:.*]] = memref.alloc() : memref<4xf32
// CHECK-DAG: %[[SRC:.*]] = memref.reinterpret_cast %arg0 to offset: [%[[OFF]]], sizes: [4], strides: [1]
// CHECK: hivm.hir.load ins(%[[SRC]] : memref<4xf32
// CHECK-SAME: outs(%[[BUFFER]] : memref<4xf32
// CHECK: %[[T:.*]] = bufferization.to_tensor %[[BUFFER]] restrict writable : memref<4xf32
// CHECK: %[[DST:.*]] = memref.subview %arg1[%[[OFF]]] [4] [1]
// CHECK: hivm.hir.local_store ins(%[[DST]] : memref<4xf32
// CHECK-SAME: %[[T]] : tensor<4xf32>)
module {
  func.func @tile_local_store(%arg0: memref<?xf32, #hivm.address_space<gm>>, %arg1: memref<8xf32, #hivm.address_space<ub>>) attributes {hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    %src = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1]>, #hivm.address_space<gm>>
    %buffer = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%src : memref<8xf32, strided<[1]>, #hivm.address_space<gm>>) outs(%buffer : memref<8xf32, #hivm.address_space<ub>>)
    %0 = bufferization.to_tensor %buffer restrict writable : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.local_store ins(%arg1 : memref<8xf32, #hivm.address_space<ub>>, %0 : tensor<8xf32>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @tile_equivalent_multi_user_slices(
// CHECK: scf.for
// CHECK: %[[SRC:.*]] = memref.subview %arg0[%{{.*}}] [4] [1]
// CHECK-NEXT: %[[TENSOR:.*]] = bufferization.to_tensor %[[SRC]] restrict writable
// CHECK: %[[DST0:.*]] = memref.subview %arg1[%{{.*}}] [4] [1]
// CHECK: hivm.hir.local_store ins(%[[DST0]]
// CHECK-SAME: %[[TENSOR]] : tensor<4xf32>
// CHECK: %[[DST1:.*]] = memref.subview %arg2[%{{.*}}] [4] [1]
// CHECK: hivm.hir.local_store ins(%[[DST1]]
// CHECK-SAME: %[[TENSOR]] : tensor<4xf32>
module {
  func.func @tile_equivalent_multi_user_slices(%arg0: memref<8xf32, #hivm.address_space<ub>>, %arg1: memref<8xf32, #hivm.address_space<ub>>, %arg2: memref<8xf32, #hivm.address_space<ub>>) attributes {hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    %tensor = bufferization.to_tensor %arg0 restrict writable : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.local_store ins(%arg1 : memref<8xf32, #hivm.address_space<ub>>, %tensor : tensor<8xf32>)
    hivm.hir.local_store ins(%arg2 : memref<8xf32, #hivm.address_space<ub>>, %tensor : tensor<8xf32>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @tile_gather_local_store_with_indices_chain(
// CHECK: scf.for
// CHECK: %[[IDXUB:.*]] = memref.subview %arg1[%{{.*}}, 0] [4, 8] [1, 1]
// CHECK: %[[IDX:.*]] = bufferization.to_tensor %[[IDXUB]] restrict writable : memref<4x8xi64
// CHECK: hivm.hir.vbrc
// CHECK: hivm.hir.vadd
// CHECK: %[[G:.*]] = hivm.hir.gather_load
// CHECK-SAME: outs(%{{.*}} : tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK: %[[DST:.*]] = memref.subview %arg4[%{{.*}}, 0] [4, 8] [1, 1]
// CHECK: hivm.hir.local_store ins(%[[DST]] : memref<4x8xf32
// CHECK-SAME: %[[G]] : tensor<4x8xf32>)
module {
  func.func @tile_gather_local_store_with_indices_chain(%arg0: memref<?xi64, #hivm.address_space<gm>>, %arg1: memref<8x8xi64, #hivm.address_space<ub>>, %arg2: memref<?xf32, #hivm.address_space<gm>>, %arg3: i32, %arg4: memref<8x8xf32, #hivm.address_space<ub>>, %arg5: i32) attributes {hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8, 8], strides: [8, 1] : memref<?xi64, #hivm.address_space<gm>> to memref<8x8xi64, strided<[8, 1]>, #hivm.address_space<gm>>
    hivm.hir.load ins(%0 : memref<8x8xi64, strided<[8, 1]>, #hivm.address_space<gm>>) outs(%arg1 : memref<8x8xi64, #hivm.address_space<ub>>)
    %1 = bufferization.to_tensor %arg1 restrict writable : memref<8x8xi64, #hivm.address_space<ub>>
    %2 = tensor.empty() : tensor<8xi32>
    %3 = hivm.hir.varange offset[%c0] strides[%c1] outs(%2 : tensor<8xi32>) -> tensor<8xi32>
    %4 = tensor.empty() : tensor<8xi32>
    %5 = hivm.hir.vmul ins(%3, %arg5 : tensor<8xi32>, i32) outs(%4 : tensor<8xi32>) -> tensor<8xi32>
    %6 = tensor.empty() : tensor<8xi64>
    %7 = hivm.hir.vcast ins(%5 : tensor<8xi32>) outs(%6 : tensor<8xi64>) -> tensor<8xi64>
    %expanded = tensor.expand_shape %7 [[0, 1]] output_shape [8, 1] : tensor<8xi64> into tensor<8x1xi64>
    %8 = tensor.empty() : tensor<8x8xi64>
    %9 = hivm.hir.vbrc ins(%expanded : tensor<8x1xi64>) outs(%8 : tensor<8x8xi64>) broadcast_dims = [1] -> tensor<8x8xi64>
    %10 = tensor.empty() : tensor<8x8xi64>
    %11 = hivm.hir.vadd ins(%9, %1 : tensor<8x8xi64>, tensor<8x8xi64>) outs(%10 : tensor<8x8xi64>) -> tensor<8x8xi64>
    %12 = tensor.empty() : tensor<8x8xf32>
    %13 = hivm.hir.gather_load ins(%arg2 : memref<?xf32, #hivm.address_space<gm>>, %11 : tensor<8x8xi64>, %arg3 : i32) outs(%12 : tensor<8x8xf32>) -> tensor<8x8xf32>
    hivm.hir.local_store ins(%arg4 : memref<8x8xf32, #hivm.address_space<ub>>, %13 : tensor<8x8xf32>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @skip_mismatched_local_stores(
// CHECK-NOT: scf.for
// CHECK: hivm.hir.local_store ins(%arg2 : memref<8xf32, #hivm.address_space<ub>>, %[[T0:.*]] : tensor<8xf32>)
// CHECK: hivm.hir.local_store ins(%arg5 : memref<10xf32, #hivm.address_space<ub>>, %[[T1:.*]] : tensor<10xf32>)
// CHECK: return
module {
  func.func @skip_mismatched_local_stores(%arg0: memref<?xf32, #hivm.address_space<gm>>, %arg1: memref<8xf32, #hivm.address_space<ub>>, %arg2: memref<8xf32, #hivm.address_space<ub>>, %arg3: memref<?xf32, #hivm.address_space<gm>>, %arg4: memref<10xf32, #hivm.address_space<ub>>, %arg5: memref<10xf32, #hivm.address_space<ub>>) attributes {hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    %0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1]>, #hivm.address_space<gm>>
    hivm.hir.load ins(%0 : memref<8xf32, strided<[1]>, #hivm.address_space<gm>>) outs(%arg1 : memref<8xf32, #hivm.address_space<ub>>)
    %1 = bufferization.to_tensor %arg1 restrict writable : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.local_store ins(%arg2 : memref<8xf32, #hivm.address_space<ub>>, %1 : tensor<8xf32>)
    %2 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [10], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<10xf32, strided<[1]>, #hivm.address_space<gm>>
    hivm.hir.load ins(%2 : memref<10xf32, strided<[1]>, #hivm.address_space<gm>>) outs(%arg4 : memref<10xf32, #hivm.address_space<ub>>)
    %3 = bufferization.to_tensor %arg4 restrict writable : memref<10xf32, #hivm.address_space<ub>>
    hivm.hir.local_store ins(%arg5 : memref<10xf32, #hivm.address_space<ub>>, %3 : tensor<10xf32>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @skip_vtranspose_local_store(
// CHECK-NOT: scf.for
// CHECK: hivm.hir.vtranspose
// CHECK: hivm.hir.local_store ins(%arg2 : memref<4x8xf32, #hivm.address_space<ub>>, %[[T:.*]] : tensor<4x8xf32>)
module {
  func.func @skip_vtranspose_local_store(%arg0: memref<?xf32, #hivm.address_space<gm>>, %arg1: memref<8x4xf32, #hivm.address_space<ub>>, %arg2: memref<4x8xf32, #hivm.address_space<ub>>) attributes {hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    %0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8, 4], strides: [4, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<8x4xf32, strided<[4, 1]>, #hivm.address_space<gm>>
    hivm.hir.load ins(%0 : memref<8x4xf32, strided<[4, 1]>, #hivm.address_space<gm>>) outs(%arg1 : memref<8x4xf32, #hivm.address_space<ub>>)
    %1 = bufferization.to_tensor %arg1 restrict writable : memref<8x4xf32, #hivm.address_space<ub>>
    %2 = tensor.empty() : tensor<4x8xf32>
    %3 = hivm.hir.vtranspose ins(%1 : tensor<8x4xf32>) outs(%2 : tensor<4x8xf32>) permutation = [1, 0] -> tensor<4x8xf32>
    hivm.hir.local_store ins(%arg2 : memref<4x8xf32, #hivm.address_space<ub>>, %3 : tensor<4x8xf32>)
    return
  }
}

// -----

// CHECK-LABEL: func.func @tile_single_user_local_load(
// CHECK: scf.for
// CHECK: %[[SRC:.*]] = memref.subview %arg0[%{{.*}}] [4] [1]
// CHECK: %[[LOAD:.*]] = hivm.hir.local_load ins(%[[SRC]] : memref<4xf32
// CHECK: %[[DST:.*]] = memref.subview %arg1[%{{.*}}] [4] [1]
// CHECK: hivm.hir.local_store ins(%[[DST]] : memref<4xf32
// CHECK-SAME: %[[LOAD]] : tensor<4xf32>
module {
  func.func @tile_single_user_local_load(%arg0: memref<8xf32, #hivm.address_space<ub>>, %arg1: memref<8xf32, #hivm.address_space<ub>>) attributes {hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    %tensor = hivm.hir.local_load ins(%arg0 : memref<8xf32, #hivm.address_space<ub>>) -> tensor<8xf32>
    hivm.hir.local_store ins(%arg1 : memref<8xf32, #hivm.address_space<ub>>, %tensor : tensor<8xf32>)
    return
  }
}

// -----

// Ensure nested extract_slice bubbling uses the immediately enclosing loop
// when reconstructing the child slice. The outer SIMT VF tile loop determines
// the four-element tile, while the original inner loop still slices it into
// one-element elementwise tiles.
//
// CHECK-LABEL: func.func @tile_nested_extract_slice(
// CHECK: scf.for %[[TILE:.*]] =
// CHECK: %[[TILE_OFF:.*]] = affine.apply {{.*}}(){{\[}}%[[TILE]]]
// CHECK: %[[SRC_VIEW:.*]] = memref.subview %arg0[0, %[[TILE_OFF]]] [1, 4] [1, 1]
// CHECK: %[[SRC_TENSOR:.*]] = bufferization.to_tensor %[[SRC_VIEW]]
// CHECK: %[[RESULT:.*]] = scf.for %[[INNER:.*]] = {{.*}} iter_args(%[[ACC:.*]] = {{.*}}) -> (tensor<4xf32>) {
// CHECK: %[[PARENT_SLICE:.*]] = tensor.extract_slice %[[SRC_TENSOR]][0, 0] [1, 4] [1, 1]
// CHECK: %[[VALUE_TILE:.*]] = tensor.extract_slice %[[PARENT_SLICE]][%[[INNER]]] [1] [1]
// CHECK: %[[VALUE_DST:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[VALUE:.*]] = hivm.hir.vln ins(%[[VALUE_TILE]] : tensor<1xf32>) outs(%[[VALUE_DST]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK: tensor.insert_slice {{.*}} into %[[ACC]][%[[INNER]]] [1] [1]
// CHECK: hivm.hir.local_store
// CHECK-SAME: %[[RESULT]] : tensor<4xf32>
// CHECK: } {hivm.simt_vf_tile_loop}
module {
  func.func @tile_nested_extract_slice(
      %src: memref<1x8xf32, #hivm.address_space<ub>>,
      %dst: memref<8xf32, #hivm.address_space<ub>>)
      attributes {
        hivm.func_core_type = #hivm.func_core_type<AIV>,
        hivm.vf_mode = #hivm.vf_mode<SIMT>,
        no_inline,
        outline
      } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %acc_init = tensor.empty() : tensor<8xf32>
    %op_init = tensor.empty() : tensor<2xf32>
    %src_tensor = bufferization.to_tensor %src restrict writable
        : memref<1x8xf32, #hivm.address_space<ub>>
    %batch_tile = tensor.extract_slice %src_tensor[0, 0] [1, 8] [1, 1]
        : tensor<1x8xf32> to tensor<8xf32>
    %result = scf.for %i = %c0 to %c8 step %c2
        iter_args(%acc = %acc_init) -> tensor<8xf32> {
      %src_tile = tensor.extract_slice %batch_tile[%i] [2] [1]
          : tensor<8xf32> to tensor<2xf32>
      %value = hivm.hir.vln
          ins(%src_tile : tensor<2xf32>)
          outs(%op_init : tensor<2xf32>) -> tensor<2xf32>
      %next = tensor.insert_slice %value into %acc[%i] [2] [1]
          : tensor<2xf32> into tensor<8xf32>
      scf.yield %next : tensor<8xf32>
    }
    hivm.hir.local_store
        ins(%dst : memref<8xf32, #hivm.address_space<ub>>,
            %result : tensor<8xf32>)
    return
  }
}

// -----

// Ensure the outer SIMT VF loop determines the two-element split size while
// the original inner loop induction variable determines the insertion offset.
//
// CHECK-LABEL: func.func @tile_nested_insert_slice_offset(
// CHECK: scf.for %[[TILE:.*]] =
// CHECK: %[[TILE_OFF:.*]] = affine.apply {{.*}}(){{\[}}%[[TILE]]]
// CHECK: %[[SRC_VIEW:.*]] = memref.subview %arg0[%[[TILE_OFF]]] [4] [1]
// CHECK: %[[SRC_TENSOR:.*]] = bufferization.to_tensor %[[SRC_VIEW]]
// CHECK: %[[RESULT:.*]] = scf.for %[[INNER:.*]] = {{.*}} iter_args(%[[ACC:.*]] = {{.*}}) -> (tensor<4xf32>) {
// CHECK: %[[INNER_OFF:.*]] = affine.apply {{.*}}(){{\[}}%[[INNER]]]
// CHECK: %[[SRC_TILE:.*]] = tensor.extract_slice %[[SRC_TENSOR]][%[[INNER_OFF]]] [2] [1]
// CHECK: %[[VALUE:.*]] = hivm.hir.vln ins(%[[SRC_TILE]] : tensor<2xf32>)
// CHECK: tensor.insert_slice %[[VALUE]] into %[[ACC]][%[[INNER_OFF]]] [2] [1]
// CHECK: hivm.hir.local_store
// CHECK-SAME: %[[RESULT]] : tensor<4xf32>
// CHECK: } {hivm.simt_vf_tile_loop}
#map = affine_map<()[s0] -> (s0 * 8)>
module {
  func.func @tile_nested_insert_slice_offset(
      %src: memref<16xf32, #hivm.address_space<ub>>,
      %dst: memref<16xf32, #hivm.address_space<ub>>)
      attributes {
        hivm.func_core_type = #hivm.func_core_type<AIV>,
        hivm.vf_mode = #hivm.vf_mode<SIMT>,
        no_inline,
        outline
      } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %init = tensor.empty() : tensor<16xf32>
    %op_init = tensor.empty() : tensor<8xf32>
    %src_tensor = bufferization.to_tensor %src restrict writable
        : memref<16xf32, #hivm.address_space<ub>>
    %result = scf.for %i = %c0 to %c2 step %c1
        iter_args(%acc = %init) -> tensor<16xf32> {
      %offset = affine.apply #map()[%i]
      %tile = tensor.extract_slice %src_tensor[%offset] [8] [1]
          : tensor<16xf32> to tensor<8xf32>
      %value = hivm.hir.vln
          ins(%tile : tensor<8xf32>)
          outs(%op_init : tensor<8xf32>) -> tensor<8xf32>
      %next = tensor.insert_slice %value into %acc[%offset] [8] [1]
          : tensor<8xf32> into tensor<16xf32>
      scf.yield %next : tensor<16xf32>
    }
    hivm.hir.local_store
        ins(%dst : memref<16xf32, #hivm.address_space<ub>>,
            %result : tensor<16xf32>)
    return
  }
}
