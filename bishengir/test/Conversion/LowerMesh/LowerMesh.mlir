// RUN: bishengir-opt -lower-mesh -canonicalize -one-shot-bufferize \
// RUN:               -func-bufferize -buffer-results-to-out-params \
// RUN:               -canonicalize -lower-mesh-host -canonicalize \
// RUN:               -convert-hivm-to-llvm -canonicalize %s | FileCheck %s

// RUN: bishengir-opt -lower-mesh="target=lccl" -canonicalize -one-shot-bufferize \
// RUN:               -func-bufferize -buffer-results-to-out-params \
// RUN:               -canonicalize -lower-mesh-host -canonicalize \
// RUN:               -convert-hivm-to-llvm -canonicalize %s | FileCheck -check-prefix=LCCL %s

// REQUIRES: issue2438
// CHECK-NOT: mesh.mesh
mesh.mesh @mesh(shape=2)
func.func @testCollectives(%arg : tensor<8x16xf16>) -> tensor<4x32xf16> attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  // CHECK-NOT: unrealized_conversion_cast
  // CHECK: _mlir_device_malloc
  // CHECK: _mlir_device_memcpy
  // CHECK: _mlir_hccl_allgather
  // LCCL: _mlir_lccl_allgather
  %0 = mesh.all_gather %arg on @mesh mesh_axes = [0] gather_axis = 0 : tensor<8x16xf16> -> tensor<16x16xf16>
  // CHECK: _mlir_hccl_allreduce
  // LCCL: _mlir_lccl_allreduce
  %1 = mesh.all_reduce %0 on @mesh mesh_axes = [0] reduction = <max> : tensor<16x16xf16> -> tensor<16x16xf16>
  // CHECK: _mlir_hccl_all2all
  // LCCL: _mlir_lccl_all2all
  %2 = mesh.all_to_all %1 on @mesh mesh_axes = [0] split_axis = 0 concat_axis = 1 : tensor<16x16xf16> -> tensor<8x32xf16>
  // CHECK: _mlir_hccl_reducescatter
  // LCCL: _mlir_lccl_reducescatter
  %3 = mesh.reduce_scatter %2 on @mesh mesh_axes = [0] reduction = <max>  scatter_axis = 0 : tensor<8x32xf16> -> tensor<4x32xf16>
  return %3 : tensor<4x32xf16>
}
