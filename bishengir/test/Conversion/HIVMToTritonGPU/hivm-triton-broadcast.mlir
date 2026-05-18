// RUN: bishengir-opt -convert-hivm-to-tritongpu %s -split-input-file -verify-diagnostics | FileCheck %s
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.simt_module, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  // CHECK-LABEL: tt.func @simple_indirect_load_2d_kernel_scope_0
  // CHECK: tt.load %{{.*}} evictionPolicy = evict_first : tensor<4x8x!tt.ptr<i64>>
  // CHECK: [[ARANGE:%.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  // CHECK: [[STRIDE:%.*]] = arith.constant 1 : i32
  // CHECK: [[STRIDE_TENSOR:%.*]] = tt.splat [[STRIDE]] : i32 -> tensor<4xi32>
  // CHECK: [[MUL:%.*]] = arith.muli [[ARANGE]], [[STRIDE_TENSOR]] : tensor<4xi32>
  // CHECK: [[OFFSET:%.*]] = arith.constant 0 : i32
  // CHECK: [[OFFSET_TENSOR:%.*]] = tt.splat [[OFFSET]] : i32 -> tensor<4xi32>
  // CHECK: [[VARANGE:%.*]] = arith.addi [[MUL]], [[OFFSET_TENSOR]] : tensor<4xi32>
  // CHECK: [[SCALE:%.*]] = tt.splat %{{.*}} : i32 -> tensor<4xi32>
  // CHECK: [[SCALED:%.*]] = arith.muli [[VARANGE]], [[SCALE]] : tensor<4xi32>
  // CHECK: [[CAST:%.*]] = arith.extsi [[SCALED]] : tensor<4xi32> to tensor<4xi64>
  // CHECK: %{{.*}} = tt.reshape [[CAST]] : tensor<4xi64> -> tensor<4x1xi64>
  // CHECK: [[BRC:%.*]] = tt.broadcast %{{.*}} : tensor<4x1xi64> -> tensor<4x8xi64>
  // CHECK: arith.addi [[BRC]], %{{.*}} : tensor<4x8xi64>
  // CHECK: tt.store %{{.*}}, %{{.*}} : tensor<4x8x!tt.ptr<f32, 6>>
  func.func @simple_indirect_load_2d_kernel_scope_0(%arg0: memref<?xi64, #hivm.address_space<gm>> {hivm.memory_effect = #hivm.memory_effect<read>}, %arg1: memref<4x8xi64, #hivm.address_space<ub>> , %arg4: i32, %arg5: memref<?xf32, #hivm.address_space<gm>> {hivm.memory_effect = #hivm.memory_effect<read>}, %arg6: i32, %arg7: memref<4x8xf32, #hivm.address_space<ub>> {hivm.memory_effect = #hivm.memory_effect<write>}, %arg8: memref<1024xi8, #hivm.address_space<ub>> {hivm.shared_memory}) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, no_inline, outline} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = tensor.empty() : tensor<4x8xf32>
    %1 = tensor.empty() : tensor<4x8xi64>
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4, 8], strides: [8, 1] : memref<?xi64, #hivm.address_space<gm>> to memref<4x8xi64, strided<[8, 1]>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast : memref<4x8xi64, strided<[8, 1]>, #hivm.address_space<gm>>) outs(%arg1 : memref<4x8xi64, #hivm.address_space<ub>>) eviction_policy = <EvictFirst>
    %2 = bufferization.to_tensor %arg1 restrict writable : memref<4x8xi64, #hivm.address_space<ub>>
    %3 = tensor.empty() : tensor<4x8xi64>
    %4 = tensor.empty() : tensor<4xi64>
    %5 = tensor.empty() : tensor<4xi32>
    %6 = tensor.empty() : tensor<4xi32>
    %7 = hivm.hir.varange offset[%c0] strides[%c1] outs(%6 : tensor<4xi32>) -> tensor<4xi32>
    %8 = hivm.hir.vmul ins(%7, %arg4 : tensor<4xi32>, i32) outs(%5 : tensor<4xi32>) -> tensor<4xi32>
    %9 = hivm.hir.vcast ins(%8 : tensor<4xi32>) outs(%4 : tensor<4xi64>) -> tensor<4xi64>
    %expanded = tensor.expand_shape %9 [[0, 1]] output_shape [4, 1] : tensor<4xi64> into tensor<4x1xi64>
    %10 = hivm.hir.vbrc ins(%expanded : tensor<4x1xi64>) outs(%3 : tensor<4x8xi64>) broadcast_dims = [1] -> tensor<4x8xi64>
    %11 = hivm.hir.vadd ins(%10, %2 : tensor<4x8xi64>, tensor<4x8xi64>) outs(%1 : tensor<4x8xi64>) -> tensor<4x8xi64>
    %12 = hivm.hir.gather_load ins(%arg5 : memref<?xf32, #hivm.address_space<gm>>, %11 : tensor<4x8xi64>, %arg6 : i32) outs(%0 : tensor<4x8xf32>) -> tensor<4x8xf32>
    hivm.hir.local_store ins(%arg7 : memref<4x8xf32, #hivm.address_space<ub>>, %12 : tensor<4x8xf32>)
    return
  }
}
