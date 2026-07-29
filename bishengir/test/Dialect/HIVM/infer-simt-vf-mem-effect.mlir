// RUN: bishengir-opt --infer-simt-vf-memory-effect --split-input-file %s | FileCheck %s

// -----

// CHECK: func.func @simple_indirect_load_kernel_scope_0(%arg0: memref<8xi64> {hivm.memory_effect = #hivm.memory_effect<read>}, %arg1: memref<?xf32> {hivm.memory_effect = #hivm.memory_effect<read>}, %arg2: i32, %arg3: memref<8xf32> {hivm.memory_effect = #hivm.memory_effect<write>})
module {
  func.func @simple_indirect_load_kernel_scope_0(%arg0: memref<8xi64>, %arg1: memref<?xf32>, %arg2: i32, %arg3: memref<8xf32>) attributes {no_inline, outline, hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    %0 = tensor.empty() : tensor<8xi64>
    %1 = bufferization.to_memref %0 : memref<8xi64>
    hivm.hir.load ins(%arg0 : memref<8xi64>) outs(%1 : memref<8xi64>)
    %2 = tensor.empty() : tensor<8xf32>
    %3 = hivm.hir.gather_load ins(%arg1 : memref<?xf32>, %0 : tensor<8xi64>, %arg2 : i32) outs(%2 : tensor<8xf32>) {cache = #hivm.cache_modifier<none>, evict = #hivm.eviction_policy<EvictLast>, isVolatile = false} -> tensor<8xf32>
    hivm.hir.store ins(%3 : tensor<8xf32>) outs(%arg3 : memref<8xf32>)
    return
  }
}

// -----

// CHECK: func.func @simple_indirect_store_kernel_scope_0(%arg0: memref<8xf32> {hivm.memory_effect = #hivm.memory_effect<read>}, %arg1: memref<8xi64> {hivm.memory_effect = #hivm.memory_effect<read>}, %arg2: memref<?xf32> {hivm.memory_effect = #hivm.memory_effect<write>}, %arg3: i32)
module {
  func.func @simple_indirect_store_kernel_scope_0(%arg0: memref<8xf32>, %arg1: memref<8xi64>, %arg2: memref<?xf32>, %arg3: i32) attributes {no_inline, outline, hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    %0 = tensor.empty() : tensor<8xf32>
    %1 = bufferization.to_memref %0 : memref<8xf32>
    hivm.hir.load ins(%arg0 : memref<8xf32>) outs(%1 : memref<8xf32>)
    %2 = tensor.empty() : tensor<8xi64>
    %3 = bufferization.to_memref %2 : memref<8xi64>
    hivm.hir.load ins(%arg1 : memref<8xi64>) outs(%3 : memref<8xi64>)
    hivm.hir.scatter_store ins(%2 : tensor<8xi64>, %0 : tensor<8xf32>, %arg3 : i32) outs(%arg2 : memref<?xf32>) {cache = #hivm.cache_modifier<none>, evict = #hivm.eviction_policy<EvictLast>}
    return
  }
}

// -----

// CHECK: func.func @simple_indirect_load_from_to_tensor_scope_0(%arg0: memref<8xi64> {hivm.memory_effect = #hivm.memory_effect<read>}, %arg1: memref<?xf32> {hivm.memory_effect = #hivm.memory_effect<read>}, %arg2: memref<8xf32> {hivm.memory_effect = #hivm.memory_effect<write>})
module {
  func.func @simple_indirect_load_from_to_tensor_scope_0(%arg0: memref<8xi64>, %arg1: memref<?xf32>, %arg2: memref<8xf32>) attributes {no_inline, outline, hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    %c1_i32 = arith.constant 1 : i32
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<8xi64>
    %1 = tensor.empty() : tensor<8xf32>
    %2 = hivm.hir.gather_load ins(%arg1 : memref<?xf32>, %0 : tensor<8xi64>, %c1_i32 : i32) outs(%1 : tensor<8xf32>) -> tensor<8xf32>
    hivm.hir.local_store ins(%arg2 : memref<8xf32>, %2 : tensor<8xf32>)
    return
  }
}

// -----

// CHECK-LABEL: func.func private @kernel_scope_0(
//  CHECK-SAME:     %arg0: memref<8xf32> {hivm.memory_effect = #hivm.memory_effect<read>}
//  CHECK-SAME:     %arg1: memref<8xf32> {hivm.memory_effect = #hivm.memory_effect<write>}
// CHECK-LABEL: func.func @caller(
//       CHECK:   bufferization.to_memref %{{.+}} read_only : memref<8xf32>
//  CHECK-NEXT:   bufferization.to_memref %{{[a-z0-9_]+}} : memref<8xf32>
module {
  func.func private @kernel_scope_0(%arg0: memref<8xf32>, %arg1: memref<8xf32>) attributes {no_inline, outline, hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    %0 = bufferization.to_tensor %arg0 restrict : memref<8xf32>
    hivm.hir.local_store ins(%arg1 : memref<8xf32>, %0 : tensor<8xf32>)
    return
  }
  func.func @caller(%t0: tensor<8xf32>, %t1: tensor<8xf32>) {
    %m0 = bufferization.to_memref %t0 : memref<8xf32>
    %m1 = bufferization.to_memref %t1 : memref<8xf32>
    func.call @kernel_scope_0(%m0, %m1) : (memref<8xf32>, memref<8xf32>) -> ()
    return
  }
}

// -----

// CHECK-LABEL: func.func private @kernel_scope_w(
//  CHECK-SAME:     %arg0: memref<8xf32> {hivm.memory_effect = #hivm.memory_effect<read>}
//  CHECK-SAME:     %arg1: memref<8xf32> {hivm.memory_effect = #hivm.memory_effect<write>}
// CHECK-LABEL: func.func @caller_out(
//       CHECK:   bufferization.to_memref %{{.+}} read_only : memref<8xf32>
//       CHECK:   bufferization.to_tensor %{{.+}} restrict writable : memref<8xf32>
module {
  func.func private @kernel_scope_w(%arg0: memref<8xf32>, %arg1: memref<8xf32>) attributes {no_inline, outline, hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    %0 = bufferization.to_tensor %arg0 restrict : memref<8xf32>
    hivm.hir.local_store ins(%arg1 : memref<8xf32>, %0 : tensor<8xf32>)
    return
  }
  func.func @caller_out(%t0: tensor<8xf32>) -> tensor<8xf32> {
    %m0 = bufferization.to_memref %t0 : memref<8xf32>
    %out = memref.alloc() : memref<8xf32>
    func.call @kernel_scope_w(%m0, %out) : (memref<8xf32>, memref<8xf32>) -> ()
    %r = bufferization.to_tensor %out restrict : memref<8xf32>
    return %r : tensor<8xf32>
  }
}

// -----

// CHECK-LABEL: func.func private @simt_vf(
//  CHECK-SAME:     %arg1: memref<8xf32> {hivm.memory_effect = #hivm.memory_effect<write>}
// CHECK-LABEL: func.func @caller_snapshot(
//       CHECK:   call @simt_vf
//       CHECK:   bufferization.to_tensor %{{.+}} restrict : memref<8xf32>
//       CHECK:   call @simt_vf
module {
  func.func private @simt_vf(%arg0: memref<8xf32>, %arg1: memref<8xf32>) attributes {no_inline, outline, hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    %0 = bufferization.to_tensor %arg0 restrict : memref<8xf32>
    hivm.hir.local_store ins(%arg1 : memref<8xf32>, %0 : tensor<8xf32>)
    return
  }
  func.func @caller_snapshot(%t0: tensor<8xf32>) -> tensor<8xf32> {
    %m0 = bufferization.to_memref %t0 : memref<8xf32>
    %buf = memref.alloc() : memref<8xf32>
    func.call @simt_vf(%m0, %buf) : (memref<8xf32>, memref<8xf32>) -> ()
    %snap = bufferization.to_tensor %buf restrict : memref<8xf32>
    func.call @simt_vf(%m0, %buf) : (memref<8xf32>, memref<8xf32>) -> ()
    return %snap : tensor<8xf32>
  }
}

// -----

// CHECK-LABEL: func.func private @kernel_rw(
//  CHECK-SAME:     %arg1: memref<8xf32> {hivm.memory_effect = #hivm.memory_effect<read_write>}
// CHECK-LABEL: func.func @caller_rw(
//       CHECK:   bufferization.to_tensor %{{.+}} restrict writable : memref<8xf32>
module {
  func.func private @kernel_rw(%arg0: memref<8xf32>, %arg1: memref<8xf32>) attributes {no_inline, outline, hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    %0 = bufferization.to_tensor %arg0 restrict : memref<8xf32>
    hivm.hir.local_store ins(%arg1 : memref<8xf32>, %0 : tensor<8xf32>)
    %1 = bufferization.to_tensor %arg1 restrict : memref<8xf32>
    hivm.hir.local_store ins(%arg1 : memref<8xf32>, %1 : tensor<8xf32>)
    return
  }
  func.func @caller_rw(%t0: tensor<8xf32>) -> tensor<8xf32> {
    %m0 = bufferization.to_memref %t0 : memref<8xf32>
    %out = memref.alloc() : memref<8xf32>
    func.call @kernel_rw(%m0, %out) : (memref<8xf32>, memref<8xf32>) -> ()
    %r = bufferization.to_tensor %out restrict : memref<8xf32>
    return %r : tensor<8xf32>
  }
}
