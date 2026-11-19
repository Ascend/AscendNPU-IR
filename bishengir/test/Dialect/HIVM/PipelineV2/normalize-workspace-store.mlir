// RUN: bishengir-opt %s -normalize-workspace-store -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @rewrite_workspace_scalar_store
// CHECK:       %[[WS:.*]] = memref_ext.alloc_workspace() : memref<1xf32>
// CHECK-NEXT:  hivm.hir.store ins(%{{.*}} : tensor<1xf32>) outs(%[[WS]] : memref<1xf32>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>, "inserted-store"}
// CHECK-NEXT:  %[[WST:.*]] = bufferization.to_tensor %[[WS]] restrict writable : memref<1xf32>
// CHECK-NOT:   annotation.mark %[[WST]]
// CHECK-NEXT:  %[[EXT:.*]] = tensor.extract %[[WST]][%{{.*}}]
// CHECK-NEXT:  return %[[EXT]] : f32
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @rewrite_workspace_scalar_store(%src: tensor<1xf32>) -> f32
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
  %c0 = arith.constant 0 : index
  %ws = memref_ext.alloc_workspace() : memref<1xf32>
  %wst = bufferization.to_tensor %ws restrict writable : memref<1xf32>
  %stored = hivm.hir.store ins(%src : tensor<1xf32>) outs(%wst : tensor<1xf32>) {"inserted-store"} -> tensor<1xf32>
  annotation.mark %stored {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : tensor<1xf32>
  %ext = tensor.extract %stored[%c0] : tensor<1xf32>
  return %ext : f32
}
}

// -----

// A loop marked by CVPipelining means that pass did pipeline. The whole
// function is left untouched, including a workspace store outside the loop.
// CHECK-LABEL: func.func @skip_when_cv_pipelining_transformed
// CHECK:       %[[WS:.*]] = memref_ext.alloc_workspace() : memref<1xi32>
// CHECK-NEXT:  %[[WST:.*]] = bufferization.to_tensor %[[WS]] restrict writable : memref<1xi32>
// CHECK-NEXT:  %[[STORED:.*]] = hivm.hir.store ins(%{{.*}} : tensor<1xi32>) outs(%[[WST]] : tensor<1xi32>) {"hivm.inserted-store"} -> tensor<1xi32>
// CHECK-NEXT:  annotation.mark %[[STORED]] {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : tensor<1xi32>
// CHECK:       } {"hivm.legacy-cv-pipeline"}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @skip_when_cv_pipelining_transformed(%src: tensor<1xi32>) -> i32
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %ws = memref_ext.alloc_workspace() : memref<1xi32>
  %wst = bufferization.to_tensor %ws restrict writable : memref<1xi32>
  %stored = hivm.hir.store ins(%src : tensor<1xi32>) outs(%wst : tensor<1xi32>) {"hivm.inserted-store"} -> tensor<1xi32>
  annotation.mark %stored {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : tensor<1xi32>
  scf.for %i = %c0 to %c1 step %c1 {
    scf.yield
  } {"hivm.legacy-cv-pipeline"}
  %ext = tensor.extract %stored[%c0] : tensor<1xi32>
  return %ext : i32
}
}

// -----

// Stores inside legacy-cv-pipeline loops or targeting non-workspace memrefs
// are left untouched.
// CHECK-LABEL: func.func @skip_legacy_and_non_workspace
// CHECK:       %[[ALLOC:.*]] = memref.alloc() : memref<1xf32>
// CHECK-NEXT:  %[[ALLOC_T:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<1xf32>
// CHECK-NEXT:  %{{.*}} = hivm.hir.store ins(%{{.*}} : tensor<1xf32>) outs(%[[ALLOC_T]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK:       scf.for
// CHECK:         %[[WS:.*]] = memref_ext.alloc_workspace() : memref<1xf32>
// CHECK-NEXT:    %[[WST:.*]] = bufferization.to_tensor %[[WS]] restrict writable : memref<1xf32>
// CHECK-NEXT:    %{{.*}} = hivm.hir.store ins(%{{.*}} : tensor<1xf32>) outs(%[[WST]] : tensor<1xf32>) -> tensor<1xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @skip_legacy_and_non_workspace(%src: tensor<1xf32>)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %alloc = memref.alloc() : memref<1xf32>
  %alloc_t = bufferization.to_tensor %alloc restrict writable : memref<1xf32>
  %s0 = hivm.hir.store ins(%src : tensor<1xf32>) outs(%alloc_t : tensor<1xf32>) -> tensor<1xf32>
  scf.for %i = %c0 to %c1 step %c1 {
    %ws = memref_ext.alloc_workspace() : memref<1xf32>
    %wst = bufferization.to_tensor %ws restrict writable : memref<1xf32>
    %s1 = hivm.hir.store ins(%src : tensor<1xf32>) outs(%wst : tensor<1xf32>) -> tensor<1xf32>
  } {"hivm.legacy-cv-pipeline"}
  return
}
}
