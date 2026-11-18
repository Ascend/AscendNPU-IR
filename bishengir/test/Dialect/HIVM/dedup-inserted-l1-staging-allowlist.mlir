// DedupInsertedL1StagingPattern uses the loop-shape registry minus
// chunk_kda_bwd_kernel_intra and
// chunk_kda_fwd_kernel_intra_safe_gate_fused. Do not pass
// bypass-shape-registry here.
// RUN: bishengir-opt %s --hivm-combine-optimized-convert-layout --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @parallel_path_fwd_kernel
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: %[[VIEW:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable
// CHECK: hivm.hir.copy ins(%{{.*}}) outs(%[[ALLOC]]
// CHECK-NOT: hivm.hir.copy
// CHECK: return %[[VIEW]], %[[VIEW]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @parallel_path_fwd_kernel(%src: tensor<8x4x16x16xf16>)
      -> (tensor<8x4x16x16xf16>, tensor<8x4x16x16xf16>) {
    %a = memref.alloc() : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
    %va = bufferization.to_tensor %a restrict writable : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
    hivm.hir.copy ins(%src : tensor<8x4x16x16xf16>)
        outs(%a : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
    %b = memref.alloc() : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
    %vb = bufferization.to_tensor %b restrict writable : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
    hivm.hir.copy ins(%src : tensor<8x4x16x16xf16>)
        outs(%b : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
    return %va, %vb : tensor<8x4x16x16xf16>, tensor<8x4x16x16xf16>
  }
}

// -----

// CHECK-LABEL: func.func @chunk_kda_bwd_kernel_intra
// CHECK-COUNT-2: hivm.hir.copy
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @chunk_kda_bwd_kernel_intra(%src: tensor<8x4x16x16xf16>)
      -> (tensor<8x4x16x16xf16>, tensor<8x4x16x16xf16>) {
    %a = memref.alloc() : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
    %va = bufferization.to_tensor %a restrict writable : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
    hivm.hir.copy ins(%src : tensor<8x4x16x16xf16>)
        outs(%a : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
    %b = memref.alloc() : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
    %vb = bufferization.to_tensor %b restrict writable : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
    hivm.hir.copy ins(%src : tensor<8x4x16x16xf16>)
        outs(%b : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
    return %va, %vb : tensor<8x4x16x16xf16>, tensor<8x4x16x16xf16>
  }
}

// -----

// CHECK-LABEL: func.func @chunk_kda_fwd_kernel_intra_safe_gate_fused
// CHECK-COUNT-2: hivm.hir.copy
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @chunk_kda_fwd_kernel_intra_safe_gate_fused(%src: tensor<8x4x16x16xf16>)
      -> (tensor<8x4x16x16xf16>, tensor<8x4x16x16xf16>) {
    %a = memref.alloc() : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
    %va = bufferization.to_tensor %a restrict writable : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
    hivm.hir.copy ins(%src : tensor<8x4x16x16xf16>)
        outs(%a : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
    %b = memref.alloc() : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
    %vb = bufferization.to_tensor %b restrict writable : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
    hivm.hir.copy ins(%src : tensor<8x4x16x16xf16>)
        outs(%b : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
    return %va, %vb : tensor<8x4x16x16xf16>, tensor<8x4x16x16xf16>
  }
}

// -----

// CHECK-LABEL: func.func @dedup_l1_staging_not_registered
// CHECK-COUNT-2: hivm.hir.copy
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dedup_l1_staging_not_registered(%src: tensor<8x4x16x16xf16>)
      -> (tensor<8x4x16x16xf16>, tensor<8x4x16x16xf16>) {
    %a = memref.alloc() : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
    %va = bufferization.to_tensor %a restrict writable : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
    hivm.hir.copy ins(%src : tensor<8x4x16x16xf16>)
        outs(%a : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
    %b = memref.alloc() : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
    %vb = bufferization.to_tensor %b restrict writable : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
    hivm.hir.copy ins(%src : tensor<8x4x16x16xf16>)
        outs(%b : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
    return %va, %vb : tensor<8x4x16x16xf16>, tensor<8x4x16x16xf16>
  }
}
