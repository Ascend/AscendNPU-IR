// RUN: bishengir-opt %s -convert-to-hivm-op -split-input-file -allow-unregistered-dialect | FileCheck %s

// Test ConvertToHIVMOp pass for distributed CustomOp

// -----
// Test memref.copy from distributed CustomOp result

module {
  func.func @copy_from_dist_result(%arg0: memref<128x128xf16>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %0 = hivm.hir.custom
        {hivm.is_distributed,
         hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         hivm.pipe = #hivm.pipe<PIPE_V>,
         symbol = "aclshmem_ptr_half"}
        "dist.aclshmem_ptr_half"
        ins(%arg0 : memref<128x128xf16>) -> memref<128x128xf16, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc() : memref<128x128xf16>
    // CHECK: hivm.hir.load
    memref.copy %0, %alloc : memref<128x128xf16, strided<[?, ?], offset: ?>> to memref<128x128xf16>
    return
  }
}

// -----
// Test memref.copy to distributed CustomOp result

module {
  func.func @copy_from_dist_result(%arg0: memref<128x128xf16>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %0 = hivm.hir.custom
        {hivm.is_distributed,
         hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         hivm.pipe = #hivm.pipe<PIPE_V>,
         symbol = "aclshmem_ptr_half"}
        "dist.aclshmem_ptr_half"
        ins(%arg0 : memref<128x128xf16>) -> memref<128x128xf16, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc() : memref<128x128xf16>
    // CHECK: hivm.hir.store
    memref.copy %alloc, %0 : memref<128x128xf16> to memref<128x128xf16, strided<[?, ?], offset: ?>>
    return
  }
}
