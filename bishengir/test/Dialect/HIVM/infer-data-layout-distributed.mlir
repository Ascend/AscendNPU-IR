// RUN: bishengir-opt %s -allow-unregistered-dialect -hivm-infer-data-layout -split-input-file | FileCheck %s

// Test populateLayoutToHIVMCustom - data layout inference for distributed CustomOp

// -----
// Test distributed custom op with aclshmem_ptr - should inherit layout from input

// CHECK-LABEL: test_distributed_layout_shmem_ptr
module {
  func.func @test_distributed_layout_shmem_ptr(%arg0: memref<128x32xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32, #hivm.address_space<cc>>
    %0 = hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "aclshmem_ptr_half"} "dist.aclshmem_ptr_half" ins(%arg0, %c0_i32 : memref<128x32xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>, i32) -> memref<128x32xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %alloc_0 = memref.alloc() : memref<128x32xf16, #hivm.address_space<cbuf>>
    // CHECK: hivm.hir.nd2nz
    hivm.hir.load ins(%0 : memref<128x32xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_0 : memref<128x32xf16, #hivm.address_space<cbuf>>)
    %alloc_1 = memref.alloc() : memref<32x128xf16, #hivm.address_space<cbuf>>
    hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%alloc_0, %alloc_1, %false, %c256, %c256, %c256 : memref<128x32xf16, #hivm.address_space<cbuf>>, memref<32x128xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc : memref<128x128xf32, #hivm.address_space<cc>>)
    return
  }
}

// -----
// Test distributed custom op with aclshmem_consume_token

// CHECK-LABEL: test_distributed_layout_consume_token
module {
  func.func @test_distributed_layout_consume_token(%arg0: memref<128x32xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32, #hivm.address_space<cc>>
    %0 = hivm.hir.custom {SrcPtrIndex = array<i32: 0>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "aclshmem_consume_token_half_ptr_2d"} "dist.aclshmem_consume_token_half_ptr_2d" ins(%arg0, %c0_i32 : memref<128x32xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>, i32) -> memref<128x32xf16, #hivm.address_space<gm>>
    %alloc_0 = memref.alloc() : memref<128x32xf16, #hivm.address_space<cbuf>>
    // CHECK: hivm.hir.nd2nz
    hivm.hir.load ins(%0 : memref<128x32xf16, #hivm.address_space<gm>>) outs(%alloc_0 : memref<128x32xf16, #hivm.address_space<cbuf>>)
    %alloc_1 = memref.alloc() : memref<32x128xf16, #hivm.address_space<cbuf>>
    hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%alloc_0, %alloc_1, %false, %c256, %c256, %c256 : memref<128x32xf16, #hivm.address_space<cbuf>>, memref<32x128xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc : memref<128x128xf32, #hivm.address_space<cc>>)
    return
  }
}
