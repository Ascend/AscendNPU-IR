// RUN: bishengir-opt %s -hivm-enable-stride-align -split-input-file -verify-diagnostics | FileCheck %s

// Verifier coverage for hivm::CopyOp on i1 buffers whose layout the DMA
// copy library cannot express (see isI1LayoutByteExpressible). Such copies
// must be rejected and decomposed into elementwise accesses instead (see
// hivm-enable-stride-align-i1-bitexact-copy.mlir).

// -----
// Regbase target + non-byte-expressible i1 layout (19 not a multiple of 8)
// must fail verification.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @bad_i1_strided_copy() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %src_storage = memref.alloc() : memref<256xi1, #hivm.address_space<ub>>
    %src = memref.reinterpret_cast %src_storage to offset: [0], sizes: [13, 19], strides: [256, 1]
      : memref<256xi1, #hivm.address_space<ub>> to memref<13x19xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    %dst_storage = memref.alloc() : memref<256xi1, #hivm.address_space<ub>>
    %dst = memref.reinterpret_cast %dst_storage to offset: [0], sizes: [13, 19], strides: [256, 1]
      : memref<256xi1, #hivm.address_space<ub>> to memref<13x19xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    // expected-error@+1 {{'hivm.hir.copy' op i1 copy layout is not byte-expressible}}
    hivm.hir.copy ins(%src : memref<13x19xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
                  outs(%dst : memref<13x19xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
    return
  }
}

// -----
// Dynamic offset cannot be proven byte-aligned, so it is rejected too.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @dyn_offset_i1_copy(%offset: index) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %src_storage = memref.alloc() : memref<512xi1, #hivm.address_space<ub>>
    %src = memref.reinterpret_cast %src_storage to offset: [%offset], sizes: [2, 8], strides: [8, 1]
      : memref<512xi1, #hivm.address_space<ub>> to memref<2x8xi1, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    %dst_storage = memref.alloc() : memref<512xi1, #hivm.address_space<ub>>
    %dst = memref.reinterpret_cast %dst_storage to offset: [0], sizes: [2, 8], strides: [8, 1]
      : memref<512xi1, #hivm.address_space<ub>> to memref<2x8xi1, strided<[8, 1]>, #hivm.address_space<ub>>
    // expected-error@+1 {{'hivm.hir.copy' op i1 copy layout is not byte-expressible}}
    hivm.hir.copy ins(%src : memref<2x8xi1, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>)
                  outs(%dst : memref<2x8xi1, strided<[8, 1]>, #hivm.address_space<ub>>)
    return
  }
}

// -----
// A non-unit innermost stride is not byte-expressible: view_as keeps the
// stride number but reinterprets it in i8-element units, so the library
// would read bits 0-7 instead of 0, 2, ..., 14.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @strided_last_dim_i1_copy() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %src_storage = memref.alloc() : memref<32xi1, #hivm.address_space<ub>>
    %src = memref.reinterpret_cast %src_storage to offset: [0], sizes: [8], strides: [2]
      : memref<32xi1, #hivm.address_space<ub>> to memref<8xi1, strided<[2]>, #hivm.address_space<ub>>
    %dst_storage = memref.alloc() : memref<32xi1, #hivm.address_space<ub>>
    %dst = memref.reinterpret_cast %dst_storage to offset: [0], sizes: [8], strides: [2]
      : memref<32xi1, #hivm.address_space<ub>> to memref<8xi1, strided<[2]>, #hivm.address_space<ub>>
    // expected-error@+1 {{'hivm.hir.copy' op i1 copy layout is not byte-expressible}}
    hivm.hir.copy ins(%src : memref<8xi1, strided<[2]>, #hivm.address_space<ub>>)
                  outs(%dst : memref<8xi1, strided<[2]>, #hivm.address_space<ub>>)
    return
  }
}

// -----
// Byte-expressible i1 layout (last dim 8, non-last stride 256, offset 0)
// keeps the plain CopyOp.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @byte_expressible_i1_copy() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %src_storage = memref.alloc() : memref<256xi1, #hivm.address_space<ub>>
    %src = memref.reinterpret_cast %src_storage to offset: [0], sizes: [13, 8], strides: [256, 1]
      : memref<256xi1, #hivm.address_space<ub>> to memref<13x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    %dst_storage = memref.alloc() : memref<256xi1, #hivm.address_space<ub>>
    %dst = memref.reinterpret_cast %dst_storage to offset: [0], sizes: [13, 8], strides: [256, 1]
      : memref<256xi1, #hivm.address_space<ub>> to memref<13x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    // CHECK-LABEL: func.func @byte_expressible_i1_copy
    // CHECK: hivm.hir.copy ins({{.*}} : memref<13x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
    hivm.hir.copy ins(%src : memref<13x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
                  outs(%dst : memref<13x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
    return
  }
}

// -----
// Non-regbase target keeps accepting the copy: the membase DMA lowering
// does not go through the byte-reinterpreting library entries.
module {
  func.func @non_regbase_i1_copy() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %src_storage = memref.alloc() : memref<256xi1, #hivm.address_space<ub>>
    %src = memref.reinterpret_cast %src_storage to offset: [0], sizes: [13, 19], strides: [256, 1]
      : memref<256xi1, #hivm.address_space<ub>> to memref<13x19xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    %dst_storage = memref.alloc() : memref<256xi1, #hivm.address_space<ub>>
    %dst = memref.reinterpret_cast %dst_storage to offset: [0], sizes: [13, 19], strides: [256, 1]
      : memref<256xi1, #hivm.address_space<ub>> to memref<13x19xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    // CHECK-LABEL: func.func @non_regbase_i1_copy
    // CHECK: hivm.hir.copy ins({{.*}} : memref<13x19xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
    hivm.hir.copy ins(%src : memref<13x19xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
                  outs(%dst : memref<13x19xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
    return
  }
}
