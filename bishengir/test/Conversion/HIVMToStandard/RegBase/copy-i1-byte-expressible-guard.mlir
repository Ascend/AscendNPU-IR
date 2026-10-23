// RUN: bishengir-opt %s -convert-hivm-to-std -split-input-file -verify-diagnostics | FileCheck %s

// Lowering coverage for the i1 byte-expressibility guard in
// CopyOpToLibraryCallPattern. The DMA library reinterprets both buffers as
// i8 via view_as; only byte-expressible layouts (see
// isI1LayoutByteExpressible) survive that bit-exactly and keep the library
// call. Everything else must not reach the library (EnableStrideAlign
// decomposes it into elementwise transfers; CopyOp::verify rejects what
// slips through).

// -----
// Byte-expressible: 13x8xi1 with 256-bit row stride and offset 0.
// CHECK-LABEL: func.func @byte_expressible_i1_copy_libcall
// CHECK: call @copy_ubuf_to_ubuf_2d_bool
// CHECK-NOT: scf.for
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @byte_expressible_i1_copy_libcall() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %src_storage = memref.alloc() : memref<256xi1, #hivm.address_space<ub>>
    %src = memref.reinterpret_cast %src_storage to offset: [0], sizes: [13, 8], strides: [256, 1]
      : memref<256xi1, #hivm.address_space<ub>> to memref<13x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    %dst_storage = memref.alloc() : memref<256xi1, #hivm.address_space<ub>>
    %dst = memref.reinterpret_cast %dst_storage to offset: [0], sizes: [13, 8], strides: [256, 1]
      : memref<256xi1, #hivm.address_space<ub>> to memref<13x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.copy ins(%src : memref<13x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
                  outs(%dst : memref<13x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>)
    return
  }
}

// -----
// Dense rank-1 i1 whose last-dim size is a multiple of 8: byte-expressible
// even though the element stride is 1 bit (view_as carries the last-dim
// stride verbatim).
// CHECK-LABEL: func.func @dense_i1_copy_libcall
// CHECK: call @copy_ubuf_to_ubuf_1d_bool
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @dense_i1_copy_libcall() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %src = memref.alloc() : memref<104xi1, #hivm.address_space<ub>>
    %dst = memref.alloc() : memref<104xi1, #hivm.address_space<ub>>
    hivm.hir.copy ins(%src : memref<104xi1, #hivm.address_space<ub>>)
                  outs(%dst : memref<104xi1, #hivm.address_space<ub>>)
    return
  }
}

// -----
// A non-unit innermost stride is not byte-expressible: view_as keeps the
// stride number but in i8-element units, so the library would read
// consecutive bytes instead of every second bit.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @strided_last_dim_i1_copy_rejected() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %src_storage = memref.alloc() : memref<4096xi1, #hivm.address_space<ub>>
    %src = memref.reinterpret_cast %src_storage to offset: [0], sizes: [13, 8], strides: [256, 2]
      : memref<4096xi1, #hivm.address_space<ub>> to memref<13x8xi1, strided<[256, 2]>, #hivm.address_space<ub>>
    %dst_storage = memref.alloc() : memref<4096xi1, #hivm.address_space<ub>>
    %dst = memref.reinterpret_cast %dst_storage to offset: [0], sizes: [13, 8], strides: [256, 2]
      : memref<4096xi1, #hivm.address_space<ub>> to memref<13x8xi1, strided<[256, 2]>, #hivm.address_space<ub>>
    // expected-error@+1 {{'hivm.hir.copy' op i1 copy layout is not byte-expressible}}
    hivm.hir.copy ins(%src : memref<13x8xi1, strided<[256, 2]>, #hivm.address_space<ub>>)
                  outs(%dst : memref<13x8xi1, strided<[256, 2]>, #hivm.address_space<ub>>)
    return
  }
}

// -----
// 19-bit rows are not byte-expressible; verification rejects the copy before
// the pattern could emit a corrupting library call.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @bad_i1_copy_rejected() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
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
// Byte-aligned rows but a non-multiple-of-8 last-dim size (19): still not
// byte-expressible, because view_as widens the last dim to
// CEIL_DIV(19, 8) = 3 bytes.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @bad_last_dim_i1_copy_rejected() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %src_storage = memref.alloc() : memref<2048xi1, #hivm.address_space<ub>>
    %src = memref.reinterpret_cast %src_storage to offset: [0], sizes: [13, 19], strides: [160, 1]
      : memref<2048xi1, #hivm.address_space<ub>> to memref<13x19xi1, strided<[160, 1]>, #hivm.address_space<ub>>
    %dst_storage = memref.alloc() : memref<2048xi1, #hivm.address_space<ub>>
    %dst = memref.reinterpret_cast %dst_storage to offset: [0], sizes: [13, 19], strides: [160, 1]
      : memref<2048xi1, #hivm.address_space<ub>> to memref<13x19xi1, strided<[160, 1]>, #hivm.address_space<ub>>
    // expected-error@+1 {{'hivm.hir.copy' op i1 copy layout is not byte-expressible}}
    hivm.hir.copy ins(%src : memref<13x19xi1, strided<[160, 1]>, #hivm.address_space<ub>>)
                  outs(%dst : memref<13x19xi1, strided<[160, 1]>, #hivm.address_space<ub>>)
    return
  }
}
