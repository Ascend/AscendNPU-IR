// REQUIRES: hivmc
// Regression for the missing `broadcast_scalar_*` payload symbols on the
// reg-based Ascend 950 (dav-c310-vec) template library: a scalar-source
// `hivm.hir.vbrc` lowers to `_mlir_ciface_broadcast_scalar_<type>_to_<dim>d`
// libcalls (see Conversion/HIVMToStandard/RegBase/vbrc-scalar-broadcast.mlir),
// and the device link must resolve them against the c310/920r1 meta_op
// bitcode (lib/Template/lib/RegBase/Vector/ScalarBroadcast.cpp). Before the
// RegBase scalar-broadcast template existed this pipeline failed with
// "undefined symbol: _mlir_ciface_broadcast_scalar_*".
//
// The kernel below forces the f32 1-D, bf16 2-D and i32 1-D entries at once.
// The second kernel forces the bool entries: a full 64xi1 1-D view takes the
// word-pattern fast path, and a 4x17 subview (last dim not a multiple of 16)
// takes the per-bit read-modify-write fallback in ScalarBroadcast.cpp.
// RUN: bishengir-compile %s --target=Ascend950PR_9589 --enable-lir-compile=false -o %t
// RUN: bishengir-compile %s --target=Ascend950PR_9589 --enable-lir-compile=false -o %t2 2>&1 | FileCheck %s --implicit-check-not=undefined

module {
  func.func @test_vbrc_scalar_fill__kernel0(%out_f32: memref<256xf32, #hivm.address_space<gm>>,
                                            %out_bf16: memref<4x16xbf16, #hivm.address_space<gm>>,
                                            %out_i32: memref<64xi32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst_f32 = arith.constant 2.5 : f32
    %ub_f32 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst_f32 : f32) outs(%ub_f32 : memref<256xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%ub_f32 : memref<256xf32, #hivm.address_space<ub>>)
                   outs(%out_f32 : memref<256xf32, #hivm.address_space<gm>>)

    %cst_bf16 = arith.constant 1.0 : bf16
    %ub_bf16 = memref.alloc() : memref<4x16xbf16, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst_bf16 : bf16) outs(%ub_bf16 : memref<4x16xbf16, #hivm.address_space<ub>>)
    hivm.hir.store ins(%ub_bf16 : memref<4x16xbf16, #hivm.address_space<ub>>)
                   outs(%out_bf16 : memref<4x16xbf16, #hivm.address_space<gm>>)

    %cst_i32 = arith.constant 7 : i32
    %ub_i32 = memref.alloc() : memref<64xi32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst_i32 : i32) outs(%ub_i32 : memref<64xi32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%ub_i32 : memref<64xi32, #hivm.address_space<ub>>)
                   outs(%out_i32 : memref<64xi32, #hivm.address_space<gm>>)

    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }

  func.func @test_vbrc_scalar_fill__kernel_bool(%out_i1: memref<64xi1, #hivm.address_space<gm>>,
                                                %out_i1_2d: memref<4x32xi1, #hivm.address_space<gm>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // 1-D bool: contiguous, offset 0, size 64 -> uint16 word-pattern path.
    // The payload has no store_ubuf_to_gm_*_bool libcall yet, so the filled
    // buffers are kept alive with ubuf-to-ubuf copies instead of GM stores.
    %cst_i1 = arith.constant true
    %ub_i1 = memref.alloc() : memref<64xi1, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst_i1 : i1) outs(%ub_i1 : memref<64xi1, #hivm.address_space<ub>>)
    %ub_i1_out = memref.alloc() : memref<64xi1, #hivm.address_space<ub>>
    hivm.hir.copy ins(%ub_i1 : memref<64xi1, #hivm.address_space<ub>>)
                  outs(%ub_i1_out : memref<64xi1, #hivm.address_space<ub>>)

    // 2-D bool 4x17 subview of a 4x32 buffer: the last dim is not a multiple
    // of 16, so the fill must not write whole trailing words; only the 68
    // logical bits of the view may change.
    %ub_i1_2d_alloca = memref.alloc() : memref<4x32xi1, #hivm.address_space<ub>>
    %ub_i1_2d = memref.subview %ub_i1_2d_alloca[0, 0][4, 17][1, 1] :
                            memref<4x32xi1, #hivm.address_space<ub>> to
                            memref<4x17xi1, strided<[32, 1], offset: 0>, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst_i1 : i1)
                  outs(%ub_i1_2d : memref<4x17xi1, strided<[32, 1], offset: 0>, #hivm.address_space<ub>>)
    %ub_i1_2d_out = memref.alloc() : memref<4x32xi1, #hivm.address_space<ub>>
    hivm.hir.copy ins(%ub_i1_2d_alloca : memref<4x32xi1, #hivm.address_space<ub>>)
                  outs(%ub_i1_2d_out : memref<4x32xi1, #hivm.address_space<ub>>)

    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}

// CHECK-NOT: undefined symbol
