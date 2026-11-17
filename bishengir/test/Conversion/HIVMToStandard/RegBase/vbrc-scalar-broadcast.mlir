// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9589 -convert-hivm-to-std -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9589 -convert-hivm-to-std="mark-libcall-noinline=true" -split-input-file | FileCheck %s --check-prefix=NOINLINE
//
// Scalar-source `hivm.hir.vbrc` must lower to the
// `broadcast_scalar_<type>_to_<dim>d` library-call family on the reg-based
// (Ascend 950) path. The c310/920r1 template payload must provide these
// symbols (see lib/Template/lib/RegBase/Vector/ScalarBroadcast.cpp); the
// end-to-end link regression lives in
// test/bishengir-compile/hivm/compile-vbrc-scalar-broadcast.mlir.

module {
  // CHECK-LABEL: vbrc_scalar_1d
  func.func @vbrc_scalar_1d() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: call @broadcast_scalar_int8_t_to_1d
    %c_i8 = arith.constant 1 : i8
    %d_i8 = memref.alloca() : memref<64xi8, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%c_i8 : i8) outs(%d_i8 : memref<64xi8, #hivm.address_space<ub>>)
    // CHECK: call @broadcast_scalar_int16_t_to_1d
    %c_i16 = arith.constant 2 : i16
    %d_i16 = memref.alloca() : memref<64xi16, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%c_i16 : i16) outs(%d_i16 : memref<64xi16, #hivm.address_space<ub>>)
    // CHECK: call @broadcast_scalar_int32_t_to_1d
    %c_i32 = arith.constant 3 : i32
    %d_i32 = memref.alloca() : memref<64xi32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%c_i32 : i32) outs(%d_i32 : memref<64xi32, #hivm.address_space<ub>>)
    // CHECK: call @broadcast_scalar_int64_t_to_1d
    %c_i64 = arith.constant 4 : i64
    %d_i64 = memref.alloca() : memref<64xi64, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%c_i64 : i64) outs(%d_i64 : memref<64xi64, #hivm.address_space<ub>>)
    // CHECK: call @broadcast_scalar_half_to_1d
    %c_f16 = arith.constant 5.0 : f16
    %d_f16 = memref.alloca() : memref<64xf16, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%c_f16 : f16) outs(%d_f16 : memref<64xf16, #hivm.address_space<ub>>)
    // CHECK: call @broadcast_scalar_bfloat16_t_to_1d
    %c_bf16 = arith.constant 6.0 : bf16
    %d_bf16 = memref.alloca() : memref<64xbf16, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%c_bf16 : bf16) outs(%d_bf16 : memref<64xbf16, #hivm.address_space<ub>>)
    // CHECK: call @broadcast_scalar_float_to_1d
    %c_f32 = arith.constant 7.0 : f32
    %d_f32 = memref.alloca() : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%c_f32 : f32) outs(%d_f32 : memref<64xf32, #hivm.address_space<ub>>)
    // CHECK: call @broadcast_scalar_bool_to_1d
    %c_i1 = arith.constant true
    %d_i1 = memref.alloca() : memref<64xi1, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%c_i1 : i1) outs(%d_i1 : memref<64xi1, #hivm.address_space<ub>>)
    return
  }
}

// -----

module {
  // CHECK-LABEL: vbrc_scalar_2d_subview
  // A strided 2-D destination keeps the `_to_2d` suffix; the strides stay in
  // the memref argument, not in the symbol name.
  func.func @vbrc_scalar_2d_subview() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: call @broadcast_scalar_int32_t_to_2d
    %c_i32 = arith.constant 1 : i32
    %d_i32_alloca = memref.alloca() : memref<4x16xi32, #hivm.address_space<ub>>
    %d_i32 = memref.subview %d_i32_alloca[0, 0][4, 10][1, 1] :
                            memref<4x16xi32, #hivm.address_space<ub>> to
                            memref<4x10xi32, strided<[16, 1], offset: 0>, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%c_i32 : i32)
                  outs(%d_i32 : memref<4x10xi32, strided<[16, 1], offset: 0>, #hivm.address_space<ub>>)
    // CHECK: call @broadcast_scalar_bfloat16_t_to_2d
    %c_bf16 = arith.constant 2.0 : bf16
    %d_bf16_alloca = memref.alloca() : memref<4x16xbf16, #hivm.address_space<ub>>
    %d_bf16 = memref.subview %d_bf16_alloca[0, 0][4, 10][1, 1] :
                             memref<4x16xbf16, #hivm.address_space<ub>> to
                             memref<4x10xbf16, strided<[16, 1], offset: 0>, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%c_bf16 : bf16)
                  outs(%d_bf16 : memref<4x10xbf16, strided<[16, 1], offset: 0>, #hivm.address_space<ub>>)
    // CHECK: call @broadcast_scalar_bool_to_2d
    %c_i1 = arith.constant false
    %d_i1_alloca = memref.alloca() : memref<4x16xi1, #hivm.address_space<ub>>
    %d_i1 = memref.subview %d_i1_alloca[0, 0][4, 10][1, 1] :
                           memref<4x16xi1, #hivm.address_space<ub>> to
                           memref<4x10xi1, strided<[16, 1], offset: 0>, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%c_i1 : i1)
                  outs(%d_i1 : memref<4x10xi1, strided<[16, 1], offset: 0>, #hivm.address_space<ub>>)
    return
  }
}

// -----

module {
  // CHECK: func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {
  // CHECK-SAME: llvm.emit_c_interface
  // NOINLINE: func.func private @broadcast_scalar_float_to_1d({{.*}}) attributes {
  // NOINLINE-SAME: hacc.noinline
  // CHECK-LABEL: vbrc_scalar_callee_signature
  // The scalar keeps its own type and the destination memref is passed as the
  // C-interface memref descriptor.
  func.func @vbrc_scalar_callee_signature() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 2.5 : f32
    %dst = memref.alloca() : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst : f32) outs(%dst : memref<256xf32, #hivm.address_space<ub>>)
    return
  }
}
