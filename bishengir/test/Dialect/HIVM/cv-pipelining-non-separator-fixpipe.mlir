// REQUIRES: asserts
// RUN: bishengir-opt -cv-pipelining="set-depth-in-unroll-mode=2" -allow-unregistered-dialect -split-input-file %s | FileCheck %s --check-prefix=CHECK-IR
// RUN: bishengir-opt -mlir-disable-threading -cv-pipelining="set-depth-in-unroll-mode=2" -debug-only=cv-pipelining -allow-unregistered-dialect -split-input-file %s 2>&1 | FileCheck %s --check-prefixes=CHECK-IR,CHECK-DEBUG

// Test 1: Cross-core fixpipe (to UB) IS a separator.
// In V1 -> C -> V2, fixpipe(UB) separates V1 and V2 across C into distinct VECTOR workitems.
// CHECK-DEBUG: [build] Separators:
// CHECK-DEBUG-NEXT:    hivm.hir.store
// CHECK-DEBUG-NEXT:    hivm.hir.fixpipe{{.*}}#hivm.address_space<ub>
// CHECK-DEBUG-NEXT:    hivm.hir.store

// CHECK-IR-LABEL: func.func @test_vcv_with_ub_fixpipe
// CHECK-IR: scf.for
// CHECK-IR:   scf.for
// CHECK-IR:   {hivm.loop_core_type = #hivm.tcore_type<VECTOR>
// CHECK-IR:   scf.for
// CHECK-IR:   {hivm.loop_core_type = #hivm.tcore_type<CUBE>
// CHECK-IR:   scf.for
// CHECK-IR:   {hivm.loop_core_type = #hivm.tcore_type<VECTOR>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_vcv_with_ub_fixpipe(%gmIn1: memref<16x16xf16>, %gmIn2: memref<16x16xf16>, %gmOut1: memref<16x16xf16>, %gmOut2: memref<16x16xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %tensor1 = bufferization.to_tensor %gmIn1 : memref<16x16xf16>
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %step = arith.constant 2 : i32
    %bound = "some_op"() : () -> i32
    %vdest = tensor.empty() : tensor<16x16xf16>
    scf.for %i = %c0 to %bound step %step : i32 {
      // V1: Vector op 1
      %v1_alloc = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%gmIn1 : memref<16x16xf16>) outs(%v1_alloc : memref<16x16xf16, #hivm.address_space<ub>>)
      %v1_t = bufferization.to_tensor %v1_alloc : memref<16x16xf16, #hivm.address_space<ub>>
      %v1_exp = hivm.hir.vexp ins(%v1_t : tensor<16x16xf16>) outs(%vdest : tensor<16x16xf16>) -> tensor<16x16xf16>
      hivm.hir.store ins(%v1_exp : tensor<16x16xf16>) outs(%gmOut1 : memref<16x16xf16>)

      // C: Cube op + Fixpipe to UB (separator)
      %allocC = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%gmIn2 : memref<16x16xf16>) outs(%allocC : memref<16x16xf16>)
      %tensor2 = bufferization.to_tensor %allocC : memref<16x16xf16>
      %dest = tensor.empty() : tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%tensor1, %tensor2, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ub0 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf16>) outs(%ub0 : memref<16x16xf16, #hivm.address_space<ub>>)

      // V2: Vector op 2
      %ub_tensor = bufferization.to_tensor %ub0 : memref<16x16xf16, #hivm.address_space<ub>>
      %v2_exp = hivm.hir.vexp ins(%ub_tensor : tensor<16x16xf16>) outs(%vdest : tensor<16x16xf16>) -> tensor<16x16xf16>
      hivm.hir.store ins(%v2_exp : tensor<16x16xf16>) outs(%gmOut2 : memref<16x16xf16>)
    }
    return
  }
}

// -----

// Test 2: Non-cross-core fixpipe (to L1 / cbuf) is NOT a separator.
// In V1 -> C -> V2, fixpipe(cbuf) is not a separator, so V1 and V2 are merged into a single VECTOR workitem.
// CHECK-DEBUG: [build] Separators:
// CHECK-DEBUG-NEXT:    hivm.hir.store
// CHECK-DEBUG-NEXT:    hivm.hir.store

// CHECK-IR-LABEL: func.func @test_vcv_with_cbuf_fixpipe
// CHECK-IR: scf.for
// CHECK-IR:   scf.for
// CHECK-IR:   {hivm.loop_core_type = #hivm.tcore_type<VECTOR>
// CHECK-IR:   scf.for
// CHECK-IR:   {hivm.loop_core_type = #hivm.tcore_type<CUBE>
// CHECK-IR: cv_unrolled_loop
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_vcv_with_cbuf_fixpipe(%gmIn1: memref<16x16xf16>, %gmIn2: memref<16x16xf16>, %gmOut1: memref<16x16xf16>, %gmOut2: memref<16x16xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %tensor1 = bufferization.to_tensor %gmIn1 : memref<16x16xf16>
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %step = arith.constant 2 : i32
    %bound = "some_op"() : () -> i32
    %vdest = tensor.empty() : tensor<16x16xf16>
    scf.for %i = %c0 to %bound step %step : i32 {
      // V1: Vector op 1
      %v1_alloc = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%gmIn1 : memref<16x16xf16>) outs(%v1_alloc : memref<16x16xf16, #hivm.address_space<ub>>)
      %v1_t = bufferization.to_tensor %v1_alloc : memref<16x16xf16, #hivm.address_space<ub>>
      %v1_exp = hivm.hir.vexp ins(%v1_t : tensor<16x16xf16>) outs(%vdest : tensor<16x16xf16>) -> tensor<16x16xf16>
      hivm.hir.store ins(%v1_exp : tensor<16x16xf16>) outs(%gmOut1 : memref<16x16xf16>)

      // C: Cube op + Fixpipe to cbuf (L1, non-cross-core, NOT a separator)
      %allocC = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%gmIn2 : memref<16x16xf16>) outs(%allocC : memref<16x16xf16>)
      %tensor2 = bufferization.to_tensor %allocC : memref<16x16xf16>
      %dest = tensor.empty() : tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%tensor1, %tensor2, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %cbuf0 = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf16>) outs(%cbuf0 : memref<16x16xf16, #hivm.address_space<cbuf>>)

      // V2: Vector op 2
      %v2_alloc = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%gmIn1 : memref<16x16xf16>) outs(%v2_alloc : memref<16x16xf16, #hivm.address_space<ub>>)
      %v2_t = bufferization.to_tensor %v2_alloc : memref<16x16xf16, #hivm.address_space<ub>>
      %v2_exp = hivm.hir.vexp ins(%v2_t : tensor<16x16xf16>) outs(%vdest : tensor<16x16xf16>) -> tensor<16x16xf16>
      hivm.hir.store ins(%v2_exp : tensor<16x16xf16>) outs(%gmOut2 : memref<16x16xf16>)
    }
    return
  }
}
