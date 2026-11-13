// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9589 -hivm-plan-memory | FileCheck %s --check-prefix=PLAN --implicit-check-not=memref.alloc
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9589 -hivm-plan-memory -convert-hivm-to-std | FileCheck %s --check-prefix=LOWER

// Compile-only coverage. This input has no cross-pipe synchronization and is
// not a runnable kernel; the caller supplies explicit events in device tests.
// Sequential use still preserves distinct physical slots: A1=2048 B1=4096.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {

  // PLAN-LABEL: func.func @slots
  // LOWER-LABEL: func.func @slots
  func.func @slots(%src_a: memref<2x2x16x16xf16, #hivm.address_space<cbuf>>, %src_b: memref<4x2x16x16xf16, #hivm.address_space<cbuf>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %m = arith.constant 32 : index
    %k = arith.constant 32 : index
    %n = arith.constant 64 : index
    %false = arith.constant false
    %true = arith.constant true
    // PLAN-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
    // PLAN-DAG: %[[ASLOT:.*]] = arith.constant 2048 : i64
    // PLAN-DAG: %[[BSLOT:.*]] = arith.constant 4096 : i64
    // PLAN-DAG: hivm.hir.pointer_cast(%[[ZERO]]) : memref<2x2x16x16xf16, #hivm.address_space<ca>>
    %a0 = memref.alloc() : memref<2x2x16x16xf16, #hivm.address_space<ca>>
    // PLAN-DAG: hivm.hir.pointer_cast(%[[ASLOT]]) : memref<2x2x16x16xf16, #hivm.address_space<ca>>
    %a1 = memref.alloc() : memref<2x2x16x16xf16, #hivm.address_space<ca>>
    // PLAN-DAG: hivm.hir.pointer_cast(%[[ZERO]]) : memref<2x4x16x16xf16, #hivm.address_space<cb>>
    %b0 = memref.alloc() : memref<2x4x16x16xf16, #hivm.address_space<cb>>
    // PLAN-DAG: hivm.hir.pointer_cast(%[[BSLOT]]) : memref<2x4x16x16xf16, #hivm.address_space<cb>>
    %b1 = memref.alloc() : memref<2x4x16x16xf16, #hivm.address_space<cb>>
    // PLAN-DAG: hivm.hir.pointer_cast(%[[ZERO]]) : memref<4x2x16x16xf32, #hivm.address_space<cc>>
    %c = memref.alloc() : memref<4x2x16x16xf32, #hivm.address_space<cc>>
    // PLAN-NOT: memref.alloc
    // LOWER: call @l12l0_cbuf_to_ca_half
    hivm.hir.l12l0 ins(%src_a : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>)
      outs(%a0 : memref<2x2x16x16xf16, #hivm.address_space<ca>>)
      {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 32}
    // LOWER: call @l12l0_cbuf_to_cb_half
    hivm.hir.l12l0 ins(%src_b : memref<4x2x16x16xf16, #hivm.address_space<cbuf>>)
      outs(%b0 : memref<2x4x16x16xf16, #hivm.address_space<cb>>)
      {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 64}
    // LOWER: call @mmadl0_a5_half_to_float(
    hivm.hir.mmadL0 ins(%a0, %b0, %m, %k, %n, %false, %false, %true :
      memref<2x2x16x16xf16, #hivm.address_space<ca>>, memref<2x4x16x16xf16, #hivm.address_space<cb>>, index, index, index, i1, i1, i1)
      outs(%c : memref<4x2x16x16xf32, #hivm.address_space<cc>>)
    // LOWER: call @l12l0_cbuf_to_ca_half
    hivm.hir.l12l0 ins(%src_a : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>)
      outs(%a1 : memref<2x2x16x16xf16, #hivm.address_space<ca>>)
      {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 32}
    // LOWER: call @l12l0_cbuf_to_cb_half
    hivm.hir.l12l0 ins(%src_b : memref<4x2x16x16xf16, #hivm.address_space<cbuf>>)
      outs(%b1 : memref<2x4x16x16xf16, #hivm.address_space<cb>>)
      {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 64}
    // LOWER: call @mmadl0_a5_half_to_float(
    hivm.hir.mmadL0 ins(%a1, %b1, %m, %k, %n, %false, %false, %false :
      memref<2x2x16x16xf16, #hivm.address_space<ca>>, memref<2x4x16x16xf16, #hivm.address_space<cb>>, index, index, index, i1, i1, i1)
      outs(%c : memref<4x2x16x16xf32, #hivm.address_space<cc>>)
    return
  }
}
