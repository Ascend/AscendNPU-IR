// RUN: bishengir-opt %s -hivm-graph-sync-solver='solver-version=v1 enable-unit-flag=true' | FileCheck %s --check-prefix=SYNC --implicit-check-not='unit_flag<ENABLED'
// RUN: bishengir-opt %s -hivm-graph-sync-solver='solver-version=v2 enable-unit-flag=true' | FileCheck %s --check-prefix=SYNC --implicit-check-not='unit_flag<ENABLED'
// RUN: bishengir-opt %s -hivm-graph-sync-solver='solver-version=v1 enable-unit-flag=false' | FileCheck %s --check-prefix=SYNC
// RUN: bishengir-opt %s -hivm-graph-sync-solver='solver-version=v2 enable-unit-flag=false' | FileCheck %s --check-prefix=SYNC
// RUN: bishengir-opt %s -hivm-graph-sync-solver='solver-version=v1 enable-unit-flag=true' -convert-hivm-to-std | FileCheck %s --check-prefix=LOWER
// RUN: bishengir-opt %s -hivm-graph-sync-solver='solver-version=v2 enable-unit-flag=true' -convert-hivm-to-std | FileCheck %s --check-prefix=LOWER
// RUN: bishengir-opt %s -convert-hivm-to-std | FileCheck %s --check-prefix=LOWER --implicit-check-not=hivm.hir.set_flag --implicit-check-not=hivm.hir.wait_flag

// Allocation-free pass tests. Operands supplied by the caller isolate the
// synchronization contract from PlanMemory. The final RUN only checks lowering
// without GSS; this unsynchronized input must not be executed on a device.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {

  // SYNC-LABEL: func.func @a_dependency
  // LOWER-LABEL: func.func @a_dependency
  func.func @a_dependency(%a: memref<2x2x16x16xf16, #hivm.address_space<ca>>, %b: memref<2x4x16x16xf16, #hivm.address_space<cb>>, %c: memref<4x2x16x16xf32, #hivm.address_space<cc>>, %src: memref<2x2x16x16xf16, #hivm.address_space<cbuf>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %m = arith.constant 32 : index
    %k = arith.constant 32 : index
    %n = arith.constant 64 : index
    %false = arith.constant false
    %true = arith.constant true
    // SYNC: hivm.hir.l12l0
    // LOWER: call @l12l0_cbuf_to_ca_half
    hivm.hir.l12l0 ins(%src : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>) outs(%a : memref<2x2x16x16xf16, #hivm.address_space<ca>>)
      {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 32}
    // SYNC: hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_M>, <[[F_a:EVENT_ID[0-9]+]]>]
    // SYNC: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_M>, <[[F_a]]>]
    // SYNC: hivm.hir.mmadL0
    // LOWER: call @mmadl0_a5_half_to_float(
    hivm.hir.mmadL0 ins(%a, %b, %m, %k, %n, %false, %false, %true :
      memref<2x2x16x16xf16, #hivm.address_space<ca>>, memref<2x4x16x16xf16, #hivm.address_space<cb>>, index, index, index, i1, i1, i1)
      outs(%c : memref<4x2x16x16xf32, #hivm.address_space<cc>>)
    // Reusing this physical slot must wait for MMAD to finish reading it.
    // SYNC: hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <[[B_a:EVENT_ID[0-9]+]]>]
    // SYNC: hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <[[B_a]]>]
    // SYNC: hivm.hir.l12l0
    // LOWER: call @l12l0_cbuf_to_ca_half
    hivm.hir.l12l0 ins(%src : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>) outs(%a : memref<2x2x16x16xf16, #hivm.address_space<ca>>)
      {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 32}
    return
  }

  // SYNC-LABEL: func.func @b_dependency
  // LOWER-LABEL: func.func @b_dependency
  func.func @b_dependency(%a: memref<2x2x16x16xf16, #hivm.address_space<ca>>, %b: memref<2x4x16x16xf16, #hivm.address_space<cb>>, %c: memref<4x2x16x16xf32, #hivm.address_space<cc>>, %src: memref<4x2x16x16xf16, #hivm.address_space<cbuf>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %m = arith.constant 32 : index
    %k = arith.constant 32 : index
    %n = arith.constant 64 : index
    %false = arith.constant false
    %true = arith.constant true
    // SYNC: hivm.hir.l12l0
    // LOWER: call @l12l0_cbuf_to_cb_half
    hivm.hir.l12l0 ins(%src : memref<4x2x16x16xf16, #hivm.address_space<cbuf>>) outs(%b : memref<2x4x16x16xf16, #hivm.address_space<cb>>)
      {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 64}
    // SYNC: hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_M>, <[[F_b:EVENT_ID[0-9]+]]>]
    // SYNC: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_M>, <[[F_b]]>]
    // SYNC: hivm.hir.mmadL0
    // LOWER: call @mmadl0_a5_half_to_float(
    hivm.hir.mmadL0 ins(%a, %b, %m, %k, %n, %false, %false, %true :
      memref<2x2x16x16xf16, #hivm.address_space<ca>>, memref<2x4x16x16xf16, #hivm.address_space<cb>>, index, index, index, i1, i1, i1)
      outs(%c : memref<4x2x16x16xf32, #hivm.address_space<cc>>)
    // Reusing this physical slot must wait for MMAD to finish reading it.
    // SYNC: hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <[[B_b:EVENT_ID[0-9]+]]>]
    // SYNC: hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <[[B_b]]>]
    // SYNC: hivm.hir.l12l0
    // LOWER: call @l12l0_cbuf_to_cb_half
    hivm.hir.l12l0 ins(%src : memref<4x2x16x16xf16, #hivm.address_space<cbuf>>) outs(%b : memref<2x4x16x16xf16, #hivm.address_space<cb>>)
      {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 64}
    return
  }

  // Even with UnitFlag optimization enabled, standalone MMAD must retain
  // ordinary M/FIX events and remain legal for its DISABLED-only lowering.
  // SYNC-LABEL: func.func @mmad_fixpipe
  // LOWER-LABEL: func.func @mmad_fixpipe
  func.func @mmad_fixpipe(%a: memref<2x2x16x16xf16, #hivm.address_space<ca>>, %b: memref<2x4x16x16xf16, #hivm.address_space<cb>>, %c: memref<4x2x16x16xf32, #hivm.address_space<cc>>, %out: memref<32x64xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %m = arith.constant 32 : index
    %k = arith.constant 32 : index
    %n = arith.constant 64 : index
    %false = arith.constant false
    %true = arith.constant true
    // SYNC: hivm.hir.mmadL0
    // LOWER: call @mmadl0_a5_half_to_float(
    hivm.hir.mmadL0 ins(%a, %b, %m, %k, %n, %false, %false, %true :
      memref<2x2x16x16xf16, #hivm.address_space<ca>>, memref<2x4x16x16xf16, #hivm.address_space<cb>>, index, index, index, i1, i1, i1)
      outs(%c : memref<4x2x16x16xf32, #hivm.address_space<cc>>)
    // SYNC: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <[[OUT:EVENT_ID[0-9]+]]>]
    // SYNC: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <[[OUT]]>]
    // SYNC: hivm.hir.fixpipe
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
      ins(%c : memref<4x2x16x16xf32, #hivm.address_space<cc>>) outs(%out : memref<32x64xf32, #hivm.address_space<gm>>)
    // SYNC: hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <[[REUSE:EVENT_ID[0-9]+]]>]
    // SYNC: hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <[[REUSE]]>]
    // SYNC: hivm.hir.mmadL0
    // LOWER: call @mmadl0_a5_half_to_float(
    hivm.hir.mmadL0 ins(%a, %b, %m, %k, %n, %false, %false, %false :
      memref<2x2x16x16xf16, #hivm.address_space<ca>>, memref<2x4x16x16xf16, #hivm.address_space<cb>>, index, index, index, i1, i1, i1)
      outs(%c : memref<4x2x16x16xf32, #hivm.address_space<cc>>)
    return
  }
}
