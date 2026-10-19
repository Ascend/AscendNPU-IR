// RUN: bishengir-opt -hivm-graph-sync-solver=solver-version=v1 -split-input-file %s | FileCheck %s
// RUN: bishengir-opt -hivm-graph-sync-solver=solver-version=v2 -split-input-file %s | FileCheck %s

// A VF's AVE memory accesses must produce dependencies at its PIPE_V call.
// Neither function-argument memory-effect attributes nor explicit events are
// supplied. The source alias checks that effects are traced back to VF args.

#b32 = #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func private @vf_load_store(
      %input: memref<64xf32, #hivm.address_space<ub>>,
      %output: memref<64xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %source = memref.subview %input[0] [64] [1]
        : memref<64xf32, #hivm.address_space<ub>> to memref<64xf32, #hivm.address_space<ub>>
    %mask = ave.hir.pge <VL16> : vector<64xi1, #b32>
    %value = ave.hir.vload <NORM> %source[%c0]
        : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32, #b32>
    ave.hir.masked_store <NORM_B32> %output[%c0], %mask, %value
        : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1, #b32>, vector<64xf32, #b32>
    return
  }

  // CHECK-LABEL: func.func @load_store(
  func.func @load_store(
      %inputGM: memref<64xf32, #hivm.address_space<gm>>,
      %outputGM: memref<64xf32, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : i64
    %c256 = arith.constant 256 : i64
    %inputUB = hivm.hir.pointer_cast(%c0)
        : memref<64xf32, #hivm.address_space<ub>>
    %outputUB = hivm.hir.pointer_cast(%c256)
        : memref<64xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.load
    hivm.hir.load ins(%inputGM : memref<64xf32, #hivm.address_space<gm>>)
        outs(%inputUB : memref<64xf32, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    // CHECK: call @vf_load_store
    call @vf_load_store(%inputUB, %outputUB)
        : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
    // CHECK: hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    // CHECK: hivm.hir.store
    hivm.hir.store ins(%outputUB : memref<64xf32, #hivm.address_space<ub>>)
        outs(%outputGM : memref<64xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----

#b32 = #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func private @vf_gather_scatter(
      %input: memref<64xf32, #hivm.address_space<ub>>,
      %output: memref<64xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %source = memref.subview %input[0] [64] [1]
        : memref<64xf32, #hivm.address_space<ub>> to memref<64xf32, #hivm.address_space<ub>>
    %mask = ave.hir.pge <VL16> : vector<64xi1, #b32>
    %indices = arith.constant dense<0> : vector<64xindex>
    %value = ave.hir.vgather %source[%c0] [%indices], %mask
        : memref<64xf32, #hivm.address_space<ub>>, vector<64xindex>, vector<64xi1, #b32> into vector<64xf32, #b32>
    ave.hir.vscatter %output[%c0] [%indices], %mask, %value
        : memref<64xf32, #hivm.address_space<ub>>, vector<64xindex>, vector<64xi1, #b32>, vector<64xf32, #b32>
    return
  }

  // CHECK-LABEL: func.func @gather_scatter(
  func.func @gather_scatter(
      %inputGM: memref<64xf32, #hivm.address_space<gm>>,
      %outputGM: memref<64xf32, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : i64
    %c256 = arith.constant 256 : i64
    %inputUB = hivm.hir.pointer_cast(%c0)
        : memref<64xf32, #hivm.address_space<ub>>
    %outputUB = hivm.hir.pointer_cast(%c256)
        : memref<64xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.load
    hivm.hir.load ins(%inputGM : memref<64xf32, #hivm.address_space<gm>>)
        outs(%inputUB : memref<64xf32, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    // CHECK: call @vf_gather_scatter
    call @vf_gather_scatter(%inputUB, %outputUB)
        : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
    // CHECK: hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    // CHECK: hivm.hir.store
    hivm.hir.store ins(%outputUB : memref<64xf32, #hivm.address_space<ub>>)
        outs(%outputGM : memref<64xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----

// The unaligned store writes through an alias of the VF output argument.

#b32 = #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func private @vf_unaligned_store(
      %input: memref<64xf32, #hivm.address_space<ub>>,
      %output: memref<64xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %source = memref.subview %input[0] [64] [1]
        : memref<64xf32, #hivm.address_space<ub>> to memref<64xf32, #hivm.address_space<ub>>
    %destination = memref.subview %output[0] [64] [1]
        : memref<64xf32, #hivm.address_space<ub>> to memref<64xf32, #hivm.address_space<ub>>
    %mask = ave.hir.pge <VL16> : vector<64xi1, #b32>
    %value = ave.hir.vload <NORM> %source[%c0]
        : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32, #b32>
    ave.hir.unaligned.masked.store %destination[%c0], %mask, %value
        : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1, #b32>, vector<64xf32, #b32>
    return
  }

  // CHECK-LABEL: func.func @unaligned_store(
  func.func @unaligned_store(
      %inputGM: memref<64xf32, #hivm.address_space<gm>>,
      %outputGM: memref<64xf32, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : i64
    %c256 = arith.constant 256 : i64
    %inputUB = hivm.hir.pointer_cast(%c0)
        : memref<64xf32, #hivm.address_space<ub>>
    %outputUB = hivm.hir.pointer_cast(%c256)
        : memref<64xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.load
    hivm.hir.load ins(%inputGM : memref<64xf32, #hivm.address_space<gm>>)
        outs(%inputUB : memref<64xf32, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    // CHECK: call @vf_unaligned_store
    call @vf_unaligned_store(%inputUB, %outputUB)
        : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
    // CHECK: hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    // CHECK: hivm.hir.store
    hivm.hir.store ins(%outputUB : memref<64xf32, #hivm.address_space<ub>>)
        outs(%outputGM : memref<64xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----

// The strided store writes through an alias of the VF output argument.

#b32 = #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func private @vf_strided_store(
      %input: memref<64xf32, #hivm.address_space<ub>>,
      %output: memref<64xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %source = memref.subview %input[0] [64] [1]
        : memref<64xf32, #hivm.address_space<ub>> to memref<64xf32, #hivm.address_space<ub>>
    %destination = memref.subview %output[0] [64] [1]
        : memref<64xf32, #hivm.address_space<ub>> to memref<64xf32, #hivm.address_space<ub>>
    %mask = ave.hir.pge <VL16> : vector<64xi1, #b32>
    %value = ave.hir.vload <NORM> %source[%c0]
        : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32, #b32>
    ave.hir.store_with_stride %destination[%c0], %c1, %mask, %value
        : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1, #b32>, vector<64xf32, #b32>
    return
  }

  // CHECK-LABEL: func.func @strided_store(
  func.func @strided_store(
      %inputGM: memref<64xf32, #hivm.address_space<gm>>,
      %outputGM: memref<64xf32, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : i64
    %c256 = arith.constant 256 : i64
    %inputUB = hivm.hir.pointer_cast(%c0)
        : memref<64xf32, #hivm.address_space<ub>>
    %outputUB = hivm.hir.pointer_cast(%c256)
        : memref<64xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.load
    hivm.hir.load ins(%inputGM : memref<64xf32, #hivm.address_space<gm>>)
        outs(%inputUB : memref<64xf32, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    // CHECK: call @vf_strided_store
    call @vf_strided_store(%inputUB, %outputUB)
        : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
    // CHECK: hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    // CHECK: hivm.hir.store
    hivm.hir.store ins(%outputUB : memref<64xf32, #hivm.address_space<ub>>)
        outs(%outputGM : memref<64xf32, #hivm.address_space<gm>>)
    return
  }
}
