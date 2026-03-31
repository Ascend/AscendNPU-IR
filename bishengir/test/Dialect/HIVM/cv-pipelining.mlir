// RUN: bishengir-opt -cv-pipelining="pipeline-depth=2" -allow-unregistered-dialect %s | FileCheck %s

// CHECK: scf.for
// CHECK: scf.for
// CHECK: hivm.loop_core_type = #hivm.tcore_type<CUBE>
// CHECK: scf.for
// CHECK: hivm.loop_core_type = #hivm.tcore_type<VECTOR>
// CHECK: scf.for
// CHECK: hivm.loop_core_type = #hivm.tcore_type<CUBE>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_pipeline(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<[true]> : vector<1xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %input1 = "some_op"() : () -> memref<16x16xf16>
    %tensor1 = bufferization.to_tensor %input1 : memref<16x16xf16>
    %input2 = "some_op"() : () -> memref<?xf16>
    %initin = memref.reinterpret_cast %input2 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16>
    %offset = "some_op"() : () -> index
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %c0i = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %step = arith.constant 2 : i32
    %bound = "some_op"() : () -> i32
    %cinit = "some_op"() : () -> tensor<16x16xf16>
    %cond = "some_op"() : () -> i1
    %gm = "some_op"() : () -> tensor<16x16xf32>
    %gm2 = "some_op"() : () -> tensor<16x16xf16>
    %vdest = tensor.empty() : tensor<16x16xf16>
    scf.for %i = %c0 to %bound step %step iter_args(%sliding_input = %initin, %inc = %c0i, %itercube = %cinit) -> (memref<16x16xf16>, index, tensor<16x16xf16>) : i32 {
      // Cube ops
      %alloc = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%sliding_input : memref<16x16xf16>) outs(%alloc : memref<16x16xf16>)
      %tensor2 = bufferization.to_tensor %alloc : memref<16x16xf16>
      %dest = tensor.empty() : tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%tensor1, %tensor2, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ub0 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf16>) outs(%ub0 : memref<16x16xf16, #hivm.address_space<ub>>)
      %ub0_cast = memref.memory_space_cast %ub0 : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
      %wst = bufferization.to_tensor %ub0_cast : memref<16x16xf16>
      %newinc = arith.addi %inc, %offset : index
      %next = memref.reinterpret_cast %input2 to offset: [%newinc], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16>

      // Vector ops
      %vdest1 = tensor.empty() : tensor<16x16xf16>
      %exp = hivm.hir.vexp ins(%wst : tensor<16x16xf16>) outs(%vdest1 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %if = scf.if %cond -> tensor<16x16xf16> {
        %newexp = hivm.hir.vexp ins(%exp : tensor<16x16xf16>) outs(%vdest1:tensor<16x16xf16>) -> tensor<16x16xf16>
        scf.yield %newexp : tensor<16x16xf16>
      } else {
        scf.yield %exp : tensor<16x16xf16>
      }
      %ws1 = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
      %ws1_cast = memref.memory_space_cast %ws1 : memref<16x16xf16, #hivm.address_space<cbuf>> to memref<16x16xf16>
      hivm.hir.copy ins(%if:tensor<16x16xf16>) outs(%ws1_cast:memref<16x16xf16>)
      %wso = bufferization.to_tensor %ws1_cast : memref<16x16xf16>

      // Another cube with iter arg/yield
      %t1 = tensor.empty() : tensor<16x16xf16>
      %dot1 = hivm.hir.mmadL1 ins(%itercube, %wso, %true, %c16, %c16, %c16: tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%t1:tensor<16x16xf16>) -> tensor<16x16xf16>
      scf.yield %next, %newinc, %dot1 : memref<16x16xf16>, index, tensor<16x16xf16>
    }
    return
  }
}
