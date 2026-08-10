// RUN: bishengir-opt -cv-pipelining="pipeline-mode=skew" -allow-unregistered-dialect -verify-diagnostics %s | FileCheck %s

// A non-DPS scf.if tensor result cannot currently be expanded for preload.
// Rejecting that work item must restore the checkpoint instead of leaving
// both the checkpoint and the partially transformed loop executable.

// CHECK-LABEL: func.func @preload_revert_unsupported_output
// CHECK-NOT: scope.scope
// CHECK: scf.for
// CHECK-NOT: scope.scope
// CHECK: hivm.hir.mmadL1
// CHECK-NOT: scope.scope
// CHECK-NOT: scf.for
// CHECK-NOT: hivm.hir.mmadL1
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @preload_revert_unsupported_output(
      %a: tensor<16x16xf16>,
      %b: tensor<16x16xf16>,
      %vector_in: tensor<16x16xf32>,
      %cond: i1) attributes {
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        hivm.func_core_type = #hivm.func_core_type<MIX>,
        mix_mode = "mix"
      } {
    %c0 = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %bound = "some_op"() : () -> i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %bound step %step : i32 {
      %vector_init = tensor.empty() : tensor<16x16xf32>
      // expected-warning@+1 {{expected to_tensor for non-tensor-empty output}}
      %selected = scf.if %cond -> tensor<16x16xf32> {
        %exp = hivm.hir.vexp ins(%vector_in : tensor<16x16xf32>) outs(%vector_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        scf.yield %exp : tensor<16x16xf32>
      } else {
        scf.yield %vector_in : tensor<16x16xf32>
      }

      %dot_init = tensor.empty() : tensor<16x16xf32>
      %dot = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %cube_out = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %cube_out_cast = memref.memory_space_cast %cube_out : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
      %cube_out_tensor = bufferization.to_tensor %cube_out_cast restrict writable : memref<16x16xf32>
      %fix = hivm.hir.fixpipe ins(%dot : tensor<16x16xf32>) outs(%cube_out_tensor : tensor<16x16xf32>) -> tensor<16x16xf32>

      %sum_init = tensor.empty() : tensor<16x16xf32>
      %sum = hivm.hir.vadd ins(%selected, %fix : tensor<16x16xf32>, tensor<16x16xf32>) outs(%sum_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      "consume"(%sum) : (tensor<16x16xf32>) -> ()
      scf.yield
    }
    return
  }
}
