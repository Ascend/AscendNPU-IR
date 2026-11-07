// RUN: bishengir-opt %s --pass-pipeline="builtin.module(hivm-v2-pipeline{target=Ascend950PR_9579 ghost-split-mix=false verify-stage=all})" | FileCheck %s

// Compile a vector prologue followed by matmul through the full v2 pipeline.
// Keep the internal data dependencies; completion at the call boundary belongs
// to the caller, without an unconditional scalar-pipe join on return.
// CHECK-LABEL: func.func @exported_mix_aic
// CHECK-SAME: export_device
// CHECK-NOT: hacc.entry
// CHECK: hivm.hir.mmadL1
// CHECK-NOT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK-NOT: hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 14
// CHECK: return
// CHECK-LABEL: func.func @exported_mix_aiv
// CHECK-SAME: export_device
// CHECK-NOT: hacc.entry
// CHECK: hivm.hir.vexp
// CHECK: hivm.hir.vcast
// CHECK: hivm.hir.copy {{.*}}outs({{.*}}#hivm.address_space<cbuf>
// CHECK-NOT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK-NOT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 14
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @exported(%lhs_mem: memref<16x16xf32>, %rhs_mem: memref<16x16xf16>, %dst: memref<16x16xf32>)
      attributes {export_device, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %lhs = bufferization.to_tensor %lhs_mem restrict : memref<16x16xf32>
    %rhs = bufferization.to_tensor %rhs_mem restrict : memref<16x16xf16>
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %exp_out = tensor.empty() : tensor<16x16xf32>
    %exp = hivm.hir.vexp ins(%lhs : tensor<16x16xf32>) outs(%exp_out : tensor<16x16xf32>) -> tensor<16x16xf32>
    %cast_out = tensor.empty() : tensor<16x16xf16>
    %cast = hivm.hir.vcast ins(%exp : tensor<16x16xf32>) outs(%cast_out : tensor<16x16xf16>) -> tensor<16x16xf16>
    %out = tensor.empty() : tensor<16x16xf32>
    %result = hivm.hir.mmadL1 {already_set_real_mkn}
        ins(%cast, %rhs, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
        outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
    bufferization.materialize_in_destination %result in writable %dst : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
    return
  }
}
