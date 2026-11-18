// RUN: bishengir-opt %s --hivm-insert-load-store-for-mix-cv="enable-legacy=false" | FileCheck %s

// A vector prologue feeding matmul needs UB -> L1 conversion in an export too.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @exported
  // CHECK-SAME: export_device
  // CHECK-NOT: hacc.entry
  // CHECK: hivm.hir.vexp
  // CHECK: hivm.hir.vcast
  // CHECK: hivm.hir.vtranspose
  // CHECK: %[[L1:.*]] = tensor.empty() {{.*}}hivm.address_space = #hivm.address_space<cbuf>
  // CHECK: %[[COPY:.*]] = hivm.hir.copy {{.*}}outs(%[[L1]]
  // CHECK: hivm.hir.mmadL1 {{.*}}ins(%[[COPY]],
  func.func @exported(%lhs: tensor<16x16xf32>, %rhs: tensor<16x16xf16>) -> tensor<16x16xf32>
      attributes {export_device, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %exp_out = tensor.empty() : tensor<16x16xf32>
    %exp = hivm.hir.vexp ins(%lhs : tensor<16x16xf32>) outs(%exp_out : tensor<16x16xf32>) -> tensor<16x16xf32>
    %cast_out = tensor.empty() : tensor<16x16xf16>
    %cast = hivm.hir.vcast ins(%exp : tensor<16x16xf32>) outs(%cast_out : tensor<16x16xf16>) -> tensor<16x16xf16>
    %out = tensor.empty() : tensor<16x16xf32>
    %result = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true}
        ins(%cast, %rhs, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
        outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %result : tensor<16x16xf32>
  }

  // Helpers must still be left to their enclosing compilation root.
  // CHECK-LABEL: func.func private @helper
  // CHECK-NOT: hivm.hir.copy
  // CHECK-NOT: hivm.hir.vtranspose
  // CHECK: hivm.hir.mmadL1
  // CHECK-NEXT: return
  func.func private @helper(%lhs: tensor<16x16xf16>, %rhs: tensor<16x16xf16>) -> tensor<16x16xf32>
      attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %exp_out = tensor.empty() : tensor<16x16xf16>
    %exp = hivm.hir.vexp ins(%lhs : tensor<16x16xf16>) outs(%exp_out : tensor<16x16xf16>) -> tensor<16x16xf16>
    %out = tensor.empty() : tensor<16x16xf32>
    %result = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true}
        ins(%exp, %rhs, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
        outs(%out : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %result : tensor<16x16xf32>
  }
}
