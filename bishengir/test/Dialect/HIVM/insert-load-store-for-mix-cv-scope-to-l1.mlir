// RUN: bishengir-opt %s -split-input-file \
// RUN:   -hivm-insert-load-store-for-mix-cv="target=Ascend910_9589" \
// RUN:   | FileCheck %s

// CHECK-LABEL: @scope_to_mmadL1
// CHECK: %[[SCOPE:.+]] = scope.scope
// Operand 0 (A) goes through expand -> vtranspose -> expand -> cbuf copy:
// CHECK: %[[EX1:.+]] = tensor.expand_shape %[[SCOPE]]
// CHECK: %[[VT:.+]] = hivm.hir.vtranspose ins(%[[EX1]]
// CHECK-SAME: permutation = [1, 0, 2]
// CHECK: %[[EX2:.+]] = tensor.expand_shape %[[VT]]
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<{{.*}}#hivm.address_space<cbuf>>
// CHECK: %[[CAST:.+]] = memref.memory_space_cast %[[ALLOC]]
// CHECK: %[[L1:.+]] = bufferization.to_tensor %[[CAST]]
// CHECK: hivm.hir.copy ins(%[[EX2]]
// CHECK: hivm.hir.mmadL1 {{.*}}ins(%[[L1]]
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @scope_to_mmadL1(%arg0: tensor<16x16xf32>)
      -> tensor<16x16xf32>
      attributes {hacc.entry,
                  hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %init = tensor.empty() : tensor<16x16xf32>
    %scope_result = scope.scope : () -> tensor<16x16xf32> {
      scope.return %arg0 : tensor<16x16xf32>
    }
    %res = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true}
        ins(%scope_result, %arg0, %true, %c16, %c16, %c16
            : tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index)
        outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %res : tensor<16x16xf32>
  }
}
