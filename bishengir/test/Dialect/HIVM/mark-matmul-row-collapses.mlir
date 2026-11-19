// RUN: bishengir-opt %s -hivm-mark-matmul-row-collapses | FileCheck %s
// RUN: bishengir-opt %s -hivm-mark-matmul-row-collapses -hivm-mark-matmul-row-collapses | FileCheck %s
// RUN: sed 's/Ascend910_9382/Ascend950PR_9579/g' %s | bishengir-opt -hivm-mark-matmul-row-collapses | FileCheck %s --check-prefix=SKIP
// RUN: sed 's/ attributes {hacc.target = #hacc.target<"Ascend910_9382">}//' %s | bishengir-opt -hivm-mark-matmul-row-collapses | FileCheck %s --check-prefix=SKIP

module attributes {hacc.target = #hacc.target<"Ascend910_9382">} {
  // Both MMAD operands can merge separate row groups, including transposed B.
  // CHECK-LABEL: func.func @row_merges
  // CHECK: %[[A:.*]] = tensor.collapse_shape
  // CHECK-NEXT: annotation.mark %[[A]] {maybeUnCollapsibleReshape}
  // CHECK-NOT: annotation.mark
  // CHECK: %[[B:.*]] = tensor.collapse_shape
  // CHECK-NEXT: annotation.mark %[[B]] {maybeUnCollapsibleReshape}
  // CHECK-NOT: annotation.mark
  // CHECK: return
  // SKIP-LABEL: func.func @row_merges
  // SKIP-NOT: annotation.mark
  // SKIP: return
  func.func @row_merges(%lhs: tensor<2x32x64xf16>,
                        %rhs: tensor<2x32x64xf16>) -> tensor<64x64xf32> {
    %true = arith.constant true
    %c64 = arith.constant 64 : index
    %a = tensor.collapse_shape %lhs [[0, 1], [2]]
        : tensor<2x32x64xf16> into tensor<64x64xf16>
    %b = tensor.collapse_shape %rhs [[0, 1], [2]]
        : tensor<2x32x64xf16> into tensor<64x64xf16>
    %empty = tensor.empty() : tensor<64x64xf32>
    %result = hivm.hir.mmadL1 {b_transpose} ins(%a, %b, %true, %c64, %c64, %c64
        : tensor<64x64xf16>, tensor<64x64xf16>, i1, index, index, index)
        outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %result : tensor<64x64xf32>
  }

  // Preserve explicit hints and avoid duplicates for a shared A/B operand.
  // CHECK-LABEL: func.func @existing_hint
  // CHECK: annotation.mark %{{.*}} {maybeUnCollapsibleReshape}
  // CHECK-NOT: annotation.mark
  // CHECK: return
  func.func @existing_hint(%src: tensor<2x32x64xf16>) -> tensor<64x64xf32> {
    %true = arith.constant true
    %c64 = arith.constant 64 : index
    %a = tensor.collapse_shape %src [[0, 1], [2]]
        : tensor<2x32x64xf16> into tensor<64x64xf16>
    annotation.mark %a {maybeUnCollapsibleReshape} : tensor<64x64xf16>
    %empty = tensor.empty() : tensor<64x64xf32>
    %result = hivm.hir.mmadL1 ins(%a, %a, %true, %c64, %c64, %c64
        : tensor<64x64xf16>, tensor<64x64xf16>, i1, index, index, index)
        outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %result : tensor<64x64xf32>
  }

  // Removing a unit dimension does not merge independently fractalized rows.
  // CHECK-LABEL: func.func @unit_dimensions
  // CHECK-NOT: annotation.mark
  // CHECK: return
  func.func @unit_dimensions(%lhs: tensor<1x64x64xf16>,
                             %rhs: tensor<64x1x64xf16>) -> tensor<64x64xf32> {
    %true = arith.constant true
    %c64 = arith.constant 64 : index
    %a = tensor.collapse_shape %lhs [[0, 1], [2]]
        : tensor<1x64x64xf16> into tensor<64x64xf16>
    %b = tensor.collapse_shape %rhs [[0, 1], [2]]
        : tensor<64x1x64xf16> into tensor<64x64xf16>
    %empty = tensor.empty() : tensor<64x64xf32>
    %result = hivm.hir.mmadL1 ins(%a, %b, %true, %c64, %c64, %c64
        : tensor<64x64xf16>, tensor<64x64xf16>, i1, index, index, index)
        outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %result : tensor<64x64xf32>
  }

  // Dynamic rows and columns retain their previous lowering.
  // CHECK-LABEL: func.func @dynamic_dimensions
  // CHECK-NOT: annotation.mark
  // CHECK: return
  func.func @dynamic_dimensions(%lhs: tensor<?x?x64xf16>,
                                %rhs: tensor<2x32x?xf16>, %m: index, %n: index)
      -> tensor<?x?xf32> {
    %true = arith.constant true
    %c64 = arith.constant 64 : index
    %a = tensor.collapse_shape %lhs [[0, 1], [2]]
        : tensor<?x?x64xf16> into tensor<?x64xf16>
    %b = tensor.collapse_shape %rhs [[0, 1], [2]]
        : tensor<2x32x?xf16> into tensor<64x?xf16>
    %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %result = hivm.hir.mmadL1 ins(%a, %b, %true, %m, %c64, %n
        : tensor<?x64xf16>, tensor<64x?xf16>, i1, index, index, index)
        outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %result : tensor<?x?xf32>
  }

  // A static collapse does not make a dynamically sliced MMAD operand static.
  // CHECK-LABEL: func.func @dynamic_slice
  // CHECK-NOT: annotation.mark
  // CHECK: return
  func.func @dynamic_slice(%src: tensor<2x32x64xf16>,
                           %rhs: tensor<64x64xf16>, %m: index)
      -> tensor<?x64xf32> {
    %true = arith.constant true
    %c64 = arith.constant 64 : index
    %collapsed = tensor.collapse_shape %src [[0, 1], [2]]
        : tensor<2x32x64xf16> into tensor<64x64xf16>
    %a = tensor.extract_slice %collapsed[0, 0] [%m, 64] [1, 1]
        : tensor<64x64xf16> to tensor<?x64xf16>
    %empty = tensor.empty(%m) : tensor<?x64xf32>
    %result = hivm.hir.mmadL1 ins(%a, %rhs, %true, %m, %c64, %c64
        : tensor<?x64xf16>, tensor<64x64xf16>, i1, index, index, index)
        outs(%empty : tensor<?x64xf32>) -> tensor<?x64xf32>
    return %result : tensor<?x64xf32>
  }

  // This pass only annotates leading row merges feeding MMAD.
  // CHECK-LABEL: func.func @column_merge
  // CHECK-NOT: annotation.mark
  // CHECK: return
  func.func @column_merge(%lhs: tensor<64x2x32xf16>,
                          %rhs: tensor<64x64xf16>) -> tensor<64x64xf32> {
    %true = arith.constant true
    %c64 = arith.constant 64 : index
    %a = tensor.collapse_shape %lhs [[0], [1, 2]]
        : tensor<64x2x32xf16> into tensor<64x64xf16>
    %empty = tensor.empty() : tensor<64x64xf32>
    %result = hivm.hir.mmadL1 ins(%a, %rhs, %true, %c64, %c64, %c64
        : tensor<64x64xf16>, tensor<64x64xf16>, i1, index, index, index)
        outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %result : tensor<64x64xf32>
  }

  // CHECK-LABEL: func.func @plain_nd_collapse
  // CHECK-NOT: annotation.mark
  // CHECK: return
  func.func @plain_nd_collapse(%src: tensor<2x32x64xf16>) -> tensor<64x64xf16> {
    %result = tensor.collapse_shape %src [[0, 1], [2]]
        : tensor<2x32x64xf16> into tensor<64x64xf16>
    return %result : tensor<64x64xf16>
  }
}
