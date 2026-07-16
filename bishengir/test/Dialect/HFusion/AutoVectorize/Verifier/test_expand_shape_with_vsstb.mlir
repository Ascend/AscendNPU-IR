// RUN: bishengir-opt %s --split-input-file --hfusion-auto-vectorize-v2 2>/dev/null | FileCheck %s

// Case 1: expand_shape + annotation.mark + 1 vsstb transpose

// CHECK-LABEL: func.func @case1()
// CHECK: %expanded = tensor.expand_shape {{.*}} output_shape [128, 8, 16] : tensor<128x128xf16> into tensor<128x8x16xf16>
// CHECK: annotation.mark %expanded
// CHECK: scf.for
// CHECK-NOT: tensor.expand_shape
// CHECK: vector.transfer_write
// CHECK: {"outlined-loop-target-1"}

module {
  func.func @case1() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %alloc = memref.alloc() : memref<8x128x16xf16>
    %0 = tensor.empty() : tensor<128x128xf16>
    %expanded = tensor.expand_shape %0 [[0], [1, 2]] output_shape [128, 8, 16] : tensor<128x128xf16> into tensor<128x8x16xf16>
    annotation.mark %expanded {tiling_dim_mapping = {"1" = 1 : index}} : tensor<128x8x16xf16>
    %1 = tensor.empty() : tensor<8x128x16xf16>
    %transposed = linalg.transpose ins(%expanded : tensor<128x8x16xf16>) outs(%1 : tensor<8x128x16xf16>) permutation = [1, 0, 2]
    hivm.hir.copy ins(%transposed : tensor<8x128x16xf16>) outs(%alloc : memref<8x128x16xf16>)
    return
  }
}

// -----

// Case 2: expand_shape + sync_block_wait + 2 vsstb transposes

// CHECK-LABEL: func.func @case2()
// CHECK: %expanded = tensor.expand_shape {{.*}} output_shape [128, 8, 16] : tensor<128x128xf16> into tensor<128x8x16xf16>
// CHECK: scf.for
// CHECK-NOT: tensor.expand_shape
// CHECK: vector.transfer_write
// CHECK: {"outlined-loop-target-1"}
// CHECK: hivm.hir.sync_block_wait
// CHECK: scf.for
// CHECK-NOT: tensor.expand_shape
// CHECK: vector.transfer_write
// CHECK: {"outlined-loop-target-2"}

module {
  func.func @case2() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %alloc1 = memref.alloc() : memref<8x128x16xf16>
    %alloc2 = memref.alloc() : memref<8x128x16xf16>
    %0 = tensor.empty() : tensor<128x128xf16>
    %expanded = tensor.expand_shape %0 [[0], [1, 2]] output_shape [128, 8, 16] : tensor<128x128xf16> into tensor<128x8x16xf16>
    %1 = tensor.empty() : tensor<8x128x16xf16>
    %t1 = linalg.transpose ins(%expanded : tensor<128x8x16xf16>) outs(%1 : tensor<8x128x16xf16>) permutation = [1, 0, 2]
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
    %2 = tensor.empty() : tensor<8x128x16xf16>
    %t2 = linalg.transpose ins(%expanded : tensor<128x8x16xf16>) outs(%2 : tensor<8x128x16xf16>) permutation = [1, 0, 2]
    hivm.hir.copy ins(%t1 : tensor<8x128x16xf16>) outs(%alloc1 : memref<8x128x16xf16>)
    hivm.hir.copy ins(%t2 : tensor<8x128x16xf16>) outs(%alloc2 : memref<8x128x16xf16>)
    return
  }
}

// -----

// Case 3: expand_shape + 2 vsstb transposes

// CHECK-LABEL: func.func @case3()
// CHECK: scf.for
// CHECK: %expanded = tensor.expand_shape {{.*}} output_shape [1, 8, 16] : tensor<1x128xf16> into tensor<1x8x16xf16>
// CHECK-COUNT-2: vector.transfer_write
// CHECK: {"outlined-loop-target-1"}

module {
  func.func @case3() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %alloc1 = memref.alloc() : memref<8x128x16xf16>
    %alloc2 = memref.alloc() : memref<8x128x16xf16>
    %0 = tensor.empty() : tensor<128x128xf16>
    %expanded = tensor.expand_shape %0 [[0], [1, 2]] output_shape [128, 8, 16] : tensor<128x128xf16> into tensor<128x8x16xf16>
    %1 = tensor.empty() : tensor<8x128x16xf16>
    %t1 = linalg.transpose ins(%expanded : tensor<128x8x16xf16>) outs(%1 : tensor<8x128x16xf16>) permutation = [1, 0, 2]
    %2 = tensor.empty() : tensor<8x128x16xf16>
    %t2 = linalg.transpose ins(%expanded : tensor<128x8x16xf16>) outs(%2 : tensor<8x128x16xf16>) permutation = [1, 0, 2]
    hivm.hir.copy ins(%t1 : tensor<8x128x16xf16>) outs(%alloc1 : memref<8x128x16xf16>)
    hivm.hir.copy ins(%t2 : tensor<8x128x16xf16>) outs(%alloc2 : memref<8x128x16xf16>)
    return
  }
}

// -----

// Case 4: expand_shape + sync nested in scf.for + vsstb transpose.

// CHECK-LABEL: func.func @case4()
// CHECK: %expanded = tensor.expand_shape {{.*}} output_shape [128, 8, 16] : tensor<128x128xf16> into tensor<128x8x16xf16>
// CHECK: hivm.hir.sync_block_wait
// CHECK: scf.for
// CHECK-NOT: tensor.expand_shape
// CHECK: vector.transfer_write
// CHECK: {"outlined-loop-target-1"}

module {
  func.func @case4() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %alloc = memref.alloc() : memref<8x128x16xf16>
    %0 = tensor.empty() : tensor<128x128xf16>
    %expanded = tensor.expand_shape %0 [[0], [1, 2]] output_shape [128, 8, 16] : tensor<128x128xf16> into tensor<128x8x16xf16>
    scf.for %i = %c0 to %c8 step %c1 {
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
    }
    %1 = tensor.empty() : tensor<8x128x16xf16>
    %transposed = linalg.transpose ins(%expanded : tensor<128x8x16xf16>) outs(%1 : tensor<8x128x16xf16>) permutation = [1, 0, 2]
    hivm.hir.copy ins(%transposed : tensor<8x128x16xf16>) outs(%alloc : memref<8x128x16xf16>)
    return
  }
}

// -----

// Baseline: expand_shape + 1 vsstb transpose

// CHECK-LABEL: func.func @baseline()
// CHECK: scf.for
// CHECK: %expanded = tensor.expand_shape {{.*}} output_shape [1, 8, 16] : tensor<1x128xf16> into tensor<1x8x16xf16>
// CHECK: vector.transfer_write
// CHECK: {"outlined-loop-target-1"}

module {
  func.func @baseline() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %alloc = memref.alloc() : memref<8x128x16xf16>
    %0 = tensor.empty() : tensor<128x128xf16>
    %expanded = tensor.expand_shape %0 [[0], [1, 2]] output_shape [128, 8, 16] : tensor<128x128xf16> into tensor<128x8x16xf16>
    %1 = tensor.empty() : tensor<8x128x16xf16>
    %transposed = linalg.transpose ins(%expanded : tensor<128x8x16xf16>) outs(%1 : tensor<8x128x16xf16>) permutation = [1, 0, 2]
    hivm.hir.copy ins(%transposed : tensor<8x128x16xf16>) outs(%alloc : memref<8x128x16xf16>)
    return
  }
}
