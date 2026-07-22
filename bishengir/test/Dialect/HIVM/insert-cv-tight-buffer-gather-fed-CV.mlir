// RUN: bishengir-opt --hivm-insert-cv-tight-coupled-buffer -split-input-file %s | FileCheck %s

// -----
// Operand A (operand index 0) -> zN layout, no transpose flag.
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  // CHECK-LABEL: func.func @gather_zn_aiv
  // CHECK-SAME:  %[[BASE:[A-Za-z0-9_]+]]: memref<?xf16>
  // CHECK-SAME:  %[[IDX:[A-Za-z0-9_]+]]: tensor<32x64xi32>
  // CHECK-SAME:  %[[B:[A-Za-z0-9_]+]]: tensor<64x64xf16>
  func.func @gather_zn_aiv(%base : memref<?xf16>,
                           %idx : tensor<32x64xi32>,
                           %b : tensor<64x64xf16>)
      -> tensor<32x64xf32> attributes {hacc.entry, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %true = arith.constant true
    %empty = tensor.empty() : tensor<32x64xf16>

    // CHECK:      %[[GATHER:.*]] = hivm.hir.gather_load
    // CHECK-SAME:   ins(%[[BASE]] : memref<?xf16>, %[[IDX]] : tensor<32x64xi32>
    // CHECK-SAME:   hivm.fractal_layout = "zN"
    // CHECK-SAME:   -> tensor<32x64xf16>
    %g = hivm.hir.gather_load ins(%base : memref<?xf16>, %idx : tensor<32x64xi32>, %c0 : i32)
                              outs(%empty : tensor<32x64xf16>)
                              {cache = #hivm.cache_modifier<none>,
                               evict = #hivm.eviction_policy<EvictLast>,
                               isVolatile = false}
                              -> tensor<32x64xf16>

    // CHECK-NOT:  tensor.expand_shape
    // CHECK-NOT:  hivm.hir.vtranspose
    // CHECK:      %[[ALLOC:.*]] = memref.alloc()
    // CHECK-SAME:   {hivm.fractal_layout = "zN"}
    // CHECK-SAME:   memref<32x64xf16, #hivm.address_space<cbuf>>
    // CHECK:      %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC]]
    // CHECK-SAME:   memref<32x64xf16, #hivm.address_space<cbuf>> to memref<32x64xf16>
    // CHECK:      %[[L1:.*]] = bufferization.to_tensor %[[CAST]] restrict writable
    // CHECK:      hivm.hir.copy ins(%[[GATHER]] : tensor<32x64xf16>) outs(%[[CAST]] : memref<32x64xf16>)

    // CHECK:      hivm.hir.mmadL1 ins(%[[L1]], %[[B]],
    // CHECK-NOT:    b_transpose
    %mm_dst = tensor.empty() : tensor<32x64xf32>
    %mm = hivm.hir.mmadL1 ins(%g, %b, %true, %c32, %c64, %c64
                            : tensor<32x64xf16>, tensor<64x64xf16>, i1, index, index, index)
                          outs(%mm_dst : tensor<32x64xf32>) -> tensor<32x64xf32>
    return %mm : tensor<32x64xf32>
  }
}

// -----
// Operand B (operand index 1) -> nZ layout, b_transpose set on mmadL1.
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  // CHECK-LABEL: func.func @gather_nz_aiv
  // CHECK-SAME:  %[[A:[A-Za-z0-9_]+]]: tensor<32x64xf16>
  // CHECK-SAME:  %[[BASE:[A-Za-z0-9_]+]]: memref<?xf16>
  // CHECK-SAME:  %[[IDX:[A-Za-z0-9_]+]]: tensor<64x64xi32>
  func.func @gather_nz_aiv(%a : tensor<32x64xf16>,
                           %base : memref<?xf16>,
                           %idx : tensor<64x64xi32>)
      -> tensor<32x64xf32> attributes {hacc.entry, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %true = arith.constant true
    %empty = tensor.empty() : tensor<64x64xf16>

    // CHECK:      %[[GATHER:.*]] = hivm.hir.gather_load
    // CHECK-SAME:   ins(%[[BASE]] : memref<?xf16>, %[[IDX]] : tensor<64x64xi32>
    // CHECK-SAME:   hivm.fractal_layout = "nZ"
    // CHECK-SAME:   -> tensor<64x64xf16>
    %g = hivm.hir.gather_load ins(%base : memref<?xf16>, %idx : tensor<64x64xi32>, %c0 : i32)
                              outs(%empty : tensor<64x64xf16>)
                              {cache = #hivm.cache_modifier<none>,
                               evict = #hivm.eviction_policy<EvictLast>,
                               isVolatile = false}
                              -> tensor<64x64xf16>

    // CHECK-NOT:  tensor.expand_shape
    // CHECK-NOT:  hivm.hir.vtranspose
    // CHECK:      %[[ALLOC:.*]] = memref.alloc()
    // CHECK-SAME:   {hivm.fractal_layout = "nZ"}
    // CHECK-SAME:   memref<64x64xf16, #hivm.address_space<cbuf>>
    // CHECK:      %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC]]
    // CHECK:      %[[L1:.*]] = bufferization.to_tensor %[[CAST]] restrict writable
    // CHECK:      hivm.hir.copy ins(%[[GATHER]] : tensor<64x64xf16>) outs(%[[CAST]] : memref<64x64xf16>)

    // CHECK:      hivm.hir.mmadL1 {b_transpose}
    // CHECK-SAME:   ins(%[[A]], %[[L1]],
    %mm_dst = tensor.empty() : tensor<32x64xf32>
    %mm = hivm.hir.mmadL1 ins(%a, %g, %true, %c32, %c64, %c64
                            : tensor<32x64xf16>, tensor<64x64xf16>, i1, index, index, index)
                          outs(%mm_dst : tensor<32x64xf32>) -> tensor<32x64xf32>
    return %mm : tensor<32x64xf32>
  }
}