// RUN: bishengir-opt %s -hivm-infer-data-layout -canonicalize | FileCheck %s --check-prefixes=CHECK,LOOP
// RUN: bishengir-opt %s -hivm-infer-data-layout='batch-matmul=true' -canonicalize | FileCheck %s --check-prefixes=CHECK,NATIVE

// Initialize the entire B tile before copying the valid batches and columns.
// CHECK-LABEL: func.func @masked_batch_operand(
// CHECK-SAME: %[[VALID_BATCHES:arg[0-9]+]]: index, %{{arg[0-9]+}}: index, %[[NEEDS_PAD:arg[0-9]+]]: i1)
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f16
// CHECK: %[[TILE:.*]] = memref.alloc() : memref<4x2x1x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: scf.if %[[NEEDS_PAD]] {
// CHECK-NEXT: hivm.hir.vbrc ins(%[[ZERO]] : f16) outs(%[[TILE]]
// CHECK-NEXT: }
// LOOP-NEXT: scf.for {{.*}} to %[[VALID_BATCHES]]
// NATIVE-NOT: scf.for
// CHECK-NOT: hivm.hir.vbrc
// CHECK: hivm.hir.nd2nz
// CHECK-NOT: hivm.hir.vbrc
// CHECK: hivm.hir.batchMmadL1
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @masked_batch_operand(
      %a: memref<4x16x16xf16, #hivm.address_space<gm>>,
      %b: memref<4x16x32xf16, #hivm.address_space<gm>>,
      %validBatches: index, %validCols: index, %needsPad: i1)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                  hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %zero = arith.constant 0.0 : f16
    %true = arith.constant true
    %l1A = memref.alloc() : memref<4x16x16xf16, #hivm.address_space<cbuf>>
    %l1B = memref.alloc() : memref<4x16x32xf16, #hivm.address_space<cbuf>>
    %l0C = memref.alloc() : memref<4x16x32xf32, #hivm.address_space<cc>>
    hivm.hir.load ins(%a : memref<4x16x16xf16, #hivm.address_space<gm>>)
                  outs(%l1A : memref<4x16x16xf16, #hivm.address_space<cbuf>>)
    %src = memref.subview %b[0, 0, 0] [%validBatches, 16, %validCols] [1, 1, 1] :
        memref<4x16x32xf16, #hivm.address_space<gm>> to
        memref<?x16x?xf16, strided<[512, 32, 1]>, #hivm.address_space<gm>>
    %dst = memref.subview %l1B[0, 0, 0] [%validBatches, 16, %validCols] [1, 1, 1] :
        memref<4x16x32xf16, #hivm.address_space<cbuf>> to
        memref<?x16x?xf16, strided<[512, 32, 1]>, #hivm.address_space<cbuf>>
    hivm.hir.load ins(%src : memref<?x16x?xf16, strided<[512, 32, 1]>, #hivm.address_space<gm>>)
        outs(%dst : memref<?x16x?xf16, strided<[512, 32, 1]>, #hivm.address_space<cbuf>>)
        pad_mode = <PadValue> pad_value = %zero : f16
        init_out_buffer = true init_condition = %needsPad : i1
    hivm.hir.batchMmadL1 ins(%l1A, %l1B, %true, %c16, %c16, %c32 :
        memref<4x16x16xf16, #hivm.address_space<cbuf>>,
        memref<4x16x32xf16, #hivm.address_space<cbuf>>, i1, index, index, index)
        outs(%l0C : memref<4x16x32xf32, #hivm.address_space<cc>>)
    return
  }
}
