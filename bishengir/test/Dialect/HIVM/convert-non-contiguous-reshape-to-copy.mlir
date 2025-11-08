// RUN: bishengir-opt -convert-non-contiguous-reshape-to-copy -allow-unregistered-dialect -split-input-file %s | FileCheck %s

// CHECK-LABEL: collapse
// CHECK: %[[ARG0:.*]]: memref<64x32x4xbf16, #hivm.address_space<ub>>)
func.func @collapse(%src : memref<64x32x4xbf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<64x32x4xbf16, #hivm.address_space<ub>>
  // CHECK: hivm.hir.copy ins(%[[ARG0]] : memref<64x32x4xbf16, #hivm.address_space<ub>>) outs(%[[ALLOC]] : memref<64x32x4xbf16, #hivm.address_space<ub>>)
  // CHECK-SAME:          collapse_reassociation = {{\[}}[0, 1, 2]]
  // CHECK: memref.collapse_shape %[[ALLOC]]
  %collapse_shape = memref.collapse_shape %src [[0], [1, 2]] : memref<64x32x4xbf16, #hivm.address_space<ub>> into memref<64x128xbf16, #hivm.address_space<ub>>
  annotation.mark %collapse_shape {maybeUnCollapsibleReshape} : memref<64x128xbf16, #hivm.address_space<ub>>
  "some_use"(%collapse_shape) : (memref<64x128xbf16, #hivm.address_space<ub>>) -> ()
}

// -----

// CHECK-LABEL: nop
func.func @nop(%src : memref<64x32x4xbf16, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  // CHECK-NOT: hivm.hir.copy
  %collapse_shape = memref.collapse_shape %src [[0], [1, 2]] : memref<64x32x4xbf16, #hivm.address_space<gm>> into memref<64x128xbf16, #hivm.address_space<gm>>
  annotation.mark %collapse_shape {maybeUnCollapsibleReshape} : memref<64x128xbf16, #hivm.address_space<gm>>
  "some_use"(%collapse_shape) : (memref<64x128xbf16, #hivm.address_space<gm>>) -> ()
}
