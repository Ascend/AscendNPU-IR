// RUN: bishengir-opt %s -split-input-file -verify-diagnostics

func.func @invalid_signal_element_type(%a: memref<16x16xf16>, %b: memref<16x16xf16>,
    %c: memref<16x16xf32>, %signal: memref<i8>, %init: i1, %n: index) {
  // expected-error@+1 {{0D memref of 32-bit signless integer values}}
  hivm.hir.mmadL1
      ins(%a, %b, %init, %n, %n, %n :
          memref<16x16xf16>, memref<16x16xf16>, i1, index, index, index)
      outs(%c : memref<16x16xf32>)
      ping_pong_signal(%signal : memref<i8>)
  return
}

// -----

func.func @invalid_signal_shape(%a: memref<16x16xf16>, %b: memref<16x16xf16>,
    %c: memref<16x16xf32>, %signal: memref<1xi32>, %init: i1, %n: index) {
  // expected-error@+1 {{0D memref of 32-bit signless integer values}}
  hivm.hir.mmadL1
      ins(%a, %b, %init, %n, %n, %n :
          memref<16x16xf16>, memref<16x16xf16>, i1, index, index, index)
      outs(%c : memref<16x16xf32>)
      ping_pong_signal(%signal : memref<1xi32>)
  return
}
