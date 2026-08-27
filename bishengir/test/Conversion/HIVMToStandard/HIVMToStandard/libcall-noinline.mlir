// RUN: bishengir-opt %s -convert-hivm-to-std="mark-libcall-noinline=true" -split-input-file | FileCheck %s --check-prefix=ENABLED
// RUN: bishengir-opt %s -convert-hivm-to-std="mark-libcall-noinline=false" -split-input-file | FileCheck %s --check-prefix=DISABLED
// RUN: bishengir-opt %s -convert-hivm-to-std -split-input-file | FileCheck %s --check-prefix=DEFAULT

module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @vector_libcall() {
    %lhs = memref.alloc() : memref<16xf16>
    %rhs = memref.alloc() : memref<16xf16>
    %dst = memref.alloc() : memref<16xf16>
    hivm.hir.vadd ins(%lhs, %rhs : memref<16xf16>, memref<16xf16>)
                  outs(%dst : memref<16xf16>)
    return
  }
}

// ENABLED-LABEL: func.func private @vadd_1d_half
// ENABLED-SAME:  hacc.noinline
// DISABLED-LABEL: func.func private @vadd_1d_half
// DISABLED-SAME:  hacc.always_inline
// DEFAULT-LABEL: func.func private @vadd_1d_half
// DEFAULT-SAME:  hacc.noinline

// -----

module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @cube_libcall() {
    %lhs = memref.alloc() : memref<256x128xf16>
    %rhs = memref.alloc() : memref<128x256xf16>
    %dst = memref.alloc() : memref<256x256xf32>
    %init = arith.constant 1 : i1
    %m = arith.constant 256 : index
    %k = arith.constant 128 : index
    %n = arith.constant 256 : index
    hivm.hir.mmadL1 ins(%lhs, %rhs, %init, %m, %k, %n :
                          memref<256x128xf16>, memref<128x256xf16>, i1,
                          index, index, index)
                    outs(%dst : memref<256x256xf32>)
    return
  }
}

// ENABLED-LABEL: func.func private @mma_tile_half_to_float
// ENABLED-SAME:  hacc.always_inline
// DISABLED-LABEL: func.func private @mma_tile_half_to_float
// DISABLED-SAME:  hacc.always_inline
// DEFAULT-LABEL: func.func private @mma_tile_half_to_float
// DEFAULT-SAME:  hacc.always_inline
