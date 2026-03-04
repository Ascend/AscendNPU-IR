// RUN: bishengir-opt -memref-remove-redundant-copy -split-input-file -allow-unregistered-dialect %s | FileCheck %s

// -----
// CHECK-LABEL: func @remove_copy_test_0
// CHECK-SAME: (%[[ARG0:.*]]: memref<2xf32>, %[[ARG1:.*]]: memref<2xf32>)
// CHECK-NEXT: memref.copy %[[ARG0]], %[[ARG1]] : memref<2xf32> to memref<2xf32>

module {
  func.func @remove_copy_test_0(%arg0: memref<2xf32>, %arg1: memref<2xf32>) {
    %src = memref.alloc() : memref<2xf32>
    memref.copy %arg0, %src : memref<2xf32> to memref<2xf32>
    memref.copy %src, %arg1 : memref<2xf32> to memref<2xf32>
    return
  }
}

// -----
// CHECK-LABEL: func @remove_copy_test_1
// CHECK-SAME: (%[[ARG0:.*]]: memref<2xf32>)
// CHECK-NEXT: %[[DST:.*]] = memref.alloca() : memref<2xf32>
// CHECK-NEXT: hivm.hir.vadd
// CHECK-NEXT: memref.copy %[[ARG0]], %[[DST]] : memref<2xf32> to memref<2xf32>

module {
  func.func @remove_copy_test_1(%arg0: memref<2xf32>) {
    %src = memref.alloca() : memref<2xf32>
    memref.copy %arg0, %src : memref<2xf32> to memref<2xf32>

    %dst = memref.alloca() : memref<2xf32>
    hivm.hir.vadd ins(%src, %src: memref<2xf32>, memref<2xf32>)
              outs(%dst : memref<2xf32>)
    memref.copy %src, %dst : memref<2xf32> to memref<2xf32>
    return
  }
}

// -----
// CHECK-LABEL: func @remove_copy_test_2
// CHECK-SAME: (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: memref<2xf32>, %[[ARG4:.*]]: memref<2xf32>)
// CHECK-NEXT: %[[TMP1:.*]] = scf.for %[[IV:.*]] = %[[ARG0]] to %[[ARG1]] step %[[ARG2]] iter_args(%[[ARG6:.*]] = %[[ARG3]])
// CHECK-NEXT: %[[TMP3:.*]] = memref.alloc()
// CHECK-NEXT: scf.yield %[[TMP3]]
// CHECK-NEXT: }
// CHECK-NEXT: memref.copy %[[TMP1]], %[[ARG4]] : memref<2xf32> to memref<2xf32>
// CHECK-NEXT: memref.dealloc %[[TMP1]]
module {
  func.func @remove_copy_test_2(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<2xf32>, %arg4: memref<2xf32>) {
    %1 = memref.alloc() : memref<2xf32>
    memref.copy %arg3, %1 : memref<2xf32> to memref<2xf32>
    %2 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %1) -> (memref<2xf32>) {
      %4 = memref.alloc() : memref<2xf32>
      %5 = memref.alloc() : memref<2xf32>
      memref.copy %4, %5 : memref<2xf32> to memref<2xf32>
      scf.yield %5 : memref<2xf32>
    }
    memref.copy %2, %arg4 : memref<2xf32> to memref<2xf32>
    memref.dealloc %2 : memref<2xf32>
    return
  }
}

// -----
module {
  func.func @unremove_copy_test_0() {
    // CHECK: %[[SRC:.*]] = memref.alloc() : memref<5xf32>
    %src = memref.alloc() : memref<5xf32>
    // CHECK: %[[DEST:.*]] = memref.alloc() : memref<5xf32>
    %dst = memref.alloc(): memref<5xf32>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    //CHECK: scf.for %arg0 = %c0 to %c8 step %c1 iter_args(%arg1 = %[[DEST]]) -> (memref<5xf32>) {
    %0 = scf.for %arg0 = %c0 to %c8 step %c1 iter_args(%arg1 = %dst) -> (memref<5xf32>) {
      //CHECK memref.copy
      memref.copy %src, %arg1 : memref<5xf32> to memref<5xf32>
      scf.yield %arg1 : memref<5xf32>
    }
    return
  }
}


// -----
// CHECK-LABEL: func @unremove_copy_test_1
// CHECK-NEXT: %[[SRC:.*]] = memref.alloca() : memref<5xf32>
// CHECK-NEXT: %[[DEST:.*]] = memref.alloca() : memref<5xf32>
// CHECK-NEXT: hivm.hir.vadd ins(%[[SRC]], %[[SRC]] : memref<5xf32>, memref<5xf32>) outs(%[[DEST]] : memref<5xf32>)
// CHECK-NEXT: memref.copy %[[SRC]], %[[DEST]] : memref<5xf32> to memref<5xf32>
// CHECK-NEXT: memref.dealloc %[[SRC]] : memref<5xf32>
// CHECK-NEXT: return %[[DEST]] : memref<5xf32>

module {
  func.func @unremove_copy_test_1() -> memref<5xf32> {
    %src = memref.alloca() : memref<5xf32>
    %dest = memref.alloca() : memref<5xf32>
    hivm.hir.vadd ins(%src, %src: memref<5xf32>, memref<5xf32>)
                  outs(%dest : memref<5xf32>)
    memref.copy %src, %dest : memref<5xf32> to memref<5xf32>
    memref.dealloc %src : memref<5xf32>
    return %dest : memref<5xf32>
  }
}

// -----
// CHECK-LABEL: func @unremove_copy_test_2
module {
  func.func @unremove_copy_test_2() {
    %c0 = arith.constant 0 : i32
    %c29 = arith.constant 29 : index
    %c128 = arith.constant 128 : index
    %c768 = arith.constant 768 : i32
    %c256 = arith.constant 256 : i32
    %true = arith.constant true
    %c256_index = arith.constant 256 : index
    %alloc = memref.alloc() : memref<29x768xf32>
    %alloc_0 = memref.alloc() : memref<29x128xf16>
    %alloc_1 = memref.alloc() : memref<29x256xf32>
    %0 = scf.for %arg0 = %c0 to %c768 step %c256 iter_args(%arg1 = %alloc) -> (memref<29x768xf32>)  : i32 {
      %alloc_2 = memref.alloc() : memref<128x256xf16>
      %alloc_3 = memref.alloc() : memref<1x256xf32>
      %1 = arith.index_cast %arg0 : i32 to index
      // CHECK: outs(%subview
      // CHECK-NOT: memref.copy
      hivm.hir.mmadL1 ins(%alloc_0, %alloc_2, %true, %c29, %c128, %c256_index, %alloc_3 : memref<29x128xf16>, memref<128x256xf16>, i1, index, index, index, memref<1x256xf32>) outs(%alloc_1 : memref<29x256xf32>)
      %subview = memref.subview %arg1[0, %1] [29, 256] [1, 1] : memref<29x768xf32> to memref<29x256xf32, strided<[768, 1], offset: ?>>
      memref.copy %alloc_1, %subview : memref<29x256xf32> to memref<29x256xf32, strided<[768, 1], offset: ?>>
      scf.yield %arg1 : memref<29x768xf32>
    }
    return
  }
}

// -----
// CHECK-LABEL: func @unremove_copy_test_3
module {
  func.func @unremove_copy_test_3() {
    %c0_i32 = arith.constant 0 : i32
    %c29 = arith.constant 29 : index
    %c128 = arith.constant 128 : index
    %c768_i32 = arith.constant 768 : i32
    %c256_i32 = arith.constant 256 : i32
    %true = arith.constant true
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<29x768xf32>
    %alloc_0 = memref.alloc() : memref<29x128xf16>
    %alloc_1 = memref.alloc() : memref<29x256xf32>
    %0 = scf.for %arg0 = %c0_i32 to %c768_i32 step %c256_i32 iter_args(%arg1 = %alloc) -> (memref<29x768xf32>)  : i32 {
      %alloc_2 = memref.alloc() : memref<128x256xf16>
      %alloc_3 = memref.alloc() : memref<1x256xf32>
      scf.for %arg2 = %arg0 to %c768_i32 step %c256_i32  : i32 {
        %1 = arith.index_cast %arg2 : i32 to index
        // CHECK: outs(%subview
        // CHECK-NOT: memref.copy
        hivm.hir.mmadL1 ins(%alloc_0, %alloc_2, %true, %c29, %c128, %c256, %alloc_3 : memref<29x128xf16>, memref<128x256xf16>, i1, index, index, index, memref<1x256xf32>) outs(%alloc_1 : memref<29x256xf32>)
        %subview = memref.subview %arg1[0, %1] [29, 256] [1, 1] : memref<29x768xf32> to memref<29x256xf32, strided<[768, 1], offset: ?>>
        memref.copy %alloc_1, %subview : memref<29x256xf32> to memref<29x256xf32, strided<[768, 1], offset: ?>>
      }
      scf.yield %arg1 : memref<29x768xf32>
    }
    return
  }
}

// -----
// CHECK-LABEL: func @unremove_copy_test_4
module {
  func.func @unremove_copy_test_4() {
    %c0_i32 = arith.constant 0 : i32
    %c29 = arith.constant 29 : index
    %c128 = arith.constant 128 : index
    %c768_i32 = arith.constant 768 : i32
    %c256_i32 = arith.constant 256 : i32
    %true = arith.constant true
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<29x768xf32>
    %alloc_0 = memref.alloc() : memref<29x128xf16>
    %alloc_1 = memref.alloc() : memref<29x256xf32>
    %0 = scf.for %arg0 = %c0_i32 to %c768_i32 step %c256_i32 iter_args(%arg1 = %alloc) -> (memref<29x768xf32>)  : i32 {
      %alloc_2 = memref.alloc() : memref<128x256xf16>
      %alloc_3 = memref.alloc() : memref<1x256xf32>
      %1 = arith.index_cast %arg0 : i32 to index
      %subview = memref.subview %arg1[0, %1] [29, 256] [1, 1] : memref<29x768xf32> to memref<29x256xf32, strided<[768, 1], offset: ?>>
      scf.for %arg2 = %arg0 to %c768_i32 step %c256_i32  : i32 {
        // CHECK: outs(%subview
        // CHECK-NOT: memref.copy
        // CHECK-NOT: memref.subview
        hivm.hir.mmadL1 ins(%alloc_0, %alloc_2, %true, %c29, %c128, %c256, %alloc_3 : memref<29x128xf16>, memref<128x256xf16>, i1, index, index, index, memref<1x256xf32>) outs(%alloc_1 : memref<29x256xf32>)
        memref.copy %alloc_1, %subview : memref<29x256xf32> to memref<29x256xf32, strided<[768, 1], offset: ?>>
      }
      scf.yield %arg1 : memref<29x768xf32>
    }
    return
  }
}
