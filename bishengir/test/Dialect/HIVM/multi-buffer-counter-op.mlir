// RUN: bishengir-opt %s -hivm-enable-multi-buffer -split-input-file | FileCheck %s --check-prefix=ENABLE
// RUN: bishengir-opt %s -hivm-lower-multi-buffer-counter -split-input-file | FileCheck %s --check-prefix=LOWER

// -----

module {
// ENABLE-LABEL: func.func @reuse_counter_op(
  func.func @reuse_counter_op(%arg0: memref<16xf16, #hivm.address_space<gm>>,
                              %arg1: memref<16xf16, #hivm.address_space<gm>>,
                              %arg2: memref<16xf16, #hivm.address_space<gm>>) {
    %c0_i64 = arith.constant 0 : i64
    %c16_i64 = arith.constant 16 : i64
    %c128_i64 = arith.constant 128 : i64
    %c144_i64 = arith.constant 144 : i64
    %c256_i64 = arith.constant 256 : i64
    %c272_i64 = arith.constant 272 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index

    // ENABLE: scf.for
    scf.for %iv = %c0 to %c4 step %c1 {
      // ENABLE: %[[CTR:.*]] = hivm.hir.multi_buffer_counter loop_id = {{[0-9]+}} -> i64
      // ENABLE: arith.remui %[[CTR]], %{{.*}} : i64
      // ENABLE: arith.remui %[[CTR]], %{{.*}} : i64
      // ENABLE-NOT: hivm.hir.multi_buffer_counter
      %0 = hivm.hir.pointer_cast(%c0_i64, %c16_i64) [] : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %0 {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      %1 = hivm.hir.pointer_cast(%c128_i64, %c144_i64) [] : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %1 {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      %2 = hivm.hir.pointer_cast(%c256_i64, %c272_i64) [] : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>)
                    outs(%0 : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.load ins(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
                    outs(%1 : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.store ins(%2 : memref<16xf16, #hivm.address_space<ub>>)
                     outs(%arg2 : memref<16xf16, #hivm.address_space<gm>>)
    }
    // ENABLE: } {hivm.multi_buffer_loop_id = {{[0-9]+}} : i64}
    return
  }
}

// -----

module {
// LOWER-LABEL: func.func @lower_counter_op(
// LOWER: %[[CTR:.*]] = memref.alloca() {hivm.multi_buffer_counter_for = 7 : i64} : memref<1xi64>
// LOWER: memref.store %{{.*}}, %[[CTR]][%{{.*}}] : memref<1xi64>
  func.func @lower_counter_op() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c1_i64 = arith.constant 1 : i64
    // LOWER: scf.for
    scf.for %iv = %c0 to %c4 step %c1 {
      // LOWER: %[[CUR:.*]] = memref.load %[[CTR]][%{{.*}}] : memref<1xi64>
      // LOWER-NOT: hivm.hir.multi_buffer_counter
      %counter = hivm.hir.multi_buffer_counter loop_id = 7 -> i64
      %use = arith.addi %counter, %c1_i64 : i64
      // LOWER: arith.addi %[[CUR]], %{{.*}} : i64
      // LOWER: memref.store %{{.*}}, %[[CTR]][%{{.*}}] : memref<1xi64>
    }
    // LOWER: } {hivm.multi_buffer_loop_id = 7 : i64}
    return
  }
}
