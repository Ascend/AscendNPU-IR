// RUN: bishengir-opt %s -hivm-enable-multi-buffer -split-input-file | FileCheck  %s

// -----

// CHECK: #map = affine_map<()[s0] -> ((s0 floordiv 4) mod 2)>
module {
// CHECK-LABEL: multi_buffer_alloc_manual(
// CHECK: %[[arg0:.*]]: memref<16xf16, #hivm.address_space<gm>>, %[[arg1:.*]]: memref<16xf16, #hivm.address_space<gm>>) {
  func.func @multi_buffer_alloc_manual(%arg0: memref<16xf16, #hivm.address_space<gm>>, %arg1: memref<16xf16, #hivm.address_space<gm>>) {
    // CHECK: %[[c16:.*]] = arith.constant 16 : index
    // CHECK: %[[c4:.*]] = arith.constant 4 : index
    // CHECK: %[[c0:.*]] = arith.constant 0 : index
    // CHECK: %[[c144_i64:.*]] = arith.constant 144 : i64
    // CHECK: %[[c128_i64:.*]] = arith.constant 128 : i64
    // CHECK: %[[c0_i64:.*]] = arith.constant 0 : i64
    // CHECK: %[[c16_i64:.*]] = arith.constant 16 : i64

    // CHECK: %[[T0:.*]] = hivm.hir.pointer_cast(%[[c0_i64]]) : memref<16xf16, #hivm.address_space<ub>>
    // CHECK: annotation.mark %[[T0]] {attr = 1 : i32} : memref<16xf16, #hivm.address_space<ub>>
    // CHECK: %[[T1:.*]] = hivm.hir.pointer_cast(%[[c16_i64]]) : memref<16xf16, #hivm.address_space<ub>>
    // CHECK: annotation.mark %[[T1]] {attr = 1 : i32} : memref<16xf16, #hivm.address_space<ub>>
    // CHECK: %[[T2:.*]] = hivm.hir.pointer_cast(%[[c128_i64]]) : memref<16xf16, #hivm.address_space<ub>>
    // CHECK: %[[T3:.*]] = hivm.hir.pointer_cast(%[[c144_i64]]) : memref<16xf16, #hivm.address_space<ub>>

    %c0_i64 = arith.constant 0 : i64
    %c16_i64 = arith.constant 16 : i64
    %c128_i64 = arith.constant 128 : i64
    %c144_i64 = arith.constant 144 : i64
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index

    // CHECK: scf.for %[[arg2:.*]] = %[[c0]] to %[[c16]] step %[[c4]] {
    scf.for %arg2 = %c0 to %c16 step %c4 {
      // CHECK: %[[T4:.*]] = affine.apply #map()[%[[arg2]]]
      // CHECK: %[[T5:.*]] = arith.index_cast %[[T4]] : index to i1
      // CHECK: %[[T6:.*]] = arith.select %[[T5]], %[[T0]], %[[T1]] : memref<16xf16, #hivm.address_space<ub>>
      // CHECK: %[[T7:.*]] = affine.apply #map()[%[[arg2]]]
      // CHECK: %[[T8:.*]] = arith.index_cast %[[T7]] : index to i1
      // CHECK: %[[T9:.*]] = arith.select %[[T8]], %[[T2]], %[[T3]] : memref<16xf16, #hivm.address_space<ub>>
      %0 = hivm.hir.pointer_cast(%c0_i64, %c16_i64) [] : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %0 {attr = 1 : i32} : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_ALL>]
      %1 = hivm.hir.pointer_cast(%c128_i64, %c144_i64) [] : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_ALL>]
      // CHECK: hivm.hir.load ins(%[[arg0]] : memref<16xf16, #hivm.address_space<gm>>) outs(%[[T6]] : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>) outs(%0 : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_ALL>]
      // CHECK: hivm.hir.vadd ins(%[[T6]], %[[T6]] : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>) outs(%[[T9]] : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%0, %0 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>) outs(%1 : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_ALL>]
      // CHECK: hivm.hir.store ins(%[[T9]] : memref<16xf16, #hivm.address_space<ub>>) outs(%[[arg1]] : memref<16xf16, #hivm.address_space<gm>>)
      hivm.hir.store ins(%1 : memref<16xf16, #hivm.address_space<ub>>) outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
    }
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}

// -----
// CHECK: #map = affine_map<()[s0, s1] -> ((s1 * 5 + s0 floordiv 3) mod 2)>
module {
// CHECK: func.func @multi_buffer_alloc_manual_2for(%[[arg0:.*]]: memref<16xf16, #hivm.address_space<gm>>, %[[arg1:.*]]: memref<16xf16, #hivm.address_space<gm>>) {
  func.func @multi_buffer_alloc_manual_2for(%arg0: memref<16xf16, #hivm.address_space<gm>>, %arg1: memref<16xf16, #hivm.address_space<gm>>) {
    // CHECK: %[[T0:.*]] = hivm.hir.pointer_cast(%[[c0_i64]]) : memref<16xf16, #hivm.address_space<ub>>
    // CHECK: %[[T1:.*]] = hivm.hir.pointer_cast(%[[c16_i64]]) : memref<16xf16, #hivm.address_space<ub>>
    // CHECK: %[[T2:.*]] = hivm.hir.pointer_cast(%[[c128_i64]]) : memref<16xf16, #hivm.address_space<ub>>
    // CHECK: %[[T3:.*]] = hivm.hir.pointer_cast(%[[c144_i64]]) : memref<16xf16, #hivm.address_space<ub>>

    %c0_i64 = arith.constant 0 : i64
    %c16_i64 = arith.constant 16 : i64
    %c128_i64 = arith.constant 128 : i64
    %c144_i64 = arith.constant 144 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c15 = arith.constant 15 : index
    %c16 = arith.constant 16 : index

    // CHECK: scf.for %[[arg2:.*]]
    scf.for %arg2 = %c0 to %c8 step %c1 {
        // CHECK: scf.for %[[arg3:.*]]
        scf.for %arg3 = %c0 to %c15 step %c3 {
          // CHECK: %[[T4:.*]] = affine.apply #map()[%arg3, %arg2]
          // CHECK: %[[T5:.*]] = arith.index_cast
          // CHECK: %[[T6:.*]] = arith.select %[[T5]], %[[T0]], %[[T1]]
          // CHECK: %[[T7:.*]] = affine.apply #map()[%arg3, %arg2]
          // CHECK: %[[T8:.*]] = arith.index_cast %[[T7]] : index to i1
          // CHECK: %[[T9:.*]] = arith.select %[[T8]], %[[T2]], %[[T3]]
          hivm.hir.pipe_barrier[<PIPE_ALL>]
          %0 = hivm.hir.pointer_cast(%c0_i64, %c16_i64) [] : memref<16xf16, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_ALL>]
          %1 = hivm.hir.pointer_cast(%c128_i64, %c144_i64) [] : memref<16xf16, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_ALL>]
          // CHECK: hivm.hir.load ins(%[[arg0]] : memref<16xf16, #hivm.address_space<gm>>) outs(%[[T6]] : memref<16xf16, #hivm.address_space<ub>>)
          hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>) outs(%0 : memref<16xf16, #hivm.address_space<ub>>)
          hivm.hir.pipe_barrier[<PIPE_ALL>]
          // CHECK: hivm.hir.vadd ins(%[[T6]], %[[T6]] : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>) outs(%[[T9]] : memref<16xf16, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%0, %0 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>) outs(%1 : memref<16xf16, #hivm.address_space<ub>>)
          hivm.hir.pipe_barrier[<PIPE_ALL>]
          // CHECK: hivm.hir.store ins(%[[T9]] : memref<16xf16, #hivm.address_space<ub>>) outs(%[[arg1]] : memref<16xf16, #hivm.address_space<gm>>)
          hivm.hir.store ins(%1 : memref<16xf16, #hivm.address_space<ub>>) outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
        }
      hivm.hir.pipe_barrier[<PIPE_ALL>]
    }

    return
  }
}

// -----
// CHECK: #map = affine_map<()[s0] -> (s0 mod 2)>
// CHECK: #map1 = affine_map<()[s0, s1] -> ((s0 + s1 * 15) mod 2)>
module {
// CHECK-LABEL: func.func @multi_buffer_alloc_manual_for_vadd(
  func.func @multi_buffer_alloc_manual_for_vadd(%arg0: memref<16xf16, #hivm.address_space<gm>>, %arg1: memref<16xf16, #hivm.address_space<gm>>) {
    %c0_i64 = arith.constant 0 : i64
    %c16_i64 = arith.constant 16 : i64
    %c128_i64 = arith.constant 128 : i64
    %c144_i64 = arith.constant 144 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c15 = arith.constant 15 : index
    %c16 = arith.constant 16 : index

    // CHECK: %[[T0:.*]] = hivm.hir.pointer_cast(%c128_i64) : memref<16xf16, #hivm.address_space<ub>>
    // CHECK: %[[T1:.*]] = hivm.hir.pointer_cast(%c144_i64) : memref<16xf16, #hivm.address_space<ub>>
    // CHECK: %[[T2:.*]] = hivm.hir.pointer_cast(%c0_i64) : memref<16xf16, #hivm.address_space<ub>>
    // CHECK: %[[T3:.*]] = hivm.hir.pointer_cast(%c16_i64) : memref<16xf16, #hivm.address_space<ub>>

    // CHECK: scf.for %[[arg2:.*]]
    scf.for %arg2 = %c0 to %c8 step %c1 {
      // CHECK: %[[T4:.*]] = affine.apply #map()
      // CHECK: %[[T5:.*]] = arith.index_cast
      // CHECK: %[[T6:.*]] = arith.select %[[T5]], %[[T0]], %[[T1]]

      %1 = hivm.hir.pointer_cast(%c128_i64, %c144_i64) [] : memref<16xf16, #hivm.address_space<ub>>
      // CHECK: scf.for %[[arg3:.*]]
      scf.for %arg3 = %c0 to %c15 step %c1 {
        // CHECK: %[[T7:.*]] = affine.apply #map1()
        // CHECK: %[[T8:.*]] = arith.index_cast
        // CHECK: %[[T9:.*]] = arith.select %[[T8]], %[[T2]], %[[T3]]
        %0 = hivm.hir.pointer_cast(%c0_i64, %c16_i64) [] : memref<16xf16, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_ALL>]
        hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>)
                      outs(%0 : memref<16xf16, #hivm.address_space<ub>>)
        hivm.hir.pipe_barrier[<PIPE_ALL>]
        hivm.hir.vadd ins(%0, %0 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>)
                      outs(%1 : memref<16xf16, #hivm.address_space<ub>>)
      }

      hivm.hir.pipe_barrier[<PIPE_ALL>]
      hivm.hir.store ins(%1 : memref<16xf16, #hivm.address_space<ub>>)
                     outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)

      hivm.hir.pipe_barrier[<PIPE_ALL>]
    }

    return
  }
}

// -----
// CHECK: #map = affine_map<()[s0] -> (s0 mod 2)>
module {
// CHECK-LABEL: func.func @multi_buffer_alloc_manual_for_vadd_vmul(
  func.func @multi_buffer_alloc_manual_for_vadd_vmul(%arg0: memref<16xf16, #hivm.address_space<gm>>, %arg1: memref<16xf16, #hivm.address_space<gm>>) {
    %c0_i64 = arith.constant 0 : i64
    %c16_i64 = arith.constant 16 : i64
    %c128_i64 = arith.constant 128 : i64
    %c144_i64 = arith.constant 144 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c15 = arith.constant 15 : index
    %c16 = arith.constant 16 : index

    // CHECK: %[[T0:.*]] = hivm.hir.pointer_cast(%c0_i64) : memref<16xf16, #hivm.address_space<ub>>
    // CHECK: %[[T1:.*]] = hivm.hir.pointer_cast(%c16_i64) : memref<16xf16, #hivm.address_space<ub>>
    // CHECK: %[[T2:.*]] = hivm.hir.pointer_cast(%c128_i64) : memref<16xf16, #hivm.address_space<ub>>
    // CHECK: %[[T3:.*]] = hivm.hir.pointer_cast(%c144_i64) : memref<16xf16, #hivm.address_space<ub>>

    scf.for %arg2 = %c0 to %c8 step %c1 {
      // CHECK: %[[T4:.*]] = affine.apply #map()
      // CHECK: %[[T5:.*]] = arith.index_cast
      // CHECK: %[[T6:.*]] = arith.select %[[T5]], %[[T0]], %[[T1]]
      // CHECK: %[[T7:.*]] = affine.apply #map()
      // CHECK: %[[T8:.*]] = arith.index_cast %[[T7]] : index to i1
      // CHECK: %[[T9:.*]] = arith.select %[[T8]], %[[T2]], %[[T3]] : memref<16xf16, #hivm.address_space<ub>>

      %0 = hivm.hir.pointer_cast(%c0_i64, %c16_i64) [] : memref<16xf16, #hivm.address_space<ub>>
      %1 = hivm.hir.pointer_cast(%c128_i64, %c144_i64) [] : memref<16xf16, #hivm.address_space<ub>>
      scf.for %arg3 = %c0 to %c15 step %c1 {
        hivm.hir.pipe_barrier[<PIPE_ALL>]
        hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>)
                      outs(%0 : memref<16xf16, #hivm.address_space<ub>>)
        hivm.hir.pipe_barrier[<PIPE_ALL>]
        hivm.hir.vadd ins(%0, %0 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>)
                      outs(%1 : memref<16xf16, #hivm.address_space<ub>>)
      }
      scf.for %arg4 = %c0 to %c15 step %c3 {
        hivm.hir.pipe_barrier[<PIPE_ALL>]
        hivm.hir.vmul ins(%0, %0 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>)
                      outs(%1 : memref<16xf16, #hivm.address_space<ub>>)
      }

      hivm.hir.pipe_barrier[<PIPE_ALL>]
      hivm.hir.store ins(%1 : memref<16xf16, #hivm.address_space<ub>>)
                     outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)

      hivm.hir.pipe_barrier[<PIPE_ALL>]
    }

    return
  }
}

// -----
// CHECK: #map = affine_map<()[s0, s1, s2] -> ((s0 + s1 * 15 + ((s2 - 1) floordiv 2) * 225) mod 2)>
module {
  func.func @multi_buffer_alloc_manual_3for(%arg0: memref<16xf16, #hivm.address_space<gm>>, %arg1: memref<16xf16, #hivm.address_space<gm>>) {
    %c0_i64 = arith.constant 0 : i64
    %c16_i64 = arith.constant 16 : i64
    %c128_i64 = arith.constant 128 : i64
    %c144_i64 = arith.constant 144 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c15 = arith.constant 15 : index
    %c16 = arith.constant 16 : index

    // CHECK: hivm.hir.pointer_cast
    // CHECK: hivm.hir.pointer_cast
    // CHECK: hivm.hir.pointer_cast
    // CHECK: hivm.hir.pointer_cast

    // CHECK: scf.for %[[arg2:.*]]
    scf.for %arg2 = %c1 to %c15 step %c2 {
      // CHECK: scf.for %[[arg3:.*]]
      scf.for %arg3 = %c0 to %c15 step %c1 {
          // CHECK: scf.for %[[arg4:.*]]
          scf.for %arg4 = %c0 to %c15 step %c1 {
            // CHECK: affine.apply #map()[%arg4, %arg3, %arg2]
            // CHECK: arith.index_cast
            // CHECK: arith.select
            // CHECK: affine.apply #map()[%arg4, %arg3, %arg2]
            // CHECK: arith.index_cast
            // CHECK: arith.select
            hivm.hir.pipe_barrier[<PIPE_ALL>]
            %0 = hivm.hir.pointer_cast(%c0_i64, %c16_i64) [] : memref<16xf16, #hivm.address_space<ub>>
            hivm.hir.pipe_barrier[<PIPE_ALL>]
            %1 = hivm.hir.pointer_cast(%c128_i64, %c144_i64) [] : memref<16xf16, #hivm.address_space<ub>>

            hivm.hir.pipe_barrier[<PIPE_ALL>]
            hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>) outs(%0 : memref<16xf16, #hivm.address_space<ub>>)
            hivm.hir.pipe_barrier[<PIPE_ALL>]
            hivm.hir.vadd ins(%0, %0 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>) outs(%1 : memref<16xf16, #hivm.address_space<ub>>)
            hivm.hir.pipe_barrier[<PIPE_ALL>]
            hivm.hir.store ins(%1 : memref<16xf16, #hivm.address_space<ub>>) outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
          }
        hivm.hir.pipe_barrier[<PIPE_ALL>]
      }
    }
    return
  }
}

// -----
module {
  func.func @test_for_yield_db_ptr(%arg0: memref<1x2048xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32})
      attributes {global_kernel = "local", hacc.entry = "", hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {

    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c49152_i32 = arith.constant 49152 : i32

    %c8192_i64 = arith.constant 8192 : i64
    %c24576_i64 = arith.constant 24576 : i64
    %c53312_i64 = arith.constant 53312 : i64
    // CHECK: scf.for
    scf.for %arg7 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
      // CHECK: arith.select
      %29 = hivm.hir.pointer_cast(%c8192_i64) : memref<1x2048xf16, #hivm.address_space<ub>>
      // CHECK: scf.for
      %31 = scf.for %arg8 = %c0_i32 to %c49152_i32 step %c2048_i32 iter_args(%arg9 = %29) -> (memref<1x2048xf16, #hivm.address_space<ub>>)  : i32 {
        // CHECK-NOT: arith.select
        %39 = hivm.hir.pointer_cast(%c24576_i64, %c53312_i64) : memref<1x2048xf16, #hivm.address_space<ub>>
        hivm.hir.load ins(%arg0 : memref<1x2048xf16, #hivm.address_space<gm>>) outs(%39 : memref<1x2048xf16, #hivm.address_space<ub>>)

        scf.yield %39 : memref<1x2048xf16, #hivm.address_space<ub>>
      }
    }

    return
  }
}

// -----
module {
  func.func @test_for_not_yield_db_ptr(
      %arg0: memref<1x2048xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32})
      attributes {global_kernel = "local", hacc.entry = "", hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {

    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c49152_i32 = arith.constant 49152 : i32

    %c8192_i64 = arith.constant 8192 : i64
    %c24576_i64 = arith.constant 24576 : i64
    %c53312_i64 = arith.constant 53312 : i64
    %c57408_i64 = arith.constant 57408 : i64
    // CHECK: scf.for
    scf.for %arg7 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
      %29 = hivm.hir.pointer_cast(%c8192_i64) : memref<1x2048xf16, #hivm.address_space<ub>>
      // CHECK: scf.for
      %31 = scf.for %arg8 = %c0_i32 to %c49152_i32 step %c2048_i32 iter_args(%arg9 = %29) -> (memref<1x2048xf16, #hivm.address_space<ub>>)  : i32 {
        // CHECK: arith.select
        %39 = hivm.hir.pointer_cast(%c24576_i64, %c53312_i64) : memref<1x2048xf16, #hivm.address_space<ub>>
        hivm.hir.load ins(%arg0 : memref<1x2048xf16, #hivm.address_space<gm>>) outs(%39 : memref<1x2048xf16, #hivm.address_space<ub>>)

        %43 = hivm.hir.pointer_cast(%c57408_i64) : memref<1x2048xf16, #hivm.address_space<ub>>
        hivm.hir.vadd ins(%39, %39 : memref<1x2048xf16, #hivm.address_space<ub>>, memref<1x2048xf16, #hivm.address_space<ub>>) outs(%43 : memref<1x2048xf16, #hivm.address_space<ub>>)
        scf.yield %43 : memref<1x2048xf16, #hivm.address_space<ub>>
      }
    }

    return
  }
}

// -----
module {
  func.func @test_three_for_inner_yield_db_ptr(%arg0: memref<1x2048xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32})
      attributes {global_kernel = "local", hacc.entry = "", hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {

    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c49152_i32 = arith.constant 49152 : i32

    %c8192_i64 = arith.constant 8192 : i64
    %c24576_i64 = arith.constant 24576 : i64
    %c53312_i64 = arith.constant 53312 : i64
    %c57408_i64 = arith.constant 57408 : i64

    %29 = hivm.hir.pointer_cast(%c8192_i64) : memref<1x2048xf16, #hivm.address_space<ub>>
    // CHECK: scf.for
    scf.for %arg6 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
      // CHECK: scf.for
      scf.for %arg7 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg9 = %29) -> (memref<1x2048xf16, #hivm.address_space<ub>>) : i32 {
        // CHECK: arith.select

        // CHECK: scf.for
        %31:2 = scf.for %arg8 = %c0_i32 to %c49152_i32 step %c2048_i32 iter_args(%arg10 = %29, %arg11 = %29) -> (memref<1x2048xf16, #hivm.address_space<ub>>, memref<1x2048xf16, #hivm.address_space<ub>>)  : i32 {
          %39 = hivm.hir.pointer_cast(%c24576_i64, %c53312_i64) : memref<1x2048xf16, #hivm.address_space<ub>>
          hivm.hir.load ins(%arg0 : memref<1x2048xf16, #hivm.address_space<gm>>) outs(%39 : memref<1x2048xf16, #hivm.address_space<ub>>)

          %43 = hivm.hir.pointer_cast(%c57408_i64) : memref<1x2048xf16, #hivm.address_space<ub>>
          hivm.hir.vadd ins(%39, %39 : memref<1x2048xf16, #hivm.address_space<ub>>, memref<1x2048xf16, #hivm.address_space<ub>>) outs(%43 : memref<1x2048xf16, #hivm.address_space<ub>>)

          scf.yield %39, %43 : memref<1x2048xf16, #hivm.address_space<ub>>, memref<1x2048xf16, #hivm.address_space<ub>>
        }

        scf.yield %31#1 : memref<1x2048xf16, #hivm.address_space<ub>>
      }
    }

    return
  }
}

// -----
module {
  func.func @test_three_for_2for_yield_db_ptr(%arg0: memref<1x2048xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32})
      attributes {global_kernel = "local", hacc.entry = "", hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {

    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c49152_i32 = arith.constant 49152 : i32

    %c8192_i64 = arith.constant 8192 : i64
    %c24576_i64 = arith.constant 24576 : i64
    %c53312_i64 = arith.constant 53312 : i64
    %c57408_i64 = arith.constant 57408 : i64

    %29 = hivm.hir.pointer_cast(%c8192_i64) : memref<1x2048xf16, #hivm.address_space<ub>>
    // CHECK: scf.for
    scf.for %arg6 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
      // CHECK: arith.select
      // CHECK: scf.for
      scf.for %arg7 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg9 = %29) -> (memref<1x2048xf16, #hivm.address_space<ub>>) : i32 {
        %31:2 = scf.for %arg8 = %c0_i32 to %c49152_i32 step %c2048_i32 iter_args(%arg10 = %29, %arg11 = %29) -> (memref<1x2048xf16, #hivm.address_space<ub>>, memref<1x2048xf16, #hivm.address_space<ub>>)  : i32 {
          %39 = hivm.hir.pointer_cast(%c24576_i64, %c53312_i64) : memref<1x2048xf16, #hivm.address_space<ub>>
          hivm.hir.load ins(%arg0 : memref<1x2048xf16, #hivm.address_space<gm>>) outs(%39 : memref<1x2048xf16, #hivm.address_space<ub>>)

          %43 = hivm.hir.pointer_cast(%c57408_i64) : memref<1x2048xf16, #hivm.address_space<ub>>
          hivm.hir.vadd ins(%39, %39 : memref<1x2048xf16, #hivm.address_space<ub>>, memref<1x2048xf16, #hivm.address_space<ub>>) outs(%43 : memref<1x2048xf16, #hivm.address_space<ub>>)

          scf.yield %39, %43 : memref<1x2048xf16, #hivm.address_space<ub>>, memref<1x2048xf16, #hivm.address_space<ub>>
        }

        scf.yield %31#0 : memref<1x2048xf16, #hivm.address_space<ub>>
      }
    }

    return
  }
}

// -----
module {
  func.func @test_for_yield_db_ptr_if_else(%arg0: memref<1x2048xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}) attributes {global_kernel = "local", hacc.entry = "", hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c49152_i32 = arith.constant 49152 : i32
    %c8192_i64 = arith.constant 8192 : i64
    %c24576_i64 = arith.constant 24576 : i64
    %c28672_i64 = arith.constant 28672 : i64
    %c53312_i64 = arith.constant 53312 : i64
    %c57408_i64 = arith.constant 57408 : i64
    %c61504_i64 = arith.constant 61504 : i64
    %true = arith.constant true

    // CHECK: scf.for
    scf.for %arg1 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
      // CHECK: arith.select
      // CHECK: arith.select
      // CHECK: arith.select

      %0 = hivm.hir.pointer_cast(%c24576_i64, %c53312_i64) : memref<1x2048xf16, #hivm.address_space<ub>>

      // CHECK: scf.for
      %1 = scf.for %arg2 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg3 = %0) -> (memref<1x2048xf16, #hivm.address_space<ub>>)  : i32 {
        %2 = hivm.hir.pointer_cast(%c61504_i64) : memref<1x2048xf16, #hivm.address_space<ub>>
        // CHECK: scf.for
        %3 = scf.for %arg4 = %c0_i32 to %c49152_i32 step %c2048_i32 iter_args(%arg5 = %2) -> (memref<1x2048xf16, #hivm.address_space<ub>>)  : i32 {
          // CHECK-NOT: arith.select
          %4 = scf.if %true -> (memref<1x2048xf16, #hivm.address_space<ub>>) {
            %5 = hivm.hir.pointer_cast(%c24576_i64, %c53312_i64) : memref<1x2048xf16, #hivm.address_space<ub>>
            hivm.hir.load ins(%arg0 : memref<1x2048xf16, #hivm.address_space<gm>>) outs(%5 : memref<1x2048xf16, #hivm.address_space<ub>>)
            scf.yield %5 : memref<1x2048xf16, #hivm.address_space<ub>>
          } else {
            %5 = hivm.hir.pointer_cast(%c28672_i64, %c57408_i64) : memref<1x2048xf16, #hivm.address_space<ub>>
            hivm.hir.load ins(%arg0 : memref<1x2048xf16, #hivm.address_space<gm>>) outs(%5 : memref<1x2048xf16, #hivm.address_space<ub>>)
            scf.yield %5 : memref<1x2048xf16, #hivm.address_space<ub>>
          }
          scf.yield %4 : memref<1x2048xf16, #hivm.address_space<ub>>
        }
        scf.yield %3 : memref<1x2048xf16, #hivm.address_space<ub>>
      }
    }
    return
  }
}
