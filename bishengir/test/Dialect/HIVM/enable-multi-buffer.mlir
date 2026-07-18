// RUN: bishengir-opt %s -hivm-enable-multi-buffer -hivm-lower-multi-buffer-counter -split-input-file | FileCheck  %s

// EnableMultiBuffer now drives slot rotation through MultiBufferLoopAdapter
// for both scf.for and scf.while. For
// scf.for parents the previous affine.apply((iv - lb)/step) % modular
// codegen has been replaced by a function-scoped memref.alloca<1xi64>
// counter plus body-head memref.load / arith.remui / arith.select cascade
// and body-tail arith.addi / memref.store-back. These CHECKs verify those
// structural markers; exact SSA names are intentionally not bound (they
// shift between unrelated runs).

// -----

module {
// CHECK-LABEL: func.func @multi_buffer_alloc_manual(
// CHECK:   memref.alloca() : memref<1xi64>
// CHECK:   memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1xi64>
// CHECK:   hivm.hir.pointer_cast(%{{.*}}) : memref<16xf16, #hivm.address_space<ub>>
// CHECK:   hivm.hir.pointer_cast(%{{.*}}) : memref<16xf16, #hivm.address_space<ub>>
  func.func @multi_buffer_alloc_manual(%arg0: memref<16xf16, #hivm.address_space<gm>>, %arg1: memref<16xf16, #hivm.address_space<gm>>) {
    %c0_i64 = arith.constant 0 : i64
    %c16_i64 = arith.constant 16 : i64
    %c128_i64 = arith.constant 128 : i64
    %c144_i64 = arith.constant 144 : i64
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index

    scf.for %arg2 = %c0 to %c16 step %c4 {
      // CHECK: scf.for
      // CHECK:   memref.load %{{.*}}[%{{.*}}] : memref<1xi64>
      // CHECK:   arith.remui %{{.*}}, %{{.*}} : i64
      // CHECK:   arith.cmpi eq, %{{.*}}, %{{.*}} : i64
      // CHECK:   arith.select %{{.*}} : memref<16xf16, #hivm.address_space<ub>>
      // CHECK:   arith.remui %{{.*}}, %{{.*}} : i64
      // CHECK:   arith.cmpi eq, %{{.*}}, %{{.*}} : i64
      // CHECK:   arith.select %{{.*}} : memref<16xf16, #hivm.address_space<ub>>
      %0 = hivm.hir.pointer_cast(%c0_i64, %c16_i64) [] : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %0 {attr = 1 : i32} : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_ALL>]
      %1 = hivm.hir.pointer_cast(%c128_i64, %c144_i64) [] : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_ALL>]
      // CHECK: hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>) outs(%{{.*}} : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>) outs(%0 : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_ALL>]
      // CHECK: hivm.hir.vadd ins(%{{.*}}, %{{.*}} : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>) outs(%{{.*}} : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%0, %0 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>) outs(%1 : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_ALL>]
      // CHECK: hivm.hir.store ins(%{{.*}} : memref<16xf16, #hivm.address_space<ub>>) outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
      hivm.hir.store ins(%1 : memref<16xf16, #hivm.address_space<ub>>) outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
      // CHECK:   arith.addi %{{.*}}, %{{.*}} : i64
      // CHECK:   memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1xi64>
    }
    // CHECK: }
    return
  }
}

// -----
module {
// CHECK-LABEL: func.func @multi_buffer_alloc_manual_2for(
// CHECK:   memref.alloca() : memref<1xi64>
// CHECK:   memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1xi64>
  func.func @multi_buffer_alloc_manual_2for(%arg0: memref<16xf16, #hivm.address_space<gm>>, %arg1: memref<16xf16, #hivm.address_space<gm>>) {
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

    // CHECK: scf.for
    scf.for %arg2 = %c0 to %c8 step %c1 {
        // CHECK: scf.for
        scf.for %arg3 = %c0 to %c15 step %c3 {
          // CHECK:   memref.load %{{.*}}[%{{.*}}] : memref<1xi64>
          // The innermost for-loop owns the counter; its body has the
          // load + remui + select cascade.
          // CHECK:   arith.remui %{{.*}}, %{{.*}} : i64
          // CHECK:   arith.cmpi eq, %{{.*}}, %{{.*}} : i64
          // CHECK:   arith.select %{{.*}} : memref<16xf16, #hivm.address_space<ub>>
          // CHECK:   arith.remui %{{.*}}, %{{.*}} : i64
          // CHECK:   arith.cmpi eq, %{{.*}}, %{{.*}} : i64
          // CHECK:   arith.select %{{.*}} : memref<16xf16, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_ALL>]
          %0 = hivm.hir.pointer_cast(%c0_i64, %c16_i64) [] : memref<16xf16, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_ALL>]
          %1 = hivm.hir.pointer_cast(%c128_i64, %c144_i64) [] : memref<16xf16, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_ALL>]
          // CHECK: hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>) outs(%{{.*}} : memref<16xf16, #hivm.address_space<ub>>)
          hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>) outs(%0 : memref<16xf16, #hivm.address_space<ub>>)
          hivm.hir.pipe_barrier[<PIPE_ALL>]
          // CHECK: hivm.hir.vadd ins(%{{.*}}, %{{.*}} : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>) outs(%{{.*}} : memref<16xf16, #hivm.address_space<ub>>)
          hivm.hir.vadd ins(%0, %0 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>) outs(%1 : memref<16xf16, #hivm.address_space<ub>>)
          hivm.hir.pipe_barrier[<PIPE_ALL>]
          // CHECK: hivm.hir.store ins(%{{.*}} : memref<16xf16, #hivm.address_space<ub>>) outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
          hivm.hir.store ins(%1 : memref<16xf16, #hivm.address_space<ub>>) outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
          // CHECK:   arith.addi %{{.*}}, %{{.*}} : i64
          // CHECK:   memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1xi64>
        }
        // CHECK: }
    }

    return
  }
}

// -----
module {
// CHECK-LABEL: func.func @multi_buffer_alloc_manual_for_vadd(
// Two distinct multi-buffer candidates parented to two different fors:
// outer for owns one counter (alloc %1 hoisted to outer body), inner for
// owns the other (alloc %0 in inner body). Two alloca<1xi64> at funcOp top,
// one per scf.for op.
// CHECK-DAG:   memref.alloca() : memref<1xi64>
// CHECK-DAG:   memref.alloca() : memref<1xi64>
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

    // CHECK: scf.for
    scf.for %arg2 = %c0 to %c8 step %c1 {
      // Outer for body-head: load + remui + select for the outer multi-buffer
      // candidate (the vadd output buffer %1).
      // CHECK:   memref.load %{{.*}}[%{{.*}}] : memref<1xi64>
      // CHECK:   arith.remui %{{.*}}, %{{.*}} : i64
      // CHECK:   arith.cmpi eq, %{{.*}}, %{{.*}} : i64
      // CHECK:   arith.select %{{.*}} : memref<16xf16, #hivm.address_space<ub>>

      %1 = hivm.hir.pointer_cast(%c128_i64, %c144_i64) [] : memref<16xf16, #hivm.address_space<ub>>
      // CHECK: scf.for
      scf.for %arg3 = %c0 to %c15 step %c1 {
        // Inner for body-head: load + remui + select for the inner candidate.
        // CHECK:   memref.load %{{.*}}[%{{.*}}] : memref<1xi64>
        // CHECK:   arith.remui %{{.*}}, %{{.*}} : i64
        // CHECK:   arith.cmpi eq, %{{.*}}, %{{.*}} : i64
        // CHECK:   arith.select %{{.*}} : memref<16xf16, #hivm.address_space<ub>>
        %0 = hivm.hir.pointer_cast(%c0_i64, %c16_i64) [] : memref<16xf16, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_ALL>]
        hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>)
                      outs(%0 : memref<16xf16, #hivm.address_space<ub>>)
        hivm.hir.pipe_barrier[<PIPE_ALL>]
        hivm.hir.vadd ins(%0, %0 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>)
                      outs(%1 : memref<16xf16, #hivm.address_space<ub>>)
        // CHECK:   arith.addi %{{.*}}, %{{.*}} : i64
        // CHECK:   memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1xi64>
      }
      // CHECK: }
      hivm.hir.pipe_barrier[<PIPE_ALL>]
      hivm.hir.store ins(%1 : memref<16xf16, #hivm.address_space<ub>>)
                     outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)

      hivm.hir.pipe_barrier[<PIPE_ALL>]
      // CHECK:   arith.addi %{{.*}}, %{{.*}} : i64
      // CHECK:   memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1xi64>
    }
    // CHECK: }
    return
  }
}

// -----
module {
// CHECK-LABEL: func.func @multi_buffer_alloc_manual_for_vadd_vmul(
// Single multi-buffer candidate parented to the outer for; the two inner
// fors host vadd/vmul uses but are not multi-buffer parents themselves
// (the alloc lives in the outer-for body), so only one counter alloca and
// one loop_id attribute show up.
// CHECK:   memref.alloca() : memref<1xi64>
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

    // CHECK: scf.for
    scf.for %arg2 = %c0 to %c8 step %c1 {
      // CHECK:   memref.load %{{.*}}[%{{.*}}] : memref<1xi64>
      // CHECK:   arith.remui %{{.*}}, %{{.*}} : i64
      // CHECK:   arith.cmpi eq, %{{.*}}, %{{.*}} : i64
      // CHECK:   arith.select %{{.*}} : memref<16xf16, #hivm.address_space<ub>>
      // CHECK:   arith.remui %{{.*}}, %{{.*}} : i64
      // CHECK:   arith.cmpi eq, %{{.*}}, %{{.*}} : i64
      // CHECK:   arith.select %{{.*}} : memref<16xf16, #hivm.address_space<ub>>

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
      // CHECK:   arith.addi %{{.*}}, %{{.*}} : i64
      // CHECK:   memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1xi64>
    }
    // CHECK: }
    return
  }
}

// -----
module {
// CHECK-LABEL: func.func @multi_buffer_alloc_manual_3for(
// 3-level nested for, multi-buffer candidate parented to innermost. One
// alloca at funcOp top, slot select cascade lives in innermost body.
// CHECK:   memref.alloca() : memref<1xi64>
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

    scf.for %arg2 = %c1 to %c15 step %c2 {
      scf.for %arg3 = %c0 to %c15 step %c1 {
          scf.for %arg4 = %c0 to %c15 step %c1 {
            // CHECK: memref.load %{{.*}}[%{{.*}}] : memref<1xi64>
            // CHECK: arith.remui %{{.*}}, %{{.*}} : i64
            // CHECK: arith.cmpi eq, %{{.*}}, %{{.*}} : i64
            // CHECK: arith.select %{{.*}} : memref<16xf16, #hivm.address_space<ub>>
            // CHECK: arith.remui %{{.*}}, %{{.*}} : i64
            // CHECK: arith.cmpi eq, %{{.*}}, %{{.*}} : i64
            // CHECK: arith.select %{{.*}} : memref<16xf16, #hivm.address_space<ub>>
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
            // CHECK: arith.addi %{{.*}}, %{{.*}} : i64
            // CHECK: memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1xi64>
          }
          // CHECK: }
      }
    }
    return
  }
}

// -----
module {
// CHECK-LABEL: func.func @test_for_yield_db_ptr(
// The double-buffer ptr (%39) is yielded by the inner for but is ALSO
// consumed (loaded) inside it, so the inner for is the rotation anchor: the
// slot-select cascade lives in the inner-for body and the outer for has none.
// (Before the isConsumedInLoop guard this incorrectly climbed to the outer
// for; see getParentLoop in EnableMultiBuffer.cpp.)
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
    // CHECK:   memref.alloca() : memref<1xi64>
    // CHECK: scf.for
    scf.for %arg7 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
      // CHECK-NOT: arith.select
      %29 = hivm.hir.pointer_cast(%c8192_i64) : memref<1x2048xf16, #hivm.address_space<ub>>
      // CHECK: scf.for
      %31 = scf.for %arg8 = %c0_i32 to %c49152_i32 step %c2048_i32 iter_args(%arg9 = %29) -> (memref<1x2048xf16, #hivm.address_space<ub>>)  : i32 {
        // CHECK: arith.select
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
// CHECK-LABEL: func.func @test_for_not_yield_db_ptr(
// db ptr is NOT yielded by inner for; multi-buffer fires on the inner for
// directly, so the select cascade is in the inner body.
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
    // CHECK:   memref.alloca() : memref<1xi64>
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
// CHECK-LABEL: func.func @test_three_for_inner_yield_db_ptr(
// 3-nested for; the db ptr (%39) is yielded up the chain but is consumed
// (load + vadd) in the innermost for, so the innermost for is the rotation
// anchor and the slot-select cascade stays there. (Before the
// isConsumedInLoop guard this climbed to the middle for.)
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
    // CHECK:   memref.alloca() : memref<1xi64>
    // CHECK: scf.for
    scf.for %arg6 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
      // CHECK: scf.for
      scf.for %arg7 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg9 = %29) -> (memref<1x2048xf16, #hivm.address_space<ub>>) : i32 {
        // CHECK-NOT: arith.select
        // CHECK: scf.for
        %31:2 = scf.for %arg8 = %c0_i32 to %c49152_i32 step %c2048_i32 iter_args(%arg10 = %29, %arg11 = %29) -> (memref<1x2048xf16, #hivm.address_space<ub>>, memref<1x2048xf16, #hivm.address_space<ub>>)  : i32 {
          // CHECK: arith.select
          %39 = hivm.hir.pointer_cast(%c24576_i64, %c53312_i64) : memref<1x2048xf16, #hivm.address_space<ub>>
          hivm.hir.load ins(%arg0 : memref<1x2048xf16, #hivm.address_space<gm>>) outs(%39 : memref<1x2048xf16, #hivm.address_space<ub>>)

          %43 = hivm.hir.pointer_cast(%c57408_i64) : memref<1x2048xf16, #hivm.address_space<ub>>
          hivm.hir.vadd ins(%39, %39 : memref<1x2048xf16, #hivm.address_space<ub>>, memref<1x2048xf16, #hivm.address_space<ub>>) outs(%43 : memref<1x2048xf16, #hivm.address_space<ub>>)

          scf.yield %39, %43 : memref<1x2048xf16, #hivm.address_space<ub>>, memref<1x2048xf16, #hivm.address_space<ub>>
        }

        scf.yield %31#1 : memref<1x2048xf16, #hivm.address_space<ub>>
      }
      scf.yield
    }

    return
  }
}

// -----
module {
// CHECK-LABEL: func.func @test_three_for_2for_yield_db_ptr(
// 3-nested for, db ptr (%39) yielded by innermost AND middle for, but
// consumed (load + vadd) in the innermost for. The innermost for is therefore
// the rotation anchor and the slot-select cascade stays there. (Before the
// isConsumedInLoop guard this climbed all the way to the outermost for.)
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
    // CHECK:   memref.alloca() : memref<1xi64>
    // CHECK: scf.for
    scf.for %arg6 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
      // CHECK-NOT: arith.select
      // CHECK: scf.for
      scf.for %arg7 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg9 = %29) -> (memref<1x2048xf16, #hivm.address_space<ub>>) : i32 {
        // CHECK-NOT: arith.select
        // CHECK: scf.for
        %31:2 = scf.for %arg8 = %c0_i32 to %c49152_i32 step %c2048_i32 iter_args(%arg10 = %29, %arg11 = %29) -> (memref<1x2048xf16, #hivm.address_space<ub>>, memref<1x2048xf16, #hivm.address_space<ub>>)  : i32 {
          // CHECK: arith.select
          %39 = hivm.hir.pointer_cast(%c24576_i64, %c53312_i64) : memref<1x2048xf16, #hivm.address_space<ub>>
          hivm.hir.load ins(%arg0 : memref<1x2048xf16, #hivm.address_space<gm>>) outs(%39 : memref<1x2048xf16, #hivm.address_space<ub>>)

          %43 = hivm.hir.pointer_cast(%c57408_i64) : memref<1x2048xf16, #hivm.address_space<ub>>
          hivm.hir.vadd ins(%39, %39 : memref<1x2048xf16, #hivm.address_space<ub>>, memref<1x2048xf16, #hivm.address_space<ub>>) outs(%43 : memref<1x2048xf16, #hivm.address_space<ub>>)

          scf.yield %39, %43 : memref<1x2048xf16, #hivm.address_space<ub>>, memref<1x2048xf16, #hivm.address_space<ub>>
        }

        scf.yield %31#0 : memref<1x2048xf16, #hivm.address_space<ub>>
      }
      scf.yield
    }

    return
  }
}

// -----
module {
// CHECK-LABEL: func.func @test_for_yield_db_ptr_if_else(
// scf.if inside a for. The outer pointer_cast (%0) is consumed by the outer
// for result chain, while the branch-local pointer_casts are consumed inside
// the innermost for/if body; both anchor levels get their own counters.
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

    // CHECK-DAG: memref.alloca() : memref<1xi64>
    // CHECK-DAG: memref.alloca() : memref<1xi64>
    // CHECK: scf.for
    scf.for %arg1 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
      // Outer-level select for %0.
      // CHECK: arith.select

      %0 = hivm.hir.pointer_cast(%c24576_i64, %c53312_i64) : memref<1x2048xf16, #hivm.address_space<ub>>

      // CHECK: scf.for
      %1 = scf.for %arg2 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg3 = %0) -> (memref<1x2048xf16, #hivm.address_space<ub>>)  : i32 {
        %2 = hivm.hir.pointer_cast(%c61504_i64) : memref<1x2048xf16, #hivm.address_space<ub>>
        // CHECK: scf.for
        %3 = scf.for %arg4 = %c0_i32 to %c49152_i32 step %c2048_i32 iter_args(%arg5 = %2) -> (memref<1x2048xf16, #hivm.address_space<ub>>)  : i32 {
          // Inner-level selects for the then/else branch pointer_casts.
          // CHECK: arith.select
          // CHECK: arith.select
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

// -----
module {
// CHECK-LABEL: func.func @test_consumed_loop_yield_no_copy(
// The yielded pointer is consumed by vmul in the loop body, so the loop is its
// rotation anchor. The value can be yielded directly without a fallback copy.
  func.func @test_consumed_loop_yield_no_copy(%arg0: memref<16xf16, #hivm.address_space<gm>>,
                                              %arg1: memref<16xf16, #hivm.address_space<gm>>,
                                              %arg2: memref<16xf16, #hivm.address_space<gm>>,
                                              %arg3: memref<16xf16, #hivm.address_space<gm>>,
                                              %arg4: f16) {
    %c0_i64 = arith.constant 0 : i64
    %c96_i64 = arith.constant 96 : i64
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c32_i64 = arith.constant 32 : i64
    %c64_i64 = arith.constant 64 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>) outs(%0 : memref<16xf16, #hivm.address_space<ub>>)
    // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG:.*]] = %{{.*}})
    %1 = scf.for %arg5 = %c0 to %c8 step %c1 iter_args(%arg6 = %0) -> (memref<16xf16, #hivm.address_space<ub>>) {
      // CHECK: arith.select
      %2 = hivm.hir.pointer_cast(%c32_i64, %c64_i64) : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %2 {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      // CHECK-NOT: arith.select
      %3 = hivm.hir.pointer_cast(%c96_i64) : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg1 : memref<16xf16, #hivm.address_space<gm>>) outs(%3 : memref<16xf16, #hivm.address_space<ub>>)
      %4 = hivm.hir.pointer_cast(%c96_i64) : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.vadd ins(%arg6, %3 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>) outs(%4 : memref<16xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.vmul ins(%{{.*}}, %{{.*}} : memref<16xf16, #hivm.address_space<ub>>, f16) outs(%[[YIELD:.*]] : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.vmul ins(%4, %arg4 : memref<16xf16, #hivm.address_space<ub>>, f16) outs(%2 : memref<16xf16, #hivm.address_space<ub>>)
      scf.if %true {
        // CHECK: hivm.hir.store
        hivm.hir.store ins(%2 : memref<16xf16, #hivm.address_space<ub>>) outs(%arg2 : memref<16xf16, #hivm.address_space<gm>>)
      }
      // CHECK-NOT: hivm.hir.copy
      // CHECK: scf.yield %[[YIELD]]
      scf.yield %2 : memref<16xf16, #hivm.address_space<ub>>
    }
    scf.if %true {
      // CHECK: hivm.hir.store
      hivm.hir.store ins(%1 : memref<16xf16, #hivm.address_space<ub>>) outs(%arg3 : memref<16xf16, #hivm.address_space<gm>>)
    }
    return
  }
}

// -----
module {
  func.func @test_parent_loop_nullptr(%arg0: memref<16xf16, #hivm.address_space<gm>>,
                                      %arg1: memref<16xf16, #hivm.address_space<gm>>) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c64_i64 = arith.constant 64 : i64
    // CHECK: hivm.hir.pointer_cast
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>) outs(%0 : memref<16xf16, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.pointer_cast
    %1 = hivm.hir.pointer_cast(%c32_i64, %c64_i64) [] : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%0, %0 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>) outs(%1 : memref<16xf16, #hivm.address_space<ub>>)
    hivm.hir.store ins(%1 : memref<16xf16, #hivm.address_space<ub>>) outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
    return
  }
}

// -----
module {
// CHECK-LABEL: func.func @test_yielded_db_ptr_consumed_by_outer_for(
// The double-buffer ptr is consumed in the inner for and yielded out. Because
// the outer for directly consumes the yielded loop result, the outer for is the
// outermost consumer loop and becomes the multi-buffer anchor.
  func.func @test_yielded_db_ptr_consumed_by_outer_for(
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

    // CHECK: memref.alloca() : memref<1xi64>
    // CHECK: scf.for
    scf.for %arg6 = %c0_i32 to %c16_i32 step %c1_i32 : i32 {
      // The select for %39 is anchored to this outer for.
      // CHECK: arith.select
      %29 = hivm.hir.pointer_cast(%c8192_i64) : memref<1x2048xf16, #hivm.address_space<ub>>
      // CHECK: scf.for
      %31 = scf.for %arg7 = %c0_i32 to %c49152_i32 step %c2048_i32
          iter_args(%arg8 = %29) -> (memref<1x2048xf16, #hivm.address_space<ub>>) : i32 {
        // CHECK-NOT: arith.select
        %39 = hivm.hir.pointer_cast(%c24576_i64, %c53312_i64) :
            memref<1x2048xf16, #hivm.address_space<ub>>
        hivm.hir.load ins(%arg0 : memref<1x2048xf16, #hivm.address_space<gm>>)
                      outs(%39 : memref<1x2048xf16, #hivm.address_space<ub>>)
        scf.yield %39 : memref<1x2048xf16, #hivm.address_space<ub>>
      }

      %43 = hivm.hir.pointer_cast(%c57408_i64) : memref<1x2048xf16, #hivm.address_space<ub>>
      hivm.hir.vadd ins(%31, %31 : memref<1x2048xf16, #hivm.address_space<ub>>,
                                    memref<1x2048xf16, #hivm.address_space<ub>>)
                    outs(%43 : memref<1x2048xf16, #hivm.address_space<ub>>)
      scf.yield
    }
    // CHECK: }
    return
  }
}
