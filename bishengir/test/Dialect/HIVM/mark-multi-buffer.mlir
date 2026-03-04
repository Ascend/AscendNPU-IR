// RUN: bishengir-opt -allow-unregistered-dialect %s             \
// RUN:   -pass-pipeline="builtin.module(                        \
// RUN:     func.func(hivm-mark-multi-buffer{enable-auto=true}),cse)" \
// RUN:   -split-input-file -verify-diagnostics | FileCheck %s

// -----
// CHECK-LABEL: func.func @test_mark_multi_buffer(
func.func @test_mark_multi_buffer(%d : index, %in : memref<8xf32, #hivm.address_space<gm>>, %out : memref<8xf32, #hivm.address_space<gm>>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %tmp4 = memref.alloca() : memref<8xf32, #hivm.address_space<ub>>
  // CHECK: memref.alloca
  scf.for %i0 = %c0 to %c16 step %c4 {
    %tmp1 = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // CHECK: %[[alloc:.*]] = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // CHECK-NOT: annotation.mark
    %tmp2 = memref.alloca() : memref<8xf32, #hivm.address_space<ub>>
    // CHECK: %[[alloca_0:.*]] = memref.alloca() : memref<8xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark %alloca_0 {hivm.multi_buffer = 2 : i32} : memref<8xf32, #hivm.address_space<ub>>
    %tmp3 = memref.alloca(%d) : memref<?xf32, #hivm.address_space<ub>>
    // CHECK: %[[alloca_1:.*]] = memref.alloca(%arg0) : memref<?xf32, #hivm.address_space<ub>>
    // CHECK-NOT: annotation.mark
    %tmp_l0c = memref.alloca() : memref<8xf32, #hivm.address_space<cc>>
    // CHECK-NOT: annotation.mark
    "some_use"(%tmp1) : (memref<8xf32, #hivm.address_space<ub>>) -> ()
    hivm.hir.load ins(%in : memref<8xf32, #hivm.address_space<gm>>) outs(%tmp2 : memref<8xf32, #hivm.address_space<ub>>)
    "some_use"(%tmp3) : (memref<?xf32, #hivm.address_space<ub>>) -> ()
    "some_use"(%tmp_l0c) : (memref<8xf32, #hivm.address_space<cc>>) -> ()
    hivm.hir.store ins(%tmp4 : memref<8xf32, #hivm.address_space<ub>>) outs(%out : memref<8xf32, #hivm.address_space<gm>>)
  }
  return
}

// -----
module {
  func.func @test_for_yield(
      %arg0: memref<1x2048xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32})
      attributes {global_kernel = "local", hacc.entry = "", hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {

    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c49152_i32 = arith.constant 49152 : i32

    %29 = memref.alloc() : memref<1x2048xf16, #hivm.address_space<ub>>
    scf.for %arg8 = %c0_i32 to %c49152_i32 step %c2048_i32 iter_args(%arg9 = %29) -> (memref<1x2048xf16, #hivm.address_space<ub>>)  : i32 {
      %39 = memref.alloc() : memref<1x2048xf16, #hivm.address_space<ub>>
      // CHECK-NOT: annotation.mark %{{.*}} {hivm.multi_buffer = 2 : i32}
      hivm.hir.load ins(%arg0 : memref<1x2048xf16, #hivm.address_space<gm>>) outs(%39 : memref<1x2048xf16, #hivm.address_space<ub>>)

      scf.yield %39 : memref<1x2048xf16, #hivm.address_space<ub>>
    }

    return
  }
}

// -----
module {
  func.func @test_2for_yield(
      %arg0: memref<1x2048xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32})
      attributes {global_kernel = "local", hacc.entry = "", hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {

    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c49152_i32 = arith.constant 49152 : i32

    scf.for %arg7 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
      %29 = memref.alloc() : memref<1x2048xf16, #hivm.address_space<ub>>

      %31 = scf.for %arg8 = %c0_i32 to %c49152_i32 step %c2048_i32 iter_args(%arg9 = %29) -> (memref<1x2048xf16, #hivm.address_space<ub>>)  : i32 {
        %39 = memref.alloc() : memref<1x2048xf16, #hivm.address_space<ub>>
        // CHECK: annotation.mark %{{.*}} {hivm.multi_buffer = 2 : i32}
        hivm.hir.load ins(%arg0 : memref<1x2048xf16, #hivm.address_space<gm>>) outs(%39 : memref<1x2048xf16, #hivm.address_space<ub>>)

        scf.yield %39 : memref<1x2048xf16, #hivm.address_space<ub>>
      }
    }

    return
  }
}

// -----
module {
  func.func @test_mark_workspace(
      %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
      %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
      %arg2: memref<64x16xf32>)
      attributes {global_kernel = "local", hacc.entry = "", hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %1 = tensor.empty() : tensor<16x16xf32>
    %2 = tensor.empty() : tensor<16x16xf32>
    scf.for %arg3 = %c0_i32 to %c4_i32 step %c1_i32 : i32 {
      %3 = tensor.empty() : tensor<16x16xf32>
      %4 = hivm.hir.mmadL1 ins(%1, %2, %true, %c16, %c16, %c16 : tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index) outs(%3 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %5 = memref_ext.alloc_workspace() from %arg1 : from memref<?xi8> to memref<16x16xf32>
      // CHECK: annotation.mark %{{.*}} {hivm.multi_buffer = 2 : i32}
      %6 = bufferization.to_tensor %5 restrict writable : memref<16x16xf32>
      %7 = hivm.hir.fixpipe {enable_nz2nd} ins(%4 : tensor<16x16xf32>) outs(%6 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %8 = tensor.empty() : tensor<16x16xf32>
      %9 = hivm.hir.load ins(%7 : tensor<16x16xf32>) outs(%8 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %10 = tensor.empty() : tensor<16x16xf32>
      %11 = tensor.empty() : tensor<16x16xf32>
      %12 = hivm.hir.vadd ins(%9, %10 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%11 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %13 = arith.index_cast %arg3 : i32 to index
      %14 = arith.muli %13, %c16 : index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [16, 16], strides: [16, 1] : memref<64x16xf32> to memref<16x16xf32, strided<[16, 1], offset: ?>>
      hivm.hir.store ins(%12 : tensor<16x16xf32>) outs(%reinterpret_cast : memref<16x16xf32, strided<[16, 1], offset: ?>>)
    }
    return
  }
}

// -----
module {
  func.func @test_3for_yield(
      %arg0: memref<1x2048xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32})
      attributes {global_kernel = "local", hacc.entry = "", hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {

    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c49152_i32 = arith.constant 49152 : i32

    %29 = memref.alloc() : memref<1x2048xf16, #hivm.address_space<ub>>
    scf.for %arg7 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg9 = %29) -> (memref<1x2048xf16, #hivm.address_space<ub>>) : i32 {

      %31:2 = scf.for %arg8 = %c0_i32 to %c49152_i32 step %c2048_i32 iter_args(%arg10 = %29, %arg11 = %29) -> (memref<1x2048xf16, #hivm.address_space<ub>>, memref<1x2048xf16, #hivm.address_space<ub>>)  : i32 {
        %39 = memref.alloc() : memref<1x2048xf16, #hivm.address_space<ub>>
        // CHECK-NOT: annotation.mark %{{.*}} {hivm.multi_buffer = 2 : i32}
        %40 = memref.alloc() : memref<1x2048xf16, #hivm.address_space<ub>>
        hivm.hir.load ins(%arg0 : memref<1x2048xf16, #hivm.address_space<gm>>) outs(%39 : memref<1x2048xf16, #hivm.address_space<ub>>)

        scf.yield %40, %39 : memref<1x2048xf16, #hivm.address_space<ub>>, memref<1x2048xf16, #hivm.address_space<ub>>
      }

      scf.yield %31#1 : memref<1x2048xf16, #hivm.address_space<ub>>
    }

    return
  }
}