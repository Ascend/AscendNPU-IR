// RUN: bishengir-opt %s -allow-unregistered-dialect -triton-global-kernel-args-to-hivm-op -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @test_args_to_hivm_op(
// CHECK: %[[A:.*]]: memref<*xf16>, %[[B:.*]]: memref<*xf16>, %[[C:.*]]: memref<*xf16>,
// CHECK-SAME: %[[ProgNumX:.*]]: i32, %[[ProgNumY:.*]]: i32, %[[ProgNumZ:.*]]: i32)
// CHECK: %[[BLOCK_IDX:.+]] = hivm.hir.get_block_idx -> i64
// CHECK: %[[CAST_OP_ID:.+]] = arith.trunci %[[BLOCK_IDX]] : i64 to i32
// CHECK: %[[ACCSHAPE_Z:.+]] = arith.constant 1 : i32
// CHECK: %[[TOTALINDEX_Z:.+]] = arith.divsi %[[CAST_OP_ID]], %[[ACCSHAPE_Z]]
// CHECK: %[[ProgZ_ID:.+]] = arith.remsi %[[TOTALINDEX_Z]], %[[ProgNumZ]]
// CHECK: %[[ACCSHAPE_Y:.+]] = arith.muli %[[ACCSHAPE_Z]], %[[ProgNumZ]]
// CHECK: %[[TOTALINDEX_Y:.+]] = arith.divsi %[[CAST_OP_ID]], %[[ACCSHAPE_Y]]
// CHECK: %[[ProgY_ID:.+]] = arith.remsi %[[TOTALINDEX_Y]], %[[ProgNumY]]
// CHECK: %[[ACCSHAPE_X:.+]] = arith.muli %[[ACCSHAPE_Y]], %[[ProgNumY]]
// CHECK: %[[TOTALINDEX_X:.+]] = arith.divsi %[[CAST_OP_ID]], %[[ACCSHAPE_X]]
// CHECK: %[[ProgX_ID:.+]] = arith.remsi %[[TOTALINDEX_X]], %[[ProgNumX]]
// CHECK: arith.muli %[[ProgX_ID]]
// CHECK: arith.muli %[[ProgY_ID]]
// CHECK: arith.addi %[[ProgZ_ID]]
module {
  func.func @test_args_to_hivm_op(%arg0: memref<*xf16>, %arg1: memref<*xf16>, %arg2: memref<*xf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // cur_idx = tl.program_id(0) * tl.num_programs(1) * tl.num_programs(2)
    //           + tl.program_id(1) * tl.num_programs(2)
    //           + tl.program_id(2)
    %0 = arith.muli %arg6, %arg4 : i32
    %1 = arith.muli %0, %arg5 : i32
    %2 = arith.muli %arg7, %arg5 : i32
    %3 = arith.addi %1, %2 : i32
    %4 = arith.addi %arg8, %3 : i32

    "some_op"(%4) : (i32) -> ()
    return
  }
}

// -----
module {
  // expected-error@+1 {{arguments program id or program num are missing}}
  func.func @test_args_lack(%arg0: memref<*xf16>, %arg1: memref<*xf16>, %arg2: memref<*xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c0_i32 = arith.constant 10 : i32
    scf.for %arg18 = %c0 to %c0_i32 step %c1  : i32 {
       "some_op"(%arg18) : (i32) -> ()
        scf.yield
      }
    return
  }
}

// -----
module {
  // expected-error@+1 {{incompatible types of arguments program id or program num}}
  func.func @test_invalid_type(%arg0: memref<*xf16>, %arg1: memref<*xf16>, %arg2: memref<*xf16>, %arg4: memref<*xf16>, %arg5: memref<*xf16>, %arg6: memref<*xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c0_i32 = arith.constant 10 : i32
    scf.for %arg18 = %c0 to %c0_i32 step %c1  : i32 {
       "some_op"(%arg18) : (i32) -> ()
        scf.yield
      }
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_args_for_scf_if
module {
  // CHECK: %arg2: memref<?xf32>, %[[ProgNumX:.*]]: i32, %[[ProgNumY:.*]]: i32, %[[ProgNumZ:.*]]: i32)
  func.func @test_args_for_scf_if(%arg0: memref<?xf32>, %arg1: memref<?xf32> , %arg2: memref<?xf32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: %[[BLOCK_IDX:.+]] = hivm.hir.get_block_idx -> i64
    // CHECK: %[[CAST_OP_ID:.+]] = arith.trunci %[[BLOCK_IDX]] : i64 to i32
    // CHECK: %[[ACCSHAPE_Z:.+]] = arith.constant 1 : i32
    // CHECK: %[[TOTALINDEX_Z:.+]] = arith.divsi %[[CAST_OP_ID]], %[[ACCSHAPE_Z]]
    // CHECK: %[[ProgZ_ID:.+]] = arith.remsi %[[TOTALINDEX_Z]], %[[ProgNumZ]]
    // CHECK: %[[ACCSHAPE_Y:.+]] = arith.muli %[[ACCSHAPE_Z]], %[[ProgNumZ]]
    // CHECK: %[[TOTALINDEX_Y:.+]] = arith.divsi %[[CAST_OP_ID]], %[[ACCSHAPE_Y]]
    // CHECK: %[[ProgY_ID:.+]] = arith.remsi %[[TOTALINDEX_Y]], %[[ProgNumY]]
    // CHECK: %[[ACCSHAPE_X:.+]] = arith.muli %[[ACCSHAPE_Y]], %[[ProgNumY]]
    // CHECK: %[[TOTALINDEX_X:.+]] = arith.divsi %[[CAST_OP_ID]], %[[ACCSHAPE_X]]
    // CHECK: %[[ProgX_ID:.+]] = arith.remsi %[[TOTALINDEX_X]], %[[ProgNumX]]
    // CHECK: %[[CST0:.+]] = arith.constant 16 : i32
    // CHECK: %[[CST1:.+]] = arith.constant 51302 : i32
    // CHECK: %[[COND:.+]] = arith.cmpi slt, %[[ProgX_ID]], %[[CST0]] : i32
    // CHECK: %[[ARG0:.+]] = scf.if %[[COND]] -> (i32) {
    // CHECK: %[[ARG1:.+]] = arith.muli %[[ProgX_ID]], %[[CST1]] : i32
    // CHECK: scf.yield %[[ARG1]] : i32
    // CHECK: } else {
    // CHECK: %[[ARG2:.+]] = arith.muli %[[ProgX_ID]], %[[CST1]] : i32
    // CHECK: %[[ARG3:.+]] = arith.addi %[[ARG2]], %[[CST0]] : i32
    // CHECK: scf.yield %[[ARG3]] : i32
    // CHECK: }
    // CHECK: return
    %c16_i32 = arith.constant 16 : i32
    %c51302_i32 = arith.constant 51302 : i32
    %0 = arith.cmpi slt, %arg6, %c16_i32 : i32
    %1 = scf.if %0 -> (i32) {
        %3 = arith.muli %arg6, %c51302_i32 : i32
        scf.yield %3 : i32
    } else {
        %3 = arith.muli %arg6, %c51302_i32 : i32
        %4 = arith.addi %3, %c16_i32 : i32
        scf.yield %4 : i32
    }
    return
  }
}
