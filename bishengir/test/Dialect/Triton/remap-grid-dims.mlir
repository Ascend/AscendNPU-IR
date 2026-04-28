// RUN: bishengir-opt %s -triton-remap | FileCheck %s

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: llvm.func @foo(%[[X:.*]]: i32 {gpu.block = #gpu.block<x>}, %[[Y:.*]]: i32 {gpu.block = #gpu.block<y>}, %[[Z:.*]]: i32 {gpu.block = #gpu.block<z>}
  // CHECK: %[[BLOCK_IDX:.*]] = "hivm.intr.hivm.GET.BLOCK.IDX"() : () -> i64
  // CHECK: %[[X_I64:.*]] = llvm.zext %[[X]] : i32 to i64
  // CHECK: %[[Y_I64:.*]] = llvm.zext %[[Y]] : i32 to i64
  // CHECK: %[[Z_I64:.*]] = llvm.zext %[[Z]] : i32 to i64
  // CHECK: %[[X_ID:.*]] = llvm.urem %[[BLOCK_IDX]], %[[X_I64]]  : i64
  // CHECK: %[[TEMP:.*]] = llvm.udiv %[[BLOCK_IDX]], %[[X_I64]]  : i64
  // CHECK: %[[Y_ID:.*]] = llvm.urem %[[TEMP]], %[[Y_I64]]  : i64
  // CHECK: %[[Z_ID:.*]] = llvm.udiv %[[TEMP]], %[[Y_I64]]  : i64
  // hivm_regbaseintrins.intrins.launch_func

  // CHECK: llvm.func @foo_vf_simt
  llvm.func @foo(%x: i32 {gpu.block = #gpu.block<x>},
                 %y: i32 {gpu.block = #gpu.block<y>},
                 %z: i32 {gpu.block = #gpu.block<z>})
  attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>} {
    llvm.return
  }
}
