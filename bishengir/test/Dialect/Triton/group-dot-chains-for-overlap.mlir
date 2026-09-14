// RUN: bishengir-opt --group-dot-chains-for-overlap %s | FileCheck %s
//
// Input is the IR immediately before GroupDotChainsForOverlap in
// dot-dump-4-w2.mlir. The two dots form a direct C dependency chain.

// CHECK-LABEL: tt.func public @two_dot_c_chain
// CHECK: %22 = tt.dot %17, %20, %cst {bishengir.dot.c_grouped_for_overlap, bishengir.dot.group_id = 0 : i64} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK: %23 = tt.dot %17, %21, %22 {bishengir.dot.c_grouped_for_overlap, bishengir.dot.group_id = 0 : i64} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, "ttg.enable-bishengir-simt-optimization" = 900100 : i32, "ttg.enable-global-scratch-allocation" = false} {
  tt.func public @two_dot_c_chain(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg6: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) {
    %0 = arith.muli %arg5, %arg6 : i32
    %1 = arith.muli %arg4, %0 : i32
    %2 = gpu.linear_block_id
    %3 = arith.index_cast %2 : index to i32
    %c64_i32 = arith.constant 64 : i32
    %4 = arith.ceildivui %1, %c64_i32 : i32
    %5 = arith.muli %3, %4 : i32
    %6 = arith.addi %5, %4 : i32
    %7 = arith.minui %6, %1 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %arg7 = %5 to %7 step %c1_i32 : i32 {
      %8 = arith.divui %arg7, %arg4 : i32
      %9 = arith.remui %arg7, %arg4 : i32
      %10 = arith.remui %8, %arg5 : i32
      %11 = arith.divui %8, %arg5 : i32
      %c0_i32 = arith.constant 0 : i32
      %c16_i32 = arith.constant 16 : i32
      %c16_i64 = arith.constant 16 : i64
      %c1_i64 = arith.constant 1 : i64
      %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
      %12 = arith.muli %9, %c16_i32 : i32
      %13 = tt.make_tensor_ptr %arg0, [%c16_i64, %c16_i64], [%c16_i64, %c1_i64], [%12, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %14 = tt.make_tensor_ptr %arg1, [%c16_i64, %c16_i64], [%c16_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %15 = tt.make_tensor_ptr %arg2, [%c16_i64, %c16_i64], [%c16_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %16 = tt.make_tensor_ptr %arg3, [%c16_i64, %c16_i64], [%c16_i64, %c1_i64], [%12, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %17 = tt.load %13 : !tt.ptr<tensor<16x16xf32>>
      %18 = tt.load %14 : !tt.ptr<tensor<16x16xf32>>
      %19 = tt.load %15 : !tt.ptr<tensor<16x16xf32>>
      %20 = arith.addf %18, %cst : tensor<16x16xf32>
      %21 = arith.addf %19, %cst : tensor<16x16xf32>
      %22 = tt.dot %17, %20, %cst : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %23 = tt.dot %17, %21, %22 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      tt.store %16, %23 : !tt.ptr<tensor<16x16xf32>>
    }
    tt.return
  }
}
