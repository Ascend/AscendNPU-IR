// RUN: bishengir-opt --tritongpu-reorder-instructions %s | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 16], order = [1, 0]}>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, "ttg.enable-bishengir-simt-optimization" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.shared = 122880 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: triton_unk_fused_cat_11
  // CHECK: tt.load
  // CHECK: tt.store
  // CHECK: tt.load
  // CHECK: tt.store
  // CHECK: tt.load
  // CHECK: tt.store
  // CHECK: tt.load
  // CHECK: tt.store
  // CHECK: tt.load
  // CHECK: tt.store
  // CHECK: tt.load
  // CHECK: tt.store
  // CHECK: tt.load
  // CHECK: tt.store
  // CHECK: tt.load
  // CHECK: tt.store
  // CHECK: tt.load
  // CHECK: tt.store
  // CHECK: tt.load
  // CHECK: tt.store
  tt.func public @triton_unk_fused_cat_11(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg10: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg11: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg12: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg13: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg14: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg15: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg16: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg17: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg18: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg19: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg20: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg24: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg25: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %c480_i32 = arith.constant 480 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<472> : tensor<1x512xi32, #blocked>
    %cst_0 = arith.constant dense<464> : tensor<1x512xi32, #blocked>
    %cst_1 = arith.constant dense<336> : tensor<1x512xi32, #blocked>
    %cst_2 = arith.constant dense<208> : tensor<1x512xi32, #blocked>
    %cst_3 = arith.constant dense<192> : tensor<1x512xi32, #blocked>
    %cst_4 = arith.constant dense<176> : tensor<1x512xi32, #blocked>
    %cst_5 = arith.constant dense<112> : tensor<1x512xi32, #blocked>
    %cst_6 = arith.constant dense<96> : tensor<1x512xi32, #blocked>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<1x512xf32, #blocked>
    %cst_8 = arith.constant dense<8> : tensor<1x512xi32, #blocked>
    %cst_9 = arith.constant dense<128> : tensor<1x512xi32, #blocked>
    %cst_10 = arith.constant dense<16> : tensor<1x512xi32, #blocked>
    %cst_11 = arith.constant dense<32> : tensor<1x512xi32, #blocked>
    %cst_12 = arith.constant dense<64> : tensor<1x512xi32, #blocked>
    %c4_i32 = arith.constant 4 : i32
    %cst_13 = arith.constant dense<0> : tensor<1x1xi64, #blocked>
    %cst_14 = arith.constant dense<289094> : tensor<1x1xi64, #blocked>
    %cst_15 = arith.constant dense<64> : tensor<1x1xi64, #blocked>
    %cst_16 = arith.constant dense<98> : tensor<1x1xi64, #blocked>
    %cst_17 = arith.constant dense<32> : tensor<1x1xi64, #blocked>
    %cst_18 = arith.constant dense<14> : tensor<1x1xi64, #blocked>
    %cst_19 = arith.constant dense<16> : tensor<1x1xi64, #blocked>
    %cst_20 = arith.constant dense<3> : tensor<1x1xi64, #blocked>
    %cst_21 = arith.constant dense<8> : tensor<1x1xi64, #blocked>
    %cst_22 = arith.constant dense<4> : tensor<1x1xi64, #blocked>
    %cst_23 = arith.constant dense<128> : tensor<1x1xi64, #blocked>
    %cst_24 = arith.constant dense<5> : tensor<1x1xi64, #blocked>
    %cst_25 = arith.constant dense<1820039> : tensor<1x1xi64, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = arith.addi %1, %c4_i32 : i32
    %3 = arith.minsi %2, %arg21 : i32
    %4 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x512xi32, #blocked>
    %6 = arith.cmpi slt, %5, %cst_12 : tensor<1x512xi32, #blocked>
    %7 = arith.cmpi slt, %5, %cst_11 : tensor<1x512xi32, #blocked>
    %8 = arith.cmpi slt, %5, %cst_10 : tensor<1x512xi32, #blocked>
    %9 = arith.cmpi slt, %5, %cst_9 : tensor<1x512xi32, #blocked>
    %10 = arith.cmpi slt, %5, %cst_8 : tensor<1x512xi32, #blocked>
    %11 = arith.extsi %5 : tensor<1x512xi32, #blocked> to tensor<1x512xi64, #blocked>
    %12 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x512x!tt.ptr<f32>, #blocked>
    %13 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<1x512x!tt.ptr<f32>, #blocked>
    %14 = tt.splat %arg5 : !tt.ptr<f32> -> tensor<1x512x!tt.ptr<f32>, #blocked>
    %15 = tt.splat %arg7 : !tt.ptr<f32> -> tensor<1x512x!tt.ptr<f32>, #blocked>
    %16 = tt.splat %arg9 : !tt.ptr<f32> -> tensor<1x512x!tt.ptr<f32>, #blocked>
    %17 = tt.splat %arg11 : !tt.ptr<f32> -> tensor<1x512x!tt.ptr<f32>, #blocked>
    %18 = tt.splat %arg13 : !tt.ptr<f32> -> tensor<1x512x!tt.ptr<f32>, #blocked>
    %19 = tt.splat %arg15 : !tt.ptr<f32> -> tensor<1x512x!tt.ptr<f32>, #blocked>
    %20 = tt.splat %arg17 : !tt.ptr<f32> -> tensor<1x512x!tt.ptr<f32>, #blocked>
    %21 = tt.splat %arg19 : !tt.ptr<f32> -> tensor<1x512x!tt.ptr<f32>, #blocked>
    %22 = arith.addi %5, %cst_12 : tensor<1x512xi32, #blocked>
    %23 = arith.addi %5, %cst_6 : tensor<1x512xi32, #blocked>
    %24 = arith.addi %5, %cst_5 : tensor<1x512xi32, #blocked>
    %25 = arith.addi %5, %cst_4 : tensor<1x512xi32, #blocked>
    %26 = arith.addi %5, %cst_3 : tensor<1x512xi32, #blocked>
    %27 = arith.addi %5, %cst_2 : tensor<1x512xi32, #blocked>
    %28 = arith.addi %5, %cst_1 : tensor<1x512xi32, #blocked>
    %29 = arith.addi %5, %cst_0 : tensor<1x512xi32, #blocked>
    %30 = arith.addi %5, %cst : tensor<1x512xi32, #blocked>
    scf.for %arg26 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %31 = arith.addi %1, %arg26 : i32
      %32 = arith.cmpi slt, %31, %3 : i32
      %33 = tt.addptr %arg0, %31 : !tt.ptr<i64>, i32
      %34 = tt.splat %33 : !tt.ptr<i64> -> tensor<1x1x!tt.ptr<i64>, #blocked>
      %35 = tt.splat %32 : i1 -> tensor<1x1xi1, #blocked>
      %36 = tt.load %34, %35, %cst_13 : tensor<1x1x!tt.ptr<i64>, #blocked>
      %37 = tt.addptr %arg2, %31 : !tt.ptr<i64>, i32
      %38 = tt.splat %37 : !tt.ptr<i64> -> tensor<1x1x!tt.ptr<i64>, #blocked>
      %39 = tt.load %38, %35, %cst_13 : tensor<1x1x!tt.ptr<i64>, #blocked>
      %40 = tt.addptr %arg4, %31 : !tt.ptr<i64>, i32
      %41 = tt.splat %40 : !tt.ptr<i64> -> tensor<1x1x!tt.ptr<i64>, #blocked>
      %42 = tt.load %41, %35, %cst_13 : tensor<1x1x!tt.ptr<i64>, #blocked>
      %43 = tt.addptr %arg6, %31 : !tt.ptr<i64>, i32
      %44 = tt.splat %43 : !tt.ptr<i64> -> tensor<1x1x!tt.ptr<i64>, #blocked>
      %45 = tt.load %44, %35, %cst_13 : tensor<1x1x!tt.ptr<i64>, #blocked>
      %46 = tt.addptr %arg8, %31 : !tt.ptr<i64>, i32
      %47 = tt.splat %46 : !tt.ptr<i64> -> tensor<1x1x!tt.ptr<i64>, #blocked>
      %48 = tt.load %47, %35, %cst_13 : tensor<1x1x!tt.ptr<i64>, #blocked>
      %49 = tt.addptr %arg10, %31 : !tt.ptr<i64>, i32
      %50 = tt.splat %49 : !tt.ptr<i64> -> tensor<1x1x!tt.ptr<i64>, #blocked>
      %51 = tt.load %50, %35, %cst_13 : tensor<1x1x!tt.ptr<i64>, #blocked>
      %52 = tt.addptr %arg12, %31 : !tt.ptr<i64>, i32
      %53 = tt.splat %52 : !tt.ptr<i64> -> tensor<1x1x!tt.ptr<i64>, #blocked>
      %54 = tt.load %53, %35, %cst_13 : tensor<1x1x!tt.ptr<i64>, #blocked>
      %55 = tt.addptr %arg14, %31 : !tt.ptr<i64>, i32
      %56 = tt.splat %55 : !tt.ptr<i64> -> tensor<1x1x!tt.ptr<i64>, #blocked>
      %57 = tt.load %56, %35, %cst_13 : tensor<1x1x!tt.ptr<i64>, #blocked>
      %58 = tt.addptr %arg16, %31 : !tt.ptr<i64>, i32
      %59 = tt.splat %58 : !tt.ptr<i64> -> tensor<1x1x!tt.ptr<i64>, #blocked>
      %60 = tt.load %59, %35, %cst_13 : tensor<1x1x!tt.ptr<i64>, #blocked>
      %61 = tt.addptr %arg18, %31 : !tt.ptr<i64>, i32
      %62 = tt.splat %61 : !tt.ptr<i64> -> tensor<1x1x!tt.ptr<i64>, #blocked>
      %63 = tt.load %62, %35, %cst_13 : tensor<1x1x!tt.ptr<i64>, #blocked>
      %64 = arith.addi %36, %cst_14 : tensor<1x1xi64, #blocked>
      %65 = arith.cmpi slt, %36, %cst_13 : tensor<1x1xi64, #blocked>
      %66 = arith.select %65, %64, %36 : tensor<1x1xi1, #blocked>, tensor<1x1xi64, #blocked>
      %67 = arith.muli %66, %cst_15 : tensor<1x1xi64, #blocked>
      %68 = tt.broadcast %67 : tensor<1x1xi64, #blocked> -> tensor<1x512xi64, #blocked>
      %69 = arith.addi %11, %68 : tensor<1x512xi64, #blocked>
      %70 = tt.addptr %12, %69 : tensor<1x512x!tt.ptr<f32>, #blocked>, tensor<1x512xi64, #blocked>
      %71 = tt.splat %32 : i1 -> tensor<1x512xi1, #blocked>
      %72 = arith.andi %71, %6 : tensor<1x512xi1, #blocked>
      %73 = tt.load %70, %72, %cst_7 : tensor<1x512x!tt.ptr<f32>, #blocked>
      %74 = arith.addi %39, %cst_16 : tensor<1x1xi64, #blocked>
      %75 = arith.cmpi slt, %39, %cst_13 : tensor<1x1xi64, #blocked>
      %76 = arith.select %75, %74, %39 : tensor<1x1xi1, #blocked>, tensor<1x1xi64, #blocked>
      %77 = arith.muli %76, %cst_17 : tensor<1x1xi64, #blocked>
      %78 = tt.broadcast %77 : tensor<1x1xi64, #blocked> -> tensor<1x512xi64, #blocked>
      %79 = arith.addi %11, %78 : tensor<1x512xi64, #blocked>
      %80 = tt.addptr %13, %79 : tensor<1x512x!tt.ptr<f32>, #blocked>, tensor<1x512xi64, #blocked>
      %81 = arith.andi %71, %7 : tensor<1x512xi1, #blocked>
      %82 = tt.load %80, %81, %cst_7 : tensor<1x512x!tt.ptr<f32>, #blocked>
      %83 = arith.addi %42, %cst_18 : tensor<1x1xi64, #blocked>
      %84 = arith.cmpi slt, %42, %cst_13 : tensor<1x1xi64, #blocked>
      %85 = arith.select %84, %83, %42 : tensor<1x1xi1, #blocked>, tensor<1x1xi64, #blocked>
      %86 = arith.muli %85, %cst_19 : tensor<1x1xi64, #blocked>
      %87 = tt.broadcast %86 : tensor<1x1xi64, #blocked> -> tensor<1x512xi64, #blocked>
      %88 = arith.addi %11, %87 : tensor<1x512xi64, #blocked>
      %89 = tt.addptr %14, %88 : tensor<1x512x!tt.ptr<f32>, #blocked>, tensor<1x512xi64, #blocked>
      %90 = arith.andi %71, %8 : tensor<1x512xi1, #blocked>
      %91 = tt.load %89, %90, %cst_7 : tensor<1x512x!tt.ptr<f32>, #blocked>
      %92 = arith.addi %45, %cst_20 : tensor<1x1xi64, #blocked>
      %93 = arith.cmpi slt, %45, %cst_13 : tensor<1x1xi64, #blocked>
      %94 = arith.select %93, %92, %45 : tensor<1x1xi1, #blocked>, tensor<1x1xi64, #blocked>
      %95 = arith.muli %94, %cst_15 : tensor<1x1xi64, #blocked>
      %96 = tt.broadcast %95 : tensor<1x1xi64, #blocked> -> tensor<1x512xi64, #blocked>
      %97 = arith.addi %11, %96 : tensor<1x512xi64, #blocked>
      %98 = tt.addptr %15, %97 : tensor<1x512x!tt.ptr<f32>, #blocked>, tensor<1x512xi64, #blocked>
      %99 = tt.load %98, %72, %cst_7 : tensor<1x512x!tt.ptr<f32>, #blocked>
      %100 = arith.addi %48, %cst_21 : tensor<1x1xi64, #blocked>
      %101 = arith.cmpi slt, %48, %cst_13 : tensor<1x1xi64, #blocked>
      %102 = arith.select %101, %100, %48 : tensor<1x1xi1, #blocked>, tensor<1x1xi64, #blocked>
      %103 = arith.muli %102, %cst_19 : tensor<1x1xi64, #blocked>
      %104 = tt.broadcast %103 : tensor<1x1xi64, #blocked> -> tensor<1x512xi64, #blocked>
      %105 = arith.addi %11, %104 : tensor<1x512xi64, #blocked>
      %106 = tt.addptr %16, %105 : tensor<1x512x!tt.ptr<f32>, #blocked>, tensor<1x512xi64, #blocked>
      %107 = tt.load %106, %90, %cst_7 : tensor<1x512x!tt.ptr<f32>, #blocked>
      %108 = arith.addi %51, %cst_22 : tensor<1x1xi64, #blocked>
      %109 = arith.cmpi slt, %51, %cst_13 : tensor<1x1xi64, #blocked>
      %110 = arith.select %109, %108, %51 : tensor<1x1xi1, #blocked>, tensor<1x1xi64, #blocked>
      %111 = arith.muli %110, %cst_19 : tensor<1x1xi64, #blocked>
      %112 = tt.broadcast %111 : tensor<1x1xi64, #blocked> -> tensor<1x512xi64, #blocked>
      %113 = arith.addi %11, %112 : tensor<1x512xi64, #blocked>
      %114 = tt.addptr %17, %113 : tensor<1x512x!tt.ptr<f32>, #blocked>, tensor<1x512xi64, #blocked>
      %115 = tt.load %114, %90, %cst_7 : tensor<1x512x!tt.ptr<f32>, #blocked>
      %116 = arith.addi %54, %cst_22 : tensor<1x1xi64, #blocked>
      %117 = arith.cmpi slt, %54, %cst_13 : tensor<1x1xi64, #blocked>
      %118 = arith.select %117, %116, %54 : tensor<1x1xi1, #blocked>, tensor<1x1xi64, #blocked>
      %119 = arith.muli %118, %cst_23 : tensor<1x1xi64, #blocked>
      %120 = tt.broadcast %119 : tensor<1x1xi64, #blocked> -> tensor<1x512xi64, #blocked>
      %121 = arith.addi %11, %120 : tensor<1x512xi64, #blocked>
      %122 = tt.addptr %18, %121 : tensor<1x512x!tt.ptr<f32>, #blocked>, tensor<1x512xi64, #blocked>
      %123 = arith.andi %71, %9 : tensor<1x512xi1, #blocked>
      %124 = tt.load %122, %123, %cst_7 : tensor<1x512x!tt.ptr<f32>, #blocked>
      %125 = arith.addi %57, %cst_20 : tensor<1x1xi64, #blocked>
      %126 = arith.cmpi slt, %57, %cst_13 : tensor<1x1xi64, #blocked>
      %127 = arith.select %126, %125, %57 : tensor<1x1xi1, #blocked>, tensor<1x1xi64, #blocked>
      %128 = arith.muli %127, %cst_23 : tensor<1x1xi64, #blocked>
      %129 = tt.broadcast %128 : tensor<1x1xi64, #blocked> -> tensor<1x512xi64, #blocked>
      %130 = arith.addi %11, %129 : tensor<1x512xi64, #blocked>
      %131 = tt.addptr %19, %130 : tensor<1x512x!tt.ptr<f32>, #blocked>, tensor<1x512xi64, #blocked>
      %132 = tt.load %131, %123, %cst_7 : tensor<1x512x!tt.ptr<f32>, #blocked>
      %133 = arith.addi %60, %cst_24 : tensor<1x1xi64, #blocked>
      %134 = arith.cmpi slt, %60, %cst_13 : tensor<1x1xi64, #blocked>
      %135 = arith.select %134, %133, %60 : tensor<1x1xi1, #blocked>, tensor<1x1xi64, #blocked>
      %136 = arith.muli %135, %cst_21 : tensor<1x1xi64, #blocked>
      %137 = tt.broadcast %136 : tensor<1x1xi64, #blocked> -> tensor<1x512xi64, #blocked>
      %138 = arith.addi %11, %137 : tensor<1x512xi64, #blocked>
      %139 = tt.addptr %20, %138 : tensor<1x512x!tt.ptr<f32>, #blocked>, tensor<1x512xi64, #blocked>
      %140 = arith.andi %71, %10 : tensor<1x512xi1, #blocked>
      %141 = tt.load %139, %140, %cst_7 : tensor<1x512x!tt.ptr<f32>, #blocked>
      %142 = arith.addi %63, %cst_25 : tensor<1x1xi64, #blocked>
      %143 = arith.cmpi slt, %63, %cst_13 : tensor<1x1xi64, #blocked>
      %144 = arith.select %143, %142, %63 : tensor<1x1xi1, #blocked>, tensor<1x1xi64, #blocked>
      %145 = arith.muli %144, %cst_21 : tensor<1x1xi64, #blocked>
      %146 = tt.broadcast %145 : tensor<1x1xi64, #blocked> -> tensor<1x512xi64, #blocked>
      %147 = arith.addi %11, %146 : tensor<1x512xi64, #blocked>
      %148 = tt.addptr %21, %147 : tensor<1x512x!tt.ptr<f32>, #blocked>, tensor<1x512xi64, #blocked>
      %149 = tt.load %148, %140, %cst_7 : tensor<1x512x!tt.ptr<f32>, #blocked>
      %150 = arith.muli %31, %c480_i32 : i32
      %151 = tt.addptr %arg20, %150 : !tt.ptr<f32>, i32
      %152 = tt.splat %151 : !tt.ptr<f32> -> tensor<1x512x!tt.ptr<f32>, #blocked>
      %153 = tt.addptr %152, %5 : tensor<1x512x!tt.ptr<f32>, #blocked>, tensor<1x512xi32, #blocked>
      tt.store %153, %73, %72 : tensor<1x512x!tt.ptr<f32>, #blocked>
      %154 = tt.addptr %152, %22 : tensor<1x512x!tt.ptr<f32>, #blocked>, tensor<1x512xi32, #blocked>
      tt.store %154, %82, %81 : tensor<1x512x!tt.ptr<f32>, #blocked>
      %155 = tt.addptr %152, %23 : tensor<1x512x!tt.ptr<f32>, #blocked>, tensor<1x512xi32, #blocked>
      tt.store %155, %91, %90 : tensor<1x512x!tt.ptr<f32>, #blocked>
      %156 = tt.addptr %152, %24 : tensor<1x512x!tt.ptr<f32>, #blocked>, tensor<1x512xi32, #blocked>
      tt.store %156, %99, %72 : tensor<1x512x!tt.ptr<f32>, #blocked>
      %157 = tt.addptr %152, %25 : tensor<1x512x!tt.ptr<f32>, #blocked>, tensor<1x512xi32, #blocked>
      tt.store %157, %107, %90 : tensor<1x512x!tt.ptr<f32>, #blocked>
      %158 = tt.addptr %152, %26 : tensor<1x512x!tt.ptr<f32>, #blocked>, tensor<1x512xi32, #blocked>
      tt.store %158, %115, %90 : tensor<1x512x!tt.ptr<f32>, #blocked>
      %159 = tt.addptr %152, %27 : tensor<1x512x!tt.ptr<f32>, #blocked>, tensor<1x512xi32, #blocked>
      tt.store %159, %124, %123 : tensor<1x512x!tt.ptr<f32>, #blocked>
      %160 = tt.addptr %152, %28 : tensor<1x512x!tt.ptr<f32>, #blocked>, tensor<1x512xi32, #blocked>
      tt.store %160, %132, %123 : tensor<1x512x!tt.ptr<f32>, #blocked>
      %161 = tt.addptr %152, %29 : tensor<1x512x!tt.ptr<f32>, #blocked>, tensor<1x512xi32, #blocked>
      tt.store %161, %141, %140 : tensor<1x512x!tt.ptr<f32>, #blocked>
      %162 = tt.addptr %152, %30 : tensor<1x512x!tt.ptr<f32>, #blocked>, tensor<1x512xi32, #blocked>
      tt.store %162, %149, %140 : tensor<1x512x!tt.ptr<f32>, #blocked>
    }
    tt.return
  }
  
  // CHECK: tt.func public @while_kernel
  tt.func public @while_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst_0 = arith.constant dense<128> : tensor<128x1xi32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x1xf16>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
    %2:3 = scf.while (%arg3 = %1, %arg4 = %c0_i32, %arg5 = %cst_1) : (tensor<128x1x!tt.ptr<f16>>, i32, tensor<128x1xf16>) -> (tensor<128x1x!tt.ptr<f16>>, i32, tensor<128x1xf16>) {
      %3 = arith.cmpi slt, %arg4, %arg2 : i32
      scf.condition(%3) %arg3, %arg4, %arg5 : tensor<128x1x!tt.ptr<f16>>, i32, tensor<128x1xf16>
    } do {
    ^bb0(%arg3: tensor<128x1x!tt.ptr<f16>>, %arg4: i32, %arg5 : tensor<128x1xf16>):
      %4 = tt.load %arg3 : tensor<128x1x!tt.ptr<f16>>
      %5 = arith.addi %arg4, %c128_i32 : i32
      %6 = tt.addptr %arg3, %cst_0 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      scf.yield %6, %5, %4 : tensor<128x1x!tt.ptr<f16>>, i32, tensor<128x1xf16>
    }
    %7 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
    tt.store %7, %2#2 : tensor<128x1x!tt.ptr<f16>>
    tt.return
  }

  // CHECK: tt.func public @for_kernel
  tt.func public @for_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst_0 = arith.constant dense<128> : tensor<128x1xi32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x1xf16>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
    %2:2 = scf.for %arg3 = %c0_i32 to %arg2 step %c128_i32 iter_args(%arg4 = %1, %arg5 = %cst_1) -> (tensor<128x1x!tt.ptr<f16>>, tensor<128x1xf16>)  : i32 {
      %4 = tt.load %arg4 : tensor<128x1x!tt.ptr<f16>>
      %5 = tt.addptr %arg4, %cst_0 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      scf.yield %5, %4 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xf16>
    }
    %6 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
    tt.store %6, %2#1 : tensor<128x1x!tt.ptr<f16>>
    tt.return
  }
}
