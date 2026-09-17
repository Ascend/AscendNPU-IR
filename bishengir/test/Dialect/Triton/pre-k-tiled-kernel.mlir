// RUN: bishengir-opt -tile-dot-loads='k-tile-size=2' %s | FileCheck %s

// Check fix for canonical tensor-of-ptrs style tiling/decomposition
// during staging when dot is pre tiled in the kernel itself

// CHECK-LABEL: @pre_tiled_dot_kernel
// CHECK: [[SCRATCH_ARG:%arg[0-9]+]]: !tt.ptr<f32, 6>
// CHECK: arith.cmpi slt
// CHECK: scf.for [[PRE_KTILE_IDX:%arg[0-9]+]]
// CHECK: [[SCRATCH_ARG_SPLAT:%[0-9]+]] = tt.splat [[SCRATCH_ARG:%arg[0-9]+]]
// CHECK: [[SCRATCH_PTRS:%[0-9]+]] = tt.addptr [[SCRATCH_ARG_SPLAT:%[0-9]+]]
// CHECK: tt.store [[SCRATCH_PTRS:%[0-9]+]]
// CHECK: tensor<1x32x!tt.ptr<f32, 6>>
// CHECK: [[TILE_RANGE:%[0-9]+]] = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// CHECK: [[PRE_KTILE_IDX_SPLAT:%[0-9]+]] = tt.splat [[PRE_KTILE_IDX:%arg[0-9]+]] : i32 -> tensor<2xi32>
// CHECK: scf.for
// CHECK: [[FIRST_ADDI:%[0-9]+]] = arith.addi {{%[0-9]+}}, [[TILE_RANGE:%[0-9]+]]
// CHECK: [[SECOND_ADDI:%[0-9]+]] = arith.addi [[FIRST_ADDI:%[0-9]+]], [[PRE_KTILE_IDX_SPLAT:%[0-9]+]]
// CHECK: [[FINAL_OUT:%[0-9]+]] = tt.expand_dims [[SECOND_ADDI:%[0-9]+]] {axis = 1 : i32}
// CHECK: arith.muli [[FINAL_OUT:%[0-9]+]]
// CHECK: arith.cmpi slt, [[FINAL_OUT:%[0-9]+]]
// CHECK: tt.dot
// CHECK-NOT: tt.dot
// CHECK: scf.yield
// CHECK: scf.yield

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, "ttg.enable-bishengir-simt-optimization" = 900101 : i32, "ttg.enable-global-scratch-allocation" = false} {
  tt.func public @pre_tiled_dot_kernel(%arg0: f32 {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x32xf32>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %pid_y = tt.get_program_id y : i32
    %pid_y_off = arith.muli %pid_y, %c32_i32 : i32
    %full_range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %pid_y_off_splat = tt.splat %pid_y_off : i32 -> tensor<32xi32>
    %row_offsets = arith.addi %pid_y_off_splat, %full_range : tensor<32xi32>
    %col_offset_multiplier = tt.splat %arg4 : i32 -> tensor<32x1xi32>
    %ptr_splat = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>>
    %row_offsets_2D = tt.expand_dims %row_offsets {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %row_offsets_full = tt.broadcast %row_offsets_2D : tensor<1x32xi32> -> tensor<32x32xi32>
    %col_lim = tt.splat %arg3 : i32 -> tensor<32x1xi32>
    %row_lim = tt.splat %arg2 : i32 -> tensor<1x32xi32>
    %row_check = arith.cmpi slt, %row_offsets_2D, %row_lim : tensor<1x32xi32>
    %row_check_full = tt.broadcast %row_check : tensor<1x32xi1> -> tensor<32x32xi1>
    %pre_tiled_dot = scf.for %arg5 = %c0_i32 to %arg3 step %c32_i32 iter_args(%arg13 = %cst_0) -> (tensor<1x32xf32>)  : i32 {
        %ktile_idx = tt.splat %arg5 : i32 -> tensor<32xi32>
        %ktile_range = arith.addi %ktile_idx, %full_range : tensor<32xi32>
        %non_load_a = tt.splat %arg0 : f32 -> tensor<1x32xf32>
        %ktile_range_2D = tt.expand_dims %ktile_range {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
        %col_offsets = arith.muli %ktile_range_2D, %col_offset_multiplier : tensor<32x1xi32>
        %col_ptrs = tt.addptr %ptr_splat, %col_offsets : tensor<32x1x!tt.ptr<f32>>, tensor<32x1xi32>
        %col_ptrs_full = tt.broadcast %col_ptrs : tensor<32x1x!tt.ptr<f32>> -> tensor<32x32x!tt.ptr<f32>>
        %full_ptrs = tt.addptr %col_ptrs_full, %row_offsets_full : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
        %col_check = arith.cmpi slt, %ktile_range_2D, %col_lim : tensor<32x1xi32>
        %col_check_full = tt.broadcast %col_check : tensor<32x1xi1> -> tensor<32x32xi1>
        %mask = arith.andi %col_check_full, %row_check_full : tensor<32x32xi1>
        %load_b = tt.load %full_ptrs, %mask, %cst : tensor<32x32x!tt.ptr<f32>>
        %dot = tt.dot %non_load_a, %load_b, %arg13 : tensor<1x32xf32> * tensor<32x32xf32> -> tensor<1x32xf32>
        scf.yield %dot : tensor<1x32xf32>
    }
    tt.return
  }
}
