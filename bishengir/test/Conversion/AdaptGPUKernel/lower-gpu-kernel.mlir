// RUN: bishengir-opt %s --adapt-gpu-kernel -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, "ttg.enable-bishengir-simt-optimization" = 100 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.shared = 128 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
    llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
    // CHECK-LABEL: @adapt_gpu_kernel_test
    // CHECK: %[[BLOCKIDX:.*]] = "hivm.intr.hivm.GET.BLOCK.IDX"() : () -> i64
    // CHECK-NEXT: %[[DIMX:.*]] = llvm.zext %[[ARG2:.*]] : i32 to i64
    // CHECK-NEXT: %[[DIMY:.*]] = llvm.zext %[[ARG3:.*]] : i32 to i64
    // CHECK-NEXT: %[[DIMZ:.*]] = llvm.zext %[[ARG4:.*]] : i32 to i64
    // CHECK-NEXT: %[[IDX:.*]] = llvm.urem %[[BLOCKIDX]], %[[DIMX]]
    // CHECK-NEXT: %[[TMP1:.*]] = llvm.udiv %[[BLOCKIDX]], %[[DIMX]]
    // CHECK-NEXT: %[[IDY:.*]] = llvm.urem %[[TMP1]], %[[DIMY]]
    // CHECK-NEXT: %[[IDZ:.*]] = llvm.udiv %[[TMP1]], %[[DIMY]]
    // CHECK: launch_func @adapt_gpu_kernel_test_vf_simt
    // CHECK-SAME: %[[DIMX]]
    // CHECK-SAME: %[[DIMY]]
    // CHECK-SAME: %[[DIMZ]]
    // CHECK-SAME: %[[IDX]]
    // CHECK-SAME: %[[IDY]]
    // CHECK-SAME: %[[IDZ]]

    // ---

    // CHECK-LABEL: llvm.func @adapt_gpu_kernel_test_vf_simt
    // CHECK-SAME: %arg0
    // CHECK-SAME: %arg1
    // CHECK-SAME: %arg2
    // CHECK-SAME: %arg3
    // CHECK-SAME: %arg4
    // CHECK-SAME: %arg5
    llvm.func @adapt_gpu_kernel_test(%arg0: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg1: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}, %arg5: !llvm.ptr<1>, %arg6: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 64>} {
        // CHECK: %arg3
        %1 = ascend_dpx.block_idx_x
        // CHECK: %arg4
        %2 = ascend_dpx.block_idx_y
        // CHECK: %arg5
        %3 = ascend_dpx.block_idx_z
        // CHECK: %arg0
        %4 = ascend_dpx.grid_dim_x
        // CHECK: %arg1
        %5 = ascend_dpx.grid_dim_y
        // CHECK: %arg2
        %6 = ascend_dpx.grid_dim_z
        %7 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
        ascend_dpx.store %7, %4 : !llvm.ptr<3>, i32
        llvm.return
    }
}
