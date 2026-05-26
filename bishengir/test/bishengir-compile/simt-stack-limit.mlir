// DEFINE: %{compile} = bishengir-compile -enable-lir-compile=true -pure-simt=true \
// DEFINE:   --enable-triton-ir-compile --num-warps=32 --threads-per-warp=32 \
// DEFINE:   -target=Ascend910_9589

// Per the layered design, this file covers only what is exclusive to the
// bishengir-compile boundary: the [DEBUG] log line and that the cmd-arg
// parses both decimal and hex. The full resolution policy (env var,
// default, 0-strict, negative-disable, per-kernel-over-env precedence) is
// owned by Triton-Ascend and tested there.

// [DEBUG] log messages — documented, not asserted. The LDBG output below is
// only emitted when hivmc is built with assertions (NDEBUG undefined), and
// the exact literals are owned by hivmc; gating that reliably from a
// cross-repo lit test is fragile, so we leave the expected behavior here
// for reference instead.
//
// With:
//   %{compile} -simt-stack-limit=1152 -hivmc-args="-debug-only=bishengir-compile" %s
//
// expect to see on stderr:
//   [DEBUG] per-thread stack limit is set to 1152
//   [DEBUG] found TLV variable .AIV_Kernel_Type_spill_kernel in section .ascend.meta.spill_kernel
//   [DEBUG] StackSize = 28

// test cmd flag is accepted as decimal
// RUN: not %{compile} -simt-stack-limit=1 %s 2>&1 | FileCheck %s --check-prefix=DECIMAL
// DECIMAL: [ERROR] SIMT per-thread stack size (28 bytes) of kernel .AIV_Kernel_Type_spill_kernel exceeds limit (1 bytes/thread)

// test cmd flag is accepted as hex
// RUN: not %{compile} -simt-stack-limit=0xa %s 2>&1 | FileCheck %s --check-prefix=HEX
// HEX: [ERROR] SIMT per-thread stack size (28 bytes) of kernel .AIV_Kernel_Type_spill_kernel exceeds limit (10 bytes/thread)

module attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:89", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 32 : i32} {
  tt.func public @spill_kernel(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %z_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %c6_i32 = arith.constant 6 : i32
    %c7_i32 = arith.constant 7 : i32
    %c16_i32 = arith.constant 16 : i32
    %c17_i32 = arith.constant 17 : i32
    %c18_i32 = arith.constant 18 : i32
    %c19_i32 = arith.constant 19 : i32
    %c20_i32 = arith.constant 20 : i32
    %c21_i32 = arith.constant 21 : i32
    %c22_i32 = arith.constant 22 : i32
    %c23_i32 = arith.constant 23 : i32
    %c32_i32 = arith.constant 32 : i32
    %c33_i32 = arith.constant 33 : i32
    %c34_i32 = arith.constant 34 : i32
    %c35_i32 = arith.constant 35 : i32
    %c36_i32 = arith.constant 36 : i32
    %c37_i32 = arith.constant 37 : i32
    %c38_i32 = arith.constant 38 : i32
    %c39_i32 = arith.constant 39 : i32
    %c48_i32 = arith.constant 48 : i32
    %c49_i32 = arith.constant 49 : i32
    %c50_i32 = arith.constant 50 : i32
    %c51_i32 = arith.constant 51 : i32
    %c52_i32 = arith.constant 52 : i32
    %c53_i32 = arith.constant 53 : i32
    %c54_i32 = arith.constant 54 : i32
    %c55_i32 = arith.constant 55 : i32
    %pid = tt.get_program_id x : i32
    %mask = arith.cmpi slt, %pid, %N : i32
    %x = tt.addptr %x_ptr, %pid : !tt.ptr<f32>, i32
    %x_0 = tt.load %x, %mask : !tt.ptr<f32>
    %p = arith.xori %pid, %c1_i32 : i32
    %p_1 = tt.addptr %x_ptr, %p : !tt.ptr<f32>, i32
    %a0 = tt.load %p_1, %mask : !tt.ptr<f32>
    %a0_2 = arith.addf %x_0, %a0 : f32
    %a1 = tt.addptr %p_1, %c1_i32 : !tt.ptr<f32>, i32
    %a1_3 = tt.load %a1, %mask : !tt.ptr<f32>
    %a1_4 = arith.addf %x_0, %a1_3 : f32
    %a2 = tt.addptr %p_1, %c2_i32 : !tt.ptr<f32>, i32
    %a2_5 = tt.load %a2, %mask : !tt.ptr<f32>
    %a2_6 = arith.addf %x_0, %a2_5 : f32
    %a3 = tt.addptr %p_1, %c3_i32 : !tt.ptr<f32>, i32
    %a3_7 = tt.load %a3, %mask : !tt.ptr<f32>
    %a3_8 = arith.addf %x_0, %a3_7 : f32
    %a4 = tt.addptr %p_1, %c4_i32 : !tt.ptr<f32>, i32
    %a4_9 = tt.load %a4, %mask : !tt.ptr<f32>
    %a4_10 = arith.addf %x_0, %a4_9 : f32
    %a5 = tt.addptr %p_1, %c5_i32 : !tt.ptr<f32>, i32
    %a5_11 = tt.load %a5, %mask : !tt.ptr<f32>
    %a5_12 = arith.addf %x_0, %a5_11 : f32
    %a6 = tt.addptr %p_1, %c6_i32 : !tt.ptr<f32>, i32
    %a6_13 = tt.load %a6, %mask : !tt.ptr<f32>
    %a6_14 = arith.addf %x_0, %a6_13 : f32
    %a7 = tt.addptr %p_1, %c7_i32 : !tt.ptr<f32>, i32
    %a7_15 = tt.load %a7, %mask : !tt.ptr<f32>
    %a7_16 = arith.addf %x_0, %a7_15 : f32
    %b0 = tt.addptr %p_1, %c16_i32 : !tt.ptr<f32>, i32
    %b0_17 = tt.load %b0, %mask : !tt.ptr<f32>
    %b0_18 = arith.addf %x_0, %b0_17 : f32
    %b1 = tt.addptr %p_1, %c17_i32 : !tt.ptr<f32>, i32
    %b1_19 = tt.load %b1, %mask : !tt.ptr<f32>
    %b1_20 = arith.addf %x_0, %b1_19 : f32
    %b2 = tt.addptr %p_1, %c18_i32 : !tt.ptr<f32>, i32
    %b2_21 = tt.load %b2, %mask : !tt.ptr<f32>
    %b2_22 = arith.addf %x_0, %b2_21 : f32
    %b3 = tt.addptr %p_1, %c19_i32 : !tt.ptr<f32>, i32
    %b3_23 = tt.load %b3, %mask : !tt.ptr<f32>
    %b3_24 = arith.addf %x_0, %b3_23 : f32
    %b4 = tt.addptr %p_1, %c20_i32 : !tt.ptr<f32>, i32
    %b4_25 = tt.load %b4, %mask : !tt.ptr<f32>
    %b4_26 = arith.addf %x_0, %b4_25 : f32
    %b5 = tt.addptr %p_1, %c21_i32 : !tt.ptr<f32>, i32
    %b5_27 = tt.load %b5, %mask : !tt.ptr<f32>
    %b5_28 = arith.addf %x_0, %b5_27 : f32
    %b6 = tt.addptr %p_1, %c22_i32 : !tt.ptr<f32>, i32
    %b6_29 = tt.load %b6, %mask : !tt.ptr<f32>
    %b6_30 = arith.addf %x_0, %b6_29 : f32
    %b7 = tt.addptr %p_1, %c23_i32 : !tt.ptr<f32>, i32
    %b7_31 = tt.load %b7, %mask : !tt.ptr<f32>
    %b7_32 = arith.addf %x_0, %b7_31 : f32
    %c0 = tt.addptr %p_1, %c32_i32 : !tt.ptr<f32>, i32
    %c0_33 = tt.load %c0, %mask : !tt.ptr<f32>
    %c0_34 = arith.addf %x_0, %c0_33 : f32
    %c1 = tt.addptr %p_1, %c33_i32 : !tt.ptr<f32>, i32
    %c1_35 = tt.load %c1, %mask : !tt.ptr<f32>
    %c1_36 = arith.addf %x_0, %c1_35 : f32
    %c2 = tt.addptr %p_1, %c34_i32 : !tt.ptr<f32>, i32
    %c2_37 = tt.load %c2, %mask : !tt.ptr<f32>
    %c2_38 = arith.addf %x_0, %c2_37 : f32
    %c3 = tt.addptr %p_1, %c35_i32 : !tt.ptr<f32>, i32
    %c3_39 = tt.load %c3, %mask : !tt.ptr<f32>
    %c3_40 = arith.addf %x_0, %c3_39 : f32
    %c4 = tt.addptr %p_1, %c36_i32 : !tt.ptr<f32>, i32
    %c4_41 = tt.load %c4, %mask : !tt.ptr<f32>
    %c4_42 = arith.addf %x_0, %c4_41 : f32
    %c5 = tt.addptr %p_1, %c37_i32 : !tt.ptr<f32>, i32
    %c5_43 = tt.load %c5, %mask : !tt.ptr<f32>
    %c5_44 = arith.addf %x_0, %c5_43 : f32
    %c6 = tt.addptr %p_1, %c38_i32 : !tt.ptr<f32>, i32
    %c6_45 = tt.load %c6, %mask : !tt.ptr<f32>
    %c6_46 = arith.addf %x_0, %c6_45 : f32
    %c7 = tt.addptr %p_1, %c39_i32 : !tt.ptr<f32>, i32
    %c7_47 = tt.load %c7, %mask : !tt.ptr<f32>
    %c7_48 = arith.addf %x_0, %c7_47 : f32
    %d0 = tt.addptr %p_1, %c48_i32 : !tt.ptr<f32>, i32
    %d0_49 = tt.load %d0, %mask : !tt.ptr<f32>
    %d0_50 = arith.addf %x_0, %d0_49 : f32
    %d1 = tt.addptr %p_1, %c49_i32 : !tt.ptr<f32>, i32
    %d1_51 = tt.load %d1, %mask : !tt.ptr<f32>
    %d1_52 = arith.addf %x_0, %d1_51 : f32
    %d2 = tt.addptr %p_1, %c50_i32 : !tt.ptr<f32>, i32
    %d2_53 = tt.load %d2, %mask : !tt.ptr<f32>
    %d2_54 = arith.addf %x_0, %d2_53 : f32
    %d3 = tt.addptr %p_1, %c51_i32 : !tt.ptr<f32>, i32
    %d3_55 = tt.load %d3, %mask : !tt.ptr<f32>
    %d3_56 = arith.addf %x_0, %d3_55 : f32
    %d4 = tt.addptr %p_1, %c52_i32 : !tt.ptr<f32>, i32
    %d4_57 = tt.load %d4, %mask : !tt.ptr<f32>
    %d4_58 = arith.addf %x_0, %d4_57 : f32
    %d5 = tt.addptr %p_1, %c53_i32 : !tt.ptr<f32>, i32
    %d5_59 = tt.load %d5, %mask : !tt.ptr<f32>
    %d5_60 = arith.addf %x_0, %d5_59 : f32
    %d6 = tt.addptr %p_1, %c54_i32 : !tt.ptr<f32>, i32
    %d6_61 = tt.load %d6, %mask : !tt.ptr<f32>
    %d6_62 = arith.addf %x_0, %d6_61 : f32
    %d7 = tt.addptr %p_1, %c55_i32 : !tt.ptr<f32>, i32
    %d7_63 = tt.load %d7, %mask : !tt.ptr<f32>
    %d7_64 = arith.addf %x_0, %d7_63 : f32
    %r = arith.addf %a0_2, %a1_4 : f32
    %r_65 = arith.addf %r, %a2_6 : f32
    %r_66 = arith.addf %r_65, %a3_8 : f32
    %r_67 = arith.addf %r_66, %a4_10 : f32
    %r_68 = arith.addf %r_67, %a5_12 : f32
    %r_69 = arith.addf %r_68, %a6_14 : f32
    %r_70 = arith.addf %r_69, %a7_16 : f32
    %r_71 = arith.addf %r_70, %b0_18 : f32
    %r_72 = arith.addf %r_71, %b1_20 : f32
    %r_73 = arith.addf %r_72, %b2_22 : f32
    %r_74 = arith.addf %r_73, %b3_24 : f32
    %r_75 = arith.addf %r_74, %b4_26 : f32
    %r_76 = arith.addf %r_75, %b5_28 : f32
    %r_77 = arith.addf %r_76, %b6_30 : f32
    %r_78 = arith.addf %r_77, %b7_32 : f32
    %r_79 = arith.addf %r_78, %c0_34 : f32
    %r_80 = arith.addf %r_79, %c1_36 : f32
    %r_81 = arith.addf %r_80, %c2_38 : f32
    %r_82 = arith.addf %r_81, %c3_40 : f32
    %r_83 = arith.addf %r_82, %c4_42 : f32
    %r_84 = arith.addf %r_83, %c5_44 : f32
    %r_85 = arith.addf %r_84, %c6_46 : f32
    %r_86 = arith.addf %r_85, %c7_48 : f32
    %r_87 = arith.addf %r_86, %d0_50 : f32
    %r_88 = arith.addf %r_87, %d1_52 : f32
    %r_89 = arith.addf %r_88, %d2_54 : f32
    %r_90 = arith.addf %r_89, %d3_56 : f32
    %r_91 = arith.addf %r_90, %d4_58 : f32
    %r_92 = arith.addf %r_91, %d5_60 : f32
    %r_93 = arith.addf %r_92, %d6_62 : f32
    %r_94 = arith.addf %r_93, %d7_64 : f32
    %0 = tt.addptr %z_ptr, %pid : !tt.ptr<f32>, i32
    tt.store %0, %r_94, %mask : !tt.ptr<f32>
    tt.return
  }
}
