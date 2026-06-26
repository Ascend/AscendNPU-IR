// REQUIRES: enable-lir-compile
// RUN: bishengir-compile %s --target=Ascend910_9589 --enable-triton-ir-compile --pure-simt --enable-sink-dpx-load=false --enable-auto-blockify-loop --num-warps=1 --threads-per-warp=32 --enable-simt-fast-div=false -o %t.o --mlir-print-ir-after=convert-ascend-dpx-to-hivmregbaseintrins 2>&1 | FileCheck %s --implicit-check-not=_mlir_ciface_simt_div_magic_shift_uint32_t --implicit-check-not=_mlir_ciface_simt_div_magic_mul_uint32_t --implicit-check-not=_mlir_ciface_simt_umulhi_uint32_t

// The kernel uses tt.get_program_id so SIMTAutoBlockify introduces div/rem to
// reconstruct logical program ids. With --enable-simt-fast-div=false, the later
// DPX div optimization must not rewrite those divisions into magic/shift paths.

// CHECK: IR Dump After ConvertAscendDPXToHIVMRegbaseIntrins (convert-ascend-dpx-to-hivmregbaseintrins)

// CHECK-LABEL: llvm.func @simt_autoblockify_fast_div_guard(
// CHECK: %[[BLOCKIDX:.*]] = "hivm.intr.hivm.GET.BLOCK.IDX"() : () -> i64
// CHECK: %[[GRIDX:.*]] = llvm.zext %arg1 : i32 to i64
// CHECK: %[[GRIDY:.*]] = llvm.zext %arg2 : i32 to i64
// CHECK: llvm.urem %[[BLOCKIDX]], %[[GRIDX]] : i64
// CHECK: llvm.udiv %[[BLOCKIDX]], %[[GRIDX]] : i64
// CHECK: llvm.urem %{{.*}}, %[[GRIDY]] : i64
// CHECK: llvm.udiv %{{.*}}, %[[GRIDY]] : i64
// CHECK: %[[SHM:.*]] = llvm.inttoptr %{{.*}} {hivm.shared_memory} : i64 to !llvm.ptr<6>
// CHECK: hivm_regbaseintrins.intrins.launch_func @simt_autoblockify_fast_div_guard_vf_simt threads in

// CHECK-LABEL: llvm.func @simt_autoblockify_fast_div_guard_vf_simt(
// CHECK: %[[PHYSICAL:.*]] = llvm.mlir.constant(64 : i32) : i32
// CHECK: %[[HWBLOCK:.*]] = hivm_regbaseintrins.block_idx
// CHECK: %[[CHUNKDIV:.*]] = llvm.sdiv %{{.*}}, %[[PHYSICAL]] : i32
// CHECK: %[[START:.*]] = llvm.mul %[[HWBLOCK]], %{{.*}} : i32
// CHECK: %[[DIVX:.*]] = llvm.sdiv %{{.*}}, %arg1 : i32
// CHECK: %[[PIDX:.*]] = llvm.urem %{{.*}}, %arg1 : i32
// CHECK: %[[PIDY:.*]] = llvm.urem %[[DIVX]], %arg2 : i32
// CHECK: %[[PIDZ:.*]] = llvm.sdiv %[[DIVX]], %arg2 : i32

module {
  tt.func public @simt_autoblockify_fast_div_guard(
      %arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32})
      attributes {noinline = false} {
    %pid_x = tt.get_program_id x : i32
    %pid_y = tt.get_program_id y : i32
    %pid_z = tt.get_program_id z : i32
    %sum0 = arith.addi %pid_x, %pid_y : i32
    %sum1 = arith.addi %sum0, %pid_z : i32
    %ptr = tt.addptr %arg0, %pid_x : !tt.ptr<i32>, i32
    tt.store %ptr, %sum1 : !tt.ptr<i32>
    tt.return
  }
}
