// RUN: bishengir-translate --mlir-to-llvmir -split-input-file %s | FileCheck %s

// CHECK-LABEL: define void @test_get_block_idx() {
// CHECK: %[[CALL:.*]] = call i64 @llvm.hivm.GET.BLOCK.IDX()
llvm.func @test_get_block_idx() {
  %0 = "hivm.intr.hivm.GET.BLOCK.IDX"() : () -> i64
  llvm.return
}

// -----

// CHECK-LABEL: define void @test_get_block_num() {
// CHECK: %[[CALL:.*]] = call i64 @llvm.hivm.GET.BLOCK.NUM()
llvm.func @test_get_block_num() {
  %0 = "hivm.intr.hivm.GET.BLOCK.NUM"() : () -> i64
  llvm.return
}

// -----

// CHECK-LABEL: define void @test_get_sub_block_idx(
// CHECK: %[[CALL:.*]] = call i64 @llvm.hivm.GET.SUBBLOCKID()
llvm.func @test_get_sub_block_idx() {
  %0 = "hivm.intr.hivm.GET.SUBBLOCKID"() : () -> i64
  llvm.return
}

// -----

// CHECK-LABEL: define void @test_get_sub_block_num(
// CHECK: %[[CALL:.*]] = call i64 @llvm.hivm.GET.SUBBLOCKDIM()
llvm.func @test_get_sub_block_num() {
  %0 = "hivm.intr.hivm.GET.SUBBLOCKDIM"() : () -> i64
  llvm.return
}

// -----

// CHECK-LABEL: define void @test_pipe_barrier() {
// CHECK: call void @llvm.hivm.BARRIER(i64 6)
llvm.func @test_pipe_barrier() {
  "hivm.intr.hivm.BARRIER"() <{pipe = 6 : i64}> : () -> ()
  llvm.return
}

// -----

// CHECK-LABEL: define void @test_set_ffts_base_addr
// CHECK: call void @llvm.hivm.SET.FFTS.BASE.ADDR(i64 %0)
llvm.func @test_set_ffts_base_addr(%arg : i64) {
  "hivm.intr.hivm.SET.FFTS.BASE.ADDR"(%arg) : (i64) -> ()
  llvm.return
}

// -----

// CHECK-LABEL: define void @test_set_mask_norm_decomposition
// CHECK: %[[CTRL:.*]] = call i64 @llvm.hivm.GET.CTRL()
// CHECK: %[[SBITSET:.*]] = call i64 @llvm.hivm.SBITSET0(i64 %[[CTRL]], i64 56)
// CHECK: call void @llvm.hivm.SET.CTRL(i64 %[[SBITSET]])
llvm.func @test_set_mask_norm_decomposition() {
  %0 = "hivm.intr.hivm.GET.CTRL"() : () -> i64
  %1 = llvm.mlir.constant(56 : i64) : i64
  %2 = "hivm.intr.hivm.SBITSET0"(%0, %1) : (i64, i64) -> i64
  "hivm.intr.hivm.SET.CTRL"(%2) : (i64) -> ()
  llvm.return
}

// -----

// Standalone entries use the same core-kind markers on C220 and C310. The C310
// 0:1 and 1:0 functions normalized by SplitMixKernel have this representation
// and need the markers when compiled with MIX enabled.
// CHECK-LABEL: define dso_local void @c310_vector_only()
// CHECK-SAME: #[[$AIV_ATTR:[0-9]+]] {
llvm.func @c310_vector_only() attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">} {
  llvm.return
}

// CHECK-LABEL: define dso_local void @c310_cube_only()
// CHECK-SAME: #[[$AIC_ATTR:[0-9]+]] {
llvm.func @c310_cube_only() attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">} {
  llvm.return
}

// Ordinary C220 entries use the same standalone markers as C310 entries.
// CHECK-LABEL: define dso_local void @c220_vector_only()
// CHECK-SAME: #[[$C220_AIV_ATTR:[0-9]+]] {
llvm.func @c220_vector_only() attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c220">} {
  llvm.return
}

// A C220 0:1 vector half retains MIX metadata; AIV does not imply standalone.
// CHECK-LABEL: define dso_local void @c220_mix_aiv()
// CHECK-SAME: #[[$C220_MIX_ATTR:[0-9]+]]
llvm.func @c220_mix_aiv() attributes {hacc.entry, hivm.core_ratio = #hivm.core_ratio<0, 1>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c220">} {
  llvm.return
}

// A non-entry vector helper does not need a kernel marker.
// CHECK-LABEL: define void @c310_vector_helper()
// CHECK-SAME: #[[$HELPER_ATTR:[0-9]+]] {
llvm.func @c310_vector_helper() attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">} {
  llvm.return
}

// An AIC MIX half without an explicit ratio must also remain unmarked.
// CHECK-LABEL: define dso_local void @c310_default_mix_aic()
// CHECK-SAME: #[[$HELPER_ATTR]] {
llvm.func @c310_default_mix_aic() attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">} {
  llvm.return
}

// CHECK-DAG: attributes #[[$AIV_ATTR]] = { {{.*}}"vec-only-kernel"{{.*}} }
// CHECK-DAG: attributes #[[$AIC_ATTR]] = { {{.*}}"cube-only-kernel"{{.*}} }
// CHECK-DAG: attributes #[[$C220_AIV_ATTR]] = { "target-cpu"="dav-c220" "target-features"="+dav-c220" "vec-only-kernel" }
// CHECK-DAG: attributes #[[$C220_MIX_ATTR]] = { "mix-kernel-core-ratio" "target-cpu"="dav-c220" "target-features"="+dav-c220" }
// CHECK-DAG: attributes #[[$HELPER_ATTR]] = { "target-cpu"="dav-c310" "target-features"="+dav-c310" }
// CHECK-DAG: !{ptr @c220_mix_aiv, !"mix-kernel-core-ratio-M", i32 0}
// CHECK-DAG: !{ptr @c220_mix_aiv, !"mix-kernel-core-ratio-N", i32 1}

// -----

// CHECK-LABEL: define void @test_dcci
// CHECK: call void @llvm.hivm.DCCI.DST(ptr addrspace(1) null, i64 1, i64 2)
llvm.func @test_dcci() {
  %0 = llvm.mlir.constant(2 : i64) : i64
  %1 = llvm.mlir.constant(1 : i64) : i64
  %2 = llvm.mlir.constant(0 : i64) : i64
  %3 = llvm.inttoptr %2 : i64 to !llvm.ptr<1>
  "hivm.intr.hivm.DCCI.DST"(%3, %1, %0) : (!llvm.ptr<1>, i64, i64) -> ()
  llvm.return
}
