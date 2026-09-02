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
