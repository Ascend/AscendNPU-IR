// REQUIRES: enable-lir-compile
// RUN: mkdir -p %t.on %t.off
// RUN: echo 'source_filename = "support"' > %t.ll
// RUN: bishengir-compile %s --target=Ascend960DT_968B7 --use-pipeline-version=2 --enable-hfusion-compile --enable-triton-kernel-compile --disable-ffts --custom-aic-number=32 --custom-aiv-number=64 --enable-auto-blockify-loop --link-aicore-bitcode=%t.ll --save-temps=%t.on -o %t.on/kernel
// RUN: FileCheck %s --check-prefix=LOOP --input-file=%t.on/kernel.ll
// RUN: bishengir-compile %s --target=Ascend960DT_968B7 --use-pipeline-version=2 --enable-hfusion-compile --enable-triton-kernel-compile --disable-ffts --custom-aic-number=32 --custom-aiv-number=64 --link-aicore-bitcode=%t.ll --save-temps=%t.off -o %t.off/kernel
// RUN: FileCheck %s --check-prefix=DIRECT --implicit-check-not="phi i32" --input-file=%t.off/kernel.ll

// Both Triton program IDs follow the induction variable. This AIV kernel
// takes its physical stride from the 64 vector cores.
// LOOP-LABEL: define {{.*}}@clear_block(
// LOOP: @llvm.hivm.SBITSET0({{.*}}i64 60)
// LOOP: %[[BLOCK:[0-9]+]] = call i64 @llvm.hivm.GET.BLOCK.IDX()
// LOOP: %[[START:[0-9]+]] = trunc i64 %[[BLOCK]] to i32
// LOOP: %[[ID:[0-9]+]] = phi i32 {{.*}}%[[START]],
// LOOP: icmp slt i32 %[[ID]],
// LOOP: srem i32 %[[ID]], %[[NX:[0-9]+]]
// LOOP: %[[ROW:[0-9]+]] = sdiv i32 %[[ID]], %[[NX]]
// LOOP: srem i32 %[[ROW]],
// LOOP: store i32 0
// LOOP: add nsw i32 %[[ID]], 64
// LOOP: @llvm.hivm.SBITSET1({{.*}}i64 60)
// DIRECT-LABEL: define {{.*}}@clear_block(
// DIRECT: %[[BLOCK:[0-9]+]] = call i64 @llvm.hivm.GET.BLOCK.IDX()
// DIRECT: %[[ID:[0-9]+]] = trunc i64 %[[BLOCK]] to i32
// DIRECT: srem i32 %[[ID]],
// DIRECT: sdiv i32 %[[ID]],
// DIRECT: store i32 0
module {
  func.func @clear_block(%lock: memref<?xi8>, %workspace: memref<?xi8>, %states: memref<?xi32>, %nx: i32, %ny: i32, %nz: i32, %x: i32, %y: i32, %z: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %xidx = arith.index_cast %x : i32 to index
    %yidx = arith.index_cast %y : i32 to index
    %width = arith.constant 4096 : index
    %tile = arith.muli %yidx, %width : index
    %index = arith.addi %xidx, %tile : index
    %zero = arith.constant 0 : i32
    memref.store %zero, %states[%index] : memref<?xi32>
    return
  }
}
