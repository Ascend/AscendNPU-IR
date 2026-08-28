// RUN: bishengir-opt %s --split-input-file --allocate-ascend-shared-memory --convert-triton-ascend-gpu-to-llvm | FileCheck %s

#layout = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [32], order = [0]}>

module attributes {"ttg.enable-bishengir-simt-optimization" = 100 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {

// CHECK: module attributes {{.*}}ttg.shared = 256 : i32
// CHECK-LABEL: llvm.func {{.*}}@scan_addf_fast
// CHECK: ascend_dpx.sync_threads
// CHECK: llvm.load
// CHECK-COUNT-7: ascend_dpx.shfl.idx
// CHECK: llvm.load
// CHECK-COUNT-6: ascend_dpx.shfl.idx
// CHECK-NOT: llvm.load
// CHECK-NOT: ascend_dpx.sync_threads
tt.func public @scan_addf_fast(%arg0: tensor<2048xf32, #layout>) attributes {noinline = false} {
  %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%lhs: f32, %rhs: f32):
    %sum = arith.addf %lhs, %rhs : f32
    tt.scan.return %sum : f32
  }) : (tensor<2048xf32, #layout>) -> tensor<2048xf32, #layout>
  tt.return
}

// CHECK-LABEL: llvm.func {{.*}}@scan_addf_reverse_fast
// CHECK: ascend_dpx.sync_threads
// CHECK: llvm.load
// CHECK-COUNT-7: ascend_dpx.shfl.idx
// CHECK: llvm.load
// CHECK-COUNT-6: ascend_dpx.shfl.idx
// CHECK-NOT: llvm.load
// CHECK-NOT: ascend_dpx.sync_threads
tt.func public @scan_addf_reverse_fast(%arg0: tensor<2048xf32, #layout>) attributes {noinline = false} {
  %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = true}> ({
  ^bb0(%lhs: f32, %rhs: f32):
    %sum = arith.addf %lhs, %rhs : f32
    tt.scan.return %sum : f32
  }) : (tensor<2048xf32, #layout>) -> tensor<2048xf32, #layout>
  tt.return
}

}

// -----

#layout_f16 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [32], order = [0]}>

module attributes {"ttg.enable-bishengir-simt-optimization" = 100 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {

// CHECK: module attributes {{.*}}ttg.shared = 128 : i32
// CHECK-LABEL: llvm.func {{.*}}@scan_addf_f16_fast
// CHECK: ascend_dpx.sync_threads
// CHECK: llvm.load {{.*}} -> f16
// CHECK-COUNT-7: ascend_dpx.shfl.idx
// CHECK: llvm.load {{.*}} -> f16
// CHECK-COUNT-6: ascend_dpx.shfl.idx
// CHECK-NOT: llvm.load
// CHECK-NOT: ascend_dpx.sync_threads
tt.func public @scan_addf_f16_fast(%arg0: tensor<2048xf16, #layout_f16>) attributes {noinline = false} {
  %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%lhs: f16, %rhs: f16):
    %sum = arith.addf %lhs, %rhs : f16
    tt.scan.return %sum : f16
  }) : (tensor<2048xf16, #layout_f16>) -> tensor<2048xf16, #layout_f16>
  tt.return
}

}

// -----

#layout_warp16 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [16], order = [0]}>

module attributes {"ttg.enable-bishengir-simt-optimization" = 100 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {

// CHECK: module attributes {{.*}}ttg.shared = 128 : i32
// CHECK-LABEL: llvm.func {{.*}}@scan_addf_warp16_fast
// CHECK: ascend_dpx.sync_threads
// CHECK: llvm.load
// CHECK: ascend_dpx.shfl.idx
// CHECK: [[W16_H1:%.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: {{%.*}} = llvm.icmp "sge" {{%.*}}, [[W16_H1]] : i32
// CHECK: ascend_dpx.shfl.idx
// CHECK: [[W16_H2:%.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: {{%.*}} = llvm.icmp "sge" {{%.*}}, [[W16_H2]] : i32
// CHECK: ascend_dpx.shfl.idx
// CHECK: [[W16_H4:%.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK-NEXT: {{%.*}} = llvm.icmp "sge" {{%.*}}, [[W16_H4]] : i32
// CHECK: ascend_dpx.shfl.idx
// CHECK: [[W16_H8:%.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK-NEXT: {{%.*}} = llvm.icmp "sge" {{%.*}}, [[W16_H8]] : i32
// One carry shuffle and one chunk-total shuffle.
// CHECK-COUNT-2: ascend_dpx.shfl.idx
// CHECK: llvm.load
// Four Sklansky stages and one carry shuffle for the last chunk.
// CHECK-COUNT-5: ascend_dpx.shfl.idx
// CHECK-NOT: llvm.load
// CHECK-NOT: ascend_dpx.sync_threads
tt.func public @scan_addf_warp16_fast(%arg0: tensor<1024xf32, #layout_warp16>) attributes {noinline = false} {
  %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%lhs: f32, %rhs: f32):
    %sum = arith.addf %lhs, %rhs : f32
    tt.scan.return %sum : f32
  }) : (tensor<1024xf32, #layout_warp16>) -> tensor<1024xf32, #layout_warp16>
  tt.return
}

}

// -----

#layout_warp8 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>

module attributes {"ttg.enable-bishengir-simt-optimization" = 100 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {

// CHECK: module attributes {{.*}}ttg.shared = 64 : i32
// CHECK-LABEL: llvm.func {{.*}}@scan_addf_warp8_fast
// CHECK: ascend_dpx.sync_threads
// CHECK: llvm.load
// CHECK: ascend_dpx.shfl.idx
// CHECK: [[W8_H1:%.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: {{%.*}} = llvm.icmp "sge" {{%.*}}, [[W8_H1]] : i32
// CHECK: ascend_dpx.shfl.idx
// CHECK: [[W8_H2:%.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: {{%.*}} = llvm.icmp "sge" {{%.*}}, [[W8_H2]] : i32
// CHECK: ascend_dpx.shfl.idx
// CHECK: [[W8_H4:%.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK-NEXT: {{%.*}} = llvm.icmp "sge" {{%.*}}, [[W8_H4]] : i32
// One carry shuffle and one chunk-total shuffle.
// CHECK-COUNT-2: ascend_dpx.shfl.idx
// CHECK: llvm.load
// Three Sklansky stages and one carry shuffle for the last chunk.
// CHECK-COUNT-4: ascend_dpx.shfl.idx
// CHECK-NOT: llvm.load
// CHECK-NOT: ascend_dpx.sync_threads
tt.func public @scan_addf_warp8_fast(%arg0: tensor<512xf32, #layout_warp8>) attributes {noinline = false} {
  %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%lhs: f32, %rhs: f32):
    %sum = arith.addf %lhs, %rhs : f32
    tt.scan.return %sum : f32
  }) : (tensor<512xf32, #layout_warp8>) -> tensor<512xf32, #layout_warp8>
  tt.return
}

}
