// RUN: bishengir-opt --allocate-ascend-shared-memory --convert-triton-ascend-gpu-to-llvm %s | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: bishengir-opt --allocate-ascend-shared-memory --convert-triton-ascend-gpu-to-llvm="enable-a5-bank-aware-swizzle=true" %s | FileCheck %s --check-prefixes=CHECK,A5

// Keep both distributed layouts fixed. Their different warp mappings force
// conversion through shared memory rather than a warp shuffle. Under the
// generic 32x4B model the f32 load has two distinct 8B words in each Ascend
// bank; the Ascend bank model permutes shared-memory bank bits to remove this.
#src = #ttg.linear<{register = [[1]], lane = [[4], [8], [16], [32], [64]], warp = [[2]], block = []}>
#dst = #ttg.linear<{register = [[2]], lane = [[1], [8], [16], [32], [64]], warp = [[4]], block = []}>

// CHECK: ttg.shared = 512 : i32
module attributes {
    "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 2 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.target = "cuda:80",
    hacc.target = #hacc.target<"Ascend950PR_957c">,
    "ttg.simt-optimization-mode" = 1111 : i32
} {
  // CHECK-LABEL: @convert_layout_f32_bank_aware
  // Swap the two low bank-coordinate bases. Both scalar stores and loads,
  // the 512B scratch allocation and the block barrier are preserved.
  // DEFAULT: llvm.mlir.constant(63 : i32)
  // A5: llvm.mlir.constant(60 : i32)
  // CHECK: ascend_dpx.store {{.*}} : <3>, vector<1xi32>
  // DEFAULT: llvm.mlir.constant(260 : i32)
  // A5: llvm.mlir.constant(264 : i32)
  // CHECK: ascend_dpx.store {{.*}} : <3>, vector<1xi32>
  // CHECK-NEXT: ascend_dpx.sync_threads
  // DEFAULT: llvm.mlir.constant(30 : i32)
  // A5: llvm.mlir.constant(28 : i32)
  // DEFAULT: llvm.mlir.constant(260 : i32)
  // A5: llvm.mlir.constant(264 : i32)
  // CHECK: ascend_dpx.load {{.*}} : (!llvm.ptr<3>) -> i32
  // CHECK: ascend_dpx.load {{.*}} : (!llvm.ptr<3>) -> i32
  // CHECK-NOT: ascend_dpx.sync_threads
  tt.func public @convert_layout_f32_bank_aware(
      %data: tensor<128xf32, #src>, %out: !tt.ptr<f32>) {
    %converted = ttg.convert_layout %data
      : tensor<128xf32, #src> -> tensor<128xf32, #dst>
    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #dst>
    %base = tt.splat %out : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #dst>
    %ptrs = tt.addptr %base, %offsets : tensor<128x!tt.ptr<f32>, #dst>, tensor<128xi32, #dst>
    tt.store %ptrs, %converted : tensor<128x!tt.ptr<f32>, #dst>
    tt.return
  }

  // The target currently opts in only for f32-sized elements.
  // CHECK-LABEL: @convert_layout_f16_default
  // CHECK: ascend_dpx.store {{.*}} : <3>, vector<1xi16>
  // CHECK: llvm.mlir.constant(132 : i32)
  // CHECK: ascend_dpx.store {{.*}} : <3>, vector<1xi16>
  // CHECK-NEXT: ascend_dpx.sync_threads
  // CHECK: ascend_dpx.load {{.*}} : (!llvm.ptr<3>) -> vector<2xi16>
  // CHECK-NOT: ascend_dpx.load {{.*}} : (!llvm.ptr<3>)
  tt.func public @convert_layout_f16_default(
      %data: tensor<128xf16, #src>, %out: !tt.ptr<f16>) {
    %converted = ttg.convert_layout %data
      : tensor<128xf16, #src> -> tensor<128xf16, #dst>
    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #dst>
    %base = tt.splat %out : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #dst>
    %ptrs = tt.addptr %base, %offsets : tensor<128x!tt.ptr<f16>, #dst>, tensor<128xi32, #dst>
    tt.store %ptrs, %converted : tensor<128x!tt.ptr<f16>, #dst>
    tt.return
  }
}
