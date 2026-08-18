// RUN: bishengir-opt -convert-triton-ascend-gpu-to-llvm %s | FileCheck %s
//
// Test that Triton cache modifier CV ("don't cache + volatile") correctly
// lowers to DPX cacheOption=NCA and volatileOption=VOLATILE.
//
// Previously, only CG and isVolatile were checked. The CV modifier was
// falling through to CA (cached) + NONVOLATILE, which is semantically
// incorrect. This test verifies the fix.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>

module attributes {
    "ttg.num-ctas"         = 1 : i32,
    "ttg.num-warps"        = 8 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared             = 4096 : i32,
    ttg.target             = "cuda:80",
    hacc.target            = #hacc.target<"Ascend910_9589">,
    "ttg.enable-bishengir-simt-optimization" = 1111 : i32
} {

  // --- Test 1: CV load -> NCA, VOLATILE ---
  // CV means "don't cache + volatile", so both cache bypass and volatile
  // must be set, even though isVolatile is false.
  // CHECK-LABEL: @load_cv
  // CHECK: ascend_dpx.load
  // CHECK-SAME: cacheOption = <LOADCACHEOPTION_NCA>
  // CHECK-SAME: volatileOption = <VOLATILE>
  tt.func @load_cv(
      %ptr: tensor<256x!tt.ptr<bf16>, #blocked>,
      %out: tensor<256x!tt.ptr<bf16>, #blocked>) {
    %0 = tt.load %ptr cacheModifier = cv : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.store %out, %0 : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  // --- Test 2: CV load with evict_last -> NCA, VOLATILE, LV hint ---
  // Combining CV with eviction policy: both should take effect independently.
  // CHECK-LABEL: @load_cv_evict_last
  // CHECK: ascend_dpx.load
  // CHECK-SAME: cacheModifier = <L2_CACHE_HINT_NORMAL_LV>
  // CHECK-SAME: cacheOption = <LOADCACHEOPTION_NCA>
  // CHECK-SAME: volatileOption = <VOLATILE>
  tt.func @load_cv_evict_last(
      %ptr: tensor<256x!tt.ptr<bf16>, #blocked>,
      %out: tensor<256x!tt.ptr<bf16>, #blocked>) {
    %0 = tt.load %ptr cacheModifier = cv evictionPolicy = evict_last : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.store %out, %0 : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  // --- Test 3: CV store -> NCA ---
  // CV on store should also bypass cache. Store has no volatileOption.
  // CHECK-LABEL: @store_cv
  // CHECK: ascend_dpx.store
  // CHECK-SAME: cacheOption = <LOADCACHEOPTION_NCA>
  tt.func @store_cv(
      %ptr: tensor<256x!tt.ptr<bf16>, #blocked>,
      %val: tensor<256xbf16, #blocked>) {
    tt.store %ptr, %val cacheModifier = cv : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  // --- Test 4: CA load (explicit) -> CA, NONVOLATILE (both elided) ---
  // Ensure CA still maps to cached, as a baseline sanity check.
  // CHECK-LABEL: @load_ca
  // CHECK: ascend_dpx.load
  // CHECK-NOT: cacheOption
  // CHECK-NOT: volatileOption
  // CHECK-SAME: : (!llvm.ptr<1>) -> vector<1xbf16>
  tt.func @load_ca(
      %ptr: tensor<256x!tt.ptr<bf16>, #blocked>,
      %out: tensor<256x!tt.ptr<bf16>, #blocked>) {
    %0 = tt.load %ptr cacheModifier = ca : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.store %out, %0 : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}
