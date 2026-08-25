// RUN: bishengir-opt -convert-triton-ascend-gpu-to-llvm %s | FileCheck %s
//
// Test that Triton cache modifier CG and isVolatile correctly lower to
// DPX cacheOption=NCA and volatileOption=VOLATILE.
//
// This covers the original support added for CG and isVolatile:
//   - CG load  -> cacheOption NCA, volatileOption NONVOLATILE
//   - volatile load -> cacheOption NCA, volatileOption VOLATILE
//   - CG store -> cacheOption NCA
//   - default load/store -> cacheOption CA (elided), volatileOption NONVOLATILE (elided)

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

  // --- Test 1: default load (no modifiers) -> CA, NONVOLATILE (both elided) ---
  // CHECK-LABEL: @load_default
  // CHECK: ascend_dpx.load
  // CHECK-NOT: cacheOption
  // CHECK-NOT: volatileOption
  // CHECK-SAME: : (!llvm.ptr<1>) -> vector<1xbf16>
  tt.func @load_default(
      %ptr: tensor<256x!tt.ptr<bf16>, #blocked>,
      %out: tensor<256x!tt.ptr<bf16>, #blocked>) {
    %0 = tt.load %ptr : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.store %out, %0 : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  // --- Test 2: CG load -> NCA, NONVOLATILE (volatile elided) ---
  // CHECK-LABEL: @load_cg
  // CHECK: ascend_dpx.load
  // CHECK-SAME: cacheOption = <LOADCACHEOPTION_NCA>
  // CHECK-NOT: volatileOption
  // CHECK-SAME: : (!llvm.ptr<1>) -> vector<1xbf16>
  tt.func @load_cg(
      %ptr: tensor<256x!tt.ptr<bf16>, #blocked>,
      %out: tensor<256x!tt.ptr<bf16>, #blocked>) {
    %0 = tt.load %ptr cacheModifier = cg : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.store %out, %0 : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  // --- Test 3: volatile load -> NCA, VOLATILE ---
  // CHECK-LABEL: @load_volatile
  // CHECK: ascend_dpx.load
  // CHECK-SAME: cacheOption = <LOADCACHEOPTION_NCA>
  // CHECK-SAME: volatileOption = <VOLATILE>
  tt.func @load_volatile(
      %ptr: tensor<256x!tt.ptr<bf16>, #blocked>,
      %out: tensor<256x!tt.ptr<bf16>, #blocked>) {
    %0 = tt.load %ptr {isVolatile = true} : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.store %out, %0 : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  // --- Test 4: default store -> CA (elided) ---
  // CHECK-LABEL: @store_default
  // CHECK: ascend_dpx.store
  // CHECK-NOT: cacheOption
  // CHECK-SAME: : <1>, vector<1xi16>
  tt.func @store_default(
      %ptr: tensor<256x!tt.ptr<bf16>, #blocked>,
      %val: tensor<256xbf16, #blocked>) {
    tt.store %ptr, %val : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  // --- Test 5: CG store -> NCA ---
  // CHECK-LABEL: @store_cg
  // CHECK: ascend_dpx.store
  // CHECK-SAME: cacheOption = <LOADCACHEOPTION_NCA>
  tt.func @store_cg(
      %ptr: tensor<256x!tt.ptr<bf16>, #blocked>,
      %val: tensor<256xbf16, #blocked>) {
    tt.store %ptr, %val cacheModifier = cg : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}
