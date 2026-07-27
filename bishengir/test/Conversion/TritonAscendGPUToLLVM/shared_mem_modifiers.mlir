// RUN: bishengir-opt -convert-triton-ascend-gpu-to-llvm %s | FileCheck %s
//
// cache_modifier on shared memory:
//   `ttg.local_load` / `ttg.local_store` (shared memory ops) do not carry a
//   `cacheModifier` attribute by design — the Ascend hardware has no L2
//   cache on shared memory, so the modifier is silently dropped during
//   `ConvertSharedPtrToMemDescPass`. After lowering, the resulting
//   `ascend_dpx.load` / `ascend_dpx.store` targets `!llvm.ptr<3>` (shared)
//   and contains no `cacheOption` attribute. This is the regression guard
//   for that contract: even when other loads in the same kernel set
//   `cacheOption = NCA` on `!llvm.ptr<1>` (global), shared-memory ops stay
//   bare.

#blocked  = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared   = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem     = #ttg.shared_memory

module attributes {
    "ttg.num-ctas"         = 1 : i32,
    "ttg.num-warps"        = 4 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared             = 16384 : i32,
    ttg.target             = "cuda:80",
    hacc.target            = #hacc.target<"Ascend910_9589">,
    "ttg.enable-bishengir-simt-optimization" = 1111 : i32
} {
  // CHECK-LABEL: @shared_mem_load_store_no_cache_option
  // CHECK:       ascend_dpx.store
  // CHECK-SAME:  : <3>
  // CHECK-NOT:   cacheOption
  //
  // CHECK:       ascend_dpx.load
  // CHECK-SAME:  : (!llvm.ptr<3>)
  // CHECK-NOT:   cacheOption
  tt.func public @shared_mem_load_store_no_cache_option(
      %data : tensor<32x32xf16, #blocked>) {
    %buf = ttg.local_alloc %data {allocation.offset = 0 : i32}
      : (tensor<32x32xf16, #blocked>) -> !ttg.memdesc<32x32xf16, #shared, #smem>
    %loaded = ttg.local_load %buf
      : !ttg.memdesc<32x32xf16, #shared, #smem>
     -> tensor<32x32xf16, #blocked>
    tt.return
  }
}
