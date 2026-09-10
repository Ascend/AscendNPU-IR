// RUN: bishengir-opt --split-input-file --convert-debug-to-ascenddpx --allocate-ascend-shared-memory -convert-triton-ascend-gpu-to-llvm --verify-diagnostics %s

#rank9_blocked = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 1, 1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 1, 1, 1, 1, 1, 32], warpsPerCTA = [1, 1, 1, 1, 1, 1, 1, 1, 1], order = [8, 7, 6, 5, 4, 3, 2, 1, 0]}>

module attributes {
    "ttg.simt-optimization-mode" = 1111 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 1 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared = 4096 : i32,
    ttg.target = "cuda:80"
} {
  tt.func public @print_tensor_rank9(
      %arg0: tensor<1x1x1x1x1x1x1x1x32xi32, #rank9_blocked>) attributes {noinline = false} {
    // expected-error@+2 {{DebugOp requires rank <= maxOpRank}}
    // expected-error@+1 {{failed to legalize operation 'ascend_dpx.print' that was explicitly marked illegal}}
    tt.print "rank-9 tensor: " {hex = false, isSigned = array<i32: 1>}
      : %arg0 : tensor<1x1x1x1x1x1x1x1x32xi32, #rank9_blocked>
    tt.return
  }
}

// -----

module attributes {
    "ttg.simt-optimization-mode" = 1111 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 1 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared = 0 : i32,
    ttg.target = "cuda:80"
} {
  tt.func public @reject_f64_before_runtime_dispatch(
      %arg0: f64) attributes {noinline = false} {
    // expected-error@+1 {{device print does not support element type 'f64'}}
    tt.print "unsupported f64: " {hex = false, isSigned = array<i32: 1>}
      : %arg0 : f64
    tt.return
  }
}

// -----

#i4_blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {
    "ttg.simt-optimization-mode" = 1111 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 1 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared = 0 : i32,
    ttg.target = "cuda:80"
} {
  tt.func public @reject_i4_before_shared_memory_staging(
      %arg0: tensor<32xi4, #i4_blocked>) attributes {noinline = false} {
    // expected-error@+1 {{device print does not support element type 'i4'}}
    tt.print "unsupported i4: " {hex = false, isSigned = array<i32: 1>}
      : %arg0 : tensor<32xi4, #i4_blocked>
    tt.return
  }
}

// -----

module attributes {
    "ttg.simt-optimization-mode" = 1111 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 1 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared = 0 : i32,
    ttg.target = "cuda:80"
} {
  tt.func public @reject_f64_at_runtime_dispatch(
      %arg0: f64) attributes {noinline = false} {
    // expected-error@+2 {{device print runtime does not support operand type 'f64'}}
    // expected-error@+1 {{failed to legalize operation 'ascend_dpx.print' that was explicitly marked illegal}}
    "ascend_dpx.print"(%arg0) {
      hex = false,
      isBool = array<i32: 0>,
      isSigned = array<i32: 1>,
      prefix = "unsupported direct f64: "
    } : (f64) -> ()
    tt.return
  }
}

// -----

module attributes {
    "ttg.simt-optimization-mode" = 1111 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 1 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared = 0 : i32,
    ttg.target = "cuda:80"
} {
  tt.func public @reject_direct_assert_i32(
      %arg0: i32) attributes {noinline = false} {
    // expected-error@+2 {{device assert runtime does not support condition type 'i32'}}
    // expected-error@+1 {{failed to legalize operation 'ascend_dpx.assert' that was explicitly marked illegal}}
    ascend_dpx.assert %arg0, "unsupported direct i32 assertion" : i32
    tt.return
  }
}

// -----

#debug_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#debug_smem = #ttg.shared_memory

module attributes {
    "ttg.simt-optimization-mode" = 1111 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 1 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared = 4096 : i32,
    ttg.target = "cuda:80"
} {
  tt.func public @reject_direct_print_non_i8_bool_memdesc() attributes {noinline = false} {
    %value = ttg.local_alloc
      : () -> !ttg.memdesc<32xi32, #debug_shared, #debug_smem, mutable>
    // expected-error@+2 {{device print runtime does not support operand type '!ttg.memdesc<32xi32}}
    // expected-error@+1 {{failed to legalize operation 'ascend_dpx.print' that was explicitly marked illegal}}
    "ascend_dpx.print"(%value) {
      hex = false,
      isBool = array<i32: 1>,
      isSigned = array<i32: 1>,
      prefix = "unsupported direct non-i8 bool memdesc: "
    } : (!ttg.memdesc<32xi32, #debug_shared, #debug_smem, mutable>) -> ()
    tt.return
  }
}

// -----

#debug_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#debug_smem = #ttg.shared_memory

module attributes {
    "ttg.simt-optimization-mode" = 1111 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 1 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared = 4096 : i32,
    ttg.target = "cuda:80"
} {
  tt.func public @reject_direct_print_i1_memdesc() attributes {noinline = false} {
    %value = ttg.local_alloc
      : () -> !ttg.memdesc<32xi1, #debug_shared, #debug_smem, mutable>
    // expected-error@+2 {{device print runtime does not support operand type '!ttg.memdesc<32xi1}}
    // expected-error@+1 {{failed to legalize operation 'ascend_dpx.print' that was explicitly marked illegal}}
    "ascend_dpx.print"(%value) {
      hex = false,
      isBool = array<i32: 1>,
      isSigned = array<i32: 0>,
      prefix = "unsupported direct i1 memdesc: "
    } : (!ttg.memdesc<32xi1, #debug_shared, #debug_smem, mutable>) -> ()
    tt.return
  }
}

// -----

#debug_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#debug_smem = #ttg.shared_memory

module attributes {
    "ttg.simt-optimization-mode" = 1111 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 1 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared = 4096 : i32,
    ttg.target = "cuda:80"
} {
  tt.func public @reject_direct_assert_i1_memdesc() attributes {noinline = false} {
    %cond = ttg.local_alloc
      : () -> !ttg.memdesc<32xi1, #debug_shared, #debug_smem, mutable>
    // expected-error@+2 {{device assert runtime does not support condition type '!ttg.memdesc<32xi1}}
    // expected-error@+1 {{failed to legalize operation 'ascend_dpx.assert' that was explicitly marked illegal}}
    ascend_dpx.assert %cond, "unsupported direct i1 memdesc assertion"
      : !ttg.memdesc<32xi1, #debug_shared, #debug_smem, mutable>
    tt.return
  }
}

// -----

#rank9_blocked = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 1, 1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 1, 1, 1, 1, 1, 32], warpsPerCTA = [1, 1, 1, 1, 1, 1, 1, 1, 1], order = [8, 7, 6, 5, 4, 3, 2, 1, 0]}>

module attributes {
    "ttg.simt-optimization-mode" = 1111 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 1 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared = 4096 : i32,
    ttg.target = "cuda:80"
} {
  tt.func public @assert_tensor_rank9(
      %arg0: tensor<1x1x1x1x1x1x1x1x32xi1, #rank9_blocked>) attributes {noinline = false} {
    // expected-error@+2 {{DebugOp requires rank <= maxOpRank}}
    // expected-error@+1 {{failed to legalize operation 'ascend_dpx.assert' that was explicitly marked illegal}}
    tt.assert %arg0, "rank-9 tensor assertion failed"
      : tensor<1x1x1x1x1x1x1x1x32xi1, #rank9_blocked>
    tt.return
  }
}
