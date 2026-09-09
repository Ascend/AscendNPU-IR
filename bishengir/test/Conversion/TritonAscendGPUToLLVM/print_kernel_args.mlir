// RUN: bishengir-opt --split-input-file --convert-debug-to-ascenddpx --allocate-ascend-shared-memory -convert-triton-ascend-gpu-to-llvm %s | FileCheck %s
// RUN: bishengir-opt --split-input-file --convert-debug-to-ascenddpx %s | FileCheck %s --check-prefix=DPX

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {
    "ttg.simt-optimization-mode" = 1111 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared = 0 : i32,
    ttg.target = "cuda:80"
} {
  // DPX-LABEL: tt.func public @print_scalar_kernel_arg
  // DPX: ascend_dpx.print "scalar arg: "
  // DPX-SAME: isBool = array<i32: 0, 0, 0, 1>
  // CHECK: llvm.mlir.global private constant @_debug_prefix_0("scalar arg: ") {addr_space = 1 : i32}
  // CHECK-LABEL: llvm.func @print_scalar_kernel_arg
  // CHECK: scf.if
  // CHECK: llvm.call @_mlir_ciface_print_scalar_int32_t_gm_simt
  // CHECK: llvm.call @_mlir_ciface_print_scalar_int16_t_gm_simt
  // CHECK: llvm.call @_mlir_ciface_print_scalar_bfloat16_t_gm_simt
  // CHECK: llvm.call @_mlir_ciface_print_scalar_bool_gm_simt
  // CHECK-NOT: tt.print
  // CHECK-NOT: ascend_dpx.print
  tt.func public @print_scalar_kernel_arg(
      %arg0: i32, %arg1: i16, %arg2: bf16,
      %arg3: i1) attributes {noinline = false} {
    tt.print "scalar arg: " {hex = false, isSigned = array<i32: 1, 1, 1, 0>}
      : %arg0, %arg1, %arg2, %arg3 : i32, i16, bf16, i1
    tt.return
  }
}

// -----

module attributes {
    "ttg.simt-optimization-mode" = 1111 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared = 0 : i32,
    ttg.target = "cuda:80"
} {
  // DPX-LABEL: tt.func public @print_loaded_scalar_bool
  // DPX: %[[LOADED:.*]] = tt.load {{.*}} {was_bool_to_int8 = true}
  // DPX: %[[BOOL:.*]] = arith.trunci %[[LOADED]] : i8 to i1
  // DPX: ascend_dpx.print "loaded scalar bool: "
  // DPX-SAME: isBool = array<i32: 1>
  // DPX-SAME: : %[[BOOL]] : i1
  // CHECK-LABEL: llvm.func @print_loaded_scalar_bool
  // CHECK: llvm.call @_mlir_ciface_print_scalar_bool_gm_simt
  // CHECK-NOT: llvm.call @_mlir_ciface_print_scalar_uint8_t_gm_simt
  tt.func public @print_loaded_scalar_bool(
      %arg0: !tt.ptr<i8>) attributes {noinline = false} {
    %0 = tt.load %arg0 {was_bool_to_int8 = true} : !tt.ptr<i8>
    tt.print "loaded scalar bool: " {hex = false, isSigned = array<i32: 0>}
      : %0 : i8
    tt.return
  }
}

// -----

module attributes {
    "ttg.simt-optimization-mode" = 1111 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 1 : i32,
    "ttg.super-block-factor" = 2 : ui32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared = 0 : i32,
    ttg.target = "cuda:80"
} {
  // CHECK-LABEL: llvm.func @print_from_second_superblock_task
  // CHECK: %[[TID:.*]] = ascend_dpx.thread_id_x
  // CHECK: %[[FACTOR:.*]] = llvm.mlir.constant(2 : i32)
  // CHECK: %[[WARP_SIZE:.*]] = llvm.mlir.constant(32 : i32)
  // CHECK: %[[LANE_ID:.*]] = llvm.urem %[[TID]], %[[WARP_SIZE]] : i32
  // CHECK: %[[WARP_ID:.*]] = llvm.udiv %[[TID]], %[[WARP_SIZE]] : i32
  // CHECK: %[[TASK_WARP_ID:.*]] = llvm.udiv %[[WARP_ID]], %[[FACTOR]] : i32
  // CHECK: %[[TASK_WARP_BASE:.*]] = llvm.mul %[[TASK_WARP_ID]], %[[WARP_SIZE]] : i32
  // CHECK: %[[LOGICAL_TID:.*]] = llvm.add %[[TASK_WARP_BASE]], %[[LANE_ID]] : i32
  // CHECK: %[[TID_Y:.*]] = ascend_dpx.thread_id_y
  // CHECK: %[[TID_Z:.*]] = ascend_dpx.thread_id_z
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
  // CHECK: %[[CMP_X:.*]] = llvm.icmp "eq" %[[LOGICAL_TID]], %[[ZERO]] : i32
  // CHECK: %[[CMP_Y:.*]] = llvm.icmp "eq" %[[TID_Y]], %[[ZERO]] : i32
  // CHECK: %[[CMP_Z:.*]] = llvm.icmp "eq" %[[TID_Z]], %[[ZERO]] : i32
  // CHECK: %[[CMP_XY:.*]] = llvm.and %[[CMP_X]], %[[CMP_Y]] : i1
  // CHECK: %[[GUARD:.*]] = llvm.and %[[CMP_XY]], %[[CMP_Z]] : i1
  // CHECK: scf.if %[[GUARD]]
  // CHECK: llvm.call @_mlir_ciface_print_scalar_int32_t_gm_simt
  tt.func public @print_from_second_superblock_task(
      %arg0: i32) attributes {noinline = false} {
    "ascend_dpx.print"(%arg0) {
      hex = false,
      isBool = array<i32: 0>,
      isSigned = array<i32: 1>,
      prefix = "logical_pid:"
    } : (i32) -> ()
    tt.return
  }
}

// -----

#fp8_blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {
    "ttg.simt-optimization-mode" = 1111 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared = 4096 : i32,
    ttg.target = "cuda:80"
} {
  // DPX-LABEL: tt.func public @print_fp8_args
  // DPX: %[[SCALAR:.*]] = tt.fp_to_fp {{.*}} : f8E5M2 -> f32
  // DPX: %[[TENSOR:.*]] = tt.fp_to_fp {{.*}} : tensor<4x8xf8E4M3FN, {{.*}}> -> tensor<4x8xf32, {{.*}}>
  // DPX: %[[STAGED:.*]] = ttg.local_alloc %[[TENSOR]]
  // DPX: ascend_dpx.print "fp8 args: " {{.*}} : %[[SCALAR]], %[[STAGED]] :
  // CHECK-LABEL: llvm.func @print_fp8_args
  // CHECK: llvm.call @_mlir_ciface_print_scalar_float_gm_simt
  // CHECK: llvm.call @_mlir_ciface_print_2d_float_ubuf_simt
  // CHECK-NOT: _mlir_ciface_print_scalar_float8
  // CHECK-NOT: _mlir_ciface_print_2d_float8
  tt.func public @print_fp8_args(
      %arg0: f8E5M2,
      %arg1: tensor<4x8xf8E4M3FN, #fp8_blocked>) attributes {noinline = false} {
    tt.print "fp8 args: " {hex = false, isSigned = array<i32: 1, 1>}
      : %arg0, %arg1 : f8E5M2, tensor<4x8xf8E4M3FN, #fp8_blocked>
    tt.return
  }
}

// -----

#bool_blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {
    "ttg.simt-optimization-mode" = 1111 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared = 8192 : i32,
    ttg.target = "cuda:80"
} {
  // DPX-LABEL: tt.func public @print_bool_and_integer_tensor_args
  // DPX: arith.extui {{.*}} : tensor<4x8xi1, #blocked> to tensor<4x8xi8, #blocked>
  // DPX: ascend_dpx.print "mixed tensor args: "
  // DPX-SAME: isBool = array<i32: 1, 0>
  // DPX-SAME: isSigned = array<i32: 0, 1>
  // CHECK-LABEL: llvm.func @print_bool_and_integer_tensor_args
  // CHECK: llvm.call @_mlir_ciface_print_2d_bool_ubuf_simt
  // CHECK: llvm.call @_mlir_ciface_print_2d_int32_t_ubuf_simt
  // CHECK-NOT: llvm.call @_mlir_ciface_print_2d_uint8_t_ubuf_simt
  // CHECK-NOT: ascend_dpx.print
  tt.func public @print_bool_and_integer_tensor_args(
      %arg0: tensor<4x8xi1, #bool_blocked>,
      %arg1: tensor<4x8xi32, #bool_blocked>) attributes {noinline = false} {
    tt.print "mixed tensor args: " {hex = false, isSigned = array<i32: 0, 1>}
      : %arg0, %arg1 : tensor<4x8xi1, #bool_blocked>, tensor<4x8xi32, #bool_blocked>
    tt.return
  }
}

// -----

#bool_load_blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {
    "ttg.simt-optimization-mode" = 1111 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared = 8192 : i32,
    ttg.target = "cuda:80"
} {
  // DPX-LABEL: tt.func public @print_loaded_bool_tensor_arg
  // DPX: tt.load {{.*}} {was_bool_to_int8 = true}
  // DPX: ascend_dpx.print "loaded bool: "
  // DPX-SAME: isBool = array<i32: 1>
  // CHECK-LABEL: llvm.func @print_loaded_bool_tensor_arg
  // CHECK: llvm.call @_mlir_ciface_print_1d_bool_ubuf_simt
  // CHECK-NOT: llvm.call @_mlir_ciface_print_1d_uint8_t_ubuf_simt
  tt.func public @print_loaded_bool_tensor_arg(
      %arg0: tensor<2x!tt.ptr<i8>, #bool_load_blocked>) attributes {noinline = false} {
    %0 = tt.load %arg0 {was_bool_to_int8 = true} : tensor<2x!tt.ptr<i8>, #bool_load_blocked>
    tt.print "loaded bool: " {hex = false, isSigned = array<i32: 0>}
      : %0 : tensor<2xi8, #bool_load_blocked>
    tt.return
  }
}

// -----

module attributes {
    "ttg.simt-optimization-mode" = 1111 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared = 0 : i32,
    ttg.target = "cuda:80"
} {
  // CHECK: llvm.mlir.global private constant @_debug_msg_0("scalar assertion failed") {addr_space = 1 : i32}
  // CHECK-LABEL: llvm.func @assert_scalar_kernel_arg
  // CHECK: scf.if
  // CHECK: llvm.call @_mlir_ciface_assert_scalar_bool_gm_simt
  // CHECK-NOT: tt.assert
  // CHECK-NOT: ascend_dpx.assert
  tt.func public @assert_scalar_kernel_arg(%arg0: i1) attributes {noinline = false} {
    tt.assert %arg0, "scalar assertion failed" : i1
    tt.return
  }
}

// -----

#assert_blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {
    "ttg.simt-optimization-mode" = 1111 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared = 4096 : i32,
    ttg.target = "cuda:80"
} {
  // CHECK: llvm.mlir.global private constant @_debug_msg_
  // CHECK-LABEL: llvm.func @assert_tensor_kernel_arg
  // CHECK: ascend_dpx.sync_threads
  // CHECK: llvm.call @_mlir_ciface_assert_2d_int8_t_ubuf_simt
  // CHECK-NOT: tt.assert
  // CHECK-NOT: ascend_dpx.assert
  tt.func public @assert_tensor_kernel_arg(%arg0: tensor<4x8xi1, #assert_blocked>) attributes {noinline = false} {
    tt.assert %arg0, "tensor assertion failed" : tensor<4x8xi1, #assert_blocked>
    tt.return
  }
}

// -----

module attributes {
    "ttg.simt-optimization-mode" = 1111 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared = 0 : i32,
    ttg.target = "cuda:80"
} {
  // CHECK: llvm.mlir.global private constant @_debug_prefix_
  // CHECK-LABEL: llvm.func @print_mixed_signedness_constexpr_values
  // CHECK: llvm.call @_mlir_ciface_print_scalar_int64_t_gm_simt
  // CHECK: llvm.call @_mlir_ciface_print_scalar_uint64_t_gm_simt
  // CHECK-NOT: tt.print
  // CHECK-NOT: ascend_dpx.print
  tt.func public @print_mixed_signedness_constexpr_values() attributes {noinline = false} {
    %signed = arith.constant -1 : i64
    %unsigned = arith.constant -1 : i64
    tt.print "mixed signedness: " {hex = false, isSigned = array<i32: 1, 0>}
      : %signed, %unsigned : i64, i64
    tt.return
  }
}

// -----

module attributes {
    "ttg.simt-optimization-mode" = 1111 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared = 0 : i32,
    ttg.target = "cuda:80"
} {
  // CHECK: llvm.mlir.global private constant @_debug_prefix_
  // CHECK-LABEL: llvm.func @print_unsigned_constexpr_value
  // CHECK: llvm.call @_mlir_ciface_print_scalar_uint64_t_gm_simt
  // CHECK-NOT: llvm.call @_mlir_ciface_print_scalar_int64_t_gm_simt
  // CHECK-NOT: tt.print
  // CHECK-NOT: ascend_dpx.print
  tt.func public @print_unsigned_constexpr_value() attributes {noinline = false} {
    %c = arith.constant -9223372036854775807 : i64
    tt.print "unsigned constexpr: " {hex = false, isSigned = array<i32: 0>} : %c : i64
    tt.return
  }
}

// -----

module attributes {
    "ttg.simt-optimization-mode" = 1111 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared = 0 : i32,
    ttg.target = "cuda:80"
} {
  // CHECK: llvm.mlir.global private constant @_debug_prefix_
  // CHECK-LABEL: llvm.func @print_pointer_kernel_arg
  // CHECK: llvm.ptrtoint
  // CHECK: llvm.mlir.constant(1 : i8)
  // CHECK: llvm.call @_mlir_ciface_print_scalar_uint64_t_gm_simt
  // CHECK-NOT: _mlir_ciface_print_scalar_UNKNOWN_gm_simt
  // CHECK-NOT: tt.print
  // CHECK-NOT: ascend_dpx.print
  tt.func public @print_pointer_kernel_arg(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    tt.print "ptr arg: " {hex = false, isSigned = array<i32: 0>} : %arg0 : !tt.ptr<f32>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {
    "ttg.simt-optimization-mode" = 1111 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared = 4096 : i32,
    ttg.target = "cuda:80"
} {
  // CHECK: llvm.mlir.global private constant @_debug_prefix_
  // CHECK-LABEL: llvm.func @print_tensor_kernel_arg
  // CHECK: ascend_dpx.sync_threads
  // CHECK: llvm.call @_mlir_ciface_print_2d_half_ubuf_simt
  // CHECK-NOT: tt.print
  // CHECK-NOT: ascend_dpx.print
  tt.func public @print_tensor_kernel_arg(%arg0: tensor<4x8xf16, #blocked>) attributes {noinline = false} {
    tt.print "tensor arg: " {hex = false, isSigned = array<i32: 1>}
      : %arg0 : tensor<4x8xf16, #blocked>
    tt.return
  }
}

// -----

module attributes {
    "ttg.simt-optimization-mode" = 1111 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared = 0 : i32,
    ttg.target = "cuda:80"
} {
  // CHECK: llvm.mlir.global private constant @_debug_prefix_
  // CHECK-LABEL: llvm.func @print_constexpr_value
  // CHECK: llvm.call @_mlir_ciface_print_scalar_int32_t_gm_simt
  // CHECK-NOT: tt.print
  // CHECK-NOT: ascend_dpx.print
  tt.func public @print_constexpr_value() attributes {noinline = false} {
    %c42_i32 = arith.constant 42 : i32
    tt.print "constexpr: " {hex = false, isSigned = array<i32: 1>} : %c42_i32 : i32
    tt.return
  }
}

// -----

#assert_rank5_blocked = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 1, 32], warpsPerCTA = [1, 1, 1, 1, 4], order = [4, 3, 2, 1, 0]}>

module attributes {
    "ttg.simt-optimization-mode" = 1111 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared = 4096 : i32,
    ttg.target = "cuda:80"
} {
  // CHECK-LABEL: llvm.func @assert_tensor_rank5
  // CHECK: llvm.call @_mlir_ciface_assert_5d_int8_t_ubuf_simt
  // CHECK-NOT: tt.assert
  // CHECK-NOT: ascend_dpx.assert
  tt.func public @assert_tensor_rank5(
      %arg0: tensor<1x1x1x1x128xi1, #assert_rank5_blocked>) attributes {noinline = false} {
    tt.assert %arg0, "rank-5 tensor assertion failed" : tensor<1x1x1x1x128xi1, #assert_rank5_blocked>
    tt.return
  }
}

// -----

#assert_rank6_blocked = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 1, 1, 32], warpsPerCTA = [1, 1, 1, 1, 1, 4], order = [5, 4, 3, 2, 1, 0]}>
#assert_rank7_blocked = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 1, 1, 1, 32], warpsPerCTA = [1, 1, 1, 1, 1, 1, 4], order = [6, 5, 4, 3, 2, 1, 0]}>
#debug_rank8_blocked = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 1, 1, 1, 1, 32], warpsPerCTA = [1, 1, 1, 1, 1, 1, 1, 4], order = [7, 6, 5, 4, 3, 2, 1, 0]}>

module attributes {
    "ttg.simt-optimization-mode" = 1111 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared = 4096 : i32,
    ttg.target = "cuda:80"
} {
  // CHECK-LABEL: llvm.func @debug_tensor_rank8
  // CHECK: llvm.call @_mlir_ciface_assert_6d_int8_t_ubuf_simt
  // CHECK: llvm.call @_mlir_ciface_assert_7d_int8_t_ubuf_simt
  // CHECK: llvm.call @_mlir_ciface_assert_8d_int8_t_ubuf_simt
  // CHECK: llvm.call @_mlir_ciface_print_8d_half_ubuf_simt
  // CHECK-NOT: tt.assert
  // CHECK-NOT: tt.print
  // CHECK-NOT: ascend_dpx.assert
  // CHECK-NOT: ascend_dpx.print
  tt.func public @debug_tensor_rank8(
      %cond6: tensor<1x1x1x1x1x128xi1, #assert_rank6_blocked>,
      %cond7: tensor<1x1x1x1x1x1x128xi1, #assert_rank7_blocked>,
      %cond: tensor<1x1x1x1x1x1x1x128xi1, #debug_rank8_blocked>,
      %value: tensor<1x1x1x1x1x1x1x128xf16, #debug_rank8_blocked>) attributes {noinline = false} {
    tt.assert %cond6, "rank-6 tensor assertion failed" : tensor<1x1x1x1x1x128xi1, #assert_rank6_blocked>
    tt.assert %cond7, "rank-7 tensor assertion failed" : tensor<1x1x1x1x1x1x128xi1, #assert_rank7_blocked>
    tt.assert %cond, "rank-8 tensor assertion failed" : tensor<1x1x1x1x1x1x1x128xi1, #debug_rank8_blocked>
    tt.print "rank-8 tensor: " {hex = false, isSigned = array<i32: 1>}
      : %value : tensor<1x1x1x1x1x1x1x128xf16, #debug_rank8_blocked>
    tt.return
  }
}
