// RUN: bishengir-opt -split-input-file -simt-fast-div %s | FileCheck %s

// Encodings shared across test cases.
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// Test 1: scalar signed division whose divisor is a direct i32 function arg.
//
// Expected transformations:
//  - Two tt.func private declarations inserted in the module.
//  - Two ttg.local_alloc + two tt.call_scalar at function entry.
//  - Two ttg.local_load (for memory ordering) before the division.
//  - arith.divsi replaced by ascend_dpx.umulhi + arith.addi + arith.shrsi.

// CHECK-DAG: llvm.func @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK-DAG: llvm.func @_mlir_ciface_simt_div_magic_mul_uint32_t

// CHECK-LABEL: @scalar_divsi
// CHECK:       [[SHM_SHIFT:%[^ ]+]] = ttg.local_alloc
// CHECK:       [[SHIFT:%[^ ]+]] = tt.call_scalar [[SHM_SHIFT]], @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK:       [[SHM_MUL:%[^ ]+]] = ttg.local_alloc
// CHECK:       [[MAGIC:%[^ ]+]] = tt.call_scalar [[SHM_MUL]], @_mlir_ciface_simt_div_magic_mul_uint32_t
// CHECK:       ttg.local_load [[SHM_SHIFT]]
// CHECK:       ttg.local_load [[SHM_MUL]]
// CHECK:       [[HI:%[^ ]+]] = ascend_dpx.umulhi {{%[^ ]+}}, [[MAGIC]]
// CHECK:       [[SUM:%[^ ]+]] = arith.addi [[HI]], {{%[^ ]+}}
// CHECK:       arith.shrsi [[SUM]], [[SHIFT]]
// CHECK-NOT:   arith.divsi
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scalar_divsi(%dividend: i32, %divisor: i32) -> i32 {
    %0 = arith.divsi %dividend, %divisor : i32
    tt.return %0 : i32
  }
}

// -----

// Test 2: scalar unsigned division whose divisor is a direct i32 function arg.
// The replacement uses arith.shrui (logical shift) instead of arith.shrsi.

// CHECK-LABEL: @scalar_divui
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_mul_uint32_t
// CHECK:       [[HI:%[^ ]+]] = ascend_dpx.umulhi
// CHECK:       [[SUM:%[^ ]+]] = arith.addi [[HI]], {{%[^ ]+}}
// CHECK:       arith.shrui [[SUM]],
// CHECK-NOT:   arith.shrsi
// CHECK-NOT:   arith.divui
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scalar_divui(%dividend: i32, %divisor: i32) -> i32 {
    %0 = arith.divui %dividend, %divisor : i32
    tt.return %0 : i32
  }
}

// -----

// Test 3: tensorized signed division — divisor is tt.splat of an i32 arg.
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
//
// Expected transformations:
//  - ttg.local_load returns tensor<1xi32, #blocked1> (dividend encoding).
//  - tt.broadcast widens tensor<1xi32> to tensor<32xi32>.
//  - tt.mulhiui performs element-wise upper-32-bit multiply.
//  - arith.shrsi performs the final tensor shift.

// CHECK-LABEL: @tensor_divsi
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_mul_uint32_t
// CHECK:       [[ML:%[^ ]+]] = ttg.local_load {{.*}} -> tensor<1xi32, #{{.*}}>
// CHECK:       [[SL:%[^ ]+]] = ttg.local_load {{.*}} -> tensor<1xi32, #{{.*}}>
// CHECK:       [[MT:%[^ ]+]] = tt.broadcast [[ML]] : tensor<1xi32, {{.*}}> -> tensor<32xi32,
// CHECK:       [[ST:%[^ ]+]] = tt.broadcast [[SL]] : tensor<1xi32, {{.*}}> -> tensor<32xi32,
// CHECK:       [[HI:%[^ ]+]] = tt.mulhiui {{%[^ ]+}}, [[MT]]
// CHECK:       [[SUM:%[^ ]+]] = arith.addi [[HI]], {{%[^ ]+}}
// CHECK:       arith.shrsi [[SUM]], [[ST]]
// CHECK-NOT:   arith.divsi
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_divsi(%dividend: tensor<32xi32, #blocked1>, %divisor: i32) -> tensor<32xi32, #blocked1> {
    %splat = tt.splat %divisor : i32 -> tensor<32xi32, #blocked1>
    %0 = arith.divsi %dividend, %splat : tensor<32xi32, #blocked1>
    tt.return %0 : tensor<32xi32, #blocked1>
  }
}

// -----

// Test 4: tensorized unsigned division.
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: @tensor_divui
// CHECK:       [[HI:%[^ ]+]] = tt.mulhiui
// CHECK:       [[SUM:%[^ ]+]] = arith.addi [[HI]], {{%[^ ]+}}
// CHECK:       arith.shrui [[SUM]],
// CHECK-NOT:   arith.shrsi
// CHECK-NOT:   arith.divui
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_divui(%dividend: tensor<32xi32, #blocked1>, %divisor: i32) -> tensor<32xi32, #blocked1> {
    %splat = tt.splat %divisor : i32 -> tensor<32xi32, #blocked1>
    %0 = arith.divui %dividend, %splat : tensor<32xi32, #blocked1>
    tt.return %0 : tensor<32xi32, #blocked1>
  }
}

// -----

// Test 5: multiple divisions sharing the same divisor argument.
// The magic computation (local_alloc + call_scalar pair) must be hoisted only
// once — not once per division.

// CHECK-LABEL: @same_divisor_twice
// CHECK-COUNT-1: @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK-COUNT-1: @_mlir_ciface_simt_div_magic_mul_uint32_t
// CHECK-NOT:     arith.divsi
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @same_divisor_twice(%a: i32, %b: i32, %divisor: i32) -> i32 {
    %0 = arith.divsi %a, %divisor : i32
    %1 = arith.divsi %b, %divisor : i32
    %2 = arith.addi %0, %1 : i32
    tt.return %2 : i32
  }
}

// -----

// Test 6: two different divisor arguments — two independent magic computations.

// CHECK-LABEL: @two_divisors
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_mul_uint32_t
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_mul_uint32_t
// CHECK-NOT:   arith.divsi
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @two_divisors(%a: i32, %d0: i32, %d1: i32) -> i32 {
    %0 = arith.divsi %a, %d0 : i32
    %1 = arith.divsi %0, %d1 : i32
    tt.return %1 : i32
  }
}

// -----

// Test 7: division by a constant — arith.constant divisors are transformed
// just like function-argument divisors.

// CHECK-LABEL: @divisor_is_constant
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_mul_uint32_t
// CHECK:       ascend_dpx.umulhi
// CHECK-NOT:   arith.divsi
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @divisor_is_constant(%dividend: i32) -> i32 {
    %c7 = arith.constant 7 : i32
    %0 = arith.divsi %dividend, %c7 : i32
    tt.return %0 : i32
  }
}

// -----

// Test 8: i64 division — element type is not i32, must not be transformed.

// CHECK-LABEL: @i64_division_not_transformed
// CHECK-NOT:   tt.call_scalar
// CHECK:       arith.divsi
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @i64_division_not_transformed(%a: i64, %b: i64) -> i64 {
    %0 = arith.divsi %a, %b : i64
    tt.return %0 : i64
  }
}

// -----

// Test 9: mixed — one divisor is a function argument, the other is a constant.
// Both divisions are now transformed; two independent magic computations are
// emitted (one per unique divisor value / argument).

// CHECK-LABEL: @mixed
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_mul_uint32_t
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_mul_uint32_t
// CHECK-NOT:   arith.divsi
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @mixed(%a: i32, %b: i32, %divisor: i32) -> i32 {
    %c3 = arith.constant 3 : i32
    %0 = arith.divsi %a, %divisor : i32
    %1 = arith.divsi %b, %c3 : i32
    %2 = arith.addi %0, %1 : i32
    tt.return %2 : i32
  }
}

// -----

// Test 10: magic ops are hoisted to the entry block even when the division
// appears inside a conditional (scf.if).

// CHECK-LABEL: @division_in_branch
// CHECK:       ttg.local_alloc
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK:       ttg.local_alloc
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_mul_uint32_t
// CHECK-NOT:   arith.divsi
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @division_in_branch(%dividend: i32, %divisor: i32, %cond: i1) -> i32 {
    %c0 = arith.constant 0 : i32
    %result = scf.if %cond -> i32 {
      %0 = arith.divsi %dividend, %divisor : i32
      scf.yield %0 : i32
    } else {
      scf.yield %c0 : i32
    }
    tt.return %result : i32
  }
}

// -----

// Test 11: two divisions sharing the same constant divisor value.
// Even though the two arith.constant ops are distinct SSA values, the pass
// deduplicates by integer value — magic is computed only once.

// CHECK-LABEL: @same_constant_twice
// CHECK-COUNT-1: @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK-COUNT-1: @_mlir_ciface_simt_div_magic_mul_uint32_t
// CHECK-NOT:     arith.divsi
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @same_constant_twice(%a: i32, %b: i32) -> i32 {
    %c7_1 = arith.constant 7 : i32
    %c7_2 = arith.constant 7 : i32
    %0 = arith.divsi %a, %c7_1 : i32
    %1 = arith.divsi %b, %c7_2 : i32
    %2 = arith.addi %0, %1 : i32
    tt.return %2 : i32
  }
}

// -----

// Test 12: two divisions with different constant divisors — two independent
// magic computations are emitted.

// CHECK-LABEL: @two_constants
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_mul_uint32_t
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_mul_uint32_t
// CHECK-NOT:   arith.divui
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @two_constants(%a: i32) -> i32 {
    %c3 = arith.constant 3 : i32
    %c7 = arith.constant 7 : i32
    %0 = arith.divui %a, %c3 : i32
    %1 = arith.divui %0, %c7 : i32
    tt.return %1 : i32
  }
}

// -----

// Test 13: 2-D tensor signed division — the memdesc and local_load result must
// have rank 2 (shape 1x1) to match the dividend's 2-D blocked encoding.  The
// quotient is consumed by a multi-dim arith.addi to exercise the full
// broadcast → element-wise pipeline on a higher-rank tensor.
#blocked2d = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>

// CHECK-LABEL: @tensor_2d_divsi
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_mul_uint32_t
// CHECK:       [[ML:%[^ ]+]] = ttg.local_load {{.*}} -> tensor<1x1xi32, #{{.*}}>
// CHECK:       [[SL:%[^ ]+]] = ttg.local_load {{.*}} -> tensor<1x1xi32, #{{.*}}>
// CHECK:       [[MT:%[^ ]+]] = tt.broadcast [[ML]] : tensor<1x1xi32, {{.*}}> -> tensor<4x32xi32,
// CHECK:       [[ST:%[^ ]+]] = tt.broadcast [[SL]] : tensor<1x1xi32, {{.*}}> -> tensor<4x32xi32,
// CHECK:       [[HI:%[^ ]+]] = tt.mulhiui {{%[^ ]+}}, [[MT]]
// CHECK:       [[SUM:%[^ ]+]] = arith.addi [[HI]], {{%[^ ]+}}
// CHECK:       [[Q:%[^ ]+]] = arith.shrsi [[SUM]], [[ST]]
// CHECK:       arith.addi [[Q]], {{%[^ ]+}}
// CHECK-NOT:   arith.divsi
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_2d_divsi(%dividend: tensor<4x32xi32, #blocked2d>, %divisor: i32) -> tensor<4x32xi32, #blocked2d> {
    %splat = tt.splat %divisor : i32 -> tensor<4x32xi32, #blocked2d>
    %0 = arith.divsi %dividend, %splat : tensor<4x32xi32, #blocked2d>
    %1 = arith.addi %0, %dividend : tensor<4x32xi32, #blocked2d>
    tt.return %1 : tensor<4x32xi32, #blocked2d>
  }
}

// -----

// Test 14: 3-D tensor divsi and remsi sharing the same constant divisor (128).
// Derived from triton_poi_fused_mul_1 where dense<128> is used by both
//   arith.divsi %33, %cst_4  and  arith.remsi %33, %cst_4.
// The magic computation must be emitted only once (dedup by constant value),
// memdesc and local_load must have rank 3, and results feed into multi-dim ops.
#blocked3d = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>

// CHECK-LABEL: @tensor_3d_div_rem_same_const
// CHECK-COUNT-1: @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK-COUNT-1: @_mlir_ciface_simt_div_magic_mul_uint32_t
//   div replacement:
// CHECK:       ttg.local_load {{.*}} -> tensor<1x1x1xi32, #{{.*}}>
// CHECK:       ttg.local_load {{.*}} -> tensor<1x1x1xi32, #{{.*}}>
// CHECK:       tt.broadcast {{.*}} : tensor<1x1x1xi32, {{.*}}> -> tensor<2x4x16xi32,
// CHECK:       tt.broadcast {{.*}} : tensor<1x1x1xi32, {{.*}}> -> tensor<2x4x16xi32,
// CHECK:       tt.mulhiui
// CHECK:       arith.shrsi
//   rem replacement (same magic, fresh loads):
// CHECK:       ttg.local_load {{.*}} -> tensor<1x1x1xi32, #{{.*}}>
// CHECK:       ttg.local_load {{.*}} -> tensor<1x1x1xi32, #{{.*}}>
// CHECK:       tt.broadcast {{.*}} : tensor<1x1x1xi32, {{.*}}> -> tensor<2x4x16xi32,
// CHECK:       tt.broadcast {{.*}} : tensor<1x1x1xi32, {{.*}}> -> tensor<2x4x16xi32,
// CHECK:       tt.mulhiui
// CHECK:       arith.shrsi
// CHECK:       arith.muli
// CHECK:       arith.subi
//   user code:
// CHECK:       arith.muli
// CHECK:       arith.addi
// CHECK-NOT:   arith.divsi
// CHECK-NOT:   arith.remsi
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_3d_div_rem_same_const(%a: tensor<2x4x16xi32, #blocked3d>) -> tensor<2x4x16xi32, #blocked3d> {
    %c128 = arith.constant 128 : i32
    %divisor = tt.splat %c128 : i32 -> tensor<2x4x16xi32, #blocked3d>
    %div = arith.divsi %a, %divisor : tensor<2x4x16xi32, #blocked3d>
    %rem = arith.remsi %a, %divisor : tensor<2x4x16xi32, #blocked3d>
    %0 = arith.muli %div, %a : tensor<2x4x16xi32, #blocked3d>
    %1 = arith.addi %0, %rem : tensor<2x4x16xi32, #blocked3d>
    tt.return %1 : tensor<2x4x16xi32, #blocked3d>
  }
}

// -----

// Test 15: 3-D tensor remsi whose divisor is a function argument via tt.splat.
// Derived from triton_poi_fused_mul_1: arith.remsi %37, %15 where
//   %15 = tt.splat %arg3 : i32 -> tensor<1x128x16xi32>.
// The remainder result is consumed by a multi-dim arith.muli.
#blocked3d = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>

// CHECK-LABEL: @tensor_3d_rem_func_arg
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_mul_uint32_t
// CHECK:       [[ML:%[^ ]+]] = ttg.local_load {{.*}} -> tensor<1x1x1xi32, #{{.*}}>
// CHECK:       [[SL:%[^ ]+]] = ttg.local_load {{.*}} -> tensor<1x1x1xi32, #{{.*}}>
// CHECK:       tt.broadcast [[ML]] : tensor<1x1x1xi32, {{.*}}> -> tensor<2x4x16xi32,
// CHECK:       tt.broadcast [[SL]] : tensor<1x1x1xi32, {{.*}}> -> tensor<2x4x16xi32,
// CHECK:       tt.mulhiui
// CHECK:       arith.shrsi
// CHECK:       arith.muli
// CHECK:       arith.subi
// CHECK:       arith.muli
// CHECK-NOT:   arith.remsi
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_3d_rem_func_arg(%a: tensor<2x4x16xi32, #blocked3d>, %divisor: i32) -> tensor<2x4x16xi32, #blocked3d> {
    %splat = tt.splat %divisor : i32 -> tensor<2x4x16xi32, #blocked3d>
    %rem = arith.remsi %a, %splat : tensor<2x4x16xi32, #blocked3d>
    %result = arith.muli %rem, %a : tensor<2x4x16xi32, #blocked3d>
    tt.return %result : tensor<2x4x16xi32, #blocked3d>
  }
}

// -----

// Test 16: 3-D tensor with two different constant divisors (128 and 384).
// Derived from triton_poi_fused_mul_1 which divides by both dense<128> and
// dense<38400>, with the results chained through arith.muli and arith.addi.
// Two independent magic computations are required.
#blocked3d = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>

// CHECK-LABEL: @tensor_3d_two_constants
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_mul_uint32_t
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_mul_uint32_t
// CHECK:       ttg.local_load {{.*}} -> tensor<1x1x1xi32, #{{.*}}>
// CHECK:       ttg.local_load {{.*}} -> tensor<1x1x1xi32, #{{.*}}>
// CHECK:       tt.broadcast {{.*}} : tensor<1x1x1xi32, {{.*}}> -> tensor<2x4x16xi32,
// CHECK:       tt.broadcast {{.*}} : tensor<1x1x1xi32, {{.*}}> -> tensor<2x4x16xi32,
// CHECK:       tt.mulhiui
// CHECK:       arith.muli
// CHECK:       ttg.local_load {{.*}} -> tensor<1x1x1xi32, #{{.*}}>
// CHECK:       ttg.local_load {{.*}} -> tensor<1x1x1xi32, #{{.*}}>
// CHECK:       tt.broadcast {{.*}} : tensor<1x1x1xi32, {{.*}}> -> tensor<2x4x16xi32,
// CHECK:       tt.broadcast {{.*}} : tensor<1x1x1xi32, {{.*}}> -> tensor<2x4x16xi32,
// CHECK:       tt.mulhiui
// CHECK:       arith.addi
// CHECK-NOT:   arith.divsi
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_3d_two_constants(%a: tensor<2x4x16xi32, #blocked3d>, %b: tensor<2x4x16xi32, #blocked3d>) -> tensor<2x4x16xi32, #blocked3d> {
    %c128 = arith.constant 128 : i32
    %c384 = arith.constant 384 : i32
    %d0 = tt.splat %c128 : i32 -> tensor<2x4x16xi32, #blocked3d>
    %d1 = tt.splat %c384 : i32 -> tensor<2x4x16xi32, #blocked3d>
    %div0 = arith.divsi %a, %d0 : tensor<2x4x16xi32, #blocked3d>
    %mul = arith.muli %div0, %b : tensor<2x4x16xi32, #blocked3d>
    %div1 = arith.divsi %mul, %d1 : tensor<2x4x16xi32, #blocked3d>
    %result = arith.addi %div1, %a : tensor<2x4x16xi32, #blocked3d>
    tt.return %result : tensor<2x4x16xi32, #blocked3d>
  }
}

// -----

// Test 17: 3-D tensor divsi inside an scf.for loop — the magic computation
// (local_alloc + call_scalar) is hoisted to function entry while the
// replacement (local_load + broadcast + mulhiui + shift) stays in the loop
// body.  Uses a non-trivial encoding with multi-dim sizePerThread,
// distributed threadsPerWarp, 2 warps, and non-standard order [0,2,1].
#blocked3d = #ttg.blocked<{sizePerThread = [1, 2, 4], threadsPerWarp = [2, 4, 4], warpsPerCTA = [1, 2, 1], order = [0, 2, 1]}>

// CHECK-LABEL: @tensor_3d_div_in_loop
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_mul_uint32_t
// CHECK:       scf.for
// CHECK:         ttg.local_load {{.*}} -> tensor<1x1x1xi32, #{{.*}}>
// CHECK:         ttg.local_load {{.*}} -> tensor<1x1x1xi32, #{{.*}}>
// CHECK:         tt.broadcast {{.*}} : tensor<1x1x1xi32, {{.*}}> -> tensor<2x16x16xi32,
// CHECK:         tt.broadcast {{.*}} : tensor<1x1x1xi32, {{.*}}> -> tensor<2x16x16xi32,
// CHECK:         tt.mulhiui
// CHECK:         arith.shrsi
// CHECK:         arith.addi
// CHECK-NOT:   arith.divsi
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_3d_div_in_loop(%a: tensor<2x16x16xi32, #blocked3d>, %divisor: i32, %n: index) -> tensor<2x16x16xi32, #blocked3d> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %splat = tt.splat %divisor : i32 -> tensor<2x16x16xi32, #blocked3d>
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %a) -> (tensor<2x16x16xi32, #blocked3d>) {
      %div = arith.divsi %acc, %splat : tensor<2x16x16xi32, #blocked3d>
      %sum = arith.addi %div, %acc : tensor<2x16x16xi32, #blocked3d>
      scf.yield %sum : tensor<2x16x16xi32, #blocked3d>
    }
    tt.return %result : tensor<2x16x16xi32, #blocked3d>
  }
}

// -----

// Test 18: 4-D tensor unsigned division — the memdesc, local_load, and
// broadcast all require rank-4 shapes (1x1x1x1 → 4x2x4x16).  Uses a
// non-trivial 4-D encoding with multi-dim sizePerThread [2,1,1,2],
// distributed threadsPerWarp [1,2,4,4], and non-standard order [3,1,2,0].
#blocked4d = #ttg.blocked<{sizePerThread = [2, 1, 1, 2], threadsPerWarp = [1, 2, 4, 4], warpsPerCTA = [1, 1, 1, 1], order = [3, 1, 2, 0]}>

// CHECK-LABEL: @tensor_4d_divui
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_mul_uint32_t
// CHECK:       ttg.local_load {{.*}} -> tensor<1x1x1x1xi32, #{{.*}}>
// CHECK:       ttg.local_load {{.*}} -> tensor<1x1x1x1xi32, #{{.*}}>
// CHECK:       tt.broadcast {{.*}} : tensor<1x1x1x1xi32, {{.*}}> -> tensor<4x2x4x16xi32,
// CHECK:       tt.broadcast {{.*}} : tensor<1x1x1x1xi32, {{.*}}> -> tensor<4x2x4x16xi32,
// CHECK:       tt.mulhiui
// CHECK:       arith.shrui
// CHECK:       arith.muli
// CHECK-NOT:   arith.divui
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_4d_divui(%a: tensor<4x2x4x16xi32, #blocked4d>, %divisor: i32) -> tensor<4x2x4x16xi32, #blocked4d> {
    %splat = tt.splat %divisor : i32 -> tensor<4x2x4x16xi32, #blocked4d>
    %div = arith.divui %a, %splat : tensor<4x2x4x16xi32, #blocked4d>
    %result = arith.muli %div, %a : tensor<4x2x4x16xi32, #blocked4d>
    tt.return %result : tensor<4x2x4x16xi32, #blocked4d>
  }
}

// -----

// Test 19: nested scf.for loops with mixed divisors (constant 128 + func arg),
// divsi + remsi on 3-D tensors, and 4 warps.  Uses a non-trivial encoding with
// multi-dim sizePerThread [1,4,2], distributed threadsPerWarp [2,2,8] and
// warpsPerCTA [1,2,2], and non-standard order [2,0,1].
#blocked3d_4w = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [2, 2, 8], warpsPerCTA = [1, 2, 2], order = [2, 0, 1]}>

// CHECK-LABEL: @nested_loop_3d_mixed
//   two magic computations at entry:
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_mul_uint32_t
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_mul_uint32_t
//   outer loop:
// CHECK:       scf.for
//   inner loop:
// CHECK:         scf.for
//   div replacement:
// CHECK:           ttg.local_load {{.*}} -> tensor<1x1x1xi32, #{{.*}}>
// CHECK:           ttg.local_load {{.*}} -> tensor<1x1x1xi32, #{{.*}}>
// CHECK:           tt.broadcast {{.*}} : tensor<1x1x1xi32, {{.*}}> -> tensor<2x16x32xi32,
// CHECK:           tt.broadcast {{.*}} : tensor<1x1x1xi32, {{.*}}> -> tensor<2x16x32xi32,
// CHECK:           tt.mulhiui
// CHECK:           arith.shrsi
//   rem replacement:
// CHECK:           ttg.local_load {{.*}} -> tensor<1x1x1xi32, #{{.*}}>
// CHECK:           ttg.local_load {{.*}} -> tensor<1x1x1xi32, #{{.*}}>
// CHECK:           tt.broadcast {{.*}} : tensor<1x1x1xi32, {{.*}}> -> tensor<2x16x32xi32,
// CHECK:           tt.broadcast {{.*}} : tensor<1x1x1xi32, {{.*}}> -> tensor<2x16x32xi32,
// CHECK:           tt.mulhiui
// CHECK:           arith.shrsi
// CHECK:           arith.muli
// CHECK:           arith.subi
//   user code:
// CHECK:           arith.addi
// CHECK-NOT:   arith.divsi
// CHECK-NOT:   arith.remsi
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @nested_loop_3d_mixed(%a: tensor<2x16x32xi32, #blocked3d_4w>, %divisor: i32, %n: index, %m: index) -> tensor<2x16x32xi32, #blocked3d_4w> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : i32
    %splat_arg = tt.splat %divisor : i32 -> tensor<2x16x32xi32, #blocked3d_4w>
    %splat_const = tt.splat %c128 : i32 -> tensor<2x16x32xi32, #blocked3d_4w>
    %outer = scf.for %i = %c0 to %n step %c1 iter_args(%o_acc = %a) -> (tensor<2x16x32xi32, #blocked3d_4w>) {
      %inner = scf.for %j = %c0 to %m step %c1 iter_args(%i_acc = %o_acc) -> (tensor<2x16x32xi32, #blocked3d_4w>) {
        %div = arith.divsi %i_acc, %splat_const : tensor<2x16x32xi32, #blocked3d_4w>
        %rem = arith.remsi %i_acc, %splat_arg : tensor<2x16x32xi32, #blocked3d_4w>
        %sum = arith.addi %div, %rem : tensor<2x16x32xi32, #blocked3d_4w>
        scf.yield %sum : tensor<2x16x32xi32, #blocked3d_4w>
      }
      scf.yield %inner : tensor<2x16x32xi32, #blocked3d_4w>
    }
    tt.return %outer : tensor<2x16x32xi32, #blocked3d_4w>
  }
}

// -----

// Test 20: same function-argument divisor used by tensors of different ranks
// (1-D and 3-D).  The pass must create separate (divisor, rank) memdesc
// entries — a rank-1 memdesc for the 1-D use and a rank-3 memdesc for the
// 3-D use — so that each local_load result shape matches its memdesc.
#blocked1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked3d = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>

// CHECK-LABEL: @same_divisor_different_ranks
//   two (divisor, rank) pairs → four call_scalar ops:
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_mul_uint32_t
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK:       tt.call_scalar {{.*}}, @_mlir_ciface_simt_div_magic_mul_uint32_t
//   1-D replacement:
// CHECK:       ttg.local_load {{.*}} -> tensor<1xi32, #{{.*}}>
// CHECK:       ttg.local_load {{.*}} -> tensor<1xi32, #{{.*}}>
// CHECK:       tt.broadcast {{.*}} : tensor<1xi32, {{.*}}> -> tensor<32xi32,
// CHECK:       tt.broadcast {{.*}} : tensor<1xi32, {{.*}}> -> tensor<32xi32,
// CHECK:       tt.mulhiui
// CHECK:       arith.shrsi
//   3-D replacement:
// CHECK:       ttg.local_load {{.*}} -> tensor<1x1x1xi32, #{{.*}}>
// CHECK:       ttg.local_load {{.*}} -> tensor<1x1x1xi32, #{{.*}}>
// CHECK:       tt.broadcast {{.*}} : tensor<1x1x1xi32, {{.*}}> -> tensor<2x4x16xi32,
// CHECK:       tt.broadcast {{.*}} : tensor<1x1x1xi32, {{.*}}> -> tensor<2x4x16xi32,
// CHECK:       tt.mulhiui
// CHECK:       arith.shrsi
// CHECK-NOT:   arith.divsi
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @same_divisor_different_ranks(%a: tensor<32xi32, #blocked1d>, %b: tensor<2x4x16xi32, #blocked3d>, %divisor: i32) -> tensor<2x4x16xi32, #blocked3d> {
    %splat1d = tt.splat %divisor : i32 -> tensor<32xi32, #blocked1d>
    %splat3d = tt.splat %divisor : i32 -> tensor<2x4x16xi32, #blocked3d>
    %div1d = arith.divsi %a, %splat1d : tensor<32xi32, #blocked1d>
    %div3d = arith.divsi %b, %splat3d : tensor<2x4x16xi32, #blocked3d>
    %sum1d = arith.addi %div1d, %a : tensor<32xi32, #blocked1d>
    %result = arith.addi %div3d, %b : tensor<2x4x16xi32, #blocked3d>
    tt.return %result : tensor<2x4x16xi32, #blocked3d>
  }
}
