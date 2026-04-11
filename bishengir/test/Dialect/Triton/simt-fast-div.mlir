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
