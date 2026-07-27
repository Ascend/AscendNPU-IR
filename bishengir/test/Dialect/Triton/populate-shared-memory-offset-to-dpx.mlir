// RUN: bishengir-opt -split-input-file -populate-shared-memory-offset-to-dpx %s | FileCheck %s

// Test 1: Scalar signed division.
//
// Input: tt.call_scalar ops backed by ttg.local_alloc with allocation.offset.
//        ttg.local_load results are unused (memory-ordering only).
//
// Expected:
//   - Each tt.call_scalar + local_alloc pair becomes ascend_dpx.call_scalar
//     carrying the offset as use_shmem_offset.
//   - The unused ttg.local_load ops are erased.
//   - No tt.call_scalar or ttg.local_alloc remain.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared  = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem    = #ttg.shared_memory

// CHECK-LABEL: @scalar_case
// CHECK-NOT:   tt.call_scalar
// CHECK-NOT:   ttg.local_alloc
// CHECK-NOT:   ttg.local_load
// CHECK:       [[SHIFT:%[^ ]+]] = ascend_dpx.call_scalar @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK-SAME:  {use_shmem_offset = 0 : i32}
// CHECK:       [[MAGIC:%[^ ]+]] = ascend_dpx.call_scalar @_mlir_ciface_simt_div_magic_mul_uint32_t
// CHECK-SAME:  {use_shmem_offset = 4 : i32}
// CHECK:       [[HI:%[^ ]+]] = ascend_dpx.umulhi {{%[^ ]+}}, [[MAGIC]]
// CHECK:       arith.shrsi [[HI]], [[SHIFT]]
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func private @_mlir_ciface_simt_div_magic_shift_uint32_t(i32) -> i32
  tt.func private @_mlir_ciface_simt_div_magic_mul_uint32_t(i32, i32) -> i32
  tt.func public @scalar_case(%arg0: i32, %arg1: i32) -> i32 {
    %shm0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %shift = tt.call_scalar %shm0, @_mlir_ciface_simt_div_magic_shift_uint32_t(%arg1) : !ttg.memdesc<1xi32, #shared, #smem, mutable>, (i32) -> i32
    %shm1 = ttg.local_alloc {allocation.offset = 4 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %magic = tt.call_scalar %shm1, @_mlir_ciface_simt_div_magic_mul_uint32_t(%arg1, %shift) : !ttg.memdesc<1xi32, #shared, #smem, mutable>, (i32, i32) -> i32
    %_0 = ttg.local_load %shm0 : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
    %_1 = ttg.local_load %shm1 : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
    %hi = ascend_dpx.umulhi %arg0, %magic : (i32, i32) -> i32
    %result = arith.shrsi %hi, %shift : i32
    tt.return %result : i32
  }
}

// -----

// Test 2: Tensor signed division.
//
// Input: tt.call_scalar backed by local_alloc; local_load feeds tt.broadcast.
//
// Expected:
//   - Each tt.call_scalar + local_alloc pair becomes ascend_dpx.call_scalar
//     carrying the offset as use_shmem_offset.
//   - Each ttg.local_load + tt.broadcast pair is replaced by tt.splat of the
//     new call's i32 result.
//   - No tt.call_scalar, ttg.local_alloc, ttg.local_load, or tt.broadcast remain.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared  = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem    = #ttg.shared_memory

// CHECK-LABEL: @tensor_case
// CHECK-NOT:   tt.call_scalar
// CHECK-NOT:   ttg.local_alloc
// CHECK-NOT:   ttg.local_load
// CHECK-NOT:   tt.broadcast
// CHECK:       [[SHIFT:%[^ ]+]] = ascend_dpx.call_scalar @_mlir_ciface_simt_div_magic_shift_uint32_t
// CHECK-SAME:  {use_shmem_offset = 0 : i32}
// CHECK:       [[MAGIC:%[^ ]+]] = ascend_dpx.call_scalar @_mlir_ciface_simt_div_magic_mul_uint32_t
// CHECK-SAME:  {use_shmem_offset = 4 : i32}
// CHECK:       tt.splat [[MAGIC]] : i32 -> tensor<32xi32,
// CHECK:       tt.splat [[SHIFT]] : i32 -> tensor<32xi32,
// CHECK:       tt.mulhiui
// CHECK:       arith.shrsi
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func private @_mlir_ciface_simt_div_magic_shift_uint32_t(i32) -> i32
  tt.func private @_mlir_ciface_simt_div_magic_mul_uint32_t(i32, i32) -> i32
  tt.func public @tensor_case(%arg0: tensor<32xi32, #blocked>, %arg1: i32) -> tensor<32xi32, #blocked> {
    %shm0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %shift = tt.call_scalar %shm0, @_mlir_ciface_simt_div_magic_shift_uint32_t(%arg1) : !ttg.memdesc<1xi32, #shared, #smem, mutable>, (i32) -> i32
    %shm1 = ttg.local_alloc {allocation.offset = 4 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %magic = tt.call_scalar %shm1, @_mlir_ciface_simt_div_magic_mul_uint32_t(%arg1, %shift) : !ttg.memdesc<1xi32, #shared, #smem, mutable>, (i32, i32) -> i32
    %ml = ttg.local_load %shm1 : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
    %sl = ttg.local_load %shm0 : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
    %mt = tt.broadcast %ml : tensor<1xi32, #blocked> -> tensor<32xi32, #blocked>
    %st = tt.broadcast %sl : tensor<1xi32, #blocked> -> tensor<32xi32, #blocked>
    %hi = tt.mulhiui %arg0, %mt : tensor<32xi32, #blocked>
    %result = arith.shrsi %hi, %st : tensor<32xi32, #blocked>
    tt.return %result : tensor<32xi32, #blocked>
  }
}

// -----

// Test 3: shm_desc comes from a function argument, not a ttg.local_alloc.
// The pass only converts tt.call_scalar ops whose shm_desc is produced by a
// local_alloc; all others are left untouched.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem   = #ttg.shared_memory

// CHECK-LABEL: @shm_from_arg_untouched
// CHECK:       tt.call_scalar
// CHECK-NOT:   ascend_dpx.call_scalar
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func private @_mlir_ciface_simt_div_magic_shift_uint32_t(i32) -> i32
  tt.func public @shm_from_arg_untouched(
      %arg0: i32, %arg1: i32,
      %shm: !ttg.memdesc<1xi32, #shared, #smem, mutable>) -> i32 {
    %shift = tt.call_scalar %shm, @_mlir_ciface_simt_div_magic_shift_uint32_t(%arg1)
        : !ttg.memdesc<1xi32, #shared, #smem, mutable>, (i32) -> i32
    tt.return %shift : i32
  }
}
