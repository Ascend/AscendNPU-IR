// RUN: bishengir-opt --hoist-simt-scalar-calls-to-simd -split-input-file %s | FileCheck %s

// CHECK: func.func private @gather_div_scope_0
// CHECK-SAME: memref<20xi8> {hivm.memory_effect = #hivm.memory_effect<read>, hivm.shared_memory

// CHECK-LABEL: func.func @gather_div
// CHECK:         %[[SH:.*]] = call @_mlir_ciface_simt_div_magic_shift_uint32_t(%arg2) : (i32) -> i32
// CHECK:         %[[V0:.*]] = memref.view %[[BUF:.*]][%{{.*}}][] : memref<20xi8> to memref<1xi32>
// CHECK:         memref.store %[[SH]], %[[V0]][%{{.*}}]
// CHECK:         %[[MG:.*]] = call @_mlir_ciface_simt_div_magic_mul_uint32_t(%arg2, %[[SH]]) : (i32, i32) -> i32
// CHECK:         %[[V1:.*]] = memref.view %[[BUF]][%{{.*}}][] : memref<20xi8> to memref<1xi32>
// CHECK:         memref.store %[[MG]], %[[V1]][%{{.*}}]
// CHECK:         call @gather_div_scope_0(%{{.*}}, %arg2, %{{.*}}, %[[BUF]],

// CHECK: func.func private @_mlir_ciface_simt_div_magic_shift_uint32_t(i32) -> i32 attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>}
// CHECK: func.func private @_mlir_ciface_simt_div_magic_mul_uint32_t(i32, i32) -> i32 attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>}
// CHECK:      llvm.func @gather_div_scope_0(%arg0: !llvm.ptr<6>, %arg1: i32, %arg2: !llvm.ptr<6>, %arg3: !llvm.ptr<6> {hivm.shared_memory}, %arg4: i32 {gpu.block = #gpu.block<x>}, %arg5: i32 {gpu.block = #gpu.block<y>}, %arg6: i32 {gpu.block = #gpu.block<z>}, %arg7: !llvm.ptr<1>)
// CHECK-NOT:  llvm.call @_mlir_ciface_simt_div_magic
// CHECK-NOT:  use_shmem_offset
// CHECK:      llvm.func @gather_div_scope_0_vf_simt(
module {
  module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
    func.func private @gather_div_scope_0(memref<64xi32> {hivm.memory_effect = #hivm.memory_effect<read>, hivm.simt_mem_scope_hint = #hivm.simt_mem_scope_hint<ub>}, i32, memref<64xf32> {hivm.memory_effect = #hivm.memory_effect<write>, hivm.simt_mem_scope_hint = #hivm.simt_mem_scope_hint<ub>}, memref<20xi8> {hivm.shared_memory, hivm.simt_mem_scope_hint = #hivm.simt_mem_scope_hint<ub>}, i32, i32, i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, no_inline, outline}

    func.func @gather_div(%arg0: memref<?xi32>, %arg1: memref<?xf32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<MIX>, mix_mode = "aiv", parallel_mode = "mix_simd_simt"} {
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [64], strides: [1] : memref<?xi32> to memref<64xi32, strided<[1]>>
      %alloc = memref.alloc() : memref<64xi32>
      hivm.hir.load ins(%reinterpret_cast : memref<64xi32, strided<[1]>>) outs(%alloc : memref<64xi32>) core_type = <VECTOR>
      %alloc_0 = memref.alloc() : memref<64xf32>
      %alloc_1 = memref.alloc() : memref<20xi8>
      call @gather_div_scope_0(%alloc, %arg2, %alloc_0, %alloc_1, %arg3, %arg4, %arg5) : (memref<64xi32>, i32, memref<64xf32>, memref<20xi8>, i32, i32, i32) -> ()
      %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1]>>
      hivm.hir.store ins(%alloc_0 : memref<64xf32>) outs(%reinterpret_cast_2 : memref<64xf32, strided<[1]>>)
      return
    }
  }
  module attributes {hacc.simt_module, hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>, "ttg.num-warps" = 4 : i32, ttg.shared = 20 : i32, "ttg.threads-per-warp" = 32 : i32} {
    llvm.func @_mlir_ciface_simt_div_magic_mul_uint32_t(i32, i32) -> i32
    llvm.func @_mlir_ciface_simt_div_magic_shift_uint32_t(i32) -> i32

    llvm.func @gather_div_scope_0(%arg0: !llvm.ptr<6>, %arg1: i32, %arg2: !llvm.ptr<6>, %arg3: !llvm.ptr<6> {hivm.shared_memory}, %arg4: i32 {gpu.block = #gpu.block<x>}, %arg5: i32 {gpu.block = #gpu.block<y>}, %arg6: i32 {gpu.block = #gpu.block<z>}, %arg7: !llvm.ptr<1>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm_regbaseintrins.kernel, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">} {
      %0 = llvm.mlir.constant(128 : i64) : i64
      %1 = llvm.mlir.constant(1 : i64) : i64
      %2 = llvm.call @_mlir_ciface_simt_div_magic_shift_uint32_t(%arg1) : (i32) -> i32
      %3 = llvm.getelementptr %arg3[0] : (!llvm.ptr<6>) -> !llvm.ptr<6>, i8
      llvm.store %2, %3 {use_shmem_offset = 0 : i32} : i32, !llvm.ptr<6>
      %4 = llvm.call @_mlir_ciface_simt_div_magic_mul_uint32_t(%arg1, %2) : (i32, i32) -> i32
      %5 = llvm.getelementptr %arg3[16] : (!llvm.ptr<6>) -> !llvm.ptr<6>, i8
      llvm.store %4, %5 {use_shmem_offset = 16 : i32} : i32, !llvm.ptr<6>
      hivm_regbaseintrins.intrins.launch_func @gather_div_scope_0_vf_simt threads in (%0, %1, %1) args(%arg0, %arg1, %arg2, %arg3) : !llvm.ptr<6>, i32, !llvm.ptr<6>, !llvm.ptr<6>
      llvm.return
    }

    llvm.func @gather_div_scope_0_vf_simt(%arg0: !llvm.ptr<6>, %arg1: i32, %arg2: !llvm.ptr<6>, %arg3: !llvm.ptr<6> {hivm.shared_memory}) attributes {hivm_regbaseintrins.cconv = #hivm_regbaseintrins.simt_entry<128>, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>} {
      %0 = llvm.getelementptr %arg3[0] : (!llvm.ptr<6>) -> !llvm.ptr<6>, i8
      %1 = llvm.load %0 : !llvm.ptr<6> -> i32
      %2 = llvm.getelementptr %arg3[16] : (!llvm.ptr<6>) -> !llvm.ptr<6>, i8
      %3 = llvm.load %2 : !llvm.ptr<6> -> i32
      %4 = ascend_dpx.thread_id_x
      %5 = ascend_dpx.umulhi %4, %3 : (i32, i32) -> i32
      %6 = llvm.add %5, %4 : i32
      %7 = llvm.ashr %6, %1 : i32
      %8 = llvm.getelementptr %arg2[%7] : (!llvm.ptr<6>, i32) -> !llvm.ptr<6>, i32
      llvm.store %7, %8 : i32, !llvm.ptr<6>
      llvm.return
    }
  }
}

// -----

// No hoisted scalar calls: the pass must not touch the caller, and in particular
// must not add a memory effect or a view into the zero-byte buffer that
// kernels without fast-div carry.

// CHECK:       func.func private @no_scalar_calls_scope_0(i32, memref<0xi8> {hivm.shared_memory}, i32, i32, i32)
// CHECK-LABEL: func.func @no_scalar_calls
// CHECK-NOT:     memref.view
// CHECK:         call @no_scalar_calls_scope_0
module {
  module {
    func.func private @no_scalar_calls_scope_0(i32, memref<0xi8> {hivm.shared_memory}, i32, i32, i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>}
    func.func @no_scalar_calls(%d: i32, %gx: i32, %gy: i32, %gz: i32) attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
      %alloc = memref.alloc() : memref<0xi8>
      call @no_scalar_calls_scope_0(%d, %alloc, %gx, %gy, %gz) : (i32, memref<0xi8>, i32, i32, i32) -> ()
      return
    }
  }
  module attributes {hacc.simt_module} {
    llvm.func @no_scalar_calls_scope_0(%arg0: i32, %arg1: !llvm.ptr<6> {hivm.shared_memory}, %arg2: i32 {gpu.block = #gpu.block<x>}, %arg3: i32 {gpu.block = #gpu.block<y>}, %arg4: i32 {gpu.block = #gpu.block<z>}) attributes {hacc.entry} {
      llvm.return
    }
  }
}

// -----

// No SIMT module at all: fast-div disabled, or nothing to optimize. The pass
// must not fail, and must not object to the second main module it has no
// reason to look at.

// CHECK-LABEL: func.func @no_simt_module
module {
  module {
    func.func @no_simt_module() {
      return
    }
  }
  module {
    func.func @second_main() {
      return
    }
  }
}

// -----

// Descriptor mode: a dynamic memref makes the whole signature convert with
// descriptors, so the memref takes 3 + 2*rank wrapper slots and the divisor
// sits at wrapper position 5. It must resolve to operand 1, the i32 -- under
// the bare-pointer layout position 5 would fall off the end of the signature.

// CHECK-LABEL: func.func @desc_div
// CHECK:         %[[SH:.*]] = call @_mlir_ciface_simt_div_magic_shift_uint32_t(%arg1) : (i32) -> i32
// CHECK:         memref.store %[[SH]], %{{.*}}
// CHECK:         call @desc_div_scope_0
module {
  module {
    func.func private @desc_div_scope_0(memref<?xf32>, i32, memref<20xi8> {hivm.shared_memory}, i32, i32, i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>}
    func.func @desc_div(%buf: memref<?xf32>, %d: i32, %gx: i32, %gy: i32, %gz: i32) attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
      %alloc = memref.alloc() : memref<20xi8>
      call @desc_div_scope_0(%buf, %d, %alloc, %gx, %gy, %gz) : (memref<?xf32>, i32, memref<20xi8>, i32, i32, i32) -> ()
      return
    }
  }
  module attributes {hacc.simt_module} {
    llvm.func @_mlir_ciface_simt_div_magic_shift_uint32_t(i32) -> i32
    llvm.func @desc_div_scope_0(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i32, %arg6: !llvm.ptr<6> {hivm.shared_memory}, %arg7: !llvm.ptr<6>, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i32 {gpu.block = #gpu.block<x>}, %arg12: i32 {gpu.block = #gpu.block<y>}, %arg13: i32 {gpu.block = #gpu.block<z>}) attributes {hacc.entry} {
      %0 = llvm.call @_mlir_ciface_simt_div_magic_shift_uint32_t(%arg5) : (i32) -> i32
      %1 = llvm.getelementptr %arg6[0] : (!llvm.ptr<6>) -> !llvm.ptr<6>, i8
      llvm.store %0, %1 {use_shmem_offset = 0 : i32} : i32, !llvm.ptr<6>
      llvm.return
    }
  }
}
