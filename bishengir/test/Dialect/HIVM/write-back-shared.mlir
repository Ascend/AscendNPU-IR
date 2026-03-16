// RUN: bishengir-opt --write-back-shared --split-input-file %s | FileCheck %s

// CHECK: %[[SHARED:.*]] = memref.alloc() : memref<[[VALUE:.*]]xi8, #hivm.address_space<ub>>
// CHECK-NEXT: call @simple_indirect_load_kernel_scope_0(%arg3, %alloc, %arg2, %c1_i32, %alloc_0, %[[SHARED]], %arg5, %arg6, %arg7) : (memref<?xi64, #hivm.address_space<gm>>, memref<8xi64, #hivm.address_space<ub>>, memref<?xf32, #hivm.address_space<gm>>, i32, memref<8xf32, #hivm.address_space<ub>>, memref<[[VALUE]]xi8, #hivm.address_space<ub>>, i32, i32, i32)
// CHECK: ttg.shared = [[VALUE]]
module {
  module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">} {
    func.func private @simple_indirect_load_kernel_scope_0(memref<?xi64, #hivm.address_space<gm>> {hivm.memory_effect = #hivm.memory_effect<read>}, memref<8xi64, #hivm.address_space<ub>>, memref<?xf32, #hivm.address_space<gm>> {hivm.memory_effect = #hivm.memory_effect<read>}, i32, memref<8xf32, #hivm.address_space<ub>> {hivm.memory_effect = #hivm.memory_effect<write>}, memref<1024xi8, #hivm.address_space<ub>>, i32, i32, i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, no_inline, outline}
    func.func @simple_indirect_load_kernel(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "mix_simd_simt"} {
      %c1_i32 = arith.constant 1 : i32
      %0 = arith.muli %arg5, %arg6 : i32
      %1 = arith.muli %0, %arg7 : i32
      annotation.mark %1 {logical_block_num} : i32
      %alloc = memref.alloc() : memref<8xi64, #hivm.address_space<ub>>
      %alloc_0 = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
      %alloc_1 = memref.alloc() {hivm.shared_memory} : memref<1024xi8, #hivm.address_space<ub>>
      call @simple_indirect_load_kernel_scope_0(%arg3, %alloc, %arg2, %c1_i32, %alloc_0, %alloc_1, %arg5, %arg6, %arg7) : (memref<?xi64, #hivm.address_space<gm>>, memref<8xi64, #hivm.address_space<ub>>, memref<?xf32, #hivm.address_space<gm>>, i32, memref<8xf32, #hivm.address_space<ub>>, memref<1024xi8, #hivm.address_space<ub>>, i32, i32, i32) -> ()
      %2 = bufferization.to_tensor %alloc_0 restrict : memref<8xf32, #hivm.address_space<ub>>
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [0], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1]>, #hivm.address_space<gm>>
      hivm.hir.store ins(%2 : tensor<8xf32>) outs(%reinterpret_cast : memref<8xf32, strided<[1]>, #hivm.address_space<gm>>)
      return
    }
  }
  module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.simt_module, hacc.target = #hacc.target<"Ascend910_9589">, "ttg.enable-bishengir-simt-optimization" = 900001 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 32 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
    llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
    llvm.func @simple_indirect_load_kernel_scope_0(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<6>, %arg2: !llvm.ptr<1>, %arg3: i32, %arg4: !llvm.ptr<6>, %arg5: !llvm.ptr<6>, %arg6: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg7: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg8: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}, %arg9: !llvm.ptr<1>, %arg10: !llvm.ptr<1>) attributes {nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>} {
      %0 = llvm.mlir.undef : vector<1xf32>
      %1 = llvm.mlir.undef : vector<1xi64>
      %2 = llvm.mlir.constant(3 : i32) : i32
      %3 = llvm.mlir.constant(24 : i32) : i32
      %4 = llvm.mlir.constant(7 : i32) : i32
      %5 = llvm.mlir.constant(5 : i32) : i32
      %6 = llvm.mlir.constant(0 : i32) : i32
      %7 = llvm.mlir.constant(32 : i32) : i32
      %8 = llvm.mlir.constant(0 : index) : i32
      %9 = nvvm.read.ptx.sreg.tid.x : i32
      %10 = llvm.urem %9, %7  : i32
      %11 = llvm.udiv %9, %7  : i32
      %12 = llvm.shl %10, %6 : i32
      %13 = llvm.or %6, %12  : i32
      %14 = llvm.shl %11, %5 : i32
      %15 = llvm.or %13, %14  : i32
      %16 = llvm.or %15, %6  : i32
      %17 = llvm.and %16, %4  : i32
      %18 = llvm.lshr %17, %6  : i32
      %19 = llvm.or %18, %6  : i32
      %20 = llvm.xor %6, %19  : i32
      %21 = llvm.xor %20, %6  : i32
      %22 = llvm.add %21, %8 : i32
      %23 = llvm.getelementptr %arg1[%22] : (!llvm.ptr<6>, i32) -> !llvm.ptr<6>, i64
      %24 = llvm.getelementptr %arg0[%22] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i64
      %25 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u64 $0, 0x0;\0A\09ld.global.L1::evict_first.b64 { $0 }, [ $1 + 0 ];", "=l,l" %24 : (!llvm.ptr<1>) -> i64
      %26 = llvm.bitcast %25 : i64 to vector<1xi64>
      %27 = llvm.extractelement %26[%8 : i32] : vector<1xi64>
      %28 = nvvm.read.ptx.sreg.tid.x : i32
      %29 = llvm.urem %28, %7  : i32
      %30 = llvm.udiv %28, %7  : i32
      %31 = llvm.and %29, %3  : i32
      %32 = llvm.icmp "eq" %31, %6 : i32
      %33 = llvm.and %30, %2  : i32
      %34 = llvm.icmp "eq" %33, %6 : i32
      %35 = llvm.and %32, %34  : i1
      %36 = llvm.insertelement %27, %1[%6 : i32] : vector<1xi64>
      %37 = llvm.bitcast %36 : vector<1xi64> to i64
      %38 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$2 st.global.b64 [ $1 + 0 ], { $0 };", "l,l,b" %37, %23, %35 : (i64, !llvm.ptr<6>, i1) -> !llvm.void
      %39 = llvm.getelementptr %arg4[%22] : (!llvm.ptr<6>, i32) -> !llvm.ptr<6>, f32
      %40 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u64 $0, 0x0;\0A\09ld.global.b64 { $0 }, [ $1 + 0 ];", "=l,l" %23 : (!llvm.ptr<6>) -> i64
      %41 = llvm.bitcast %40 : i64 to vector<1xi64>
      %42 = llvm.extractelement %41[%8 : i32] : vector<1xi64>
      %43 = llvm.getelementptr %arg2[%42] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      %44 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, 0x0;\0A\09ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l" %43 : (!llvm.ptr<1>) -> i32
      %45 = llvm.bitcast %44 : i32 to vector<1xf32>
      %46 = llvm.extractelement %45[%8 : i32] : vector<1xf32>
      %47 = nvvm.read.ptx.sreg.tid.x : i32
      %48 = llvm.urem %47, %7  : i32
      %49 = llvm.udiv %47, %7  : i32
      %50 = llvm.and %48, %3  : i32
      %51 = llvm.icmp "eq" %50, %6 : i32
      %52 = llvm.and %49, %2  : i32
      %53 = llvm.icmp "eq" %52, %6 : i32
      %54 = llvm.and %51, %53  : i1
      %55 = llvm.insertelement %46, %0[%6 : i32] : vector<1xf32>
      %56 = llvm.bitcast %55 : vector<1xf32> to i32
      %57 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b" %56, %39, %54 : (i32, !llvm.ptr<6>, i1) -> !llvm.void
      llvm.return
    }
  }
}
