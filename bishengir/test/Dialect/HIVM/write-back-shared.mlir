// REQUIRES: a5

// RUN: bishengir-opt --write-back-shared --split-input-file %s | FileCheck %s

// CHECK: %[[SHARED:.*]] = memref.alloc() : memref<32xi8, #hivm.address_space<ub>>
// CHECK-NEXT: call @simple_indirect_load_kernel_scope_0(%arg3, %alloc, %arg2, %c1_i32, %alloc_0, %[[SHARED]], %arg5, %arg6, %arg7)
// CHECK-NOT: memref<1024xi8, #hivm.address_space<ub>>
module {
  module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
    func.func private @simple_indirect_load_kernel_scope_0(memref<?xi64, #hivm.address_space<gm>>, memref<8xi64, #hivm.address_space<ub>>, memref<?xf32, #hivm.address_space<gm>>, i32, memref<8xf32, #hivm.address_space<ub>>, memref<1024xi8, #hivm.address_space<ub>>, i32, i32, i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, no_inline, outline}

    func.func @simple_indirect_load_kernel(%arg0: memref<?xi8, #hivm.address_space<gm>>, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xf32, #hivm.address_space<gm>>, %arg3: memref<?xi64, #hivm.address_space<gm>>, %arg4: memref<?xf32, #hivm.address_space<gm>>, %arg5: i32, %arg6: i32, %arg7: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "mix_simd_simt"} {
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

  module attributes {hivm.simt_module, ttg.shared = 32 : i32} {
    llvm.func @simple_indirect_load_kernel_scope_0() attributes {nvvm.kernel = 1 : ui1} {
      llvm.return
    }
  }
}
