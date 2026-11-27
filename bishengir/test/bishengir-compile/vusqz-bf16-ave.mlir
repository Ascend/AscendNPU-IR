// REQUIRES: hivmc
//
// RUN: bishengir-compile %s --target=Ascend950PR_9579 --enable-triton-kernel-compile=true --enable-tla-mode --save-temps=%t -o %t.o
// RUN: FileCheck %s --input-file=%t/module.hivm.opt.mlir --check-prefix=AVE
//
// The bf16 zero initializer feeds VBR, and VBR feeds the prefix count.
// AVE: 0.000000e+00 : bf16
// AVE: "hivm_regbaseintrins.intr.hivm.vbr"
// AVE: "hivm_regbaseintrins.intr.hivm.vusqz.m"

#b16 = #ave.vector_layout<{mem = #ave.vec_mem_type<b16>}>

module attributes {
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  func.func @vusqz_bf16_kernel(
      %sync: memref<?xi8>,
      %workspace: memref<?xi8>,
      %outputBase: memref<?xbf16> {
        tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32
      },
      %gridX: i32, %gridY: i32, %gridZ: i32,
      %programIdX: i32, %programIdY: i32, %programIdZ: i32)
      attributes {
        SyncBlockLockArgIdx = 0 : i64,
        WorkspaceArgIdx = 1 : i64,
        global_kernel = "local",
        hivm.func_core_type = #hivm.func_core_type<AIV>,
        hivm.vf_mode = #hivm.vf_mode<SIMD>,
        mix_mode = "aiv",
        parallel_mode = "simd"
      } {
    scope.scope : () -> () {
      %outputGM = memref.reinterpret_cast %outputBase to
          offset: [0], sizes: [128], strides: [1]
          : memref<?xbf16> to memref<128xbf16, strided<[1]>>
      %outputStorage = memref.alloc()
          : memref<128xbf16, #hivm.address_space<ub>>
      annotation.mark %outputStorage {mem_unique}
          : memref<128xbf16, #hivm.address_space<ub>>
      %outputUB = memref.subview %outputStorage[0] [128] [1]
          : memref<128xbf16, #hivm.address_space<ub>> to
            memref<128xbf16, #hivm.address_space<ub>>

      scope.scope : () -> () {
        %c0 = arith.constant 0 : index
        %mask = ave.hir.pge <ALL>
            {functionType = #ave.func_dist_type<pb16>}
            : vector<128xi1, #b16>
        // Every active lane becomes its exclusive prefix count: 0 .. 127,
        // encoded as bf16 values, which are exact over that range.
        %counts = ave.hir.vusqz %mask
            : vector<128xi1, #b16> -> vector<128xbf16, #b16>
        ave.hir.masked_store <NORM_B16> %outputStorage[%c0], %mask, %counts
            {functionType = #ave.func_dist_type<norm>}
            : memref<128xbf16, #hivm.address_space<ub>>,
              vector<128xi1, #b16>, vector<128xbf16, #b16>
        scope.return
      } {
        hivm.func_core_type = #hivm.func_core_type<AIV>,
        hivm.vector_function,
        hivm.vf_mode = #hivm.vf_mode<SIMD>,
        no_inline,
        outline
      }

      hivm.hir.store
          ins(%outputUB : memref<128xbf16, #hivm.address_space<ub>>)
          outs(%outputGM : memref<128xbf16, strided<[1]>>)
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    return
  }
}
