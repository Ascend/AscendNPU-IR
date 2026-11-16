// REQUIRES: hivmc
//
// Regression test for the Triton-only gating of the extra 2K VF stack
// options (MR !4011): compileDeviceKernel appends
// --cce-disable-asc-reserved-ubuf and -mllvm -cce-vf-stack-size=0x2000 to
// the external bisheng device invocation. hivmc also serves the catlass DSL
// and other frontends, which must keep the default reserved AscendC UB
// layout and the 6 KB (0x1800) VF stack. The frontend is identified by the
// existing triton markers: --enable-triton-kernel-compile (triton-adaptor
// flow) or --enable-triton-ir-compile (pure-SIMT triton flow); every other
// producer sets neither.
//
// Both runs compile the same kernel through the TLA affinity pipeline and
// echo the executed ccec command lines via --save-temps; only the frontend
// marker differs. BISHENG_BACKEND=process is pinned: the assertions match
// the external ccec command line, which only the Process backend echoes
// (the Native backend compiles in-process and never executes ccec; on a
// host where libbisheng-llc.so resolves, the default backend is Native).
//
// Triton frontend: the echoed device compile carries both options.
// RUN: BISHENG_BACKEND=process bishengir-compile %s --target=Ascend950PR_9579 --enable-tla-mode --enable-triton-kernel-compile=true --save-temps=%t.triton -o %t.triton.o 2> %t.triton.log
// RUN: test -s %t.triton.o
// RUN: FileCheck --check-prefix=TRITON --input-file=%t.triton.log %s
//
// Catlass DSL shape (no triton marker): neither option may appear anywhere
// in the log; --implicit-check-not scans the whole file.
// RUN: BISHENG_BACKEND=process bishengir-compile %s --target=Ascend950PR_9579 --enable-tla-mode --save-temps=%t.catlass -o %t.catlass.o 2> %t.catlass.log
// RUN: test -s %t.catlass.o
// RUN: FileCheck --check-prefix=CATLASS --input-file=%t.catlass.log %s --implicit-check-not=--cce-disable-asc-reserved-ubuf --implicit-check-not=-cce-vf-stack-size=0x2000
//
// Both runs must reach the external device compile, otherwise the checks
// below are vacuous: the matched line is the bisheng/ccec device invocation.
// TRITON: [save-temps] Executing: {{.*}}--cce-aicore-only{{.*}}--cce-disable-asc-reserved-ubuf{{.*}}-cce-vf-stack-size=0x2000
// CATLASS: [save-temps] Executing: {{.*}}--cce-aicore-only

#b32 = #ave.vector_layout<{mem = #ave.vec_mem_type<b32>}>

module attributes {
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  func.func @auto_sync_ave_kernel(
      %sync: memref<?xi8>,
      %workspace: memref<?xi8>,
      %inputBase: memref<?xf32> {
        tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32
      },
      %outputBase: memref<?xf32> {
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
      %inputGM = memref.reinterpret_cast %inputBase to
          offset: [0], sizes: [16], strides: [1]
          : memref<?xf32> to memref<16xf32, strided<[1]>>
      %outputGM = memref.reinterpret_cast %outputBase to
          offset: [0], sizes: [16], strides: [1]
          : memref<?xf32> to memref<16xf32, strided<[1]>>
      %inputStorage = memref.alloc()
          : memref<64xf32, #hivm.address_space<ub>>
      %outputStorage = memref.alloc()
          : memref<64xf32, #hivm.address_space<ub>>
      annotation.mark %inputStorage {mem_unique}
          : memref<64xf32, #hivm.address_space<ub>>
      annotation.mark %outputStorage {mem_unique}
          : memref<64xf32, #hivm.address_space<ub>>
      %inputUB = memref.subview %inputStorage[0] [16] [1]
          : memref<64xf32, #hivm.address_space<ub>> to
            memref<16xf32, #hivm.address_space<ub>>
      %outputUB = memref.subview %outputStorage[0] [16] [1]
          : memref<64xf32, #hivm.address_space<ub>> to
            memref<16xf32, #hivm.address_space<ub>>

      hivm.hir.load
          ins(%inputGM : memref<16xf32, strided<[1]>>)
          outs(%inputUB : memref<16xf32, #hivm.address_space<ub>>)
          eviction_policy = <EvictFirst> core_type = <VECTOR>

      scope.scope : () -> () {
        %c0 = arith.constant 0 : index
        %mask = ave.hir.pge <VL16>
            {functionType = #ave.func_dist_type<pb32>}
            : vector<64xi1, #b32>
        %inputVec = ave.hir.vload <NORM> %inputStorage[%c0]
            {functionType = #ave.func_dist_type<norm>}
            : memref<64xf32, #hivm.address_space<ub>> into
              vector<64xf32, #b32>
        %outputVec = ave.hir.vexp %inputVec, %mask
            : vector<64xf32, #b32>, vector<64xi1, #b32>
        ave.hir.masked_store <NORM_B32> %outputStorage[%c0], %mask, %outputVec
            {functionType = #ave.func_dist_type<norm>}
            : memref<64xf32, #hivm.address_space<ub>>,
              vector<64xi1, #b32>, vector<64xf32, #b32>
        scope.return
      } {
        hivm.func_core_type = #hivm.func_core_type<AIV>,
        hivm.vector_function,
        hivm.vf_mode = #hivm.vf_mode<SIMD>,
        no_inline,
        outline
      }

      hivm.hir.store
          ins(%outputUB : memref<16xf32, #hivm.address_space<ub>>)
          outs(%outputGM : memref<16xf32, strided<[1]>>)
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}

    return
  }
}
