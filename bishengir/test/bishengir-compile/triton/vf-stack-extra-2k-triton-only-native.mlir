// Regression test for the Triton-only gating of the extra 2K VF stack
// options (MR !4011) on the Native backend (in-process libbisheng-llc.so):
// compileDeviceKernel translates --cce-disable-asc-reserved-ubuf and
// -mllvm -cce-vf-stack-size=0x2000 into the extraOpts entries
// -user-reserved-ub-size=0 and -cce-vf-stack-size=0x2000, and both must be
// appended only for Triton-originated kernels. hivmc also serves the
// catlass DSL and other frontends, which must keep the default reserved
// AscendC UB layout and the 6 KB (0x1800) VF stack. The Process-backend
// command line is covered by the sibling file
// vf-stack-extra-2k-triton-only.mlir.
//
// The Native path never echoes the ccec argv, so this test asserts on the
// BISHENG_NATIVE_DUMP line, which prints the extraOpts list after it is
// fully built (HIVMCMainA5.cpp, "Debug: dump native compile inputs").
// The compile's exit status is dropped via `;` sequencing: the dump is
// printed before submitCompile, so the FileCheck status alone decides and
// the test stays valid even if the vendor SO fails or crashes afterwards.
//
// Requires the build to have BISHENGIR_HAVE_BISHENG_LLC=1, the vendor SO
// libbisheng-llc.so present, AND host glibc >= 2.35 (or
// BISHENG_NATIVE_DLMOPEN_FORCE set) so the driver does not silently fall
// back to Process mode. The lit feature `bisheng-native` encodes all three
// conditions; without it this test is UNSUPPORTED (skipped).
//
// The Native llc infrastructure is not exercised on aarch64 hosts by any
// other test (compile-triton-vf-stack-overflow-retry-native.mlir skips the
// platform), so this test follows the same guard instead of asserting on
// unverified native behavior there.
// UNSUPPORTED: host-aarch64
// REQUIRES: enable-lir-compile, bisheng-native
//
// Triton frontend: the dumped extraOpts carry both translated entries, in
// the order compileDeviceKernel pushes them (reserved-ub first).
// RUN: BISHENG_BACKEND=native BISHENG_NATIVE_DUMP=1 bishengir-compile %s --target=Ascend950PR_9579 --enable-tla-mode --enable-triton-kernel-compile=true -o %t.triton.o 2> %t.triton.log ; FileCheck --check-prefix=TRITON-NATIVE --input-file=%t.triton.log %s
//
// Catlass DSL shape (no triton marker): neither entry may appear anywhere
// in the log; --implicit-check-not scans the whole file.
// RUN: BISHENG_BACKEND=native BISHENG_NATIVE_DUMP=1 bishengir-compile %s --target=Ascend950PR_9579 --enable-tla-mode -o %t.catlass.o 2> %t.catlass.log ; FileCheck --check-prefix=CATLASS-NATIVE --input-file=%t.catlass.log %s --implicit-check-not=-user-reserved-ub-size=0 --implicit-check-not=-cce-vf-stack-size=0x2000
//
// Both runs must reach the native device compile, otherwise the checks
// below are vacuous: the matched line is the per-kernel dump.
// TRITON-NATIVE: {{\[}}BishengLlcBridge dump] cpu={{.*}}extraOpts={{.*}}-user-reserved-ub-size=0{{.*}}-cce-vf-stack-size=0x2000
// CATLASS-NATIVE: {{\[}}BishengLlcBridge dump] cpu={{.*}}extraOpts=

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
