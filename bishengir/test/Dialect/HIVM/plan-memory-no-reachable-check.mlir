// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory=disable-vf-reachable-check=true -split-input-file -verify-diagnostics | FileCheck %s

module {
  func.func @vf_a(%kill_1: memref<32768xf32, #hivm.address_space<ub>>, %gen: memref<32768xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %kill_1[%c0], %cst {in_bounds = [true]} : memref<32768xf32, #hivm.address_space<ub>>, vector<32768xf32>
    vector.transfer_write %0, %gen[%c0] {in_bounds = [true]} : vector<32768xf32>, memref<32768xf32, #hivm.address_space<ub>>
    return
  }

  func.func @vf_b(%kill: memref<32768xf32, #hivm.address_space<ub>>, %gen: memref<32768xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %kill[%c0], %cst {in_bounds = [true]} : memref<32768xf32, #hivm.address_space<ub>>, vector<32768xf32>
    vector.transfer_write %0, %gen[%c0] {in_bounds = [true]} : vector<32768xf32>, memref<32768xf32, #hivm.address_space<ub>>
    return
  }
// CHECK: hivm.hir.pointer_cast({{.*}})
  func.func @kernel(%gm: memref<32768xf32, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMD>} {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    scf.for %iv = %c0 to %c256 step %c64 {
      %kill_1 = memref.alloc() : memref<32768xf32, #hivm.address_space<ub>>
      hivm.hir.load ins(%gm : memref<32768xf32, #hivm.address_space<gm>>) outs(%kill_1 : memref<32768xf32, #hivm.address_space<ub>>) eviction_policy = <EvictFirst>
      %gen = memref.alloc() : memref<32768xf32, #hivm.address_space<ub>>
      func.call @vf_a(%kill_1, %gen) {hivm.vector_function, no_inline} : (memref<32768xf32, #hivm.address_space<ub>>, memref<32768xf32, #hivm.address_space<ub>>) -> ()
      %store = memref.alloc() : memref<32768xf32, #hivm.address_space<ub>>
      func.call @vf_b(%gen, %store) {hivm.vector_function, no_inline} : (memref<32768xf32, #hivm.address_space<ub>>, memref<32768xf32, #hivm.address_space<ub>>) -> ()
      hivm.hir.store ins(%store : memref<32768xf32, #hivm.address_space<ub>>) outs(%gm : memref<32768xf32, #hivm.address_space<gm>>)
    }
    return
  }
}
