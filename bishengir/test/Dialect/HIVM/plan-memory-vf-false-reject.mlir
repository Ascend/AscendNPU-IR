// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory-regbase -split-input-file | FileCheck %s

// Two kill buffers cannot both be reused by gen (overlap); gen and store can
// share. Expect two physical UB slots (offsets 0 and 256).
// CHECK-LABEL: func.func @kernel_two_kill
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[C256:.*]] = arith.constant 256 : i64
// CHECK-DAG: hivm.hir.pointer_cast(%[[C0]])
// CHECK-DAG: hivm.hir.pointer_cast(%[[C256]])
// CHECK-NOT: pointer_cast(%{{.*}}512

module {
  func.func @vf_a(%kill_1: memref<64xf32, #hivm.address_space<ub>>, %kill_2: memref<64xf32, #hivm.address_space<ub>>, %gen: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %kill_1[%c0], %cst {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %1 = vector.transfer_read %kill_2[%c0], %cst {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %2 = arith.mulf %0, %1 : vector<64xf32>
    vector.transfer_write %2, %gen[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }

  func.func @vf_b(%kill: memref<64xf32, #hivm.address_space<ub>>, %gen: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %kill[%c0], %cst {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    vector.transfer_write %0, %gen[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }

  func.func @kernel_two_kill(%gm: memref<64xf32, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMD>} {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    scf.for %iv = %c0 to %c256 step %c64 {
      %kill_1 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
      %kill_2 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.load ins(%gm : memref<64xf32, #hivm.address_space<gm>>) outs(%kill_1 : memref<64xf32, #hivm.address_space<ub>>) eviction_policy = <EvictFirst>
      hivm.hir.load ins(%gm : memref<64xf32, #hivm.address_space<gm>>) outs(%kill_2 : memref<64xf32, #hivm.address_space<ub>>) eviction_policy = <EvictFirst>

      %gen = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
      func.call @vf_a(%kill_1, %kill_2, %gen) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()

      %store = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
      func.call @vf_b(%gen, %store) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
      hivm.hir.store ins(%store : memref<64xf32, #hivm.address_space<ub>>) outs(%gm : memref<64xf32, #hivm.address_space<gm>>)
    }
    return
  }
}

// -----

// Single kill can be reused through gen into store — one physical UB slot.
// CHECK-LABEL: func.func @kernel_one_kill
// CHECK: pointer_cast(%{{.*}}0
// CHECK-NOT: pointer_cast(%{{.*}}256

module {
  func.func @vf_a(%kill_1: memref<64xf32, #hivm.address_space<ub>>, %gen: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %kill_1[%c0], %cst {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    vector.transfer_write %0, %gen[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }

  func.func @vf_b(%kill: memref<64xf32, #hivm.address_space<ub>>, %gen: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %kill[%c0], %cst {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    vector.transfer_write %0, %gen[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }

  func.func @kernel_one_kill(%gm: memref<64xf32, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMD>} {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    scf.for %iv = %c0 to %c256 step %c64 {
      %kill_1 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.load ins(%gm : memref<64xf32, #hivm.address_space<gm>>) outs(%kill_1 : memref<64xf32, #hivm.address_space<ub>>) eviction_policy = <EvictFirst>

      %gen = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
      func.call @vf_a(%kill_1, %gen) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()

      %store = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
      func.call @vf_b(%gen, %store) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
      hivm.hir.store ins(%store : memref<64xf32, #hivm.address_space<ub>>) outs(%gm : memref<64xf32, #hivm.address_space<gm>>)
    }
    return
  }
}
