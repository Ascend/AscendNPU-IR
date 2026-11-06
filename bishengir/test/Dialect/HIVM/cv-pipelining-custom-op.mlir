// RUN: bishengir-opt -cv-pipelining="pipeline-mode=unroll set-depth-in-unroll-mode=2" -allow-unregistered-dialect -split-input-file %s | FileCheck %s

// Test 1: A CUBE custom op inside a pipelined loop must be assigned to the
// CUBE workitem, together with the other CUBE ops (mmadL1). The VECTOR ops
// (vexp/store) form a separate VECTOR workitem.
// CHECK-LABEL: func.func @test_cube_custom_op
// CHECK: scf.for
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.hir.custom
// CHECK-SAME: hivm.tcore_type = #hivm.tcore_type<CUBE>
// CHECK-SAME: symbol = "cube_custom"
// CHECK: hivm.loop_core_type = #hivm.tcore_type<CUBE>
// CHECK: scf.for
// CHECK: hivm.hir.vexp
// CHECK: hivm.loop_core_type = #hivm.tcore_type<VECTOR>
func.func @test_cube_custom_op(%gm: memref<16x16xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
  %c0 = arith.constant 0 : i32
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %step = arith.constant 2 : i32
  %bound = arith.constant 8 : i32
  %vdest = tensor.empty() : tensor<16x16xf16>
  %cdest = tensor.empty() : tensor<16x16xf32>
  %gmT = bufferization.to_tensor %gm : memref<16x16xf16>
  scf.for %i = %c0 to %bound step %step : i32 {
    // Cube: standard mmad
    %dot = hivm.hir.mmadL1 ins(%gmT, %gmT, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%cdest : tensor<16x16xf32>) -> tensor<16x16xf32>
    // Cube: custom op
    %cdest2 = tensor.empty() : tensor<16x16xf32>
    %custom = hivm.hir.custom {hivm.pipe = #hivm.pipe<PIPE_M>, hivm.tcore_type = #hivm.tcore_type<CUBE>, symbol = "cube_custom"} "cube_custom" ins(%gmT : tensor<16x16xf16>) outs(%cdest2 : tensor<16x16xf32>) -> tensor<16x16xf32>
    // Vector: standard vexp
    %exp = hivm.hir.vexp ins(%gmT : tensor<16x16xf16>) outs(%vdest : tensor<16x16xf16>) -> tensor<16x16xf16>
    hivm.hir.store ins(%exp : tensor<16x16xf16>) outs(%gm : memref<16x16xf16>)
  }
  return
}

// -----

// Test 2: A VECTOR custom op must be assigned to the VECTOR workitem, next
// to the other VECTOR ops (vexp/store). The mmadL1 forms the CUBE workitem.
// CHECK-LABEL: func.func @test_vector_custom_op
// CHECK: scf.for
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.loop_core_type = #hivm.tcore_type<CUBE>
// CHECK: scf.for
// CHECK: hivm.hir.custom
// CHECK-SAME: hivm.tcore_type = #hivm.tcore_type<VECTOR>
// CHECK-SAME: symbol = "vector_custom"
// CHECK: hivm.hir.vexp
// CHECK: hivm.loop_core_type = #hivm.tcore_type<VECTOR>
func.func @test_vector_custom_op(%gm: memref<16x16xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
  %c0 = arith.constant 0 : i32
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %step = arith.constant 2 : i32
  %bound = arith.constant 8 : i32
  %vdest = tensor.empty() : tensor<16x16xf16>
  %cdest = tensor.empty() : tensor<16x16xf32>
  %gmT = bufferization.to_tensor %gm : memref<16x16xf16>
  scf.for %i = %c0 to %bound step %step : i32 {
    // Cube: standard mmad
    %dot = hivm.hir.mmadL1 ins(%gmT, %gmT, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%cdest : tensor<16x16xf32>) -> tensor<16x16xf32>
    // Vector: custom op
    %vdest2 = tensor.empty() : tensor<16x16xf16>
    %custom = hivm.hir.custom {hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "vector_custom"} "vector_custom" ins(%gmT : tensor<16x16xf16>) outs(%vdest2 : tensor<16x16xf16>) -> tensor<16x16xf16>
    // Vector: standard vexp
    %exp = hivm.hir.vexp ins(%gmT : tensor<16x16xf16>) outs(%vdest : tensor<16x16xf16>) -> tensor<16x16xf16>
    hivm.hir.store ins(%exp : tensor<16x16xf16>) outs(%gm : memref<16x16xf16>)
  }
  return
}

// -----

// Test 3: A CUBE custom op in memref form (non-tensor operands) is also
// assigned to the CUBE workitem.
// CHECK-LABEL: func.func @test_memref_custom_op
// CHECK: scf.for
// CHECK: hivm.hir.custom
// CHECK-SAME: hivm.tcore_type = #hivm.tcore_type<CUBE>
// CHECK-SAME: symbol = "cube_custom_memref"
// CHECK: hivm.loop_core_type = #hivm.tcore_type<CUBE>
// CHECK: scf.for
// CHECK: hivm.hir.vexp
// CHECK: hivm.loop_core_type = #hivm.tcore_type<VECTOR>
func.func @test_memref_custom_op(%gm: memref<16x16xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
  %c0 = arith.constant 0 : i32
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %step = arith.constant 2 : i32
  %bound = arith.constant 8 : i32
  %vdest = tensor.empty() : tensor<16x16xf16>
  %cdest = tensor.empty() : tensor<16x16xf32>
  %gmT = bufferization.to_tensor %gm : memref<16x16xf16>
  scf.for %i = %c0 to %bound step %step : i32 {
    // Cube: standard mmad
    %dot = hivm.hir.mmadL1 ins(%gmT, %gmT, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%cdest : tensor<16x16xf32>) -> tensor<16x16xf32>
    // Cube: custom op writing to a UB buffer
    %ub = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
    hivm.hir.custom {hivm.pipe = #hivm.pipe<PIPE_M>, hivm.tcore_type = #hivm.tcore_type<CUBE>, symbol = "cube_custom_memref"} "cube_custom_memref" ins(%gm : memref<16x16xf16>) outs(%ub : memref<16x16xf16, #hivm.address_space<ub>>)
    // Vector: standard vexp
    %exp = hivm.hir.vexp ins(%gmT : tensor<16x16xf16>) outs(%vdest : tensor<16x16xf16>) -> tensor<16x16xf16>
    hivm.hir.store ins(%exp : tensor<16x16xf16>) outs(%gm : memref<16x16xf16>)
  }
  return
}
