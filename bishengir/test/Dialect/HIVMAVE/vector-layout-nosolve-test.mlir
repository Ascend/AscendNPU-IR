// RUN: not bishengir-opt -analyze-vector-layout %s -split-input-file 2>&1 | FileCheck %s

// CHECK: No Solve
// CHECK-NEXT: ========== Vector Layout Analysis Failure ==========
// CHECK-NEXT: Location:
// CHECK-NEXT: Operation: {{.*}}ave.hir.vtrc
// CHECK-NEXT: Opcode: ave.hir.vtrc
// CHECK-NEXT: Operand types:
// CHECK-NEXT:   vector<64xf32>
// CHECK-NEXT:   vector<64xi1>
// CHECK-NEXT: Result types:
// CHECK-NEXT:   vector<64xf16>
// CHECK-NEXT: Candidates in the solution space:
// CHECK-NEXT: Input states from last valid candidates:
// CHECK:      State: b16
// CHECK:      Possible causes and solutions:
// CHECK-NEXT:   1. The operation may not have been lowered to the HIVMAVE dialect before VectorLayout analysis.
// CHECK:         2. The operation type may not be handled in solveProblem TypeSwitch.
// CHECK-NEXT:      -> Add a new Case for ave.hir.vtrc in solveProblem().
// CHECK:         3. The specific VecMemType combination is not supported by this op.
// CHECK:         4. Conflicting layout requirements from multiple consumers.
// CHECK:         5. Unsupported element bitwidth (only 1/8/16/32 are supported).
// CHECK-NEXT:      -> Verify all vector types have supported bitwidths.
// CHECK-NEXT: ======================================================
func.func @test_vtrc_nosolve(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64xf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %v = ave.hir.vload <NORM> %arg0[%c0] : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
  %r = ave.hir.vtrc %v, <rint>, %mask : vector<64xf32>, vector<64xi1>, vector<64xf16>
  ave.hir.masked_store <NORM_B16> %arg1[%c0], %mask, %r : memref<64xf16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf16>
  return
}
