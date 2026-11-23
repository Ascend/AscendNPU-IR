// REQUIRES: hivmc
// RUN: bishengir-compile %s --target=Ascend950PR_9579 \
// RUN:   --enable-hfusion-compile=false --enable-lir-compile=false \
// RUN:   --only-run-hivm-pipeline=true \
// RUN:   --mlir-print-ir-after=hivm-infer-mem-scope -o %t \
// RUN:   2>&1 | FileCheck %s --implicit-check-not=func.call

// Tree-reduction tiling leaves affine.apply in outlined fusion helpers. The
// affine inliner rejects these operations when the call is inside scf.for.
// Lower only helper affine.apply before retrying inlining: memory-scope
// inference requires the helper's local allocation to be in the caller,
// rather than an unscoped memref result.
// Exercise the compiler pipeline itself so removing the lowering regresses
// this test. Keep affine.apply in the caller for later affine optimizations.

// CHECK: IR Dump After InferHIVMMemScope
// CHECK-NOT: func.func private @fused
// CHECK: func.func @kernel(%[[OUTPUT:.*]]: memref<16xf32, #hivm.address_space<gm>>
// CHECK: scf.for
// CHECK: %[[BUFFER:.*]] = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
// CHECK: %[[OFFSET:.*]] = arith.addi
// CHECK: memref.store %{{.*}}, %[[BUFFER]][%[[OFFSET]]]
// CHECK: %[[CALLER_OFFSET:.*]] = affine.apply
// CHECK: memref.load %[[BUFFER]][%[[CALLER_OFFSET]]]
// CHECK: memref.store %{{.*}}, %[[OUTPUT]]
// CHECK: return

module {
  func.func private @fused(%offset: index) -> memref<16xf32>
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %one = arith.constant 1.0 : f32
    %buffer = memref.alloc() : memref<16xf32>
    %next = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%offset]
    memref.store %one, %buffer[%next] : memref<16xf32>
    return %buffer : memref<16xf32>
  }
  func.func @kernel(%output: memref<16xf32>, %upper: index)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %upper step %c1 {
      %buffer = func.call @fused(%i) : (index) -> memref<16xf32>
      %next = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%i]
      %value = memref.load %buffer[%next] : memref<16xf32>
      memref.store %value, %output[%i] : memref<16xf32>
    }
    return
  }
}
