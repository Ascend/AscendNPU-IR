// REQUIRES: execution-engine
// RUN: bishengir-opt %s --hfusion-decompose="hfusion-decompose-phase=before-lower-to-loops" \
// RUN:   --convert-vector-to-scf --convert-scf-to-cf --convert-arith-to-llvm \
// RUN:   --convert-vector-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:   --reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

module {
  // ===----------------------------------------------------------------------===
  // Function: test_atomic_xchg_1d
  // Lowering: hfusion.atomic_xchg -> scf.for + memref.atomic_rmw
  // ===----------------------------------------------------------------------===
  func.func @test_atomic_xchg_1d(%arg0: memref<?xi16>) {
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    
    // Allocate source buffer and initialize with new value (7)
    %alloc = memref.alloc() : memref<256xi16>
    %c7 = arith.constant 7 : i16
    scf.for %i = %c0 to %c256 step %c1 {
      memref.store %c7, %alloc[%i] : memref<256xi16>
    }

    // Cast dynamic memref to fixed-size strided memref
    %reinterpret_cast = memref.reinterpret_cast %arg0 
      to offset: [0], sizes: [256], strides: [1] 
      : memref<?xi16> to memref<256xi16, strided<[1]>>
    
    // Execute atomic exchange operation
    hfusion.atomic_xchg ins(%alloc : memref<256xi16>) 
                        outs(%reinterpret_cast : memref<256xi16, strided<[1]>>)
    return
  }

  // ===----------------------------------------------------------------------===
  // Main Entry Point for Verification
  // ===----------------------------------------------------------------------===
  func.func @main() {
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i16 = arith.constant 0 : i16
    
    // 1. Initialize destination memory with 0
    %mem = memref.alloc(%c256) : memref<?xi16>
    scf.for %i = %c0 to %c256 step %c1 {
      memref.store %c0_i16, %mem[%i] : memref<?xi16>
    }

    // 2. Print initial values (Expected: all 0s)
    // CHECK: ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    %t = vector.transfer_read %mem[%c0], %c0_i16 : memref<?xi16>, vector<10xi16>
    vector.print %t : vector<10xi16>

    // 3. Perform atomic exchange
    call @test_atomic_xchg_1d(%mem) : (memref<?xi16>) -> ()

    // 4. Print updated values (Expected: all 7s)
    // CHECK: ( 7, 7, 7, 7, 7, 7, 7, 7, 7, 7 )
    %v = vector.transfer_read %mem[%c0], %c0_i16 : memref<?xi16>, vector<10xi16>
    vector.print %v : vector<10xi16>

    memref.dealloc %mem : memref<?xi16>
    return
  }
}