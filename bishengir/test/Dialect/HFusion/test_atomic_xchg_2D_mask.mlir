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
  // Function: test_atomic_xchg_masked_2d
  // Description: Core logic to test 2D atomic exchange with mask and edge cases.
  // ===----------------------------------------------------------------------===
  func.func @test_atomic_xchg_masked_2d(%arg0: memref<?x?xi16>, %arg1: memref<4x8xi1>) {
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    
    %alloc = memref.alloc() : memref<4x8xi16>
    %c7 = arith.constant 7 : i16
    
    // Boundary and overflow values for i16
    %max_i16 = arith.constant 32767 : i16
    %min_i16 = arith.constant -32768 : i16
    %over_i16 = arith.constant 32768 : i16

    scf.for %i = %c0 to %c4 step %c1 {
      scf.for %j = %c0 to %c8 step %c1 {
        memref.store %c7, %alloc[%i, %j] : memref<4x8xi16>
      }
    }

    // Insert edge cases in the first row
    memref.store %max_i16, %alloc[%c0, %c0] : memref<4x8xi16>
    memref.store %min_i16, %alloc[%c0, %c1] : memref<4x8xi16>
    %c2_idx = arith.constant 2 : index
    memref.store %over_i16, %alloc[%c0, %c2_idx] : memref<4x8xi16>

    %reinterpret_cast = memref.reinterpret_cast %arg0 
        to offset: [0], sizes: [4, 8], strides: [8, 1] 
        : memref<?x?xi16> to memref<4x8xi16, strided<[8, 1]>>
    
    hfusion.atomic_xchg ins(%alloc : memref<4x8xi16>) 
                        outs(%reinterpret_cast : memref<4x8xi16, strided<[8, 1]>>) 
                        mask(%arg1 : memref<4x8xi1>)
    return
  }

  // ===----------------------------------------------------------------------===
  // Main Driver Function
  // ===----------------------------------------------------------------------===
  func.func @main() {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i16 = arith.constant 0 : i16

    %mem = memref.alloc(%c4, %c8) : memref<?x?xi16>
    %mask = memref.alloc() : memref<4x8xi1>

    %c100 = arith.constant 100 : i16
    %true = arith.constant 1 : i1
    %false = arith.constant 0 : i1
    
    scf.for %i = %c0 to %c4 step %c1 {
      scf.for %j = %c0 to %c8 step %c1 {
        memref.store %c100, %mem[%i, %j] : memref<?x?xi16>
        %c2 = arith.constant 2 : index
        %is_upper = arith.cmpi slt, %i, %c2 : index
        scf.if %is_upper {
          memref.store %true, %mask[%i, %j] : memref<4x8xi1>
        } else {
          memref.store %false, %mask[%i, %j] : memref<4x8xi1>
        }
      }
    }

    // --- Verification Stage 1: Initial Values ---
    // All values should be 100
    // CHECK: ( 100, 100, 100, 100, 100, 100, 100, 100 )
    // CHECK: ( 100, 100, 100, 100, 100, 100, 100, 100 )
    // CHECK: ( 100, 100, 100, 100, 100, 100, 100, 100 )
    // CHECK: ( 100, 100, 100, 100, 100, 100, 100, 100 )
    scf.for %i = %c0 to %c4 step %c1 {
      %v_before = vector.transfer_read %mem[%i, %c0], %c0_i16 : memref<?x?xi16>, vector<8xi16>
      vector.print %v_before : vector<8xi16>
    }

    call @test_atomic_xchg_masked_2d(%mem, %mask) : (memref<?x?xi16>, memref<4x8xi1>) -> ()

    // --- Verification Stage 2: Updated Values ---
    // Row 0: Updated with edge cases (32767, -32768, -32768)
    // CHECK: ( 32767, -32768, -32768, 7, 7, 7, 7, 7 )
    // Row 1: Updated with 7s
    // CHECK: ( 7, 7, 7, 7, 7, 7, 7, 7 )
    // Row 2-3: Masked, remain 100
    // CHECK: ( 100, 100, 100, 100, 100, 100, 100, 100 )
    // CHECK: ( 100, 100, 100, 100, 100, 100, 100, 100 )
    scf.for %i = %c0 to %c4 step %c1 {
      %v_after = vector.transfer_read %mem[%i, %c0], %c0_i16 : memref<?x?xi16>, vector<8xi16>
      vector.print %v_after : vector<8xi16>
    }

    memref.dealloc %mem : memref<?x?xi16>
    memref.dealloc %mask : memref<4x8xi1>
    return
  }
}