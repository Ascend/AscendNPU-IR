// RUN: bishengir-opt --hivm-memref-alloc-to-alloca -split-input-file %s | FileCheck %s

// Alloc to Alloca.
// CHECK-LABEL: func @test_hivm_memory_scope_l1
// CHECK:         %[[M:.*]] = memref.alloca() : memref<32x32xi32, #hivm.address_space<cbuf>>
// CHECK-NEXT:    return %[[M]] : memref<32x32xi32, #hivm.address_space<cbuf>>
func.func @test_hivm_memory_scope_l1() -> memref<32x32xi32, #hivm.address_space<cbuf>> {
    %m = memref.alloc() : memref<32x32xi32, #hivm.address_space<cbuf>>
    return %m : memref<32x32xi32, #hivm.address_space<cbuf>>
}

// -----

// No conversions.
// CHECK-LABEL: func @test_hivm_memory_scope_gm
// CHECK:         %[[M:.*]] = memref.alloc() : memref<32x32xi32, #hivm.address_space<gm>>
// CHECK-NEXT:    return %[[M]] : memref<32x32xi32, #hivm.address_space<gm>>
func.func @test_hivm_memory_scope_gm() -> memref<32x32xi32, #hivm.address_space<gm>> {
    %m = memref.alloc() : memref<32x32xi32, #hivm.address_space<gm>>
    return %m : memref<32x32xi32, #hivm.address_space<gm>>
}

// -----

// No conversions.
// CHECK-LABEL: func @test_other_memory_scope
// CHECK:         %[[M:.*]] = memref.alloc() : memref<32x32xi32, 6>
// CHECK-NEXT:    return %[[M]] : memref<32x32xi32, 6>
func.func @test_other_memory_scope() -> memref<32x32xi32, 6> {
    %m = memref.alloc() : memref<32x32xi32, 6>
    return %m : memref<32x32xi32, 6>
}

// -----

// No conversions.
// CHECK-LABEL: func @test_no_memory_scope
// CHECK:         %[[M:.*]] = memref.alloc() : memref<32x32xi32>
// CHECK-NEXT:    return %[[M]] : memref<32x32xi32>
func.func @test_no_memory_scope() -> memref<32x32xi32> {
    %m = memref.alloc() : memref<32x32xi32>
    return %m : memref<32x32xi32>
}