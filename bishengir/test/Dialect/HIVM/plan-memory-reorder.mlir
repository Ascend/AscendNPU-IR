// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend910B1 -hivm-plan-memory=plan-memory-strategy=largest-first -split-input-file -verify-diagnostics | FileCheck %s

// -----

// CHECK: hivm.hir.pointer_cast(%{{.*}})
func.func @test_change_mem_plan_order(%arg0: memref<48x256xf32, #hivm.address_space<gm>>, %arg1: memref<128x256xf32, #hivm.address_space<gm>>, %arg2: memref<16x256xf32, #hivm.address_space<gm>>,  %arg3: memref<60x256xf32, #hivm.address_space<gm>>) {
  %cst_0 = arith.constant 0x00000000 : f32
  %alloc = memref.alloc() : memref<48x256xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0: memref<48x256xf32, #hivm.address_space<gm>>) outs(%alloc : memref<48x256xf32, #hivm.address_space<ub>>)
  %alloc_0 = memref.alloc() : memref<128x256xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg1: memref<128x256xf32, #hivm.address_space<gm>>) outs(%alloc_0 : memref<128x256xf32, #hivm.address_space<ub>>)
  hivm.hir.debug {debugtype = "print", hex = false, prefix = " %alloc: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %alloc : memref<48x256xf32, #hivm.address_space<ub>>
  %alloc_1 = memref.alloc() : memref<16x256xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg2: memref<16x256xf32, #hivm.address_space<gm>>) outs(%alloc_1 : memref<16x256xf32, #hivm.address_space<ub>>)
  hivm.hir.debug {debugtype = "print", hex = false, prefix = " %alloc_1: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %alloc_1 : memref<16x256xf32, #hivm.address_space<ub>>
  %alloc_2 = memref.alloc() : memref<60x256xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg3: memref<60x256xf32, #hivm.address_space<gm>>) outs(%alloc_2 : memref<60x256xf32, #hivm.address_space<ub>>)
  hivm.hir.debug {debugtype = "print", hex = false, prefix = " %alloc_2: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %alloc_2 : memref<60x256xf32, #hivm.address_space<ub>>
  hivm.hir.debug {debugtype = "print", hex = false, prefix = " %alloc_0: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %alloc_0 : memref<128x256xf32, #hivm.address_space<ub>>
  return
}
