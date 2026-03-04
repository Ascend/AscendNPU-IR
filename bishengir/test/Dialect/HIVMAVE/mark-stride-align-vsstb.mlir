// RUN: bishengir-opt -hivm-mark-stride-align -split-input-file %s | FileCheck %s

// CHECK-LABEL:func.func @test_vsstb_align()
// CHECK-NOT: annotation.mark %alloc
// CHECK-NOT: annotation.mark %alloc_0
// CHECK: annotation.mark %alloc_1 {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 130>}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @vf(%arg0: memref<64x8x16xf32, #hivm.address_space<ub>>,
                %arg1: memref<64x8x16xf16, #hivm.address_space<ub>>,
                %arg2: memref<8x64x16xf16, #hivm.address_space<ub>>)
                attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    scf.for %arg3 = %c0 to %c64 step %c1 {
      scf.for %arg4 = %c0 to %c8 step %c4 {
        %subview = memref.subview %arg0[%arg3, %arg4, 0] [1, 4, 16] [1, 1, 1] : memref<64x8x16xf32, #hivm.address_space<ub>> to memref<1x4x16xf32, strided<[128, 16, 1], offset: ?>, #hivm.address_space<ub>>
        %0 = vector.transfer_read %subview[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x4x16xf32, strided<[128, 16, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x4x16xf32>
        %1 = arith.truncf %0 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<1x4x16xf32> to vector<1x4x16xf16>
        %subview_0 = memref.subview %arg2[%arg4, %arg3, 0] [4, 1, 16] [1, 1, 1] : memref<8x64x16xf16, #hivm.address_space<ub>> to memref<4x1x16xf16, strided<[1024, 16, 1], offset: ?>, #hivm.address_space<ub>>
        %2 = vector.transpose %1, [1, 0, 2] : vector<1x4x16xf16> to vector<4x1x16xf16>
        vector.transfer_write %2, %subview_0[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<4x1x16xf16>, memref<4x1x16xf16, strided<[1024, 16, 1], offset: ?>, #hivm.address_space<ub>>
      } {unroll_for_vsstb}
    }
    return
  }

  func.func @test_vsstb_align() attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix", parallel_mode = "simd"} {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x8x16xf32, #hivm.address_space<ub>>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x8x16xf16, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<8x64x16xf16, #hivm.address_space<ub>>
    func.call @vf(%alloc, %alloc_0, %alloc_1) {hivm.vector_function, no_inline} : (memref<64x8x16xf32, #hivm.address_space<ub>>, memref<64x8x16xf16, #hivm.address_space<ub>>, memref<8x64x16xf16, #hivm.address_space<ub>>) -> ()
    return
  }
}
