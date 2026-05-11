// RUN: bishengir-opt --hivmave-scalar-broadcast-to-vload %s | FileCheck %s

// CHECK: func.func private @_attn_fwd_scope_0
// CHECK: ave.hir.vload <BRC_B32>
// CHECK: ave.hir.vload <BRC_B32>
func.func private @_attn_fwd_scope_0(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<64x64xf32, #hivm.address_space<ub>>, %arg4: memref<64x64xf32, #hivm.address_space<ub>>, %arg5: memref<64xf32, #hivm.address_space<ub>>, %arg6: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, noinline, outline = true, vector_mode = "simd"} {
  %c0 = arith.constant 0 : index
  scf.for %arg7 = %arg0 to %arg1 step %arg2  : i32 {
    %0 = arith.index_cast %arg7 : i32 to index
    %subview = memref.subview %arg3[%0, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg4[%0, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg5[%0] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_2 = memref.subview %arg6[%0] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %1 = memref.load %subview_1[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %2 = memref.load %subview_2[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_3 = memref.subview %subview[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %3 = ave.hir.vload <NORM> %subview_3[%c0] : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
    %subview_4 = memref.subview %subview_0[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %4 = ave.hir.vload <NORM> %subview_4[%c0] : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
    %5 = ave.hir.scalar_broadcast %1 : f32 -> vector<64xf32>
    %6 = ave.hir.pge <ALL> : vector<64xi1>
    %7 = ave.hir.vmul %3, %5, %6 : vector<64xf32>, vector<64xi1>
    %8 = ave.hir.pge <ALL> : vector<64xi1>
    %9 = ave.hir.vadd %7, %4, %8 : vector<64xf32>, vector<64xi1>
    %10 = ave.hir.scalar_broadcast %2 : f32 -> vector<64xf32>
    %11 = ave.hir.pge <ALL> : vector<64xi1>
    %12 = ave.hir.vdiv %9, %10, %11 : vector<64xf32>, vector<64xi1>
    %13 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %subview_3[%c0], %13, %12 : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
  }
  return
}