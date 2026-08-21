// RUN: bishengir-opt %s -triton-remap | FileCheck %s

module attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:89", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @triton_evict_last(%src_ptr: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %dst_ptr: !llvm.ptr<1> {tt.divisibility = 16 : i32}, %n_elements: i32 {tt.divisibility = 16 : i32}, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
// CHECK:         llvm.func @triton_evict_last(
// CHECK-SAME:       %[[src_ptr:[^:]*]]: !llvm.ptr<1>
    %0 = llvm.mlir.undef : vector<4xi8>
    %1 = llvm.mlir.constant(3 : index) : i32
    %2 = llvm.mlir.constant(2 : index) : i32
    %3 = llvm.mlir.constant(1 : index) : i32
    %4 = llvm.mlir.constant(2 : i32) : i32
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.mlir.constant(3 : i32) : i32
    %offs = llvm.mlir.constant(127 : i32) : i32
    %7 = llvm.mlir.constant(5 : i32) : i32
    %8 = llvm.mlir.constant(0 : i32) : i32
    %9 = llvm.mlir.constant(32 : i32) : i32
    %10 = llvm.mlir.constant(0 : index) : i32
    %11 = llvm.mlir.constant(1024 : i32) : i32
// CHECK:      %[[cst_1:.*]] = llvm.mlir.constant(1 : i32) : i32
    %pid = nvvm.read.ptx.sreg.ctaid.x : i32
    %offs_0 = llvm.mul %pid, %11 : i32
    %offs_1 = nvvm.read.ptx.sreg.tid.x : i32
    %offs_2 = llvm.urem %offs_1, %9 : i32
    %offs_3 = llvm.udiv %offs_1, %9 : i32
    %offs_4 = llvm.shl %offs_2, %8 : i32
    %offs_5 = llvm.or %8, %offs_4 : i32
    %offs_6 = llvm.shl %offs_3, %7 : i32
    %offs_7 = llvm.or %offs_5, %offs_6 : i32
    %offs_8 = llvm.or %offs_7, %8 : i32
    %offs_9 = llvm.and %offs_8, %offs : i32
    %offs_10 = llvm.shl %offs_9, %6 : i32
    %offs_11 = llvm.or %offs_10, %8 : i32
    %offs_12 = llvm.xor %8, %offs_11 : i32
    %offs_13 = llvm.xor %offs_12, %8 : i32
    %offs_14 = llvm.add %offs_13, %10 : i32
    %offs_15 = llvm.add %offs_0, %offs_14 : i32
    %mask = llvm.icmp "slt" %offs_15, %n_elements : i32
    %src_addr = llvm.getelementptr %src_ptr[%offs_15] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8
// CHECK:  %[[src_addr:.*]] = llvm.getelementptr %[[src_ptr]]
    %dst_addr = llvm.getelementptr %dst_ptr[%offs_15] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8
    %vals1 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u64 $0, 0x0;\0A\09createpolicy.fractional.L2::evict_last.b64 $0, 1.0;", "=l"  : () -> i64
    %vals1_16 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, 0x0;\0A\09mov.u32 $1, 0x0;\0A\09@$4 ld.global.L1::evict_last.L2::cache_hint.v2.b32 { $0, $1 }, [ $2 + 0 ], $3;", "=r,=r,l,l,b" %src_addr, %vals1, %mask : (!llvm.ptr<1>, i64, i1) -> !llvm.struct<(i32, i32)>
// CHECK:     %[[vals1:.*]] = llvm.call_intrinsic "llvm.hivm.ldg.cache.s64"(%[[src_addr]], %[[cst_1]]) : (!llvm.ptr<1>, i32) -> i64
// CHECK:     %[[vals1_cast:.*]] = llvm.bitcast %[[vals1]] : i64 to vector<2xi32>
    %vals1_17 = llvm.extractvalue %vals1_16[0] : !llvm.struct<(i32, i32)> 
    %vals1_18 = llvm.bitcast %vals1_17 : i32 to vector<4xi8>
    %vals1_19 = llvm.extractvalue %vals1_16[1] : !llvm.struct<(i32, i32)> 
    %vals1_20 = llvm.bitcast %vals1_19 : i32 to vector<4xi8>
    %vals1_21 = llvm.extractelement %vals1_18[%10 : i32] : vector<4xi8>
    %vals1_22 = llvm.extractelement %vals1_18[%3 : i32] : vector<4xi8>
    %vals1_23 = llvm.extractelement %vals1_18[%2 : i32] : vector<4xi8>
    %vals1_24 = llvm.extractelement %vals1_18[%1 : i32] : vector<4xi8>
    %vals1_25 = llvm.extractelement %vals1_20[%10 : i32] : vector<4xi8>
    %vals1_26 = llvm.extractelement %vals1_20[%3 : i32] : vector<4xi8>
    %vals1_27 = llvm.extractelement %vals1_20[%2 : i32] : vector<4xi8>
    %vals1_28 = llvm.extractelement %vals1_20[%1 : i32] : vector<4xi8>
    %vals2 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, 0x0;\0A\09mov.u32 $1, 0x0;\0A\09@$3 ld.global.v2.b32 { $0, $1 }, [ $2 + 0 ];", "=r,=r,l,b" %src_addr, %mask : (!llvm.ptr<1>, i1) -> !llvm.struct<(i32, i32)>
// CHECK:     %[[vals2:.*]] = llvm.load %[[src_addr]] : !llvm.ptr<1> -> vector<2xi32>
    %vals2_29 = llvm.extractvalue %vals2[0] : !llvm.struct<(i32, i32)> 
    %vals2_30 = llvm.bitcast %vals2_29 : i32 to vector<4xi8>
    %vals2_31 = llvm.extractvalue %vals2[1] : !llvm.struct<(i32, i32)> 
    %vals2_32 = llvm.bitcast %vals2_31 : i32 to vector<4xi8>
    %vals2_33 = llvm.extractelement %vals2_30[%10 : i32] : vector<4xi8>
    %vals2_34 = llvm.extractelement %vals2_30[%3 : i32] : vector<4xi8>
    %vals2_35 = llvm.extractelement %vals2_30[%2 : i32] : vector<4xi8>
    %vals2_36 = llvm.extractelement %vals2_30[%1 : i32] : vector<4xi8>
    %vals2_37 = llvm.extractelement %vals2_32[%10 : i32] : vector<4xi8>
    %vals2_38 = llvm.extractelement %vals2_32[%3 : i32] : vector<4xi8>
    %vals2_39 = llvm.extractelement %vals2_32[%2 : i32] : vector<4xi8>
    %vals2_40 = llvm.extractelement %vals2_32[%1 : i32] : vector<4xi8>
    %12 = llvm.add %vals1_21, %vals2_33 : i8
    %13 = llvm.add %vals1_22, %vals2_34 : i8
    %14 = llvm.add %vals1_23, %vals2_35 : i8
    %15 = llvm.add %vals1_24, %vals2_36 : i8
    %16 = llvm.add %vals1_25, %vals2_37 : i8
    %17 = llvm.add %vals1_26, %vals2_38 : i8
    %18 = llvm.add %vals1_27, %vals2_39 : i8
    %19 = llvm.add %vals1_28, %vals2_40 : i8
    %20 = llvm.insertelement %12, %0[%8 : i32] : vector<4xi8>
    %21 = llvm.insertelement %13, %20[%5 : i32] : vector<4xi8>
    %22 = llvm.insertelement %14, %21[%4 : i32] : vector<4xi8>
    %23 = llvm.insertelement %15, %22[%6 : i32] : vector<4xi8>
    %24 = llvm.bitcast %23 : vector<4xi8> to i32
    %25 = llvm.insertelement %16, %0[%8 : i32] : vector<4xi8>
    %26 = llvm.insertelement %17, %25[%5 : i32] : vector<4xi8>
    %27 = llvm.insertelement %18, %26[%4 : i32] : vector<4xi8>
    %28 = llvm.insertelement %19, %27[%6 : i32] : vector<4xi8>
    %29 = llvm.bitcast %28 : vector<4xi8> to i32
    %30 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$3 st.global.v2.b32 [ $2 + 0 ], { $0, $1 };", "r,r,l,b" %24, %29, %dst_addr, %mask : (i32, i32, !llvm.ptr<1>, i1) -> !llvm.void
    llvm.return
  }
}
