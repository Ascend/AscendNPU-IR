// RUN: bishengir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK: simt_entry
// CHECK: simt_func
// CHECK-SAME: #[[ATTR:[0-9]+]]
// CHECK-SAME: annotation ![[MD2:[0-9]+]]
// CHECK-SAME: annotation ![[MD3:[0-9]+]]
llvm.func @simt_func(%arg0 : !llvm.ptr<1>) attributes {hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, hivm_regbaseintrins.cconv = #hivm_regbaseintrins.simt_entry<4096>} {
  // CHECK: call i32 @llvm.hivm.get.TID.X()
  %xid = hivm_regbaseintrins.thread_id_x
  // CHECK: call i32 @llvm.hivm.get.TID.Y()
  %yid = hivm_regbaseintrins.thread_id_y
  // CHECK: call i32 @llvm.hivm.get.TID.Z()
  %zid = hivm_regbaseintrins.thread_id_z
  // CHECK: call i32 @llvm.hivm.get.BLOCK.DIM.X()
  %xdim = hivm_regbaseintrins.block_dim_x
  // CHECK: call i32 @llvm.hivm.get.BLOCK.DIM.Y()
  %ydim = hivm_regbaseintrins.block_dim_y
  // CHECK: call i32 @llvm.hivm.get.BLOCK.DIM.Z()
  %zdim = hivm_regbaseintrins.block_dim_z
  llvm.return
}

// CHECK: scalar_func(
llvm.func @scalar_func(%arg : !llvm.ptr<1>) attributes {hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, hivm_regbaseintrins.kernel}{
  %c8 = llvm.mlir.constant (8 : i64) : i64
  %c16 = llvm.mlir.constant (16 : i64) : i64
  %c32 = llvm.mlir.constant (32 : i64) : i64

  // 34360786976 = 0x8 0010 0020 (8 16 32)
  // CHECK: call void @llvm.hivm.store.vfsimt.info(i64 34360786976)
  // CHECK: call simt_entry void @simt_func
  hivm_regbaseintrins.intrins.launch_func @simt_func threads in (%c32, %c16, %c8) args(%arg) : !llvm.ptr<1>
  llvm.return
}

// CHECK-LABEL: @vcpadd
// CHECK: call <128 x half> @llvm.hivm.vcpadd.x
llvm.func @vcpadd(%src: vector<128xf16>, %mask: vector<256xi1>)
    -> vector<128xf16> {
  %res = "hivm_regbaseintrins.intr.hivm.vcpadd.x"(%src, %mask)
      : (vector<128xf16>, vector<256xi1>) -> vector<128xf16>
  llvm.return %res : vector<128xf16>
}

// CHECK-LABEL: @vsqz_stored
// CHECK: call <256 x i8> @llvm.hivm.vsqz.x.v256i8({{.*}}, i32 1)
llvm.func @vsqz_stored(%src: vector<256xi8>, %mask: vector<256xi1>)
    -> vector<256xi8> {
  %store = llvm.mlir.constant(1 : i32) : i32
  %res = "hivm_regbaseintrins.intr.hivm.vsqz.x"(%src, %mask, %store)
      : (vector<256xi8>, vector<256xi1>, i32) -> vector<256xi8>
  llvm.return %res : vector<256xi8>
}

// CHECK-LABEL: @vsqz_no_stored
// CHECK: call <256 x i8> @llvm.hivm.vsqz.x.v256i8({{.*}}, i32 0)
llvm.func @vsqz_no_stored(%src: vector<256xi8>, %mask: vector<256xi1>)
    -> vector<256xi8> {
  %store = llvm.mlir.constant(0 : i32) : i32
  %res = "hivm_regbaseintrins.intr.hivm.vsqz.x"(%src, %mask, %store)
      : (vector<256xi8>, vector<256xi1>, i32) -> vector<256xi8>
  llvm.return %res : vector<256xi8>
}

// CHECK-LABEL: @vusqz
// CHECK: call <64 x i32> @llvm.hivm.vusqz.m
llvm.func @vusqz(%src: vector<64xi32>, %mask: vector<256xi1>)
    -> vector<64xi32> {
  %res = "hivm_regbaseintrins.intr.hivm.vusqz.m"(%src, %mask)
      : (vector<64xi32>, vector<256xi1>) -> vector<64xi32>
  llvm.return %res : vector<64xi32>
}

// CHECK-LABEL: @vaddcs
// CHECK: call { <64 x i32>, <256 x i1> } @llvm.hivm.vaddcs.v64s32
// CHECK: extractvalue { <64 x i32>, <256 x i1> } {{.*}}, 0
// CHECK: extractvalue { <64 x i32>, <256 x i1> } {{.*}}, 1
llvm.func @vaddcs(%src0: vector<64xi32>, %src1: vector<64xi32>,
                  %carry_in: vector<256xi1>, %mask: vector<256xi1>)
    -> vector<64xi32> {
  %agg = "hivm_regbaseintrins.intr.hivm.vaddcs.v64s32"(%src0, %src1, %carry_in, %mask)
      : (vector<64xi32>, vector<64xi32>, vector<256xi1>, vector<256xi1>)
        -> !llvm.struct<(vector<64xi32>, vector<256xi1>)>
  %res = llvm.extractvalue %agg[0]
      : !llvm.struct<(vector<64xi32>, vector<256xi1>)>
  %carry = llvm.extractvalue %agg[1]
      : !llvm.struct<(vector<64xi32>, vector<256xi1>)>
  llvm.return %res : vector<64xi32>
}

// CHECK-LABEL: @vsubcs
// CHECK: call { <64 x i32>, <256 x i1> } @llvm.hivm.vsubcs.v64s32
// CHECK: extractvalue { <64 x i32>, <256 x i1> } {{.*}}, 0
// CHECK: extractvalue { <64 x i32>, <256 x i1> } {{.*}}, 1
llvm.func @vsubcs(%src0: vector<64xi32>, %src1: vector<64xi32>,
                  %carry_in: vector<256xi1>, %mask: vector<256xi1>)
    -> vector<64xi32> {
  %agg = "hivm_regbaseintrins.intr.hivm.vsubcs.v64s32"(%src0, %src1, %carry_in, %mask)
      : (vector<64xi32>, vector<64xi32>, vector<256xi1>, vector<256xi1>)
        -> !llvm.struct<(vector<64xi32>, vector<256xi1>)>
  %res = llvm.extractvalue %agg[0]
      : !llvm.struct<(vector<64xi32>, vector<256xi1>)>
  %carry = llvm.extractvalue %agg[1]
      : !llvm.struct<(vector<64xi32>, vector<256xi1>)>
  llvm.return %res : vector<64xi32>
}

// CHECK-LABEL: @vaddcs_unsigned
// CHECK: call { <64 x i32>, <256 x i1> } @llvm.hivm.vaddcs.v64u32
// CHECK: extractvalue { <64 x i32>, <256 x i1> } {{.*}}, 0
// CHECK: extractvalue { <64 x i32>, <256 x i1> } {{.*}}, 1
llvm.func @vaddcs_unsigned(%src0: vector<64xi32>, %src1: vector<64xi32>,
                           %carry_in: vector<256xi1>, %mask: vector<256xi1>)
    -> vector<64xi32> {
  %agg = "hivm_regbaseintrins.intr.hivm.vaddcs.v64u32"(%src0, %src1, %carry_in, %mask)
      : (vector<64xi32>, vector<64xi32>, vector<256xi1>, vector<256xi1>)
        -> !llvm.struct<(vector<64xi32>, vector<256xi1>)>
  %res = llvm.extractvalue %agg[0]
      : !llvm.struct<(vector<64xi32>, vector<256xi1>)>
  %carry = llvm.extractvalue %agg[1]
      : !llvm.struct<(vector<64xi32>, vector<256xi1>)>
  llvm.return %res : vector<64xi32>
}

// CHECK-LABEL: @vsubcs_unsigned
// CHECK: call { <64 x i32>, <256 x i1> } @llvm.hivm.vsubcs.v64u32
// CHECK: extractvalue { <64 x i32>, <256 x i1> } {{.*}}, 0
// CHECK: extractvalue { <64 x i32>, <256 x i1> } {{.*}}, 1
llvm.func @vsubcs_unsigned(%src0: vector<64xi32>, %src1: vector<64xi32>,
                           %carry_in: vector<256xi1>, %mask: vector<256xi1>)
    -> vector<64xi32> {
  %agg = "hivm_regbaseintrins.intr.hivm.vsubcs.v64u32"(%src0, %src1, %carry_in, %mask)
      : (vector<64xi32>, vector<64xi32>, vector<256xi1>, vector<256xi1>)
        -> !llvm.struct<(vector<64xi32>, vector<256xi1>)>
  %res = llvm.extractvalue %agg[0]
      : !llvm.struct<(vector<64xi32>, vector<256xi1>)>
  %carry = llvm.extractvalue %agg[1]
      : !llvm.struct<(vector<64xi32>, vector<256xi1>)>
  llvm.return %res : vector<64xi32>
}

// CHECK-LABEL: @vintlv_struct
// CHECK: call { <256 x i8>, <256 x i8> } @llvm.hivm.vintlv.v256i8
// CHECK: ret { <256 x i8>, <256 x i8> }
llvm.func @vintlv_struct(%a: vector<256xi8>, %b: vector<256xi8>)
    -> !llvm.struct<(vector<256xi8>, vector<256xi8>)> {
  %r = "hivm_regbaseintrins.intr.hivm.vintlv"(%a, %b)
      : (vector<256xi8>, vector<256xi8>)
        -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  llvm.return %r : !llvm.struct<(vector<256xi8>, vector<256xi8>)>
}

// CHECK-LABEL: @vintlv_field0
// CHECK: call { <256 x i8>, <256 x i8> } @llvm.hivm.vintlv.v256i8
// CHECK: extractvalue { <256 x i8>, <256 x i8> } {{.*}}, 0
// CHECK: ret <256 x i8>
llvm.func @vintlv_field0(%a: vector<256xi8>, %b: vector<256xi8>)
    -> vector<256xi8> {
  %r = "hivm_regbaseintrins.intr.hivm.vintlv"(%a, %b)
      : (vector<256xi8>, vector<256xi8>)
        -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  %r0 = llvm.extractvalue %r[0]
      : !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  llvm.return %r0 : vector<256xi8>
}

// CHECK: attributes #[[ATTR2:[0-9]+]] = {
// CHECK-SAME-DAG: "target-cpu"="dav-c310"
// CHECK-SAME-DAG: "target-features"="+dav-c310"

// CHECK: !hivm.annotations = !{![[MD1:[0-9]+]]

// CHECK-DAG: ![[MD1]] = !{ptr @scalar_func, !"kernel", i32 1}
// CHECK-DAG: ![[MD2B:[0-9]+]] = !{!"simt_entry", i32 1}
// CHECK-DAG: ![[MD3B:[0-9]+]] = !{!"simt-max-threads", i32 4096}
