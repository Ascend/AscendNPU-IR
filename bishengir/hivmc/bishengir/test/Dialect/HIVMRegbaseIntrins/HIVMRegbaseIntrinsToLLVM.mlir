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

// CHECK: attributes #[[ATTR]] = {
// CHECK-SAME-DAG: "target-cpu"="dav-c310"
// CHECK-SAME-DAG: "target-features"="+dav-c310"

// CHECK: !hivm.annotations = !{![[MD1:[0-9]+]]

// CHECK-DAG: ![[MD1]] = !{ptr @scalar_func, !"kernel", i32 1}
// CHECK-DAG: ![[MD2]] = !{!"simt_entry", i32 1}
// CHECK-DAG: ![[MD3]] = !{!"simt-max-threads", i32 4096}
