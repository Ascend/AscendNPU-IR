// RUN: bishengir-translate --mlir-to-llvmir -split-input-file %s | FileCheck %s

// Test that BMU (an A6 feature) intrinsic wrappers export to llvm.hivm calls.
// Non-BT buffers allocate/free a pointer in the buffer's address space; BT is
// addressed by an i64 scalar (proto "ULi") and has no address space.
// LLVM sanitizes underscores in intrinsic names, so the wrapper
// bmu.intr.hivm.__ubuf_alloc becomes llvm.hivm...ubuf.alloc, etc.

// CHECK-LABEL: define void @test_ubuf_alloc_free
llvm.func @test_ubuf_alloc_free(%size: i64) {
  // CHECK: %[[PTR:.*]] = call ptr addrspace(6) @llvm.hivm...ubuf.alloc(i64 %{{.*}}, i64 0, i64 0)
  %0 = "bmu.intr.hivm.__ubuf_alloc"(%size) <{pipe = 0 : i64, seg = 0 : i64}> : (i64) -> !llvm.ptr<6>
  // CHECK: call void @llvm.hivm...ubuf.free(ptr addrspace(6) %[[PTR]], i64 %{{.*}}, i64 0)
  "bmu.intr.hivm.__ubuf_free"(%0, %size) <{pipe = 0 : i64}> : (!llvm.ptr<6>, i64) -> ()
  llvm.return
}

// -----

// CHECK-LABEL: define void @test_cbuf_alloci_freei
llvm.func @test_cbuf_alloci_freei() {
  // CHECK: %[[PTR:.*]] = call ptr addrspace(2) @llvm.hivm...cbuf.alloci(i64 4, i64 0, i64 1)
  %0 = "bmu.intr.hivm.__cbuf_alloci"() <{pipe = 1 : i64, seg = 0 : i64, size = 4 : i64}> : () -> !llvm.ptr<2>
  // CHECK: call void @llvm.hivm...cbuf.freei(ptr addrspace(2) %[[PTR]], i64 4, i64 1)
  "bmu.intr.hivm.__cbuf_freei"(%0) <{pipe = 1 : i64, size = 4 : i64}> : (!llvm.ptr<2>) -> ()
  llvm.return
}

// -----

// BT is the exception: llvm.hivm...bt.alloc returns i64 and llvm.hivm...bt.free
// takes an i64 address.
// CHECK-LABEL: define void @test_bt_alloc_free
llvm.func @test_bt_alloc_free(%size: i64) {
  // CHECK: %[[ADDR:.*]] = call i64 @llvm.hivm...bt.alloc(i64 %{{.*}}, i64 0, i64 0)
  %0 = "bmu.intr.hivm.__bt_alloc"(%size) <{pipe = 0 : i64, seg = 0 : i64}> : (i64) -> i64
  // CHECK: call void @llvm.hivm...bt.free(i64 %[[ADDR]], i64 %{{.*}}, i64 0)
  "bmu.intr.hivm.__bt_free"(%0, %size) <{pipe = 0 : i64}> : (i64, i64) -> ()
  llvm.return
}

// -----

// CHECK-LABEL: define void @test_segm
llvm.func @test_segm(%val: i64) {
  // CHECK: call void @llvm.hivm.set.BMU.SEGM.UB(i64 %{{.*}})
  "bmu.intr.hivm.set_BMU_SEGM_UB"(%val) : (i64) -> ()
  // CHECK: %[[SEGM:.*]] = call i64 @llvm.hivm.get.BMU.SEGM.L0C()
  %0 = "bmu.intr.hivm.get_BMU_SEGM_L0C"() : () -> i64
  llvm.return
}
