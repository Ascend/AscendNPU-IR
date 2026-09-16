// RUN: bishengir-translate --mlir-to-llvmir -split-input-file %s | FileCheck %s
#alias_scope_domain = #llvm.alias_scope_domain<id = distinct[0]<>>
#alias_scope = #llvm.alias_scope<id = distinct[1]<>, domain = #alias_scope_domain>
#alias_scope1 = #llvm.alias_scope<id = distinct[2]<>, domain = #alias_scope_domain>
// CHECK-LABEL: define void @test_store(<32 x i64> %0, ptr addrspace(6) %1, i32 %2, i32 %3, i32 %4, <256 x i1> %5) {
// CHECK: call void @llvm.hivm.vstsx1.v32u64(<32 x i64> %0, ptr addrspace(6) %1, i32 %2, i32 %3, i32 %4, <256 x i1> %5)
llvm.func @test_store(%data: vector<32xi64>, %base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32, %mask: vector<256 x i1>) -> () {
  "hivm_regbaseintrins.intr.hivm.vstsx1.v32u64"(%data, %base, %offset, %dis, %mode, %mask) : (vector<32xi64>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
  llvm.return
}

// CHECK-LABEL: define void @test_store_post(<32 x i64> %0, ptr addrspace(6) %1, i32 %2, i32 %3, i32 %4, <256 x i1> %5) {
// CHECK: %[[CALL:.*]] = call ptr addrspace(6) @llvm.hivm.vstsx1.post.v32u64(<32 x i64> %0, ptr addrspace(6) %1, i32 %2, i32 %3, i32 %4, <256 x i1> %5)
llvm.func @test_store_post(%data: vector<32xi64>, %base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32, %mask: vector<256 x i1>) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vstsx1.post.v32u64"(%data, %base, %offset, %dis, %mode, %mask) : (vector<32xi64>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> (!llvm.ptr<6>)
  llvm.return
}

// CHECK-LABEL: define void @test_store_x2(<128 x i16> %0, <128 x i16> %1, ptr addrspace(6) %2, i32 %3, i32 %4, i32 %5, <256 x i1> %6) {
// CHECK: call void @llvm.hivm.vstsx2.v128b16(<128 x i16> %0, <128 x i16> %1, ptr addrspace(6) %2, i32 %3, i32 %4, i32 %5, <256 x i1> %6)
llvm.func @test_store_x2(%data0: vector<128xi16>, %data1: vector<128xi16>, %base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32, %mask: vector<256 x i1>) -> () {
  "hivm_regbaseintrins.intr.hivm.vstsx2.v128b16"(%data0, %data1, %base, %offset, %dis, %mode, %mask) : (vector<128xi16>, vector<128xi16>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
  llvm.return
}

// CHECK-LABEL: define void @test_store_alias(<32 x i64> %0, ptr addrspace(6) %1, i32 %2, i32 %3, i32 %4, <256 x i1> %5) {
// CHECK: call void @llvm.hivm.vstsx1.v32u64(<32 x i64> %0, ptr addrspace(6) %1, i32 %2, i32 %3, i32 %4, <256 x i1> %5), !alias.scope ![[scope0:.*]], !noalias ![[scope1:.*]]
llvm.func @test_store_alias(%data: vector<32xi64>, %base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32, %mask: vector<256 x i1>) -> () {
  "hivm_regbaseintrins.intr.hivm.vstsx1.v32u64"(%data, %base, %offset, %dis, %mode, %mask) {hivm.has_alias_scopes,llvm.alias_scope = [#alias_scope], llvm.noalias_scopes = [#alias_scope1]}: (vector<32xi64>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
  llvm.return
}
