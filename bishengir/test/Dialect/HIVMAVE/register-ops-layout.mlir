// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9579 -analyze-vector-layout -remove-vector-layout-attr -ave-normalize-ops -canonicalize -convert-hivmave-to-ave-intrin -canonicalize | FileCheck %s

// A producer with multiple mask uses and a live scalar result must survive.
// CHECK-LABEL: @pack_plt_multiple_uses
// CHECK: "hivm_regbaseintrins.intr.hivm.plt.b32.v300"
// CHECK: "hivm_regbaseintrins.intr.hivm.ppack.z"
// CHECK: "hivm_regbaseintrins.intr.hivm.ppack.z"
// CHECK-NOT: ave.hir.preg
// CHECK: return %{{.*}} : index
func.func @pack_plt_multiple_uses(%count: index, %dst32: memref<64xi32, #hivm.address_space<ub>>, %dst16: memref<128xi16, #hivm.address_space<ub>>) -> index {
  %c0 = arith.constant 0 : index
  %p, %remaining = ave.hir.plt %count : vector<64xi1>, index
  %lo = ave.hir.preg.pack %p, 0 : vector<64xi1> -> vector<128xi1>
  %hi = ave.hir.preg.pack %p, 1 : vector<64xi1> -> vector<128xi1>
  %c1 = arith.constant 1 : i32
  %ones32 = ave.hir.scalar_broadcast %c1 : i32 -> vector<64xi32>
  ave.hir.masked_store <NORM_B32> %dst32[%c0], %p, %ones32 : memref<64xi32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>
  %c2 = arith.constant 2 : i16
  %twos16 = ave.hir.scalar_broadcast %c2 : i16 -> vector<128xi16>
  %c3 = arith.constant 3 : i16
  %threes16 = ave.hir.scalar_broadcast %c3 : i16 -> vector<128xi16>
  %v = ave.hir.vsel %hi, %twos16, %threes16 : vector<128xi1>, vector<128xi16>
  ave.hir.masked_store <NORM_B16> %dst16[%c0], %lo, %v : memref<128xi16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi16>
  return %remaining : index
}

// CHECK-LABEL: @pack_layout_64_0
// CHECK: "hivm_regbaseintrins.intr.hivm.pge.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.ppack.z"
// CHECK-NOT: ave.hir.preg
func.func @pack_layout_64_0(%dst: memref<128xi16, #hivm.address_space<ub>>) {
  %c0 = arith.constant 0 : index
  %p = ave.hir.pge <ALL> : vector<64xi1>
  %packed = ave.hir.preg.pack %p, 0 : vector<64xi1> -> vector<128xi1>
  %c1 = arith.constant 1 : i16
  %ones = ave.hir.scalar_broadcast %c1 : i16 -> vector<128xi16>
  %c0v = arith.constant 0 : i16
  %zero = ave.hir.scalar_broadcast %c0v : i16 -> vector<128xi16>
  %v = ave.hir.vsel %packed, %ones, %zero : vector<128xi1>, vector<128xi16>
  %all = ave.hir.pge <ALL> : vector<128xi1>
  ave.hir.masked_store <NORM_B16> %dst[%c0], %all, %v : memref<128xi16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi16>
  return
}

// CHECK-LABEL: @pack_layout_64_1
// CHECK: "hivm_regbaseintrins.intr.hivm.pge.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.ppack.z"
// CHECK-NOT: ave.hir.preg
func.func @pack_layout_64_1(%dst: memref<128xi16, #hivm.address_space<ub>>) {
  %c0 = arith.constant 0 : index
  %p = ave.hir.pge <ALL> : vector<64xi1>
  %packed = ave.hir.preg.pack %p, 1 : vector<64xi1> -> vector<128xi1>
  %c1 = arith.constant 1 : i16
  %ones = ave.hir.scalar_broadcast %c1 : i16 -> vector<128xi16>
  %c0v = arith.constant 0 : i16
  %zero = ave.hir.scalar_broadcast %c0v : i16 -> vector<128xi16>
  %v = ave.hir.vsel %packed, %ones, %zero : vector<128xi1>, vector<128xi16>
  %all = ave.hir.pge <ALL> : vector<128xi1>
  ave.hir.masked_store <NORM_B16> %dst[%c0], %all, %v : memref<128xi16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi16>
  return
}

// CHECK-LABEL: @pack_layout_128_0
// CHECK: "hivm_regbaseintrins.intr.hivm.pge.b16"
// CHECK: "hivm_regbaseintrins.intr.hivm.ppack.z"
// CHECK-NOT: ave.hir.preg
func.func @pack_layout_128_0(%dst: memref<256xi8, #hivm.address_space<ub>>) {
  %c0 = arith.constant 0 : index
  %p = ave.hir.pge <ALL> : vector<128xi1>
  %packed = ave.hir.preg.pack %p, 0 : vector<128xi1> -> vector<256xi1>
  %c1 = arith.constant 1 : i8
  %ones = ave.hir.scalar_broadcast %c1 : i8 -> vector<256xi8>
  %c0v = arith.constant 0 : i8
  %zero = ave.hir.scalar_broadcast %c0v : i8 -> vector<256xi8>
  %v = ave.hir.vsel %packed, %ones, %zero : vector<256xi1>, vector<256xi8>
  %all = ave.hir.pge <ALL> : vector<256xi1>
  ave.hir.masked_store <NORM_B8> %dst[%c0], %all, %v : memref<256xi8, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  return
}

// CHECK-LABEL: @pack_layout_128_1
// CHECK: "hivm_regbaseintrins.intr.hivm.pge.b16"
// CHECK: "hivm_regbaseintrins.intr.hivm.ppack.z"
// CHECK-NOT: ave.hir.preg
func.func @pack_layout_128_1(%dst: memref<256xi8, #hivm.address_space<ub>>) {
  %c0 = arith.constant 0 : index
  %p = ave.hir.pge <ALL> : vector<128xi1>
  %packed = ave.hir.preg.pack %p, 1 : vector<128xi1> -> vector<256xi1>
  %c1 = arith.constant 1 : i8
  %ones = ave.hir.scalar_broadcast %c1 : i8 -> vector<256xi8>
  %c0v = arith.constant 0 : i8
  %zero = ave.hir.scalar_broadcast %c0v : i8 -> vector<256xi8>
  %v = ave.hir.vsel %packed, %ones, %zero : vector<256xi1>, vector<256xi8>
  %all = ave.hir.pge <ALL> : vector<256xi1>
  ave.hir.masked_store <NORM_B8> %dst[%c0], %all, %v : memref<256xi8, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  return
}

// CHECK-LABEL: @group_widen_layout
// CHECK: "hivm_regbaseintrins.intr.hivm.pge.b16"
// CHECK: "hivm_regbaseintrins.intr.hivm.vcgadd.u.x"
// CHECK: llvm.bitcast {{.*}} : vector<128xi16> to vector<64xi32>
// CHECK: "hivm_regbaseintrins.intr.hivm.pge.b32"
// CHECK-NOT: ave.hir.group_reduction
func.func @group_widen_layout(%src: memref<128xi16, #hivm.address_space<ub>>, %dst: memref<64xi32, #hivm.address_space<ub>>) {
  %c0 = arith.constant 0 : index
  %x = ave.hir.vload <NORM> %src[%c0] : memref<128xi16, #hivm.address_space<ub>> into vector<128xi16>
  %p = ave.hir.pge <H> : vector<128xi1>
  %r = ave.hir.group_reduction <add> %x, %p {cast = #hivm.cast<cast_unsigned>} : vector<128xi16>, vector<128xi1> -> vector<64xi32>
  %all = ave.hir.pge <ALL> : vector<64xi1>
  ave.hir.masked_store <NORM_B32> %dst[%c0], %all, %r : memref<64xi32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>
  return
}

// CHECK-LABEL: @interleave_layout_64
// CHECK: "hivm_regbaseintrins.intr.hivm.pintlv.b32"
// CHECK: "hivm_regbaseintrins.intr.hivm.pdintlv.b32"
// CHECK-NOT: ave.hir.preg
func.func @interleave_layout_64(%dst: memref<64xi32, #hivm.address_space<ub>>) {
  %c0 = arith.constant 0 : index
  %a = ave.hir.pge <ALL> : vector<64xi1>
  %b = ave.hir.pge <H> : vector<64xi1>
  %lo, %hi = ave.hir.preg.interleave %a, %b : vector<64xi1>
  %even, %odd = ave.hir.preg.deinterleave %lo, %hi : vector<64xi1>
  %c1 = arith.constant 1 : i32
  %one = ave.hir.scalar_broadcast %c1 : i32 -> vector<64xi32>
  %c0v = arith.constant 0 : i32
  %zero = ave.hir.scalar_broadcast %c0v : i32 -> vector<64xi32>
  %v = ave.hir.vsel %even, %one, %zero : vector<64xi1>, vector<64xi32>
  %out = ave.hir.vsel %odd, %v, %one : vector<64xi1>, vector<64xi32>
  ave.hir.masked_store <NORM_B32> %dst[%c0], %a, %out : memref<64xi32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>
  return
}

// CHECK-LABEL: @interleave_layout_128
// CHECK: "hivm_regbaseintrins.intr.hivm.pintlv.b16"
// CHECK: "hivm_regbaseintrins.intr.hivm.pdintlv.b16"
// CHECK-NOT: ave.hir.preg
func.func @interleave_layout_128(%dst: memref<128xi16, #hivm.address_space<ub>>) {
  %c0 = arith.constant 0 : index
  %a = ave.hir.pge <ALL> : vector<128xi1>
  %b = ave.hir.pge <H> : vector<128xi1>
  %lo, %hi = ave.hir.preg.interleave %a, %b : vector<128xi1>
  %even, %odd = ave.hir.preg.deinterleave %lo, %hi : vector<128xi1>
  %c1 = arith.constant 1 : i16
  %one = ave.hir.scalar_broadcast %c1 : i16 -> vector<128xi16>
  %c0v = arith.constant 0 : i16
  %zero = ave.hir.scalar_broadcast %c0v : i16 -> vector<128xi16>
  %v = ave.hir.vsel %even, %one, %zero : vector<128xi1>, vector<128xi16>
  %out = ave.hir.vsel %odd, %v, %one : vector<128xi1>, vector<128xi16>
  ave.hir.masked_store <NORM_B16> %dst[%c0], %a, %out : memref<128xi16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi16>
  return
}

// CHECK-LABEL: @interleave_layout_256
// CHECK: "hivm_regbaseintrins.intr.hivm.pintlv.b8"
// CHECK: "hivm_regbaseintrins.intr.hivm.pdintlv.b8"
// CHECK-NOT: ave.hir.preg
func.func @interleave_layout_256(%dst: memref<256xi8, #hivm.address_space<ub>>) {
  %c0 = arith.constant 0 : index
  %a = ave.hir.pge <ALL> : vector<256xi1>
  %b = ave.hir.pge <H> : vector<256xi1>
  %lo, %hi = ave.hir.preg.interleave %a, %b : vector<256xi1>
  %even, %odd = ave.hir.preg.deinterleave %lo, %hi : vector<256xi1>
  %c1 = arith.constant 1 : i8
  %one = ave.hir.scalar_broadcast %c1 : i8 -> vector<256xi8>
  %c0v = arith.constant 0 : i8
  %zero = ave.hir.scalar_broadcast %c0v : i8 -> vector<256xi8>
  %v = ave.hir.vsel %even, %one, %zero : vector<256xi1>, vector<256xi8>
  %out = ave.hir.vsel %odd, %v, %one : vector<256xi1>, vector<256xi8>
  ave.hir.masked_store <NORM_B8> %dst[%c0], %a, %out : memref<256xi8, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  return
}
