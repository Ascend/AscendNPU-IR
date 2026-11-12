// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9579 -canonicalize -convert-hivmave-to-ave-intrin -canonicalize -convert-func-to-llvm -reconcile-unrealized-casts | bishengir-translate -mlir-to-llvmir | FileCheck %s
// New public register ops must export valid intrinsic signatures, including
// logical predicates and the legacy i16 group-add ABI. Canonicalization must
// not fold a logical pack into an all-true predicate of the larger length.

// CHECK-LABEL: define <128 x i16> @pack_64_0(
// CHECK: call <256 x i1> @llvm.hivm.pge.b32
// CHECK: call <256 x i1> @llvm.hivm.ppack.z({{.*}}i32 0)
// CHECK: call <128 x i16> @llvm.hivm.vsel
func.func @pack_64_0() -> vector<128xi16> {
  %p = ave.hir.pge <ALL> : vector<64xi1>
  %r = ave.hir.preg.pack %p, 0 : vector<64xi1> -> vector<128xi1>
  %one = arith.constant dense<1> : vector<128xi16>
  %zero = arith.constant dense<0> : vector<128xi16>
  %out = ave.hir.vsel %r, %one, %zero : vector<128xi1>, vector<128xi16>
  return %out : vector<128xi16>
}

// CHECK-LABEL: define <64 x i32> @unpack_128_0(
// CHECK: call <256 x i1> @llvm.hivm.pge.b16
// CHECK: call <256 x i1> @llvm.hivm.punpack({{.*}}i32 0)
// CHECK: call <64 x i32> @llvm.hivm.vsel
func.func @unpack_128_0() -> vector<64xi32> {
  %p = ave.hir.pge <ALL> : vector<128xi1>
  %r = ave.hir.preg.unpack %p, 0 : vector<128xi1> -> vector<64xi1>
  %one = arith.constant dense<1> : vector<64xi32>
  %zero = arith.constant dense<0> : vector<64xi32>
  %out = ave.hir.vsel %r, %one, %zero : vector<64xi1>, vector<64xi32>
  return %out : vector<64xi32>
}

// CHECK-LABEL: define <128 x i16> @pack_64_1(
// CHECK: call <256 x i1> @llvm.hivm.pge.b32
// CHECK: call <256 x i1> @llvm.hivm.ppack.z({{.*}}i32 1)
// CHECK: call <128 x i16> @llvm.hivm.vsel
func.func @pack_64_1() -> vector<128xi16> {
  %p = ave.hir.pge <ALL> : vector<64xi1>
  %r = ave.hir.preg.pack %p, 1 : vector<64xi1> -> vector<128xi1>
  %one = arith.constant dense<1> : vector<128xi16>
  %zero = arith.constant dense<0> : vector<128xi16>
  %out = ave.hir.vsel %r, %one, %zero : vector<128xi1>, vector<128xi16>
  return %out : vector<128xi16>
}

// CHECK-LABEL: define <64 x i32> @unpack_128_1(
// CHECK: call <256 x i1> @llvm.hivm.pge.b16
// CHECK: call <256 x i1> @llvm.hivm.punpack({{.*}}i32 1)
// CHECK: call <64 x i32> @llvm.hivm.vsel
func.func @unpack_128_1() -> vector<64xi32> {
  %p = ave.hir.pge <ALL> : vector<128xi1>
  %r = ave.hir.preg.unpack %p, 1 : vector<128xi1> -> vector<64xi1>
  %one = arith.constant dense<1> : vector<64xi32>
  %zero = arith.constant dense<0> : vector<64xi32>
  %out = ave.hir.vsel %r, %one, %zero : vector<64xi1>, vector<64xi32>
  return %out : vector<64xi32>
}

// CHECK-LABEL: define <256 x i8> @pack_128_0(
// CHECK: call <256 x i1> @llvm.hivm.pge.b16
// CHECK: call <256 x i1> @llvm.hivm.ppack.z({{.*}}i32 0)
// CHECK: call <256 x i8> @llvm.hivm.vsel
func.func @pack_128_0() -> vector<256xi8> {
  %p = ave.hir.pge <ALL> : vector<128xi1>
  %r = ave.hir.preg.pack %p, 0 : vector<128xi1> -> vector<256xi1>
  %one = arith.constant dense<1> : vector<256xi8>
  %zero = arith.constant dense<0> : vector<256xi8>
  %out = ave.hir.vsel %r, %one, %zero : vector<256xi1>, vector<256xi8>
  return %out : vector<256xi8>
}

// CHECK-LABEL: define <128 x i16> @unpack_256_0(
// CHECK: call <256 x i1> @llvm.hivm.pge.b8
// CHECK: call <256 x i1> @llvm.hivm.punpack({{.*}}i32 0)
// CHECK: call <128 x i16> @llvm.hivm.vsel
func.func @unpack_256_0() -> vector<128xi16> {
  %p = ave.hir.pge <ALL> : vector<256xi1>
  %r = ave.hir.preg.unpack %p, 0 : vector<256xi1> -> vector<128xi1>
  %one = arith.constant dense<1> : vector<128xi16>
  %zero = arith.constant dense<0> : vector<128xi16>
  %out = ave.hir.vsel %r, %one, %zero : vector<128xi1>, vector<128xi16>
  return %out : vector<128xi16>
}

// CHECK-LABEL: define <256 x i8> @pack_128_1(
// CHECK: call <256 x i1> @llvm.hivm.pge.b16
// CHECK: call <256 x i1> @llvm.hivm.ppack.z({{.*}}i32 1)
// CHECK: call <256 x i8> @llvm.hivm.vsel
func.func @pack_128_1() -> vector<256xi8> {
  %p = ave.hir.pge <ALL> : vector<128xi1>
  %r = ave.hir.preg.pack %p, 1 : vector<128xi1> -> vector<256xi1>
  %one = arith.constant dense<1> : vector<256xi8>
  %zero = arith.constant dense<0> : vector<256xi8>
  %out = ave.hir.vsel %r, %one, %zero : vector<256xi1>, vector<256xi8>
  return %out : vector<256xi8>
}

// CHECK-LABEL: define <128 x i16> @unpack_256_1(
// CHECK: call <256 x i1> @llvm.hivm.pge.b8
// CHECK: call <256 x i1> @llvm.hivm.punpack({{.*}}i32 1)
// CHECK: call <128 x i16> @llvm.hivm.vsel
func.func @unpack_256_1() -> vector<128xi16> {
  %p = ave.hir.pge <ALL> : vector<256xi1>
  %r = ave.hir.preg.unpack %p, 1 : vector<256xi1> -> vector<128xi1>
  %one = arith.constant dense<1> : vector<128xi16>
  %zero = arith.constant dense<0> : vector<128xi16>
  %out = ave.hir.vsel %r, %one, %zero : vector<128xi1>, vector<128xi16>
  return %out : vector<128xi16>
}

// CHECK-LABEL: define <64 x i32> @interleave_64(
// CHECK: call {{.*}} @llvm.hivm.pintlv.b32
// CHECK: extractvalue {{.*}}, 0
// CHECK: extractvalue {{.*}}, 1
// CHECK: call {{.*}} @llvm.hivm.pdintlv.b32
func.func @interleave_64() -> vector<64xi32> {
  %a = ave.hir.pge <ALL> : vector<64xi1>
  %b = ave.hir.pge <H> : vector<64xi1>
  %lo, %hi = ave.hir.preg.interleave %a, %b : vector<64xi1>
  %even, %odd = ave.hir.preg.deinterleave %lo, %hi : vector<64xi1>
  %one = arith.constant dense<1> : vector<64xi32>
  %zero = arith.constant dense<0> : vector<64xi32>
  %x = ave.hir.vsel %even, %one, %zero : vector<64xi1>, vector<64xi32>
  %out = ave.hir.vsel %odd, %x, %one : vector<64xi1>, vector<64xi32>
  return %out : vector<64xi32>
}

// CHECK-LABEL: define <128 x i16> @interleave_128(
// CHECK: call {{.*}} @llvm.hivm.pintlv.b16
// CHECK: extractvalue {{.*}}, 0
// CHECK: extractvalue {{.*}}, 1
// CHECK: call {{.*}} @llvm.hivm.pdintlv.b16
func.func @interleave_128() -> vector<128xi16> {
  %a = ave.hir.pge <ALL> : vector<128xi1>
  %b = ave.hir.pge <H> : vector<128xi1>
  %lo, %hi = ave.hir.preg.interleave %a, %b : vector<128xi1>
  %even, %odd = ave.hir.preg.deinterleave %lo, %hi : vector<128xi1>
  %one = arith.constant dense<1> : vector<128xi16>
  %zero = arith.constant dense<0> : vector<128xi16>
  %x = ave.hir.vsel %even, %one, %zero : vector<128xi1>, vector<128xi16>
  %out = ave.hir.vsel %odd, %x, %one : vector<128xi1>, vector<128xi16>
  return %out : vector<128xi16>
}

// CHECK-LABEL: define <256 x i8> @interleave_256(
// CHECK: call {{.*}} @llvm.hivm.pintlv.b8
// CHECK: extractvalue {{.*}}, 0
// CHECK: extractvalue {{.*}}, 1
// CHECK: call {{.*}} @llvm.hivm.pdintlv.b8
func.func @interleave_256() -> vector<256xi8> {
  %a = ave.hir.pge <ALL> : vector<256xi1>
  %b = ave.hir.pge <H> : vector<256xi1>
  %lo, %hi = ave.hir.preg.interleave %a, %b : vector<256xi1>
  %even, %odd = ave.hir.preg.deinterleave %lo, %hi : vector<256xi1>
  %one = arith.constant dense<1> : vector<256xi8>
  %zero = arith.constant dense<0> : vector<256xi8>
  %x = ave.hir.vsel %even, %one, %zero : vector<256xi1>, vector<256xi8>
  %out = ave.hir.vsel %odd, %x, %one : vector<256xi1>, vector<256xi8>
  return %out : vector<256xi8>
}

// CHECK-LABEL: @i8_gather(
// CHECK: call <256 x i8> @llvm.hivm.vselr.v256i8(<256 x i8>
// CHECK: call <256 x i8> @llvm.hivm.vbr.v256i8(i8 0)
// CHECK: call <256 x i8> @llvm.hivm.vsel.v256i8
func.func @i8_gather(%src: vector<256xi8>, %idx: vector<256xi8>) -> vector<256xi8> {
  %mask = ave.hir.pge <H> : vector<256xi1>
  %r = ave.hir.vselr %src, %idx, %mask : vector<256xi8>, vector<256xi8>, vector<256xi1>
  return %r : vector<256xi8>
}

// CHECK-LABEL: @i16_gather(
// CHECK: call <128 x i16> @llvm.hivm.vselr.v128i16(<128 x i16>
// CHECK: call <128 x i16> @llvm.hivm.vsel.v128i16
func.func @i16_gather(%src: vector<128xi16>, %idx: vector<128xi16>) -> vector<128xi16> {
  %mask = ave.hir.pge <H> : vector<128xi1>
  %r = ave.hir.vselr %src, %idx, %mask : vector<128xi16>, vector<128xi16>, vector<128xi1>
  return %r : vector<128xi16>
}

// CHECK-LABEL: @i32_gather(
// CHECK: call <64 x i32> @llvm.hivm.vselr.v64i32(<64 x i32>
// CHECK: call <64 x i32> @llvm.hivm.vsel.v64i32
func.func @i32_gather(%src: vector<64xi32>, %idx: vector<64xi32>) -> vector<64xi32> {
  %mask = ave.hir.pge <H> : vector<64xi1>
  %r = ave.hir.vselr %src, %idx, %mask : vector<64xi32>, vector<64xi32>, vector<64xi1>
  return %r : vector<64xi32>
}

// CHECK-LABEL: @f16_gather(
// CHECK: call <128 x i16> @llvm.hivm.vselr.v128i16(<128 x i16>
// CHECK: call <128 x i16> @llvm.hivm.vsel.v128i16
func.func @f16_gather(%src: vector<128xf16>, %idx: vector<128xi16>) -> vector<128xf16> {
  %mask = ave.hir.pge <H> : vector<128xi1>
  %r = ave.hir.vselr %src, %idx, %mask : vector<128xf16>, vector<128xi16>, vector<128xi1>
  return %r : vector<128xf16>
}

// CHECK-LABEL: @bf16_gather(
// CHECK: call <128 x i16> @llvm.hivm.vselr.v128i16(<128 x i16>
// CHECK: call <128 x i16> @llvm.hivm.vsel.v128i16
func.func @bf16_gather(%src: vector<128xbf16>, %idx: vector<128xi16>) -> vector<128xbf16> {
  %mask = ave.hir.pge <H> : vector<128xi1>
  %r = ave.hir.vselr %src, %idx, %mask : vector<128xbf16>, vector<128xi16>, vector<128xi1>
  return %r : vector<128xbf16>
}

// CHECK-LABEL: @f32_gather(
// CHECK: call <64 x i32> @llvm.hivm.vselr.v64i32(<64 x i32>
// CHECK: call <64 x i32> @llvm.hivm.vsel.v64i32
func.func @f32_gather(%src: vector<64xf32>, %idx: vector<64xi32>) -> vector<64xf32> {
  %mask = ave.hir.pge <H> : vector<64xi1>
  %r = ave.hir.vselr %src, %idx, %mask : vector<64xf32>, vector<64xi32>, vector<64xi1>
  return %r : vector<64xf32>
}

// CHECK-LABEL: @f8E4M3FN_gather(
// CHECK: call <256 x i8> @llvm.hivm.vselr.v256i8(<256 x i8>
// CHECK: call <256 x i8> @llvm.hivm.vsel.v256i8
func.func @f8E4M3FN_gather(%src: vector<256xf8E4M3FN>, %idx: vector<256xi8>) -> vector<256xf8E4M3FN> {
  %mask = ave.hir.pge <H> : vector<256xi1>
  %r = ave.hir.vselr %src, %idx, %mask : vector<256xf8E4M3FN>, vector<256xi8>, vector<256xi1>
  return %r : vector<256xf8E4M3FN>
}

// CHECK-LABEL: @f8E5M2_gather(
// CHECK: call <256 x i8> @llvm.hivm.vselr.v256i8(<256 x i8>
// CHECK: call <256 x i8> @llvm.hivm.vsel.v256i8
func.func @f8E5M2_gather(%src: vector<256xf8E5M2>, %idx: vector<256xi8>) -> vector<256xf8E5M2> {
  %mask = ave.hir.pge <H> : vector<256xi1>
  %r = ave.hir.vselr %src, %idx, %mask : vector<256xf8E5M2>, vector<256xi8>, vector<256xi1>
  return %r : vector<256xf8E5M2>
}

// CHECK-LABEL: @add_i16_s(
// CHECK: call <128 x i16> @llvm.hivm.vcgadd.s.x.v128i16
// CHECK: bitcast <128 x i16> {{.*}} to <64 x i32>
// CHECK: call <256 x i1> @llvm.hivm.pge.b32(i32 5, i32 0)
// CHECK: call <64 x i32> @llvm.hivm.vbr.v64i32(i32 0)
// CHECK: call {{.*}} @llvm.hivm.vsel
func.func @add_i16_s(%src: vector<128xi16>) -> vector<64xi32> {
  %mask = ave.hir.pge <H> : vector<128xi1>
  %r = ave.hir.group_reduction <add> %src, %mask {cast = #hivm.cast<cast_signed>} : vector<128xi16>, vector<128xi1> -> vector<64xi32>
  return %r : vector<64xi32>
}

// CHECK-LABEL: @add_i16_u(
// CHECK: call <128 x i16> @llvm.hivm.vcgadd.u.x.v128i16
// CHECK: bitcast <128 x i16> {{.*}} to <64 x i32>
// CHECK: call <256 x i1> @llvm.hivm.pge.b32(i32 5, i32 0)
// CHECK: call {{.*}} @llvm.hivm.vsel
func.func @add_i16_u(%src: vector<128xi16>) -> vector<64xi32> {
  %mask = ave.hir.pge <H> : vector<128xi1>
  %r = ave.hir.group_reduction <add> %src, %mask {cast = #hivm.cast<cast_unsigned>} : vector<128xi16>, vector<128xi1> -> vector<64xi32>
  return %r : vector<64xi32>
}

// CHECK-LABEL: @add_i32_s(
// CHECK: call <64 x i32> @llvm.hivm.vcgadd.s.x.v64i32
// CHECK: call <256 x i1> @llvm.hivm.pge.b32(i32 5, i32 0)
// CHECK: call {{.*}} @llvm.hivm.vsel
func.func @add_i32_s(%src: vector<64xi32>) -> vector<64xi32> {
  %mask = ave.hir.pge <H> : vector<64xi1>
  %r = ave.hir.group_reduction <add> %src, %mask {cast = #hivm.cast<cast_signed>} : vector<64xi32>, vector<64xi1> -> vector<64xi32>
  return %r : vector<64xi32>
}

// CHECK-LABEL: @add_i32_u(
// CHECK: call <64 x i32> @llvm.hivm.vcgadd.u.x.v64i32
// CHECK: call <256 x i1> @llvm.hivm.pge.b32(i32 5, i32 0)
// CHECK: call {{.*}} @llvm.hivm.vsel
func.func @add_i32_u(%src: vector<64xi32>) -> vector<64xi32> {
  %mask = ave.hir.pge <H> : vector<64xi1>
  %r = ave.hir.group_reduction <add> %src, %mask {cast = #hivm.cast<cast_unsigned>} : vector<64xi32>, vector<64xi1> -> vector<64xi32>
  return %r : vector<64xi32>
}

// CHECK-LABEL: @add_f16_s(
// CHECK: call <128 x half> @llvm.hivm.vcgadd.s.x.v128f16
// CHECK: call <256 x i1> @llvm.hivm.pge.b16(i32 5, i32 0)
// CHECK: call {{.*}} @llvm.hivm.vsel
func.func @add_f16_s(%src: vector<128xf16>) -> vector<128xf16> {
  %mask = ave.hir.pge <H> : vector<128xi1>
  %r = ave.hir.group_reduction <add> %src, %mask {cast = #hivm.cast<cast_signed>} : vector<128xf16>, vector<128xi1> -> vector<128xf16>
  return %r : vector<128xf16>
}

// CHECK-LABEL: @add_f32_s(
// CHECK: call <64 x float> @llvm.hivm.vcgadd.s.x.v64f32
// CHECK: call <256 x i1> @llvm.hivm.pge.b32(i32 5, i32 0)
// CHECK: call {{.*}} @llvm.hivm.vsel
func.func @add_f32_s(%src: vector<64xf32>) -> vector<64xf32> {
  %mask = ave.hir.pge <H> : vector<64xi1>
  %r = ave.hir.group_reduction <add> %src, %mask {cast = #hivm.cast<cast_signed>} : vector<64xf32>, vector<64xi1> -> vector<64xf32>
  return %r : vector<64xf32>
}

// CHECK-LABEL: @min_i16_s(
// CHECK: call <128 x i16> @llvm.hivm.vcgmin.s.x.v128i16
// CHECK: call <256 x i1> @llvm.hivm.pge.b16(i32 5, i32 0)
// CHECK: call {{.*}} @llvm.hivm.vsel
func.func @min_i16_s(%src: vector<128xi16>) -> vector<128xi16> {
  %mask = ave.hir.pge <H> : vector<128xi1>
  %r = ave.hir.group_reduction <min> %src, %mask {cast = #hivm.cast<cast_signed>} : vector<128xi16>, vector<128xi1> -> vector<128xi16>
  return %r : vector<128xi16>
}

// CHECK-LABEL: @min_i16_u(
// CHECK: call <128 x i16> @llvm.hivm.vcgmin.u.x.v128i16
// CHECK: call <256 x i1> @llvm.hivm.pge.b16(i32 5, i32 0)
// CHECK: call {{.*}} @llvm.hivm.vsel
func.func @min_i16_u(%src: vector<128xi16>) -> vector<128xi16> {
  %mask = ave.hir.pge <H> : vector<128xi1>
  %r = ave.hir.group_reduction <min> %src, %mask {cast = #hivm.cast<cast_unsigned>} : vector<128xi16>, vector<128xi1> -> vector<128xi16>
  return %r : vector<128xi16>
}

// CHECK-LABEL: @min_i32_s(
// CHECK: call <64 x i32> @llvm.hivm.vcgmin.s.x.v64i32
// CHECK: call <256 x i1> @llvm.hivm.pge.b32(i32 5, i32 0)
// CHECK: call {{.*}} @llvm.hivm.vsel
func.func @min_i32_s(%src: vector<64xi32>) -> vector<64xi32> {
  %mask = ave.hir.pge <H> : vector<64xi1>
  %r = ave.hir.group_reduction <min> %src, %mask {cast = #hivm.cast<cast_signed>} : vector<64xi32>, vector<64xi1> -> vector<64xi32>
  return %r : vector<64xi32>
}

// CHECK-LABEL: @min_i32_u(
// CHECK: call <64 x i32> @llvm.hivm.vcgmin.u.x.v64i32
// CHECK: call <256 x i1> @llvm.hivm.pge.b32(i32 5, i32 0)
// CHECK: call {{.*}} @llvm.hivm.vsel
func.func @min_i32_u(%src: vector<64xi32>) -> vector<64xi32> {
  %mask = ave.hir.pge <H> : vector<64xi1>
  %r = ave.hir.group_reduction <min> %src, %mask {cast = #hivm.cast<cast_unsigned>} : vector<64xi32>, vector<64xi1> -> vector<64xi32>
  return %r : vector<64xi32>
}

// CHECK-LABEL: @min_f16_s(
// CHECK: call <128 x half> @llvm.hivm.vcgmin.s.x.v128f16
// CHECK: call <256 x i1> @llvm.hivm.pge.b16(i32 5, i32 0)
// CHECK: call {{.*}} @llvm.hivm.vsel
func.func @min_f16_s(%src: vector<128xf16>) -> vector<128xf16> {
  %mask = ave.hir.pge <H> : vector<128xi1>
  %r = ave.hir.group_reduction <min> %src, %mask {cast = #hivm.cast<cast_signed>} : vector<128xf16>, vector<128xi1> -> vector<128xf16>
  return %r : vector<128xf16>
}

// CHECK-LABEL: @min_f32_s(
// CHECK: call <64 x float> @llvm.hivm.vcgmin.s.x.v64f32
// CHECK: call <256 x i1> @llvm.hivm.pge.b32(i32 5, i32 0)
// CHECK: call {{.*}} @llvm.hivm.vsel
func.func @min_f32_s(%src: vector<64xf32>) -> vector<64xf32> {
  %mask = ave.hir.pge <H> : vector<64xi1>
  %r = ave.hir.group_reduction <min> %src, %mask {cast = #hivm.cast<cast_signed>} : vector<64xf32>, vector<64xi1> -> vector<64xf32>
  return %r : vector<64xf32>
}

// CHECK-LABEL: @max_i16_s(
// CHECK: call <128 x i16> @llvm.hivm.vcgmax.s.x.v128i16
// CHECK: call <256 x i1> @llvm.hivm.pge.b16(i32 5, i32 0)
// CHECK: call {{.*}} @llvm.hivm.vsel
func.func @max_i16_s(%src: vector<128xi16>) -> vector<128xi16> {
  %mask = ave.hir.pge <H> : vector<128xi1>
  %r = ave.hir.group_reduction <max> %src, %mask {cast = #hivm.cast<cast_signed>} : vector<128xi16>, vector<128xi1> -> vector<128xi16>
  return %r : vector<128xi16>
}

// CHECK-LABEL: @max_i16_u(
// CHECK: call <128 x i16> @llvm.hivm.vcgmax.u.x.v128i16
// CHECK: call <256 x i1> @llvm.hivm.pge.b16(i32 5, i32 0)
// CHECK: call {{.*}} @llvm.hivm.vsel
func.func @max_i16_u(%src: vector<128xi16>) -> vector<128xi16> {
  %mask = ave.hir.pge <H> : vector<128xi1>
  %r = ave.hir.group_reduction <max> %src, %mask {cast = #hivm.cast<cast_unsigned>} : vector<128xi16>, vector<128xi1> -> vector<128xi16>
  return %r : vector<128xi16>
}

// CHECK-LABEL: @max_i32_s(
// CHECK: call <64 x i32> @llvm.hivm.vcgmax.s.x.v64i32
// CHECK: call <256 x i1> @llvm.hivm.pge.b32(i32 5, i32 0)
// CHECK: call {{.*}} @llvm.hivm.vsel
func.func @max_i32_s(%src: vector<64xi32>) -> vector<64xi32> {
  %mask = ave.hir.pge <H> : vector<64xi1>
  %r = ave.hir.group_reduction <max> %src, %mask {cast = #hivm.cast<cast_signed>} : vector<64xi32>, vector<64xi1> -> vector<64xi32>
  return %r : vector<64xi32>
}

// CHECK-LABEL: @max_i32_u(
// CHECK: call <64 x i32> @llvm.hivm.vcgmax.u.x.v64i32
// CHECK: call <256 x i1> @llvm.hivm.pge.b32(i32 5, i32 0)
// CHECK: call {{.*}} @llvm.hivm.vsel
func.func @max_i32_u(%src: vector<64xi32>) -> vector<64xi32> {
  %mask = ave.hir.pge <H> : vector<64xi1>
  %r = ave.hir.group_reduction <max> %src, %mask {cast = #hivm.cast<cast_unsigned>} : vector<64xi32>, vector<64xi1> -> vector<64xi32>
  return %r : vector<64xi32>
}

// CHECK-LABEL: @max_f16_s(
// CHECK: call <128 x half> @llvm.hivm.vcgmax.s.x.v128f16
// CHECK: call <256 x i1> @llvm.hivm.pge.b16(i32 5, i32 0)
// CHECK: call {{.*}} @llvm.hivm.vsel
func.func @max_f16_s(%src: vector<128xf16>) -> vector<128xf16> {
  %mask = ave.hir.pge <H> : vector<128xi1>
  %r = ave.hir.group_reduction <max> %src, %mask {cast = #hivm.cast<cast_signed>} : vector<128xf16>, vector<128xi1> -> vector<128xf16>
  return %r : vector<128xf16>
}

// CHECK-LABEL: @max_f32_s(
// CHECK: call <64 x float> @llvm.hivm.vcgmax.s.x.v64f32
// CHECK: call <256 x i1> @llvm.hivm.pge.b32(i32 5, i32 0)
// CHECK: call {{.*}} @llvm.hivm.vsel
func.func @max_f32_s(%src: vector<64xf32>) -> vector<64xf32> {
  %mask = ave.hir.pge <H> : vector<64xi1>
  %r = ave.hir.group_reduction <max> %src, %mask {cast = #hivm.cast<cast_signed>} : vector<64xf32>, vector<64xi1> -> vector<64xf32>
  return %r : vector<64xf32>
}
