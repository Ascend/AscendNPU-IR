//===- AscendDPXOps.td - Ascend SIMT Ops Definition --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDDPX_OPS
#define ASCENDDPX_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "bishengir/Dialect/AscendDPX/IR/AscendDPXBase.td"
include "bishengir/Dialect/AscendDPX/IR/AscendDPXAttrs.td"

//===----------------------------------------------------------------------===//
// Program Info Ops
//===----------------------------------------------------------------------===//
class AscendDPX_InfoOp<string mnemonic, list<Trait> traits = [Pure]>
    : AscendDPX_Op<mnemonic, traits> {
  let results = (outs I32:$res);
  let assemblyFormat = "attr-dict";
}

def ThreadIdXOp : AscendDPX_InfoOp<"thread_id_x">;
def ThreadIdYOp : AscendDPX_InfoOp<"thread_id_y">;
def ThreadIdZOp : AscendDPX_InfoOp<"thread_id_z">;

def BlockIdxXOp : AscendDPX_InfoOp<"block_idx_x">;
def BlockIdxYOp : AscendDPX_InfoOp<"block_idx_y">;
def BlockIdxZOp : AscendDPX_InfoOp<"block_idx_z">;
def BlockIdxOp : AscendDPX_InfoOp<"block_idx">;

def BlockDimXOp : AscendDPX_InfoOp<"block_dim_x">;
def BlockDimYOp : AscendDPX_InfoOp<"block_dim_y">;
def BlockDimZOp : AscendDPX_InfoOp<"block_dim_z">;

def GridDimXOp : AscendDPX_InfoOp<"grid_dim_x">;
def GridDimYOp : AscendDPX_InfoOp<"grid_dim_y">;
def GridDimZOp : AscendDPX_InfoOp<"grid_dim_z">;

def Clock32Op : AscendDPX_InfoOp<"clock32", []>;
def Clock64Op : AscendDPX_InfoOp<"clock64", []> {
  let results = (outs I64:$res);
}

def CoreIdOp : AscendDPX_InfoOp<"core_id">;

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//
def LoadOp : AscendDPX_Op<"load", [AttrSizedOperandSegments, MemoryEffects<[MemRead]>]> {
  let summary = "David SIMT load operation";
  let description = [{
    Load operation with masking and cache policy support. If the mask is true, loads from the given pointer.
  }];
  let arguments = (ins
    LLVM_AnyPointer:$ptr,
    Optional<I1>:$mask,
    Optional<LLVM_Type>:$falseVal,
    DefaultValuedAttr<AscendDPX_LoadCachePolicyAttr, 
                      "::mlir::ascend_dpx::AscendDPXLoadCachePolicy::L2_CACHE_HINT_NORMAL_FV">:$cacheHint,
    DefaultValuedAttr<AscendDPX_CacheOptionAttr,
                      "::mlir::ascend_dpx::AscendDPXCacheOption::LOADCACHEOPTION_CA">:$cacheOption,
    DefaultValuedAttr<AscendDPX_VolatileOptionAttr,
                      "::mlir::ascend_dpx::AscendDPXVolatileOption::NONVOLATILE">:$volatileOption
  );

  let results = (outs LLVM_Type:$result);

  let assemblyFormat = [{
    $ptr (`,` $mask^)? (`,` $falseVal^)?
    oilist(`cacheModifier` `=` $cacheHint | `cacheOption` `=` $cacheOption | `volatileOption` `=` $volatileOption)
    attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//
def StoreOp : AscendDPX_Op<"store",[MemoryEffects<[MemWrite]>]> {
  let summary = "David SIMT store operation";
  let description = [{
    Store operation with masking and cache policy support. If the mask is true, stores to the given pointer.
  }];
  let arguments = (ins
    LLVM_AnyPointer:$ptr,
    LLVM_Type:$value,
    Optional<I1>:$mask,
    DefaultValuedAttr<AscendDPX_StoreCachePolicyAttr, 
                      "::mlir::ascend_dpx::AscendDPXStoreCachePolicy::L2_CACHE_HINT_NORMAL_FV">:$cacheHint,
    DefaultValuedAttr<AscendDPX_CacheOptionAttr,
                      "::mlir::ascend_dpx::AscendDPXCacheOption::LOADCACHEOPTION_CA">:$cacheOption
  );

  let assemblyFormat = [{
    $ptr `,` $value (`,` $mask^)?
    oilist(`cacheModifier` `=` $cacheHint | `cacheOption` `=` $cacheOption)
    attr-dict `:` type($ptr) `,` type($value)
  }];
}

//===----------------------------------------------------------------------===//
// Warp Shuffle/Sync Ops
//===----------------------------------------------------------------------===//
def SyncThreadsOp : AscendDPX_Op<"sync_threads"> {
  let assemblyFormat = "attr-dict";
}

class WarpShuffleOp<string mnemonic> : 
  AscendDPX_Op<mnemonic> {
  let arguments = (ins LLVM_Type:$src,
                       I32:$lane_mask,
                       I32:$clamp,
                       I32:$index);
  let results = (outs LLVM_Type:$res);
  let assemblyFormat = "$src `,` $lane_mask `,` $clamp `,` $index attr-dict `:` functional-type(operands, results)";
}

def ShflUpOp : WarpShuffleOp<"shfl.up">;
def ShflDownOp : WarpShuffleOp<"shfl.down">;
def ShflButterflyOp : WarpShuffleOp<"shfl.bfly">;
def ShflIndexOp : WarpShuffleOp<"shfl.idx">;

//===----------------------------------------------------------------------===//
// Warp Reduction Ops
//===----------------------------------------------------------------------===//
class WarpReduceOp<string mnemonic> :
  AscendDPX_Op<mnemonic> {
  let arguments = (ins LLVM_Type:$src);
  let results = (outs LLVM_Type:$res);
  let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";
}

def ReduceAdd : WarpReduceOp<"reduce.add">;
def ReduceMax : WarpReduceOp<"reduce.max">;
def ReduceMin : WarpReduceOp<"reduce.min">;
def ReduceUMax : WarpReduceOp<"reduce.umax">;
def ReduceUMin : WarpReduceOp<"reduce.umin">;

//===----------------------------------------------------------------------===//
// Atomic Ops
//===----------------------------------------------------------------------===//
class AscendDPX_AtomicOp<string mnemonic> : 
  AscendDPX_Op<mnemonic> {
  let arguments = (ins LLVM_AnyPointer:$src,
                       LLVM_Type:$data,
                       DefaultValuedAttr<AscendDPX_StoreCachePolicyAttr, 
                                         "::mlir::ascend_dpx::AscendDPXStoreCachePolicy::L2_CACHE_HINT_NORMAL_FV">:$cache);
  let results = (outs LLVM_Type:$res);
  let assemblyFormat = [{$src `,` $data
    oilist(`cacheModifier` `=` $cache)
    attr-dict `:` functional-type(operands, results)}];
}

def AtomicAndOp : AscendDPX_AtomicOp<"atomic_and">;
def AtomicOrOp : AscendDPX_AtomicOp<"atomic_or">;
def AtomicXorOp : AscendDPX_AtomicOp<"atomic_xor">;
def AtomicIncOp : AscendDPX_AtomicOp<"atomic_inc">;
def AtomicDecOp : AscendDPX_AtomicOp<"atomic_dec">;
def AtomicMaxOp : AscendDPX_AtomicOp<"atomic_max">;
def AtomicMinOp : AscendDPX_AtomicOp<"atomic_min">;
def AtomicAddOp : AscendDPX_AtomicOp<"atomic_add">;
def AtomicSubOp : AscendDPX_AtomicOp<"atomic_sub">;
def AtomicUMaxOp : AscendDPX_AtomicOp<"atomic_umax">;
def AtomicUMinOp : AscendDPX_AtomicOp<"atomic_umin">;
def AtomicExchangeOp : AscendDPX_AtomicOp<"atomic_exchange">;
def AtomicCASOp : AscendDPX_AtomicOp<"atomic_cas"> {
  let arguments = (ins LLVM_AnyPointer:$src,
                       LLVM_Type:$data,
                       LLVM_Type:$other,
                       DefaultValuedAttr<AscendDPX_StoreCachePolicyAttr, 
                                         "::mlir::ascend_dpx::AscendDPXStoreCachePolicy::L2_CACHE_HINT_NORMAL_FV">:$cache);
  let assemblyFormat = "$src `,` $data `,` $other attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//
def CastOp : AscendDPX_Op<"cast"> {
  let summary = "Generic type conversion operation";
  let description = [{
    Converts between compatible types while preserving structure.
  }];
  let arguments = (ins 
    LLVM_Type:$in,
    DefaultValuedAttr<AscendDPX_CastKindAttr, 
                      "::mlir::ascend_dpx::AscendDPXCastKind::SIGNED_TO_FLOAT">:$kind
  );
  let results = (outs LLVM_Type:$out);
  
  let assemblyFormat = "$in `kind` $kind attr-dict `:` type($in) `to` type($out)";
}

//===----------------------------------------------------------------------===//
// Custom Math Ops
//===----------------------------------------------------------------------===//
// All ops implemented by template library or HIVM intrinsics are here

class AscendDPX_UnaryMathOp<string mnemonic>
    : AscendDPX_Op<mnemonic> {
  let arguments = (ins LLVM_Type:$ins);
  let results = (outs LLVM_Type:$res);
  let assemblyFormat = "$ins attr-dict `:` functional-type(operands, results)";
}

// sorted alphabetically
def AbsOp : AscendDPX_UnaryMathOp<"abs">;
def AcosOp : AscendDPX_UnaryMathOp<"acos">;
def AcoshOp : AscendDPX_UnaryMathOp<"acosh">;
def AsinOp : AscendDPX_UnaryMathOp<"asin">;
def AsinhOp : AscendDPX_UnaryMathOp<"asinh">;
def AtanOp : AscendDPX_UnaryMathOp<"atan">;
def AtanhOp : AscendDPX_UnaryMathOp<"atanh">;
def BrevOp : AscendDPX_UnaryMathOp<"brev">;
def CbrtOp : AscendDPX_UnaryMathOp<"cbrt">;
def CeilOp : AscendDPX_UnaryMathOp<"ceil">;
def ClzOp : AscendDPX_UnaryMathOp<"clz">;
def CosOp : AscendDPX_UnaryMathOp<"cos">;
def CoshOp : AscendDPX_UnaryMathOp<"cosh">;
def CospiOp : AscendDPX_UnaryMathOp<"cospi">;
def CylBesselI0Op : AscendDPX_UnaryMathOp<"cyl_bessel_i0">;
def CylBesselI1Op : AscendDPX_UnaryMathOp<"cyl_bessel_i1">;
def ErfOp : AscendDPX_UnaryMathOp<"erf">;
def ErfcOp : AscendDPX_UnaryMathOp<"erfc">;
def ErfcinvOp : AscendDPX_UnaryMathOp<"erfcinv">;
def ErfcxOp : AscendDPX_UnaryMathOp<"erfcx">;
def ErfinvOp : AscendDPX_UnaryMathOp<"erfinv">;
def Exp10Op : AscendDPX_UnaryMathOp<"exp10">;
def Exp2Op : AscendDPX_UnaryMathOp<"exp2">;
def ExpOp : AscendDPX_UnaryMathOp<"exp">;
def Expm1Op : AscendDPX_UnaryMathOp<"expm1">;
def FastCosfOp : AscendDPX_UnaryMathOp<"fast_cosf">;
def FastExp10fOp : AscendDPX_UnaryMathOp<"fast_exp10f">;
def FastExpfOp : AscendDPX_UnaryMathOp<"fast_expf">;
def FastLog10fOp : AscendDPX_UnaryMathOp<"fast_log10f">;
def FastLog2fOp : AscendDPX_UnaryMathOp<"fast_log2f">;
def FastLogfOp : AscendDPX_UnaryMathOp<"fast_logf">;
def FastSinfOp : AscendDPX_UnaryMathOp<"fast_sinf">;
def FastTanfOp : AscendDPX_UnaryMathOp<"fast_tanf">;
def Float2IntRnOp : AscendDPX_UnaryMathOp<"float2int_rn">;
def Float2IntRzOp : AscendDPX_UnaryMathOp<"float2int_rz">;
def Float2IntRdOp : AscendDPX_UnaryMathOp<"float2int_rd">;
def Float2IntRuOp : AscendDPX_UnaryMathOp<"float2int_ru">;
def Float2LlRnOp : AscendDPX_UnaryMathOp<"float2ll_rn">;
def Float2LlRzOp : AscendDPX_UnaryMathOp<"float2ll_rz">;
def Float2LlRdOp : AscendDPX_UnaryMathOp<"float2ll_rd">;
def Float2LlRuOp : AscendDPX_UnaryMathOp<"float2ll_ru">;
def Float2UintRnOp : AscendDPX_UnaryMathOp<"float2uint_rn">;
def Float2UintRzOp : AscendDPX_UnaryMathOp<"float2uint_rz">;
def Float2UintRdOp : AscendDPX_UnaryMathOp<"float2uint_rd">;
def Float2UintRuOp : AscendDPX_UnaryMathOp<"float2uint_ru">;
def Float2UllRnOp : AscendDPX_UnaryMathOp<"float2ull_rn">;
def Float2UllRzOp : AscendDPX_UnaryMathOp<"float2ull_rz">;
def Float2UllRdOp : AscendDPX_UnaryMathOp<"float2ull_rd">;
def Float2UllRuOp : AscendDPX_UnaryMathOp<"float2ull_ru">;
def FloatAsIntOp : AscendDPX_UnaryMathOp<"float_as_int">;
def FloatAsUintOp : AscendDPX_UnaryMathOp<"float_as_uint">;
def FloorOp : AscendDPX_UnaryMathOp<"floor">;
def FfsOp : AscendDPX_UnaryMathOp<"ffs">;
def GammaOp : AscendDPX_UnaryMathOp<"gamma">;
def ILogbOp : AscendDPX_UnaryMathOp<"ilogb">;
def Int2FloatRnOp : AscendDPX_UnaryMathOp<"int2float_rn">;
def Int2FloatRzOp : AscendDPX_UnaryMathOp<"int2float_rz">;
def Int2FloatRdOp : AscendDPX_UnaryMathOp<"int2float_rd">;
def Int2FloatRuOp : AscendDPX_UnaryMathOp<"int2float_ru">;
def IntAsFloatOp : AscendDPX_UnaryMathOp<"int_as_float">;
def J0Op : AscendDPX_UnaryMathOp<"j0">;
def J1Op : AscendDPX_UnaryMathOp<"j1">;
def Ll2FloatRnOp : AscendDPX_UnaryMathOp<"ll2float_rn">;
def Ll2FloatRzOp : AscendDPX_UnaryMathOp<"ll2float_rz">;
def Ll2FloatRdOp : AscendDPX_UnaryMathOp<"ll2float_rd">;
def Ll2FloatRuOp : AscendDPX_UnaryMathOp<"ll2float_ru">;
def LlrintOp : AscendDPX_UnaryMathOp<"llrint">;
def LlroundOp : AscendDPX_UnaryMathOp<"llround">;
def Log10Op : AscendDPX_UnaryMathOp<"log10">;
def Log1pOp : AscendDPX_UnaryMathOp<"log1p">;
def Log2Op : AscendDPX_UnaryMathOp<"log2">;
def LogOp : AscendDPX_UnaryMathOp<"log">;
def LogbOp : AscendDPX_UnaryMathOp<"logb">;
def LgammaOp : AscendDPX_UnaryMathOp<"lgamma">;
def NearbyintOp : AscendDPX_UnaryMathOp<"nearbyint">;
def NormcdfOp : AscendDPX_UnaryMathOp<"normcdf">;
def NormcdfinvOp : AscendDPX_UnaryMathOp<"normcdfinv">;
def PopcOp : AscendDPX_UnaryMathOp<"popc">;
def RcpRdOp : AscendDPX_UnaryMathOp<"rcp_rd">;
def RcpRnOp : AscendDPX_UnaryMathOp<"rcp_rn">;
def RcpRuOp : AscendDPX_UnaryMathOp<"rcp_ru">;
def RcpRzOp : AscendDPX_UnaryMathOp<"rcp_rz">;
def RcbrtOp : AscendDPX_UnaryMathOp<"rcbrt">;
def RecipOp : AscendDPX_UnaryMathOp<"recip">;
def ReluOp : AscendDPX_UnaryMathOp<"relu">;
def RintOp : AscendDPX_UnaryMathOp<"rint">;
def RoundOp : AscendDPX_UnaryMathOp<"round">;
def RSqrtOp : AscendDPX_UnaryMathOp<"rsqrt">;
def RsqrtRdOp : AscendDPX_UnaryMathOp<"rsqrt_rd">;
def RsqrtRnOp : AscendDPX_UnaryMathOp<"rsqrt_rn">;
def RsqrtRuOp : AscendDPX_UnaryMathOp<"rsqrt_ru">;
def RsqrtRzOp : AscendDPX_UnaryMathOp<"rsqrt_rz">;
def SaturatefOp : AscendDPX_UnaryMathOp<"saturatef">;
def SignbitOp : AscendDPX_UnaryMathOp<"signbit">;
def SinOp : AscendDPX_UnaryMathOp<"sin">;
def SinhOp : AscendDPX_UnaryMathOp<"sinh">;
def SinpiOp : AscendDPX_UnaryMathOp<"sinpi">;
def SqrtOp : AscendDPX_UnaryMathOp<"sqrt">;
def SqrtRdOp : AscendDPX_UnaryMathOp<"sqrt_rd">;
def SqrtRnOp : AscendDPX_UnaryMathOp<"sqrt_rn">;
def SqrtRuOp : AscendDPX_UnaryMathOp<"sqrt_ru">;
def SqrtRzOp : AscendDPX_UnaryMathOp<"sqrt_rz">;
def TanOp : AscendDPX_UnaryMathOp<"tan">;
def TanhOp : AscendDPX_UnaryMathOp<"tanh">;
def TgammaOp : AscendDPX_UnaryMathOp<"tgamma">;
def TruncOp : AscendDPX_UnaryMathOp<"trunc">;
def Uint2FloatRnOp : AscendDPX_UnaryMathOp<"uint2float_rn">;
def Uint2FloatRzOp : AscendDPX_UnaryMathOp<"uint2float_rz">;
def Uint2FloatRdOp : AscendDPX_UnaryMathOp<"uint2float_rd">;
def Uint2FloatRuOp : AscendDPX_UnaryMathOp<"uint2float_ru">;
def UintAsFloatOp : AscendDPX_UnaryMathOp<"uint_as_float">;
def Ull2FloatRnOp : AscendDPX_UnaryMathOp<"ull2float_rn">;
def Ull2FloatRzOp : AscendDPX_UnaryMathOp<"ull2float_rz">;
def Ull2FloatRdOp : AscendDPX_UnaryMathOp<"ull2float_rd">;
def Ull2FloatRuOp : AscendDPX_UnaryMathOp<"ull2float_ru">;
def Y0Op : AscendDPX_UnaryMathOp<"y0">;
def Y1Op : AscendDPX_UnaryMathOp<"y1">;
def ReciprocalOp : AscendDPX_UnaryMathOp<"reciprocal">;

class AscendDPX_BinaryMathOp<string mnemonic>
    : AscendDPX_Op<mnemonic> {
  let arguments = (ins LLVM_Type:$in1, LLVM_Type:$in2);
  let results = (outs LLVM_Type:$res);
  let assemblyFormat = "$in1 `,` $in2 attr-dict `:` functional-type(operands, results)";
}

def DivOp : AscendDPX_BinaryMathOp<"div">;
def UDivOp : AscendDPX_BinaryMathOp<"udiv">;
def PowOp : AscendDPX_BinaryMathOp<"pow">;
def UmulhiOp : AscendDPX_BinaryMathOp<"umulhi">;
def LdExpOp : AscendDPX_BinaryMathOp<"ldexp">;
def CopysignOp : AscendDPX_BinaryMathOp<"copysign">;
def Atan2Op : AscendDPX_BinaryMathOp<"atan2">;
def NextafterOp : AscendDPX_BinaryMathOp<"nextafter">;
def HypotOp : AscendDPX_BinaryMathOp<"hypot">;
def MulhiOp : AscendDPX_BinaryMathOp<"mulhi">;
def Mul24Op : AscendDPX_BinaryMathOp<"mul24">;
def HaddOp : AscendDPX_BinaryMathOp<"hadd">;
def RhaddOp : AscendDPX_BinaryMathOp<"rhadd">;
def ScalbnOp : AscendDPX_BinaryMathOp<"scalbn">;
def FdimOp : AscendDPX_BinaryMathOp<"fdim">;
def FastDividefOp : AscendDPX_BinaryMathOp<"fast_dividef">;
def DivRnOp : AscendDPX_BinaryMathOp<"div_rn">;
def DivRzOp : AscendDPX_BinaryMathOp<"div_rz">;
def DivRdOp : AscendDPX_BinaryMathOp<"div_rd">;
def DivRuOp : AscendDPX_BinaryMathOp<"div_ru">;
def FmodOp : AscendDPX_BinaryMathOp<"fmod">;
def RemainderOp : AscendDPX_BinaryMathOp<"remainder">;
def AddRnOp : AscendDPX_BinaryMathOp<"add_rn">;
def AddRzOp : AscendDPX_BinaryMathOp<"add_rz">;
def AddRdOp : AscendDPX_BinaryMathOp<"add_rd">;
def AddRuOp : AscendDPX_BinaryMathOp<"add_ru">;
def MulRnOp : AscendDPX_BinaryMathOp<"mul_rn">;
def MulRzOp : AscendDPX_BinaryMathOp<"mul_rz">;
def MulRdOp : AscendDPX_BinaryMathOp<"mul_rd">;
def MulRuOp : AscendDPX_BinaryMathOp<"mul_ru">;
def FastPowfOp : AscendDPX_BinaryMathOp<"fast_powf">;
def JnOp : AscendDPX_BinaryMathOp<"jn">;
def RhypotOp : AscendDPX_BinaryMathOp<"rhypot">;
def SubRdOp : AscendDPX_BinaryMathOp<"sub_rd">;
def SubRnOp : AscendDPX_BinaryMathOp<"sub_rn">;
def SubRuOp : AscendDPX_BinaryMathOp<"sub_ru">;
def SubRzOp : AscendDPX_BinaryMathOp<"sub_rz">;
def YnOp : AscendDPX_BinaryMathOp<"yn">;

class AscendDPX_TernaryMathOp<string mnemonic>
    : AscendDPX_Op<mnemonic> {
  let arguments = (ins LLVM_Type:$in1, LLVM_Type:$in2, LLVM_Type:$in3);
  let results = (outs LLVM_Type:$res);
  let assemblyFormat = "$in1 `,` $in2 `,` $in3 attr-dict `:` functional-type(operands, results)";
}

def BytePermOp : AscendDPX_TernaryMathOp<"byte_perm">;
def FmaRnOp : AscendDPX_TernaryMathOp<"fma_rn">;
def FmaRzOp : AscendDPX_TernaryMathOp<"fma_rz">;
def FmaRdOp : AscendDPX_TernaryMathOp<"fma_rd">;
def FmaRuOp : AscendDPX_TernaryMathOp<"fma_ru">;
def FmaOp : AscendDPX_TernaryMathOp<"fma">;
def Norm3dOp : AscendDPX_TernaryMathOp<"norm3d">;
def Rnorm3dOp : AscendDPX_TernaryMathOp<"rnorm3d">;
def SadOp : AscendDPX_TernaryMathOp<"sad">;

class AscendDPX_QuaternaryMathOp<string mnemonic>
    : AscendDPX_Op<mnemonic> {
  let arguments = (ins LLVM_Type:$in1, LLVM_Type:$in2, LLVM_Type:$in3, LLVM_Type:$in4);
  let results = (outs LLVM_Type:$res);
  let assemblyFormat = "$in1 `,` $in2 `,` $in3 `,` $in4 attr-dict `:` functional-type(operands, results)";
}

def Norm4dOp : AscendDPX_QuaternaryMathOp<"norm4d">;
def Rnorm4dOp : AscendDPX_QuaternaryMathOp<"rnorm4d">;

class AscendDPX_NumberCheckOp<string mnemonic>
    : AscendDPX_Op<mnemonic> {
  let arguments = (ins LLVM_Type:$in);
  let results = (outs I1:$res);
  let assemblyFormat = "$in attr-dict `:` functional-type(operands, results)";
}

def FinitefOp : AscendDPX_NumberCheckOp<"finitef">;
def IsFiniteOp : AscendDPX_NumberCheckOp<"isfinite">;
def IsInfOp : AscendDPX_NumberCheckOp<"isinf">;
def IsNanOp : AscendDPX_NumberCheckOp<"isnan">;

//===----------------------------------------------------------------------===//
// CallScalarOp
//===----------------------------------------------------------------------===//

def CallScalarOp : AscendDPX_Op<"call_scalar",
    [DeclareOpInterfaceMethods<CallOpInterface>,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Call a scalar function";

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$call_args,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    OptionalAttr<I32Attr>:$use_shmem_offset
  );

  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $callee `(` $call_args `)` attr-dict `:` functional-type($call_args, results)
  }];
}

#endif // ASCENDDPX_OPS
