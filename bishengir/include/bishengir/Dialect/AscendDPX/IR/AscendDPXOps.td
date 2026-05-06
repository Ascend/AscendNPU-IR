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
def AtanOp : AscendDPX_UnaryMathOp<"atan">;
def CeilOp : AscendDPX_UnaryMathOp<"ceil">;
def CosOp : AscendDPX_UnaryMathOp<"cos">;
def ErfOp : AscendDPX_UnaryMathOp<"erf">;
def Exp2Op : AscendDPX_UnaryMathOp<"exp2">;
def ExpOp : AscendDPX_UnaryMathOp<"exp">;
def FloatAsIntOp : AscendDPX_UnaryMathOp<"float_as_int">;
def FloorOp : AscendDPX_UnaryMathOp<"floor">;
def ILogbOp : AscendDPX_UnaryMathOp<"ilogb">;
def Log1pOp : AscendDPX_UnaryMathOp<"log1p">;
def Log2Op : AscendDPX_UnaryMathOp<"log2">;
def LogOp : AscendDPX_UnaryMathOp<"log">;
def RecipOp : AscendDPX_UnaryMathOp<"recip">;
def ReluOp : AscendDPX_UnaryMathOp<"relu">;
def RintOp : AscendDPX_UnaryMathOp<"rint">;
def RoundOp : AscendDPX_UnaryMathOp<"round">;
def RSqrtOp : AscendDPX_UnaryMathOp<"rsqrt">;
def SinOp : AscendDPX_UnaryMathOp<"sin">;
def SqrtOp : AscendDPX_UnaryMathOp<"sqrt">; 
def TanhOp : AscendDPX_UnaryMathOp<"tanh">;
def TanOp : AscendDPX_UnaryMathOp<"tan">;
def TruncOp : AscendDPX_UnaryMathOp<"trunc">;
def NearbyintOp : AscendDPX_UnaryMathOp<"nearbyint">;
def Log10Op : AscendDPX_UnaryMathOp<"log10">;
def AsinOp : AscendDPX_UnaryMathOp<"asin">;
def AcosOp : AscendDPX_UnaryMathOp<"acos">;
def SinhOp : AscendDPX_UnaryMathOp<"sinh">;
def CoshOp : AscendDPX_UnaryMathOp<"cosh">;
def AsinhOp : AscendDPX_UnaryMathOp<"asinh">;
def AcoshOp : AscendDPX_UnaryMathOp<"acosh">;
def AtanhOp : AscendDPX_UnaryMathOp<"atanh">;
def Expm1Op : AscendDPX_UnaryMathOp<"expm1">;
def CylBesselI0Op : AscendDPX_UnaryMathOp<"cyl_bessel_i0">;
def ErfinvOp : AscendDPX_UnaryMathOp<"erfinv">;
def LgammaOp : AscendDPX_UnaryMathOp<"lgamma">;
def SignbitOp : AscendDPX_UnaryMathOp<"signbit">;
def ClzOp : AscendDPX_UnaryMathOp<"clz">;
def PopcOp : AscendDPX_UnaryMathOp<"popc">;
def FfsOp : AscendDPX_UnaryMathOp<"ffs">;
def AbsOp : AscendDPX_UnaryMathOp<"abs">;
def SaturatefOp : AscendDPX_UnaryMathOp<"saturatef">;
def Exp10Op : AscendDPX_UnaryMathOp<"exp10">;
def RcpRnOp : AscendDPX_UnaryMathOp<"rcp_rn">;
def RcpRzOp : AscendDPX_UnaryMathOp<"rcp_rz">;
def RcpRdOp : AscendDPX_UnaryMathOp<"rcp_rd">;
def RcpRuOp : AscendDPX_UnaryMathOp<"rcp_ru">;
def RsqrtRnOp : AscendDPX_UnaryMathOp<"rsqrt_rn">;
def BrevOp : AscendDPX_UnaryMathOp<"brev">;

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
def SadOp : AscendDPX_TernaryMathOp<"sad">;

class AscendDPX_NumberCheckOp<string mnemonic>
    : AscendDPX_Op<mnemonic> {
  let arguments = (ins LLVM_Type:$in);
  let results = (outs I1:$res);
  let assemblyFormat = "$in attr-dict `:` functional-type(operands, results)";
}

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
