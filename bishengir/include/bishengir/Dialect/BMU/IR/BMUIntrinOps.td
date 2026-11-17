//===- BMUIntrinOps.td - BMU intrinsic wrapper operations ----*- tablegen -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//===----------------------------------------------------------------------===//
//
// Low-level BMU intrinsic wrapper operations. These are produced by the
// ConvertBMUToLLVM pass and translated to LLVM IR by the BMU LLVM
// translation interface. The wrappers map to llvm.hivm intrinsics declared
// under the A6 feature macro BSPUB_DAVINCI_BISHENGIR_A6, so building them
// requires an A6-enabled LLVM. Each buffer has its own set of 4 alloc/free
// wrappers (28 total) plus 2 SPR wrappers (14 total).
//
// Parameter order matches BiSheng-CCE HiIPUGenIntrinsics.td:
//   ALLOC: (size, seg, pipe) — ArgNames = "size,seg,pipe"
//   FREE:  (addr, size, pipe) — ArgNames = "src,size,pipe" (no seg)
// All integer types are i64 (BiSheng-CCE maps all integers to llvm_i64_ty).
// SPR intrinsic names use uppercase suffixes: set_BMU_SEGM_UB, get_BMU_SEGM_UB.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_BMU_IR_BMUINTRINOPS_TD
#define BISHENGIR_DIALECT_BMU_IR_BMUINTRINOPS_TD

include "bishengir/Dialect/BMU/IR/BMUBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

//===----------------------------------------------------------------------===//
// BMU intrinsic wrapper base class.
//===----------------------------------------------------------------------===//

class BMU_IntrOp<string mnemonic, list<int> overloadOperands = [],
                 list<Trait> traits = [], int numResults = 1,
                 list<int> immArgPositions = [],
                 list<string> immArgAttrNames = []>
    : LLVM_IntrOpBase<
          /*Dialect dialect=*/BMU_Dialect,
          /*string opName=*/"intr."#mnemonic,
          /*string enumName=*/"HIVMIntrinsics::"#!subst(".", "_", mnemonic),
          /*list<int> overloadedResults=*/[],
          /*list<int> overloadedOperands=*/overloadOperands,
          /*list<Trait> traits=*/traits,
          /*int numResults=*/numResults,
          /*requiresAccessGroup=*/0,
          /*requiresAliasAnalysis=*/0,
          /*requiresFastmath=*/0, immArgPositions, immArgAttrNames>;

//===----------------------------------------------------------------------===//
// ALLOC REG wrappers: (i64 size, i64 seg, i64 pipe) -> ptr<AS>
// immArgPositions: seg=1, pipe=2 (size is SSA operand at position 0)
//===----------------------------------------------------------------------===//

class BMU_AllocRegIntrOp<string mnemonic>
    : BMU_IntrOp<
          /*string opName=*/mnemonic,
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/1,
          /*immArgPositions=*/[1, 2],
          /*immArgAttrNames=*/["seg", "pipe"]> {
  let arguments = (ins I64:$size, I64Attr:$seg, I64Attr:$pipe);
}

def UBufAllocRegIntrOp    : BMU_AllocRegIntrOp<"hivm.__ubuf_alloc">;
def CBufAllocRegIntrOp    : BMU_AllocRegIntrOp<"hivm.__cbuf_alloc">;
def CAAllocRegIntrOp      : BMU_AllocRegIntrOp<"hivm.__ca_alloc">;
def CBAllocRegIntrOp      : BMU_AllocRegIntrOp<"hivm.__cb_alloc">;
def CCAllocRegIntrOp      : BMU_AllocRegIntrOp<"hivm.__cc_alloc">;
def FBufAllocRegIntrOp    : BMU_AllocRegIntrOp<"hivm.__fbuf_alloc">;
def BTAllocRegIntrOp      : BMU_AllocRegIntrOp<"hivm.__bt_alloc">;

//===----------------------------------------------------------------------===//
// ALLOC IMM wrappers: () -> ptr<AS>, all three args are immediates.
// immArgPositions: size=0, seg=1, pipe=2
//===----------------------------------------------------------------------===//

class BMU_AllocImmIntrOp<string mnemonic>
    : BMU_IntrOp<
          /*string opName=*/mnemonic,
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/1,
          /*immArgPositions=*/[0, 1, 2],
          /*immArgAttrNames=*/["size", "seg", "pipe"]> {
  let arguments = (ins I64Attr:$size, I64Attr:$seg, I64Attr:$pipe);
}

def UBufAllocImmIntrOp    : BMU_AllocImmIntrOp<"hivm.__ubuf_alloci">;
def CBufAllocImmIntrOp    : BMU_AllocImmIntrOp<"hivm.__cbuf_alloci">;
def CAAllocImmIntrOp      : BMU_AllocImmIntrOp<"hivm.__ca_alloci">;
def CBAllocImmIntrOp      : BMU_AllocImmIntrOp<"hivm.__cb_alloci">;
def CCAllocImmIntrOp      : BMU_AllocImmIntrOp<"hivm.__cc_alloci">;
def FBufAllocImmIntrOp    : BMU_AllocImmIntrOp<"hivm.__fbuf_alloci">;
def BTAllocImmIntrOp      : BMU_AllocImmIntrOp<"hivm.__bt_alloci">;

//===----------------------------------------------------------------------===//
// FREE REG wrappers: (ptr<AS> addr, i64 size, i64 pipe) -> void
// No seg parameter — BiSheng-CCE BUF_FREE ArgNames = "src,size,pipe".
// immArgPositions: pipe=2 (addr=0, size=1 are SSA operands)
// BT is the exception: llvm.hivm.__bt_free takes an i64 address.
//===----------------------------------------------------------------------===//

class BMU_FreeRegIntrOp<string mnemonic>
    : BMU_IntrOp<
          /*string opName=*/mnemonic,
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/0,
          /*immArgPositions=*/[2],
          /*immArgAttrNames=*/["pipe"]> {
  let arguments = (ins AnyTypeOf<[LLVM_AnyPointer, I64]>:$addr, I64:$size, I64Attr:$pipe);
}

def UBufFreeRegIntrOp    : BMU_FreeRegIntrOp<"hivm.__ubuf_free">;
def CBufFreeRegIntrOp    : BMU_FreeRegIntrOp<"hivm.__cbuf_free">;
def CAFreeRegIntrOp      : BMU_FreeRegIntrOp<"hivm.__ca_free">;
def CBFreeRegIntrOp      : BMU_FreeRegIntrOp<"hivm.__cb_free">;
def CCFreeRegIntrOp      : BMU_FreeRegIntrOp<"hivm.__cc_free">;
def FBufFreeRegIntrOp    : BMU_FreeRegIntrOp<"hivm.__fbuf_free">;
def BTFreeRegIntrOp      : BMU_FreeRegIntrOp<"hivm.__bt_free">;

//===----------------------------------------------------------------------===//
// FREE IMM wrappers: (ptr<AS> addr, i64 size_imm, i64 pipe) -> void
// No seg parameter. immArgPositions: size=1, pipe=2 (addr=0 is SSA operand)
// BT is the exception: llvm.hivm.__bt_freei takes an i64 address.
//===----------------------------------------------------------------------===//

class BMU_FreeImmIntrOp<string mnemonic>
    : BMU_IntrOp<
          /*string opName=*/mnemonic,
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/0,
          /*immArgPositions=*/[1, 2],
          /*immArgAttrNames=*/["size", "pipe"]> {
  let arguments = (ins AnyTypeOf<[LLVM_AnyPointer, I64]>:$addr, I64Attr:$size, I64Attr:$pipe);
}

def UBufFreeImmIntrOp    : BMU_FreeImmIntrOp<"hivm.__ubuf_freei">;
def CBufFreeImmIntrOp    : BMU_FreeImmIntrOp<"hivm.__cbuf_freei">;
def CAFreeImmIntrOp      : BMU_FreeImmIntrOp<"hivm.__ca_freei">;
def CBFreeImmIntrOp      : BMU_FreeImmIntrOp<"hivm.__cb_freei">;
def CCFreeImmIntrOp      : BMU_FreeImmIntrOp<"hivm.__cc_freei">;
def FBufFreeImmIntrOp    : BMU_FreeImmIntrOp<"hivm.__fbuf_freei">;
def BTFreeImmIntrOp      : BMU_FreeImmIntrOp<"hivm.__bt_freei">;

//===----------------------------------------------------------------------===//
// SET_SEGM wrappers: (i64 value) -> void
// SPR names are uppercase in BiSheng-CCE: set_BMU_SEGM_UB, etc.
//===----------------------------------------------------------------------===//

class BMU_SetSegmIntrOp<string mnemonic>
    : BMU_IntrOp<
          /*string opName=*/mnemonic,
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/0> {
  let arguments = (ins I64:$value);
}

def SetSegmUbIntrOp  : BMU_SetSegmIntrOp<"hivm.set_BMU_SEGM_UB">;
def SetSegmL1IntrOp  : BMU_SetSegmIntrOp<"hivm.set_BMU_SEGM_L1">;
def SetSegmL0aIntrOp : BMU_SetSegmIntrOp<"hivm.set_BMU_SEGM_L0A">;
def SetSegmL0bIntrOp : BMU_SetSegmIntrOp<"hivm.set_BMU_SEGM_L0B">;
def SetSegmL0cIntrOp : BMU_SetSegmIntrOp<"hivm.set_BMU_SEGM_L0C">;
def SetSegmFbIntrOp  : BMU_SetSegmIntrOp<"hivm.set_BMU_SEGM_FB">;
def SetSegmBtIntrOp  : BMU_SetSegmIntrOp<"hivm.set_BMU_SEGM_BT">;

//===----------------------------------------------------------------------===//
// GET_SEGM wrappers: () -> i64
// SPR names are uppercase in BiSheng-CCE: get_BMU_SEGM_UB, etc.
//===----------------------------------------------------------------------===//

class BMU_GetSegmIntrOp<string mnemonic>
    : BMU_IntrOp<
          /*string opName=*/mnemonic,
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/1> {
}

def GetSegmUbIntrOp  : BMU_GetSegmIntrOp<"hivm.get_BMU_SEGM_UB">;
def GetSegmL1IntrOp  : BMU_GetSegmIntrOp<"hivm.get_BMU_SEGM_L1">;
def GetSegmL0aIntrOp : BMU_GetSegmIntrOp<"hivm.get_BMU_SEGM_L0A">;
def GetSegmL0bIntrOp : BMU_GetSegmIntrOp<"hivm.get_BMU_SEGM_L0B">;
def GetSegmL0cIntrOp : BMU_GetSegmIntrOp<"hivm.get_BMU_SEGM_L0C">;
def GetSegmFbIntrOp  : BMU_GetSegmIntrOp<"hivm.get_BMU_SEGM_FB">;
def GetSegmBtIntrOp  : BMU_GetSegmIntrOp<"hivm.get_BMU_SEGM_BT">;

#endif // BISHENGIR_DIALECT_BMU_IR_BMUINTRINOPS_TD
