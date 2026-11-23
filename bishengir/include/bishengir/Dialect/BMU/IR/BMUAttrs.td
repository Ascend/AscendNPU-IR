//===- BMUAttrs.td - BMU dialect attributes --------------------*- tablegen -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_BMU_IR_BMUATTRS_TD
#define BISHENGIR_DIALECT_BMU_IR_BMUATTRS_TD

include "bishengir/Dialect/BMU/IR/BMUBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Buffer enum: seven on-chip buffer types.
//===----------------------------------------------------------------------===//

def BMU_Buffer_UB  : I32EnumAttrCase<"UB",  0, "ubuf">;
def BMU_Buffer_L1  : I32EnumAttrCase<"L1",  1, "cbuf">;
def BMU_Buffer_L0A : I32EnumAttrCase<"L0A", 2, "ca">;
def BMU_Buffer_L0B : I32EnumAttrCase<"L0B", 3, "cb">;
def BMU_Buffer_L0C : I32EnumAttrCase<"L0C", 4, "cc">;
def BMU_Buffer_FB  : I32EnumAttrCase<"FB",  5, "fbuf">;
def BMU_Buffer_BT  : I32EnumAttrCase<"BT",  6, "bt">;

def BMU_BufferEnum : I32EnumAttr<
  "Buffer", "BMU buffer type", [
    BMU_Buffer_UB,
    BMU_Buffer_L1,
    BMU_Buffer_L0A,
    BMU_Buffer_L0B,
    BMU_Buffer_L0C,
    BMU_Buffer_FB,
    BMU_Buffer_BT
  ]> {
  let genSpecializedAttr = 0;
}

def BMU_BufferAttr : EnumAttr<BMU_Dialect, BMU_BufferEnum, "buffer"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Pipe enum: matches HIVM pipe_t encoding.
//===----------------------------------------------------------------------===//

def BMU_PIPE_S    : I32EnumAttrCase<"PIPE_S",    0>;
def BMU_PIPE_V    : I32EnumAttrCase<"PIPE_V",    1>;
def BMU_PIPE_M    : I32EnumAttrCase<"PIPE_M",    2>;
def BMU_PIPE_MTE1 : I32EnumAttrCase<"PIPE_MTE1", 3>;
def BMU_PIPE_MTE2 : I32EnumAttrCase<"PIPE_MTE2", 4>;
def BMU_PIPE_MTE3 : I32EnumAttrCase<"PIPE_MTE3", 5>;
def BMU_PIPE_ALL  : I32EnumAttrCase<"PIPE_ALL",  6>;
def BMU_PIPE_MTE4 : I32EnumAttrCase<"PIPE_MTE4", 7>;
def BMU_PIPE_MTE5 : I32EnumAttrCase<"PIPE_MTE5", 8>;
def BMU_PIPE_V2   : I32EnumAttrCase<"PIPE_V2",   9>;
def BMU_PIPE_FIX  : I32EnumAttrCase<"PIPE_FIX",  10>;

def BMU_PipeEnum : I32EnumAttr<
  "Pipe", "BMU pipe type", [
    BMU_PIPE_S,
    BMU_PIPE_V,
    BMU_PIPE_M,
    BMU_PIPE_MTE1,
    BMU_PIPE_MTE2,
    BMU_PIPE_MTE3,
    BMU_PIPE_ALL,
    BMU_PIPE_MTE4,
    BMU_PIPE_MTE5,
    BMU_PIPE_V2,
    BMU_PIPE_FIX
  ]> {
  let genSpecializedAttr = 0;
}

def BMU_PipeAttr : EnumAttr<BMU_Dialect, BMU_PipeEnum, "pipe"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // BISHENGIR_DIALECT_BMU_IR_BMUATTRS_TD
