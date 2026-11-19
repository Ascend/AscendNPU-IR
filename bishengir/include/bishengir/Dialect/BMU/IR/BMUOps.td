//===- BMUOps.td - BMU dialect semantic operations -----------*- tablegen -*-===//
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
// Semantic operations for the BMU dialect (an A6 feature). These are the
// user-facing operations that are lowered to bmu.intr.* intrinsic wrappers
// by the ConvertBMUToLLVM pass.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_BMU_IR_BMUOPS_TD
#define BISHENGIR_DIALECT_BMU_IR_BMUOPS_TD

include "bishengir/Dialect/BMU/IR/BMUAttrs.td"

//===----------------------------------------------------------------------===//
// ALLOC / FREE operations (four variants).
// Parameter order matches BiSheng-CCE: ALLOC=(size,seg,pipe), FREE=(addr,size,pipe).
// FREE does not carry seg — BiSheng-CCE BUF_FREE Inst{21-20} is fixed 0b00.
// All integer types use i64 to match BiSheng-CCE CCEIntrinsicEmitter
// (all integers → llvm_i64_ty).
//
// `pipe` names the pipeline that issues the allocation. When the free space of
// `buffer` is smaller than the demanded allocation size, the allocation stalls
// the execution of that pipeline until the free space is large enough, so the
// operand constrains the schedule and is not a pure annotation.
//===----------------------------------------------------------------------===//

def BMUAllocRegOp : BMU_Op<"alloc_reg"> {
  let summary = "BMU buffer allocation with register size";
  let description = [{
    Allocate `size` slices of on-chip buffer `buffer` on `pipe` with segment
    `seg`. The slice granularity is `buf_size/64`, so the slice size differs per
    buffer (UB = 6 KB, L1 = 8 KB, L0C = 4 KB). The size is a runtime i64 SSA
    value (REG variant).
  }];
  let arguments = (ins
    I64:$size,
    BMU_BufferAttr:$buffer,
    BMU_PipeAttr:$pipe,
    I64Attr:$seg
  );
  let results = (outs LLVM_AnyPointer:$addr);
  let assemblyFormat = "`(` $size `)` attr-dict `:` type($size) `->` type($addr)";
}

def BMUAllocImmOp : BMU_Op<"alloc_imm"> {
  let summary = "BMU buffer allocation with immediate size";
  let description = [{
    Allocate `size` slices of on-chip buffer `buffer` on `pipe` with segment
    `seg`. The slice granularity is `buf_size/64`, so the slice size differs per
    buffer (UB = 6 KB, L1 = 8 KB, L0C = 4 KB). The size is a compile-time i64
    attribute (IMM variant). The 7-bit instruction field bounds the encodable
    size, but that bound is not checked at the IR level.
  }];
  let arguments = (ins
    I64Attr:$size,
    BMU_BufferAttr:$buffer,
    BMU_PipeAttr:$pipe,
    I64Attr:$seg
  );
  let results = (outs LLVM_AnyPointer:$addr);
  let assemblyFormat = "attr-dict `:` `(` `)` `->` type($addr)";
}

def BMUFreeRegOp : BMU_Op<"free_reg"> {
  let summary = "BMU buffer free with register size";
  let description = [{
    Free the buffer at `addr` with runtime `size` (REG variant).
    FREE does not carry seg — matches BiSheng-CCE BUF_FREE encoding.
  }];
  let arguments = (ins
    LLVM_AnyPointer:$addr,
    I64:$size,
    BMU_BufferAttr:$buffer,
    BMU_PipeAttr:$pipe
  );
  let assemblyFormat = "`(` $addr `,` $size `)` attr-dict `:` type($addr) `,` type($size)";
}

def BMUFreeImmOp : BMU_Op<"free_imm"> {
  let summary = "BMU buffer free with immediate size";
  let description = [{
    Free the buffer at `addr` with compile-time `size` (IMM variant); `size` has
    the same slice granularity as `bmu.alloc_imm`. The 7-bit instruction field
    bounds the encodable size, but that bound is not checked at the IR level.
    FREE does not carry seg — matches BiSheng-CCE BUF_FREE encoding.
  }];
  let arguments = (ins
    LLVM_AnyPointer:$addr,
    I64Attr:$size,
    BMU_BufferAttr:$buffer,
    BMU_PipeAttr:$pipe
  );
  let assemblyFormat = "`(` $addr `)` attr-dict `:` type($addr)";
}

//===----------------------------------------------------------------------===//
// Segment register operations (SPR access).
// SPR value type is i64 — BiSheng-CCE WRITE_SPR_8bits proto "vULi" (uint64_t),
// READ_SPR proto "Li" (int64). All integers map to llvm_i64_ty.
//===----------------------------------------------------------------------===//

def BMUSetSegmOp : BMU_Op<"set_segm"> {
  let summary = "Set BMU segment register";
  let description = [{
    Set the segment register for `buffer` to the i64 `value`.
  }];
  let arguments = (ins
    I64:$value,
    BMU_BufferAttr:$buffer
  );
  let assemblyFormat = "`(` $value `)` attr-dict `:` type($value)";
}

def BMUGetSegmOp : BMU_Op<"get_segm"> {
  let summary = "Get BMU segment register";
  let description = [{
    Read the segment register for `buffer`, returning an i64 value.
  }];
  let arguments = (ins
    BMU_BufferAttr:$buffer
  );
  let results = (outs I64:$value);
  let assemblyFormat = "attr-dict `:` `(` `)` `->` type($value)";
}

#endif // BISHENGIR_DIALECT_BMU_IR_BMUOPS_TD
