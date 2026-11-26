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
//
// The handle is a ranked memref in the buffer's HIVM address space, so an
// allocation can be consumed directly by memref/HIVM operations. BT is the
// exception: it is addressed by an i64 value (proto "ULi"), not through an
// address space. The verifier enforces both directions of this mapping.
//===----------------------------------------------------------------------===//

def BMUAllocRegOp : BMU_Op<"alloc_reg"> {
  let summary = "BMU buffer allocation with register size";
  let description = [{
    Allocate `size` slices of on-chip buffer `buffer` on `pipe` with segment
    `seg`, and return a handle to the allocation.

    The slice size is the buffer size divided by the ring's slice count, read
    from the module's device spec: `(UbSize + 8 KiB compiler reserve) / 48` for
    UB, `L1Size / 64` for L1 and `L0cSize / 64` for L0C. It is derived from the
    target rather than hardcoded, so a corrected spec moves it with it. The size
    is a runtime i64 SSA value (REG variant).

    The handle is a ranked memref whose memory space is the HIVM address space of
    `buffer` — `#hivm.address_space<ub>` for UB, `#hivm.address_space<cbuf>` for
    L1, and so on — so the result can be used by memref and HIVM operations
    directly. BT is the exception: it is addressed by an i64 scalar (proto
    "ULi") and has no address space.
  }];
  let arguments = (ins
    I64:$size,
    BMU_BufferAttr:$buffer,
    BMU_PipeAttr:$pipe,
    I64Attr:$seg
  );
  let results = (outs BMU_BufferHandle:$addr);
  let assemblyFormat = "`(` $size `)` attr-dict `:` type($size) `->` type($addr)";
}

def BMUAllocImmOp : BMU_Op<"alloc_imm"> {
  let summary = "BMU buffer allocation with immediate size";
  let description = [{
    Allocate `size` slices of on-chip buffer `buffer` on `pipe` with segment
    `seg`, and return a handle to the allocation.

    The slice size is the same as for `bmu.alloc_reg`: the target buffer size
    divided by the ring's slice count, read from the module's device spec. The
    size is a compile-time i64 attribute (IMM variant). The 7-bit instruction
    field bounds the encodable size, but that bound is not checked at the IR
    level.

    The handle is the same as for `bmu.alloc_reg`: a ranked memref in the HIVM
    address space of `buffer`, or an i64 scalar for BT.
  }];
  let arguments = (ins
    I64Attr:$size,
    BMU_BufferAttr:$buffer,
    BMU_PipeAttr:$pipe,
    I64Attr:$seg
  );
  let results = (outs BMU_BufferHandle:$addr);
  let assemblyFormat = "attr-dict `:` `(` `)` `->` type($addr)";
}

def BMUFreeRegOp : BMU_Op<"free_reg"> {
  let summary = "BMU buffer free with register size";
  let description = [{
    Free the allocation handle `addr` with runtime `size` (REG variant). `addr`
    has the same form as the result of `bmu.alloc_reg`: a ranked memref in the
    buffer's HIVM address space, or an i64 scalar for BT.
    FREE does not carry seg — matches BiSheng-CCE BUF_FREE encoding.
  }];
  let arguments = (ins
    BMU_BufferHandle:$addr,
    I64:$size,
    BMU_BufferAttr:$buffer,
    BMU_PipeAttr:$pipe
  );
  let assemblyFormat = "`(` $addr `,` $size `)` attr-dict `:` type($addr) `,` type($size)";
}

def BMUFreeImmOp : BMU_Op<"free_imm"> {
  let summary = "BMU buffer free with immediate size";
  let description = [{
    Free the allocation handle `addr` with compile-time `size` (IMM variant);
    `size` has the same slice granularity as `bmu.alloc_imm`. The 7-bit
    instruction field bounds the encodable size, but that bound is not checked at
    the IR level.

    `addr` has the same form as the result of `bmu.alloc_imm`: a ranked memref in
    the buffer's HIVM address space, or an i64 scalar for BT.
    FREE does not carry seg — matches BiSheng-CCE BUF_FREE encoding.
  }];
  let arguments = (ins
    BMU_BufferHandle:$addr,
    I64Attr:$size,
    BMU_BufferAttr:$buffer,
    BMU_PipeAttr:$pipe
  );
  let assemblyFormat = "`(` $addr `)` attr-dict `:` type($addr)";
}

//===----------------------------------------------------------------------===//
// design: bmu#s.3.3.6
//
// REGION operations (design 3.3.6).
//
// A region allocation reserves `size` contiguous slices and returns the raw i64
// base address of the region. Its members are addressed as `base + constant
// offset` and materialized with `hivm.hir.pointer_cast`, so the region needs one
// alloc/free pair instead of one per member. The base is an i64 for every buffer
// because the ring hands out a byte address, not a typed view: unlike
// alloc_reg/alloc_imm the result is not a memref.
//
// The pair lowers to the existing slice alloc/free intrinsics plus an integer
// cast at the LLVM boundary, so no new hardware intrinsic is required.
//===----------------------------------------------------------------------===//

def BMUAllocRegionImmOp : BMU_Op<"alloc_region_imm"> {
  let summary = "BMU region allocation with immediate size";
  let description = [{
    Allocate `size` contiguous slices of `buffer` on `pipe` with segment `seg`
    and return their base address as an i64.

    The allocation is the region of design 3.3.6: one reservation serves a group
    of members, each of which is addressed as `base + offset` and materialized
    with `hivm.hir.pointer_cast`. The slice size is the same as for
    `bmu.alloc_imm`: the target buffer size divided by the ring's slice count.

    BT is rejected: a region is addressed by the ring, and BT allocations have no
    slice size in the dialect.
  }];
  let arguments = (ins
    I64Attr:$size,
    BMU_BufferAttr:$buffer,
    BMU_PipeAttr:$pipe,
    I64Attr:$seg
  );
  let results = (outs I64:$base);
  let assemblyFormat = "attr-dict `:` `(` `)` `->` type($base)";
}

def BMUFreeRegionImmOp : BMU_Op<"free_region_imm"> {
  let summary = "BMU region free with immediate size";
  let description = [{
    Free the region whose base address is `base`, with compile-time `size`
    slices. The pair of `bmu.alloc_region_imm`; FREE does not carry seg, matching
    BiSheng-CCE BUF_FREE encoding.
  }];
  let arguments = (ins
    I64:$base,
    I64Attr:$size,
    BMU_BufferAttr:$buffer,
    BMU_PipeAttr:$pipe
  );
  let assemblyFormat = "`(` $base `)` attr-dict `:` type($base)";
}

//===----------------------------------------------------------------------===//
// REGION planning operations (design 3.3.9).
//
// While a region is planned the members are addressed by `region_addr` and the
// reservation is `region_free`; `hivm-bmu-materialize` replaces the token with
// the i64 base of `alloc_region_imm` and turns each `region_addr` into
// `base + offset` (constant addi) followed by `pointer_cast`.
//===----------------------------------------------------------------------===//

def BMURegionAllocOp : BMU_Op<"region_alloc"> {
  let summary = "BMU region allocation (planning form)";
  let description = [{
    Reserve `size` contiguous slices of `buffer` on `pipe` in segment `seg` and
    return the region token. The members of the region are the allocations
    marked with `annotation.mark {bmu.region = id}` / addressed through
    `bmu.region_addr`; the token itself is not an address.

    BT is rejected, exactly like `bmu.alloc_region_imm`.
  }];
  let arguments = (ins
    I64Attr:$size,
    BMU_BufferAttr:$buffer,
    BMU_PipeAttr:$pipe,
    I64Attr:$seg
  );
  let results = (outs BMU_RegionType:$region);
  let assemblyFormat = "attr-dict `:` `(` `)` `->` type($region)";
}

def BMURegionAddrOp : BMU_Op<"region_addr"> {
  let summary = "address of one region member during planning";
  let description = [{
    The byte offset of a member inside the region `region`. The static planner
    fills the offset (design 3.3.5); materialization turns it into
    `arith.addi(base, offset)`.
  }];
  let arguments = (ins
    BMU_RegionType:$region,
    I64Attr:$offset
  );
  let results = (outs I64:$addr);
  let assemblyFormat = "`(` $region `)` attr-dict `:` type($region) `->` type($addr)";
}

def BMURegionFreeOp : BMU_Op<"region_free"> {
  let summary = "release of a BMU region (planning form)";
  let description = [{
    Release the reservation named by `region` on `pipe`. The pipe is the last
    reader's pipe; fences that cover the other reader pipes are emitted before
    it (design 3.3.7).
  }];
  let arguments = (ins
    BMU_RegionType:$region,
    BMU_PipeAttr:$pipe
  );
  let assemblyFormat = "`(` $region `)` attr-dict `:` type($region)";
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
