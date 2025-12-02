//===- HIVMOps.td - HIVM dialect operation definitions -----*- tablegen -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for HIVM dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMOPS_TD
#define BISHENGIR_DIALECT_HIVM_IR_HIVMOPS_TD

include "bishengir/Dialect/HIVM/IR/HIVMBase.td"
include "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"
include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.td"
include "bishengir/Dialect/HIVM/IR/HIVMTraits.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Dialect/LLVMIR/LLVMTypes.td"

//===----------------------------------------------------------------------===//
// HIVM Dialect operations.
//===----------------------------------------------------------------------===//

def GetBlockIdxOp : HIVM_Op<"get_block_idx", [Pure, CubeVectorCoreTypeTrait]> {
  let summary = "Get block idx of the current device thread used for parallelization.";
  let description = [{
    This op gets the block idx of the current device thread.
    This op will be lowered to `GetBlockIdxInstrOp`.
  }];
  let results = (outs I64:$result);
  let assemblyFormat = "attr-dict `->` type($result)";
}

def GetBlockNumOp : HIVM_Op<"get_block_num", [Pure, CubeVectorCoreTypeTrait]> {
  let summary = "Get block number of the current device thread used for parallelization.";
  let description = [{
    This op gets the block number of the current device thread.
    This op will be lowered to `GetBlockNumInstrOp`.
  }];
  let results = (outs I64:$result);
  let assemblyFormat = "attr-dict `->` type($result)";
}

def GetSubBlockIdxOp : HIVM_Op<"get_sub_block_idx", [Pure, CubeVectorCoreTypeTrait]> {
  let summary = "Get sub block idx of the current device thread used for parallelization.";
  let description = [{
    This op gets the sub block idx of the current device thread.
    This op will be lowered to GetSubBlockIdxInstrOp.
  }];
  let results = (outs I64:$result);
  let assemblyFormat = "attr-dict `->` type($result)";
}

def GetSubBlockNumOp : HIVM_Op<"get_sub_block_num", [Pure, CubeVectorCoreTypeTrait]> {
  let summary = "Get sub block number of the current device thread used for parallelization.";
  let description = [{
    This op gets the sub block number of the current device thread.
    This op will be lowered to GetSubBlockNumInstrOp.
  }];
  let results = (outs I64:$result);
  let assemblyFormat = "attr-dict `->` type($result)";
}

def GetSysCntOp : HIVM_Op<"get_sys_cnt", [Pure, CubeVectorCoreTypeTrait]> {
  let summary = "Get sys cnt of the current device";
  let description = [{
    This op get the sys cnt of the current device.
    This op will be lowered to `GetSysCntInstrOp`.
  }];
  let results = (outs I64:$result);
  let assemblyFormat = "attr-dict `->` type($result)";
}

def SetMaskNormOp : HIVM_Op<"set_mask_norm"> {
  let summary = "hivm set mask norm";
  let assemblyFormat = "attr-dict";
}

def LoadScalarOp : HIVM_Op<"load_scalar"> {
  let summary = "hivm load scalar";
  let arguments = (ins LLVMPointerType:$addr);
  let results = (outs AnyTypeOf<[AnyInteger, AnyFloat]>:$result);
  let assemblyFormat = "attr-dict $addr `:` type($addr) `->` type($result)";
  let hasVerifier = 1;
}

def DCCIOp : HIVM_Op<"dcci"> {
  let summary = "hivm dcci op ";
  let description = [{
    This op cleans(writes back) and invalidates one cacheline or the entire data cache
  }];
  let arguments = (ins HIVM_DCCIModeAttr:$mode,
      HIVM_DataCacheKindAttr:$dataCacheKind, Optional<AnyMemRef>:$ptr);
  let assemblyFormat = [{
    attr-dict `(` $mode `,` $dataCacheKind (`,` $ptr^ `:` type($ptr))? `)`
  }];
}

def ConvertLayoutOp : HIVM_Op<"convert_layout",
   [Pure,
    ViewLikeOpInterface,
    SameOperandsAndResultElementType,
    DeclareOpInterfaceMethods<HIVMInferCoreTypeInterface, ["inferCoreType"]>]> {
  let summary = "HIVM layout conversion operation.";
  let description = [{
    The `convert_layout` operation converts a memref with one layout
    to another. The data is not copied or modified.
  }];

  let arguments = (ins AnyRankedOrUnrankedMemRef:$source,
                       HIVM_DataLayoutAttr:$srcLayout,
                       HIVM_DataLayoutAttr:$dstLayout);
  let results = (outs AnyRankedOrUnrankedMemRef:$result);

  let assemblyFormat = [{
    $source attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    Value getViewSource() { return getSource(); }
  }];
}

def PointerCastOp : HIVM_Op<"pointer_cast",
  [CubeVectorCoreTypeTrait, AttrSizedOperandSegments]> {
  let summary = "HIVM pointer cast op at specific i64 addr";
  let description = [{
    The specific i64 addrs are stored in `$addrs`, which is variadic.

    Constraints:
    1. The type of each address should be i64.
    2. addrs should have at least one addr.

    Examples:
    ```mlir
    %addr = arith.constant 1234 : i64
    %tmp = hivm.hir.pointer_cast(%addr) : memref<32xf32>

    %addr2 = arith.constant 1600 : i64
    %addr3 = arith.constant 3200 : i64
    %tmp2 = hivm.hir.pointer_cast(%addr, %addr2) : memref<32xf32>
    %tmp3 = hivm.hir.pointer_cast(%addr, %addr2, %addr3) : memref<32xf32>
    ```
  }];
  let arguments = (ins Variadic<I64>:$addrs,
                       Variadic<Index>:$dynamicSizes
  );
  let results = (outs Res<AnyMemRef, "", [MemAlloc]>:$result);
  let assemblyFormat = [{
    `(`$addrs `)` (`[` $dynamicSizes^`]`)? attr-dict `:` type($result)
  }];
  let builders= [
    OpBuilder<(ins "Type":$result, "Value":$addr)>,
    OpBuilder<(ins "Type":$result, "ValueRange":$addrs)>,
    OpBuilder<(ins "Type":$result, "Value":$addr, "ValueRange":$dynamicSizes)>,
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$addr, "ValueRange":$dynamicSizes)>
  ];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    TypedValue<IntegerType> getSingleAddr();
  }];
}

def BitcastOp : HIVM_Op<"bitcast",
  [AlwaysSpeculatable, Elementwise, SameOperandsAndResultShape,
   NoMemoryEffect, Pure]> {
  let summary = "Reinterprets the bits of a shaped value without changing data";
  let description = [{
    The `bitcast` operation converts a tensor/memref from one element type to another
    while preserving the underlying bit representation. The operation requires:

    1. Same shape for input and output (2x3 != 3x2)
    2. Same total bit-width (element_bitwidth * num_elements)
    3. Same memory layout/strides (for memrefs)
  }];
  let arguments = (ins AnyType:$src);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
      $src `:` type($src) `->` type($result) attr-dict
  }];
}

def SetFFTSBaseAddrOp : HIVM_Op<"set_ffts_base_addr", [CubeVectorCoreTypeTrait]> {
  let summary = "set base addr for ffts sync machenism.";
  let arguments = (ins I64:$ffts_base_addr);
  let assemblyFormat = [{
    attr-dict $ffts_base_addr
  }];
}

//===----------------------------------------------------------------------===//
// HIVM Debug op definitions.
//===----------------------------------------------------------------------===//

def DebugOp :
  HIVM_Op<"debug", 
          [MemoryEffects<[MemRead, MemWrite]>,
           DeclareOpInterfaceMethods<HIVMInferCoreTypeInterface, ["inferCoreType"]>]>,
  Arguments<(ins StrAttr:$debugtype,
                 StrAttr:$prefix,
                 BoolAttr:$hex,
                 AnyTypeOf<[AnyInteger, AnyFloat, TensorOrMemref]>:$arg,
                 OptionalAttr<HIVM_TCoreTypeAttr>:$tcoretype)> {
  let summary = "Device-side debugging";
  let assemblyFormat = [{
    attr-dict $arg `:` type($arg)
  }];
  let extraClassDeclaration = [{
    static StringRef getOpName() { return "debug"; }
  }];
}

def InitDebugOp : HIVM_Op<"init_debug", [CubeVectorCoreTypeTrait]> {
  let summary = "Init func for device-side debugging";
  let assemblyFormat = [{
    attr-dict
  }];
}

def FinishDebugOp : HIVM_Op<"finish_debug", [CubeVectorCoreTypeTrait]> {
  let summary = "Finish func for device-side debugging";
  let assemblyFormat = [{
    attr-dict
  }];
}

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMOPS_TD
