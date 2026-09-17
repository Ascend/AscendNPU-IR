//===- FixpipePreQuantContract.td - pre-quant contract -----*- tablegen -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#ifndef BISHENGIR_DIALECT_HIVM_IR_CONTRACTS_FIXPIPEPREQUANTCONTRACT_TD
#define BISHENGIR_DIALECT_HIVM_IR_CONTRACTS_FIXPIPEPREQUANTCONTRACT_TD

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

// Exact src/dst element-type signature of one Fixpipe pre-quant mode. The
// pair is kept whole; splitting it into two independent type sets would form
// an unintended Cartesian product.
class ElemTypeSignature<Type srcType, Type dstType> {
  Type SrcType = srcType;
  Type DstType = dstType;
}

def Sig_F32_F32 : ElemTypeSignature<F32, F32>;
def Sig_I32_I32 : ElemTypeSignature<I32, I32>;
def Sig_F32_F16 : ElemTypeSignature<F32, F16>;
def Sig_I32_I8 : ElemTypeSignature<I32, I8>;
def Sig_F32_BF16 : ElemTypeSignature<F32, BF16>;

// Typed Fixpipe pre-quant enum case: the case record itself is the single
// source of truth for both the mode identity and its target-independent
// src/dst element-type contract. The enum/attr generation keeps consuming the
// plain I32EnumAttrCase fields; the schema generator consumes
// TypeSignatures.
class HIVM_FixpipePreQuantCase<
    string symbol, int value, list<ElemTypeSignature> signatures>
    : I32EnumAttrCase<symbol, value> {
  list<ElemTypeSignature> TypeSignatures = signatures;
}

#endif // BISHENGIR_DIALECT_HIVM_IR_CONTRACTS_FIXPIPEPREQUANTCONTRACT_TD
