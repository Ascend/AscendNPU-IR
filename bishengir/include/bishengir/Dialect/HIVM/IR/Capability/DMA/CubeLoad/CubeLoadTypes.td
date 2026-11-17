//===- CubeLoadTypes.td - Same-type DMA signatures -----------*- tablegen
//-*-===//
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
//
// Element-type pairs shared by cube DMA schemas. A pair stays whole: f16 to
// f16 does not admit f16 to f32.

#ifndef BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_CUBELOAD_CUBELOADTYPES_TD
#define BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_CUBELOAD_CUBELOADTYPES_TD

include "bishengir/Dialect/HIVM/IR/Capability/DMA/Common.td"

class CubeLoadSameType<Type element>
    : TypeSignature<[SrcType, DstType], [element, element]>;

def CubeLoadF16 : CubeLoadSameType<F16>;
def CubeLoadBF16 : CubeLoadSameType<BF16>;
def CubeLoadF32 : CubeLoadSameType<F32>;
def CubeLoadI8 : CubeLoadSameType<I8>;
def CubeLoadUI8 : CubeLoadSameType<UI8>;
def CubeLoadI16 : CubeLoadSameType<I16>;
def CubeLoadUI16 : CubeLoadSameType<UI16>;
def CubeLoadI32 : CubeLoadSameType<I32>;
def CubeLoadUI32 : CubeLoadSameType<UI32>;
def CubeLoadI64 : CubeLoadSameType<I64>;
def CubeLoadUI64 : CubeLoadSameType<UI64>;
def CubeLoadF8E4M3FN : CubeLoadSameType<F8E4M3FN>;
def CubeLoadF8E5M2 : CubeLoadSameType<F8E5M2>;

defvar cubeLoadCommonTypes = [CubeLoadF16, CubeLoadBF16, CubeLoadF32,
                              CubeLoadI8, CubeLoadUI8, CubeLoadI16,
                              CubeLoadUI16, CubeLoadI32, CubeLoadUI32];
defvar cubeLoadWideTypes =
    !listconcat(cubeLoadCommonTypes, [CubeLoadI64, CubeLoadUI64]);
defvar cubeLoadFloatTypes = [CubeLoadF16, CubeLoadBF16, CubeLoadF32];

#endif // BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_CUBELOAD_CUBELOADTYPES_TD
