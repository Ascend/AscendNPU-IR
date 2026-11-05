//===- CubeLoadCapabilityBinding.h - CubeLoad request builders ---*- C++
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
// Header-only builders for the CubeLoad execution requests. The query library
// does not include this header and does not link HIVM.

#ifndef BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_CUBELOAD_CUBELOADCAPABILITYBINDING_H
#define BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_CUBELOAD_CUBELOADCAPABILITYBINDING_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Target/NPU/DMA/CubeLoad/CubeLoadCapability.h"

#include <optional>

#include "bishengir/Dialect/HIVM/IR/Capability/DMA/CubeLoad/CubeLoadCapabilityBinding.h.inc"

#endif // BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_DMA_CUBELOAD_CUBELOADCAPABILITYBINDING_H
