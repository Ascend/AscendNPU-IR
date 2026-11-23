//===- SyncCapabilityBinding.h - Sync request builders ------*- C++ -*-===//
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
// Header-only builders for the synchronization execution requests. The query
// library does not include this header and does not link HIVM.

#ifndef BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_SYNC_SYNCCAPABILITYBINDING_H
#define BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_SYNC_SYNCCAPABILITYBINDING_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Target/NPU/Sync/SyncCapability.h"

#include <optional>
#include <utility>

#include "bishengir/Dialect/HIVM/IR/Capability/Sync/SyncCapabilityBinding.h.inc"

#endif // BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_SYNC_SYNCCAPABILITYBINDING_H
