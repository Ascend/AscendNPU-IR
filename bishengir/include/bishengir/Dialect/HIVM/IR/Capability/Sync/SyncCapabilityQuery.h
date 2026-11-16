//===- SyncCapabilityQuery.h - Inline sync capability helpers -*- C++ -*-===//
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
// Shared helpers for sync-op consumers that query the production capability
// catalogue. Missing #hacc.target is not Ascend910B. Only Native is treated
// as an affirmative lowering permit.

#ifndef BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_SYNC_SYNCCAPABILITYQUERY_H
#define BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_SYNC_SYNCCAPABILITYQUERY_H

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/Capability/Sync/SyncCapabilityBinding.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

namespace mlir {
namespace hivm {

inline std::optional<bishengir::npu::NPUArchitectureKind>
syncArchitecture(Operation *anchor) {
  auto module = anchor->getParentOfType<ModuleOp>();
  if (!module)
    return std::nullopt;
  auto device = hacc::utils::getTargetDevice(module);
  if (!device)
    return std::nullopt;
  return hacc::getArchitecture(*device);
}

inline bool
isNativeSyncRequest(Operation *anchor,
                    bishengir::npu::sync::CapabilityRequestStorage storage) {
  auto architecture = syncArchitecture(anchor);
  if (!architecture)
    return false;
  return bishengir::npu::sync::queryNativeSupport(*architecture,
                                                  storage.getRequest()) ==
         bishengir::npu::sync::NativeSupport::Native;
}

/// Query a try_wait candidate that does not exist as IR yet.
inline bool isNativeTryWaitCreate(Operation *anchor, TryWaitSyncMode syncMode) {
  return isNativeSyncRequest(
      anchor,
      bishengir::npu::sync::TryWaitCapabilityRequestBuilder::create(syncMode)
          .build());
}

/// Query a get_buf candidate. The schema is field-free; the result only
/// depends on the enclosing module target.
inline bool isNativeGetBufCreate(Operation *anchor) {
  return isNativeSyncRequest(
      anchor,
      bishengir::npu::sync::GetBufCapabilityRequestBuilder::create().build());
}

/// Query a rls_buf candidate. The schema is field-free; the result only
/// depends on the enclosing module target.
inline bool isNativeRlsBufCreate(Operation *anchor) {
  return isNativeSyncRequest(
      anchor,
      bishengir::npu::sync::RlsBufCapabilityRequestBuilder::create().build());
}

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_SYNC_SYNCCAPABILITYQUERY_H
