//===- SyncCapabilityBinding.td - Sync Op bindings --------*- tablegen -*-===//
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
//===----------------------------------------------------------------------===//
//
// Reads the synchronization execution schemas from the HIVM sync ops. The
// generator accepts exactly one binding per capability key, so get_buf and
// rls_buf bind to separate keys. The buffer sync schemas are field-free, so
// their bindings have no extractors and from(op) always succeeds.

#ifndef BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_SYNC_SYNCCAPABILITYBINDING_TD
#define BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_SYNC_SYNCCAPABILITYBINDING_TD

include "bishengir/Dialect/HIVM/IR/HIVMDoc.td"
include "bishengir/Dialect/HIVM/IR/Capability/Sync/SyncEmission.td"
include "bishengir/Dialect/HIVM/IR/Capability/Sync/SyncCapability.td"

def TryWaitCapabilityBinding : CapabilityBinding<TryWait, TryWaitOp, [
  CapabilityExtractor<AttributeBindingSource<"sync_mode">, IdentityProjection,
                      SyncMode>
]>;

def GetBufCapabilityBinding : CapabilityBinding<GetBuf, GetBufOp, []>;
def RlsBufCapabilityBinding : CapabilityBinding<RlsBuf, RlsBufOp, []>;

#endif // BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_SYNC_SYNCCAPABILITYBINDING_TD
