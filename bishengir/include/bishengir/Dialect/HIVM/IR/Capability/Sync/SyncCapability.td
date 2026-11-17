//===- SyncCapability.td - Synchronization schema ---------*- tablegen -*-===//
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
// Synchronization execution-capability vocabulary. Architecture support rows
// live in Target/NPU/Sync/SyncCapabilityProfile.td. This file does not
// include HACC. The generic capability base does not include this file.

#ifndef BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_SYNC_SYNCCAPABILITY_TD
#define BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_SYNC_SYNCCAPABILITY_TD

include "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"
include "bishengir/Target/Capability/CapabilityBase.td"

// The synchronization counter a try_wait query reads. The parameter domain is
// the TryWait ODS enum, which intentionally exposes only intra_block
// (hardware mode 1) today. A mode added to the enum later is not Native on
// any architecture until a support row confirms it.
def SyncMode : EnumAttrCapabilityParameter<HIVM_TryWaitSyncModeAttr>;

// The flag id selects the counter instance; it is not a capability field.
def TryWaitSchema : RequestSchema<[], [SyncMode]>;
def TryWait : CapabilityKey<TryWaitSchema> {
  let DocNote = "Only the intra_block counter query has support rows. "
                "The flag id is a counter selector, not a capability field.";
}

// Buffer token acquire/release support does not depend on the pipe, the
// blocking mode, or the buffer id, so these schemas are field-free.
def GetBufSchema : RequestSchema<[], []>;
def GetBuf : CapabilityKey<GetBufSchema> {
  let DocNote = "Buffer token acquire. Pipe, blocking mode, and buffer id "
                "are not capability fields.";
}

def RlsBufSchema : RequestSchema<[], []>;
def RlsBuf : CapabilityKey<RlsBufSchema> {
  let DocNote = "Buffer token release. Pipe, blocking mode, and buffer id "
                "are not capability fields.";
}

#endif // BISHENGIR_DIALECT_HIVM_IR_CAPABILITY_SYNC_SYNCCAPABILITY_TD
