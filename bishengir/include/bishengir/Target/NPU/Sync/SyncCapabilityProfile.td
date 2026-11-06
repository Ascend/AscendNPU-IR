//===- SyncCapabilityProfile.td - Synchronization profiles -*- tablegen -*-===//
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
// TableGen entry for the Sync catalogue. Each architecture has one profile.
// The dav-c310 rows are the confirmed template-library rows: the intrinsic
// header names the dav-c310 family for try_wait, get_buf, and rls_buf.
// dav-920r1 is not yet surveyed for these sync intrinsics, so it inherits
// every row under the forward-compatibility policy: a confirmed row on one
// regbase architecture stays open on later regbase ones. dav-920r1 also
// matches the old lowering gates, which allowed try_wait on every regbase
// architecture. Memory-based architectures (dav-c220, dav-m300) have no
// RegBase sync at all: they get surveyed coverage records with no rows, so
// they answer Unknown, like the old gates, instead of no-profile Unknown.
//
// Only Native rows are listed. Architectures without a coverage for one of
// these capabilities answer Unknown, never Native. All coverages stay
// Partial until the support database is surveyed like the cube library.

#ifndef BISHENGIR_TARGET_NPU_SYNC_SYNCCAPABILITYPROFILE_TD
#define BISHENGIR_TARGET_NPU_SYNC_SYNCCAPABILITYPROFILE_TD

include "bishengir/Dialect/HIVM/IR/Capability/Sync/SyncEmission.td"
include "bishengir/Dialect/HIVM/IR/Capability/Sync/SyncCapability.td"
include "bishengir/Target/NPU/NPUCapabilityProfileBase.td"

defvar intraBlockOnly = [HIVM_TRY_WAIT_INTRA_BLOCK];

// Ascend950 (RegBase): try_wait intra_block query and buffer token
// acquire/release are native.
def C310TryWaitRow : SupportCase<TryWait, [EnumAnyOf<SyncMode, intraBlockOnly>]>;
def C310GetBufRow : SupportCase<GetBuf, []>;
def C310RlsBufRow : SupportCase<RlsBuf, []>;

// Ascend960 (dav-920r1): not surveyed for the sync intrinsics yet, so every
// confirmed row is open under the forward-compatibility policy. get_buf and
// rls_buf are field-free schemas, so the row is the full support case; there
// is no field an AnyValue constraint could relax further.
def Dav920R1TryWaitRow
    : SupportCase<TryWait, [EnumAnyOf<SyncMode, intraBlockOnly>]>;
def Dav920R1GetBufRow : SupportCase<GetBuf, []>;
def Dav920R1RlsBufRow : SupportCase<RlsBuf, []>;

def C310TryWaitCoverage
    : ArchitectureCapabilityCoverage<TryWait, [C310TryWaitRow], PartialProfile>;
def C310GetBufCoverage
    : ArchitectureCapabilityCoverage<GetBuf, [C310GetBufRow], PartialProfile>;
def C310RlsBufCoverage
    : ArchitectureCapabilityCoverage<RlsBuf, [C310RlsBufRow], PartialProfile>;
def Dav920R1TryWaitCoverage
    : ArchitectureCapabilityCoverage<TryWait, [Dav920R1TryWaitRow],
                                     PartialProfile>;
def Dav920R1GetBufCoverage
    : ArchitectureCapabilityCoverage<GetBuf, [Dav920R1GetBufRow],
                                     PartialProfile>;
def Dav920R1RlsBufCoverage
    : ArchitectureCapabilityCoverage<RlsBuf, [Dav920R1RlsBufRow],
                                     PartialProfile>;

// Memory-based architectures have no RegBase synchronization. Give them
// surveyed coverage records with no rows: they answer Unknown like the old
// gates did, and "surveyed: no" stays distinct from "no profile yet".
def C220TryWaitCoverage
    : ArchitectureCapabilityCoverage<TryWait, [], PartialProfile>;
def C220GetBufCoverage
    : ArchitectureCapabilityCoverage<GetBuf, [], PartialProfile>;
def C220RlsBufCoverage
    : ArchitectureCapabilityCoverage<RlsBuf, [], PartialProfile>;
def M300TryWaitCoverage
    : ArchitectureCapabilityCoverage<TryWait, [], PartialProfile>;
def M300GetBufCoverage
    : ArchitectureCapabilityCoverage<GetBuf, [], PartialProfile>;
def M300RlsBufCoverage
    : ArchitectureCapabilityCoverage<RlsBuf, [], PartialProfile>;

def Dav_C310SyncProfile
    : ArchitectureCapabilityProfile<
          Dav_C310, [C310TryWaitCoverage, C310GetBufCoverage,
                     C310RlsBufCoverage]>;
def Dav_920R1SyncProfile
    : ArchitectureCapabilityProfile<
          Dav_920R1, [Dav920R1TryWaitCoverage, Dav920R1GetBufCoverage,
                      Dav920R1RlsBufCoverage]>;
def Dav_C220SyncProfile
    : ArchitectureCapabilityProfile<
          Dav_C220, [C220TryWaitCoverage, C220GetBufCoverage,
                     C220RlsBufCoverage]>;
def Dav_M300SyncProfile
    : ArchitectureCapabilityProfile<
          Dav_M300, [M300TryWaitCoverage, M300GetBufCoverage,
                     M300RlsBufCoverage]>;

#endif // BISHENGIR_TARGET_NPU_SYNC_SYNCCAPABILITYPROFILE_TD
