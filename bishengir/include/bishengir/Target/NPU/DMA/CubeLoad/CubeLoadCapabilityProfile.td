//===- CubeLoadCapabilityProfile.td - Per-arch cube DMA ------*- tablegen
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
// TableGen entry for the CubeLoad catalogue. Each architecture has one
// profile. The support rows live with their ops.

#ifndef BISHENGIR_TARGET_NPU_DMA_CUBELOAD_CUBELOADCAPABILITYPROFILE_TD
#define BISHENGIR_TARGET_NPU_DMA_CUBELOAD_CUBELOADCAPABILITYPROFILE_TD

include "bishengir/Dialect/HIVM/IR/Capability/DMA/CubeLoad/CubeLoadEmission.td"
include "bishengir/Target/NPU/DMA/CubeLoad/Nd2NzCapabilityProfile.td"
include "bishengir/Target/NPU/DMA/CubeLoad/Nz2NdCapabilityProfile.td"
include "bishengir/Target/NPU/DMA/CubeLoad/L12UBCapabilityProfile.td"
include "bishengir/Target/NPU/DMA/CubeLoad/NCHW2NC1HWC0CapabilityProfile.td"
include "bishengir/Target/NPU/DMA/CubeLoad/NCHW2C1HWNC0CapabilityProfile.td"
include "bishengir/Target/NPU/DMA/CubeLoad/LoadScaleCapabilityProfile.td"

def Dav_C220CubeLoadProfile
    : ArchitectureCapabilityProfile<Dav_C220, [C220Nd2NzCoverage,
                                               C220Nz2NdCoverage]>;
def Dav_C310CubeLoadProfile
    : ArchitectureCapabilityProfile<
          Dav_C310, [C310Nd2NzCoverage, C310Nd2NzBatchCoverage,
                     C310L12UBCoverage, C310Nchw2Nc1hwc0Coverage,
                     C310Nchw2C1hwnc0Coverage, C310LoadScaleCoverage]>;
// dav-920r1 compiles the same RegBase cube library as C310. AnyValue still
// requires each field to be present, so a missing memory scope stays Unknown.
class Dav920R1AnyRow<CapabilityKey capability,
                     list<CapabilityConstraint> extras = []>
    : SupportCase<capability,
                  !listconcat([AnyValue<SrcType>, AnyValue<SrcSpace>,
                               AnyValue<DstType>, AnyValue<DstSpace>],
                              extras)>;

def Dav920R1AnyNd2Nz : Dav920R1AnyRow<Nd2Nz, [AnyValue<Nd2NzInitOutBuffer>]>;
def Dav920R1AnyNd2NzBatch : Dav920R1AnyRow<Nd2NzBatch>;
def Dav920R1AnyNz2Nd : Dav920R1AnyRow<Nz2Nd>;
def Dav920R1AnyL12UB : Dav920R1AnyRow<L12UB>;
def Dav920R1AnyNchw2Nc1hwc0 : Dav920R1AnyRow<Nchw2Nc1hwc0>;
def Dav920R1AnyNchw2C1hwnc0 : Dav920R1AnyRow<Nchw2C1hwnc0>;
def Dav920R1AnyLoadScale
    : Dav920R1AnyRow<LoadScale, [AnyValue<LoadScaleIsTransposed>]>;

def Dav920R1Nd2NzCoverage
    : ArchitectureCapabilityCoverage<Nd2Nz, [Dav920R1AnyNd2Nz], PartialProfile>;
def Dav920R1Nd2NzBatchCoverage
    : ArchitectureCapabilityCoverage<Nd2NzBatch, [Dav920R1AnyNd2NzBatch],
                                     PartialProfile>;
def Dav920R1Nz2NdCoverage
    : ArchitectureCapabilityCoverage<Nz2Nd, [Dav920R1AnyNz2Nd], PartialProfile>;
def Dav920R1L12UBCoverage
    : ArchitectureCapabilityCoverage<L12UB, [Dav920R1AnyL12UB], PartialProfile>;
def Dav920R1Nchw2Nc1hwc0Coverage
    : ArchitectureCapabilityCoverage<Nchw2Nc1hwc0, [Dav920R1AnyNchw2Nc1hwc0],
                                     PartialProfile>;
def Dav920R1Nchw2C1hwnc0Coverage
    : ArchitectureCapabilityCoverage<Nchw2C1hwnc0, [Dav920R1AnyNchw2C1hwnc0],
                                     PartialProfile>;
def Dav920R1LoadScaleCoverage
    : ArchitectureCapabilityCoverage<LoadScale, [Dav920R1AnyLoadScale],
                                     PartialProfile>;

def Dav_920R1CubeLoadProfile
    : ArchitectureCapabilityProfile<
          Dav_920R1, [Dav920R1Nd2NzCoverage, Dav920R1Nd2NzBatchCoverage,
                      Dav920R1Nz2NdCoverage, Dav920R1L12UBCoverage,
                      Dav920R1Nchw2Nc1hwc0Coverage,
                      Dav920R1Nchw2C1hwnc0Coverage, Dav920R1LoadScaleCoverage]>;

#endif // BISHENGIR_TARGET_NPU_DMA_CUBELOAD_CUBELOADCAPABILITYPROFILE_TD
