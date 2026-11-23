//===- HIVMPass.td - HIVM pass base ------------------------*- tablegen -*-===//
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
// Pass base for HIVM passes. It carries the kernel kinds a pass applies to
// and whether function core types should be re-inferred after the pass runs.
// mlir-tblgen ignores these arguments. bishengir-options-tblgen
// -gen-bishengir-pass-metadata emits them, and rejects a concrete pass that
// does not derive from HIVMPass.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_HIVMPASS_TD
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_HIVMPASS_TD

include "mlir/Pass/PassBase.td"

// Which kernel kinds a pass is intended for. A pass lists every kind it
// applies to. The module core type selects one kind (MIX → MixKernel,
// AIV → PureAivKernel, AIC → PureAicKernel). The pass does not run when
// that kind is not in the list. A missing core type runs every pass.
class KernelKind;
def MixKernel : KernelKind;
def PureAivKernel : KernelKind;
def PureAicKernel : KernelKind;

// Default: the pass runs for mix, pure AIV, and pure AIC kernels.
defvar AllKernelKinds = [MixKernel, PureAivKernel, PureAicKernel];

// No operation anchor. The pass is OperationPass<> and runs on the operation
// the pipeline schedules it on. Pass this when a later argument is set and
// the operation argument cannot be omitted.
defvar AnyOp = "";

// Same C++ base as mlir::Pass, plus the kernel kinds the pass applies to and
// whether the function core type should be re-inferred after the pass runs.
// Omitted arguments use the defaults: AnyOp, every kernel kind, and
// classifyKernel = false.
class HIVMPass<string passArg, string operation = AnyOp,
               list<KernelKind> kernelKinds = AllKernelKinds,
               bit classifyKernel = false>
    : PassBase<passArg, "::mlir::OperationPass<"#operation#">"> {
  // Copy the arguments onto the record. TableGen drops template arguments
  // that the class body does not reference.
  list<KernelKind> appliedKernelKinds = kernelKinds;
  bit appliedClassifyKernel = classifyKernel;
}

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_HIVMPASS_TD
