
//===-- HIVMOps.td - Entry point for HIVM bind ---------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------------------------===//

#ifndef PYTHON_BINDINGS_HIVM_OPS
#define PYTHON_BINDINGS_HIVM_OPS

include "bishengir/Dialect/HIVM/IR/HIVMOps.td"
include "bishengir/Dialect/HIVM/IR/HIVMIntrinOps.td"
include "bishengir/Dialect/HIVM/IR/HIVMMacroOps.td"
include "bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td"
include "bishengir/Dialect/HIVM/IR/HIVMDMAOps.td"
include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.td"
#endif
