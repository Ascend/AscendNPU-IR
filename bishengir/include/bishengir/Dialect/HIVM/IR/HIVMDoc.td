//===- HIVMDoc.td - HIVM documentation ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This documentation files exists to circumvent limitations on mixing different
// .td files in cases one does not want to have all ops belong to the same
// logical unit. This file should only include other .td files only and be used
// for the purpose of generating documentation.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMDOC_TD
#define BISHENGIR_DIALECT_HIVM_IR_HIVMDOC_TD

// Ops
include "bishengir/Dialect/HIVM/IR/HIVMBase.td"
include "bishengir/Dialect/HIVM/IR/HIVMDMAOps.td"
include "bishengir/Dialect/HIVM/IR/HIVMIntrinOps.td"
include "bishengir/Dialect/HIVM/IR/HIVMMacroOps.td"
include "bishengir/Dialect/HIVM/IR/HIVMOps.td"
include "bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td"
include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.td"

// Attributes
include "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMDOC_TD
