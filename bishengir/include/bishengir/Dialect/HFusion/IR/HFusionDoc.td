//===- HFusionDoc.td - HFusion documentation ---------------*- tablegen -*-===//
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

#ifndef BISHENGIR_DIALECT_HFUSION_IR_HFUSIONDOC_TD
#define BISHENGIR_DIALECT_HFUSION_IR_HFUSIONDOC_TD

// Ops
include "bishengir/Dialect/HFusion/IR/HFusionBase.td"
include "bishengir/Dialect/HFusion/IR/HFusionOps.td"
include "bishengir/Dialect/HFusion/IR/HFusionStructuredOps.td"

// Attributes
include "bishengir/Dialect/HFusion/IR/HFusionAttrs.td"

#endif // BISHENGIR_DIALECT_HFUSION_IR_HFUSIONDOC_TD
