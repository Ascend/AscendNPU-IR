//===- HFusionTraits.td - HFusion library ops --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Traits used by HFusion ops.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HFUSION_IR_HFUSIONTRAITS_TD
#define BISHENGIR_DIALECT_HFUSION_IR_HFUSIONTRAITS_TD

include "mlir/IR/OpBase.td"

/// Currently only attached to manually defined structured ops. However this may
/// be attached to HFusionStructuredBase_Op if suitable.
def ResultOnlyIfTensor
    : PredOpTrait<"Result must only exist if using tensor semantics",
                  CPred<"($_op.getNumResults() == 0) ^ "
                        "(cast<DestinationStyleOpInterface>(&$_op)."
                        "hasPureTensorSemantics())">>;

/// Check to make sure that `attr` is less than the rank of `shaped`. Used for
/// GatherOp.
class AttrWithinRankOf<string attr, string shaped>
    : PredOpTrait<attr#" must be less than rank of "#shaped,
                  CPred<"$" # attr # " < static_cast<uint64_t>(cast<ShapedType>"
                        # "($" # shaped # ".getType()).getRank())">>;

#endif // BISHENGIR_DIALECT_HFUSION_IR_HFUSIONTRAITS_TD
