//===- AscendDPXToHIVMRegbaseIntrins.h --===//
//===- Convert Ascend DPX dialect to HIVMRegbaseIntrins dialect --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_ASCENDDPXTOHIVMREGBASEINTRINS_ASCENDDPXTOHIVMREGBASEINTRINS_H_
#define BISHENGIR_CONVERSION_ASCENDDPXTOHIVMREGBASEINTRINS_ASCENDDPXTOHIVMREGBASEINTRINS_H_

#include <memory>

namespace mlir {
class Pass;

#define GEN_PASS_DECL_CONVERTASCENDDPXTOHIVMREGBASEINTRINS
#include "bishengir/Conversion/Passes.h.inc"

/// Create a pass to convert HIVMAVE ops to AVEIntrin ops.
std::unique_ptr<Pass> createConvertAscendDPXToHIVMRegbaseIntrinPass();

} // namespace mlir

#endif // BISHENGIR_CONVERSION_ASCENDDPXTOHIVMREGBASEINTRINS_ASCENDDPXTOHIVMREGBASEINTRINS_H_
