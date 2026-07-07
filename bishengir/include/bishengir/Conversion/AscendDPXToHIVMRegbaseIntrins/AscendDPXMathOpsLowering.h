//===- AscendDPMathOpsLowering.h --===//
//===- Convert Ascend DPX Math Ops to HIVMRegbaseIntrins dialect --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_ASCENDDPXMATHOPSLOWERING_ASCENDDPXMATHOPSLOWERING_H_
#define BISHENGIR_CONVERSION_ASCENDDPXMATHOPSLOWERING_ASCENDDPXMATHOPSLOWERING_H_

namespace mlir {

class RewritePatternSet;
class LLVMTypeConverter;

void addAscendDPXMathOpsLoweringPatterns(RewritePatternSet& patterns, LLVMTypeConverter& converter);

}

#endif // BISHENGIR_CONVERSION_ASCENDDPXMATHOPSLOWERING_ASCENDDPXMATHOPSLOWERING_H_
