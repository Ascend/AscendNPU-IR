//===- TestPasses.h -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//============================================================================//

#ifndef TEST_TESTPASSES_H
#define TEST_TESTPASSES_H
namespace bishengir_test {
// Macro to generate function declarations and the calls
// When used with a pass name like "BiShengSegmenterPass",

// function declaration: void registerBiShengSegmenterPass()
#define DECLARE_PASS(name) void register##name()
// the function call: registerBiShengSegmenterPass()
#define REGISTER_PASS(name) register##name()

// X-Macro pattern: This macro takes another macro (X) as a parameter
// and applies it to each pass name in the list.
// This allows us to generate different code for the same list of passes
// by passing different macros as X.
#define PASS_LIST(X)                                                           \
  X(BiShengSegmenterPass);                                                     \
  X(InstructionMarkerPass);                                                    \
  X(TestAssignFusionKindAttrs);                                                \
  X(TestBufferUtilsPass);                                                      \
  X(TestCanFusePass);                                                          \
  X(TestDimensionAnalyzer);                                                    \
  X(TestFlattenInterface);                                                     \
  X(TestFunctionCallPass);                                                     \
  X(TestHIVMDimensionAnalyzer);                                                \
  X(TestHIVMTransformsPass);                                                   \
  X(ValidPropagatedReshapePass)

// Generate declarations
PASS_LIST(DECLARE_PASS);
#undef DECLARE_PASS

inline void registerAllTestPasses() {
  // Generate registration calls
  PASS_LIST(REGISTER_PASS);
}
#undef REGISTER_PASS
#undef PASS_LIST
} // namespace bishengir_test
#endif // TEST_TESTPASSES_H
