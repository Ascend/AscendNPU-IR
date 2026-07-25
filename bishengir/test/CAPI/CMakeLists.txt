function(_add_capi_test_executable name)
  cmake_parse_arguments(ARG
    ""
    ""
    "LINK_LIBS"
    ${ARGN})
  set(LLVM_LINK_COMPONENTS
    )
  add_llvm_executable(${name}
    PARTIAL_SOURCES_INTENDED
    ${ARG_UNPARSED_ARGUMENTS})
  llvm_update_compile_flags(${name})
  if(MLIR_BUILD_MLIR_C_DYLIB)
    target_link_libraries(${name} PRIVATE
      MLIR-C)
  else()
    target_link_libraries(${name} PRIVATE
      ${ARG_LINK_LIBS})
  endif()
endfunction(_add_capi_test_executable)

if(MLIR_INCLUDE_TESTS)
  set(TEST_LIBS
    BiShengIRTestDialectHFusion
    MLIRTestDialect
    MLIRTestTransformDialect
    MLIRTestTransforms
    BiShengIRTestDialectUtils
  )
endif()

get_property(BISHENGIR_LIBS GLOBAL PROPERTY BISHENGIR_LIBS)
get_property(BISHENGIR_DIALECT_LIBS GLOBAL PROPERTY BISHENGIR_DIALECT_LIBS)
get_property(BISHENGIR_CONVERSION_LIBS GLOBAL PROPERTY BISHENGIR_CONVERSION_LIBS)
get_property(BISHENGIR_TRANSLATION_LIBS GLOBAL PROPERTY BISHENGIR_TRANSLATION_LIBS)
get_property(BISHENGIR_EXTENSION_LIBS GLOBAL PROPERTY BISHENGIR_EXTENSION_LIBS)
get_property(triton_dialect_registration_deps GLOBAL PROPERTY TRITON_DIALECT_REGISTRATION_DEPS)
get_property(BISHENGIR_TRITON_LIBS GLOBAL PROPERTY TRITON_LIBS)

set(LIBS
  ${TEST_LIBS}
  ${BISHENGIR_LIBS}
  ${BISHENGIR_DIALECT_LIBS}
  ${BISHENGIR_CONVERSION_LIBS}
  ${BISHENGIR_TRANSLATION_LIBS}
  ${BISHENGIR_EXTENSION_LIBS}
  ${BISHENGIR_TRITON_LIBS}
  BiShengIRToolUtils
  MLIRMlirOptMain
)

_add_capi_test_executable(bishengir-capi-ir-test
  ir.c
  LINK_LIBS PRIVATE
    BiShengIRCAPIRegisterEverything
    MLIRCAPIIR
    ${LIBS}
)

_add_capi_test_executable(bishengir-capi-pass-test
  pass.c
  LINK_LIBS PRIVATE
    BiShengIRCAPIRegisterEverything
    BiShengIRCAPIAnnotation
    MLIRCAPIFunc
    BiShengIRCAPIHFusion
    BiShengIRCAPIHIVM
    MLIRCAPIIR
    MLIRCAPIRegisterEverything
    MLIRCAPITransforms
    ${LIBS}
)
