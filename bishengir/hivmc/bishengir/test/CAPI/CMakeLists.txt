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


_add_capi_test_executable(bishengir-capi-ir-test
  ir.c
  LINK_LIBS PRIVATE
    BiShengIRCAPIRegisterEverything
    MLIRCAPIIR
)

_add_capi_test_executable(bishengir-capi-pass-test
  pass.c
  LINK_LIBS PRIVATE
    BiShengIRCAPIRegisterEverything
    MLIRCAPIAnnotation
    MLIRCAPIFunc
    MLIRCAPIHFusion
    MLIRCAPIHIVM
    MLIRCAPIIR
    MLIRCAPIMesh
    MLIRCAPIRegisterEverything
    MLIRCAPITransforms
)
