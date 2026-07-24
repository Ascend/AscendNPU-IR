get_property(BISHENGIR_LIBS GLOBAL PROPERTY BISHENGIR_LIBS)
get_property(BISHENGIR_DIALECT_LIBS GLOBAL PROPERTY BISHENGIR_DIALECT_LIBS)
get_property(BISHENGIR_CONVERSION_LIBS GLOBAL PROPERTY BISHENGIR_CONVERSION_LIBS)
get_property(BISHENGIR_TRANSLATION_LIBS GLOBAL PROPERTY BISHENGIR_TRANSLATION_LIBS)
get_property(BISHENGIR_EXTENSION_LIBS GLOBAL PROPERTY BISHENGIR_EXTENSION_LIBS)
get_property(BISHENGIR_TRITON_LIBS GLOBAL PROPERTY TRITON_LIBS)

if(MLIR_INCLUDE_TESTS)
  set(TEST_LIBS
    MLIRAffineTransformsTestPasses
    MLIRShapeTestPasses
    MLIRSPIRVTestPasses
    MLIRTestAnalysis
    MLIRTestDialect
    MLIRTestDynDialect
    MLIRTestIR
    MLIRTestPass
    MLIRTestReducer
    )
  set(TEST_LIBS
    ${TEST_LIBS}
    MLIRTestTransformDialect
    MLIRTestTransforms)
endif()

set(LIBS
  ${TEST_LIBS}
  ${BISHENGIR_LIBS}
  ${BISHENGIR_DIALECT_LIBS}
  ${BISHENGIR_CONVERSION_LIBS}
  ${BISHENGIR_TRANSLATION_LIBS}
  ${BISHENGIR_TRITON_LIBS}
  ${BISHENGIR_EXTENSION_LIBS}
)

add_mlir_tool(bishengir-lsp-server
  bishengir-lsp-server.cpp
  DEPENDS
  ${LIBS}
)

target_link_libraries(bishengir-lsp-server PRIVATE 
  MLIRLspServerLib    
  ${LIBS}
)

mlir_check_all_link_libraries(bishengir-lsp-server)
