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

set(LIBS
  ${TEST_LIBS}
  ${BISHENGIR_LIBS}
  ${BISHENGIR_DIALECT_LIBS}
  ${BISHENGIR_CONVERSION_LIBS}
  ${BISHENGIR_TRANSLATION_LIBS}
  ${BISHENGIR_EXTENSION_LIBS}
  BiShengIRToolUtils
  MLIRMlirOptMain
)

add_bishengir_publish_tool(bishengir-opt
  bishengir-opt.cpp

  DEPENDS
  ${LIBS}
)

target_include_directories(bishengir-opt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(MLIR_INCLUDE_TESTS)
  target_include_directories(bishengir-opt PRIVATE ${BISHENGIR_SRC_DIR}/test/lib/Dialect)
endif()

target_link_libraries(bishengir-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(bishengir-opt)
