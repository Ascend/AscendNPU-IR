if(MLIR_INCLUDE_TESTS)
  set(TEST_LIBS
    BiShengIRTestDialect
    MLIRTestDialect
    MLIRTestTransformDialect
    MLIRTestTransforms
  )
endif()

get_property(PRIVATE_BISHENGIR_LIBS GLOBAL PROPERTY BISHENGIR_LIBS)
get_property(PRIVATE_BISHENGIR_CONVERSION_LIBS GLOBAL PROPERTY BISHENGIR_CONVERSION_LIBS)
get_property(PRIVATE_BISHENGIR_TRANSLATION_LIBS GLOBAL PROPERTY BISHENGIR_TRANSLATION_LIBS)
get_property(PUBLIC_BISHENGIR_CONVERSION_LIBS GLOBAL PROPERTY BISHENGIR_CONVERSION_OPEN_SOURCE_LIBS)

set(LIBS
  ${TEST_LIBS}
  ${PRIVATE_BISHENGIR_LIBS}
  ${PRIVATE_BISHENGIR_CONVERSION_LIBS}
  ${PRIVATE_BISHENGIR_TRANSLATION_LIBS}
  ${PUBLIC_BISHENGIR_CONVERSION_LIBS}
)

add_bishengir_publish_tool(bishengir-hivm-compile
  bishengir-hivm-compile.cpp

  DEPENDS
  ${LIBS}
)

target_include_directories(bishengir-hivm-compile PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bishengir-hivm-compile PRIVATE ${LIBS})
mlir_check_all_link_libraries(bishengir-hivm-compile)
