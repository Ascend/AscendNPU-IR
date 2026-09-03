set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(TRANSLATION_LIBS GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

set(LIBS
  ${DIALECT_LIBS}
  ${TRANSLATION_LIBS}
  MLIRHFusionToHIVM
  MLIRIR
)

add_mlir_tool(bishengir-translate
  bishengir-translate.cpp

  DEPENDS
  ${LIBS}
  )

llvm_update_compile_flags(bishengir-translate)

target_include_directories(bishengir-translate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(bishengir-translate PRIVATE ${LIBS})

mlir_check_link_libraries(bishengir-translate)
