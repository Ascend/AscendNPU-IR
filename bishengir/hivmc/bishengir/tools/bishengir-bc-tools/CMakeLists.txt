set(LIBS
  LLVMSupport
  MLIRSupport
)

add_mlir_tool(embed-bc-to-cpp
  embed-bc-to-cpp.cpp

  DEPENDS
  ${LIBS}
)

target_link_libraries(embed-bc-to-cpp PRIVATE ${LIBS})
mlir_check_all_link_libraries(embed-bc-to-cpp)
