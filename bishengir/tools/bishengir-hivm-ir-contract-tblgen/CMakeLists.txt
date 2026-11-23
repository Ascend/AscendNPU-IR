set(LLVM_LINK_COMPONENTS
  CodeGenTypes
  Demangle
  Support
  TableGen
)

add_tablegen(bishengir-hivm-ir-contract-tblgen BISHENGIR_HIVM_IR_CONTRACT
  bishengir-hivm-ir-contract-tblgen.cpp
  FixpipePreQuantContractGen.cpp
)

target_link_libraries(bishengir-hivm-ir-contract-tblgen
  PRIVATE
  MLIRTblgenLib
)

mlir_check_all_link_libraries(bishengir-hivm-ir-contract-tblgen)
