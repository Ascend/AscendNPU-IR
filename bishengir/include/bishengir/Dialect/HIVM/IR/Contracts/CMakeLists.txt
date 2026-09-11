set(tblgen_feat_list)
if(BISHENGIR_BUILD_STANDALONE_IR_ONLY)
  list(APPEND tblgen_feat_list -DBISHENGIR_BUILD_STANDALONE_IR_ONLY)
endif()

set(LLVM_TARGET_DEFINITIONS ../HIVMAttrs.td)
tablegen(BISHENGIR_HIVM_IR_CONTRACT FixpipePreQuantContract.h.inc
  -gen-fixpipe-pre-quant-contract-decls ${tblgen_feat_list})
tablegen(BISHENGIR_HIVM_IR_CONTRACT FixpipePreQuantContract.cpp.inc
  -gen-fixpipe-pre-quant-contract-defs ${tblgen_feat_list})
add_public_tablegen_target(BiShengIRFixpipePreQuantContractIncGen)
add_dependencies(mlir-headers BiShengIRFixpipePreQuantContractIncGen)
