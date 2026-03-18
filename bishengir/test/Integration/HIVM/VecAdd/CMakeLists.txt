add_llvm_executable(bishengir-npu-hivm-vec-add
  main.cpp

  DEPENDS
  bishengir-compile
)

set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})

target_include_directories(bishengir-npu-hivm-vec-add PUBLIC
  ${ASCEND_HOME_PATH}/include
  ${ASCEND_HOME_PATH}/include/experiment/msprof
)

target_link_directories(bishengir-npu-hivm-vec-add PUBLIC
  ${ASCEND_HOME_PATH}/lib64
)

target_link_libraries(bishengir-npu-hivm-vec-add PUBLIC
  runtime
  ascendcl
)