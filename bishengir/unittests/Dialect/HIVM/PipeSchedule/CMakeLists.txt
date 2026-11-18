add_bishengir_unittest(BiShengIRHIVMMsTests
  SmsSchedulerTest.cpp
  MsDagBuilderTest.cpp
  MsPartitionerTest.cpp
  MSCostModelTest.cpp

  ../../../../lib/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/DDG.cpp
  ../../../../lib/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/ResMII.cpp
  ../../../../lib/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/AsapAlap.cpp
  ../../../../lib/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/DepthHeight.cpp
  ../../../../lib/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/SwingOrder.cpp
  ../../../../lib/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/MRT.cpp
  ../../../../lib/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/SmsScheduler.cpp
  ../../../../lib/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/DagVerifier.cpp
  ../../../../lib/Dialect/HIVM/Utils/PipeSchedule/CostModel/MSCostModel.cpp
)

target_link_libraries(BiShengIRHIVMMsTests PRIVATE
  MLIRIR
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRParser
  MLIRSCFDialect
  MLIRTensorDialect
  BiShengIRAnnotationDialect
  BiShengIRHIVMDialect
  BiShengIRHIVMTransforms
  BiShengIRMemRefExtDialect
  BiShengIRScopeDialect
)
