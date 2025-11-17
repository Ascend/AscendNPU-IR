/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// REQUIRES: enable-lir-compile

// DEFINE: %{compile_hivm} = -enable-hivm-compile=true
// DEFINE: %{source_mlir} = \
// DEFINE:   "%bishengir_src_root/test/Integration/HIVM/VecAdd/add.mlir"

// RUN: bishengir-compile %{compile_hivm} \
// RUN:   -o kernel.o %{source_mlir}
// RUN: bishengir-npu-hivm-vec-add 2>&1 | FileCheck %s

// CHECK: i0       Expect: 1                             Result: 1
// CHECK: i1       Expect: 2                             Result: 2
// CHECK: i2       Expect: 3                             Result: 3
// CHECK: i3       Expect: 4                             Result: 4
// CHECK: i4       Expect: 5                             Result: 5
// CHECK: i5       Expect: 6                             Result: 6
// CHECK: i6       Expect: 7                             Result: 7
// CHECK: i7       Expect: 8                             Result: 8
// CHECK: i8       Expect: 9                             Result: 9
// CHECK: i9       Expect: 10                            Result: 10
// CHECK: i10      Expect: 11                            Result: 11
// CHECK: i11      Expect: 12                            Result: 12
// CHECK: i12      Expect: 13                            Result: 13
// CHECK: i13      Expect: 14                            Result: 14
// CHECK: i14      Expect: 15                            Result: 15
// CHECK: i15      Expect: 16                            Result: 16

#include "acl/acl.h"
#include "acl/error_codes/rt_error_codes.h"
#include "experiment/runtime/runtime/rt.h"

#include <cstdio>
#include <fstream>
#include <stdlib.h>

#define EXPECT_EQ(a, b, msg)                                                   \
  do {                                                                         \
    if ((a) != (b)) {                                                          \
      printf("[failed] %s\n", (msg));                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

char *readBinFile(const char *fileName, uint32_t *fileSize) {
  std::filebuf *pbuf;
  std::ifstream filestr;
  size_t size;
  filestr.open(fileName, std::ios::binary);
  if (!filestr) {
    printf("file:%s open failed!", fileName);
    return nullptr;
  }

  pbuf = filestr.rdbuf();
  size = pbuf->pubseekoff(0, std::ios::end, std::ios::in);
  pbuf->pubseekpos(0, std::ios::in);
  char *buffer = (char *)malloc(size);
  if (!buffer) {
    printf("NULL == buffer!");
    return nullptr;
  }
  pbuf->sgetn(buffer, size);
  *fileSize = size;

  printf("[success] file:%s read succ!\n", fileName);
  filestr.close();
  return buffer;
}

void *registerBinaryKernel(const char *filePath, char **buffer,
                           const char *stubFunc, const char *kernelName) {
  void *binHandle = nullptr;
  uint32_t bufferSize = 0;
  *buffer = readBinFile(filePath, &bufferSize);
  if (*buffer == nullptr) {
    printf("readBinFile: %s failed!\n", filePath);
    return binHandle;
  }

  rtDevBinary_t binary;
  binary.data = *buffer;
  binary.length = bufferSize;
  binary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  binary.version = 0;
  rtError_t rtRet = rtDevBinaryRegister(&binary, &binHandle);
  if (rtRet != RT_ERROR_NONE) {
    printf("rtDevBinaryRegister: %s failed, errorCode=%d!\n", kernelName,
           rtRet);
    return binHandle;
  }

  rtRet = rtFunctionRegister(binHandle, (const void *)stubFunc, kernelName,
                             (void *)kernelName, 0);
  if (rtRet != RT_ERROR_NONE) {
    printf("rtFunctionRegister: %s failed, errorCode=%d!\n", kernelName, rtRet);
    return binHandle;
  }

  return binHandle;
}

int main() {
  // Initialize
  aclError error = ACL_RT_SUCCESS;
  error = aclInit(nullptr);
  EXPECT_EQ(error, ACL_RT_SUCCESS, "init failed");

  error = aclrtSetDevice(0);
  EXPECT_EQ(error, ACL_RT_SUCCESS, "set device failed");
  aclrtStream stream;
  error = aclrtCreateStream(&stream);
  EXPECT_EQ(error, ACL_RT_SUCCESS, "create stream failed");
  printf("[success] initialize success\n");

  // Register the kernel
  char *buffer;
  const char *stubFunc = "add";
  void *binHandle =
      registerBinaryKernel("./kernel.o", &buffer, stubFunc, stubFunc);
  if (!binHandle)
    exit(1);
  printf("[success] register kernel success\n");

  // Prepare data
  int16_t expectedValue[] = {1, 2,  3,  4,  5,  6,  7,  8,
                             9, 10, 11, 12, 13, 14, 15, 16};
  void *outputDevice = nullptr;
  error = aclrtMalloc((void **)&outputDevice, sizeof(expectedValue),
                      ACL_MEM_MALLOC_HUGE_FIRST);
  EXPECT_EQ(error, ACL_RT_SUCCESS, "alloc output on device failed");

  int16_t input0Value[] = {0, 1, 2,  3,  4,  5,  6,  7,
                           8, 9, 10, 11, 12, 13, 14, 15};
  void *input0Device = nullptr;
  error = aclrtMalloc((void **)&input0Device, sizeof(input0Value),
                      ACL_MEM_MALLOC_HUGE_FIRST);
  EXPECT_EQ(error, ACL_RT_SUCCESS, "alloc input0 on device failed");
  error = aclrtMemcpy((void *)input0Device, sizeof(input0Value), input0Value,
                      sizeof(input0Value), ACL_MEMCPY_HOST_TO_DEVICE);
  EXPECT_EQ(error, ACL_RT_SUCCESS, "memcopy input0 to device failed");

  int16_t input1Value[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  void *input1Device = nullptr;
  error = aclrtMalloc((void **)&input1Device, sizeof(input1Value),
                      ACL_MEM_MALLOC_HUGE_FIRST);
  EXPECT_EQ(error, ACL_RT_SUCCESS, "alloc input1 on device failed");
  error = aclrtMemcpy((void *)input1Device, sizeof(input1Value), input1Value,
                      sizeof(input1Value), ACL_MEMCPY_HOST_TO_DEVICE);
  EXPECT_EQ(error, ACL_RT_SUCCESS, "memcopy input1 to device failed");
  printf("[success] memcpy host to device success\n");

  // Invoke the kernel
  void *args[] = {input0Device, input1Device, outputDevice};
  rtKernelLaunch(stubFunc, 1, static_cast<void *>(&args), sizeof(args), nullptr,
                 stream);
  error = aclrtSynchronizeStream(stream);
  EXPECT_EQ(error, ACL_RT_SUCCESS, "stream synchronize failed");
  printf("[success] stream synchronize success\n");

  // Get the result
  int16_t *outHost = nullptr;
  error = aclrtMallocHost((void **)&outHost, sizeof(expectedValue));
  EXPECT_EQ(error, ACL_RT_SUCCESS, "alloc output on host failed");
  error = aclrtMemcpy(outHost, sizeof(expectedValue), outputDevice,
                      sizeof(expectedValue), ACL_MEMCPY_DEVICE_TO_HOST);
  EXPECT_EQ(error, ACL_RT_SUCCESS, "memcpy output to host failed");
  printf("[success] memcpy device to host success\n");

  for (int i = 0; i < sizeof(expectedValue) / sizeof(int16_t); i++) {
    printf("i%d\t Expect: %d\t\t\t\tResult: %d\n", i, expectedValue[i],
           outHost[i]);
  }
  printf("[success] compare output success\n");

  free(buffer);
  aclrtFreeHost(outHost);
  aclrtFree(outputDevice);
  aclrtFree(input0Device);
  aclrtFree(input1Device);

  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();
  return 0;
}
