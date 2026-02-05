# AscendNPU IR编译&执行用力

本样例展示如何编译链接，并基于CANN提供的runtime接口实现上板执行。

## IR编译

我们先准备一段`VecAdd`的IR，该IR可以从其他IR转换得到。
```mlir
; add.mlir
module {
  func.func @add(%arg0: memref<16xi16, #hivm.address_space<gm>>, %arg1: memref<16xi16, #hivm.address_space<gm>>, %arg2: memref<16xi16, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %alloc = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<16xi16, #hivm.address_space<gm>>) outs(%alloc : memref<16xi16, #hivm.address_space<ub>>)
    %alloc_0 = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<16xi16, #hivm.address_space<gm>>) outs(%alloc_0 : memref<16xi16, #hivm.address_space<ub>>)
    %alloc_1 = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%alloc, %alloc_0 : memref<16xi16, #hivm.address_space<ub>>, memref<16xi16, #hivm.address_space<ub>>) outs(%alloc_1 : memref<16xi16, #hivm.address_space<ub>>)
    hivm.hir.store ins(%alloc_1 : memref<16xi16, #hivm.address_space<ub>>) outs(%arg2 : memref<16xi16, #hivm.address_space<gm>>)
    return
  }
}
```

使用`bishengir-compile`构建device端二进制文件。
```bash
bishengir-compile add.mlir -enable-hivm-compile -o kernel.o
```

这时的`kernel.o`即为可上班执行的二进制。

## runtime注册上板执行

通过一段c++代码，实现runtime需要的注册接口与执行接口，编译后即可上板执行。

```cpp
// main.cpp

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
```

编译可执行文件， `main.cpp`会读取 `kernel.o`的内容并加载执行。
```bash
source /usr/local/Ascend/cann/set_env.sh
g++ main.cpp -I ${ASCEND_HOME_PATH}/include -I${ASCEND_HOME_PATH}/include/experiment/msprof -L ${ASCEND_HOME_PATH}/lib64 -l runtime -l ascendcl -o vec-add
```

执行`vec-add`
```bash
./vec-add
```

输出结果（部分省略）
```bash
    i0       Expect: 1                         Result: 1
    i1       Expect: 2                         Result: 2
    i2       Expect: 3                         Result: 3
    i3       Expect: 4                         Result: 4
```