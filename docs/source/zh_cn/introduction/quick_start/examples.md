# 编译与执行示例

本示例展示如何用`bishengir-compile`将IR编译为设备端二进制，并基于CANN提供的runtime接口完成注册与上板执行。

**前置条件**：已完成[构建安装](installing_guide.md)，且`bishengir-compile`已加入`PATH`，CANN环境已安装并完成`set_env.sh`配置。

## IR编译

准备一段`VecCumsum`的MLIR（可从其他IR转换得到）：

```mlir
// test_cumsum.mlir
module {
  func.func @test_vcumsum(
    %arg0: memref<8xf32, #hivm.address_space<gm>>, 
    %arg1: memref<8xf32, #hivm.address_space<gm>>
  ) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %alloc = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<8xf32, #hivm.address_space<gm>>) outs(%alloc : memref<8xf32, #hivm.address_space<ub>>)
    %alloc_1 = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.vcumsum ins(%alloc : memref<8xf32, #hivm.address_space<ub>>) outs(%alloc_1 : memref<8xf32, #hivm.address_space<ub>>) cum_dims = [0] reverse = false
    hivm.hir.store ins(%alloc_1 : memref<8xf32, #hivm.address_space<ub>>) outs(%arg1 : memref<8xf32, #hivm.address_space<gm>>)
    return
  }
}
```

使用`bishengir-compile`生成设备端二进制：

```bash
# 确保使用的bishengir-compile是从CANN安装包中获取；若从源码自编译该工具，需添加 -t（--build-bishengir-template）参数构建模板库，端到端编译必须依赖该组件。
# --target=${soc_version} 用于指定目标硬件型号，默认为 Ascend910B1。

# 编译命令
bishengir-compile test_cumsum.mlir --enable-hivm-compile --target=${soc_version} -o kernel.o
```

生成的`kernel.o`即为可在NPU上执行的算子二进制。

## Runtime注册与上板执行

以下C++代码实现CANN runtime所需的算子注册与调用接口，编译后与`kernel.o`一起即可上板执行。

```cpp
// main.cpp

#include "acl/acl.h"
#include "acl/error_codes/rt_error_codes.h"
#include "runtime/runtime/rt.h"

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
  const char *stubFunc = "test_vcumsum";
  void *binHandle =
      registerBinaryKernel("./kernel.o", &buffer, stubFunc, stubFunc);
  if (!binHandle)
    exit(1);
  printf("[success] register kernel success\n");

  // Prepare data
  float expectedValue[] = {0., 1.,  3.,  6.,  10.,  15.,  21.,  28.,};
  void *outputDevice = nullptr;
  error = aclrtMalloc((void **)&outputDevice, sizeof(expectedValue),
                      ACL_MEM_MALLOC_HUGE_FIRST);
  EXPECT_EQ(error, ACL_RT_SUCCESS, "alloc output on device failed");

  float input0Value[] = {0., 1., 2.,  3.,  4.,  5.,  6.,  7.};
  void *input0Device = nullptr;
  error = aclrtMalloc((void **)&input0Device, sizeof(input0Value),
                      ACL_MEM_MALLOC_HUGE_FIRST);
  EXPECT_EQ(error, ACL_RT_SUCCESS, "alloc input0 on device failed");
  error = aclrtMemcpy((void *)input0Device, sizeof(input0Value), input0Value,
                      sizeof(input0Value), ACL_MEMCPY_HOST_TO_DEVICE);
  EXPECT_EQ(error, ACL_RT_SUCCESS, "memcopy input0 to device failed");

  printf("[success] memcpy host to device success\n");

  // Invoke the kernel
  void *args[] = {input0Device, outputDevice};
  rtKernelLaunch(stubFunc, 1, static_cast<void *>(&args), sizeof(args), nullptr,
                 stream);
  error = aclrtSynchronizeStream(stream);
  EXPECT_EQ(error, ACL_RT_SUCCESS, "stream synchronize failed");
  printf("[success] stream synchronize success\n");

  // Get the result
  float *outHost = nullptr;
  error = aclrtMallocHost((void **)&outHost, sizeof(expectedValue));
  EXPECT_EQ(error, ACL_RT_SUCCESS, "alloc output on host failed");
  error = aclrtMemcpy(outHost, sizeof(expectedValue), outputDevice,
                      sizeof(expectedValue), ACL_MEMCPY_DEVICE_TO_HOST);
  EXPECT_EQ(error, ACL_RT_SUCCESS, "memcpy output to host failed");
  printf("[success] memcpy device to host success\n");

  for (int i = 0; i < sizeof(expectedValue) / sizeof(float); i++) {
    printf("i%d\t Expect: %f\t\t\t\tResult: %f\n", i, expectedValue[i],
           outHost[i]);
  }
  printf("[success] compare output success\n");

  free(buffer);
  aclrtFreeHost(outHost);
  aclrtFree(outputDevice);
  aclrtFree(input0Device);

  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();
  return 0;
}
```

编译可执行文件（`main.cpp`会读取当前目录下的`kernel.o`并完成注册与调用）：

```bash
# 加载CANN运行环境（若已写入Shell配置可省略）；路径以实际安装目录为准，详情参考文档「快速开始-安装与构建」
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 配置相应的头文件路径与链接路径，若 CANN 安装在自定义路径，请将 ASCEND_HOME_PATH 替换为实际路径，或使用 set_env.sh 所设置的环境变量名
RT_INC=${ASCEND_HOME_PATH}/include
PROF_INC=${ASCEND_HOME_PATH}/include/experiment/msprof
PKG_INC=${ASCEND_HOME_PATH}/pkg_inc
RT_LIB=${ASCEND_HOME_PATH}/lib64

g++ main.cpp -I${RT_INC} -I${PROF_INC} -I${PKG_INC} -L ${RT_LIB} -l runtime -l ascendcl -o vec_vcumsum
```

运行示例：

```bash
./vec_vcumsum
```

预期输出（片段）：

```text
    i0       Expect: 0.000000                         Result: 0.000000
    i1       Expect: 1.000000                         Result: 1.000000
    i2       Expect: 3.000000                         Result: 3.000000
    i3       Expect: 6.000000                         Result: 6.000000
    i4       Expect: 10.000000                        Result: 10.000000
    i5       Expect: 15.000000                        Result: 15.000000
    i6       Expect: 21.000000                        Result: 21.000000
    i7       Expect: 28.000000                        Result: 28.000000
    ...
```
