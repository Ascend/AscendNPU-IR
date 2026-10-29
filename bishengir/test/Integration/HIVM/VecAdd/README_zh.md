# HIVM Vector Add

这是一个简单示例，展示了：
- 如何使用`bishengir-compile`将`.mlir`文件编译为可在Ascend NPU上执行的二进制文件
- 如何编写一个可以在Ascend NPU上执行的host端封装程序

## 要求

要在Ascend NPU上运行此示例，请安装CANN软件包，并将环境变量`ASCEND_HOME_PATH`设置为安装路径。例如：

```bash
export ASCEND_HOME_PATH=/usr/local/ascend-toolkit/latest
```

## 如何构建示例

1. 准备好上述列出的各项要求。
2. 在cmake构建选项中添加`-DBISHENGIR_BUILD_EXAMPLES=ON`。例如：

    ```bash
    cmake -G Ninja .. \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_EXTERNAL_PROJECTS="bishengir" \
    -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR="AscendNPU-IR" \ # 项目根目录
    -DBSPUB_DAVINCI_BISHENGIR=ON # 必须项！用于使能AscendNPU IR对于三方仓库的扩展
    [其他您需要的 CMake 选项]
    -DBISHENGIR_BUILD_EXAMPLES=ON # 编译集成测试用例
    ```

3. 构建项目。构建完成后，可执行文件`hivm-vec-add`会出现在`./bin`目录中。
4. 使用`bishengir-compile`构建device端二进制文件。
    ```bash
    bishengir-compile add.mlir -enable-hivm-compile -o kernel.o
    ```
5. 执行`hivm-vec-add`
    ```bash
    ./hivm-vec-add
    ```
6. 您应该会看到以下结果（部分文本已省略）：
    ```bash
    i0       Expect: 1                         Result: 1
    i1       Expect: 2                         Result: 2
    i2       Expect: 3                         Result: 3
    i3       Expect: 4                         Result: 4
    ```
