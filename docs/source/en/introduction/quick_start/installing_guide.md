# Build and Installation

This document describes the dependency installation, build method (source/binary), and test execution steps for AscendNPU IR.

## Environment Dependency Preparation

**Compiler and Toolchain Requirements**:

- CMake >= 3.28
- Ninja >= 1.12.0

**Recommended Tools**:

- Clang >= 10
- LLD >= 10 (Using LLVM LLD significantly improves the build speed.)

**Source Code Retrieval and Submodule Initialization**:

This project depends on third-party libraries such as LLVM and Torch-MLIR, which need to be retrieved and updated to the specified `commit id`.

```bash
# Clone the main repository and enter the project root directory.
git clone https://gitcode.com/Ascend/ascendnpu-ir.git
cd ascendnpu-ir
# Initialize and update the submodules.
git submodule update --init --recursive
```

**CANN package installation**:

End-to-end execution of AscendNPU IR depends on the CANN environment.

1. Download the CANN package: You need to download the CANN Toolkit package and the `ops` package corresponding to your hardware, which can be obtained from the [Ascend Community CANN download page](https://www.hiascend.com/cann/download).

2. Install the CANN package:

   ```bash
   # Take installing the CANN package on an x86 Atlas A3 series product as an example, where {version} is the CANN version, such as 9.0.0.
   chmod +x Ascend-cann_{version}_linux-x86_64.run
   chmod +x Ascend-cann-A3-ops_{version}_linux-x86_64.run
   ./Ascend-cann_{version}_linux-x86_64.run --full [--install-path=${PATH-TO-CANN}]
   ./Ascend-cann-A3-ops_{version}_linux-x86_64.run --install [--install-path=${PATH-TO-CANN}]
   ```

3. Set the environment variables:

   ```bash
   # For version 8.5.0 and earlier, the path is ${PATH-TO-CANN}/ascend-toolkit/set_env.sh.
   source ${PATH-TO-CANN}/cann/set_env.sh
   ```

## Build Method

### One-Click Build Script build.sh (Recommended)

Run `./build-tools/build.sh` in the project root directory to complete configuration, building, and installation. The script automatically handles CMake configuration, Ninja compilation, and installation steps.

```bash
# First build (initialize submodules first).
./build-tools/build.sh -o ./build --build-type Release

# Existing build directory, incremental compilation.
./build-tools/build.sh -o ./build --build-type Release

# Clear the build directory and perform a full rebuild.
./build-tools/build.sh -o ./build --build-type Release -r
```

**Script parameters**:

| Parameter | Description | Default Value |
|------|------|--------|
| `-o`, `--build PATH` | Build output directory | `./build` |
| `--build-type TYPE` | Build type | `Release` |
| `-r`, `--rebuild` | Clear the build directory and reconfigure | Off |
| `-j`, `--jobs N` | Number of parallel compilation threads | 3/4 of the CPU core count |
| `--install-prefix PATH` | Installation path | `BUILD_DIR/install` |
| `--c-compiler PATH` | `C` compiler path | clang |
| `--cxx-compiler PATH` | C++ compiler path | clang++ |
| `--llvm-source-dir DIR` | LLVM source directory | `third-party/llvm-project` |
| `--build-test` | Build and run tests | Off |
| `--build-bishengir-doc` | Build BiShengIR documentation | Off |
| `-t`, `--build-bishengir-template` | Build the BiShengIR template library. **Enabling this option requires CANN 9.0.0, and it must be enabled to run end-to-end test cases.** | Off |
| `--bisheng-compiler PATH` | Directory of the bisheng compiler. **Must be specified when building the template library.** | None |
| `--build-torch-mlir` | Build torch-mlir as well | Off |
| `--python-binding` | Enable MLIR python-binding | Off |
| `--enable-cpu-runner` | Enable the CPU runner | Off |
| `--disable-ccache` | Disable ccache | Enabled (if installed) |
| `--enable-assertion` | Enable assertions | Off |
| `--fast-build` | Skip the installation step | Off |
| `--add-cmake-options OPTIONS` | Append CMake options | None |

**Common examples**:

```bash
# Debug build and run tests.
./build-tools/build.sh -o ./build --build-type Debug --build-test

# Specify the compiler and the number of threads.
./build-tools/build.sh -o ./build --c-compiler /usr/bin/clang-15 --cxx-compiler /usr/bin/clang++-15 -j 256

# Quick build (without installation)
./build-tools/build.sh -o ./build --fast-build

# Rebuild and build the template library (end-to-end test case execution depends on the template library)
./build-tools/build.sh -r -o ./build --fast-build -t --bisheng-compiler=/usr/Ascend/cann/bin
```

### Manual CMake+Ninja Build (Custom Compilation Parameters)

This method applies to scenarios that require fine-grained control over compilation parameters. Prerequisite: submodule initialization has been completed (`git submodule update --init --recursive`).

```bash
# In the project root directory.
mkdir -p build
cd build

# LLVM source path: third-party/llvm-project/llvm.
export LLVM_SOURCE_DIR="$(realpath ../third-party/llvm-project)"
cmake ${LLVM_SOURCE_DIR}/llvm -G Ninja \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_EXTERNAL_PROJECTS="bishengir" \
    -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR="$(realpath ..)" \
    -DBSPUB_DAVINCI_BISHENGIR=ON \
    # [-DCMAKE_INSTALL_PREFIX="${PWD}/install"] \
    # [-DLLVM_MAJOR_VERSION_21_COMPATIBLE=ON] \
    # [-DLLVM_ENABLE_ASSERTIONS=ON] \
    # [-DMLIR_ENABLE_BINDINGS_PYTHON=ON] \
    # [-DLLVM_TARGETS_TO_BUILD="host;Native"] \
    # [-DBISHENGIR_PUBLISH=OFF] \
    # [-DBISHENGIR_BUILD_TEMPLATE=ON -DBISHENG_COMPILER_PATH=/path/to/bisheng-compiler] \
    # [-Dother options=value]

ninja -j32
```

> **Note**:
>
> `[]` indicates an optional item. When using an option, remove the leading `#` and `[]`, and add the parameter to the command.

**Optional extension parameters**:

| Parameter | Description |
|----------|------|
| `-DCMAKE_INSTALL_PREFIX="${PWD}/install"` | Installation path |
| `-DLLVM_MAJOR_VERSION_21_COMPATIBLE=ON` | Required when the LLVM version is 21 or later |
| `-DLLVM_ENABLE_ASSERTIONS=ON` | Enables assertions (commonly used in Debug builds) |
| `-DMLIR_ENABLE_BINDINGS_PYTHON=ON` | Enables MLIR python-binding |
| `-DLLVM_TARGETS_TO_BUILD="host;Native"` | Enables the CPU runner |
| `-DBISHENGIR_PUBLISH=OFF` | Disables unpublished features |
| `-DBISHENGIR_BUILD_TEMPLATE=ON -DBISHENG_COMPILER_PATH=...` | Builds the BiShengIR template library |

### Binary Installation (No Compilation Required)

The AscendNPU IR binaries are installed together with the CANN Toolkit package. See the "CANN Package Installation" content in [Environment Dependency Preparation](#environment-dependency-preparation).

## Using Docker Images Directly

Use Docker images for development and verification without configuring the environment.

### Reusing the CANN Image

The CANN Toolkit package includes the complete AscendNPU IR binaries, so the Docker image can reuse the CANN image.

You can find the relevant tags based on the hardware platform and CANN version at [https://quay.io/repository/ascend/cann?tab=tags](https://quay.io/repository/ascend/cann?tab=tags). For example, `ascend/cann:9.0.0-a3-openeuler24.03-py3.12`

### Building a Local Image

Developers can also build a local image independently. For details, refer to the `Dockerfile` files for different architectures provided in the `docker` directory.

For each architecture, the image includes the CANN Toolkit package and the latest compiled AscendNPU IR. You can install the `ops` package and Torch-related components on your own to explore more features.

The base images for different architectures are as follows:

- x86_64 image: based on `ubuntu:22.04`
- aarch64 image: based on `openeuler/openeuler:24.03`

Build method:

```bash
# Build the x86_64 image.
docker build -t ascendnpu-ir:latest -f docker/Dockerfile.x86_64 .
```

### Using the Image

```bash
IMAGE_NAME="ascendnpu-ir:latest"       # The CANN image can be ascend/cann:9.0.0-a3-openeuler24.03-py3.12.
# After entering, the environment can directly use AscendNPU IR related tools, such as bisheng-compile.
docker run -it \
  --net=host --privileged \            # Develop in host mode.
  --security-opt seccomp=unconfined \  # Disable security restrictions.
  --device=/dev/davinci0 \             # Mount the NPU device.
  --device=/dev/davinci1 \
  --device=/dev/davinci2 \
  --device=/dev/davinci3 \
  --device=/dev/davinci4 \
  --device=/dev/davinci5 \
  --device=/dev/davinci6 \
  --device=/dev/davinci7 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \ # Mount devices such as dcmi.
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  --name ascendnpu-ir   \              # Container name.
  -v $(pwd):/workspace  \              # Mount the current directory to the container.
  -w /workspace         \
  $IMAGE_NAME  /bin/bash

bishengir-compile --version # View the version information of AscendNPU IR.
```

## Running Tests

**Compile test target**:

```bash
# In the `build` directory.
cmake --build . --target "check-mlir;check-bishengir"
```

This command executes `check-mlir` and `check-bishengir`, running the BiShengIR test suite through `llvm-lit`.

**Typical output example**:

When the test passes:

```text
-- Testing: 388 tests, 8 workers --
...

Testing Time: 45.23s

Total Discovered Tests: 388
  Unsupported: 89  (22.94%)
  Passed     : 299 (77.06%)
```

When the test fails:

```text
-- Testing: 388 tests, 8 workers --
PASS: bishengir :: bishengir-compile/commandline.mlir (1 of 388)
...
FAIL: bishengir :: test/failing-case.mlir (42 of 388)
******************** TEST 'FAIL: bishengir :: test/failing-case1.mlir' FAILED ********************
...(failure details)...

********************
FAIL: bishengir :: test/failing-case.mlir (256 of 388)
******************** TEST 'FAIL: bishengir :: test/failing-case2.mlir' FAILED ********************
...(failure details)...

********************
********************
Failed Tests (2):
  bishengir :: test/failing-case1.mlir
  bishengir :: test/failing-case2.mlir

Testing Time: 38.12s

Total Discovered Tests: 388
  Unsupported:  86 (22.16%)
  Passed     : 300 (77.32%)
  Failed     :   2 (0.52%)
  ...
```

**Test pass criteria**:

- **Test passed**: The command exit code is 0 and `Failed` is 0. The following results are all counted as passed:

    - **PASS**: The test passes normally.

    - **UNSUPPORTED**: The current environment does not support it (for example, `UNSUPPORTED: bishengir_published`).

    - **XFAIL**: Expected to fail and actually fails.

- **Test failed**: The command exit code is not 0, or `Failed` > 0. The following results are all counted as failed:

    - **FAIL**: The test execution fails.

    - **XPASS**: Expected to fail but actually passes.

    - **UNRESOLVED**: The result cannot be determined.

    - **TIMEOUT**: Timeout.

**Use `LLVM-LIT` to execute the test suite**:

```bash
# In the build directory
./bin/llvm-lit ../bishengir/test
```

You can directly specify the test path, for example: `./bin/llvm-lit ../bishengir/test/bishengir-compile/commandline.mlir`.

## FAQ

For common issues during build and installation, see [FAQ-Build and Installation](../../faq/faq.md#build-and-installation).
