# Build and Installation

This document describes dependency installation, build methods (source and binary), and running tests for AscendNPU IR.

## Installing Dependencies

### Build Dependencies

#### Compiler and Toolchain Requirements

Basic compiler and toolchain requirements:

- CMake >= 3.28
- Ninja >= 1.12.0

Recommended:

- Clang >= 10
- LLD >= 10 (LLVM LLD significantly speeds up builds)

#### Source Code Preparation

1. Clone the repository (then go to the repository directory, typically `ascendnpu-ir`).

   ```bash
   git clone https://gitcode.com/Ascend/ascendnpu-ir.git
   cd ascendnpu-ir
   ```

2. Initialize and update submodules.

   This project depends on third-party libraries such as LLVM and Torch-MLIR. You need to pull and update them to the specified commit IDs.

   ```bash
   # Recursively pull all submodules.
   git submodule update --init --recursive
   ```

### Runtime Dependencies

#### Installing the CANN Package

The end-to-end operation of the AscendNPU-IR depends on the CANN environment.

1. Download the CANN package: Download the toolkit package and the ops package for your hardware from the [Ascend Community CANN download page](https://www.hiascend.com/cann/download).

2. Install the CANN package.

   ```bash
   # In the x86 A3 environment, {version} indicates the CANN version, for example, 9.0.0.
   chmod +x Ascend-cann_{version}_linux-x86_64.run
   chmod +x Ascend-cann-A3-ops_{version}_linux-x86_64.run
   ./Ascend-cann_{version}_linux-x86_64.run --full [--install-path=${PATH-TO-CANN}]
   ./Ascend-cann-A3-ops_{version}_linux-x86_64.run --install [--install-path=${PATH-TO-CANN}]
   ```

3. Set environment variables.

   ```bash
   # If the version is earlier than 8.5.0, the path is ${PATH-TO-CANN}/ascend-toolkit/set_env.sh.
   source ${PATH-TO-CANN}/cann/set_env.sh
   ```

## Build Commands

### Source Build

#### Using the Build Script (Recommended)

Run `./build-tools/build.sh` in the root directory of the project to complete the configuration, build, and installation. The script automatically handles the CMake configuration, Ninja build, and installation.

**First build** (initialize submodules and apply patches):

```bash
# In the root directory of the project:
./build-tools/build.sh -o ./build --build-type Release --apply-patches
```

**Subsequent builds** (when the build directory already exists):

```bash
./build-tools/build.sh -o ./build --build-type Release
```

**Rebuild** (clear the build directory and reconfigure it):

```bash
./build-tools/build.sh -o ./build --build-type Release -r
```

#### Script parameters

| Parameter| Description| Default Value|
|------|------|--------|
| `-o`, `--build PATH` | Build output directory.| `./build` |
| `--build-type TYPE` | Build type.| `Release` |
| `--apply-patches` | Apply patches to third-party submodules; required for the first build.| Off|
| `-r`, `--rebuild` | Clear the build directory and reconfigure it.| Off|
| `-j`, `--jobs N` | Number of parallel build threads.| 3/4 of CPU cores|
| `--install-prefix PATH` | Installation path.| `BUILD_DIR/install` |
| `--c-compiler PATH` | C compiler path.| `clang` |
| `--cxx-compiler PATH` | C++ compiler path.| `clang++` |
| `--llvm-source-dir DIR` | LLVM source code directory.| `third-party/llvm-project` |
| `--build-test` | Build and run tests.| Off|
| `--build-bishengir-doc` | Build BiShengIR documentation.| Off|
| `-t`, `--build-bishengir-template` | Build the BiShengIR template library, **required for E2E cases. To enable this option, you need to install CANN 9.0.0.**| Off|
| `--bisheng-compiler PATH` | Path to the BiSheng compiler, required when building the template library.| None|
| `--build-torch-mlir` | Also build torch-mlir.| Off|
| `--python-binding` | Enable MLIR python-binding.| Off|
| `--enable-cpu-runner` | Enabling CPU runner.| Off|
| `--disable-ccache` | Disable ccache.| On (if installed)|
| `--enable-assertion` | Enable assertions.| Off|
| `--fast-build` | Skip the installation step.| Off|
| `--add-cmake-options OPTIONS` | Append CMake options.| None|

#### Common Examples

```bash
# Debug build with tests
./build-tools/build.sh -o ./build --build-type Debug --apply-patches --build-test

# Specify the compiler and number of threads
./build-tools/build.sh -o ./build --c-compiler /usr/bin/clang-15 --cxx-compiler /usr/bin/clang++-15 -j 256

# Quick build (skip installation)
./build-tools/build.sh -o ./build --fast-build

# Rebuild and build the template library (E2E case execution depends on the template library)
./build-tools/build.sh -r -o ./build --fast-build -t --bisheng-compiler=/usr/Ascend/cann/bin
```

#### Manual Build (Advanced)

For full manual control, perform the following steps:

**Prerequisites**: Submodules have been initialized (`git submodule update --init --recursive`) and patches have been applied (`source build-tools/apply_patches.sh`).

```bash
# In the root directory of the project:
mkdir -p build
cd build

# LLVM source code path: third-party/llvm-project/llvm
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
    # [-DOther options=Value]

ninja -j32
```

**Note**: `[]` indicates that the option is optional. To use an option, remove the leading `# and`[]` and add the parameter to the command.

| Optional Option| Description|
|----------|------|
| `-DCMAKE_INSTALL_PREFIX="${PWD}/install"` | Installation path.|
| `-DLLVM_MAJOR_VERSION_21_COMPATIBLE=ON` | Required when the LLVM version is 21 or later.|
| `-DLLVM_ENABLE_ASSERTIONS=ON` | Enable assertions (common for debug).|
| `-DMLIR_ENABLE_BINDINGS_PYTHON=ON` | Enable MLIR python-binding.|
| `-DLLVM_TARGETS_TO_BUILD="host;Native"` | Enabling CPU runner.|
| `-DBISHENGIR_PUBLISH=OFF` | Disable unpublished functions.|
| `-DBISHENGIR_BUILD_TEMPLATE=ON -DBISHENG_COMPILER_PATH=...` | Build the BiShengIR template library.|

### Binary Installation

AscendNPU IR binaries are installed with CANN Toolkit. For details, see [CANN Package Installation](#installing-the-cann-package).

## Running Tests

### Build Test Target

```bash
# In the build directory:
cmake --build . --target "check-mlir;check-bishengir"
```

This command runs `check-mlir` and `check-bishengir`, executing the BiShengIR test suite using llvm-lit.

#### Typical Output

When the test passes:

```text
-- Testing: 388 tests, 8 workers --
...

Testing Time: 45.23s

Total Discovered Tests: 388
  Unsupported: 89  (22.94%)
  Passed     : 299 (77.06)
```

When the test fails:

```text
-- Testing: 388 tests, 8 workers --
PASS: bishengir :: bishengir-compile/commandline.mlir (1 of 388)
...
FAIL: bishengir :: test/failing-case.mlir (42 of 388)
******************** TEST 'FAIL: bishengir :: test/failing-case1.mlir' FAILED ********************
... (failure details)...

********************
FAIL: bishengir :: test/failing-case.mlir (256 of 388)
******************** TEST 'FAIL: bishengir :: test/failing-case2.mlir' FAILED ********************
... (failure details)...

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

#### Test Result Criteria

**Pass**: The command exit code is 0 and the value of `Failed` is 0. The following results count as **pass**:

- **PASS**: The test passes.
- **UNSUPPORTED**: The current environment does not support the test (for example, `UNSUPPORTED: bishengir_published`).
- **XFAIL**: Expected to fail and did fail.

**Fail**: The command exit code is not 0, or the value of `Failed` is greater than 0. The following results count as **fail**:

- **FAIL**: The test fails.
- **XPASS**: Expected to fail but passed.
- **UNRESOLVED**: The result cannot be determined.
- **TIMEOUT**: The test timed out.

### Running Tests with LLVM-LIT

```bash
# In the build directory:
./bin/llvm-lit ../bishengir/test
```

You can directly specify the test path, for example, `./bin/llvm-lit ../bishengir/test/bishengir-compile/commandline.mlir`.

## FAQ

Q: What should I do if the error message "ninja: error: loading 'build.ninja': No such file or directory" is displayed when I run the `build-tools/build.sh` script?
A: Add the `-r` option to `build-tools/build.sh`, and run CMake again to generate a new `build.ninja` file.

Q: What should I do if the error message "Too many open files" is displayed during build?
A: The number of open files exceeds the system limit. Run the `ulimit -n xxx` command (for example, `ulimit -n 65535`) to raise the upper limit.

Q: What should I do if the following error is reported during build?

```bash
 The CMAKE_CXX_COMPILER:

  clang++

 is not a full path and was not found in the PATH.
```

A: The C++ compiler is not specified or the compiler binary is invalid. Use the `--cxx-compiler=${CXX-COMPILER-PATH}` command to specify the C++ compiler. If the error persists, reinstall the compiler or use another version (e.g., the recommended `clang++-15`).
