# Triton Access #

[Triton Ascend](https://gitcode.com/Ascend/triton-ascend/)	It is an important component that helps Triton access the Ascend platform. After the Triton Ascend is built and installed, you can use the Ascend as the backend when executing the Triton operator.

## Installation guide ##

### Environment preparation ###

#### Python version requirements ####

Currently, the Python version required by Triton-Ascend is py3.9-py3.11.

#### Installing the Ascend CANN ####

Heterogeneous Computing Architecture for Neural Networks (CANN) is a heterogeneous computing architecture developed by Ascend for AI scenarios. CANN supports multiple AI frameworks, including MindSpore, PyTorch, and TensorFlow, and provides AI processors and programming services. Plays a key role in connecting the previous and the following and is a key platform for improving the computing efficiency of Ascend AI processors.

You can visit the Ascend community official website and follow the software installation guide provided by the website to install and configure the Ascend.

During installation, CANN version "\{version\}" Please select one of the following versions:

**CANN version:**

 *  Commercial version

| Triton-Ascend Version | CANN commercial version | CANN Release Date |
| --------------------- | ----------------------- | ----------------- |
| 3.2.0                 | CANN 8.5.0              | 2026 / 01 / 16    |
| 3.2.0rc4              | CANN 8.3.RC2            | 2025 / 11 / 20    |
|                       | CANN 8.3.RC1            | 2025 / 10 / 30    |

 *  Community edition

| Triton-Ascend Version | CANN Community Version | CANN Release Date |
| --------------------- | ---------------------- | ----------------- |
| 3.2.0                 | CANN 8.5.0             | 2026 / 01 / 16    |
| 3.2.0rc4              | CANN 8.3.RC2           | 2025 / 11 / 20    |
|                       | CANN 8.5.0.alpha001    | 2025 / 11 / 12    |
|                       | CANN 8.3.RC1           | 2025 / 10 / 30    |

Specify the software package corresponding to the CPU architecture \{arch\} (aarch64/x86_64) and software version \{version\} based on the actual environment.

You are advised to download and install version 8.5. 0:

| Software Type | Software Package Description | Software Package Name                              |
| ------------- | ---------------------------- | -------------------------------------------------- |
| Toolkit       | CANN Development Kit Package | Ascend-cann-toolkit_\{version\}_linux-\{arch\}.run |
| Ops           | CANN binary operator package | Ascend-cann-A3-ops_\{version\}_linux-\{arch\}.run  |

Note 1: The Ops package name of the A2 series is slightly different from that of the A3 series. For details, see the format (Ascend-cann-910b-ops_\{version\}_linux-\{arch\}.run).

Note 2: The Ops packages in versions earlier than 8.5.0 are named in the (Atlas-A3-cann-kernels_\{version\}_linux-\{arch\}.run) format.

[Community download link](https://www.hiascend.com/developer/download/community/result?module=cann)	You can find the corresponding software package.

[Community Installation Guide Link](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit)	Provides a complete installation process description and dependency configuration recommendations for users who need to deploy a comprehensive CANN environment.

**Script for installing the CANN.**

Take A3 CANN version 8.5.0 as an example. Scripted installation is provided for your reference.

```
#Changing the Execution Permission of the Run Package
chmod +x Ascend-cann-toolkit_8.5.0_linux-aarch64.run
chmod +x Ascend-cann-A3-ops_8.5.0_linux-aarch64.run

#Common installation (default installation path: /usr/local/Ascend)
sudo ./Ascend-cann-toolkit_8.5.0_linux-aarch64.run --install
#Default installation path (same as the Toolkit package: /usr/local/Ascend)
sudo ./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install
#The environment variables in the default path take effect.
source /usr/local/Ascend/ascend-toolkit/set_env.sh

#Installing the Python Dependency of CANN
pip install attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pyyaml
```

 *  Note: If the user does not specify the installation path, the software will be installed in the default path. The default installation path is as follows:`/usr/local/Ascend`For non-root users:`${HOME}/Ascend`, $\{HOME\} indicates the directory of the current user. The preceding environment variables take effect only in the current window. You can set`source ${HOME}/Ascend/ascend-toolkit/set_env.sh`Write the command to the environment variable configuration file, such as the .bashrc file.

#### Installing torch_npu ####

Currently, the torch_npu version is 2.7.1.

```
pip install torch_npu==2.7.1
```

Note: If an error is reported,`ERROR: No matching distribution found for torch==2.7.1+cpu`You can manually install the torch and then install the torch_npu.

```
pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

### Installing Triton-Asscend Using pip ###

#### Latest stable version ####

You can install the latest stable version of Triton-Ascend through pip.

```
pip install triton-ascend
```

 *  Note: If the community Triton is already installed, uninstall the community Triton first. Then install Triton-Ascend to avoid conflicts.

```
pip uninstall triton
pip install triton-ascend
```

#### Nightly build version ####

We provide a daily updated nightly package for users, which can be installed by the following command.

```
pip install -i https://test.pypi.org/simple/ "triton-ascend<3.2.0rc" --pre --no-cache-dir
```

At the same time, users can also[History List](https://test.pypi.org/project/triton-ascend/#history)	Find all nightly build packages in.

Note that if you are performing`pip install`If an SSL-related error is reported, append the error.`--trusted-host test.pypi.org --trusted-host test-files.pythonhosted.org`Option to resolve.

### Installing Triton-Asscend Using the Source Code ###

If you need to develop or customize the Triton-Ascend, you should compile and install the Triton-Ascend using the source code. This allows you to adjust the source code to meet project requirements and to compile and install a customized version of Triton-Ascend.

#### System Requirements ####

 *  GCC >= 9.4.0
 *  GLIBC >= 2.27

#### Depends on ####

**Installing the System Library Dependency**

Install zlib1g-dev/lld/clang. You can install the ccache package to accelerate build.

 *  Recommended version: clang >= 15
 *  Recommended LLD >= 15

```
以ubuntu系统为例：
sudo apt update
sudo apt install zlib1g-dev clang-15 lld-15
sudo apt install ccache #optional
```

The build of Triton-Asscend strongly depends on zlib1g-dev. If you use the yum source, run the following command to install it:

```
sudo yum install -y zlib-devel
```

**Installing the Python Dependency**

```
pip install ninja cmake wheel pybind11 #build-time dependencies
```

#### Built based on LLVM ####

Triton uses LLVM20 to generate code for the GPU and CPU. Similarly, the Bisheng compiler of Ascend also relies on the LLVM to generate the NPU code. Therefore, the LLVM source code must be compiled to use it. Pay attention to the specific version of the LLVM that depends on. The LLVM can be built in either of the following modes. You do not need to repeat the steps.

**Code preparation:`git checkout`Check out the LLVM of the specified version.**

```
git clone --no-checkout https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout b5cc222d7429fe6f18c787f633d5262fac2e676f
```

**Method 1: Install the LLVM in Clang.**

 *  Step 1: You are advised to use clang to install LLVM. Install clang and lld in the environment and specify the versions (recommended version: clang >= 15, lld >= 15). If the LLVM is not installed, To install clang, lld, and ccache, run the following commands:
    
    ```
    apt-get install -y clang-15 lld-15 ccache
    ```
 *  Step 2: Set the environment variable LLVM_INSTALL_PREFIX to the target installation path.
    
    ```
    export LLVM_INSTALL_PREFIX=/path/to/llvm-install
    ```
 *  Run the following commands to build and install the LLVM:
    
    ```
    cd $HOME/llvm-project  #Path of the LLVM code pulled by the git clone user
    mkdir build
    cd build
    cmake ../llvm \
      -G Ninja \
      -DCMAKE_C_COMPILER=/usr/bin/clang-15 \
      -DCMAKE_CXX_COMPILER=/usr/bin/clang++-15 \
      -DCMAKE_LINKER=/usr/bin/lld-15 \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
      -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
      -DLLVM_ENABLE_LLD=ON \
      -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX}
    ninja install
    ```

**Method 2: Install the LLVM on the GCC.**

 *  Step 1: Clang is recommended. If only GCC can be used, pay attention to the following:[Note 1](#note1)	 [Note 2](#note2)	. Set the environment variable LLVM_INSTALL_PREFIX to your target installation path:
    
    ```
    export LLVM_INSTALL_PREFIX=/path/to/llvm-install
    ```
 *  Step 2: Run the following commands to build and install the software:
    
    ```
    cd $HOME/llvm-project  #your clone of LLVM.
    mkdir build
    cd build
    cmake -G Ninja  ../llvm  \
       -DLLVM_CCACHE_BUILD=OFF \
       -DCMAKE_BUILD_TYPE=Release \
       -DLLVM_ENABLE_ASSERTIONS=ON \
       -DLLVM_ENABLE_PROJECTS="mlir;llvm"  \
       -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
       -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX}
    ninja install
    ```

Note 1: If an error occurs during compilation,`ld.lld: error: undefined symbol`. You can add the settings in step 2.`-DLLVM_ENABLE_LLD=ON`.

Note 2: If the ccache has been installed and is running properly, you can set this parameter.`-DLLVM_CCACHE_BUILD=ON`Do not enable this function unless it is used to accelerate the build.

**Clone Triton-Ascend**

```
git clone https://gitcode.com/Ascend/triton-ascend.git && cd triton-ascend/python
```

**Building Triton-Ascend**

1.  Source code installation

 *  Step 1: Ensure that the target path $\{LLVM_INSTALL_PREFIX\} for installing the LLVM has been set in \[Building Based on LLVM\] section.
 *  Step 2: Ensure that clang >=15, lld >=15, and ccache have been installed.
    
    ```
    LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
    TRITON_BUILD_WITH_CCACHE=true \
    TRITON_BUILD_WITH_CLANG_LLD=true \
    TRITON_BUILD_PROTON=OFF \
    TRITON_WHEEL_NAME="triton-ascend" \
    TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
    python3 setup.py install
    ```
 *  Note 3: It is recommended that GCC be greater than or equal to 9.4.0. If GCC is smaller than 9.4.0, the error ld.lld: error: unable to find library -lstdc++fs may be reported, indicating that the linker cannot find the stdc++fs library. This library is used to support file system features prior to GCC 9. In this case, you need to manually comment out the related code snippets in the CMake file.
 *  triton-ascend/CMakeLists.txt
    
    ```
    if (NOT WIN32 AND NOT APPLE)
    link_libraries(stdc++fs)
    endif()
    ```
    
    Uncomment and rebuild the project can resolve the problem.

2.  Running the Triton sample
    
    Install the runtime dependency as follows:
    
    ```
    cd triton-ascend && pip install -r requirements_dev.txt
    ```
    
    Run the sample file:`third_party/ascend/tutorials/01-vector-add.py`
    
    ```
    #Setting CANN Environment Variables (The default installation path of the root user is `/usr/local/Ascend`.)
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    #To run the tutorials sample:
    python3 ./triton-ascend/third_party/ascend/tutorials/01-vector-add.py
    ```
    
    If information similar to the preceding is displayed, the environment is correctly configured.
    
    ```
    tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
    tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
    The maximum difference between torch and triton is 0.0
    ```

### Invoke the Triton Kernel. ###

After the Triton-Ascend is successfully installed, you can call the related Triton Kernel. For details, see the following source code. You can run the`pytest -sv <file>.py`Verify the functions after the installation. If the function is correct, the terminal displays`PASS`.

```
from typing import Optional
import pytest
import triton
import triton.language as tl
import torch
import torch_npu

def generate_tensor(shape, dtype):
    if dtype == 'float32' or dtype == 'float16' or dtype == 'bfloat16':
        return torch.randn(size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16':
        return torch.randint(low=0, high=2000, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int8':
        return torch.randint(low=0, high=127, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'bool':
        return torch.randint(low=0, high=2, size=shape).bool()
    elif dtype == 'uint8':
        return torch.randint(low=0, high=255, size=shape, dtype=torch.uint8)
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))

def validate_cmp(dtype, y_cal, y_ref, overflow_mode: Optional[str] = None):
    y_cal=y_cal.npu()
    y_ref=y_ref.npu()
    if overflow_mode == "saturate":
        if dtype in ['float32', 'float16']:
            min_value = -torch.finfo(dtype).min
            max_value = torch.finfo(dtype).max
        elif dtype in ['int32', 'int16', 'int8']:
            min_value = torch.iinfo(dtype).min
            max_value = torch.iinfo(dtype).max
        elif dtype == 'bool':
            min_value = 0
            max_value = 1
        else:
            raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))
        y_ref = torch.clamp(y_ref, min=min_value, max=max_value)
    if dtype == 'float16':
        torch.testing.assert_close(y_ref, y_cal,  rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'bfloat16':
        torch.testing.assert_close(y_ref.to(torch.float32), y_cal.to(torch.float32),  rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'float32':
        torch.testing.assert_close(y_ref, y_cal,  rtol=1e-04, atol=1e-04, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'int8':
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'uint8' or dtype == 'uint16' or dtype == 'uint32' or dtype == 'uint64':
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))

def torch_lt(x0, x1):
    return x0 < x1

@triton.jit
def triton_lt(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr,XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop1 in range(loops1):
        x_index = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + x_index, None)
        tmp1 = tl.load(in_ptr1 + x_index, None)
        tmp2 = tmp0 < tmp1
        tl.store(out_ptr0 + x_index, tmp2, None)

@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (32,), 1, 32, 32],
                         ])
def test_lt(param_list):
    #Generate Data
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = generate_tensor(shape, dtype).npu()
    x1 = generate_tensor(shape, dtype).npu()
    #Torch results
    torch_res = torch_lt(x0, x1).to(eval('torch.' + dtype))
    #triton results
    triton_res = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_lt[ncore, 1, 1](x0, x1, triton_res, xblock, xblock_sub)
    #Compare Results
    validate_cmp(dtype, triton_res, torch_res)
```

**Dynamic tiling support: The parallel granularity is configured by the grid parameter in \[\], and the tiling size is controlled by the XBLOCK and XBLOCK_SUB parameters. Users can adjust the size as required.**

**Dynamic shape support: The kernel automatically adapts to 1D tensors of any length. Users only need to transfer the actual shape data.**

## Triton Op to Ascend NPU IR Op Conversion ##

Triton Ascend degrades the advanced GPU abstraction operations of the Triton dialect to target dialects such as Linalg, HFusion, and HIVM, resulting in optimized intermediate representations that can be efficiently executed on the Ascend NPU. The following table details the various Triton operations and their corresponding Ascend NPU IR operations in the fall process.

| Triton Op                         | Target Ascend NPU IR Op                                                                                                                        | Description                                                                  |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Storage Access Op**             |                                                                                                                                                |                                                                              |
| `triton::StoreOp`                 | `memref::copy`                                                                                                                                 | Store data in memory.                                                        |
| `triton::LoadOp`                  | `memref::copy`\+`bufferization::ToTensorOp`                                                                                                    | Loads data from memory.                                                      |
| `triton::AtomicRMWOp`             | `hivm::StoreOp`or the`hfusion::AtomicXchgOp`                                                                                                   | Perform atomic read-modify-write operations.                                 |
| `triton::AtomicCASOp`             | `linalg::GenericOp`                                                                                                                            | Performs atomic compare and swap operations.                                 |
| `triton::GatherOp`                | First convert to`func::CallOp`(Invokes`triton_gather`) and then convert to`hfusion::GatherOp`                                                  | Collect data based on the index.                                             |
| **Pointer operation class Op.**   |                                                                                                                                                |                                                                              |
| `triton::AddPtrOp`                | `memref::ReinterpretCast`                                                                                                                      | Performs an offset operation on the pointer.                                 |
| `triton::PtrToIntOp`              | `arith::IndexCastOp`                                                                                                                           | Converts a pointer to an integer.                                            |
| `triton::IntToPtrOp`              | `hivm::PointerCastOp`                                                                                                                          | Converts an integer to a pointer.                                            |
| `triton::AdvanceOp`               | `memref::ReinterpretCastOp`                                                                                                                    | Push the pointer in position.                                                |
| **Program information operation** |                                                                                                                                                |                                                                              |
| `triton::GetProgramIdOp`          | `functionOp`Parameters for                                                                                                                     | Gets the ID of the current program.                                          |
| `triton::GetNumProgramsOp`        | `functionOp`Parameters for                                                                                                                     | Gets the total number of programs.                                           |
| `triton::AssertOp`                | First convert to`func::CallOp`(Invokes`triton_assert`) and then converted to`hfusion::AssertOp`                                                | Assertion operation.                                                         |
| `triton::PrintOp`                 | First convert to`func::CallOp`(Invokes`triton_print`) and then convert to`hfusion::PrintOp`                                                    | Print operation.                                                             |
| **Tensor operation operation**    |                                                                                                                                                |                                                                              |
| `triton::ReshapeOp`               | `tensor::ReshapeOp`                                                                                                                            | Change the tensor shape.                                                     |
| `triton::ExpandDimsOp`            | `tensor::ExpandShapeOp`                                                                                                                        | Extended tensor dimension.                                                   |
| `triton::BroadcastOp`             | `linalg::BroadcastOp`                                                                                                                          | Broadcast tensor.                                                            |
| `triton::TransOp`                 | `linalg::TransposeOp`                                                                                                                          | Transpose tensors.                                                           |
| `triton::SplitOp`                 | `tensor::ExtractSliceOp`                                                                                                                       | Split tensor.                                                                |
| `triton::JoinOp`                  | `tensor::InsertSliceOp`                                                                                                                        | Connection tensor.                                                           |
| `triton::CatOp`                   | `tensor::InsertSliceOp`                                                                                                                        | Splice tensor.                                                               |
| `triton::MakeRangeOp`             | `linalg::GenericOp`                                                                                                                            | Creates a tensor that contains consecutive integers.                         |
| `triton::SplatOp`                 | `linalg::FillOp`                                                                                                                               | Fills the tensor with scalar values.                                         |
| `triton::SortOp`                  | First convert to`func::CallOp`(Invokes`triton_sort`) and then convert to`hfusion::SortOp`                                                      | Sorts tensors.                                                               |
| **Value calculation type**        |                                                                                                                                                |                                                                              |
| `triton::MulhiUIOp`               | `arith::MulSIExtendedOp`                                                                                                                       | Unsigned integer multiplication, returning the high order result.            |
| `triton::PreciseDivFOp`           | `arith::DivFOp`                                                                                                                                | Performs a high precision floating-point division.                           |
| `triton::PreciseSqrtOp`           | `math::SqrtOp`                                                                                                                                 | Performs a high precision floating-point square root.                        |
| `triton::BitcastOp`               | `arith::BitcastOp`                                                                                                                             | Bit reinterpretation between different types.                                |
| `triton::ClampFOp`                | `tensor::EmptyOp`\+`linalg::FillOp`                                                                                                            | Limits floating point numbers to a specified range.                          |
| `triton::DotOp`                   | `linalg::MatmulOp`                                                                                                                             | Performs a general matrix multiplication.                                    |
| `triton::DotScaledOp`             | `linalg::MatmulOp`                                                                                                                             | Performs a matrix multiplication with a scaling factor.                      |
| `triton::ascend::FlipOp`          | First convert to`func::CallOp`(Invokes`triton_flip`) and then converted to`hfusion::FlipOp`                                                    | Performs a matrix multiplication with a scaling factor.                      |
| **Reduced Op**                    |                                                                                                                                                |                                                                              |
| `triton::ArgMinOp`                | `linalg::ReduceOp`                                                                                                                             | Returns the index of the smallest value in the tensor.                       |
| `triton::ArgMaxOp`                | `linalg::ReduceOp`                                                                                                                             | Returns the index of the maximum value in a tensor.                          |
| `triton::ReduceOp`                | `linalg::ReduceOp`                                                                                                                             | General reduction operation.                                                 |
| `triton::ScanOp`                  | First convert to`func::CallOp`(Invokes`triton_cumsum`or the`triton_cumprod`) and then convert to`hfusion::CumsumOp`and the`hfusion::CumprodOp` | Perform scanning operations (such as cumulative sum and cumulative product). |

## Triton's only extended operations ##

We provide triton with multiple language features for Ascend. To enable the capabilities, you need to import the following modules

```
import triton.language.extra.cann.extension as al
```

The relevant Ascend Language (al) unique interface can then be used

In addition, Ascend Language provides bottom-layer interfaces, which are not compatible.

### Synchronizing and Debugging Operations ###

#### debug_barrier ####

The Ascend provides multiple synchronization modes and supports the internal synchronization mode of the vector pipeline for fine-grained synchronization control during debugging and performance optimization.

For details about the mode, see section SYNC_IN_VF.

##### Parameter Description #####

| Parameter name | Type       | Description                          |
| -------------- | ---------- | ------------------------------------ |
| `sync_mode`    | SYNC_IN_VF | Vector pipeline synchronization mode |

**Use Example**

```
@triton.jit
def kernel_debug_barrier(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(core_mode="vector"):
        al.debug_barrier(al.SYNC_IN_VF.VV_ALL)
        x = tl.load(x_ptr + i, mask=i < n)
        y = tl.load(y_ptr + i, mask=i < n)
        result = x + y
        tl.store(out_ptr + i, result, mask=i < n)
```

#### sync_block_set ####

The Ascend supports setting synchronization events between the compute unit and the vector unit.

**Parameter Description**

| Parameter name        | Type         | Description            |
| --------------------- | ------------ | ---------------------- |
| `sender`              | str          | Sending unit type      |
| `receiver`            | str          | Receiving unit type    |
| `event_id`            | TensorHandle | Event Identifier       |
| `sender_pipe_value`   | -            | Send Pipe Value        |
| `receiver_pipe_value` | -            | Receive pipeline value |

**Use Example**

```
@triton.jit
def triton_matmul_exp(A_ptr, B_ptr, C_ptr, TBuff_ptr,
                      M, N, K: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)

    offs_i = tl.arange(0, 1)[:, None]
    offs_j = tl.arange(0, 1)[None, :]
    offs_k = tl.arange(0, K)
    a_ptrs = A_ptr + (row + offs_i) * K + offs_k[None, :]
    a_vals = tl.load(a_ptrs)
    b_ptrs = B_ptr + offs_k[:, None] * N + (col + offs_j)
    b_vals = tl.load(b_ptrs)

    tbuff_ptrs = TBuff_ptr + (row + offs_i) * N + (col + offs_j)
    acc_11 = tl.dot(a_vals, b_vals)
    tl.store(tbuff_ptrs, acc_11)
    
    extension.sync_block_set("cube", "vector", 5, pipe.PIPE_MTE1, pipe.PIPE_MTE3)
    extension.sync_block_wait("cube", "vector", 5, pipe.PIPE_MTE1, pipe.PIPE_MTE3)
    
    acc_11_reload = tl.load(tbuff_ptrs)
    c_ptrs = C_ptr + (row + offs_i) * N + (col + offs_j)
    tl.store(c_ptrs, tl.exp(acc_11_reload))
```

#### sync_block_wait ####

Ascend supports waiting for a synchronization event between a compute unit and a vector unit.

**Parameter Description**

| Parameter name        | Type         | Description            |
| --------------------- | ------------ | ---------------------- |
| `sender`              | str          | Sending unit type      |
| `receiver`            | str          | Receiving unit type    |
| `event_id`            | TensorHandle | Event Identifier       |
| `sender_pipe_value`   | -            | Send Pipe Value        |
| `receiver_pipe_value` | -            | Receive pipeline value |

**Use Example**

```
@triton.jit
def triton_matmul_exp(A_ptr, B_ptr, C_ptr, TBuff_ptr,
                      M, N, K: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)

    offs_i = tl.arange(0, 1)[:, None]
    offs_j = tl.arange(0, 1)[None, :]
    offs_k = tl.arange(0, K)
    a_ptrs = A_ptr + (row + offs_i) * K + offs_k[None, :]
    a_vals = tl.load(a_ptrs)
    b_ptrs = B_ptr + offs_k[:, None] * N + (col + offs_j)
    b_vals = tl.load(b_ptrs)

    tbuff_ptrs = TBuff_ptr + (row + offs_i) * N + (col + offs_j)
    acc_11 = tl.dot(a_vals, b_vals)
    tl.store(tbuff_ptrs, acc_11)
    
    extension.sync_block_set("cube", "vector", 5, pipe.PIPE_MTE1, pipe.PIPE_MTE3)
    extension.sync_block_wait("cube", "vector", 5, pipe.PIPE_MTE1, pipe.PIPE_MTE3)
    
    acc_11_reload = tl.load(tbuff_ptrs)
    c_ptrs = C_ptr + (row + offs_i) * N + (col + offs_j)
    tl.store(c_ptrs, tl.exp(acc_11_reload))
```

#### sync_block_all ####

Ascend supports global synchronization of the entire computing block, ensuring that all computing cores of a specified type complete the current operation.

**Parameter Description**

| Parameter name | Type | Description                                            | Valid Value                                            |
| -------------- | ---- | ------------------------------------------------------ | ------------------------------------------------------ |
| `mode`         | str  | Sync mode, specifying the core type to be synchronized | `"all_cube"`,`"all_vector"`,`"all"`,`"all_sub_vector"` |
| `event_id`     | int  | synchronization event identifier                       | `0`\-`15`                                              |

**Synchronization mode details**

| mode               | Description                           | Synchronization Range                                    |
| ------------------ | ------------------------------------- | -------------------------------------------------------- |
| `"all_cube"`       | Synchronizing All Cube Cores          | All Cube cores on the current AI core                    |
| `"all_vector"`     | Synchronizing All Vector Cores        | All vectoring cores on the current AI core               |
| `"all"`            | Synchronize all cores                 | All computing cores (Cube+Vector) on the current AI core |
| `"all_sub_vector"` | Synchronizing all sub vectoring cores | All sub vectoring cores on the current AI core           |

**Use Example**

```
@triton.jit
def test_sync_block_all():
    al.sync_block_all("all_cube", 8)
    al.sync_block_all("all_vector", 9)
    al.sync_block_all("all", 10)
    al.sync_block_all("all_sub_vector", 11)
```

### Hardware query and control operations ###

#### sub_vec_id ####

The Ascend supports obtaining the vector core index on the current AI core.

**Use Example**

```
@triton.jit
def triton_matmul_exp(
    A_ptr,
    B_ptr,
    C_ptr,
    TBuff_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    sub_M: tl.constexpr,
):
    row_matmul = tl.program_id(0)
    col = tl.program_id(1)

    offs_i = tl.arange(0, tl.constexpr(M))[:, None]
    offs_j = tl.arange(0, N)[None, :]
    offs_k = tl.arange(0, K)

    a_ptrs = A_ptr + (row_matmul + offs_i) * K + offs_k[None, :]
    a_vals = tl.load(a_ptrs)

    b_ptrs = B_ptr + offs_k[:, None] * N + (col + offs_j)
    b_vals = tl.load(b_ptrs)

    tbuff_ptrs = TBuff_ptr + (row_matmul + offs_i) * N + (col + offs_j)

    acc_11 = tl.dot(a_vals, b_vals)
    tl.store(tbuff_ptrs, acc_11)
    sub_vec_id = al.sub_vec_id()
    row_exp = row_matmul + sub_M * sub_vec_id
    offs_exp_i = tl.arange(0, tl.constexpr(sub_M))[:, None]
    tbuff_exp_ptrs = TBuff_ptr + (row_exp + offs_exp_i) * N + (col + offs_j)
    acc_11_reload = tl.load(tbuff_exp_ptrs)
    c_ptrs = C_ptr + (row_exp + offs_exp_i) * N + (col + offs_j)
    tl.store(c_ptrs, tl.exp(acc_11_reload))
```

#### sub_vec_num ####

The Ascend supports obtaining the number of vectoring cores on a single AI core.

**Use Example**

```
@triton.jit
def triton_matmul_exp(
    A_ptr,
    B_ptr,
    C_ptr,
    TBuff_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    row_matmul = tl.program_id(0)
    col = tl.program_id(1)

    offs_i = tl.arange(0, tl.constexpr(M))[:, None]
    offs_j = tl.arange(0, N)[None, :]
    offs_k = tl.arange(0, K)

    a_ptrs = A_ptr + (row_matmul + offs_i) * K + offs_k[None, :]
    a_vals = tl.load(a_ptrs)

    b_ptrs = B_ptr + offs_k[:, None] * N + (col + offs_j)
    b_vals = tl.load(b_ptrs)

    tbuff_ptrs = TBuff_ptr + (row_matmul + offs_i) * N + (col + offs_j)

    acc_11 = tl.dot(a_vals, b_vals)
    tl.store(tbuff_ptrs, acc_11)

    sub_vec_id = al.sub_vec_id()
    row_exp = row_matmul + (M //al.sub_vec_num()) * sub_vec_id
    offs_exp_i = tl.arange(0, M //al.sub_vec_num())[:, None]
    tbuff_exp_ptrs = TBuff_ptr + (row_exp + offs_exp_i) * N + (col + offs_j)
    acc_11_reload = tl.load(tbuff_exp_ptrs)
    c_ptrs = C_ptr + (row_exp + offs_exp_i) * N + (col + offs_j)
    tl.store(c_ptrs, tl.exp(acc_11_reload))
```

#### parallel ####

Ascend extends the Python standard`range`capability, adding parallel execution semantics`parallel`Iterator.

**Parameter Description**

| Parameters           | Type | Description                          | Example                                 |
| -------------------- | ---- | ------------------------------------ | --------------------------------------- |
| `arg1`               | int  | Start or End Value                   | `parallel(10)`                          |
| `arg2`               | int  | End Value (Optional)                 | `parallel(0, 10)`                       |
| `step`               | int  | Step (Optional)                      | `parallel(0, 10, 2)`                    |
| `num_stages`         | int  | Number of pipeline phases (optional) | `parallel(0, 10, num_stages=3)`         |
| `loop_unroll_factor` | int  | Cycle spread factor (optional)       | `parallel(0, 10, loop_unroll_factor=4)` |

**Restriction**

Currently, the OptiX RTN 910B supports a maximum of two vectoring cores.

**Use Example**

```
@triton.jit
def triton_add(in_ptr0, in_ptr1, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    idx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    x0 = tl.load(in_ptr0 + idx)
    x1 = tl.load(in_ptr1 + idx)
    ret = x0 + x1
    
    for _ in extension.parallel(2, 5, 2):
        ret = ret + x1
    
    for _ in extension.parallel(2, 10, 3):
        ret = ret + x0
    
    odx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    tl.store(out_ptr0 + odx, ret)
```

### Compilation Optimization Tips ###

#### compile_hint ####

Ascend supports passing optimization prompts to the compiler to guide code generation and performance tuning.

**Parameter Description**

| Parameter name | Type           | Description                   |
| -------------- | -------------- | ----------------------------- |
| `ptr`          | tensor         | Pointer to the target tensor. |
| `hint_name`    | str            | Prompt name                   |
| `hint_val`     | Multiple types | Prompt Value (Optional)       |

**Use Example**

```
@triton.jit
def triton_where_lt_case1(in_ptr0, in_ptr1, cond_ptr, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        in0 = tl.load(in_ptr0 + xindex, xmask)
        in1 = tl.load(in_ptr1 + xindex, xmask)
        cond = tl.load(cond_ptr + xindex, xmask)
        mask = tl.where(cond, in1, in0)
        al.compile_hint(mask, "bitwise_mask")
        tl.store(out_ptr0 + (xindex), mask, xmask)
```

#### multibuffer ####

`multibuffer`is a function used to set up Double Buffering for existing tensors, optimizing data flow and computational overlap through compiler hints.

**Parameter Description**

| Parameters | Type   | Description                    |
| ---------- | ------ | ------------------------------ |
| `src`      | tensor | Tensor to be multiple buffered |
| `size`     | int    | Number of buffered copies      |

**Use Example**

```
@triton.jit
def triton_compile_hint(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        al.multibuffer(tmp0, 2)
        tl.store(out_ptr0 + (xindex), tmp0, xmask)
```

#### scope ####

Ascend supports scope managers, adding hint information to a section of locale code, one use of which is through`core_mode`Specifies the cube or vector type.

**Parameter Description**

| Parameter name | Type | Description                                                                                                                                   |
| -------------- | ---- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `core_mode`    | str  | Core type, which specifies the computing core used by operations in a block. Only the core type is accepted.`"cube"`or the`"vector"`Two modes |

**Core Mode Options**

| mode       | Description                           |
| ---------- | ------------------------------------- |
| `"cube"`   | Use the Cube core for calculation.    |
| `"vector"` | Using the Vector Core for Calculation |

**Use Example**

```
@triton.jit
def kernel_debug_barrier(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(core_mode="vector"):
        al.debug_barrier(al.SYNC_IN_VF.VV_ALL)
        x = tl.load(x_ptr + i, mask=i < n)
        y = tl.load(y_ptr + i, mask=i < n)
        result = x + y
        tl.store(out_ptr + i, result, mask=i < n)
```

### Tensor slice operation ###

#### insert_slice ####

Ascend supports inserting one tensor into another based on the offset, size, and step parameters of the operation.

**Parameter Description**

| Parameter name | Type          | Description                                |
| -------------- | ------------- | ------------------------------------------ |
| `ful`          | Tensor        | Receive the inserted target tensor         |
| `sub`          | Tensor        | Source tensor to be inserted               |
| `offsets`      | Integer Tuple | Start offset of the insert operation.      |
| `sizes`        | Integer Tuple | Size range of insert operations            |
| `strides`      | Integer Tuple | Step parameter of the insertion operation. |

**Use Example**

```
@triton.jit
def triton_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr, SLICE_OFFSET: tl.constexpr, SLICE_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    x_sub = al.extract_slice(x, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    y_sub = al.extract_slice(y, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    output_sub = x_sub + y_sub
    output = tl.load(output_ptr + offsets, mask=mask)
    output = al.insert_slice(output, output_sub, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    tl.store(output_ptr + offsets, output, mask=mask)
```

#### extract_slice ####

Ascend supports extracting a specified slice from another tensor based on the offset, size, and step parameters of the operation.

**Parameter Description**

| Parameter name | Type          | Description                                |
| -------------- | ------------- | ------------------------------------------ |
| `ful`          | Tensor        | Source tensor of the slice to be extracted |
| `offsets`      | Integer Tuple | Start offset of the fetch operation        |
| `sizes`        | Integer Tuple | Size range for fetch operations            |
| `strides`      | Integer Tuple | Step parameter of the extraction operation |

**Use Example**

```
@triton.jit
def triton_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr, SLICE_OFFSET: tl.constexpr, SLICE_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    x_sub = al.extract_slice(x, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    y_sub = al.extract_slice(y, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    output_sub = x_sub + y_sub
    output = tl.load(output_ptr + offsets, mask=mask)
    output = al.insert_slice(output, output_sub, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    tl.store(output_ptr + offsets, output, mask=mask)
```

#### get_element ####

Ascend supports reading a single element value at a specified index location from a tensor.

**Parameter Description**

| Parameter name | Type      | Description                                                |
| -------------- | --------- | ---------------------------------------------------------- |
| `src`          | tensor    | Source Tensor to Access                                    |
| `indice`       | int tuple | Specifies the index location of the element to be obtained |

**Use Example**

```
@triton.jit
def index_select_manual_kernel(in_ptr, indices_ptr, out_ptr, dim,
                                g_stride: tl.constexpr, indice_length: tl.constexpr,
                                g_block: tl.constexpr, g_block_sub: tl.constexpr, 
                                other_block: tl.constexpr):
    g_begin = tl.program_id(0) * g_block
    for goffs in range(0, g_block, g_block_sub):
        g_idx = tl.arange(0, g_block_sub) + g_begin + goffs
        g_mask = g_idx < indice_length
        indices = tl.load(indices_ptr + g_idx, g_mask, other=0)

        for other_offset in range(0, g_stride, other_block):
            tmp_buf = tl.zeros((g_block_sub, other_block), in_ptr.dtype.element_ty)
            other_idx = tl.arange(0, other_block) + other_offset
            other_mask = other_idx < g_stride
            
            for i in range(0, g_block_sub):
                gather_offset = al.get_element(indices, (i,)) * g_stride
                val = tl.load(in_ptr + gather_offset + other_idx, other_mask)
                tmp_buf = al.insert_slice(
                    tmp_buf, val[None, :],
                    offsets=(i, 0), sizes=(1, other_block), strides=(1, 1)
                )
            
            tl.store(out_ptr + g_idx[:, None] * g_stride + other_idx[None, :], 
                     tmp_buf, g_mask[:, None] & other_mask[None, :])
```

### Tensor Calculation Operations ###

#### sort ####

Ascend supports sorting input tensors along specified dimensions.

**Parameter Description**

| Parameter name | Type                         | Description                                                                              | Default value |
| -------------- | ---------------------------- | ---------------------------------------------------------------------------------------- | ------------- |
| `ptr`          | tensor                       | Input Tensor                                                                             | -             |
| `dim`          | int or tl.constexpr\[int\]   | Dimension to sort                                                                        | `-1`          |
| `descending`   | bool or tl.constexpr\[bool\] | Sorting direction,`True`Indicates the descending order,`False`Indicates ascending order. | `False`       |

**Use Example**

```
@triton.jit
def sort_kernel_2d(X, Z, N: tl.constexpr, M: tl.constexpr, descending: tl.constexpr):
    offx = tl.arange(0, M)
    offy = tl.arange(0, N) * M
    off2d = offx[None, :] + offy[:, None]
    x = tl.load(X + off2d)
    x = al.sort(x, descending=descending, dim=1)
    tl.store(Z + off2d, x)
```

#### flip ####

Ascend supports the flip operation on the input tensor along the specified dimension.

**Parameter Description**

| Parameter name | Type                       | Description       |
| -------------- | -------------------------- | ----------------- |
| `ptr`          | tensor                     | Input Tensor      |
| `dim`          | int or tl.constexpr\[int\] | Dimension to flip |

**Use Example**

```
flipped_feature = flip(input_tensor, dim=2)
```

#### cast ####

The Ascend supports the conversion of tensors to specified data types, including numerical conversion, bit conversion, and overflow processing.

**Parameter Description**

| Parameter name         | Type           | Description                                                  | Default value |
| ---------------------- | -------------- | ------------------------------------------------------------ | ------------- |
| `input`                | tensor         | Input Tensor                                                 | -             |
| `dtype`                | dtype          | Target Data Type                                             | -             |
| `fp_downcast_rounding` | str, optional  | Rounding mode when a floating point number is converted down | `None`        |
| `bitcast`              | bool, optional | Whether to perform bit conversion (not numeric conversion)   | `False`       |
| `overflow_mode`        | str, optional  | Overflow handling mode                                       | `None`        |

**Use Example**

```
@triton.jit
def cast_to_bool(output_ptr, x_ptr, dims: Dimensions, overflow_mode: tl.constexpr):
    xidx = tl.arange(0, dims.XB)
    yidx = tl.arange(0, dims.YB)
    zidx = tl.arange(0, dims.ZB)
    idx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]

    X = tl.load(x_ptr + idx)
    overflow_mode = "trunc" if overflow_mode == 0 else "saturate"
    ret = tl.cast(X, dtype=tl.int1, overflow_mode=overflow_mode)

    tl.store(output_ptr + idx, ret)
```

### Indexing and Collection Operations ###

#### _index_select ####

The Ascend collects data in specified dimensions based on the index UB tensor in the source GM tensor and uses the SIMT template to collect values in the output UB tensor. This operation supports 2D-5D tensors.

**Parameter Description**

| Parameter name    | Type         | Description                                          |
| ----------------- | ------------ | ---------------------------------------------------- |
| `src`             | pointer type | Source Tensor Pointer (in GM)                        |
| `index`           | tensor       | Index tensor for collection (on UB)                  |
| `dim`             | int          | Dimension along which the collection takes place     |
| `bound`           | int          | Upper boundary of the index value                    |
| `end_offset`      | int tuple    | End offset of each dimension of the index tensor.    |
| `start_offset`    | int tuple    | Start offset of each dimension of the source tensor. |
| `src_stride`      | int tuple    | Step size of each dimension of the source tensor.    |
| `other`(Optional) | scalar value | Default value when index is out of bounds (on UB)    |
| `out`             | tensor       | Output Tensor (in UB)                                |

**Use Example**

```
_index_select(
    src=src_3d_ptr,
    index=index_2d_tile,
    dim=1,
    bound=50,
    end_offset=(2, 4, 64),
    start_offset=(0, 8, 0),
    src_stride=(256, 64, 1),
    other=0.0
)
```

#### index_put ####

The Ascend supports placing the value tensor in the target tensor based on the index tensor.

**Parameter Description**

| Parameter name   | Type                  | Description                                           |
| ---------------- | --------------------- | ----------------------------------------------------- |
| `ptr`            | Tensor (pointer type) | Target Tensor Pointer (in GM)                         |
| `index`          | tensor                | Index for placement (on UB)                           |
| `value`          | tensor                | Value to store (on UB)                                |
| `dim`            | int32                 | Dimension along which the index is placed             |
| `index_boundary` | int64                 | Upper boundary of the index value                     |
| `end_offset`     | int tuple             | End offset of the placement area in each dimension.   |
| `start_offset`   | int tuple             | Start offset of the placement area of each dimension. |
| `dst_stride`     | int tuple             | Step size of each dimension of the target tensor.     |

**Index Placement Rules**

 *  2D Index Placement
    
     *  dim = 0:`out[index[i]][start_offset[1]:end_offset[1]] = value[i][0:end_offset[1]-start_offset[1]]`
 *  3D Index Placement
    
     *  dim = 0:`out[index[i]][start_offset[1]:end_offset[1]][start_offset[2]:end_offset[2]] = value[i][0:end_offset[1]-start_offset[1]][0:end_offset[2]-start_offset[2]]`
     *  dim = 1:`out[start_offset[0]:end_offset[0]][index[j]][start_offset[2]:end_offset[2]] = value[0:end_offset[0]-start_offset[0]][j][0:end_offset[2]-start_offset[2]]`

**Constraints**

 *  `ptr`And to the`value`Must have the same rank
 *  `ptr.dtype`Currently, only the`float16`,`bfloat16`,`float32`
 *  `index`Must be an integer tensor. If`index.rank`! = 1, will be remodeled as 1D
 *  `index.numel`Must be equal to`value.shape[dim]`
 *  `value`Supports two to five-dimensional tensors.
 *  `dim`Must be valid (0 ≤ dim < rank(value) - 1)

**Use Example**

```
index_put(
    ptr=dst_ptr,
    index=index_tile,
    value=value_tile,
    dim=0,
    index_boundary=4,
    end_offset=(2, 2),
    start_offset=(0, 0),
    dst_stride=(2, 1)
)
```

#### gather_out_to_ub ####

Ascend can collect data from scatterpoints in the GM and save the data to the UB in a specified dimension. This operation supports index boundary check, ensuring efficient and secure data transfer.

**Parameter Description**

| Parameter name   | Type                    | Description                                         |
| ---------------- | ----------------------- | --------------------------------------------------- |
| `src`            | Tensor (pointer type)   | Source Tensor Pointer (in GM)                       |
| `index`          | tensor                  | Index tensor for collection (on UB)                 |
| `index_boundary` | int64                   | Indicates the upper boundary of the index value.    |
| `dim`            | int32                   | Dimension along which the collection takes place    |
| `src_stride`     | int64 tuple             | Step size of each dimension of the source tensor.   |
| `end_offset`     | int32 tuple             | End offset of each dimension of the index tensor.   |
| `start_offset`   | int32 tuple             | Start offset of each dimension of the index tensor. |
| `other`          | Scalar Value (Optional) | Default value used when index out of bounds (on UB) |

**Return Value**

 *  **Type: tensor**
 *  **Description: Result tensor in UB, shape vs.**`index.shape`Same as

**Scatter collection rule**

 *  One-dimensional index collection
    
     *  dim = 0:`out[i] = src[start_offset[0] + index[i]]`
 *  2D Index Collection
    
     *  dim = 0:`out[i][j] = src[start_offset[0] + index[i][j]][start_offset[1] + j]`
     *  dim = 1:`out[i][j] = src[start_offset[0] + i][start_offset[1] + index[i][j]]`
 *  3D Index Collection
    
     *  dim = 0:`out[i][j][k] = src[start_offset[0] + index[i][j][k]][start_offset[1] + j][start_offset[2] + k]`
     *  dim = 1:`out[i][j][k] = src[start_offset[0] + i][start_offset[1] + index[i][j][k]][start_offset[2] + k]`
     *  dim = 2:`out[i][j][k] = src[start_offset[0] + i][start_offset[1] + j][start_offset[2] + index[i][j][k]]`

**Constraints**

 *  `src`And to the`index`Must have the same rank
 *  `src.dtype`Currently, only the`float16`,`bfloat16`,`float32`
 *  `index`Must be an integer tensor with a rank between 1 and 5
 *  `dim`Must be valid (0 ≤ dim < rank(index))
 *  `other`Must be a scalar value
 *  For each not equal to`dim`Dimension of`i`,`index.size[i]`≤`src.size[i]`
 *  Output Shape vs.`index.shape`Same. if`index`None, the output tensor will be the same as the`index`Empty tensors of the same shape

**Use Example**

```
gathered = gather_out_to_ub(
    src=src_ptr,
    index=index,
    index_boundary=4,
    dim=0,
    src_stride=(2, 1),
    end_offset=(2, 2),
    start_offset=(0, 0)
)
```

#### scatter_ub_to_out ####

Ascend stores data from scatterpoints in UB to GM along the specified dimension. This operation supports index boundary check, ensuring efficient and secure data transfer.

**Parameter Description**

| Parameter name   | Type                  | Description                                         |
| ---------------- | --------------------- | --------------------------------------------------- |
| `ptr`            | Tensor (pointer type) | Target Tensor Pointer (in GM)                       |
| `value`          | tensor                | Block value to store (on UB)                        |
| `index`          | tensor                | Index used for scatter storage (in UB)              |
| `index_boundary` | int64                 | Indicates the upper boundary of the index value.    |
| `dim`            | int32                 | Dimension along which scatter-stored                |
| `dst_stride`     | int64 tuple           | Step size of each dimension of the target tensor.   |
| `end_offset`     | int32 tuple           | End offset of each dimension of the index tensor.   |
| `start_offset`   | int32 tuple           | Start offset of each dimension of the index tensor. |

**Scatter storage rule**

 *  one-dimensional index scatter
    
     *  dim = 0:`out[start_offset[0] + index[i]] = value[i]`
 *  2D Index Scatter
    
     *  dim = 0:`out[start_offset[0] + index[i][j]][start_offset[1] + j] = value[i][j]`
     *  dim = 1:`out[start_offset[0] + i][start_offset[1] + index[i][j]] = value[i][j]`
 *  3D Index Scatter
    
     *  dim = 0:`out[start_offset[0] + index[i][j][k]][start_offset[1] + j][start_offset[2] + k] = value[i][j][k]`
     *  dim = 1:`out[start_offset[0] + i][start_offset[1] + index[i][j][k]][start_offset[2] + k] = value[i][j][k]`
     *  dim = 2:`out[start_offset[0] + i][start_offset[1] + j][start_offset[2] + index[i][j][k]] = value[i][j][k]`

**Constraints**

 *  `ptr`,`index`And to the`value`Must have the same rank
 *  `ptr.dtype`Currently, only the`float16`,`bfloat16`,`float32`
 *  `index`Must be an integer tensor with a rank between 1 and 5
 *  `dim`Must be valid (0 ≤ dim < rank(index))
 *  For each not equal to`dim`Dimension of`i`,`index.size[i]`≤`ptr.size[i]`
 *  Output Shape vs.`index.shape`Same. if`index`None, the output tensor will be the same as the`index`Empty tensors of the same shape

**Use Example**

```
scatter_ub_to_out(
    ptr=dst_ptr,
    value=value,
    index=index,
    index_boundary=4,
    dim=0,
    dst_stride=(2, 1),
    end_offset=(2, 2),
    start_offset=(0, 0)
)
```

#### index_select_simd ####

The Ascend supports parallel index selection. Data is directly loaded from the GM to the UB, implementing zero copy and efficient read.

**Parameter Description**

| Parameter name | Type                         | Description                                                     |
| -------------- | ---------------------------- | --------------------------------------------------------------- |
| `src`          | tensor (pointer type)        | Source Tensor Pointer (in GM)                                   |
| `dim`          | int or constexpr             | Dimension along which the index is selected                     |
| `index`        | tensor                       | One-dimensional tensor of the index to select (in UB)           |
| `src_shape`    | List\[Union\[int, tensor\]\] | Full shape of the source tensor (can be an integer or a tensor) |
| `src_offset`   | List\[Union\[int, tensor\]\] | Read start offset (can be an integer or a tensor)               |
| `read_shape`   | List\[Union\[int, tensor\]\] | Size to read (Block shape, which can be an integer or a tensor) |

**Constraints**

 *  `read_shape[dim]`Must be for the`-1`
 *  `src_offset[dim]`Can be used for`-1`(will be ignored)
 *  Boundary processing: when`src_offset + read_shape > src_shape`is automatically truncated to`src_shape`bordering
 *  **Not to be checked**`index`Indicates whether to contain out-of-bounds values.

**Return Value**

 *  **Return type: tensor**
 *  **Description: Resulting tensor in UB, whose shape**`dim`Dimension is replaced with`index`Length of

**Use Example**

```
@triton.jit
def index_select_extension_kernel(in_ptr, indices_ptr, out_ptr, dim: tl.constexpr,
                                   other_numel: tl.constexpr,
                                   g_stride: tl.constexpr, indice_length: tl.constexpr,
                                   g_block: tl.constexpr, g_block_sub: tl.constexpr, 
                                   other_block: tl.constexpr):
    g_begin = tl.program_id(0) * g_block
    for goffs in range(0, g_block, g_block_sub):
        g_idx = tl.arange(0, g_block_sub) + g_begin + goffs
        g_mask = g_idx < indice_length
        indices = tl.load(indices_ptr + g_idx, g_mask, other=0)
        
        for other_offset in range(0, g_stride, other_block):
            other_idx = tl.arange(0, other_block) + other_offset
            other_mask = other_idx < g_stride
            
            tmp_buf = al.index_select_simd(
                src=in_ptr,
                dim=dim,
                index=indices,
                src_shape=(other_numel, g_stride),
                src_offset=(-1, 0),
                read_shape=(-1, other_block)
            )
            
            tl.store(out_ptr + g_idx[:, None] * g_stride + other_idx[None, :],
                     tmp_buf, g_mask[:, None] & other_mask[None, :])
```

## Customized operations unique to Triton ##

Triton-Ascend's Custom Op allows users to customize their own actions and use it. Customization operations are converted into calling the implementation functions on the device side during running. The functions can call the existing library functions or the implementation functions generated by the source code or bytecode compilation provided by the user.

### Basic Usage ###

#### Registering Customized Operations ####

The functions related to customization operations are provided by the triton Ascend extension package. User-defined customization operations can be used only after registration. The customized operations can be provided by the extension package.`register_custom_op`Decorate a class to define and register the custom action:

```
import triton.language.extra.cann.extension as al

@al.register_custom_op
class my_custom_op:
    name = 'my_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
```

To register a simplest customization operation, at least the basic attributes such as name, core, pipe, and mode must be provided.

 *  name: operation name, which is the unique identifier of the custom operation. If the operation name is omitted, the class name is used by default.
 *  core indicates the Ascend core on which the AVP runs.
 *  pipe indicates a pipeline.
 *  mode indicates the programming mode used.

#### Use custom actions ####

Registered custom actions are available through the Ascend expansion pack`custom()`The function is invoked. The name and parameters of the customized operation must be provided.

```
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@triton.jit
def my_kernel(...):
    ...
    res = al.custom('my_custom_op', src, index, dim=0, out=dst)
    ...
```

`custom()`The parameters of the include the operation name, input parameters, and optional output parameters.

 *  Operation name, which must be the same as the registered operation name.
 *  Input parameters. Different operations have different input parameters.
 *  Output parameter (optional). The output parameter is defined by`out`Specifies the output of the operation.

If it's passed`out`If the parameter specifies the output variable, the return value of the customization operation is the same as the output variable. Otherwise, the return value of the operation is unavailable.

### Built-in Customization Operations ###

The name of the built-in customization operation starts with`"__builtin_"`Start with the customized operations built in triton-ascend and can be used without registration. For example:

```
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@triton.jit
def my_kernel(...):
    ...
    dst = tl.full(dst_shape, 0, tl.float32)
    x = al.custom('__builtin_indirect_load', src, index, mask, other, out=dst)
    ...
```

For details about the built-in customization operations, see the documentation of the related version.

### Parameter Validity Check ###

Without constraint, the user can give`al.custom()`If the number of parameters and parameter types are not the expected ones, an error occurs during the running.

To avoid this problem and improve the user experience of the customization operation, we can provide constructors for the registered customization class to describe the parameter list and check the parameter validity. For example:

```
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@al.register_custom_op
class my_custom_op:
    name = 'my_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT

    def __init__(self, src, index, dim, out=None):
        assert index.dtype.is_int(), "index must be an integer tensor"
        assert isinstance(dim, int), "dim must be an integer"
        assert out, "out is required"
        assert out.shape == index.shape, "out should have same shape as index"
        ...
```

The parameter list of the constructor function of the registration class is the parameter list required for invoking the customization operation. When invoking the custom operation, you need to provide the parameters that meet the requirements. For example:

```
res = al.custom('my_custom_op', src_ptr, index, dim=1, out=dst)
```

If the provided parameters are incorrect, an error will be reported during compilation. For example, the dim parameter must be an integer constant. If the dim parameter is a floating point number, the following error message is displayed:

```
...
    res = al.custom('my_custom_op', src_ptr, index, dim=1.0, out=dst)
          ^
AssertionError('dim must be an integer')
```

### Output Parameters and Return Values ###

`al.custom`The output parameter specified by the out parameter is returned. For example:

```
x = al.custom('my_custom_op', src, index, out=dst)
```

dst is returned to x.

The out parameter can specify multiple output parameters,`al.custom`Returns a tuple containing these output parameters:

```
x, y = al.custom('my_custom_op', src, index, out=(dst1, dst2))
```

dst1 is returned to x and dst2 is returned to y.

If the out parameter is not specified,`al.custom`No value is returned (None is returned).

### Symbolic name of the called function. ###

The customization operation will eventually be converted into the calling of the implementation function on the device side. We can register the custom action class`symbol`Property to configure the symbolic name of the function; if not set`symbol`Property, the name of the custom operation is used as the function name by default.

#### Static Symbol Name ####

If a custom operation always calls a device-side function, you can set the symbol name statically.

```
@al.register_custom_op
class my_custom_op:
    name = 'my_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    symbol = '_my_custom_op_symbol_name_'
```

In this way,`al.custom('my_custom_op', ...)`will fix the corresponding device side`_my_custom_op_symbol_name_(...)`function.

#### Dynamic symbol name ####

In most cases, the same customization operation needs to invoke different functions on the device side based on the dimension and type of the input parameter. In this case, the symbol name needs to be set dynamically. Similar to the parameter validity check, you can dynamically set the symbol name in the constructor of the registered custom operation class. For example:

```
@al.register_custom_op
class my_custom_op:
    name = 'my_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT

    def __init__(self, src, index, dim, out=None):
        ...
        self.symbol = f"my_func_{len(index.shape)}d_{src.dtype.element_ty.cname}_{index.dtype.cname}"
        ...
```

When the input src is a pointer pointing to the float32 type and the index is a 3-dimensional tensor of the int32 type, the device function symbol corresponding to the preceding customization operation is as follows:`"my_func_3d_float_int32_t"`; Different input parameters correspond to different symbol names.

Note that the type name is used here.`cname`\: indicates the name of the corresponding type in the AscendC language. For example, the cname corresponding to int32 is`int32_t`. Since we usually declare these functions as macros and embed the related type name into the function name,`cname`It's more common.

### Source code and compilation ###

If the functions that implement customization operations need to be compiled from source code or bytecode, configure the functions when registering the customization operation class.`source`And to the`compile`Attribute:

 *  source code or bytecode file path for implementing the customized operation function;
 *  The compile command implements the compilation command of the customized operation function. You can use the`%<`And to the`%@`Indicates the source file and target file, respectively (similar to Makefile).

Similar to symbol names, these two attributes can be configured statically or dynamically in the registration class constructor, for example:

```
@al.register_custom_op
class my_custom_op:
    name = 'my_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    ...
    source = "workspace/my_custom_op.cce"
    compile = "bisheng -std=c++17 -O2 -o $@ -c $<"
```

### Parameter Conversion Rule ###

#### Parameter Sequence ####

Customized operations are converted into corresponding function calls. The parameter sequence is the same as that on the Python side. The output parameter (out, if any) is always placed at the end. For example, the following Python code is used:

```
al.custom('my_custom_op', src, index, dim, out=dst)
```

Converting to a function call is equivalent to:

```
my_custom_op(src, index, dim, dst);
```

#### List and Tuple Parameters ####

The tuple or list parameter on the Python side is flattened. For example:

```
al.custom('my_custom_op', src, index, offsets=(1, 2, 3), out=dst)
```

When converted to a function call, the offsets parameter is flattened:

```
my_custom_op(src, index, 1, 2, 3, dst);
```

### Constant Parameter Type ###

Customization operations support the constant parameter types of integers and floating points. However, the integer and floating point types of Python do not distinguish the bit widths. Therefore, by default, only integers are mapped to the int32_t type and floating point numbers are mapped to the float type. When the constant parameter of the implementation function is of other type, for example, int64_t, the function signature does not match, causing errors.

For example, the implementation function signature for the following customized operations is available:

```
custom_op_impl_func(memref_t<...> *src, memref_t<...> *idx, int64_t bound);
```

The bound parameter must be an integer of the int64_t type.

When the customization operation is invoked on the Python side, the value of the bound constant parameter is provided.

```
al.custom('my_custom_op', src, idx, bound=1024)
```

Because the integer constants of Python do not distinguish the bit width, we can only map bound to int32_t by default. As a result, the signature does not match the implementation function and an error occurs.

To avoid this problem, you are advised to implement the function parameters. For integers, use int32_t, and for floating point numbers, use float. In some specific scenarios, we also provide the following methods to specify the specific type:

#### Specify the integer bit width by using al.int64. ####

By default, integer constants are mapped to the int32_t type. If the implementation function requires an int64_t type, you can use the`al.int64`Wrap integers, for example:

```
al.custom('my_custom_op', src, idx, bound=al.int64(1024))
```

#### Specify the type by type hint. ####

In the constructor function of the registered class, you can add type annotations to the corresponding parameters. For example:

```
@al.register_custom_op
class my_custom_op:
    name = 'my_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT

    def __init__(self, src, idx, bound: tl.int64):
        ...
```

In this way, the bound parameter is always mapped to the int64_t type.

#### Dynamically Specifying Parameter Types ####

Another extreme case is that the parameter type varies depending on the other parameters. For example, the bound type must be the same as the idx data type. In this case, you can use arg_type to dynamically specify the type in the constructor. For example:

```
@al.register_custom_op
class my_custom_op:
    name = 'my_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT

    def __init__(self, src, idx, bound):
        ...
        self.arg_type['bound'] = idx.dtype
```

### Encapsulation Customization Operations ###

Direct use`al.custom`Invoking a customized operation is a little troublesome, especially when there are output parameters. Before invoking the operation, prepare the output parameters. For example:

```
dst = tl.full(index.shape, 0, tl.float32)
x = al.custom('my_custom_op', src, index, out=dst)
```

A customized operation can be encapsulated as an operation function for easy use. For example:

```
@al.builtin
def my_custom_op(src, index, _builder=None):
    dst = tl.semantic.full(index.shape, 0, src.dtype.element_ty, _builder)
    return al.custom_semantic(_my_custom_op.name, src, index, out=dst, _builder=_builder)
```

Encapsulated operation functions need to be`al.builtin`Decorate, and pass through`al.custom_semantic`Invoke the customization operation. It's also possible to use`tl.semantic`Provide the function preparation output parameters; Note: When encapsulating the operation function, you need to give an additional`_builder`Parameter, and pass to all senmtic functions.

The encapsulated operation function can be directly invoked like the native operation.

```
@triton.jit
def my_kernel(...):
    ...
    x = my_custom_op(src, index)
    ...
```

## Extended Enumeration Unique to Triton ##

#### SYNC_IN_VF ####

| Enumerated Value | Description                                                                                               |
| ---------------- | --------------------------------------------------------------------------------------------------------- |
| `VV_ALL`         | Blocks execution of vector load/store instructions until all vector load/store instructions are complete  |
| `VST_VLD`        | Blocks execution of the vector load instruction until all vector store instructions are complete.         |
| `VLD_VST`        | Blocks execution of vector store instructions until all vector load instructions are complete.            |
| `VST_VST`        | Blocks execution of vector storage instructions until all vector storage instructions are complete        |
| `VS_ALL`         | Block execution of scalar load/store instructions until all vector load/store instructions are complete   |
| `VST_LD`         | Blocks execution of scalar load instructions until all vector store instructions are complete.            |
| `VLD_ST`         | Blocks execution of scalar store instructions until all vector load instructions are complete.            |
| `VST_ST`         | Blocks execution of scalar storage instructions until all vector storage instructions are complete.       |
| `SV_ALL`         | Blocks execution of vector load/store instructions until all scalar load/store instructions are complete. |
| `ST_VLD`         | Blocks execution of vector load instructions until all scalar store instructions are complete.            |
| `LD_VST`         | Blocks execution of vector store instructions until all scalar load instructions are complete.            |
| `ST_VST`         | Blocks execution of vector store instructions until all scalar store instructions are complete.           |

#### FixpipeDMAMode ####

| Enumerated Value | Description                                                                      |
| ---------------- | -------------------------------------------------------------------------------- |
| `NZ2DN`          | Data Conversion from Nonzero Storage Format to Column/Column Major Order Format  |
| `NZ2ND`          | Data Conversion from Nonzero Storage Format to Column and Row Major Format       |
| `NZ2NZ`          | Data conversion between nonzero storage formats (preserving the original format) |

#### FixpipeDualDstMode ####

| Enumerated Value | Description                                                               |
| ---------------- | ------------------------------------------------------------------------- |
| `NO_DUAL`        | Do not use dual target mode, data is written to a single target           |
| `COLUMN_SPLIT`   | Column split dual target mode, splitting data by column to two targets    |
| `ROW_SPLIT`      | Row splitting dual target mode, which splits data into two targets by row |

#### FixpipePreQuantMode ####

| Enumerated Value | Description                                                                        |
| ---------------- | ---------------------------------------------------------------------------------- |
| `NO_QUANT`       | Do not perform prequantization and retain the original data format.                |
| `F322BF16`       | Floating Point 32-bit to bfloat16 Format Quantization Conversion                   |
| `F322F16`        | Quantization Conversion from 32-bit Floating Point to 16-bit Floating Point Format |
| `S322I8`         | Signed 32-bit integer to 8-bit integer quantization conversion                     |

#### FixpipePreReluMode ####

| Enumerated Value | Description                                        |
| ---------------- | -------------------------------------------------- |
| `LEAKY_RELU`     | Processing of the Leaky ReLU activation function   |
| `NO_RELU`        | ReLU activation is not performed.                  |
| `NORMAL_RELU`    | Standard ReLU activation function processing       |
| `P_RELU`         | Processes the Parametric ReLU activation function. |

#### CORE ####

| Enumerated Value  | Description                   |
| ----------------- | ----------------------------- |
| `VECTOR`          | Vector core                   |
| `CUBE`            | Cube core                     |
| `CUBE_OR_VECTOR`  | Cube or Vector core (either)  |
| `CUBE_AND_VECTOR` | Cube and Vector cores (mixed) |

#### MODE ####

| Enumerated Value | Description                                     |
| ---------------- | ----------------------------------------------- |
| `SIMD`           | single instruction multiple data execution mode |
| `SIMT`           | single-instruction multi-thread execution mode  |
| `MIX`            | Hybrid Execution Mode                           |

#### PIPE ####

| Enumerated Value | Description                       |
| ---------------- | --------------------------------- |
| `PIPE_S`         | scalar computing pipeline         |
| `PIPE_V`         | vector computing pipeline         |
| `PIPE_M`         | memory operation pipeline         |
| `PIPE_MTE1`      | Memory transfer engine 1 pipeline |
| `PIPE_MTE2`      | Memory transfer engine 2 pipeline |
| `PIPE_MTE3`      | Memory transfer engine 3 pipeline |
| `PIPE_ALL`       | All pipelines                     |
| `PIPE_FIX`       | Fixed functional pipeline         |

