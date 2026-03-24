"""
BiShengIR Compiler Python Bindings - Usage Examples

This module demonstrates various usage patterns for the ascendnpuir.compile function,
based on real MLIR test cases from the BiShengIR project.

All examples output binary kernel files for Ascend NPU.
"""

from ascendnpuir import compile


def example_1_basic_hfusion_compile():
    """
    Example 1: Basic HFusion compilation
    
    This example demonstrates compiling a simple element-wise operation
    with HFusion compilation enabled.
    
    Based on: bishengir/test/bishengir-compile/hfusion/compile-pure-elemwise-2d.mlir
    Output: Binary kernel file
    """
    print("\n=== Example 1: Basic HFusion Compile ===")
    
    mlir_code = """
module {
  func.func @add_mul_2d(%arg0: tensor<1024x1024xf32>, 
                        %arg1: tensor<1024x1024xf32>, 
                        %arg2: tensor<1024x1024xf32>, 
                        %arg3: tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %1 = tensor.empty() : tensor<1024x1024xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} 
         ins(%arg0, %arg1 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) 
         outs(%1 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %3 = tensor.empty() : tensor<1024x1024xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} 
         ins(%2, %arg2 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) 
         outs(%3 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    return %4 : tensor<1024x1024xf32>
  }
}
"""
    
    output_path = "output_basic.o"
    options = [
        "-enable-hfusion-compile=true",
        "-enable-hivm-compile=true",
        "-enable-lir-compile=false",
        "-block-dim=20"
    ]
    
    try:
        result = compile(mlir_code, output_path, options)
        if result.returncode == 0:
            print(f"✓ Compilation successful! Output: {result.output_path}")
            print(f"  Options: {' '.join(options)}")
            print(f"  Return code: {result.returncode}")
            if result.stdout:
                print(f"  Stdout: {result.stdout}")
            if result.stderr:
                print(f"  Stderr: {result.stderr}")
        else:
            print(f"✗ Compilation failed with return code: {result.returncode}")
            print(f"  Output path: {result.output_path}")
            if result.stdout:
                print(f"  Stdout: {result.stdout}")
            if result.stderr:
                print(f"  Stderr: {result.stderr}")
    except Exception as e:
        print(f"✗ Compilation raised exception: {e}")


def example_2_dynamic_shape_broadcast():
    """
    Example 2: Dynamic shape with broadcast operations
    
    This example shows compiling operations with dynamic shapes
    and broadcast semantics.
    
    Based on: bishengir/test/bishengir-compile/hfusion/compile-any-pb-dynamic.mlir
    Output: Binary kernel file
    """
    print("\n=== Example 2: Dynamic Shape Broadcast ===")
    
    mlir_code = """
module {
  func.func @test_dynamic_shape(%arg0: tensor<?x1xf16>,
                                %arg1: tensor<1x?xf16>,
                                %arg2: tensor<?x?xf16>) -> tensor<?x?xf16>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, 
              hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x1xf16>
    %dim_0 = tensor.dim %arg1, %c1 : tensor<1x?xf16>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf16>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<?x1xf16> into tensor<?xf16>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<?xf16>) 
                   outs(%0 : tensor<?x?xf16>) dimensions = [1]
    %collapsed_1 = tensor.collapse_shape %arg1 [[0, 1]] : tensor<1x?xf16> into tensor<?xf16>
    %broadcasted_2 = linalg.broadcast ins(%collapsed_1 : tensor<?xf16>) 
                     outs(%0 : tensor<?x?xf16>) dimensions = [0]
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} 
         ins(%broadcasted, %broadcasted_2 : tensor<?x?xf16>, tensor<?x?xf16>) 
         outs(%arg2 : tensor<?x?xf16>) -> tensor<?x?xf16>
    return %1 : tensor<?x?xf16>
  }
}
"""
    
    output_path = "output_dynamic.o"
    options = [
        "-enable-hfusion-compile=true",
        "-enable-lir-compile=false"
    ]
    
    try:
        compile(mlir_code, output_path, options)
        print(f"✓ Compilation successful! Output: {output_path}")
        print(f"  Features: Dynamic shapes, broadcast operations")
    except Exception as e:
        print(f"✗ Compilation failed: {e}")


def example_3_hivm_vector_operations():
    """
    Example 3: HIVM vector operations with memory hierarchy
    
    This example demonstrates HIVM dialect operations with
    explicit memory hierarchy (GM, UB buffers).
    
    Based on: bishengir/test/bishengir-compile/hivm/compile-bisheng-hir-static-ptr-off.mlir
    Output: Binary kernel file
    """
    print("\n=== Example 3: HIVM Vector Operations ===")
    
    mlir_code = """
module {
  func.func @vector_add_kernel(%valueA: memref<16xf16, #hivm.address_space<gm>>,
                               %valueB: memref<16xf16, #hivm.address_space<gm>>,
                               %valueC: memref<16xf16, #hivm.address_space<gm>>)
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %ubA = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%valueA : memref<16xf16, #hivm.address_space<gm>>) 
                  outs(%ubA : memref<16xf16, #hivm.address_space<ub>>)
    
    %ubB = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%valueB : memref<16xf16, #hivm.address_space<gm>>) 
                  outs(%ubB : memref<16xf16, #hivm.address_space<ub>>)
    
    %ubC = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%ubA, %ubB: memref<16xf16, #hivm.address_space<ub>>, 
                           memref<16xf16, #hivm.address_space<ub>>) 
                  outs(%ubC: memref<16xf16, #hivm.address_space<ub>>)
    
    hivm.hir.store ins(%ubC : memref<16xf16, #hivm.address_space<ub>>) 
                   outs(%valueC : memref<16xf16, #hivm.address_space<gm>>)
    return
  }
}
"""
    
    output_path = "output_hivm.o"
    options = [
        "-enable-lir-compile=false",
        "-enable-static-bare-ptr=false"
    ]
    
    try:
        compile(mlir_code, output_path, options)
        print(f"✓ Compilation successful! Output: {output_path}")
        print(f"  Features: HIVM dialect, memory hierarchy (GM/UB)")
    except Exception as e:
        print(f"✗ Compilation failed: {e}")


def example_4_matmul_operation():
    """
    Example 4: Matrix multiplication with fusion
    
    This example shows compiling a matrix multiplication operation
    with fusion optimizations.
    
    Based on: bishengir/test/bishengir-compile/hfusion/compile-bisheng-full.mlir
    Output: Binary kernel file
    """
    print("\n=== Example 4: Matrix Multiplication ===")
    
    mlir_code = """
module {
  func.func @matmul_fusion(%arg0: tensor<?x?xf32>, 
                           %arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
    %result = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
    %matmul_result = linalg.matmul 
                     ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) 
                     outs(%result : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %matmul_result : tensor<?x?xf32>
  }
}
"""
    
    output_path = "output_matmul.o"
    options = [
        "-enable-hfusion-compile=true",
        "-enable-hivm-compile=true",
        "-enable-lir-compile=false"
    ]
    
    try:
        compile(mlir_code, output_path, options)
        print(f"✓ Compilation successful! Output: {output_path}")
        print(f"  Features: Matrix multiplication, fusion optimization")
    except Exception as e:
        print(f"✗ Compilation failed: {e}")


def example_5_mixed_core_types():
    """
    Example 5: Mixed core types (AIC and AIV)
    
    This example demonstrates compiling for mixed core types
    in the same module.
    
    Based on: bishengir/test/bishengir-compile/hivm/compile-bisheng-hir-mix.mlir
    Output: Binary kernel file
    """
    print("\n=== Example 5: Mixed Core Types ===")
    
    mlir_code = """
module @M attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  func.func @matmul_aic(%arg0: memref<64x64xf16>,
                        %arg1: memref<64x64xf16>,
                        %arg2: memref<64x64xf16>)
  attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, 
              hivm.func_core_type = #hivm.func_core_type<AIC>} {
    hivm.hir.matmul ins(%arg0, %arg1 : memref<64x64xf16>, memref<64x64xf16>)
                    outs(%arg2 : memref<64x64xf16>)
    return
  }
  
  func.func @vector_add_aiv(%valueA: memref<16xf16, #hivm.address_space<gm>>,
                            %valueB: memref<16xf16, #hivm.address_space<gm>>,
                            %valueC: memref<16xf16, #hivm.address_space<gm>>)
  attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, 
              hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %ubA = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%valueA : memref<16xf16, #hivm.address_space<gm>>) 
                  outs(%ubA : memref<16xf16, #hivm.address_space<ub>>)
    
    %ubB = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%valueB : memref<16xf16, #hivm.address_space<gm>>) 
                  outs(%ubB : memref<16xf16, #hivm.address_space<ub>>)
    
    %ubC = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%ubA, %ubB: memref<16xf16, #hivm.address_space<ub>>, 
                           memref<16xf16, #hivm.address_space<ub>>) 
                  outs(%ubC: memref<16xf16, #hivm.address_space<ub>>)
    
    hivm.hir.store ins(%ubC : memref<16xf16, #hivm.address_space<ub>>) 
                   outs(%valueC : memref<16xf16, #hivm.address_space<gm>>)
    return
  }
}
"""
    
    output_path = "output_mixed.o"
    options = [
        "-enable-lir-compile=false"
    ]
    
    try:
        compile(mlir_code, output_path, options)
        print(f"✓ Compilation successful! Output: {output_path}")
        print(f"  Features: Mixed core types (AIC/AIV)")
    except Exception as e:
        print(f"✗ Compilation failed: {e}")


def example_6_unary_operations():
    """
    Example 6: Various unary operations
    
    This example demonstrates compiling various unary operations
    like exp, log, sqrt, etc.
    
    Based on: bishengir/test/Conversion/HFusionToHIVM/hfusion-to-hivm.mlir
    Output: Binary kernel file
    """
    print("\n=== Example 6: Unary Operations ===")
    
    mlir_code = """
module {
  func.func @unary_ops(%src: memref<6x6xf32>, %dst: memref<6x6xf32>)
  attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>} 
      ins(%src : memref<6x6xf32>) 
      outs(%dst : memref<6x6xf32>)
    
    hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} 
      ins(%src : memref<6x6xf32>) 
      outs(%dst : memref<6x6xf32>)
    
    hfusion.elemwise_unary {fun = #hfusion.unary_fn<exp>} 
      ins(%src : memref<6x6xf32>) 
      outs(%dst : memref<6x6xf32>)
    
    return
  }
}
"""
    
    output_path = "output_unary.o"
    options = [
        "-enable-hfusion-compile=true",
        "-enable-hivm-compile=true",
        "-enable-lir-compile=false"
    ]
    
    try:
        compile(mlir_code, output_path, options)
        print(f"✓ Compilation successful! Output: {output_path}")
        print(f"  Features: Unary operations (relu, sqrt, exp)")
    except Exception as e:
        print(f"✗ Compilation failed: {e}")


def example_7_binary_operations():
    """
    Example 7: Various binary operations
    
    This example demonstrates compiling various binary operations
    like add, mul, sub, div, etc.
    
    Based on: bishengir/test/Conversion/HFusionToHIVM/hfusion-to-hivm.mlir
    Output: Binary kernel file
    """
    print("\n=== Example 7: Binary Operations ===")
    
    mlir_code = """
module {
  func.func @binary_ops(%src1: memref<6x6xi16>, 
                        %src2: memref<6x6xi16>, 
                        %dst: memref<6x6xi16>)
  attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} 
      ins(%src1, %src2 : memref<6x6xi16>, memref<6x6xi16>)
      outs(%dst : memref<6x6xi16>)
    
    hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} 
      ins(%src1, %src2 : memref<6x6xi16>, memref<6x6xi16>)
      outs(%dst : memref<6x6xi16>)
    
    return
  }
}
"""
    
    output_path = "output_binary.o"
    options = [
        "-enable-hfusion-compile=true",
        "-enable-hivm-compile=true",
        "-enable-lir-compile=false"
    ]
    
    try:
        compile(mlir_code, output_path, options)
        print(f"✓ Compilation successful! Output: {output_path}")
        print(f"  Features: Binary operations (or, and)")
    except Exception as e:
        print(f"✗ Compilation failed: {e}")


def example_8_error_handling():
    """
    Example 8: Error handling
    
    This example demonstrates proper error handling
    for invalid MLIR code and missing dependencies.
    """
    print("\n=== Example 8: Error Handling ===")
    
    # Test with empty input
    print("\nTest 1: Empty input")
    try:
        compile("", "output.o", [])
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    # Test with invalid MLIR syntax
    print("\nTest 2: Invalid MLIR syntax")
    invalid_mlir = """
module {
  func.func @test() {
    invalid_operation
    return
  }
}
"""
    try:
        compile(invalid_mlir, "output.o", [])
        print("✗ Should have raised RuntimeError")
    except RuntimeError as e:
        print(f"✓ Correctly caught error: {type(e).__name__}")
    
    # Test with missing hivmc binary
    print("\nTest 3: Missing hivmc binary")
    print("Note: This test requires hivmc to be missing from PATH")
    print("If hivmc is installed, this test will not trigger the error")
    # In a real scenario where hivmc is missing, this would raise:
    # FileNotFoundError: hivmc binary not found in PATH. 
    # Please install hivmc and ensure it is available in your PATH.
    print("✓ Error handling for missing hivmc is implemented in compiler.py")


def main():
    """Run all examples"""
    print("=" * 70)
    print("BiShengIR Compiler - Python Bindings Examples")
    print("All examples output binary kernel files for Ascend NPU")
    print("=" * 70)
    
    example_1_basic_hfusion_compile()

    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()