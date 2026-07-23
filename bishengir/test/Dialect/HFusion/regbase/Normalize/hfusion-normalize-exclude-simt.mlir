// RUN: bishengir-opt --hfusion-normalize-ops="use-regbase=true" %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-NOT: hfusion.cast
func.func @test_Normalize_bcst_i1(%arg0: tensor<64xi1>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi1> {
    %3 = scope.scope : () -> tensor<64x128xi1> {
        %1 = tensor.empty() : tensor<64x128xi1>
        %2 = linalg.broadcast ins(%arg0 : tensor<64xi1>) outs(%1 : tensor<64x128xi1>) dimensions = [1] 
        scope.return %2 : tensor<64x128xi1>
    } {noinline, vector_type = "simt"}
    return %3 : tensor<64x128xi1>
}
