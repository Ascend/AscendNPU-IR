# RUN: python3 %s | FileCheck %s

from bishengir.ir import *
from bishengir.passmanager import *

from bishengir._mlir_libs._bishengirRegisterEverything import *


def run(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        f()
    return f


# CHECK: ; ModuleID = 'LLVMDialectModule'
def compileCase1():
    module = Module.parse(
        r"""
module {
    func.func @func_arg_as_init(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>) -> tensor<1024xf32>
        attributes {hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
          %0 = tensor.empty() : tensor<1024xf32>
          %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
          %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %arg2 : tensor<1024xf32>, tensor<1024xf32>) outs(%arg3 : tensor<1024xf32>) -> tensor<1024xf32>
          return %2 : tensor<1024xf32>
    }
}
        """
    )

    pm = PassManager("builtin.module")
    pm.enable_ir_printing()
    pm.add("hacc-append-device-spec{target=Ascend910B1}")
    pm.add(
        "bishengir-compile{enable-hfusion-compile=true "
        "enable-hivm-compile=true "
        "enable-lir-compile=false}")
    pm.run(module.operation)

# CHECK: ; ModuleID = 'LLVMDialectModule'
def compileCaseDB():
    module = Module.parse(
        r"""
module {
    func.func @multi_buffer_use_parent_for(%in_gm: memref<16xf16, #hivm.address_space<gm>>) {
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c16 = arith.constant 16 : index

        scf.for %i0 = %c0 to %c16 step %c4 {
          %tmp_ub = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
          hivm.hir.load ins(%in_gm : memref<16xf16, #hivm.address_space<gm>>)
                        outs(%tmp_ub : memref<16xf16, #hivm.address_space<ub>> )

        }
        return
    }
}
        """
    )
    pm = PassManager("builtin.module")
    pm.enable_ir_printing()
    pm.add("hacc-append-device-spec{target=Ascend910B1}")
    pm.add(
        "bishengir-compile{enable-hfusion-compile=false "
        "enable-hivm-compile=true "
        "enable-lir-compile=false "
        "enable-auto-multi-buffer=true}")
    pm.run(module.operation)



# CHECK: ; ModuleID = 'LLVMDialectModule'
def compileCaseLong():
    module = Module.parse(
        r"""
module {
  func.func @Fused_AddN_Sqrt_fusion_13699286479483544998_hostapi_test_1_float32_evb_1981_aiv__kernel0(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>, %arg5: tensor<1xf32>, %arg6: tensor<1xf32>, %arg7: tensor<1xf32>, %arg8: tensor<1xf32>, %arg9: tensor<1xf32>, %arg10: tensor<1xf32>, %arg11: tensor<1xf32>, %arg12: tensor<1xf32>, %arg13: tensor<1xf32>, %arg14: tensor<1xf32>, %arg15: tensor<1xf32>, %arg16: tensor<1xf32>, %arg17: tensor<1xf32>, %arg18: tensor<1xf32>, %arg19: tensor<1xf32>, %arg20: tensor<1xf32>, %arg21: tensor<1xf32>, %arg22: tensor<1xf32>, %arg23: tensor<1xf32>, %arg24: tensor<1xf32>, %arg25: tensor<1xf32>, %arg26: tensor<1xf32>, %arg27: tensor<1xf32>, %arg28: tensor<1xf32>, %arg29: tensor<1xf32>, %arg30: tensor<1xf32>, %arg31: tensor<1xf32>, %arg32: tensor<1xf32>, %arg33: tensor<1xf32>, %arg34: tensor<1xf32>, %arg35: tensor<1xf32>, %arg36: tensor<1xf32>, %arg37: tensor<1xf32>, %arg38: tensor<1xf32>, %arg39: tensor<1xf32>, %arg40: tensor<1xf32>, %arg41: tensor<1xf32>, %arg42: tensor<1xf32>, %arg43: tensor<1xf32>, %arg44: tensor<1xf32>, %arg45: tensor<1xf32>, %arg46: tensor<1xf32>, %arg47: tensor<1xf32>, %arg48: tensor<1xf32>) -> tensor<1xf32> attributes {OperatorType = "Elementwise", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<1xf32>, tensor<1xf32>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = tensor.empty() : tensor<1xf32>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %arg2 : tensor<1xf32>, tensor<1xf32>) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
    %4 = tensor.empty() : tensor<1xf32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %arg3 : tensor<1xf32>, tensor<1xf32>) outs(%4 : tensor<1xf32>) -> tensor<1xf32>
    %6 = tensor.empty() : tensor<1xf32>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%5, %arg4 : tensor<1xf32>, tensor<1xf32>) outs(%6 : tensor<1xf32>) -> tensor<1xf32>
    %8 = tensor.empty() : tensor<1xf32>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%7, %arg5 : tensor<1xf32>, tensor<1xf32>) outs(%8 : tensor<1xf32>) -> tensor<1xf32>
    %10 = tensor.empty() : tensor<1xf32>
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%9, %arg6 : tensor<1xf32>, tensor<1xf32>) outs(%10 : tensor<1xf32>) -> tensor<1xf32>
    %12 = tensor.empty() : tensor<1xf32>
    %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%11, %arg7 : tensor<1xf32>, tensor<1xf32>) outs(%12 : tensor<1xf32>) -> tensor<1xf32>
    %14 = tensor.empty() : tensor<1xf32>
    %15 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%13, %arg8 : tensor<1xf32>, tensor<1xf32>) outs(%14 : tensor<1xf32>) -> tensor<1xf32>
    %16 = tensor.empty() : tensor<1xf32>
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%15, %arg9 : tensor<1xf32>, tensor<1xf32>) outs(%16 : tensor<1xf32>) -> tensor<1xf32>
    %18 = tensor.empty() : tensor<1xf32>
    %19 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%17, %arg10 : tensor<1xf32>, tensor<1xf32>) outs(%18 : tensor<1xf32>) -> tensor<1xf32>
    %20 = tensor.empty() : tensor<1xf32>
    %21 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%19, %arg11 : tensor<1xf32>, tensor<1xf32>) outs(%20 : tensor<1xf32>) -> tensor<1xf32>
    %22 = tensor.empty() : tensor<1xf32>
    %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%21, %arg12 : tensor<1xf32>, tensor<1xf32>) outs(%22 : tensor<1xf32>) -> tensor<1xf32>
    %24 = tensor.empty() : tensor<1xf32>
    %25 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%23, %arg13 : tensor<1xf32>, tensor<1xf32>) outs(%24 : tensor<1xf32>) -> tensor<1xf32>
    %26 = tensor.empty() : tensor<1xf32>
    %27 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%25, %arg14 : tensor<1xf32>, tensor<1xf32>) outs(%26 : tensor<1xf32>) -> tensor<1xf32>
    %28 = tensor.empty() : tensor<1xf32>
    %29 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%27, %arg15 : tensor<1xf32>, tensor<1xf32>) outs(%28 : tensor<1xf32>) -> tensor<1xf32>
    %30 = tensor.empty() : tensor<1xf32>
    %31 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%29, %arg16 : tensor<1xf32>, tensor<1xf32>) outs(%30 : tensor<1xf32>) -> tensor<1xf32>
    %32 = tensor.empty() : tensor<1xf32>
    %33 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%31, %arg17 : tensor<1xf32>, tensor<1xf32>) outs(%32 : tensor<1xf32>) -> tensor<1xf32>
    %34 = tensor.empty() : tensor<1xf32>
    %35 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%33, %arg18 : tensor<1xf32>, tensor<1xf32>) outs(%34 : tensor<1xf32>) -> tensor<1xf32>
    %36 = tensor.empty() : tensor<1xf32>
    %37 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%35, %arg19 : tensor<1xf32>, tensor<1xf32>) outs(%36 : tensor<1xf32>) -> tensor<1xf32>
    %38 = tensor.empty() : tensor<1xf32>
    %39 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%37, %arg20 : tensor<1xf32>, tensor<1xf32>) outs(%38 : tensor<1xf32>) -> tensor<1xf32>
    %40 = tensor.empty() : tensor<1xf32>
    %41 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%39, %arg21 : tensor<1xf32>, tensor<1xf32>) outs(%40 : tensor<1xf32>) -> tensor<1xf32>
    %42 = tensor.empty() : tensor<1xf32>
    %43 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%41, %arg22 : tensor<1xf32>, tensor<1xf32>) outs(%42 : tensor<1xf32>) -> tensor<1xf32>
    %44 = tensor.empty() : tensor<1xf32>
    %45 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%43, %arg23 : tensor<1xf32>, tensor<1xf32>) outs(%44 : tensor<1xf32>) -> tensor<1xf32>
    %46 = tensor.empty() : tensor<1xf32>
    %47 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%45, %arg24 : tensor<1xf32>, tensor<1xf32>) outs(%46 : tensor<1xf32>) -> tensor<1xf32>
    %48 = tensor.empty() : tensor<1xf32>
    %49 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%47, %arg25 : tensor<1xf32>, tensor<1xf32>) outs(%48 : tensor<1xf32>) -> tensor<1xf32>
    %50 = tensor.empty() : tensor<1xf32>
    %51 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%49, %arg26 : tensor<1xf32>, tensor<1xf32>) outs(%50 : tensor<1xf32>) -> tensor<1xf32>
    %52 = tensor.empty() : tensor<1xf32>
    %53 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%51, %arg27 : tensor<1xf32>, tensor<1xf32>) outs(%52 : tensor<1xf32>) -> tensor<1xf32>
    %54 = tensor.empty() : tensor<1xf32>
    %55 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%53, %arg28 : tensor<1xf32>, tensor<1xf32>) outs(%54 : tensor<1xf32>) -> tensor<1xf32>
    %56 = tensor.empty() : tensor<1xf32>
    %57 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%55, %arg29 : tensor<1xf32>, tensor<1xf32>) outs(%56 : tensor<1xf32>) -> tensor<1xf32>
    %58 = tensor.empty() : tensor<1xf32>
    %59 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%57, %arg30 : tensor<1xf32>, tensor<1xf32>) outs(%58 : tensor<1xf32>) -> tensor<1xf32>
    %60 = tensor.empty() : tensor<1xf32>
    %61 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%59, %arg31 : tensor<1xf32>, tensor<1xf32>) outs(%60 : tensor<1xf32>) -> tensor<1xf32>
    %62 = tensor.empty() : tensor<1xf32>
    %63 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%61, %arg32 : tensor<1xf32>, tensor<1xf32>) outs(%62 : tensor<1xf32>) -> tensor<1xf32>
    %64 = tensor.empty() : tensor<1xf32>
    %65 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%63, %arg33 : tensor<1xf32>, tensor<1xf32>) outs(%64 : tensor<1xf32>) -> tensor<1xf32>
    %66 = tensor.empty() : tensor<1xf32>
    %67 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%65, %arg34 : tensor<1xf32>, tensor<1xf32>) outs(%66 : tensor<1xf32>) -> tensor<1xf32>
    %68 = tensor.empty() : tensor<1xf32>
    %69 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%67, %arg35 : tensor<1xf32>, tensor<1xf32>) outs(%68 : tensor<1xf32>) -> tensor<1xf32>
    %70 = tensor.empty() : tensor<1xf32>
    %71 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%69, %arg36 : tensor<1xf32>, tensor<1xf32>) outs(%70 : tensor<1xf32>) -> tensor<1xf32>
    %72 = tensor.empty() : tensor<1xf32>
    %73 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%71, %arg37 : tensor<1xf32>, tensor<1xf32>) outs(%72 : tensor<1xf32>) -> tensor<1xf32>
    %74 = tensor.empty() : tensor<1xf32>
    %75 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%73, %arg38 : tensor<1xf32>, tensor<1xf32>) outs(%74 : tensor<1xf32>) -> tensor<1xf32>
    %76 = tensor.empty() : tensor<1xf32>
    %77 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%75, %arg39 : tensor<1xf32>, tensor<1xf32>) outs(%76 : tensor<1xf32>) -> tensor<1xf32>
    %78 = tensor.empty() : tensor<1xf32>
    %79 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%77, %arg40 : tensor<1xf32>, tensor<1xf32>) outs(%78 : tensor<1xf32>) -> tensor<1xf32>
    %80 = tensor.empty() : tensor<1xf32>
    %81 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%79, %arg41 : tensor<1xf32>, tensor<1xf32>) outs(%80 : tensor<1xf32>) -> tensor<1xf32>
    %82 = tensor.empty() : tensor<1xf32>
    %83 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%81, %arg42 : tensor<1xf32>, tensor<1xf32>) outs(%82 : tensor<1xf32>) -> tensor<1xf32>
    %84 = tensor.empty() : tensor<1xf32>
    %85 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%83, %arg43 : tensor<1xf32>, tensor<1xf32>) outs(%84 : tensor<1xf32>) -> tensor<1xf32>
    %86 = tensor.empty() : tensor<1xf32>
    %87 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%85, %arg44 : tensor<1xf32>, tensor<1xf32>) outs(%86 : tensor<1xf32>) -> tensor<1xf32>
    %88 = tensor.empty() : tensor<1xf32>
    %89 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%87, %arg45 : tensor<1xf32>, tensor<1xf32>) outs(%88 : tensor<1xf32>) -> tensor<1xf32>
    %90 = tensor.empty() : tensor<1xf32>
    %91 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%89, %arg46 : tensor<1xf32>, tensor<1xf32>) outs(%90 : tensor<1xf32>) -> tensor<1xf32>
    %92 = tensor.empty() : tensor<1xf32>
    %93 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%91, %arg47 : tensor<1xf32>, tensor<1xf32>) outs(%92 : tensor<1xf32>) -> tensor<1xf32>
    %94 = tensor.empty() : tensor<1xf32>
    %95 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%93, %arg48 : tensor<1xf32>, tensor<1xf32>) outs(%94 : tensor<1xf32>) -> tensor<1xf32>
    %96 = tensor.empty() : tensor<1xf32>
    %97 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%95 : tensor<1xf32>) outs(%96 : tensor<1xf32>) -> tensor<1xf32>
    return %97 : tensor<1xf32>
  }
}
        """
    )

    pm = PassManager("builtin.module")
    pm.enable_ir_printing()
    pm.add("hacc-append-device-spec{target=Ascend910B1}")
    pm.add(
        "bishengir-compile{enable-hfusion-compile=true "
        "enable-hivm-compile=true "
        "enable-lir-compile=false}")
    pm.run(module.operation)


# CHECK-LABEL: testCompilePass
@run
def testCompilePass():
    with Context() as ctx:
        register_dialects(ctx)
        register_translations(ctx)

        compileCase1()
        compileCaseDB()
        compileCaseLong()
