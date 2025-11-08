# RUN: python3 %s | FileCheck %s

from bishengir.ir import *
from bishengir.passmanager import *

from bishengir._mlir_libs._bishengirRegisterEverything import register_dialects


def run(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        f()
    return f


str_case = r"""
module {
  func.func @test_basic__kernel0(%valueA: memref<16xf16, #hivm.address_space<gm>>,
                                 %valueB: memref<16xf16, #hivm.address_space<gm>>,
                                 %valueC: memref<16xf16, #hivm.address_space<gm>>)
                                 attributes {hacc.entry}
  {
    %ubA = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%valueA : memref<16xf16, #hivm.address_space<gm>>) outs(%ubA : memref<16xf16, #hivm.address_space<ub>>)

    %ubB = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%valueB : memref<16xf16, #hivm.address_space<gm>>) outs(%ubB : memref<16xf16, #hivm.address_space<ub>>)

    %ubC = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%ubA, %ubB: memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>) outs(%ubC: memref<16xf16, #hivm.address_space<ub>>)

    hivm.hir.store ins(%ubC : memref<16xf16, #hivm.address_space<ub>>) outs(%valueC : memref<16xf16, #hivm.address_space<gm>>)
    return
  }
}
    """

# CHECK-LABEL: testHIVMPipeline
@run
def testHIVMPipeline():
    with Context() as ctx:
        register_dialects(ctx)
        module = Module.parse(str_case)
        pm = PassManager("builtin.module")
        pm.enable_ir_printing()
        pm.add("optimize-hivm-pipeline")
        pm.run(module.operation)
        print('--- module: ', module)


# CHECK-LABEL: testHIVMPlanMemory
@run
def testHIVMPlanMemory():
    with Context() as ctx:
        register_dialects(ctx)
        module = Module.parse(str_case)
        pm = PassManager("builtin.module")
        pm.enable_ir_printing()
        pm.add("func.func(hivm-plan-memory)")
        pm.run(module.operation)
        print('--- module: ', module)

# CHECK-LABEL: testHIVMPlanMemoryPMParse
@run
def testHIVMPlanMemoryPMParse():
    with Context() as ctx, Location.unknown():
        register_dialects(ctx)
        module = Module.parse(str_case)
        pm = PassManager.parse("builtin.module(func.func(hivm-plan-memory))")  # work
        pm.enable_ir_printing()
        pm.run(module.operation)
        print('--- module: ', module)
