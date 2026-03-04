# RUN: python3 %s | FileCheck %s

# REQUIRES: issue2438
from bishengir.ir import *
import bishengir.extras.types as Ty
from bishengir.helper import *
from bishengir.dialects import arith, func, hivm, memref
from bishengir.passmanager import *
from bishengir.dialects.linalg.opdsl.lang import *
from bishengir.bishengtile import *

from bishengir._mlir_libs._bishengirRegisterEverything import register_dialects, register_translations

with Context() as ctx, Location.unknown():
    ctx.allow_unregistered_dialects = True
    register_dialects(ctx)
    register_translations(ctx)
    module = Module.create()
    UB = hivm.addressSpaceAttr.get(hivm.addressSpace.UB)
    GM = hivm.addressSpaceAttr.get(hivm.addressSpace.GM)
    gmTy = Ty.memref(64, Ty.i32(),memory_space=GM)
    ubTy = Ty.memref(64, Ty.i32(),memory_space=UB)
    # CHECK: test_vf
    # CHECK: hivm.hir.load
    # CHECK: hivm.hir.store
    # CHECK: hivm.hir.copy
    with InsertionPoint(module.body), Location.unknown():
        funcOp = func.FuncOp("test_vf", ([gmTy,gmTy], []))
        m0 =  memref.alloc(gmTy, [], [])
        m1 =  memref.alloc(ubTy, [], [])
        m2 =  memref.alloc(ubTy, [], [])
        Ascend.copy_in(m0,m1)
        Ascend.copy_out(m1,m0)
        Ascend.copy(m1,m2)

    print(module)