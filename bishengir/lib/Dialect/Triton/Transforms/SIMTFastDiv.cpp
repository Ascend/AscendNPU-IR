//===- SIMTFastDiv.cpp - SIMT Fast Division Optimization ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// For each 32-bit integer division whose divisor is a function argument or an
// arith.constant (or a tt.splat of either), this pass replaces the division
// with:
//
//   shift = _mlir_ciface_simt_div_magic_shift_uint32_t(divisor)
//   magic = _mlir_ciface_simt_div_magic_mul_uint32_t(divisor, shift)
//   hi    = upper_32_bits( zero_extend(dividend) * zero_extend(magic) )
//   result= hi >> shift
//
// The two scalar computations are hoisted to the very start of the function
// entry block via ascend_dpx.call_scalar ops, each backed by a
// ttg.local_alloc shared-memory buffer sized for one i32 (= return type size).
//
// The shared-memory results are pulled back via ttg.local_load ops placed
// immediately before each use site:
//   - For scalar i32 divisions the load result type is tensor<1xi32> (the
//     minimum valid ranked tensor type); the captured scalar SSA values from
//     call_scalar are used for the arithmetic replacement because local_load
//     cannot return a bare scalar.
//   - For tensor<Nxi32> divisions the load result type is tensor<1xi32> using
//     the dividend's own encoding.  BroadcastOp (which requires matching
//     encodings on source and result) then broadcasts tensor<1xi32,#enc> to
//     tensor<Nxi32,#enc>, giving element-wise magic multiply and shift.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/AscendDPX/IR/AscendDPX.h"
#include "bishengir/Dialect/Triton/IR/TritonExtension.h"
#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace bishengir {
namespace triton {

#define GEN_PASS_DEF_SIMTFASTDIV
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

static constexpr char kMagicShiftFn[] =
    "_mlir_ciface_simt_div_magic_shift_uint32_t";
static constexpr char kMagicMulFn[] =
    "_mlir_ciface_simt_div_magic_mul_uint32_t";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return the i32 BlockArgument that ultimately drives \p divisor, or nullptr.
/// Return true if \p ba is an argument of the entry block of its parent
/// tt.func (i.e. a genuine function argument, not a loop-carried value).
static bool isFuncArg(BlockArgument ba) {
  Block *blk = ba.getOwner();
  if (!blk)
    return false;
  auto *parentOp = blk->getParentOp();
  auto funcOp = dyn_cast_or_null<mlir::triton::FuncOp>(parentOp);
  return funcOp && &funcOp.getBody().front() == blk;
}

/// Returns the i32 BlockArgument driving \p divisor (direct or via tt.splat),
/// only if it is a genuine entry-block function argument. Returns {} otherwise.
static BlockArgument getI32FuncArgDivisor(Value divisor) {
  Value scalar = divisor;
  if (auto splat = divisor.getDefiningOp<SplatOp>())
    scalar = splat.getSrc();
  if (auto ba = dyn_cast<BlockArgument>(scalar))
    if (ba.getType().isInteger(32) && isFuncArg(ba))
      return ba;
  return {};
}

/// Returns the compile-time i32 value if \p divisor is an arith.constant
/// (direct or via tt.splat). Returns std::nullopt otherwise.
static std::optional<int32_t> getI32ConstDivisor(Value divisor) {
  Value scalar = divisor;
  if (auto splat = divisor.getDefiningOp<SplatOp>())
    scalar = splat.getSrc();
  if (!scalar.getType().isInteger(32))
    return std::nullopt;
  if (auto constOp = scalar.getDefiningOp<arith::ConstantOp>())
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return static_cast<int32_t>(intAttr.getInt());
  return std::nullopt;
}

/// Insert a private llvm.func declaration in the module if it does not exist.
static LLVM::LLVMFuncOp ensureDecl(OpBuilder &b, ModuleOp mod, Location loc,
                                   StringRef name, FunctionType fnTy) {
  if (auto fn = mod.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return fn;
  // Convert mlir::FunctionType → LLVM::LLVMFunctionType.
  auto llvmFnTy = LLVM::LLVMFunctionType::get(
      fnTy.getNumResults() > 0 ? fnTy.getResult(0)
                               : LLVM::LLVMVoidType::get(b.getContext()),
      fnTy.getInputs());
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(mod.getBody());
  auto decl = b.create<LLVM::LLVMFuncOp>(loc, name, llvmFnTy);
  decl.setLinkage(LLVM::Linkage::External);
  return decl;
}

/// !ttg.memdesc<1xi32, #swizzled_shared<vec=1,perPhase=1,maxPhase=1,order=[0]>,
///             #smem, mutable>
/// One i32 element — matching the i32 return-type size of the helper funcs.
static MemDescType buildI32MemDescType(MLIRContext *ctx) {
  auto smem = SharedMemorySpaceAttr::get(ctx);
  auto ctaLayout =
      CTALayoutAttr::get(ctx, /*ctasPerCGA=*/{1}, /*ctasSplitNum=*/{1},
                         /*ctasOrder=*/{0});
  auto sharedEnc = SwizzledSharedEncodingAttr::get(
      ctx, /*vec=*/1, /*perPhase=*/1, /*maxPhase=*/1, /*order=*/{0}, ctaLayout);
  return MemDescType::get({1}, IntegerType::get(ctx, 32), sharedEnc, smem,
                          /*mutableMemory=*/true);
}

/// tensor<1xi32> result type for local_load from memdesc<1xi32>, carrying
/// the provided encoding attribute.  Passing \p enc=nullptr uses a fallback
/// blocked encoding whose threadsPerWarp and warpsPerCTA are taken from the
/// module attributes so the layout passes Triton's verifier.
static RankedTensorType buildLoadType(MLIRContext *ctx, Attribute enc,
                                      unsigned threadsPerWarp = 32,
                                      unsigned numWarps = 1) {
  auto i32Ty = IntegerType::get(ctx, 32);
  if (enc)
    return RankedTensorType::get({1}, i32Ty, enc);
  auto ctaLayout = CTALayoutAttr::get(ctx, {1}, {1}, {0});
  auto blk = BlockedEncodingAttr::get(ctx, {1}, {threadsPerWarp}, {numWarps},
                                      {0}, ctaLayout);
  return RankedTensorType::get({1}, i32Ty, blk);
}

// ---------------------------------------------------------------------------
// Per-divisor info built at function entry
// ---------------------------------------------------------------------------

struct MagicInfo {
  Value shmShift;    ///< memdesc<1xi32> buffer for the shift result
  Value shmMul;      ///< memdesc<1xi32> buffer for the magic multiplier result
  Value shiftScalar; ///< i32 SSA value from the shift call_scalar
  Value magicScalar; ///< i32 SSA value from the magic call_scalar
};

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------

class SIMTFastDivPass : public impl::SIMTFastDivBase<SIMTFastDivPass> {
public:
  void runOnOperation() override {
    mlir::triton::FuncOp func = getOperation();
    if (func.getBody().empty())
      return;

    MLIRContext *ctx = &getContext();
    Location loc = func.getLoc();
    OpBuilder builder(ctx);

    // ------------------------------------------------------------------
    // 1. Walk the function and gather every 32-bit integer division or
    //    remainder whose divisor traces back to a function argument or
    //    an arith.constant.
    // ------------------------------------------------------------------
    struct DivRemWork {
      Operation *op;
      bool isTensor;
      bool isConst;
      bool isRem;               ///< true for remainder, false for division
      BlockArgument divisorArg; ///< valid when !isConst
      int32_t constVal;         ///< valid when isConst
    };
    SmallVector<DivRemWork> work;

    func.walk([&](Operation *op) {
      Value divisor;
      bool isRem = false;
      if (auto d = dyn_cast<arith::DivSIOp>(op))
        divisor = d.getRhs();
      else if (auto d = dyn_cast<arith::DivUIOp>(op))
        divisor = d.getRhs();
      else if (auto r = dyn_cast<arith::RemSIOp>(op)) {
        divisor = r.getRhs();
        isRem = true;
      } else if (auto r = dyn_cast<arith::RemUIOp>(op)) {
        divisor = r.getRhs();
        isRem = true;
      } else
        return;

      bool isTensor = false;
      if (auto tt = dyn_cast<RankedTensorType>(divisor.getType())) {
        if (!tt.getElementType().isInteger(32))
          return;
        isTensor = true;
      } else if (!divisor.getType().isInteger(32)) {
        return;
      }

      if (auto arg = getI32FuncArgDivisor(divisor))
        work.push_back({op, isTensor, /*isConst=*/false, isRem, arg, 0});
      else if (auto cv = getI32ConstDivisor(divisor))
        work.push_back({op, isTensor, /*isConst=*/true, isRem, {}, *cv});
    });

    if (work.empty())
      return;

    // ------------------------------------------------------------------
    // 2. Declare the helper functions in the parent module.
    // ------------------------------------------------------------------
    auto mod = func->getParentOfType<ModuleOp>();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto threadsPerWarp =
        mod->getAttrOfType<IntegerAttr>(AttrNumThreadsPerWarp)
            ? (unsigned)mod->getAttrOfType<IntegerAttr>(AttrNumThreadsPerWarp)
                  .getInt()
            : 32u;
    auto numWarps =
        mod->getAttrOfType<IntegerAttr>(AttrNumWarpsName)
            ? (unsigned)mod->getAttrOfType<IntegerAttr>(AttrNumWarpsName)
                  .getInt()
            : 1u;
    ensureDecl(builder, mod, loc, kMagicShiftFn,
               FunctionType::get(ctx, {i32Ty}, {i32Ty}));
    ensureDecl(builder, mod, loc, kMagicMulFn,
               FunctionType::get(ctx, {i32Ty, i32Ty}, {i32Ty}));

    auto memDescTy = buildI32MemDescType(ctx);

    // ------------------------------------------------------------------
    // 3. For each unique divisor, emit local_alloc + call_scalar at the
    //    very start of the function entry block.  Deduplication is keyed
    //    by argument number (for func args) or constant value (for consts).
    // ------------------------------------------------------------------
    builder.setInsertionPointToStart(&func.getBody().front());

    llvm::SmallDenseMap<unsigned, MagicInfo> argToMagic;
    llvm::SmallDenseMap<int32_t, MagicInfo> constToMagic;

    // Emit the local_alloc + call_scalar pair for a given scalar i32 value.
    auto emitMagic = [&](Value scalarDivisor) -> MagicInfo {
      Value shmShift = builder.create<LocalAllocOp>(loc, memDescTy);
      Value shiftScalar =
          builder
              .create<mlir::triton::CallScalarOp>(
                  loc, TypeRange{i32Ty}, shmShift,
                  FlatSymbolRefAttr::get(ctx, kMagicShiftFn),
                  ValueRange{scalarDivisor},
                  /*arg_attrs=*/ArrayAttr{}, /*res_attrs=*/ArrayAttr{})
              .getResult(0);
      Value shmMul = builder.create<LocalAllocOp>(loc, memDescTy);
      Value magicScalar =
          builder
              .create<mlir::triton::CallScalarOp>(
                  loc, TypeRange{i32Ty}, shmMul,
                  FlatSymbolRefAttr::get(ctx, kMagicMulFn),
                  ValueRange{scalarDivisor, shiftScalar},
                  /*arg_attrs=*/ArrayAttr{}, /*res_attrs=*/ArrayAttr{})
              .getResult(0);
      return {shmShift, shmMul, shiftScalar, magicScalar};
    };

    for (auto &d : work) {
      if (!d.isConst) {
        unsigned idx = d.divisorArg.getArgNumber();
        if (!argToMagic.count(idx))
          argToMagic[idx] = emitMagic(d.divisorArg);
      } else {
        if (!constToMagic.count(d.constVal)) {
          Value constV = builder.create<arith::ConstantOp>(
              loc, builder.getIntegerAttr(i32Ty, d.constVal));
          constToMagic[d.constVal] = emitMagic(constV);
        }
      }
    }

    // ------------------------------------------------------------------
    // 4. Replace each division / remainder.  local_load ops are emitted
    //    immediately before each use site to make the shm reads explicit.
    //    Both div and rem share the same magic info (call_scalar ops are
    //    deduplicated per divisor in step 3).
    //
    //    div:  quotient = umulhi(dividend, magic) >> shift
    //    rem:  quotient = umulhi(dividend, magic) >> shift
    //          result   = dividend - quotient * divisor
    // ------------------------------------------------------------------
    for (auto &d : work) {
      MagicInfo &mi = d.isConst ? constToMagic[d.constVal]
                                : argToMagic[d.divisorArg.getArgNumber()];
      builder.setInsertionPoint(d.op);
      bool isSigned = isa<arith::DivSIOp>(d.op) || isa<arith::RemSIOp>(d.op);

      // Extract dividend and divisor uniformly for div and rem ops.
      Value dividend, origDivisor;
      if (d.isRem) {
        dividend = isSigned ? cast<arith::RemSIOp>(d.op).getLhs()
                            : cast<arith::RemUIOp>(d.op).getLhs();
        origDivisor = isSigned ? cast<arith::RemSIOp>(d.op).getRhs()
                               : cast<arith::RemUIOp>(d.op).getRhs();
      } else {
        dividend = isSigned ? cast<arith::DivSIOp>(d.op).getLhs()
                            : cast<arith::DivUIOp>(d.op).getLhs();
        origDivisor = isSigned ? cast<arith::DivSIOp>(d.op).getRhs()
                               : cast<arith::DivUIOp>(d.op).getRhs();
      }

      // Compute the quotient using magic multiply + shift.
      Value quotient;
      if (!d.isTensor) {
        // ---- Scalar i32 ------------------------------------------------
        auto loadTy =
            buildLoadType(ctx, /*enc=*/nullptr, threadsPerWarp, numWarps);
        builder.create<LocalLoadOp>(loc, loadTy, mi.shmShift);
        builder.create<LocalLoadOp>(loc, loadTy, mi.shmMul);

        Value hi32 = builder.create<mlir::ascend_dpx::UmulhiOp>(
            loc, i32Ty, dividend, mi.magicScalar);
        Value sum = builder.create<arith::AddIOp>(loc, hi32, dividend);
        quotient =
            isSigned ? builder.create<arith::ShRSIOp>(loc, sum, mi.shiftScalar)
                           .getResult()
                     : builder.create<arith::ShRUIOp>(loc, sum, mi.shiftScalar)
                           .getResult();

      } else {
        // ---- Tensor<Nxi32> ---------------------------------------------
        auto resTy = cast<RankedTensorType>(d.op->getResultTypes()[0]);
        Attribute enc = resTy.getEncoding();

        auto loadTy = buildLoadType(ctx, enc);
        Value magicLoad = builder.create<LocalLoadOp>(loc, loadTy, mi.shmMul);
        Value shiftLoad = builder.create<LocalLoadOp>(loc, loadTy, mi.shmShift);

        Value magicTensor = builder.create<BroadcastOp>(loc, resTy, magicLoad);
        Value shiftTensor = builder.create<BroadcastOp>(loc, resTy, shiftLoad);

        Value hi32 =
            builder.create<triton::MulhiUIOp>(loc, dividend, magicTensor);
        Value sum = builder.create<arith::AddIOp>(loc, hi32, dividend);
        quotient = isSigned
                       ? builder.create<arith::ShRSIOp>(loc, sum, shiftTensor)
                             .getResult()
                       : builder.create<arith::ShRUIOp>(loc, sum, shiftTensor)
                             .getResult();
      }

      // For division the result is the quotient directly.
      // For remainder: result = dividend - quotient * divisor.
      Value result = quotient;
      if (d.isRem) {
        Value mul = builder.create<arith::MulIOp>(loc, quotient, origDivisor);
        result = builder.create<arith::SubIOp>(loc, dividend, mul);
      }

      d.op->getResult(0).replaceAllUsesWith(result);
      d.op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createSIMTFastDivPass() {
  return std::make_unique<SIMTFastDivPass>();
}

} // namespace triton
} // namespace bishengir
