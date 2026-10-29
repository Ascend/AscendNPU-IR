//===- BishengAsyncRuntime.cpp - True-async Worker runtime ----------------===//
//
// Implementation of the per-lane Worker runtime over the v2 BishengLlcBridge.
// See BishengAsyncRuntime.h for the design.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Tools/hivmc/BishengAsyncRuntime.h"

#include "bishengir/Dialect/HIVM/Utils/Utils.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <sys/syscall.h>

namespace bishengir {

namespace {

/// Get the OS thread id (Linux).
uint64_t getOsTid() {
#ifdef SYS_gettid
  return static_cast<uint64_t>(syscall(SYS_gettid));
#else
  return static_cast<uint64_t>(::gettid());
#endif
}

/// Get a monotonic nanosecond timestamp (steady_clock).
int64_t nowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool traceEnabled() {
  const char *e = getenv("BISHENG_ASYNC_TRACE");
  return e && e[0] != '\0';
}

std::string traceFile() {
  const char *e = getenv("BISHENG_ASYNC_TRACE_FILE");
  if (e && e[0] != '\0')
    return std::string(e);
  return "";
}

LaneKind laneForMaskIndex(size_t i) { return i == 0 ? LaneKind::AIC : LaneKind::AIV; }

const char *laneName(LaneKind k) { return k == LaneKind::AIC ? "AIC" : "AIV"; }

} // namespace

// ---------------------------------------------------------------------------
// BishengLlcWorker
// ---------------------------------------------------------------------------

BishengLlcWorker::BishengLlcWorker(LaneKind kind, BishengAsyncRuntime *runtime)
    : kind_(kind), runtime_(runtime) {}

BishengLlcWorker::~BishengLlcWorker() { shutdownWorker(); }

void BishengLlcWorker::startWorker() {
  if (workerState_.load() != WorkerState::Cold)
    return;
  workerState_.store(WorkerState::Starting);
  workerThread_ = std::thread([this] { workerLoop(); });
}

CommandTicket BishengLlcWorker::submitLoadInit(llvm::StringRef soPath) {
  CommandTicket ticket;
  WorkerCommand cmd;
  cmd.kind = WorkerCommand::Kind::LoadInit;
  {
    std::lock_guard<std::mutex> lk(mtx_);
    cmd.generation = ++commandGeneration_;
    ticket.generation = cmd.generation;
    cmd.soPath = soPath.str();
    commandQueue_.push_back(std::move(cmd));
  }
  cv_.notify_one();
  return ticket;
}

CommandTicket BishengLlcWorker::submitPreloadBase(
    const std::vector<std::string> &baseAux) {
  CommandTicket ticket;
  WorkerCommand cmd;
  cmd.kind = WorkerCommand::Kind::PreloadBase;
  {
    std::lock_guard<std::mutex> lk(mtx_);
    cmd.generation = ++commandGeneration_;
    ticket.generation = cmd.generation;
    cmd.baseAux = baseAux;
    commandQueue_.push_back(std::move(cmd));
  }
  cv_.notify_one();
  return ticket;
}

CommandTicket BishengLlcWorker::submitPreloadFinalDelta(
    const std::vector<std::string> &deltaAux) {
  CommandTicket ticket;
  WorkerCommand cmd;
  cmd.kind = WorkerCommand::Kind::PreloadFinalDelta;
  {
    std::lock_guard<std::mutex> lk(mtx_);
    cmd.generation = ++commandGeneration_;
    ticket.generation = cmd.generation;
    cmd.finalDeltaAux = deltaAux;
    commandQueue_.push_back(std::move(cmd));
  }
  cv_.notify_one();
  return ticket;
}

CommandTicket BishengLlcWorker::submitCompile(
    const std::string &irBytes,
    const llvm::SmallVectorImpl<std::string> &fullAux,
    llvm::StringRef targetCpu, uint32_t optLevel,
    const std::vector<std::string> &extraOpts, llvm::StringRef outputPath) {
  CommandTicket ticket;
  WorkerCommand cmd;
  cmd.kind = WorkerCommand::Kind::Compile;
  {
    std::lock_guard<std::mutex> lk(mtx_);
    cmd.generation = ++commandGeneration_;
    ticket.generation = cmd.generation;
    cmd.irBytes = irBytes;
    cmd.fullAux.assign(fullAux.begin(), fullAux.end());
    cmd.targetCpu = targetCpu.str();
    cmd.optLevel = optLevel;
    cmd.extraOpts = extraOpts;
    cmd.outputPath = outputPath.str();
    commandQueue_.push_back(std::move(cmd));
  }
  cv_.notify_one();
  return ticket;
}

bool BishengLlcWorker::waitLoadInit(CommandTicket ticket) {
  if (!ticket.valid())
    return true;
  CommandCompletion comp;
  bool ok = waitCommand(ticket, comp);
  return ok && comp.success;
}

bool BishengLlcWorker::waitPreloadBase(CommandTicket ticket) {
  if (!ticket.valid())
    return true;
  CommandCompletion comp;
  bool ok = waitCommand(ticket, comp);
  return ok && comp.success;
}

bool BishengLlcWorker::waitPreloadFinalDelta(CommandTicket ticket) {
  if (!ticket.valid())
    return true;
  CommandCompletion comp;
  bool ok = waitCommand(ticket, comp);
  return ok && comp.success;
}

BishengLlcCompileResult BishengLlcWorker::waitCompile(CommandTicket ticket) {
  BishengLlcCompileResult result;
  if (!ticket.valid()) {
    result.diagnostic = "waitCompile: invalid ticket";
    return result;
  }
  CommandCompletion comp;
  if (!waitCommand(ticket, comp)) {
    result.diagnostic = "waitCompile: worker terminated before completion";
    return result;
  }
  return comp.compileResult;
}

void BishengLlcWorker::shutdownWorker() {
  bool wasStarted = workerState_.load() != WorkerState::Cold;
  if (!wasStarted) {
    // Worker never started; ensure state is Stopped.
    workerState_.store(WorkerState::Stopped);
    return;
  }
  if (workerState_.load() == WorkerState::Stopped)
    return;
  {
    std::lock_guard<std::mutex> lk(mtx_);
    WorkerCommand cmd;
    cmd.kind = WorkerCommand::Kind::Shutdown;
    cmd.generation = ++commandGeneration_;
    commandQueue_.push_back(std::move(cmd));
  }
  cv_.notify_one();
  if (workerThread_.joinable())
    workerThread_.join();
  workerState_.store(WorkerState::Stopped);
}

void BishengLlcWorker::resetForRetry() {
  // Recreate this lane so a fresh Compile can run on a NEW Vendor context +
  // NEW dlmopen namespace.  Called by BishengAsyncRuntime::recreateLane when
  // the previous Compile consumed the context (workerState_ == Consumed), so
  // the A5 multi-attempt loop can re-LoadInit + re-PreloadBase +
  // re-PreloadFinalDelta + re-Compile on this lane without leaking the old
  // namespace / context / TLS.
  //
  // Join-before-recreate: ensure the worker thread is fully joined and any
  // in-flight command has completed before we touch `bridge_` / state.  We
  // do this by submitting a Shutdown command and joining (same as
  // shutdownWorker), then resetting the bridge and per-worker state under
  // the lock.
  //
  // `preloadedPaths_` is mutated only under `mtx_` to guarantee no data race
  // even if a future caller invokes resetForRetry while the worker is still
  // running; `commandGeneration_` is kept monotonically increasing (never
  // reset to 0) so generation values remain unique identifiers for the
  // lifetime of the worker.
  WorkerState st = workerState_.load();
  if (st == WorkerState::Cold) {
    // Never started: nothing to recreate.  Still reset bridge + state for
    // a clean slate (idempotent).
    bridge_.reset();
    {
      std::lock_guard<std::mutex> lk(mtx_);
      commandQueue_.clear();
      completions_.clear();
      preloadedPaths_.clear();
      // commandGeneration_ intentionally NOT reset: keep monotonic.
    }
    workerState_.store(WorkerState::Cold);
    lmid_.store(0);
    workerTid_.store(0);
    return;
  }
  // Started (Starting/Ready/BaseAuxReady/Compiling/Consumed/Failed/Stopped):
  // submit Shutdown + join to guarantee the worker thread is no longer
  // touching `bridge_`.  Pending commands (e.g. a PreloadFinalDelta still
  // queued) are dropped: the runtime's recreateLane has already drained the
  // retained tickets, and dropping queued commands is safe because the
  // caller will re-submit the full LoadInit -> ... -> Compile sequence.
  if (st != WorkerState::Stopped) {
    {
      std::lock_guard<std::mutex> lk(mtx_);
      WorkerCommand cmd;
      cmd.kind = WorkerCommand::Kind::Shutdown;
      cmd.generation = ++commandGeneration_;
      commandQueue_.push_back(std::move(cmd));
    }
    cv_.notify_one();
    if (workerThread_.joinable())
      workerThread_.join();
  }
  // Now the worker thread is joined; safe to reset bridge.  Per-worker
  // state that may be observed by future commands is cleared under the
  // lock to keep the data-race surface explicit.
  bridge_.reset();
  {
    std::lock_guard<std::mutex> lk(mtx_);
    commandQueue_.clear();
    completions_.clear();
    preloadedPaths_.clear();
    // commandGeneration_ intentionally NOT reset: keep monotonic so ticket
    // values stay unique for the worker's lifetime.
  }
  workerState_.store(WorkerState::Cold);
  lmid_.store(0);
  workerTid_.store(0);
  if (runtime_)
    runtime_->emitWorkerEvent(kind_, "lane_recreate", "point", "", 0, 0);
}

void BishengLlcWorker::completeCommand(uint64_t generation, bool success,
                                       const BishengLlcCompileResult &res) {
  {
    std::lock_guard<std::mutex> lk(mtx_);
    CommandCompletion &c = completions_[generation];
    c.done = true;
    c.success = success;
    c.compileResult = res;
  }
  cv_.notify_all();
}

bool BishengLlcWorker::waitCommand(CommandTicket ticket,
                                   CommandCompletion &out) {
  std::unique_lock<std::mutex> lk(mtx_);
  cv_.wait(lk, [this, ticket] {
    auto it = completions_.find(ticket.generation);
    if (it != completions_.end() && it->second.done)
      return true;
    // Worker terminated and will not produce more completions.
    return workerState_.load() == WorkerState::Failed ||
           workerState_.load() == WorkerState::Stopped;
  });
  auto it = completions_.find(ticket.generation);
  if (it != completions_.end() && it->second.done) {
    out = it->second;
    completions_.erase(it);
    return true;
  }
  // Worker died without completing.
  return false;
}

void BishengLlcWorker::workerLoop() {
  workerTid_.store(getOsTid());
  while (true) {
    WorkerCommand cmd;
    {
      std::unique_lock<std::mutex> lk(mtx_);
      cv_.wait(lk, [this] { return !commandQueue_.empty(); });
      cmd = std::move(commandQueue_.front());
      commandQueue_.pop_front();
    }
    if (cmd.kind == WorkerCommand::Kind::Shutdown) {
      // Destroy bridge if loaded, then exit.
      bridge_.destroy();
      workerState_.store(WorkerState::Stopped);
      // Wake any waiters.
      {
        std::lock_guard<std::mutex> lk(mtx_);
      }
      cv_.notify_all();
      return;
    }
    // The build is compiled with -fno-exceptions, so handlers cannot throw.
    // Each handler is responsible for calling completeCommand() on every
    // return path (including fail-closed early returns); the waitCommand()
    // predicate also unblocks on workerState_ == Failed or Stopped, so a
    // Wait-forever is impossible as long as handlers either complete the
    // command or transition the worker to a terminal state.  The
    // handleCompile state guard ensures a Compile on a non-Ready lane
    // completes with failure immediately rather than leaving the ticket
    // uncompleted.
    switch (cmd.kind) {
    case WorkerCommand::Kind::LoadInit:
      handleLoadInit(cmd);
      break;
    case WorkerCommand::Kind::PreloadBase:
      handlePreloadBase(cmd);
      break;
    case WorkerCommand::Kind::PreloadFinalDelta:
      handlePreloadFinalDelta(cmd);
      break;
    case WorkerCommand::Kind::Compile: {
      CommandCompletion comp;
      handleCompile(cmd, comp);
      completeCommand(cmd.generation, comp.success, comp.compileResult);
      break;
    }
    case WorkerCommand::Kind::Shutdown:
      // Already handled above; keep for completeness.
      break;
    }
  }
}

void BishengLlcWorker::handleLoadInit(const WorkerCommand &cmd) {
  auto emitW = [&](llvm::StringRef phase, llvm::StringRef event,
                   llvm::StringRef status) {
    if (runtime_)
      runtime_->emitWorkerEvent(kind_, phase, event, status, lmid_.load(),
                               workerTid_.load());
  };
  // LoadInit only does dlmopen + four dlsym + bishengLlcInit.  It does NOT
  // preload any aux path (LoadInit must be independent of aux paths).
  // Trace split: `llc_load` strictly wraps bridge_.load (dlmopen + dlsym)
  // and `llc_init` strictly wraps bridge_.init (bishengLlcInit).
  emitW("llc_load", "begin", "");
  if (!bridge_.load(cmd.soPath)) {
    workerState_.store(WorkerState::Failed);
    BishengLlcCompileResult res;
    res.diagnostic = "loadinit: bridge.load failed";
    emitW("llc_load", "end", "fail");
    completeCommand(cmd.generation, false, res);
    return;
  }
  lmid_.store(bridge_.lmid());
  emitW("llc_load", "end", "ok");
  emitW("llc_init", "begin", "");
  if (!bridge_.init()) {
    workerState_.store(WorkerState::Failed);
    BishengLlcCompileResult res;
    res.diagnostic = "loadinit: bridge.init failed";
    emitW("llc_init", "end", "fail");
    completeCommand(cmd.generation, false, res);
    return;
  }
  // LoadInit succeeds with workerState_ = Ready.  Base aux preload is a
  // separate PreloadBase command.
  workerState_.store(WorkerState::Ready);
  BishengLlcCompileResult res;
  res.success = true;
  emitW("llc_init", "end", "ok");
  completeCommand(cmd.generation, true, res);
}

void BishengLlcWorker::handlePreloadBase(const WorkerCommand &cmd) {
  auto emitW = [&](llvm::StringRef phase, llvm::StringRef event,
                   llvm::StringRef status) {
    if (runtime_)
      runtime_->emitWorkerEvent(kind_, phase, event, status, lmid_.load(),
                               workerTid_.load());
  };
  // fail-closed guard: PreloadBase requires LoadInit done (workerState_ ==
  // Ready or BaseAuxReady).  If LoadInit failed (Failed) or never ran, do
  // not attempt preload.
  WorkerState st = workerState_.load();
  if (st != WorkerState::Ready && st != WorkerState::BaseAuxReady) {
    workerState_.store(WorkerState::Failed);
    BishengLlcCompileResult res;
    res.diagnostic =
        "preload_base: worker not Ready (state=" + std::to_string((int)st) +
        "); LoadInit must precede PreloadBase";
    emitW("base_aux", "end", "fail");
    completeCommand(cmd.generation, false, res);
    return;
  }
  emitW("base_aux", "begin", "");
  for (const auto &path : cmd.baseAux) {
    if (path.empty())
      continue;
    // Skip paths already preloaded (idempotent).
    bool already = false;
    for (const auto &p : preloadedPaths_) {
      if (p == path) {
        already = true;
        break;
      }
    }
    if (already)
      continue;
    if (!bridge_.preloadAux(path)) {
      workerState_.store(WorkerState::Failed);
      BishengLlcCompileResult res;
      res.diagnostic = "preload_base: preloadAux failed: " + path;
      emitW("base_aux", "end", "fail");
      completeCommand(cmd.generation, false, res);
      return;
    }
    preloadedPaths_.push_back(path);
  }
  workerState_.store(WorkerState::BaseAuxReady);
  BishengLlcCompileResult res;
  res.success = true;
  emitW("base_aux", "end", "ok");
  completeCommand(cmd.generation, true, res);
}

void BishengLlcWorker::handlePreloadFinalDelta(const WorkerCommand &cmd) {
  auto emitW = [&](llvm::StringRef phase, llvm::StringRef event,
                   llvm::StringRef status) {
    if (runtime_)
      runtime_->emitWorkerEvent(kind_, phase, event, status, lmid_.load(),
                               workerTid_.load());
  };
  // fail-closed guard: PreloadFinalDelta requires LoadInit done.  It does
  // NOT consume the context, so BaseAuxReady / Ready are both acceptable.
  WorkerState st = workerState_.load();
  if (st != WorkerState::Ready && st != WorkerState::BaseAuxReady) {
    workerState_.store(WorkerState::Failed);
    BishengLlcCompileResult res;
    res.diagnostic = "preload_final_delta: worker not Ready (state=" +
                     std::to_string((int)st) + ")";
    emitW("final_aux", "end", "fail");
    completeCommand(cmd.generation, false, res);
    return;
  }
  emitW("final_aux", "begin", "");
  for (const auto &path : cmd.finalDeltaAux) {
    if (path.empty())
      continue;
    bool already = false;
    for (const auto &p : preloadedPaths_) {
      if (p == path) {
        already = true;
        break;
      }
    }
    if (already)
      continue;
    if (!bridge_.preloadAux(path)) {
      workerState_.store(WorkerState::Failed);
      BishengLlcCompileResult res;
      res.diagnostic = "preload_final_delta: preloadAux failed: " + path;
      emitW("final_aux", "end", "fail");
      completeCommand(cmd.generation, false, res);
      return;
    }
    preloadedPaths_.push_back(path);
  }
  // PreloadFinalDelta does not change workerState_ (still BaseAuxReady or
  // Ready) and does not consume the context.
  BishengLlcCompileResult res;
  res.success = true;
  emitW("final_aux", "end", "ok");
  completeCommand(cmd.generation, true, res);
}

void BishengLlcWorker::handleCompile(const WorkerCommand &cmd,
                                     CommandCompletion &comp) {
  // Trace split: the delta preload loop (paths in fullAux not yet preloaded)
  // is traced as a `final_aux` event; `llc_compile` strictly wraps
  // bridge_.compile (bishengLlcCompile()) only.  llc_compile begin/end do
  // NOT include PreloadFinalDelta (independent command) nor the in-Compile
  // delta preload.
  auto emitW = [&](llvm::StringRef phase, llvm::StringRef event,
                   llvm::StringRef status) {
    if (runtime_)
      runtime_->emitWorkerEvent(kind_, phase, event, status, lmid_.load(),
                               workerTid_.load());
  };
  // Fail-closed state guard.  Compile requires LoadInit done (Ready or
  // BaseAuxReady).  Rejecting Consumed (context already used by a previous
  // Compile — bishengLlcCompile is at most once per Context), Failed (lane
  // unrecoverable), Cold (LoadInit never submitted — safe supplementary
  // start was missed), and Stopped (worker shutting down).  Marking Failed
  // here surfaces the error to the A5 retry loop, which aborts on a Failed
  // lane rather than silently recreating it (a Failed state at a retry
  // boundary indicates an environment failure or a sync bug that a retry
  // with different tiling parameters will not fix).  This also prevents
  // Wait-forever if a Compile is somehow submitted to a never-started lane:
  // the command completes with failure immediately.
  WorkerState st = workerState_.load();
  if (st != WorkerState::Ready && st != WorkerState::BaseAuxReady) {
    workerState_.store(WorkerState::Failed);
    comp.done = true;
    comp.success = false;
    comp.compileResult.success = false;
    comp.compileResult.diagnostic =
        "compile: worker not Ready (state=" + std::to_string((int)st) +
        "); LoadInit must precede Compile and bishengLlcCompile is at most"
        " once per Context";
    emitW("llc_compile", "end", "fail");
    return;
  }
  workerState_.store(WorkerState::Compiling);
  // Delta preload: paths in fullAux not yet preloaded.  This is the in-Compile
  // fallback for paths PreloadFinalDelta missed.  Traced as `final_aux`, NOT
  // `llc_compile` (llc_compile must strictly wrap bishengLlcCompile only).
  bool anyDelta = false;
  for (const auto &path : cmd.fullAux) {
    if (path.empty())
      continue;
    bool already = false;
    for (const auto &p : preloadedPaths_) {
      if (p == path) {
        already = true;
        break;
      }
    }
    if (!already) {
      if (!anyDelta) {
        emitW("final_aux", "begin", "");
        anyDelta = true;
      }
      if (!bridge_.preloadAux(path)) {
        workerState_.store(WorkerState::Failed);
        comp.done = true;
        comp.success = false;
        comp.compileResult.success = false;
        comp.compileResult.diagnostic =
            "compile: delta preloadAux failed: " + path;
        if (anyDelta)
          emitW("final_aux", "end", "fail");
        return;
      }
      preloadedPaths_.push_back(path);
    }
  }
  if (anyDelta)
    emitW("final_aux", "end", "ok");
  // llc_compile: STRICTLY wraps bishengLlcCompile() (bridge_.compile).  No
  // preload inside this scope.
  emitW("llc_compile", "begin", "");
  auto res = bridge_.compile(cmd.irBytes, cmd.fullAux, cmd.targetCpu,
                             cmd.optLevel, cmd.extraOpts, cmd.outputPath);
  workerState_.store(WorkerState::Consumed);
  comp.done = true;
  comp.success = res.success;
  comp.compileResult = res;
  emitW("llc_compile", "end", res.success ? "ok" : "fail");
}

// ---------------------------------------------------------------------------
// BishengAsyncRuntime
// ---------------------------------------------------------------------------

BishengAsyncRuntime::BishengAsyncRuntime(llvm::StringRef soPath)
    : frozenSoPath_(soPath.str()) {
  workers_[0] = std::make_unique<BishengLlcWorker>(LaneKind::AIC, this);
  workers_[1] = std::make_unique<BishengLlcWorker>(LaneKind::AIV, this);
  initTrace();
}

BishengAsyncRuntime::~BishengAsyncRuntime() {
  // Drain all retained tickets (best-effort) then join all workers.
  drainAllTickets();
  shutdownAll();
}

void BishengAsyncRuntime::submitLoadInit(LaneMask mask) {
  if (static_cast<uint8_t>(mask) & static_cast<uint8_t>(LaneMask::AIC)) {
    auto &w = *workers_[0];
    if (!w.isWorkerStarted())
      w.startWorker();
    loadInitTickets_[0] = w.submitLoadInit(frozenSoPath_);
    emitEvent("AIC", "llc_load", "point");
  }
  if (static_cast<uint8_t>(mask) & static_cast<uint8_t>(LaneMask::AIV)) {
    auto &w = *workers_[1];
    if (!w.isWorkerStarted())
      w.startWorker();
    loadInitTickets_[1] = w.submitLoadInit(frozenSoPath_);
    emitEvent("AIV", "llc_load", "point");
  }
}

void BishengAsyncRuntime::submitPreloadBase(
    LaneMask mask, const std::vector<std::string> &aicBaseAux,
    const std::vector<std::string> &aivBaseAux) {
  if (static_cast<uint8_t>(mask) & static_cast<uint8_t>(LaneMask::AIC)) {
    auto &w = *workers_[0];
    if (!w.isWorkerStarted())
      w.startWorker();
    baseAuxTickets_[0] = w.submitPreloadBase(aicBaseAux);
    emitEvent("AIC", "base_aux", "point");
  }
  if (static_cast<uint8_t>(mask) & static_cast<uint8_t>(LaneMask::AIV)) {
    auto &w = *workers_[1];
    if (!w.isWorkerStarted())
      w.startWorker();
    baseAuxTickets_[1] = w.submitPreloadBase(aivBaseAux);
    emitEvent("AIV", "base_aux", "point");
  }
}

void BishengAsyncRuntime::submitPreloadFinalDelta(
    LaneKind lane, const std::vector<std::string> &deltaAux) {
  size_t idx = lane == LaneKind::AIC ? 0 : 1;
  auto &w = *workers_[idx];
  if (!w.isWorkerStarted())
    w.startWorker();
  finalDeltaTickets_[idx] = w.submitPreloadFinalDelta(deltaAux);
  emitEvent(lane == LaneKind::AIC ? "AIC" : "AIV", "final_aux", "point");
}

bool BishengAsyncRuntime::drainLoadInitTicket(LaneKind lane) {
  size_t idx = lane == LaneKind::AIC ? 0 : 1;
  CommandTicket ticket = loadInitTickets_[idx];
  loadInitTickets_[idx] = {};
  if (!ticket.valid())
    return true;
  return workers_[idx]->waitLoadInit(ticket);
}

bool BishengAsyncRuntime::drainBaseAuxTicket(LaneKind lane) {
  size_t idx = lane == LaneKind::AIC ? 0 : 1;
  CommandTicket ticket = baseAuxTickets_[idx];
  baseAuxTickets_[idx] = {};
  if (!ticket.valid())
    return true;
  return workers_[idx]->waitPreloadBase(ticket);
}

bool BishengAsyncRuntime::drainFinalDeltaTicket(LaneKind lane) {
  size_t idx = lane == LaneKind::AIC ? 0 : 1;
  CommandTicket ticket = finalDeltaTickets_[idx];
  finalDeltaTickets_[idx] = {};
  if (!ticket.valid())
    return true;
  return workers_[idx]->waitPreloadFinalDelta(ticket);
}

bool BishengAsyncRuntime::drainAllTickets() {
  bool ok = true;
  ok &= drainLoadInitTicket(LaneKind::AIC);
  ok &= drainLoadInitTicket(LaneKind::AIV);
  ok &= drainBaseAuxTicket(LaneKind::AIC);
  ok &= drainBaseAuxTicket(LaneKind::AIV);
  ok &= drainFinalDeltaTicket(LaneKind::AIC);
  ok &= drainFinalDeltaTicket(LaneKind::AIV);
  return ok;
}

void BishengAsyncRuntime::shutdownAll() {
  workers_[0]->shutdownWorker();
  workers_[1]->shutdownWorker();
  // Lifecycle observability: report per-lane Vendor context Init/Destroy
  // pairing.  At quiescence init == destroy for each lane; a mismatch means
  // a context leak (init > destroy) or a double-destroy (init < destroy).
  // Gated on BISHENG_LLC_DEBUG so it does not spam normal runs.
  if (getenv("BISHENG_LLC_DEBUG")) {
    for (int idx = 0; idx < 2; ++idx) {
      const char *lane = (idx == 0) ? "AIC" : "AIV";
      uint64_t ini = workers_[idx]->initCount();
      uint64_t des = workers_[idx]->destroyCount();
      llvm::errs() << "[BishengAsyncRuntime] " << lane
                   << " vendor ctx: init=" << ini << " destroy=" << des;
      if (ini != des)
        llvm::errs() << "  *** MISMATCH (";
      else
        llvm::errs() << "  (paired";
      llvm::errs() << (ini > des ? "LEAK" : (ini < des ? "DOUBLE-DESTROY" : "OK"))
                   << ") ***\n";
    }
  }
}

bool BishengAsyncRuntime::drainFinalDeltaTicketForRetry(LaneKind lane) {
  // Drain (wait on) a previous attempt's retained PreloadFinalDelta ticket
  // BEFORE the new attempt's submitPreloadFinalDelta overwrites
  // `finalDeltaTickets_[idx]`.  Without this, a previous attempt's ticket
  // could still be in-flight when overwritten, and its completion would
  // never be waited on (leak).  Returns true if the old ticket succeeded or
  // was absent; returns false if the old ticket failed.  Either way the
  // retained slot is cleared.  The caller (A5 retry loop) inspects the
  // lane's WorkerState after this drain to decide between recreateLane
  // (Consumed) and ensureLaneLoadInit (Ready/Cold); a Failed lane is not
  // retried — the error propagates to the RegBase driver.
  size_t idx = lane == LaneKind::AIC ? 0 : 1;
  CommandTicket ticket = finalDeltaTickets_[idx];
  finalDeltaTickets_[idx] = {};
  if (!ticket.valid())
    return true;
  return workers_[idx]->waitPreloadFinalDelta(ticket);
}

void BishengAsyncRuntime::recreateLane(LaneKind lane) {
  // Recreate one lane's worker so a fresh Compile can run on a NEW Vendor
  // context + NEW dlmopen namespace.  Called by the A5 multi-attempt loop
  // when the previous attempt's Compile consumed the context
  // (workerState_ == Consumed) — bishengLlcCompile is at most once per
  // Context, so a retry must build a fresh one.  After recreateLane the
  // worker is Cold; the caller must re-submit LoadInit + PreloadBase before
  // any PreloadFinalDelta / Compile on this lane.
  //
  // Join-before-recreate: drain retained tickets (best-effort), then
  // resetForRetry (which submits Shutdown + joins the worker thread + resets
  // the bridge + clears state).
  size_t idx = lane == LaneKind::AIC ? 0 : 1;
  // Drain retained tickets for this lane (best-effort; clears slots).
  emitEvent(lane == LaneKind::AIC ? "AIC" : "AIV", "lane_recreate", "begin");
  (void)drainLoadInitTicket(lane);
  (void)drainBaseAuxTicket(lane);
  (void)drainFinalDeltaTicket(lane);
  // Clear retained slots explicitly (drain already clears, but be defensive).
  loadInitTickets_[idx] = {};
  baseAuxTickets_[idx] = {};
  finalDeltaTickets_[idx] = {};
  // Recreate the worker (join + fresh bridge + reset state).
  workers_[idx]->resetForRetry();
  emitEvent(lane == LaneKind::AIC ? "AIC" : "AIV", "lane_recreate", "end");
}

void BishengAsyncRuntime::ensureLaneLoadInit(LaneKind lane) {
  // Safe supplementary start.  If a lane's LoadInit was never submitted
  // (worker still Cold) but the final topology turns out to need this lane,
  // submit LoadInit now so `drainLoadInitTicket` + `submitCompile` never
  // blocks forever waiting on a LoadInit that was never submitted.  This
  // covers the "early topology unknown" case: the early `submitLoadInit`
  // used a conservative mask (LaneMask::Both) so both lanes are normally
  // started, but if a future caller submits a narrower mask and the final
  // topology needs the other lane, this is the safety net.
  //
  // Idempotent: a no-op when the worker is already alive (Starting/Ready/
  // BaseAuxReady).  Only Cold triggers a start + LoadInit.  Callers must
  // NOT route Failed/Stopped/Compiling/Consumed lanes here — Failed/Stopped
  // are unrecoverable (the A5 retry loop surfaces them as errors rather
  // than silently recreating; see runDeviceBiShengLIRCompile), and Consumed
  // requires recreateLane (a fresh Vendor Context), not a supplementary
  // start.
  size_t idx = lane == LaneKind::AIC ? 0 : 1;
  auto &w = *workers_[idx];
  WorkerState st = w.state();
  if (st == WorkerState::Starting || st == WorkerState::Ready ||
      st == WorkerState::BaseAuxReady)
    return; // already started; LoadInit either in-flight or done
  if (st != WorkerState::Cold) {
    // Failed/Stopped/Compiling/Consumed — caller bug: these states are
    // handled by the retry loop (Consumed -> recreateLane, others -> error)
    // and must not reach the supplementary-start path.  Assert rather than
    // silently recreate, to surface the caller's state-machine bug.
    llvm::errs() << "[ERROR] ensureLaneLoadInit: lane "
                 << (lane == LaneKind::AIC ? "AIC" : "AIV")
                 << " in non-Cold state " << static_cast<int>(st)
                 << "; expected Cold or already-started\n";
    return;
  }
  // Worker is Cold: start it + submit LoadInit with the frozen SO path.
  w.startWorker();
  loadInitTickets_[idx] = w.submitLoadInit(frozenSoPath_);
  emitEvent(lane == LaneKind::AIC ? "AIC" : "AIV", "llc_load", "point",
            "supplementary");
}

void BishengAsyncRuntime::cleanupAfterLaneFailure(
    LaneMask mask, const std::vector<std::string> &plannedOutputs) {
  // One-lane-failure cleanup.  Drain + join all Writers for the lanes in
  // `mask` BEFORE removing planned output files (join-before-remove), then
  // unlink every path in `plannedOutputs` (best-effort; missing files are
  // silently ignored).  This guarantees:
  //   1. The other Lane's in-flight Writer is drained + joined before any
  //      partial `.o` is unlinked (no use-after-free on the output file
  //      being written by a still-running Worker thread).
  //   2. No stale `.o` from a previous attempt remains (a retry that
  //      succeeds would otherwise see the stale `.o` and might falsely pass
  //      or produce a wrong mixed result).
  //   3. Aborts the lane (fail-closed); no silent recovery.
  emitEvent("MAIN", "failure_cleanup", "begin");
  // Step 1: drain retained tickets for the masked lanes (best-effort).
  if (static_cast<uint8_t>(mask) & static_cast<uint8_t>(LaneMask::AIC)) {
    (void)drainLoadInitTicket(LaneKind::AIC);
    (void)drainBaseAuxTicket(LaneKind::AIC);
    (void)drainFinalDeltaTicket(LaneKind::AIC);
  }
  if (static_cast<uint8_t>(mask) & static_cast<uint8_t>(LaneMask::AIV)) {
    (void)drainLoadInitTicket(LaneKind::AIV);
    (void)drainBaseAuxTicket(LaneKind::AIV);
    (void)drainFinalDeltaTicket(LaneKind::AIV);
  }
  // Step 2: shutdown (join) the masked workers so any in-flight Compile
  // finishes writing its `.o` before we unlink.  Join-before-remove.
  if (static_cast<uint8_t>(mask) & static_cast<uint8_t>(LaneMask::AIC))
    workers_[0]->shutdownWorker();
  if (static_cast<uint8_t>(mask) & static_cast<uint8_t>(LaneMask::AIV))
    workers_[1]->shutdownWorker();
  // Step 3: unlink planned output files (best-effort).
  for (const auto &path : plannedOutputs) {
    if (path.empty())
      continue;
    std::error_code ec = llvm::sys::fs::remove(path);
    // Ignore ec: a missing file is fine (the failed lane may not have
    // produced one).  An existing file that we failed to remove is logged.
    if (ec) {
      llvm::errs() << "[async cleanup] warning: failed to remove planned output '"
                   << path << "': " << ec.message() << "\n";
    }
  }
  emitEvent("MAIN", "failure_cleanup", "end");
}

void BishengAsyncRuntime::initTrace() {
  pid_ = static_cast<int>(getpid());
  mainTid_ = getOsTid();
  // run_id = pid + timestamp suffix (short).
  std::ostringstream oss;
  oss << "r" << pid_ << "_" << (nowNs() % 1000000);
  runId_ = oss.str();
  traceEnabled_ = traceEnabled();
  traceFile_ = traceFile();
}

void BishengAsyncRuntime::emitEvent(llvm::StringRef lane, llvm::StringRef phase,
                                    llvm::StringRef event,
                                    llvm::StringRef status) {
  if (!traceEnabled_)
    return;
  std::lock_guard<std::mutex> lk(traceMtx_);
  std::ostringstream os;
  os << "{\"run_id\":\"" << runId_ << "\",\"pid\":" << pid_
     << ",\"tid\":" << mainTid_ << ",\"lane\":\"" << lane.str()
     << "\",\"phase\":\"" << phase.str() << "\",\"event\":\"" << event.str()
     << "\",\"time_ns\":" << nowNs();
  if (!status.empty())
    os << ",\"status\":\"" << status.str() << "\"";
  os << "}\n";
  std::string line = os.str();
  if (!traceFile_.empty()) {
    std::ofstream f(traceFile_, std::ios::app);
    if (f)
      f << line;
  } else {
    llvm::errs() << line;
  }
}

void BishengAsyncRuntime::emitWorkerEvent(LaneKind lane, llvm::StringRef phase,
                                          llvm::StringRef event,
                                          llvm::StringRef status, long lmid,
                                          uint64_t tid) {
  if (!traceEnabled_)
    return;
  std::lock_guard<std::mutex> lk(traceMtx_);
  std::ostringstream os;
  os << "{\"run_id\":\"" << runId_ << "\",\"pid\":" << pid_
     << ",\"tid\":" << tid << ",\"lane\":\"" << laneName(lane)
     << "\",\"phase\":\"" << phase.str() << "\",\"event\":\"" << event.str()
     << "\",\"time_ns\":" << nowNs() << ",\"lmid\":" << lmid;
  if (!status.empty())
    os << ",\"status\":\"" << status.str() << "\"";
  os << "}\n";
  std::string line = os.str();
  if (!traceFile_.empty()) {
    std::ofstream f(traceFile_, std::ios::app);
    if (f)
      f << line;
  } else {
    llvm::errs() << line;
  }
}

LaneMask inferCompileIntent(mlir::ModuleOp mod) {
  // Conservative: if the module has a MIX core type or is a mix entry,
  // warm up both lanes.  Otherwise warm up only the lanes whose bitcode
  // attributes are present.
  bool hasAic = static_cast<bool>(mod->getAttr(mlir::hivm::AIC_BITCODEAttr::name));
  bool hasAiv = static_cast<bool>(mod->getAttr(mlir::hivm::AIV_BITCODEAttr::name));
  bool hasMixAic = static_cast<bool>(mod->getAttr(mlir::hivm::MIX_AIC_BITCODEAttr::name));
  bool hasMixAiv = static_cast<bool>(mod->getAttr(mlir::hivm::MIX_AIV_BITCODEAttr::name));
  bool isMix = mlir::hivm::getModuleCoreTypeAttr(mod) &&
               mlir::hivm::getModuleCoreTypeAttr(mod).getModuleCoreType() ==
                   mlir::hivm::TModuleCoreType::MIX;
  if (isMix || hasMixAic || hasMixAiv)
    return LaneMask::Both;
  uint8_t mask = 0;
  if (hasAic)
    mask |= static_cast<uint8_t>(LaneMask::AIC);
  if (hasAiv)
    mask |= static_cast<uint8_t>(LaneMask::AIV);
  if (mask == 0)
    return LaneMask::Both; // uncertain → conservative
  return static_cast<LaneMask>(mask);
}

} // namespace bishengir