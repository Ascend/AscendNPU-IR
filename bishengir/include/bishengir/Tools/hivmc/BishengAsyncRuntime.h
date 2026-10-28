//===- BishengAsyncRuntime.h - True-async Worker runtime for v2 LLC bridge ===//
//
// True-async scheduling layer over the synchronous BishengLlcBridge (v2
// libbisheng-llc.so, 4-symbol C ABI).  Provides two per-lane Worker threads
// (AIC / AIV), each owning one dlmopen namespace + one BishengLlcContext,
// a command queue with generation-based tickets, and early-start support
// so that dlmopen + dlsym + bishengLlcInit overlap with the main thread's
// MLIR lowering.
//
// Worker commands (FIFO per lane, ordered by submission):
//   LoadInit           dlmopen + four dlsym + bishengLlcInit (no aux preload).
//                      Submit-able before any aux path is known.
//   PreloadBase        Preload distinct base aux paths (after they are
//                      resolved by the path calculator).
//   PreloadFinalDelta  Preload remaining delta aux (after the final topology
//                      is known).
//   Compile            Preload remaining delta + bishengLlcCompile (consumes
//                      the context).
//   Shutdown           Destroy + join.
//
// Design invariants:
//   - One BishengLlcWorker per lane; the worker thread owns ALL Vendor API
//     calls (dlmopen, dlsym, init, preload, compile, destroy, dlclose).
//     The main thread only submits commands and waits on tickets.
//   - Strict FIFO command queue per worker: LoadInit -> PreloadBase ->
//     PreloadFinalDelta -> Compile -> Shutdown, ordered by submission.
//     No need to wait between submits; the queue guarantees order.
//   - Generation-based tickets avoid stale-completion pollution: each
//     command has a unique generation, and wait* matches by generation.
//   - Two overlap windows:
//       Early Start:  Main lowers MLIR  ||  Worker does LoadInit (+ PreloadBase).
//       Backend:       Main serializes lane N+1 IR  ||  Worker compiles lane N.
//   - compile() receives the COMPLETE ordered aux path list (duplicates
//     preserved); preload() is only a cache hint for distinct paths.
//   - One compile per context (success or failure); retry requires a new
//     worker + new namespace + new context (recreateLane).
//   - Failure cleanup: drain/wait every submitted ticket, join workers,
//     THEN remove planned output files (join-before-remove).
//   - fail-closed: any Worker stage failure propagates to the main thread
//     via drain*/wait* returning false; the caller surfaces the error and
//     aborts the lane (no silent recovery).
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TOOLS_HIVMC_BISHENG_ASYNC_RUNTIME_H
#define BISHENGIR_TOOLS_HIVMC_BISHENG_ASYNC_RUNTIME_H

#include "bishengir/Tools/Utils/Utils.h" // SubCoreTarget, IRFilePair
#include "bishengir/Tools/hivmc/BishengLlcBridge.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace bishengir {

/// Lane kind (AIC / AIV).
enum class LaneKind : int { AIC = 0, AIV = 1 };

/// Worker execution state (per lane).  Updated only on the worker thread
/// except for the atomic readback used by trace/diagnostics.
enum class WorkerState {
  Cold,          ///< Worker thread not started.
  Starting,      ///< Worker thread running, bridge load in progress.
  Ready,         ///< Bridge loaded + init done.
  BaseAuxReady,  ///< Base aux preload done.
  Compiling,     ///< Compile in progress.
  Consumed,      ///< Compile finished (success or failure); context consumed.
  Stopped,       ///< Worker joined.
  Failed,        ///< Unrecoverable failure.
};

/// Ticket returned by submit* calls.  The generation uniquely identifies the
/// command so wait* avoids consuming a stale completion from a previous
/// command.
struct CommandTicket {
  uint64_t generation = 0;
  bool valid() const { return generation != 0; }
};

/// Completion record for one command generation.
struct CommandCompletion {
  bool done = false;
  bool success = false;
  BishengLlcCompileResult compileResult; ///< Only meaningful for Compile.
};

/// Compile intent: early lane mask, decided before MLIR lowering.
enum class LaneMask : uint8_t {
  None = 0,
  AIC = 1,
  AIV = 2,
  Both = 3,
};

/// Compile-scope async runtime owning two lane workers (AIC / AIV) and the
/// retained LoadInit / PreloadBase / PreloadFinalDelta tickets.  Created by
/// the A3/A5 CLI entry before MLIR lowering (native mode only); destroyed
/// after device compile completes.
class BishengAsyncRuntime;

/// One per-lane worker owning a BishengLlcBridge (v2) and a command queue.
/// The Worker commands are LoadInit / PreloadBase / PreloadFinalDelta /
/// Compile / Shutdown (see WorkerCommand::Kind below).
class BishengLlcWorker {
public:
  explicit BishengLlcWorker(LaneKind kind, BishengAsyncRuntime *runtime = nullptr);
  ~BishengLlcWorker();

  BishengLlcWorker(const BishengLlcWorker &) = delete;
  BishengLlcWorker &operator=(const BishengLlcWorker &) = delete;

  LaneKind kind() const { return kind_; }

  /// Start the worker thread.  The thread begins loading the bridge
  /// (dlmopen + dlsym + init) as soon as a LoadInit command is submitted;
  /// startWorker alone does not block.  Idempotent.
  void startWorker();

  bool isWorkerStarted() const {
    return workerState_.load() != WorkerState::Cold;
  }

  /// Submit a LoadInit command: dlmopen the SO + four dlsym + bishengLlcInit.
  /// Does NOT preload any aux (LoadInit must be independent of aux paths).
  /// \p soPath  frozen absolute path to libbisheng-llc.so (typically
  ///            resolved once by the runtime from BISHENG_LLC_LIBRARY env).
  /// Returns a ticket; the main thread does NOT wait immediately.
  CommandTicket submitLoadInit(llvm::StringRef soPath);

  /// Submit a PreloadBase command: preload distinct base aux paths into the
  /// vendor cache (cache hint only).  Must follow LoadInit (FIFO guarantees
  /// LoadInit is processed first).  Worker must be Ready (LoadInit done) for
  /// preload to succeed; otherwise the command fails fail-closed.
  CommandTicket submitPreloadBase(const std::vector<std::string> &baseAux);

  /// Submit a PreloadFinalDelta command: preload delta aux paths (paths not
  /// yet preloaded) into the vendor cache.  Cache hint only; does not change
  /// link order or consume the context.
  CommandTicket submitPreloadFinalDelta(
      const std::vector<std::string> &deltaAux);

  /// Submit a Compile command: preload any delta aux (paths in \p fullAux
  /// not yet preloaded), then compile.  Consumes the context.
  CommandTicket submitCompile(const std::string &irBytes,
                              const llvm::SmallVectorImpl<std::string> &fullAux,
                              llvm::StringRef targetCpu, uint32_t optLevel,
                              const std::vector<std::string> &extraOpts,
                              llvm::StringRef outputPath);

  /// Wait for a LoadInit command to complete.  Returns true on success,
  /// false on failure or worker termination.  Consumes the ticket.
  bool waitLoadInit(CommandTicket ticket);

  /// Wait for a PreloadBase command to complete.  Returns true on success
  /// (or when no ticket is outstanding).  Consumes the ticket.
  bool waitPreloadBase(CommandTicket ticket);

  /// Wait for a PreloadFinalDelta command to complete.  Returns true on
  /// success (or when no ticket is outstanding).  Consumes the ticket.
  bool waitPreloadFinalDelta(CommandTicket ticket);

  /// Wait for a Compile command to complete.  Returns the compile result
  /// (check `success`).  Consumes the ticket.
  BishengLlcCompileResult waitCompile(CommandTicket ticket);

  /// Permanently shut down the worker thread (submit Shutdown + join).
  /// Used by cleanupDirectFailure and runtime teardown.
  void shutdownWorker();

  /// Reset this lane so a fresh Compile can run on a NEW Vendor context
  /// + NEW dlmopen namespace.  Called after the previous Compile consumed the
  /// context (workerState_ == Consumed), so the A5 multi-attempt loop can
  /// submit a new LoadInit + PreloadBase + PreloadFinalDelta + Compile
  /// sequence on this lane without leaking the old namespace/context.
  /// Idempotent: a Cold/Stopped worker is a no-op.
  ///
  /// Steps (join-before-recreate):
  ///   1. Drain any in-flight command (best-effort: drop queued commands and
  ///      wait for the worker thread to finish the current one).
  ///   2. Submit Shutdown and join the worker thread.
  ///   3. Replace `bridge_` with a fresh BishengLlcBridge (new dlmopen handle
  ///      on next load, new LMID).
  ///   4. Reset worker state to Cold, clear `preloadedPaths_` and pending
  ///      command queue / completions.  Generation tracking is kept
  ///      monotonically increasing (NOT reset) so ticket values remain
  ///      unique identifiers for the worker's lifetime.
  ///   5. The caller must then submit LoadInit + PreloadBase + ... again
  ///      (BishengAsyncRuntime::recreateLane does this for the runtime's
  ///      retained tickets).
  ///
  /// After reset, `isWorkerStarted()` returns false (state == Cold) and the
  /// worker thread is not running.  `startWorker()` + `submitLoadInit()` must
  /// be called before any Preload/Compile.
  void resetForRetry();

  /// Worker OS thread id (for trace).  Valid after startWorker.
  uint64_t workerTid() const { return workerTid_.load(); }

  /// LMID of this worker's dlmopen namespace (for trace).  Valid after the
  /// bridge is loaded (Ready state).
  long lmid() const { return lmid_.load(); }

  /// Vendor context Init/Destroy pairing counters (forwarded from the bridge).
  /// At quiescence initCount() == destroyCount(); a mismatch indicates a
  /// context leak (init > destroy) or a double-destroy (init < destroy).
  uint64_t initCount() const { return bridge_.initCount(); }
  uint64_t destroyCount() const { return bridge_.destroyCount(); }

  WorkerState state() const { return workerState_.load(); }

private:
  struct WorkerCommand {
    enum class Kind {
      LoadInit,           ///< dlmopen + dlsym + bishengLlcInit (no aux preload).
      PreloadBase,        ///< Preload distinct base aux paths (after LoadInit).
      PreloadFinalDelta,  ///< Preload delta aux paths (cache hint only).
      Compile,            ///< Preload remaining delta + compile (consumes ctx).
      Shutdown,           ///< Destroy bridge and exit the worker loop.
    } kind;
    uint64_t generation = 0;
    // LoadInit input
    std::string soPath;
    // PreloadBase / PreloadFinalDelta inputs
    std::vector<std::string> baseAux;       ///< PreloadBase distinct paths.
    std::vector<std::string> finalDeltaAux; ///< PreloadFinalDelta delta paths.
    // Compile inputs
    std::string irBytes;
    llvm::SmallVector<std::string, 4> fullAux;
    std::string targetCpu;
    uint32_t optLevel = 2;
    std::vector<std::string> extraOpts;
    std::string outputPath;
  };

  void workerLoop();
  void handleLoadInit(const WorkerCommand &cmd);
  void handlePreloadBase(const WorkerCommand &cmd);
  void handlePreloadFinalDelta(const WorkerCommand &cmd);
  void handleCompile(const WorkerCommand &cmd, CommandCompletion &completion);
  void completeCommand(uint64_t generation, bool success,
                       const BishengLlcCompileResult &compileResult);
  bool waitCommand(CommandTicket ticket, CommandCompletion &out);

  LaneKind kind_;
  // Non-owning back-pointer to the owning runtime, used only for trace
  // emission. Lifetime invariant: the runtime (owned by CompileScope) outlives
  // all workers — workers are joined in ~BishengAsyncRuntime before the
  // runtime is destroyed, so runtime_ never dangles.
  BishengAsyncRuntime *runtime_ = nullptr;
  BishengLlcBridge bridge_;
  std::thread workerThread_;
  std::atomic<WorkerState> workerState_{WorkerState::Cold};
  std::atomic<uint64_t> workerTid_{0};
  std::atomic<long> lmid_{0};

  std::deque<WorkerCommand> commandQueue_;
  std::map<uint64_t, CommandCompletion> completions_;
  uint64_t commandGeneration_ = 0;
  // Paths already preloaded on this worker (for delta preload).
  std::vector<std::string> preloadedPaths_;

  mutable std::mutex mtx_;
  std::condition_variable cv_;
};

/// Compile-scope async runtime owning two lane workers (AIC / AIV) and the
/// retained LoadInit / PreloadBase / PreloadFinalDelta tickets.  Created by
/// the A3/A5 CLI entry before MLIR lowering (native mode only); destroyed
/// after device compile completes.
class BishengAsyncRuntime {
public:
  /// \p soPath is the absolute path to libbisheng-llc.so frozen for the
  /// whole compile scope; workers dlmopen this path during LoadInit.
  explicit BishengAsyncRuntime(llvm::StringRef soPath);
  ~BishengAsyncRuntime();

  BishengAsyncRuntime(const BishengAsyncRuntime &) = delete;
  BishengAsyncRuntime &operator=(const BishengAsyncRuntime &) = delete;

  BishengLlcWorker &aicWorker() { return *workers_[0]; }
  BishengLlcWorker &aivWorker() { return *workers_[1]; }

  BishengLlcWorker &getWorkerForCore(SubCoreTarget core) {
    if (core == SubCoreTarget::AIC || core == SubCoreTarget::MIX_AIC)
      return *workers_[0];
    return *workers_[1];
  }

  /// Start workers for the lanes in \p mask and submit LoadInit commands
  /// (dlmopen + dlsym + bishengLlcInit, no aux preload).  The runtime's
  /// frozen SO path is used (resolved from BISHENG_LLC_LIBRARY env at
  /// construction).  Retains the LoadInit tickets for later draining.
  void submitLoadInit(LaneMask mask);

  /// Submit PreloadBase for the lanes in \p mask with their distinct base
  /// aux paths.  Retains the BaseAux tickets for later draining.
  void submitPreloadBase(LaneMask mask,
                         const std::vector<std::string> &aicBaseAux,
                         const std::vector<std::string> &aivBaseAux);

  /// Submit PreloadFinalDelta for one \p lane.
  void submitPreloadFinalDelta(LaneKind lane,
                               const std::vector<std::string> &deltaAux);

  /// Drain (wait) the retained LoadInit ticket for one lane.  Returns true
  /// when no LoadInit is outstanding or it succeeded.
  bool drainLoadInitTicket(LaneKind lane);

  /// Drain (wait) the retained PreloadBase ticket for one lane.  Returns
  /// true when no PreloadBase is outstanding or it succeeded.
  bool drainBaseAuxTicket(LaneKind lane);

  /// Drain (wait) the retained PreloadFinalDelta ticket for one lane.
  /// Returns true when no PreloadFinalDelta is outstanding or it succeeded.
  bool drainFinalDeltaTicket(LaneKind lane);

  /// Drain the retained PreloadFinalDelta ticket for one lane WITHOUT
  /// propagating failure to the caller's compile path.  Used at the top of
  /// A5's per-attempt `runDeviceBiShengLIRCompile` so a previous attempt's
  /// `finalDeltaTickets_[lane]` is consumed (waited on) BEFORE being
  /// overwritten by a new `submitPreloadFinalDelta`.  Ensures the old ticket
  /// is always drained (and its completion waited on) before a new one is
  /// retained.  Returns true if the old ticket succeeded or was absent;
  /// returns false if the old ticket failed (the lane is in a bad state and
  /// the caller should `recreateLane` before reusing it).  Either way the
  /// retained ticket slot is cleared.
  bool drainFinalDeltaTicketForRetry(LaneKind lane);

  /// Drain all retained tickets (LoadInit + PreloadBase + PreloadFinalDelta
  /// for both lanes).  Returns false if any failed.
  bool drainAllTickets();

  /// Recreate one lane's worker so a fresh Compile can run on a NEW
  /// Vendor context + NEW dlmopen namespace.  Required after a Compile
  /// consumed the context (workerState_ == Consumed), so the A5 multi-attempt
  /// loop can re-LoadInit + re-PreloadBase + re-PreloadFinalDelta + re-Compile
  /// on this lane without leaking the old namespace/context/TLS.
  ///
  /// Steps (join-before-recreate):
  ///   1. Drain (wait) this lane's retained LoadInit/PreloadBase/PreloadFinal
  ///      Delta tickets (best-effort; clears the retained slots).
  ///   2. Call the worker's `resetForRetry()` (drain in-flight + Shutdown +
  ///      join + fresh bridge + reset state).
  ///   3. Clear this lane's retained tickets (loadInitTickets_/baseAuxTickets_
  ///      /finalDeltaTickets_).
  /// The caller must then re-submit LoadInit + PreloadBase before any
  /// Compile on this lane.
  void recreateLane(LaneKind lane);

  /// Safe supplementary start for a lane whose LoadInit was never
  /// submitted (e.g. early topology was unknown and only one lane was
  /// initially warmed up, but the final topology turns out to need the
  /// other lane).  Submits LoadInit for \p lane if its worker is still Cold
  /// (LoadInit never submitted).  Idempotent: a no-op when the worker is
  /// already alive (Starting/Ready/BaseAuxReady).  Callers must NOT route
  /// Failed/Stopped/Compiling/Consumed lanes here — the A5 retry loop
  /// handles those (Consumed -> recreateLane, others -> error).
  void ensureLaneLoadInit(LaneKind lane);

  /// One-lane-failure cleanup.  Drain + join all Writers for the lanes
  /// in \p mask BEFORE removing planned output files (join-before-remove),
  /// then unlink every path in \p plannedOutputs (best-effort; missing files
  /// are silently ignored).  Used by `lowerDeviceToBinary` (A3/A5) when one
  /// Lane's Compile failed, so the other Lane's in-flight Writer is drained
  /// and joined before any partial `.o` is unlinked, and no stale `.o` from
  /// a previous attempt remains.
  void cleanupAfterLaneFailure(LaneMask mask,
                               const std::vector<std::string> &plannedOutputs);

  /// Shutdown both workers (join).  Called by destructor and failure cleanup.
  void shutdownAll();

  /// Whether any worker has been started (runtime is active).
  bool isActive() const {
    return workers_[0]->isWorkerStarted() || workers_[1]->isWorkerStarted();
  }

  /// Frozen absolute path to libbisheng-llc.so (resolved once at
  /// construction from BISHENG_LLC_LIBRARY env).  Empty when the env was
  /// unset; in that case the bridge falls back to reading env on load.
  llvm::StringRef frozenSoPath() const { return frozenSoPath_; }

  /// Trace helpers. Emit a JSONL event line to the trace sink (stderr when
  /// BISHENG_ASYNC_TRACE=1, or a file when BISHENG_ASYNC_TRACE_FILE set).
  void emitEvent(llvm::StringRef lane, llvm::StringRef phase,
                 llvm::StringRef event, llvm::StringRef status = "ok");
  void emitWorkerEvent(LaneKind lane, llvm::StringRef phase,
                       llvm::StringRef event, llvm::StringRef status,
                       long lmid, uint64_t tid);

  /// Record the run id (for trace correlation). Defaults to pid.
  void initTrace();

  /// Free-function trace helpers that accept a nullptr `runtime` and no-op.
  /// Lets call sites write `emitTrace(runtime, ...)` without an outer
  /// `if (runtime)` guard.
  static inline void emitTrace(BishengAsyncRuntime *runtime, llvm::StringRef lane,
                               llvm::StringRef phase, llvm::StringRef event,
                               llvm::StringRef status = "ok") {
    if (runtime)
      runtime->emitEvent(lane, phase, event, status);
  }
  static inline void emitTrace(BishengAsyncRuntime *runtime, LaneKind lane,
                               llvm::StringRef phase, llvm::StringRef event,
                               llvm::StringRef status, long lmid, uint64_t tid) {
    if (runtime)
      runtime->emitWorkerEvent(lane, phase, event, status, lmid, tid);
  }

private:
  std::array<std::unique_ptr<BishengLlcWorker>, 2> workers_;
  // Per-lane retained tickets for LoadInit / PreloadBase / PreloadFinalDelta.
  // An invalid (generation == 0) ticket means "nothing outstanding"; the
  // corresponding drain* returns true.
  std::array<CommandTicket, 2> loadInitTickets_{};
  std::array<CommandTicket, 2> baseAuxTickets_{};
  std::array<CommandTicket, 2> finalDeltaTickets_{};
  // Frozen absolute path to libbisheng-llc.so (resolved once at construction
  // by the caller and passed to the ctor).  Workers dlmopen this path.
  std::string frozenSoPath_;
  // Trace state
  std::string runId_;
  uint64_t mainTid_ = 0;
  int pid_ = 0;
  std::mutex traceMtx_;
  // Atomic so the owner-thread write in initTrace() and the worker-thread
  // reads in emitEvent/emitWorkerEvent form a well-defined happens-before
  // relationship (no data-race UB).
  std::atomic<bool> traceEnabled_{false};
  std::string traceFile_;
};

/// Infer the early compile intent (lane mask) from a module, before MLIR
/// lowering.  Conservative: when the module is (or may be) mix, returns Both.
LaneMask inferCompileIntent(mlir::ModuleOp mod);

} // namespace bishengir

#endif // BISHENGIR_TOOLS_HIVMC_BISHENG_ASYNC_RUNTIME_H