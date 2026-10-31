//===- BufferLifeVec.h - Lives grouped by interval ---------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENG_DIALECT_HIVM_TRANSFORMS_BUFFER_LIFE_VEC_H
#define BISHENG_DIALECT_HIVM_TRANSFORMS_BUFFER_LIFE_VEC_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <optional>
#include <utility>

namespace mlir {
namespace hivm {

/// One buffer and the alloc/free times that belong to that buffer.
/// These times are not rewritten to the start and end of a merged span.
struct BufferLifeView {
  Value buffer;
  int64_t allocTime = -1;
  int64_t freeTime = -1;
};

/// Buffer lifetimes stored in a tree ordered by interval start.
///
/// Storage is `std::map<int64_t, Interval>`, a red-black tree. Each node is
/// one pair: the key (`first`) is the interval start, the value (`second`)
/// is the interval. The tree keeps nodes sorted by that key, so a walk from
/// `begin()` to `end()` visits intervals from the earliest start to the
/// latest. There is no "element i". Insert looks the start up with `find`
/// and adds a node with `emplace`. `buffersAt` uses `upper_bound` to find
/// the interval covering a time.
///
/// A cursor into the tree is an iterator. `begin()` points at the smallest
/// key. `end()` is a sentinel past the last node, not a real node; a loop
/// stops when the cursor compares equal to `end()`. `++it` moves to the next
/// larger key. `std::prev(it)` moves one key backward. `it->first` reads the
/// key and `it->second` reads the interval.
///
/// Touching intervals, including two that only meet as `end + 1 == start`,
/// are merged on insert. After the merge the remaining nodes do not touch, so
/// each start is unique and can be used as the map key.
///
/// A node keeps two lists of buffers:
///   * `started`: buffers whose alloc time equals the interval start. The
///     extra time on each entry is that buffer's own free time.
///   * `ended`: buffers whose free time equals the interval end. The extra
///     time is that buffer's own alloc time.
/// A buffer that fills the whole interval is in both lists.
///
/// Example: `(1,2,A) (2,3,B) (3,4,C) (4,4,D)` becomes one node `[1,4]` with
/// started `{A}` and ended `{C,D}`. B is dropped because it neither starts at
/// 1 nor ends at 4. A later insert that moves the end past 4 drops C and D
/// too. Times stored on the kept buffers stay the originals.
///
/// `llvm::IntervalMap` is also a tree of intervals, but it folds touching
/// intervals into a single value and cannot store these two lists separately.
class BufferLifeVec {
public:
  /// Insert one life and merge it with every stored interval it touches.
  /// `std::move` gives `interval` to `insertInterval` without copying its
  /// buffer lists. `interval` itself is empty afterwards.
  void append(Value buffer, int64_t allocTime, int64_t freeTime) {
    Interval interval;
    interval.start = allocTime;
    interval.end = freeTime;
    interval.buffersTouchingStart.push_back(Endpoint{buffer, freeTime});
    interval.buffersTouchingEnd.push_back(Endpoint{buffer, allocTime});
    insertInterval(std::move(interval));
  }

  /// Merge every interval of `other` into this tree.
  ///
  /// `for (const auto &entry : other.byStart)` walks that tree from the
  /// smallest start to the largest. `entry` is the node's (key, interval)
  /// pair, so `entry.second` is the interval. The call copies it into this
  /// tree. `other` must not be `*this`: inserting rebalances the tree the
  /// loop is reading, and that invalidates the cursor.
  void append(const BufferLifeVec &other) {
    for (const auto &entry : other.byStart)
      insertInterval(entry.second);
  }

  /// Exchange the two trees. No interval is copied.
  void swap(BufferLifeVec &other) { std::swap(byStart, other.byStart); }

  bool empty() const { return byStart.empty(); }

  /// The buffer of a memory-unique entry, which has a single buffer.
  /// `begin()` is the node with the smallest start. `->second` is the
  /// interval stored in that node.
  Value frontBuffer() const {
    const Interval &interval = byStart.begin()->second;
    if (!interval.buffersTouchingStart.empty())
      return interval.buffersTouchingStart.front().buffer;
    return interval.buffersTouchingEnd.front().buffer;
  }

  /// Call `fn` once per stored buffer, in order of interval start.
  /// A buffer listed in both `started` and `ended` fills the whole interval
  /// and is reported only from `started`.
  template <typename Fn> void forEach(Fn &&fn) const {
    for (const auto &entry : byStart) {
      const Interval &interval = entry.second;
      for (const Endpoint &ep : interval.buffersTouchingStart)
        fn(BufferLifeView{ep.buffer, interval.start, ep.otherTime});
      for (const Endpoint &ep : interval.buffersTouchingEnd) {
        // Same buffer as the `started` entry: alloc time is this start.
        if (ep.otherTime == interval.start)
          continue;
        fn(BufferLifeView{ep.buffer, ep.otherTime, interval.end});
      }
    }
  }

  /// True when `fn` returns true for every stored buffer. Stops at the first
  /// false result. Same buffer order as `forEach`.
  template <typename Fn> bool allOf(Fn &&fn) const {
    for (const auto &entry : byStart) {
      const Interval &interval = entry.second;
      for (const Endpoint &ep : interval.buffersTouchingStart) {
        if (!fn(BufferLifeView{ep.buffer, interval.start, ep.otherTime}))
          return false;
      }
      for (const Endpoint &ep : interval.buffersTouchingEnd) {
        if (ep.otherTime == interval.start)
          continue;
        if (!fn(BufferLifeView{ep.buffer, ep.otherTime, interval.end}))
          return false;
      }
    }
    return true;
  }

  /// The stored life of `buffer`, or an empty result when this buffer is not
  /// an endpoint of any interval. `std::optional` holds a value or nothing;
  /// `if (!found)` is the "nothing" case. An interior buffer dropped by merge
  /// is in that case.
  std::optional<BufferLifeView> find(Value buffer) const {
    std::optional<BufferLifeView> found;
    forEach([&](BufferLifeView life) {
      if (!found && life.buffer == buffer)
        found = life;
    });
    return found;
  }

  /// True when any stored intervals overlap, including a touch at one time.
  /// The two cursors start at the earliest interval of each tree. On each
  /// step the cursor whose interval ends first moves forward (`++`), which is
  /// the same walk as merging two sorted lists. `end()` means that cursor has
  /// passed the last node.
  bool intersects(const BufferLifeVec &other) const {
    if (empty() || other.empty())
      return false;
    auto lhs = byStart.begin();
    auto rhs = other.byStart.begin();
    while (lhs != byStart.end() && rhs != other.byStart.end()) {
      int64_t lo = std::max(lhs->second.start, rhs->second.start);
      int64_t hi = std::min(lhs->second.end, rhs->second.end);
      if (lo <= hi)
        return true;
      if (lhs->second.end < rhs->second.end)
        ++lhs;
      else
        ++rhs;
    }
    return false;
  }

  /// Each closed time overlap `[lo, hi]` of the stored intervals.
  /// `lo == hi` is a single time. That includes a `[t, t]` interval sitting
  /// strictly inside a longer interval of `other`: the overlap is `[t, t]`,
  /// not the longer span. The walk is the same two-cursor walk as
  /// `intersects`.
  SmallVector<std::pair<int64_t, int64_t>>
  timeOverlaps(const BufferLifeVec &other) const {
    SmallVector<std::pair<int64_t, int64_t>> overlaps;
    if (empty() || other.empty())
      return overlaps;
    auto lhsIt = byStart.begin();
    auto rhsIt = other.byStart.begin();
    while (lhsIt != byStart.end() && rhsIt != other.byStart.end()) {
      const Interval &lhs = lhsIt->second;
      const Interval &rhs = rhsIt->second;
      int64_t lo = std::max(lhs.start, rhs.start);
      int64_t hi = std::min(lhs.end, rhs.end);
      if (lo <= hi)
        overlaps.emplace_back(lo, hi);
      if (lhs.end < rhs.end)
        ++lhsIt;
      else
        ++rhsIt;
    }
    return overlaps;
  }

  /// Buffers whose own alloc/free range contains `time`.
  ///
  /// `upper_bound(time)` returns a cursor to the first node whose start is
  /// strictly greater than `time`, or `end()` when every start is `<= time`.
  /// The only node that can cover `time` is the one just before that cursor.
  /// `--it` steps back to it. If that node's end is still before `time`, no
  /// stored interval covers `time`.
  ///
  /// A `[t, t]` node has start `t`, so this finds it: it is the last node
  /// whose start is `<= t`. The other tree's interval may have started much
  /// earlier; that does not matter, because each tree is searched on its own.
  SmallVector<BufferLifeView> buffersAt(int64_t time) const {
    SmallVector<BufferLifeView> lives;
    auto it = byStart.upper_bound(time);
    if (it == byStart.begin())
      return lives;
    --it;
    const Interval &interval = it->second;
    if (interval.end < time)
      return lives;
    for (const Endpoint &ep : interval.buffersTouchingStart) {
      if (time <= ep.otherTime)
        lives.push_back(
            BufferLifeView{ep.buffer, interval.start, ep.otherTime});
    }
    for (const Endpoint &ep : interval.buffersTouchingEnd) {
      // Already reported from `started` when the buffer fills the interval.
      if (ep.otherTime == interval.start)
        continue;
      if (ep.otherTime <= time)
        lives.push_back(BufferLifeView{ep.buffer, ep.otherTime, interval.end});
    }
    return lives;
  }

private:
  /// One buffer tied to an interval endpoint.
  /// In `buffersTouchingStart`, `otherTime` is the buffer's free time.
  /// In `buffersTouchingEnd`, `otherTime` is the buffer's alloc time.
  struct Endpoint {
    Value buffer;
    int64_t otherTime = -1;
  };

  /// One merged span. `buffersTouchingStart` / `buffersTouchingEnd` are small
  /// arrays (`SmallVector`): `push_back` appends, `front` is the first element,
  /// and a range-for walks them in insertion order.
  struct Interval {
    int64_t start = -1;
    int64_t end = -1;
    SmallVector<Endpoint, 1> buffersTouchingStart;
    SmallVector<Endpoint, 1> buffersTouchingEnd;
  };

  /// Key = interval start. Nodes do not touch, so the key is unique.
  std::map<int64_t, Interval> byStart;

  /// True when `eps` already lists `buffer`. The list is the buffers that
  /// touch one endpoint, so it stays short and a scan is enough.
  static bool containsBuffer(const SmallVectorImpl<Endpoint> &eps,
                             Value buffer) {
    for (const Endpoint &ep : eps)
      if (ep.buffer == buffer)
        return true;
    return false;
  }

  static void appendNew(SmallVectorImpl<Endpoint> &dst,
                        const SmallVectorImpl<Endpoint> &src) {
    for (const Endpoint &ep : src)
      if (!containsBuffer(dst, ep.buffer))
        dst.push_back(ep);
  }

  /// Fold `src` into `dst` in place. `src` starts at or after `dst`, so
  /// `dst.start` (the map key) does not change. No new `Interval` is built.
  /// A buffer is kept only when it still touches the union's start or end.
  static void absorb(Interval &dst, Interval &&src) {
    if (src.start == dst.start)
      appendNew(dst.buffersTouchingStart, src.buffersTouchingStart);
    if (src.end > dst.end) {
      dst.end = src.end;
      dst.buffersTouchingEnd = std::move(src.buffersTouchingEnd);
    } else if (src.end == dst.end) {
      appendNew(dst.buffersTouchingEnd, src.buffersTouchingEnd);
    }
  }

  /// Insert `incoming` and absorb every stored node it touches.
  ///
  /// Stored nodes do not touch each other, so the new interval meets at most
  /// one node on the left and a run of nodes on the right. The surviving node
  /// is always the one whose start is the union start; the others are erased.
  void insertInterval(Interval incoming) {
    // lower_bound: first node whose start is >= incoming.start, or end().
    auto it = byStart.lower_bound(incoming.start);
    if (it != byStart.end() && it->first == incoming.start) {
      // This start is already a key. Fold into that node.
      absorb(it->second, std::move(incoming));
    } else if (it != byStart.begin() &&
               std::prev(it)->second.end + 1 >= incoming.start) {
      // Empty tree: begin()==end(), so prev is not called. When every stored
      // start is smaller, it==end() and prev is the last node, which can
      // still touch incoming.
      // The previous node already touches this start. Its key is the union
      // start, so keep that node and point `it` at it. incoming is not
      // inserted as its own node.
      it = std::prev(it);
      absorb(it->second, std::move(incoming));
    } else {
      int64_t start = incoming.start;
      // emplace_hint inserts just before `it` and returns the new node.
      it = byStart.emplace_hint(it, start, std::move(incoming));
    }

    // Nodes with a later start. Absorbing one does not change this key.
    // Erasing `next` leaves `it` valid.
    auto next = std::next(it);
    while (next != byStart.end() && next->second.start <= it->second.end + 1) {
      absorb(it->second, std::move(next->second));
      byStart.erase(next);
      next = std::next(it);
    }
  }
};

} // namespace hivm
} // namespace mlir

#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_BUFFER_LIFE_VEC_H
