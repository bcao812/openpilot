#pragma once

#include <cstdint>

// Only the high-resolution road files follow this gate. The encoder, qcamera,
// camera timestamps and other rlog services continue while longitudinal is off.
class RoadRecordingGate {
public:
  static constexpr uint64_t GRACE_NS = 10ULL * 1000 * 1000 * 1000;
  static constexpr uint64_t MAX_MESSAGE_AGE_NS = 1000ULL * 1000 * 1000;

  void update(bool long_active, bool valid, uint64_t message_ns, uint64_t now_ns) {
    if (!valid || message_ns == 0 || message_ns <= last_message_ns_ ||
        message_ns > now_ns || now_ns - message_ns > MAX_MESSAGE_AGE_NS) return;

    const bool window_open = message_ns < record_until_ns_;
    if (long_active) {
      if (!window_open) ++generation_;
      record_until_ns_ = message_ns + GRACE_NS;
    } else if (long_active_ && window_open) {
      // Only the transition extends the tail; repeated inactive packets do not.
      record_until_ns_ = message_ns + GRACE_NS;
    }
    long_active_ = long_active;
    last_message_ns_ = message_ns;
  }

  bool should_record(uint64_t now_ns) const { return now_ns < record_until_ns_; }
  uint64_t generation() const { return generation_; }

private:
  uint64_t last_message_ns_ = 0;
  uint64_t record_until_ns_ = 0;
  uint64_t generation_ = 0;
  bool long_active_ = false;
};
