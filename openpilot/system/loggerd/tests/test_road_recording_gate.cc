#include <cassert>
#include <cstdint>

#include "../road_recording.h"

int main() {
  constexpr uint64_t S = 1000000000ULL;
  RoadRecordingGate gate;
  assert(!gate.should_record(S));
  gate.update(false, true, S, S);
  assert(!gate.should_record(S));  // startup and lateral-only do not start a tail
  gate.update(true, false, 2*S, 2*S);
  gate.update(true, true, 4*S, 3*S);  // future message
  gate.update(true, true, S, 3*S);    // stale message
  assert(!gate.should_record(3*S));

  gate.update(true, true, 4*S, 4*S);
  assert(gate.should_record(4*S));
  assert(gate.generation() == 1);
  gate.update(false, true, 5*S, 5*S);
  gate.update(false, true, 9*S, 9*S);
  assert(gate.should_record(15*S - 1));
  assert(!gate.should_record(15*S));  // ten seconds, not ten seconds per message

  gate.update(true, true, 16*S, 16*S);
  assert(gate.generation() == 2);
  gate.update(false, true, 17*S, 17*S);
  gate.update(true, true, 20*S, 20*S);
  assert(gate.generation() == 2);  // a brief override does not interrupt the GOP
  gate.update(false, true, 19*S, 20*S);  // old inactive packet cannot end control
  gate.update(true, true, 20*S, 21*S);   // duplicate cannot extend the timeout
  assert(gate.should_record(30*S - 1));
  assert(!gate.should_record(30*S));    // silent publisher also times out
  gate.update(false, true, 31*S, 31*S);
  assert(!gate.should_record(31*S));    // inactive reconnect must not restart video
  gate.update(true, true, 32*S, 32*S);
  assert(gate.generation() == 3);      // resume needs a keyframe even with no video in the gap
  gate.update(false, true, 33*S, 33*S);
  gate.update(true, true, 31*S, 34*S);  // stale active packet cannot reopen the gate
  gate.update(true, false, 40*S, 40*S);
  assert(!gate.should_record(43*S));
}
