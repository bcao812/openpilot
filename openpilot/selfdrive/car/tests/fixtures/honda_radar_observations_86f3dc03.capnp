# Frozen wire-compatibility fixture from:
# 86f3dc03e4bbbe7816da3185f3d9b1b14c11758a:openpilot/cereal/custom.capnp
# HondaRadarObservations is copied verbatim; unrelated structs and
# C++ annotations are omitted. Keep the original file and nested type IDs.
# Do not regenerate this baseline from the schema under test.

@0xb526ba661d550a59;

struct HondaRadarObservations {
  schemaVersion @0 :UInt16 = 1;
  decoderVersion @1 :Text;
  sessionId @2 :Text;
  sequence @3 :UInt64;
  active @4 :Bool;
  reason @5 :Text;
  radarBus @6 :UInt8;
  lastRadarFrameMonoTime @7 :UInt64;
  lastPositionMonoTime @8 :UInt64;
  observations @9 :List(Observation);
  counters @10 :Counters;
  scope @11 :Scope;
  calibrated @12 :Bool;  # false until an independent calibration is qualified
  processingTimeNs @13 :UInt64;
  rejectedEnvelopes @14 :UInt64;
  backlogCount @15 :UInt64;
  fusion @16 :ShadowFusion;  # telemetry, not lead authority; mode-3 assistance uses an independent bounded consumer
  mode @17 :UInt8;  # 4 produces perception independently of planner/DEC readiness

  struct ShadowFusion {
    version @0 :UInt16;
    algorithmVersion @1 :Text;
    enabled @2 :Bool;
    active @3 :Bool;
    calibrated @4 :Bool;  # remains false; mode-3 assistance does not claim calibration
    controlAllowed @5 :Bool;  # remains false; no lead, independent braking or launch authority
    reason @6 :Text;
    modelMonoTime @7 :UInt64;
    modelTimestampEof @8 :UInt64;
    egoMonoTime @9 :UInt64;
    processingTimeNs @10 :UInt64;
    tracks @11 :List(FusionTrack);
    leads @12 :List(FusionLead);  # model hypotheses, not independent vehicles
    counters @13 :FusionCounters;
  }

  enum PathState {
    unknown @0;
    inPath @1;
    outOfPath @2;
    entering @3;
    edge @4;  # outer body-overlap caution, not proven entry into the swept path
  }

  enum Motion {
    unknown @0;
    stationary @1;
    moving @2;
    oncoming @3;
  }

  enum FusionState {
    visionOnly @0;
    associating @1;
    fused @2;
    radarDistance @3;
    coasting @4;
  }

  struct FusionCounters {
    positions @0 :UInt64;
    duplicate @1 :UInt64;
    old @2 :UInt64;
    resets @3 :UInt64;
    rejectedPosition @4 :UInt64;
    rejectedVelocity @5 :UInt64;
    ambiguous @6 :UInt64;
    expired @7 :UInt64;
    capacity @8 :UInt64;
  }

  struct FusionTrack {
    trackId @0 :UInt64;
    oemTrackId @1 :UInt16;
    slot @2 :UInt8;
    positionSeq @3 :UInt64;
    positionMonoTime @4 :UInt64;
    velocityMonoTime @5 :UInt64;
    dRel @6 :Float64;
    yRel @7 :Float64;
    groundSpeed @8 :Float64;
    lateralSpeed @9 :Float64;
    positionVariance @10 :Float64;  # engineering filter value, not calibrated sigma
    velocityVariance @11 :Float64;
    observations @12 :UInt32;
    mature @13 :Bool;
    velocityGood @14 :Bool;
    coasting @15 :Bool;
    pathState @16 :PathState;
    pathOffset @17 :Float64;  # meaningful only when pathState != unknown
    entryTime @18 :Float64;  # meaningful only when pathState == entering
    motion @19 :Motion;
    u10Raw @20 :UInt16;
    active @21 :Bool;  # explicit position validity; separate from ordinary missing scans
    generation @22 :UInt64;  # observer slot generation, not a physical-object identifier
    birthReason @23 :Text;
    lastInvalidReason @24 :Text;
    lastInvalidMonoTime @25 :UInt64;
    frameIndex @26 :UInt8;
    lifecycleRaw @27 :UInt16;
    previousPositionSeq @28 :UInt64;
    previousPositionMonoTime @29 :UInt64;
    previousDRel @30 :Float64;
    previousYRel @31 :Float64;
    acceleration @32 :Float64;
    accelerationGood @33 :Bool;
    lateralPositionVariance @34 :Float64;  # engineering covariance, not calibrated sigma
    lateralVelocityVariance @35 :Float64;
    lateralVelocityValid @36 :Bool;
  }

  struct FusionLead {
    modelIndex @0 :UInt8;
    present @1 :Bool;
    probability @2 :Float64;
    trackId @3 :UInt64;
    state @4 :FusionState;
    reason @5 :Text;
    dRel @6 :Float64;
    yRel @7 :Float64;
    vRel @8 :Float64;
    vLead @9 :Float64;
    aLead @10 :Float64;
    distanceWeight @11 :Float64;
    velocityWeight @12 :Float64;
    pathState @13 :PathState;
    positionMonoTime @14 :UInt64;
    velocityMonoTime @15 :UInt64;
    sourceAge @16 :Float64;
    identityState @17 :Text;  # same/different/unresolved are algorithm judgments, not identity truth
    identityReason @18 :Text;
    candidateTrackId @19 :UInt64;  # best geometric candidate, including a rejected association
    bindingStartMonoTime @20 :UInt64;
    bindingSamples @21 :UInt16;
    recoverySamples @22 :UInt16;
    independentPair @23 :Bool;  # both model source and position advanced this update
    residualX @24 :Float64;
    residualY @25 :Float64;
    residualV @26 :Float64;  # actual velocity-gate operand
    residualDeltaX @27 :Float64;
    residualDeltaY @28 :Float64;
    bindingAnchorX @29 :Float64;
    bindingAnchorY @30 :Float64;
    rejectionMonoTime @31 :UInt64;
    gateX @32 :Float64;
    gateY @33 :Float64;
    visionXStd @34 :Float64;  # actual floored gate inputs, not calibrated sensor sigma
    visionYStd @35 :Float64;
    visionVStd @36 :Float64;
    blend @37 :Float64;
    coastWeight @38 :Float64;
    velocityGate @39 :Float64;
  }

  struct Scope {
    alphaLong @0 :Bool;
    experimentalSetting @1 :Bool;
    decSetting @2 :Bool;
    experimentalActive @3 :Bool;
    decEnabled @4 :Bool;
    decActive @5 :Bool;
    decMode @6 :UInt8;  # LongitudinalPlanSP.dec.state: acc=0, blended=1
    selfdriveMonoTime @7 :UInt64;
    planMonoTime @8 :UInt64;
    longActive @9 :Bool;
  }

  struct Counters {
    frames @0 :UInt64;
    badLength @1 :UInt64;
    duplicate @2 :UInt64;
    late @3 :UInt64;
    conflict @4 :UInt64;
    incomplete @5 :UInt64;
    expired @6 :UInt64;
    dropped @7 :UInt64;
  }

  struct FrameReference {
    part @0 :UInt8;  # MAIN F0..F3 = 0..3, AUX = 4
    address @1 :UInt32;
    monoTime @2 :UInt64;  # CAN envelope time, NOT a sensor sample timestamp
    frameInEnvelope @3 :UInt32;
  }

  struct RawFields {
    statusRaw @0 :UInt16;
    rangeRaw @1 :UInt16;
    azimuthRaw @2 :UInt16;
    rangeSigmaRaw @3 :UInt16;
    existenceRaw @4 :UInt16;
    angularWidthRaw @5 :UInt16;
    lifecycleRaw @6 :UInt16;
    normalizedClosingRaw @7 :UInt16;
    closingSigmaRaw @8 :UInt16;
    oemTrackId @9 :UInt16;
    edgeARaw @10 :UInt16;
    edgeBRaw @11 :UInt16;
    edgeSigmaARaw @12 :UInt16;
    u11Raw @13 :UInt16;
    u10Raw @14 :UInt16;
    edgeSigmaBRaw @15 :UInt16;
    ratioRaw @16 :UInt16;
  }

  struct Observation {
    slot @0 :UInt8;
    frameIndex @1 :UInt8;
    frameMask @2 :UInt8;  # Also defines presence of RawFields; zero is a real value
    generation @3 :UInt64;  # Slot occupancy generation, scoped by sessionId
    positionSeq @4 :UInt64;
    velocitySeq @5 :UInt64;
    positionMonoTime @6 :UInt64;
    velocityMonoTime @7 :UInt64;
    dRel @8 :Float64;  # provisional forward-axis distance, not calibrated
    yRel @9 :Float64;  # provisional left-positive position
    vRel @10 :Float64;  # direct U11 candidate only; never ratio-derived
    positionValid @11 :Bool;  # syntactically valid candidate, NOT control permission
    velocityValid @12 :Bool;
    reason @13 :Text;
    raw @14 :RawFields;
    frames @15 :List(FrameReference);
  }
}
