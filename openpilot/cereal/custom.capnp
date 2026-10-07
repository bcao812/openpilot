using Cxx = import "/include/c++.capnp";
$Cxx.namespace("cereal");

@0xb526ba661d550a59;

# custom.capnp: a home for empty structs reserved for custom forks
# These structs are guaranteed to remain reserved and empty in mainline
# cereal, so use these if you want custom events in your fork.

# DO rename the structs
# DON'T change the identifier (e.g. @0x81c2f05a394cf4af)

struct ModularAssistiveDrivingSystem {
  state @0 :ModularAssistiveDrivingSystemState;
  enabled @1 :Bool;
  active @2 :Bool;
  available @3 :Bool;

  enum ModularAssistiveDrivingSystemState {
    disabled @0;
    paused @1;
    enabled @2;
    softDisabling @3;
    overriding @4;
  }
}

struct IntelligentCruiseButtonManagement {
  state @0 :IntelligentCruiseButtonManagementState;
  sendButton @1 :SendButtonState;
  vTarget @2 :Float32;

  enum IntelligentCruiseButtonManagementState {
    inactive @0;      # No button press or default state
    preActive @1;     # Pre-active state before transitioning to increasing or decreasing
    increasing @2;    # Increasing speed
    decreasing @3;    # Decreasing speed
    holding @4;       # Holding steady speed
  }

  enum SendButtonState {
    none @0;
    increase @1;
    decrease @2;
  }
}

# Source evidence shared by the formal radar publisher and its SP mirror.
# modelProb remains the model's probability; these fields never replace it.
struct RadarLeadEvidence {
  version @0 :UInt16;
  source @1 :Source;
  sessionId @2 :Text;
  trackId @3 :UInt64;
  generation @4 :UInt64;
  positionMonoTime @5 :UInt64;
  velocityMonoTime @6 :UInt64;
  estimateMonoTime @7 :UInt64;
  positionStd @8 :Float32;
  velocityStd @9 :Float32;
  positionValid @10 :Bool;
  velocityValid @11 :Bool;
  accelerationValid @12 :Bool;
  followQualified @13 :Bool;
  brakeQualified @14 :Bool;
  stopQualified @15 :Bool;
  departureQualified @16 :Bool;
  fcwQualified @17 :Bool;
  coasting @18 :Bool;
  pathState @19 :HondaRadarObservations.PathState;
  pathOffset @20 :Float32;
  conflictStart @21 :Float32;  # seconds from estimateMonoTime
  conflictEnd @22 :Float32;  # finite occupancy interval, not permanent lead presence
  lateralSpeed @23 :Float32;
  pathSource @24 :Text;
  visionIndex @25 :Int16 = -1;
  reason @26 :Text;
  modelMonoTime @27 :UInt64;
  cautionQualified @28 :Bool;  # edge occupancy may only withdraw positive acceleration
  # Unpadded, single continuous 1.10 m point-sweep interval under the current
  # motion hypothesis. No quality/spatial extension, coast or synthetic path.
  # Independent mitigation evidence only; this does not change ordinary roles.
  physicalConflictValid @29 :Bool = false;
  physicalConflictStart @30 :Float64;  # seconds from estimateMonoTime
  physicalConflictEnd @31 :Float64;

  enum Source {
    legacy @0;
    vision @1;
    radar @2;
    fused @3;
    predicted @4;
  }
}

# Read-only attribution of an adopted object constraint. Strength describes
# request magnitude, not collision probability or counterfactual contribution.
struct LongitudinalObjectInfluence {
  version @0 :UInt8;
  sessionId @1 :Text;
  trackId @2 :UInt64;
  modelMonoTime @3 :UInt64;
  visionIndex @4 :Int8 = -1;
  kind @5 :Kind;
  strength @6 :Float32;

  enum Kind {
    follow @0;
    braking @1;
    hold @2;
    release @3;
    caution @4;
  }
}

# Same live fields as Log.RadarState.LeadData.
struct LeadData {
  dRel @0 :Float32;
  yRel @1 :Float32;
  vRel @2 :Float32;
  aRel @3 :Float32;
  vLead @4 :Float32;
  dPath @6 :Float32;
  vLat @7 :Float32;
  vLeadK @8 :Float32;
  aLeadK @9 :Float32;
  fcw @10 :Bool;
  status @11 :Bool;
  aLeadTau @12 :Float32;
  modelProb @13 :Float32;
  radar @14 :Bool;
  radarTrackId @15 :Int32 = -1;
  evidence @16 :RadarLeadEvidence;

  aLeadDEPRECATED @5 :Float32;
}

struct SelfdriveStateSP @0x81c2f05a394cf4af {
  mads @0 :ModularAssistiveDrivingSystem;
  intelligentCruiseButtonManagement @1 :IntelligentCruiseButtonManagement;
  buttonsPressed @2 :UInt16;
  buttonsReleaseToggle @3 :UInt16;

  enum AudibleAlert {
    none @0;

    engage @1;
    disengage @2;
    refuse @3;

    warningSoft @4;
    warningImmediate @5;

    prompt @6;
    promptRepeat @7;
    promptDistracted @8;

    # unused, these are reserved for upstream events so we don't collide
    reserved9 @9;
    reserved10 @10;
    reserved11 @11;
    reserved12 @12;
    reserved13 @13;
    reserved14 @14;
    reserved15 @15;
    reserved16 @16;
    reserved17 @17;
    reserved18 @18;
    reserved19 @19;
    reserved20 @20;
    reserved21 @21;
    reserved22 @22;
    reserved23 @23;
    reserved24 @24;
    reserved25 @25;
    reserved26 @26;
    reserved27 @27;
    reserved28 @28;
    reserved29 @29;
    reserved30 @30;

    promptSingleLow @31;
    promptSingleHigh @32;
  }
}

struct ModelManagerSP @0xaedffd8f31e7b55d {
  activeBundle @0 :ModelBundle;
  selectedBundle @1 :ModelBundle;
  availableBundles @2 :List(ModelBundle);

  struct DownloadUri {
    uri @0 :Text;
    sha256 @1 :Text;
  }

  enum DownloadStatus {
    notDownloading @0;
    downloading @1;
    downloaded @2;
    cached @3;
    failed @4;
    verifying @5;
  }

  struct DownloadProgress {
    status @0 :DownloadStatus;
    progress @1 :Float32;
    eta @2 :UInt32;
  }

  struct Chunk {
    fileName @0 :Text;
    sha256 @1 :Text;
  }

  struct Artifact {
    fileName @0 :Text;
    downloadUri @1 :DownloadUri;
    downloadProgress @2 :DownloadProgress;
    chunks @3 :List(Chunk);
  }

  struct Model {
    type @0 :Type;
    artifact @1 :Artifact;  # Main artifact
    metadata @2 :Artifact;  # Metadata artifact

    enum Type {
      supercombo @0;
      navigation @1;
      vision @2;
      policy @3;
      offPolicy @4;
      onPolicy @5;
      chunked @6;
    }
  }

  enum Runner {
    snpe @0;
    tinygrad @1;
    stock @2;
  }

  struct Override {
    key @0 :Text;
    value @1 :Text;
  }

  struct ModelBundle {
    index @0 :UInt32;
    internalName @1 :Text;
    displayName @2 :Text;
    models @3 :List(Model);
    status @4 :DownloadStatus;
    generation @5 :UInt32;
    environment @6 :Text;
    runner @7 :Runner;
    is20hz @8 :Bool;
    ref @9 :Text;
    minimumSelectorVersion @10 :UInt32;
    overrides @11 :List(Override);
  }
}

struct LongitudinalPlanSP @0xf35cc4560bbf6ec2 {
  dec @0 :DynamicExperimentalControl;
  longitudinalPlanSource @1 :LongitudinalPlanSource;
  smartCruiseControl @2 :SmartCruiseControl;
  speedLimit @3 :SpeedLimit;
  vTarget @4 :Float32;
  aTarget @5 :Float32;
  events @6 :List(OnroadEventSP.Event);
  e2eAlerts @7 :E2eAlerts;
  longitudinalPlanMonoTime @8 :UInt64;

  struct DynamicExperimentalControl {
    state @0 :DynamicExperimentalControlState;
    enabled @1 :Bool;
    active @2 :Bool;

    enum DynamicExperimentalControlState {
      acc @0;
      blended @1;
    }
  }

  struct SmartCruiseControl {
    vision @0 :Vision;
    map @1 :Map;

    struct Vision {
      state @0 :VisionState;
      vTarget @1 :Float32;
      aTarget @2 :Float32;
      currentLateralAccel @3 :Float32;
      maxPredictedLateralAccel @4 :Float32;
      enabled @5 :Bool;
      active @6 :Bool;
    }

    struct Map {
      state @0 :MapState;
      vTarget @1 :Float32;
      aTarget @2 :Float32;
      enabled @3 :Bool;
      active @4 :Bool;
    }

    enum VisionState {
      disabled @0; # System disabled or inactive.
      enabled @1; # No predicted substantial turn on vision range.
      entering @2; # A substantial turn is predicted ahead, adapting speed to turn comfort levels.
      turning @3; # Actively turning. Managing acceleration to provide a roll on turn feeling.
      leaving @4; # Road ahead straightens. Start to allow positive acceleration.
      overriding @5; # System overriding with manual control.
    }

    enum MapState {
      disabled @0; # System disabled or inactive.
      enabled @1; # No predicted substantial turn on map range.
      turning @2; # Actively turning. Managing acceleration to provide a roll on turn feeling.
      overriding @3; # System overriding with manual control.
    }
  }

  struct SpeedLimit {
    resolver @0 :Resolver;
    assist @1 :Assist;

    struct Resolver {
      speedLimit @0 :Float32;
      distToSpeedLimit @1 :Float32;
      source @2 :Source;
      speedLimitOffset @3 :Float32;
      speedLimitLast @4 :Float32;
      speedLimitFinal @5 :Float32;
      speedLimitFinalLast @6 :Float32;
      speedLimitValid @7 :Bool;
      speedLimitLastValid @8 :Bool;
    }

    struct Assist {
      state @0 :AssistState;
      enabled @1 :Bool;
      active @2 :Bool;
      vTarget @3 :Float32;
      aTarget @4 :Float32;
    }

    enum Source {
      none @0;
      car @1;
      map @2;
    }

    enum AssistState {
      disabled @0;
      inactive @1; # No speed limit set or not enabled by parameter.
      preActive @2;
      pending @3; # Awaiting new speed limit.
      adapting @4; # Reducing speed to match new speed limit.
      active @5; # Cruising at speed limit.
    }
  }

  enum LongitudinalPlanSource {
    cruise @0;
    sccVision @1;
    sccMap @2;
    speedLimitAssist @3;
  }

  struct E2eAlerts {
    greenLightAlert @0 :Bool;
    leadDepartAlert @1 :Bool;
  }
}

struct OnroadEventSP @0xda96579883444c35 {
  events @0 :List(Event);

  struct Event {
    name @0 :EventName;

    # event types
    enable @1 :Bool;
    noEntry @2 :Bool;
    warning @3 :Bool;   # alerts presented only when  enabled or soft disabling
    userDisable @4 :Bool;
    softDisable @5 :Bool;
    immediateDisable @6 :Bool;
    preEnable @7 :Bool;
    permanent @8 :Bool; # alerts presented regardless of openpilot state
    overrideLateral @10 :Bool;
    overrideLongitudinal @9 :Bool;
  }

  enum EventName {
    lkasEnable @0;
    lkasDisable @1;
    manualSteeringRequired @2;
    manualLongitudinalRequired @3;
    silentLkasEnable @4;
    silentLkasDisable @5;
    silentBrakeHold @6;
    silentWrongGear @7;
    silentReverseGear @8;
    silentDoorOpen @9;
    silentSeatbeltNotLatched @10;
    silentParkBrake @11;
    controlsMismatchLateral @12;
    hyundaiRadarTracksConfirmed @13;
    experimentalModeSwitched @14;
    wrongCarModeAlertOnly @15;
    pedalPressedAlertOnly @16;
    laneTurnLeft @17;
    laneTurnRight @18;
    speedLimitPreActive @19;
    speedLimitActive @20;
    speedLimitChanged @21;
    speedLimitPending @22;
    e2eChime @23;
    laneChangeRoadEdge @24;
    bigModelReady @25;
  }
}

struct CarParamsSP @0x80ae746ee2596b11 {
  flags @0 :UInt32;        # flags for car specific quirks in sunnypilot
  safetyParam @1 : Int16;  # flags for sunnypilot's custom safety flags
  pcmCruiseSpeed @3 :Bool;
  intelligentCruiseButtonManagementAvailable @4 :Bool;
  enableGasInterceptor @5 :Bool;

  neuralNetworkLateralControl @2 :NeuralNetworkLateralControl;

  struct NeuralNetworkLateralControl {
    model @0 :Model;
    fuzzyFingerprint @1 :Bool;

    struct Model {
      path @0 :Text;
      name @1 :Text;
    }
  }
}

struct CarControlSP @0xa5cd762cd951a455 {
  mads @0 :ModularAssistiveDrivingSystem;
  params @1 :List(Param);
  leadOne @2 :LeadData;
  leadTwo @3 :LeadData;
  intelligentCruiseButtonManagement @4 :IntelligentCruiseButtonManagement;

  struct Param {
    key @0 :Text;
    type @2 :ParamType;
    value @3 :Data;

    valueDEPRECATED @1 :Text; # The data type change may cause issues with backwards compatibility.
  }

  enum ParamType {
    string @0;
    bool @1;
    int @2;
    float @3;
    time @4;
    json @5;
    bytes @6;
  }
}

struct BackupManagerSP @0xf98d843bfd7004a3 {
  backupStatus @0 :Status;
  restoreStatus @1 :Status;
  backupProgress @2 :Float32;
  restoreProgress @3 :Float32;
  lastError @4 :Text;
  currentBackup @5 :BackupInfo;
  backupHistory @6 :List(BackupInfo);

  enum Status {
    idle @0;
    inProgress @1;
    completed @2;
    failed @3;
  }

  struct Version {
    major @0 :UInt16;
    minor @1 :UInt16;
    patch @2 :UInt16;
    build @3 :UInt16;
    branch @4 :Text;
  }

  struct MetadataEntry {
    key @0 :Text;
    value @1 :Text;
    tags @2 :List(Text);
  }

  struct BackupInfo {
    deviceId @0 :Text;
    version @1 :UInt32;
    config @2 :Text;
    isEncrypted @3 :Bool;
    createdAt @4 :Text;  # ISO timestamp
    updatedAt @5 :Text;  # ISO timestamp
    sunnypilotVersion @6 :Version;
    backupMetadata @7 :List(MetadataEntry);
  }
}

struct CarStateSP @0xb86e6369214c01c8 {
  speedLimit @0 :Float32;
}

struct LiveMapDataSP @0xf416ec09499d9d19 {
  speedLimitValid @0 :Bool;
  speedLimit @1 :Float32;
  speedLimitAheadValid @2 :Bool;
  speedLimitAhead @3 :Float32;
  speedLimitAheadDistance @4 :Float32;
  roadName @5 :Text;
}

struct ModelDataV2SP @0xa1680744031fdb2d {
  laneTurnDirection @0 :TurnDirection;
  leftLaneChangeEdgeBlock @1 :Bool;
  rightLaneChangeEdgeBlock @2 :Bool;

  enum TurnDirection {
    none @0;
    turnLeft @1;
    turnRight @2;
  }
}

struct CustomReserved10 @0xcb9fd56c7057593a {
}

struct CustomReserved11 @0xc2243c65e0340384 {
}

struct CustomReserved12 @0x9ccdc8676701b412 {
}

struct CustomReserved13 @0xcd96dafb67a082d0 {
}

struct CustomReserved14 @0xb057204d7deadf3f {
}

struct CustomReserved15 @0xbd443b539493bc68 {
}

struct CustomReserved16 @0xfc6241ed8877b611 {
}

struct CustomReserved17 @0xa30662f84033036c {
}

struct CustomReserved18 @0xc86a3d38d13eb3ef {
}

struct CustomReserved19 @0xa4f1eb3323f5f582 {
}

# Independent, receive-only Accord radar telemetry; no lead authority is granted.
# Mode 3 uses a separately gated consumer to bound positive acceleration only.
# Events are updates, not a complete object snapshot: absence is not deletion.
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
