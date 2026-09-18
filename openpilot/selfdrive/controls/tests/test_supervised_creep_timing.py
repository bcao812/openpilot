"""Measured-input tests for the SUPERVISED anti-creep clock migration.

The recorded boundary below is route 0000005a--0f5cae6edb, Q11
R-7d9151776fadaaae / J-a9f9815e7b634916. Source: the 2026-09-16
supervised-q11-residuals-v5 raw-inputs, control-timing and owner-reconstruction
artifacts, plus the corresponding core execution-contract integer ID edges.
The two earliest OFF control rows lack a captured planner carState sample and
are omitted; every included speed is a real carState measurement. The raw stop,
moving-lead and release facts are the existing focused reconstruction's stated
assumptions. This is an input-boundary regression, not native replay fidelity
or a claim about delivered vehicle braking.
"""
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.controls.lib.drive_helpers import CREEP_MAX_T, CREEP_SPEED, CREEP_STOP_T
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.motion_supervisor import MOTION_INTENT_VERSION, MotionIntent, MotionPhase, MotionSupervisor, StopReason


# Fixed wire order: carState, radarState, modelV2, controlsState, carControl,
# selfdriveState, vehicleParameters. This is independent raw provenance.
RAW_SERVICES = ('carState', 'radarState', 'modelV2', 'controlsState', 'carControl', 'selfdriveState', 'vehicleParameters')
FIRST_ROUND_STOP_CONTROL_NS = 2253275731795
SECOND_ROUND_STOP_CONTROL_NS = 2253315097941


def raw_intent(timestamp=2_000_000_000, *, sources=None, speed=0.7, reset=False, allow_creep=True,
               moving_lead=False, release_allowed=False, reasons=0, version=MOTION_INTENT_VERSION, **changes):
  # A real wire-shaped message lets the pre-fix decoder run and ignore the added
  # fields: the before test fails on timing, not an absent Python class/import.
  message = SimpleNamespace(
    version=version, mode=int(LongitudinalMode.SUPERVISED), reasons=reasons,
    movingLead=moving_lead, releaseAllowed=release_allowed, allowCreep=allow_creep,
    valid=True, ready=True, brakingValid=False, brakingAccel=0.0,
    creepEvidenceValid=True, creepSourceMonoTimes=list(sources or (timestamp,) * len(RAW_SERVICES)),
    creepMeasuredSpeed=speed, creepReset=reset,
  )
  for key, value in changes.items():
    setattr(message, key, value)
  return MotionIntent.from_message(message)


# plan ID: (raw source IDs, measured speed, off/reset observation, scalar request)
RECORDED_PLANS = {
  2252764478097: (
    (2252757035962, 2252711340753, 2252758196795, 2252753848462, 2252753993097, 2252759513670, 2252722927993),
    0.7278876304626465, True, -0.22158890962600708,
  ),
  2252814597785: (
    (2252809097681, 2252765469972, 2252808101378, 2252803500233, 2252803648306, 2252801912837, 2252780784399),
    0.7246951460838318, False, -0.22628508508205414,
  ),
  2252864024035: (
    (2252857268566, 2252815595806, 2252857793097, 2252853067108, 2252853199451, 2252851414556, 2252831208253),
    0.7197140455245972, False, -0.23024921119213104,
  ),
  2252911468201: (
    (2252897626691, 2252865049660, 2252904811951, 2252902322680, 2252902459712, 2252900712576, 2252879425545),
    0.7199581265449524, False, -0.23620042204856873,
  ),
  2252964480337: (
    (2252957965441, 2252912386014, 2252957931326, 2252952888618, 2252953026587, 2252951262993, 2252927139920),
    0.7336621284484863, False, -0.24270719289779663,
  ),
  2253014087628: (
    (2253009022680, 2252965588878, 2253007811847, 2253003683358, 2253003826847, 2253002092472, 2252979823983),
    0.7632907629013062, False, -0.2515009343624115,
  ),
  2253064238983: (
    (2253057119243, 2253015109712, 2253057950805, 2253052901014, 2253053035597, 2253059730910, 2253029550858),
    0.8037394881248474, False, -0.2596777379512787,
  ),
  2253113948618: (
    (2253109851482, 2253065235545, 2253107976014, 2253102833514, 2253102975805, 2253101106587, 2253079863045),
    0.845602810382843, False, -0.26828888058662415,
  ),
  2253164668410: (
    (2253159375232, 2253114933826, 2253158326222, 2253155487420, 2253155620701, 2253153921066, 2253131375441),
    0.8678404092788696, False, -0.27614155411720276,
  ),
  2253214006847: (
    (2253200514607, 2253165782316, 2253207663253, 2253205756482, 2253205885389, 2253204216066, 2253180313514),
    0.8749183416366577, False, -0.2846113443374634,
  ),
  2253264315962: (
    (2253258596170, 2253214956222, 2253257912785, 2253254428566, 2253254569764, 2253252828566, 2253230116274),
    0.8913870453834534, False, -0.29317012429237366,
  ),
  2253314607264: (
    (2253309872732, 2253265298826, 2253308143201, 2253303494660, 2253303631691, 2253301976743, 2253280225857),
    0.9104865789413452, False, -0.303636372089386,
  ),
  2253360315962: (
    (2253350397524, 2253315767472, 2253354347472, 2253355376326, 2253355518774, 2253353993930, 2253330790493),
    0.9212225675582886, False, -0.3145325481891632,
  ),
  2253414359191: (
    (2253400303566, 2253361243670, 2253407740389, 2253404778462, 2253404922524, 2253403288097, 2253375926534),
    0.9264730215072632, False, -0.32468879222869873,
  ),
  2253464571899: (
    (2253458539139, 2253415283826, 2253457783774, 2253454089451, 2253454238774, 2253452630389, 2253427949243),
    0.918983519077301, False, -0.3354385793209076,
  ),
  2253514229712: (
    (2253509272055, 2253465532368, 2253507556222, 2253504280493, 2253504422784, 2253502804764, 2253477491743),
    0.9116157293319702, False, -0.34875595569610596,
  ),
  2253564303097: (
    (2253558690493, 2253515195232, 2253558332628, 2253554528618, 2253554672211, 2253552989034, 2253529095545),
    0.8948317170143127, False, -0.36013031005859375,
  ),
  2253614308045: (
    (2253599819659, 2253565223670, 2253607744191, 2253604454347, 2253604594086, 2253602940024, 2253580221482),
    0.8701549172401428, False, -0.36911770701408386,
  ),
}

# control ID, plan ID, actual carState ID, vEgo, aEgo, gas, brake, baseline request
RECORDED_CONTROLS = [
  (2252773110389, 2252764478097, 2252768009139, 0.7263655066490173, 0.28692370653152466, True, False, 0.0),
  (2252783037993, 2252764478097, 2252777556639, 0.7270630598068237, 0.2662218511104584, True, False, 0.0),
  (2252791984556, 2252764478097, 2252787514503, 0.7264651656150818, 0.23514443635940552, True, False, 0.0),
  (2252803648306, 2252764478097, 2252798894608, 0.7256605625152588, 0.20505908131599426, False, False, -0.22158890962600708),
  (2252813923618, 2252764478097, 2252809097681, 0.7246951460838318, 0.176308736205101, False, False, -0.22158890962600708),
  (2252822772733, 2252814597785, 2252818171847, 0.7236102819442749, 0.1491602212190628, False, False, -0.22628508508205414),
  (2252833468878, 2252814597785, 2252827439503, 0.7224428057670593, 0.12381178885698318, False, False, -0.22628508508205414),
  (2252843062785, 2252814597785, 2252838297003, 0.7212250232696533, 0.10040055215358734, False, False, -0.22628508508205414),
  (2252853199451, 2252814597785, 2252847813983, 0.7199850678443909, 0.07900993525981903, False, False, -0.22628508508205414),
  (2252861545076, 2252814597785, 2252857268566, 0.7197140455245972, 0.06889478862285614, False, False, -0.22628508508205414),
  (2252874047889, 2252864024035, 2252868986431, 0.7208395600318909, 0.07305643707513809, False, False, -0.23024921119213104),
  (2252883655701, 2252864024035, 2252877924035, 0.7193933129310608, 0.052305325865745544, False, False, -0.23024921119213104),
  (2252893340337, 2252864024035, 2252888336535, 0.7189582586288452, 0.04317208379507065, False, False, -0.23024921119213104),
  (2252902459712, 2252864024035, 2252897626691, 0.7199581265449524, 0.048587821424007416, False, False, -0.23024921119213104),
  (2252913593774, 2252911468201, 2252908870076, 0.7208380699157715, 0.05234455689787865, False, False, -0.23620042204856873),
  (2252922028045, 2252911468201, 2252916995389, 0.7220859527587891, 0.059250302612781525, False, False, -0.23620042204856873),
  (2252935446170, 2252911468201, 2252928909243, 0.7246361970901489, 0.07791262865066528, False, False, -0.23620042204856873),
  (2252943318462, 2252911468201, 2252938796951, 0.7254786491394043, 0.07851634174585342, False, False, -0.23620042204856873),
  (2252953026587, 2252911468201, 2252947568045, 0.7305319905281067, 0.11920357495546341, False, False, -0.23620042204856873),
  (2252962397837, 2252911468201, 2252957965441, 0.7336621284484863, 0.137678861618042, False, False, -0.23620042204856873),
  (2252972804555, 2252964480337, 2252967894764, 0.7398167252540588, 0.18322378396987915, False, False, -0.24270719289779663),
  (2252983273670, 2252964480337, 2252977462680, 0.7439049482345581, 0.20472948253154755, False, False, -0.24270719289779663),
  (2252992076795, 2252964480337, 2252987449660, 0.7499141693115234, 0.24249698221683502, False, False, -0.24270719289779663),
  (2253003826847, 2252964480337, 2252998827889, 0.7557385563850403, 0.2749027609825134, False, False, -0.24270719289779663),
  (2253014292837, 2252964480337, 2253009022680, 0.7632907629013062, 0.3206895887851715, False, False, -0.24270719289779663),
  (2253022722680, 2253014087628, 2253017522785, 0.7704697847366333, 0.3585543930530548, False, False, -0.2515009343624115),
  (2253033240597, 2253014087628, 2253027356483, 0.7796788811683655, 0.41216161847114563, False, False, -0.2515009343624115),
  (2253042356326, 2253014087628, 2253037899764, 0.7854035496711731, 0.42744338512420654, False, False, -0.2515009343624115),
  (2253053035597, 2253014087628, 2253047701066, 0.7941526770591736, 0.47009873390197754, False, False, -0.2515009343624115),
  (2253061369608, 2253014087628, 2253057119243, 0.8037394881248474, 0.5166733264923096, False, False, -0.2515009343624115),
  (2253073761326, 2253064238983, 2253068881222, 0.8135738372802734, 0.5611681342124939, False, False, -0.2596777379512787),
  (2253083555805, 2253064238983, 2253077823566, 0.8240753412246704, 0.6077813506126404, False, False, -0.2596777379512787),
  (2253092453253, 2253064238983, 2253087655753, 0.8312811255455017, 0.6185336709022522, False, False, -0.2596777379512787),
  (2253102975805, 2253064238983, 2253097427680, 0.8387906551361084, 0.6311569213867188, False, False, -0.2596777379512787),
  (2253114461378, 2253064238983, 2253109851482, 0.845602810382843, 0.6359289288520813, False, False, -0.2596777379512787),
  (2253124567003, 2253113948618, 2253119590128, 0.8546614646911621, 0.6616611480712891, False, False, -0.26828888058662415),
  (2253135640649, 2253113948618, 2253129356066, 0.860466718673706, 0.6539265513420105, False, False, -0.26828888058662415),
  (2253144380910, 2253113948618, 2253139709607, 0.8632501363754272, 0.6181234121322632, False, False, -0.26828888058662415),
  (2253155620701, 2253113948618, 2253150126170, 0.8651910424232483, 0.5777018666267395, False, False, -0.26828888058662415),
  (2253164088410, 2253113948618, 2253159375232, 0.8678404092788696, 0.5478869080543518, False, False, -0.26828888058662415),
  (2253174116951, 2253164668410, 2253169172524, 0.8716644644737244, 0.5321123003959656, False, False, -0.27614155411720276),
  (2253184727055, 2253164668410, 2253178782107, 0.8732146620750427, 0.4961654245853424, False, False, -0.27614155411720276),
  (2253193677680, 2253164668410, 2253188731430, 0.8751025795936584, 0.4668644070625305, False, False, -0.27614155411720276),
  (2253205885389, 2253164668410, 2253200514607, 0.8749183416366577, 0.4206037223339081, False, False, -0.27614155411720276),
  (2253215463097, 2253164668410, 2253210428982, 0.8767210841178894, 0.3976939916610718, False, False, -0.27614155411720276),
  (2253223645441, 2253214006847, 2253218892316, 0.8799149394035339, 0.3902292549610138, False, False, -0.2846113443374634),
  (2253235026899, 2253214006847, 2253228732420, 0.8815112709999084, 0.3682469427585602, False, False, -0.2846113443374634),
  (2253243898670, 2253214006847, 2253239292160, 0.8840603828430176, 0.35744309425354004, False, False, -0.2846113443374634),
  (2253254569764, 2253214006847, 2253249027264, 0.8865412473678589, 0.34701862931251526, False, False, -0.2846113443374634),
  (2253263328826, 2253214006847, 2253258596170, 0.8913870453834534, 0.360132098197937, False, False, -0.2846113443374634),
  (2253275731795, 2253264315962, 2253270467732, 0.895037055015564, 0.3605960011482239, False, False, -0.30461135506629944),
  (2253285187160, 2253264315962, 2253279121899, 0.8970893621444702, 0.34578561782836914, False, False, -0.3246113359928131),
  (2253293925805, 2253264315962, 2253289156899, 0.9015373587608337, 0.3552241921424866, False, False, -0.34461134672164917),
  (2253303631691, 2253264315962, 2253298755701, 0.90385502576828, 0.3434552550315857, False, False, -0.36461135745048523),
  (2253315097941, 2253264315962, 2253309872732, 0.9104865789413452, 0.3739311397075653, False, False, -0.3846113383769989),
  (2253323127472, 2253314607264, 2253318369295, 0.9133675694465637, 0.365749329328537, False, False, -0.40461134910583496),
  (2253335821118, 2253314607264, 2253329625597, 0.9146983027458191, 0.3435690701007843, False, False, -0.42461133003234863),
  (2253345318826, 2253314607264, 2253340313045, 0.9175096154212952, 0.33761700987815857, False, False, -0.4446113407611847),
  (2253355518774, 2253314607264, 2253350397524, 0.9212225675582886, 0.34082740545272827, False, False, -0.46461135149002075),
  (2253362894972, 2253360315962, 2253358338514, 0.9238378405570984, 0.33326804637908936, False, False, -0.4846113324165344),
  (2253374245128, 2253360315962, 2253369450649, 0.9235047698020935, 0.2983240485191345, False, False, -0.5046113729476929),
  (2253385169555, 2253360315962, 2253378940180, 0.9233638048171997, 0.2685416340827942, False, False, -0.5246113538742065),
  (2253393321066, 2253360315962, 2253388630909, 0.925366997718811, 0.2620384693145752, False, False, -0.5446113348007202),
  (2253404922524, 2253360315962, 2253400303566, 0.9264730215072632, 0.2476024180650711, False, False, -0.5646113157272339),
  (2253415401326, 2253360315962, 2253410484451, 0.925308108329773, 0.212895005941391, False, False, -0.5846113562583923),
  (2253423227472, 2253414359191, 2253418651378, 0.9230319261550903, 0.1709022969007492, False, False, -0.604611337184906),
  (2253435870024, 2253414359191, 2253429627941, 0.9216989874839783, 0.14190448820590973, False, False, -0.6246113181114197),
  (2253443769295, 2253414359191, 2253439174191, 0.9232091307640076, 0.1427726000547409, False, False, -0.6446113586425781),
  (2253454238774, 2253414359191, 2253449037055, 0.9210805296897888, 0.10887179523706436, False, False, -0.6646113395690918),
  (2253463197107, 2253414359191, 2253458539139, 0.918983519077301, 0.07850281149148941, False, False, -0.6846113204956055),
  (2253475370857, 2253464571899, 2253470414034, 0.9169477224349976, 0.051613401621580124, False, False, -0.7046113610267639),
  (2253485344295, 2253464571899, 2253479368878, 0.9174149632453918, 0.05114704743027687, False, False, -0.7246113419532776),
  (2253493324243, 2253464571899, 2253488698045, 0.9158622026443481, 0.03146924078464508, False, False, -0.7446113228797913),
  (2253504422784, 2253464571899, 2253499633566, 0.9134159088134766, 0.005149782635271549, False, False, -0.7646113634109497),
  (2253514843826, 2253464571899, 2253509272055, 0.9116157293319702, -0.012501548044383526, False, False, -0.7846113443374634),
  (2253523609034, 2253514229712, 2253518865701, 0.9099523425102234, -0.027165928855538368, False, False, -0.804611325263977),
  (2253536060597, 2253514229712, 2253528930076, 0.908431887626648, -0.03907039016485214, False, False, -0.8246113657951355),
  (2253544710493, 2253514229712, 2253540177003, 0.9056065082550049, -0.06227912753820419, False, False, -0.8446113467216492),
  (2253554672211, 2253514229712, 2253549330909, 0.8982058763504028, -0.12689034640789032, False, False, -0.8646113276481628),
  (2253562998305, 2253514229712, 2253558690493, 0.8948317170143127, -0.14695857465267181, False, False, -0.8846113681793213),
  (2253573337107, 2253564303097, 2253568456847, 0.8903937339782715, -0.17525552213191986, False, False, -0.904611349105835),
  (2253585176743, 2253564303097, 2253579286743, 0.8859617114067078, -0.2007976621389389, False, False, -0.9246113300323486),
  (2253593309243, 2253564303097, 2253588673045, 0.8791447281837463, -0.24664025008678436, False, False, -0.9446113705635071),
  (2253604594086, 2253564303097, 2253599819659, 0.8701549172401428, -0.3088260591030121, False, False, -0.9646113514900208),
  (2253618074868, 2253614308045, 2253613345545, 0.8635584712028503, -0.3422681391239166, False, False, -0.9846113324165344),
  (2253626771430, 2253614308045, 2253622132889, 0.8563252687454224, -0.378592312335968, False, True, 0.0),
  (2253638322889, 2253614308045, 2253631828982, 0.8485373854637146, -0.41674190759658813, False, True, 0.0),
  (2253645704086, 2253614308045, 2253641329920, 0.8383390307426453, -0.47423258423805237, False, True, 0.0),
]


def recorded_inputs():
  for now_ns, plan_ns, car_state_ns, speed, accel, gas, brake, baseline_request in RECORDED_CONTROLS:
    sources, measured_speed, reset, target = RECORDED_PLANS[plan_ns]
    intent = raw_intent(sources=sources, speed=measured_speed, reset=reset, allow_creep=not reset)
    yield now_ns, plan_ns, sources[2], car_state_ns, speed, accel, gas, brake, baseline_request, target, intent


def test_recorded_pedal_reentry_meets_first_round_stopping_deadline():
  owner = MotionSupervisor(LongitudinalMode.SUPERVISED)
  first_stop = None
  for now_ns, plan_ns, model_ns, _, speed, accel, gas, brake, _, target, intent in recorded_inputs():
    stop = owner.update(not gas and not brake, speed, accel, False, target, intent,
                        now_ns=now_ns, plan_time_ns=plan_ns, model_time_ns=model_ns)
    if gas or brake:
      assert not stop
      assert owner.phase == MotionPhase.OFF
    elif stop and first_stop is None:
      assert owner.reasons & StopReason.CREEP
      first_stop = now_ns
  assert first_stop is not None
  assert first_stop <= FIRST_ROUND_STOP_CONTROL_NS, (
    f"stop entry {first_stop} still later than {FIRST_ROUND_STOP_CONTROL_NS}; " +
    f"second-round observed entry was {SECOND_ROUND_STOP_CONTROL_NS}"
  )


def sample_step(owner, timestamp, *, intent=None, v_ego=0.7, a_ego=0.0, a_target=-0.2,
                active=True, now_ns=None, plan_ns=None, model_ns=None, **kwargs):
  return owner.update(active, v_ego, a_ego, False, a_target, raw_intent(timestamp) if intent is None else intent,
                      plan_time_ns=timestamp + 1_000_000 if plan_ns is None else plan_ns,
                      model_time_ns=timestamp if model_ns is None else model_ns,
                      now_ns=timestamp + 2_000_000 if now_ns is None else now_ns, **kwargs)


def test_off_observation_is_bounded_and_old_plan_cannot_rearm():
  owner = MotionSupervisor(LongitudinalMode.SUPERVISED)
  base = 2_000_000_000
  for index in range(30):
    timestamp = base + index * 50_000_000
    off_intent = raw_intent(timestamp, reset=True, allow_creep=False)
    assert not sample_step(owner, timestamp, intent=off_intent, active=False)
    assert owner.phase == MotionPhase.OFF
    assert owner.creep_total_t == 0.0
  assert not sample_step(owner, timestamp, intent=off_intent, now_ns=timestamp + 12_000_000)
  assert owner.waiting_for_fresh_input and not owner.degraded
  assert owner.creep_total_t == 0.0
  assert not sample_step(owner, timestamp + 50_000_000)
  assert not owner.degraded
  assert owner.creep_total_t == pytest.approx(0.1)


def test_ineligible_off_observation_still_sets_rearm_barrier():
  owner = MotionSupervisor(LongitudinalMode.SUPERVISED)
  base = 2_000_000_000
  intent = raw_intent(base, reset=True, allow_creep=False, speed=5.0)
  assert not sample_step(owner, base, intent=intent, active=False, v_ego=5.0)
  assert not sample_step(owner, base, intent=intent, v_ego=5.0, now_ns=base + 12_000_000)
  assert owner.waiting_for_fresh_input and not owner.degraded
  assert owner.creep_total_t == 0.0


def test_off_speed_excursion_cannot_rebank_the_same_old_observation():
  owner = MotionSupervisor(LongitudinalMode.SUPERVISED)
  base = 2_000_000_000
  off_intent = raw_intent(base, reset=True, allow_creep=False)
  sample_step(owner, base, intent=off_intent, active=False)
  sample_step(owner, base, intent=off_intent, active=False, v_ego=CREEP_SPEED, now_ns=base + 12_000_000)
  sample_step(owner, base, intent=off_intent, active=False, now_ns=base + 22_000_000)
  sample_step(owner, base + 50_000_000)
  assert owner.creep_total_t == pytest.approx(0.05)


@pytest.mark.parametrize('off_valid', [True, False])
@pytest.mark.parametrize('reentry_now_offset', [60_000_000, 90_000_000])
def test_persistent_off_clock_rejects_rollback_and_duplicates_even_after_invalid_input(off_valid, reentry_now_offset):
  owner = MotionSupervisor(LongitudinalMode.SUPERVISED)
  base = 2_000_000_000
  sample_step(owner, base - 50_000_000)
  off_intent = raw_intent(base, reset=True, allow_creep=False)
  sample_step(owner, base, intent=off_intent, active=False, now_ns=base + 90_000_000, plan_valid=off_valid)
  assert owner.phase == MotionPhase.OFF
  sample_step(owner, base + 50_000_000, now_ns=base + reentry_now_offset)
  assert owner.degraded
  assert owner.creep_total_t == 0.0
  # Recovery needs a forward consumption clock and independently fresh sources.
  # The rejected transition must not carry the earlier off credit into it.
  sample_step(owner, base + 100_000_000, now_ns=base + 110_000_000)
  assert not owner.degraded
  assert owner.creep_total_t == pytest.approx(0.05)


@pytest.mark.parametrize('gap_ns', [101_000_000, 200_000_000, 300_000_000])
def test_off_sample_outside_continuity_window_has_no_borrowed_credit(gap_ns):
  owner = MotionSupervisor(LongitudinalMode.SUPERVISED)
  base = 2_000_000_000
  sample_step(owner, base, intent=raw_intent(base, reset=True, allow_creep=False), active=False)
  assert not sample_step(owner, base + gap_ns)
  assert owner.creep_total_t == pytest.approx(0.05)


@pytest.mark.parametrize('changes', [
  {'creepEvidenceValid': False}, {'creepEvidenceValid': 1}, {'creepSourceMonoTimes': []},
  {'creepSourceMonoTimes': [0] * 7}, {'creepSourceMonoTimes': [True] * 7},
  {'creepMeasuredSpeed': float('nan')}, {'creepMeasuredSpeed': float('inf')},
  {'creepReset': 1},
])
def test_missing_or_malformed_raw_evidence_is_not_a_normal_supervised_plan(changes):
  intent = raw_intent(**changes)
  assert not intent.is_valid()
  owner = MotionSupervisor(LongitudinalMode.SUPERVISED)
  sample_step(owner, 2_000_000_000, intent=intent)
  assert owner.degraded
  assert owner.creep_total_t == 0.0


@pytest.mark.parametrize('version', [0, 1, 2, MOTION_INTENT_VERSION + 1])
def test_old_or_unknown_protocol_cannot_opt_in_to_new_timing(version):
  assert not raw_intent(version=version).is_valid()


@pytest.mark.parametrize('source_index', range(len(RAW_SERVICES)))
@pytest.mark.parametrize('offset_ns', [-300_000_000, 3_000_000])
def test_each_raw_source_has_an_independent_lease(source_index, offset_ns):
  owner = MotionSupervisor(LongitudinalMode.SUPERVISED)
  base = 2_000_000_000
  sources = [base] * len(RAW_SERVICES)
  sources[source_index] += offset_ns
  sample_step(owner, base, intent=raw_intent(sources=sources))
  assert owner.degraded
  assert owner.creep_total_t == 0.0


@pytest.mark.parametrize('source_index', [0, 3, 4, 5])
def test_recent_model_cannot_hide_authority_snapshot_skew(source_index):
  owner = MotionSupervisor(LongitudinalMode.SUPERVISED)
  base = 2_000_000_000
  sources = [base] * len(RAW_SERVICES)
  sources[source_index] -= 125_000_000
  sample_step(owner, base, intent=raw_intent(sources=sources))
  assert owner.degraded


@pytest.mark.parametrize('source_index', range(len(RAW_SERVICES)))
def test_reordered_raw_source_is_rejected(source_index):
  owner = MotionSupervisor(LongitudinalMode.SUPERVISED)
  base = 2_000_000_000
  sample_step(owner, base)
  sources = [base + 50_000_000] * len(RAW_SERVICES)
  sources[source_index] = base - 1
  sample_step(owner, base + 50_000_000, intent=raw_intent(sources=sources))
  assert owner.degraded
  assert owner.creep_total_t == 0.0


@pytest.mark.parametrize('rewritten', ['speed', 'reset'])
def test_same_raw_source_identity_cannot_change_its_measurement(rewritten):
  owner = MotionSupervisor(LongitudinalMode.SUPERVISED)
  base = 2_000_000_000
  sample_step(owner, base)
  sources = [base + 50_000_000] * len(RAW_SERVICES)
  sources[0 if rewritten == 'speed' else 3] = base
  sample_step(owner, base + 50_000_000,
              intent=raw_intent(sources=sources, speed=0.6 if rewritten == 'speed' else 0.7, reset=rewritten == 'reset'))
  assert owner.degraded


def test_control_repeats_do_not_create_extra_qualified_samples():
  owner = MotionSupervisor(LongitudinalMode.SUPERVISED)
  base = 2_000_000_000
  intent = raw_intent(base)
  sample_step(owner, base, intent=intent)
  for tick in range(1, 24):
    assert not sample_step(owner, base, intent=intent, now_ns=base + 2_000_000 + tick * 10_000_000)
    assert owner.creep_total_t == pytest.approx(0.05)
  sample_step(owner, base, intent=intent, now_ns=base + 262_000_000)
  assert owner.degraded
  assert owner.creep_total_t == 0.0


@pytest.mark.parametrize('repeated_source', ['carState', 'modelV2'])
def test_new_publication_with_repeated_measurement_or_model_is_not_new_evidence(repeated_source):
  owner = MotionSupervisor(LongitudinalMode.SUPERVISED)
  base = 2_000_000_000
  sample_step(owner, base)
  for index in range(1, 5):
    timestamp = base + index * 50_000_000
    sources = [timestamp] * len(RAW_SERVICES)
    sources[RAW_SERVICES.index(repeated_source)] = base
    # Keep each raw authority snapshot inside its independent 100 ms bound.
    if repeated_source == 'carState' and index > 2:
      break
    assert not sample_step(owner, timestamp, intent=raw_intent(sources=sources), model_ns=sources[2])
    assert owner.creep_total_t == pytest.approx(0.05)


@pytest.mark.parametrize('excursion', [CREEP_SPEED, CREEP_SPEED + 0.01, -0.01])
def test_current_control_measurement_breaks_qualification_between_plans(excursion):
  owner = MotionSupervisor(LongitudinalMode.SUPERVISED)
  base = 2_000_000_000
  intent = raw_intent(base)
  sample_step(owner, base, intent=intent)
  sample_step(owner, base, intent=intent, v_ego=excursion, now_ns=base + 12_000_000)
  assert owner.creep_total_t == 0.0
  sample_step(owner, base, intent=intent, now_ns=base + 22_000_000)
  assert owner.creep_total_t == 0.0
  sample_step(owner, base + 50_000_000)
  assert owner.creep_total_t == pytest.approx(0.05)


def test_rewritten_positive_target_and_manual_override_revoke_old_sample_credit():
  owner = MotionSupervisor(LongitudinalMode.SUPERVISED)
  base = 2_000_000_000
  intent = raw_intent(base)
  sample_step(owner, base, intent=intent)
  sample_step(owner, base, intent=intent, a_target=0.1, now_ns=base + 12_000_000)
  assert owner.degraded
  assert owner.creep_total_t == 0.0
  sample_step(owner, base, intent=intent, now_ns=base + 22_000_000)
  assert owner.degraded
  sample_step(owner, base + 50_000_000)
  sample_step(owner, base + 50_000_000, active=False, now_ns=base + 62_000_000)
  assert owner.phase == MotionPhase.OFF
  assert owner.creep_total_t == 0.0
  sample_step(owner, base + 50_000_000, now_ns=base + 72_000_000)
  assert owner.waiting_for_fresh_input and not owner.degraded
  assert owner.creep_total_t == 0.0


@pytest.mark.parametrize('field', ['v_ego', 'a_ego', 'a_target'])
def test_nonfinite_current_input_clears_timing_and_requires_new_evidence(field):
  owner = MotionSupervisor(LongitudinalMode.SUPERVISED)
  base = 2_000_000_000
  sample_step(owner, base)
  sample_step(owner, base, now_ns=base + 12_000_000, **{field: float('nan')})
  assert owner.degraded
  assert owner.creep_total_t == 0.0
  sample_step(owner, base, now_ns=base + 22_000_000)
  assert owner.degraded
  assert owner.creep_total_t == 0.0
  sample_step(owner, base + 50_000_000)
  assert not owner.degraded
  assert owner.creep_total_t == pytest.approx(0.05)


@pytest.mark.parametrize('jitter_ns', [(-3_000_000, 3_000_000), (7_000_000, -7_000_000), (0, 0)])
def test_first_fresh_moving_lead_drops_off_credit_and_shares_the_sample_clock(jitter_ns):
  owner = MotionSupervisor(LongitudinalMode.SUPERVISED)
  timestamp = 2_000_000_000
  sample_step(owner, timestamp, intent=raw_intent(timestamp, reset=True, allow_creep=False), active=False)
  for index in range(35):
    timestamp += 50_000_000 + jitter_ns[index % len(jitter_ns)]
    intent = raw_intent(timestamp, moving_lead=True)
    for tick in range(4):
      assert not sample_step(owner, timestamp, intent=intent, now_ns=timestamp + 2_000_000 + tick * 10_000_000)
      assert not owner.degraded
      assert not owner.reasons & StopReason.CREEP
      if index == 0:
        assert owner.creep_total_t == pytest.approx(0.05)
  assert owner.moving_lead_t == pytest.approx(CREEP_STOP_T)
  assert owner.creep_total_t == 0.0


@pytest.mark.parametrize('first_moving_sample', [1, 2])
@pytest.mark.parametrize('control_phase_ns', [2_000_000, 8_000_000])
def test_later_fresh_moving_lead_revokes_off_credit_before_creep_can_win(first_moving_sample, control_phase_ns):
  owner = MotionSupervisor(LongitudinalMode.SUPERVISED)
  base = 2_000_000_000
  for frame in range(35):
    timestamp = base + frame * 50_000_000
    intent = raw_intent(timestamp, reset=frame == 0, allow_creep=frame > 0, moving_lead=frame >= first_moving_sample)
    for tick in range(5):
      assert not sample_step(owner, timestamp, intent=intent, active=frame > 0,
                             now_ns=timestamp + control_phase_ns + tick * 10_000_000)
      assert not owner.degraded
  assert owner.moving_lead_t == pytest.approx(CREEP_STOP_T)
  assert not owner.reasons & StopReason.CREEP


@pytest.mark.parametrize('first_moving_sample', [1, 2, 5, 12])
@pytest.mark.parametrize('repeated_pulse', [False, True])
def test_transient_moving_evidence_can_remove_off_credit_only_once(first_moving_sample, repeated_pulse):
  owner = MotionSupervisor(LongitudinalMode.SUPERVISED)
  base = 2_000_000_000
  off = raw_intent(base, speed=1.15, reset=True, allow_creep=False)
  sample_step(owner, base, intent=off, v_ego=1.15, active=False)
  first_stop_frame = None
  for frame in range(1, 23):
    timestamp = base + frame * 50_000_000
    speed = 1.15 * (1.0 - 0.6 * 0.05) ** (frame - 1)
    moving = frame >= first_moving_sample and (frame - first_moving_sample) % 2 == 0 if repeated_pulse else frame == first_moving_sample
    stop = sample_step(owner, timestamp, intent=raw_intent(timestamp, speed=speed, moving_lead=moving), v_ego=speed)
    if stop and first_stop_frame is None:
      first_stop_frame = frame
  # Prompt physical slowing passes the half-second window. One or many short
  # moving pulses must still reach the existing total deadline at frame 20.
  assert first_stop_frame == 20
  assert owner.creep_total_t == pytest.approx(CREEP_MAX_T)
  assert owner.reasons & StopReason.CREEP


def test_revoking_off_credit_restores_the_automatic_measurement_speed_anchor():
  borrowed = MotionSupervisor(LongitudinalMode.SUPERVISED)
  without_off = MotionSupervisor(LongitudinalMode.SUPERVISED)
  base = 2_000_000_000
  off = raw_intent(base, speed=1.1, reset=True, allow_creep=False)
  sample_step(borrowed, base, intent=off, v_ego=1.1, active=False)
  first_stop_frame = None
  for frame in range(1, 16):
    timestamp = base + frame * 50_000_000
    speed = 0.8 if frame == 1 else 0.78
    intent = raw_intent(timestamp, speed=speed, moving_lead=frame == 2)
    stopped = sample_step(borrowed, timestamp, intent=intent, v_ego=speed)
    expected_stop = sample_step(without_off, timestamp, intent=intent, v_ego=speed)
    if frame >= 2:
      assert stopped == expected_stop
      assert borrowed.creep_total_t == without_off.creep_total_t
      assert borrowed.creep_t == without_off.creep_t
      assert borrowed.creep_v == without_off.creep_v
    if stopped and first_stop_frame is None:
      first_stop_frame = frame
  # Keeping the old 1.1 m/s off anchor would falsely treat this as sufficient
  # deceleration and postpone the stop beyond the actual automatic window.
  assert first_stop_frame == 11


def test_moving_evidence_cannot_rewrite_an_already_established_creep_stop():
  owner = MotionSupervisor(LongitudinalMode.SUPERVISED)
  base = 2_000_000_000
  for frame in range(12):
    sample_step(owner, base + frame * 50_000_000)
  assert owner.reasons & StopReason.CREEP
  elapsed = owner.creep_total_t
  for frame in range(12, 35):
    timestamp = base + frame * 50_000_000
    assert sample_step(owner, timestamp, intent=raw_intent(timestamp, moving_lead=True))
    assert owner.phase == MotionPhase.APPROACH
    assert owner.creep_total_t == elapsed


def test_window_boundary_uses_actual_snapshot_time_and_strict_half_second():
  owner = MotionSupervisor(LongitudinalMode.SUPERVISED)
  base = 2_000_000_000
  for elapsed in range(0, 451_000_000, 50_000_000):
    assert not sample_step(owner, base + elapsed)
  assert owner.creep_total_t == pytest.approx(CREEP_STOP_T)
  assert sample_step(owner, base + 450_000_001)
  assert owner.reasons & StopReason.CREEP


def test_prompt_deceleration_and_total_creep_deadline_are_independent():
  owner = MotionSupervisor(LongitudinalMode.SUPERVISED)
  base = 2_000_000_000
  for index in range(22):
    timestamp = base + index * 50_000_000
    speed = 1.15 * (1.0 - 0.6 * 0.05) ** index
    stopped = sample_step(owner, timestamp, intent=raw_intent(timestamp, speed=speed), v_ego=speed)
    if index < 19:
      assert not stopped
    else:
      assert stopped
      assert owner.reasons & StopReason.CREEP
  assert owner.creep_total_t == pytest.approx(CREEP_MAX_T)


def test_unified_keeps_its_existing_control_time_contract():
  owner = MotionSupervisor(LongitudinalMode.UNIFIED)
  base = 2_000_000_000
  intent = MotionIntent()
  for tick in range(20):
    assert not sample_step(owner, base, intent=intent, now_ns=base + 2_000_000 + tick * 10_000_000)
  assert owner.creep_total_t == pytest.approx(0.2)


def test_recorded_longcontrol_ramp_inherits_request_and_manual_brake_wins():
  from opendbc.car.honda.interface import CarInterface
  from opendbc.car.honda.values import CAR, CarControllerParams
  from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState

  params = CarInterface.get_non_essential_params(CAR.HONDA_ACCORD)
  params.openpilotLongitudinalControl = True
  extension = CarInterface.get_non_essential_params_sp(params, CAR.HONDA_ACCORD)
  controller = LongControl(params, extension, supervised=True)
  first_stop = None
  braking_rows = 0
  for now_ns, plan_ns, model_ns, _, speed, accel, gas, brake, baseline_request, target, intent in recorded_inputs():
    car_state = SimpleNamespace(vEgo=speed, aEgo=accel, standstill=False, gasPressed=gas, brakePressed=brake,
                                cruiseState=SimpleNamespace(standstill=False))
    output = float(controller.update(True, car_state, target, False,
                                     (CarControllerParams.BOSCH_ACCEL_MIN, CarControllerParams.BOSCH_ACCEL_MAX),
                                     intent=intent, plan_time_ns=plan_ns, model_time_ns=model_ns, now_ns=now_ns))
    if gas or brake:
      assert output == 0.0
      assert controller.long_control_state == LongCtrlState.off
      assert controller.motion_phase == MotionPhase.OFF
    elif controller.long_control_state == LongCtrlState.stopping:
      if first_stop is None:
        first_stop = now_ns
      assert output <= baseline_request + 1e-6
      braking_rows += 1
  assert first_stop is not None and first_stop <= FIRST_ROUND_STOP_CONTROL_NS
  assert braking_rows > 20


def test_real_producer_publishes_measured_sources_through_capnp_wire():
  from openpilot.common.prefix import OpenpilotPrefix
  from openpilot.selfdrive.controls.tests.test_supervised_longitudinal_planner import make_supervised, make_supervised_state
  from openpilot.selfdrive.controls.tests.test_unified_longitudinal_planner import Publisher

  with OpenpilotPrefix():
    planner, _ = make_supervised(speed=0.7)
    state = make_supervised_state(speed=0.7)
    base = 2_000_000_000
    sources = tuple(base + index * 100_000 for index in range(len(RAW_SERVICES)))
    for service, timestamp in zip(RAW_SERVICES, sources, strict=True):
      state.logMonoTime[service] = timestamp
    planner.update(state, now_ns=base + 1_000_000)
    assert planner.motion_intent.is_valid()
    # Distinguish a reference node from the original measured input at the
    # actual publish boundary. No control or vehicle consumes this sentinel.
    planner.v_desired_trajectory[0] = 8.0
    publisher = Publisher()
    planner.publish(state, publisher)
    plan = publisher.messages['longitudinalPlan'].longitudinalPlan
    assert plan.motionIntent.version == MOTION_INTENT_VERSION
    assert plan.motionIntent.creepEvidenceValid
    assert tuple(plan.motionIntent.creepSourceMonoTimes) == sources
    assert plan.motionIntent.creepMeasuredSpeed == state['carState'].vEgo
    assert plan.motionIntent.creepMeasuredSpeed != plan.speeds[0]
    assert MotionIntent.from_message(plan.motionIntent) == planner.motion_intent


@pytest.mark.parametrize('warm_hold', [False, True])
def test_missing_wire_evidence_cannot_authorize_supervised_movement(warm_hold):
  from openpilot.common.prefix import OpenpilotPrefix
  from openpilot.selfdrive.controls.tests.test_unified_controlsd import ControlsHarness

  with OpenpilotPrefix():
    harness = ControlsHarness(unified=False, supervised=True)
    if warm_hold:
      harness.hold()
    harness.plan.aTarget = 0.5
    harness.plan.motionIntent.reasons = 0
    harness.plan.motionIntent.releaseAllowed = True
    harness.plan.motionIntent.creepEvidenceValid = False
    command, motion = harness.step()
    assert motion.degraded
    assert command.actuators.accel <= -0.2
    assert not command.cruiseControl.resume
    if warm_hold:
      assert motion.phase == 'hold'
