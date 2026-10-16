"""Recorded Q11 risk inputs retained after SUPERVISED sampled-creep retirement.

The recorded boundary is route 0000005a--0f5cae6edb, Q11
R-7d9151776fadaaae / J-a9f9815e7b634916. Source: 2026-09-16
supervised-q11-residuals-v5 raw-inputs, control-timing and owner-reconstruction
artifacts, plus the core execution-contract integer ID edges. Raw source IDs
and measured speeds remain provenance; they do not recreate the retired
sample/off-credit policy. The unchanged reconstructed stop/moving/release facts
are now carried by the current UNIFIED intent.

The recorded inputs contain no explicit stop reason. The current policy keeps
their negative tracking requests without recreating the historical independent
creep stop. Synthetic budget tests below supply explicit stop evidence and
separate that immediate stop from the later CREEP bit. This fixed-input owner
regression is neither full planner replay nor proof of physical braking.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import CREEP_SPEED, CREEP_STOP_T, STOP_SPEED
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionIntent, MotionPhase, MotionReferenceKind, MotionSupervisor, StopReason


FIRST_ROUND_STOP_CONTROL_NS = 2253275731795

SECOND_ROUND_STOP_CONTROL_NS = 2253315097941

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
    sources, _, reset, target = RECORDED_PLANS[plan_ns]
    intent = MotionIntent(allow_creep=not reset)
    yield now_ns, plan_ns, sources[2], car_state_ns, speed, accel, gas, brake, baseline_request, target, intent


def sample_step(owner, timestamp, *, intent=None, v_ego=0.7, a_ego=0.0, a_target=-0.2,
                active=True, now_ns=None, plan_ns=None, model_ns=None, **kwargs):
  return owner.update(active, v_ego, a_ego, False, a_target, MotionIntent(reasons=StopReason.E2E) if intent is None else intent,
                      plan_time_ns=timestamp + 1_000_000 if plan_ns is None else plan_ns,
                      model_time_ns=timestamp if model_ns is None else model_ns,
                      now_ns=timestamp + 2_000_000 if now_ns is None else now_ns, **kwargs)


def test_recorded_pedal_reentry_does_not_invent_a_stop_without_current_evidence():
  owner = MotionSupervisor()
  active_rows = 0
  for now_ns, plan_ns, model_ns, _, speed, accel, gas, brake, _, target, intent in recorded_inputs():
    stop = owner.update(not gas and not brake, speed, accel, False, target, intent,
                        now_ns=now_ns, plan_time_ns=plan_ns, model_time_ns=model_ns)
    if gas or brake:
      assert not stop
      assert owner.phase == MotionPhase.OFF
    else:
      assert not intent.reasons
      assert not stop
      active_rows += 1
    assert not owner.reasons & StopReason.CREEP
    assert owner.creep_total_t == 0.0
  assert active_rows > 50


def test_unified_keeps_its_existing_control_time_contract():
  owner = MotionSupervisor()
  base = 2_000_000_000
  intent = MotionIntent(reasons=StopReason.E2E)
  for tick in range(20):
    assert sample_step(owner, base, intent=intent, now_ns=base + 2_000_000 + tick * 10_000_000)
    assert not owner.reasons & StopReason.CREEP
  assert owner.creep_total_t == pytest.approx(0.2)


def test_recorded_longcontrol_keeps_the_negative_reference_and_manual_brake_wins():
  from opendbc.car.honda.interface import CarInterface
  from opendbc.car.honda.values import CAR, CarControllerParams
  from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState

  params = CarInterface.get_non_essential_params(CAR.HONDA_ACCORD)
  params.openpilotLongitudinalControl = True
  extension = CarInterface.get_non_essential_params_sp(params, CAR.HONDA_ACCORD)
  controller = LongControl(params, extension)
  tracking_rows = 0
  for now_ns, plan_ns, model_ns, _, speed, accel, gas, brake, _baseline_request, target, intent in recorded_inputs():
    car_state = SimpleNamespace(vEgo=speed, aEgo=accel, standstill=False, gasPressed=gas, brakePressed=brake,
                                cruiseState=SimpleNamespace(standstill=False))
    output = float(controller.update(True, car_state, target, False,
                                     (CarControllerParams.BOSCH_ACCEL_MIN, CarControllerParams.BOSCH_ACCEL_MAX),
                                     intent=intent, plan_time_ns=plan_ns, model_time_ns=model_ns, now_ns=now_ns))
    if gas or brake:
      assert output == 0.0
      assert controller.long_control_state == LongCtrlState.off
      assert controller.motion_phase == MotionPhase.OFF
    elif not controller.motion_input_constrained:
      assert controller.long_control_state == LongCtrlState.pid
      assert not controller.resolved_should_stop
      assert output == pytest.approx(target)
      tracking_rows += 1
    assert not controller.motion_reasons & StopReason.CREEP
  assert tracking_rows > 50


@pytest.mark.parametrize('off_valid', [True, False])
@pytest.mark.parametrize('reentry_now_offset', [60_000_000, 90_000_000])
def test_off_clock_rejects_duplicate_or_backward_reentry(off_valid, reentry_now_offset):
  owner = MotionSupervisor()
  base = 2_000_000_000
  sample_step(owner, base - 50_000_000)
  sample_step(owner, base, active=False, now_ns=base + 90_000_000, plan_valid=off_valid)
  assert owner.phase == MotionPhase.OFF
  sample_step(owner, base + 50_000_000, now_ns=base + reentry_now_offset)
  assert owner.degraded
  assert owner.creep_total_t == 0.0
  sample_step(owner, base + 100_000_000, now_ns=base + 110_000_000)
  assert not owner.degraded
  assert 0.0 < owner.creep_total_t <= 0.02


def test_off_observation_never_accumulates_active_creep_time():
  owner = MotionSupervisor()
  base = 2_000_000_000
  for index in range(30):
    timestamp = base + index * 50_000_000
    assert not sample_step(owner, timestamp, active=False)
    assert owner.phase == MotionPhase.OFF
    assert owner.creep_total_t == 0.0
  assert not sample_step(owner, timestamp, now_ns=timestamp + 12_000_000)
  assert owner.waiting_for_fresh_input and not owner.degraded
  assert owner.creep_total_t == 0.0
  assert sample_step(owner, timestamp + 50_000_000)
  assert not owner.degraded
  assert owner.creep_total_t == pytest.approx(0.04)


def _progress_window(speed_for_frame, *, frames=60, intent_for_frame=None, target=-0.22, rotating_producer=False):
  """Current measurements at 100 Hz, with independently advancing 20 Hz plans."""
  owner = MotionSupervisor()
  base = 2_000_000_000
  history = []
  previous_speed = speed_for_frame(0)
  for frame in range(frames):
    producer_ns = base + frame // 5 * 50_000_000
    instance_ns = base + (frame // 10 * 100_000_000 if rotating_producer else 0)
    intent = MotionIntent(reasons=StopReason.E2E) if intent_for_frame is None else intent_for_frame(frame)
    intent = replace(intent, producer_instance_ns=instance_ns)
    speed = speed_for_frame(frame)
    stopped = sample_step(owner, producer_ns, intent=intent, v_ego=speed,
                          a_ego=(speed - previous_speed) / DT_CTRL, a_target=target,
                          now_ns=base + 2_000_000 + frame * 10_000_000)
    assert not owner.degraded
    if intent.reasons:
      assert stopped
    history.append({'resolved_stop': stopped, 'creep': bool(owner.reasons & StopReason.CREEP),
                    'phase': owner.phase, 'elapsed': (frame + 1) * DT_CTRL, 'speed': speed,
                    'previous_speed': previous_speed, 'anchor': owner.creep_v, 'window': owner.creep_t,
                    'distance': owner.creep_distance, 'total': owner.creep_total_t, 'moving': owner.moving_lead_t})
    previous_speed = speed
  return owner, history


@pytest.mark.parametrize('profile', ['constant', 'growing', 'prompt_deceleration'])
def test_distance_budget_changes_only_growing_speed_without_stopping_progress(profile):
  speeds = {
    'constant': lambda frame: 0.7,
    'growing': lambda frame: 0.7 + 0.004 * frame,
    'prompt_deceleration': lambda frame: 0.95 - 0.008 * frame,
  }
  _, rows = _progress_window(speeds[profile])
  first = next((row for row in rows if row['creep']), None)
  if profile == 'prompt_deceleration':
    assert first is None
    # A completed prompt-deceleration window clears its distance along with
    # its time; the next window is anchored in the current measured speed.
    assert rows[50]['window'] == rows[50]['distance'] == 0.0
    assert rows[51]['anchor'] == pytest.approx(rows[51]['speed'])
    assert rows[51]['distance'] == pytest.approx(rows[51]['speed'] * DT_CTRL)
  elif profile == 'constant':
    assert first['elapsed'] == pytest.approx(CREEP_STOP_T + DT_CTRL)
  else:
    assert first['elapsed'] < CREEP_STOP_T
    assert min(first['previous_speed'], first['speed']) > first['anchor'] >= STOP_SPEED
    assert first['distance'] > first['anchor'] * CREEP_STOP_T


@pytest.mark.parametrize('spike_frame', [0, 1, 10, 25, 45, 49])
def test_one_speed_spike_cannot_advance_the_constant_speed_window(spike_frame):
  _, rows = _progress_window(lambda frame: 1.1 if frame == spike_frame else 0.7)
  assert not any(row['creep'] for row in rows[:50])


@pytest.mark.parametrize('spike_frame', [41, 43, 44, 45, 46, 48, 49])
@pytest.mark.parametrize('repeated_controls', [2, 3])
def test_repeated_consumption_of_one_speed_spike_cannot_confirm_early_stop(spike_frame, repeated_controls):
  _, rows = _progress_window(lambda frame: 1.1 if spike_frame <= frame < spike_frame + repeated_controls else 0.4)
  assert not any(row['creep'] for row in rows[:50])


@pytest.mark.parametrize('advance_identity', ['neither', 'plan_only', 'model_only'])
def test_distance_excess_needs_fresh_plan_and_model_confirmation(advance_identity):
  owner, _ = _progress_window(lambda frame: 0.7 + 0.004 * frame, frames=43)
  # Above-anchor producer observations already exist. A later high-speed
  # control sample spends the distance budget, but cached/one-sided identities
  # still cannot establish the additional early stop cause.
  plan_ns, model_ns = owner._last_plan_time_ns, owner._last_model_time_ns
  for frame in (43, 44):
    if advance_identity == 'plan_only':
      plan_ns += 1
    elif advance_identity == 'model_only':
      model_ns += 1
    sample_step(owner, 2_400_000_000, v_ego=1.1, a_target=-0.22,
                intent=MotionIntent(reasons=StopReason.E2E, producer_instance_ns=2_000_000_000), plan_ns=plan_ns, model_ns=model_ns,
                now_ns=2_002_000_000 + frame * 10_000_000)
    assert not owner.reasons & StopReason.CREEP
  if advance_identity == 'model_only':
    assert owner.degraded
  else:
    assert not owner.degraded
    assert owner.creep_distance > owner.creep_v * CREEP_STOP_T


@pytest.mark.parametrize('entry_speed', [0.0, 1e-6, 0.05, STOP_SPEED - 1e-6])
def test_near_zero_distance_budget_does_not_turn_small_motion_into_early_stop(entry_speed):
  _, rows = _progress_window(lambda frame: entry_speed if frame == 0 else entry_speed + 0.01, frames=50)
  assert not any(row['creep'] for row in rows)


def test_pending_moving_lead_does_not_start_a_budget_without_stop_evidence():
  _, rows = _progress_window(lambda frame: 0.7 + 0.004 * frame,
                            intent_for_frame=lambda frame: MotionIntent(moving_lead=True))
  assert not any(row['creep'] for row in rows)
  assert all(row['distance'] == row['total'] == 0.0 and not row['resolved_stop'] for row in rows)
  assert rows[-1]['moving'] == pytest.approx(CREEP_STOP_T)
  assert rows[-1]['window'] == rows[-1]['distance'] == rows[-1]['total'] == 0.0


def test_late_moving_evidence_is_not_required_to_avoid_an_unrequested_stop():
  _, rows = _progress_window(lambda frame: 0.7 + 0.004 * frame,
                            intent_for_frame=lambda frame: MotionIntent(moving_lead=frame >= 25))
  assert all(not row['resolved_stop'] and row['total'] == 0.0 for row in rows)


@pytest.mark.parametrize('growing', [False, True])
def test_rolling_permission_without_stop_evidence_has_no_creep_deadline(growing):
  # The current negative target still brakes. Neither fresh permission nor
  # its repeated consumption creates a separate stop-assistance budget.
  _, rows = _progress_window(lambda frame: 0.7 + (0.004 * frame if growing else 0.0), frames=110,
                            intent_for_frame=lambda frame: MotionIntent(reference_kind=MotionReferenceKind.FORWARD,
                                                                       release_allowed=True))
  assert not any(row['creep'] for row in rows)
  assert all(row['total'] == 0.0 and not row['resolved_stop'] for row in rows)


@pytest.mark.parametrize('profile', ['growing', 'prompt_deceleration'])
def test_transient_moving_evidence_cannot_create_a_stop_deadline(profile):
  speed = (lambda frame: 0.7 + 0.004 * frame) if profile == 'growing' else (lambda frame: 1.15 - 0.006 * frame)
  _, rows = _progress_window(speed, frames=110,
                            intent_for_frame=lambda frame: MotionIntent(moving_lead=frame >= 25 and frame % 10 >= 5))
  assert any(row['moving'] > 0.0 for row in rows)
  assert all(row['total'] == 0.0 and not row['resolved_stop'] for row in rows)


@pytest.mark.parametrize('target', [0.0, 0.01, 0.2])
def test_positive_or_neutral_target_cannot_spend_the_distance_budget(target):
  _, rows = _progress_window(lambda frame: 0.7 + 0.004 * frame, target=target)
  assert not any(row['creep'] for row in rows)
  assert all(row['distance'] == row['total'] == 0.0 for row in rows)


def test_withheld_creep_qualification_cannot_spend_distance():
  _, rows = _progress_window(lambda frame: 0.7 + 0.004 * frame,
                            intent_for_frame=lambda frame: MotionIntent(reasons=StopReason.E2E, allow_creep=False))
  assert not any(row['creep'] for row in rows)
  assert all(row['distance'] == row['total'] == 0.0 for row in rows)


@pytest.mark.parametrize('fault', ['repeat_clock', 'backward_clock', 'clock_gap', 'invalid_plan', 'expired_plan', 'inactive', 'out_of_band'])
def test_original_window_revocation_paths_clear_distance(fault):
  owner, rows = _progress_window(lambda frame: 0.7 + 0.004 * frame, frames=30)
  assert rows[-1]['distance'] > 0.0 and not rows[-1]['creep']
  timestamp = 2_300_000_000
  options = {'now_ns': timestamp + 2_000_000, 'intent': MotionIntent(reasons=StopReason.E2E, producer_instance_ns=2_000_000_000)}
  if fault in ('repeat_clock', 'backward_clock'):
    options['now_ns'] = owner._last_now_ns - int(fault == 'backward_clock')
  elif fault == 'clock_gap':
    options['now_ns'] = owner._last_now_ns + 100_000_001
  elif fault == 'invalid_plan':
    options['plan_valid'] = False
  elif fault == 'expired_plan':
    options.update(plan_ns=2_001_000_000, model_ns=2_000_000_000)
  elif fault == 'inactive':
    options['active'] = False
  elif fault == 'out_of_band':
    options['v_ego'] = CREEP_SPEED
  sample_step(owner, timestamp, **options)
  assert owner.creep_t == owner.creep_total_t == owner.creep_distance == 0.0


def test_continuous_owner_deadline_cannot_be_extended_by_rotating_planner_instances():
  _, original = _progress_window(lambda frame: 0.7 + 0.004 * frame)
  _, restarted = _progress_window(lambda frame: 0.7 + 0.004 * frame, rotating_producer=True)
  assert restarted == original


def test_distance_stop_preserves_stronger_braking_and_qualified_release():
  owner, rows = _progress_window(lambda frame: 0.7 + 0.004 * frame)
  assert owner.reasons & StopReason.CREEP
  distance = owner.creep_distance
  # A stronger requested brake cannot turn a stopped phase into a release or
  # start a new budget. The controller test below checks its actual output.
  assert sample_step(owner, 2_650_000_000, intent=MotionIntent(reasons=StopReason.E2E, producer_instance_ns=2_000_000_000),
                     a_target=-2.0)
  assert owner.phase == MotionPhase.APPROACH
  assert owner.creep_distance == distance
  assert not sample_step(owner, 2_700_000_000,
                         intent=MotionIntent(release_allowed=True, reference_kind=MotionReferenceKind.FORWARD,
                                             producer_instance_ns=2_000_000_000), a_target=0.3)
  assert owner.phase == MotionPhase.RELEASE
  assert owner.creep_distance == owner.creep_total_t == 0.0


@pytest.mark.parametrize('pedal', ['gasPressed', 'brakePressed'])
def test_distance_stop_keeps_stronger_controller_braking_and_immediate_pedal_revocation(pedal):
  from opendbc.car.honda.interface import CarInterface
  from opendbc.car.honda.values import CAR, CarControllerParams
  from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState

  params = CarInterface.get_non_essential_params(CAR.HONDA_ACCORD)
  params.openpilotLongitudinalControl = True
  extension = CarInterface.get_non_essential_params_sp(params, CAR.HONDA_ACCORD)
  controller = LongControl(params, extension)
  limits = (CarControllerParams.BOSCH_ACCEL_MIN, CarControllerParams.BOSCH_ACCEL_MAX)
  cs = SimpleNamespace(vEgo=0.7, aEgo=0.4, standstill=False, gasPressed=False, brakePressed=False,
                       cruiseState=SimpleNamespace(standstill=False))
  intent = MotionIntent(reasons=StopReason.E2E, producer_instance_ns=2_000_000_000)
  for frame in range(50):
    cs.vEgo = 0.7 + 0.004 * frame
    model_ns = 2_000_000_000 + frame // 5 * 50_000_000
    output = controller.update(True, cs, -0.22, False, limits, intent=intent,
                               plan_time_ns=model_ns + 1_000_000, model_time_ns=model_ns,
                               now_ns=2_002_000_000 + frame * 10_000_000)
    assert output <= -0.22
  assert controller.motion_reasons & StopReason.CREEP
  assert controller.motion_supervisor.creep_t < CREEP_STOP_T
  output = controller.update(True, cs, -2.5, False, limits, intent=intent,
                             plan_time_ns=2_501_000_000, model_time_ns=2_500_000_000, now_ns=2_502_000_000)
  assert output <= -2.5
  setattr(cs, pedal, True)
  assert controller.update(True, cs, -2.5, False, limits, intent=intent,
                           plan_time_ns=2_501_000_000, model_time_ns=2_500_000_000, now_ns=2_512_000_000) == 0.0
  assert controller.long_control_state == LongCtrlState.off
  assert controller.motion_phase == MotionPhase.OFF
  assert controller.motion_supervisor.creep_distance == controller.motion_supervisor.creep_total_t == 0.0
