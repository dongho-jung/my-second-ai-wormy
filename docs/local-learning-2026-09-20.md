# 로컬 신경망 학습으로 갈 수 있는가 — 조사 결과와 계획

2026-09-20 조사. 결론부터: **된다.** 막고 있던 것은 사양이 아니라 "게임이 실시간
1배속으로만 돌아간다"는 가정이었고, 그 가정이 틀렸다.

이 문서의 경로는 전부 이 저장소(`my-second-ai-wormy`) 기준이다. 앞선 저장소를 가리킬
때는 `/Users/dongho/projects/my-first-ai-wormy`로 절대경로를 쓴다.

## 0. 이 저장소의 현재 위치

**실게임은 여전히 보기만 한다.** 상태 수집·지형 추출·대시보드에 조작 API는 없고,
그건 그대로다.

**오프라인 환경은 생겼다** (2026-09-20, 5절 1단계). `src/env/`가 공식 v20 번들을
브라우저 없이 돌려 관측·보상·결정론까지 준다. `npm test`가 49개를 검증하고,
`npm run rollout`이 실제로 에피소드를 돌린다.

아직 없는 것:

- **학습기** — 5절 2~3단계. 지금 돌릴 수 있는 정책은 랜덤뿐이다.
- **실게임 조작 경로** — 키 입력. 앞 저장소의 `/Users/dongho/projects/my-first-ai-wormy/src/controls.js`가
  참고 구현이다(키 바인딩을 게임 설정에 심고, 누른 키를 최소 100ms 유지해야 60Hz 샘플링에 잡힌다).

## 1. 확인된 사실 (전부 이 기계에서 측정)

### 게임 엔진이 브라우저 없이 돈다

공식 v20 번들은 클라이언트가 월드를 전부 시뮬레이션한다. `World.update()`가 한 틱이고
RNG는 시드 LCG 하나다. 브라우저도 서버도 렌더링도 없이 **순수 Node에서 그대로 실행된다.**
번들 SHA-256은 `b3c7b33c…`로, `src/adapter-v20.js`가 잠가둔 것과 **같은 파일**이다.

| 측정 (M2 Pro, 웜 2마리 이동+사격+리스폰) | 결과 |
|---|---|
| 1코어 | 46만 틱/s = **7,700배속** (한산한 맵 130만~200만) |
| 6프로세스 | 각 34만~37만, 합계 **약 220만 틱/s ≈ 36,000배속** |
| 결정론 | 같은 시드 2회 → 2만 틱 뒤 좌표·체력 소수점까지 동일 |

4틱 프레임스킵이면 **초당 55만 에이전트 스텝**. 1천만 스텝이 환경 기준 몇 분이다.

### 조작은 비트마스크 하나 + 메시지 둘

> **2026-09-20 정정.** 처음 이 문서는 비트 256을 "무기교체"로 적었다. 틀렸다.
> 번들의 `worm.tw()`에서 256은 `worm.nv()`를 부르고, `nv()`는 웜 앞 2·4px 지점의
> 흙을 깎는다 — **굴착**이다. 무기교체는 비트가 아니라 별도 메시지다.
> `test/env-engine.test.js`가 실제로 흙이 줄어드는 것과 슬롯이 바뀌는 것을 각각
> 확인한다.

웜에 `worm.Wa`를 쓰고 `world.update()`를 부르면 끝이다.

`1 좌 | 2 우 | 4 조준↑ | 8 조준↓ | 16 발사 | 32 점프 | 64/128 로프 길이 | 256 굴착`

**점프(32)와 굴착(256)은 누른 순간에만 동작한다.** 엔진이 직전 틱의 상태를 기억해서,
떼었다가 다시 눌러야 한 번 더 먹는다. 비트를 계속 켜 두는 정책은 딱 한 번 점프한다.

비트가 아닌 것이 둘 있고, 실제 방도 이 둘을 따로 보낸다.

- **로프** — 투척 `worm.kx(world)` / 해제 `worm.Pw()`. 네트워크 명령 하나(`Cb.hm`).
- **무기교체** — `worm.oq(worm.Ka + offset)`. **상대 이동**이며, 네트워크 명령(`Ab.offset`)도
  상대값이다. 스폰 때 `oq(a.Bf)`로 직전 슬롯을 복원한다.

실제 온라인 방도 똑같이 한다 — `room.ea`가 각 플레이어의 `Lb`를 `worm.Wa`에 복사하고
`world.update()`를 부른다. **학습한 정책이 실게임 키 입력에 1:1로 대응한다.**
그래서 `src/env/actions.js`의 액션도 `{ keys, rope, weapon }` 세 조각으로 나눠 두었다.
비트 하나로 합치면 실게임 이식에서 다시 쪼개야 한다.

### 맵은 엔진이 만들어 준다 (시드 재현 가능)

방을 만들 때 게임이 쓰는 생성기가 그대로 있다: `level.Jp(rng, settings, 504)` →
504x350 "Random Dirt". **`.lev` 파일이 아예 없어도 학습할 수 있다.**

단서가 하나 있다. 지형 모양을 만드는 Perlin 순열표가 `Math.random()`으로 섞여서,
시드만으로는 재현되지 않는다. `src/env/engine.js`의 `randomLevel(seed)`는 생성하는
동안만 `Math.random`을 시드 고정 LCG로 바꿔 끼우고 원래 것을 돌려놓는다. 그래서
같은 시드 → 바이트까지 같은 맵이다.

그리고 **키트에 받아둔 `.lev`들은 굴착이 의미가 없다**: Simple2는 흙이 0픽셀(전부
바위 아니면 공기), Arena_1도 176,400 중 1,457픽셀뿐이다. 생성 레벨은 흙이 99,223
픽셀로 절반이 넘는다. 지형을 파는 Liero를 학습시키려면 생성 레벨이 맞다.
레벨 생성은 1장에 약 7ms이므로, 에피소드마다 새로 만들면 짧은 에피소드에서는
무시 못 할 비용이다. 미리 N장 만들어 돌려쓰면 된다(`--level-pool N`).

### 학습된 모델 돌리는 비용

배치 1, torch 스레드 1개 (= 실게임과 같은 조건):

| 모델 | 크기 | 1회 판단 | 60Hz일 때 |
|---|---|---|---|
| 0.63M 파라미터 | 2.4 MB | **0.075 ms** | 코어 1개의 0.45% |
| 2.45M | 9.4 MB | 0.145 ms | 0.87% |
| 9.66M | 36.9 MB | 1.07 ms | 6.4% |

**추론에 GPU는 영원히 불필요.** 앞 저장소에서 유료 모델 한 번이 3.5~4.4초였으므로 약 5만 배 빨라진다.

### 학습 비용 (fwd+bwd+Adam, 배치 1024)

| | CPU 1스레드 | CPU 10스레드 | MPS |
|---|---|---|---|
| 지형 conv 정책 | 11,532/s | 5,984/s | **108,731/s** |
| 벡터 관측만 | **696,713/s** | 478,621/s | 1,209,578/s |

- **GPU 없어도 된다.** 벡터 관측이면 CPU 1스레드가 환경보다 빠르다.
- conv를 쓸 거면 MPS가 10배. 그래도 CPU만으로 1천만 스텝에 15분.
- **torch에 코어를 많이 주면 오히려 느리다** (10스레드가 1스레드의 절반).
  torch는 1~2스레드로 묶고 나머지 6~8코어는 환경 워커에 준다.

## 2. 엔진 로딩 — 저장소 코드와 키트

`src/env/engine.js`가 번들을 직접 읽어 `vm.runInThisContext`로 평가한다. **키트의
`probe.mjs`나 `engine.js`(패치된 사본)에 의존하지 않는다.** 필요한 건 내려받은 에셋
네 개뿐이고, 그 넷은 `fetch-assets.sh`가 받는 것과 정확히 같다.

- 실행 직전에 `game-v20.orig.js`의 SHA-256을 `src/adapter-v20.js`의 `CLIENT_SHA256`과
  대조해서 다르면 거부한다. 대시보드가 읽는 파일과 **같은 파일**이어야 물리가 같다.
- 패치는 메모리에서만 한다: 맨 끝 `u.js()`(UI 부팅) 호출을 클로저 안 클래스들을
  `globalThis`로 내보내는 한 줄로 치환. 파일을 만들지 않으므로 블랙박스가 남지 않는다.
- 클래스는 이름을 붙여서 꺼낸다(`World`=`Pa`, `Worm`=`V`, `Level`=`Z`, `Rng`=`eb`,
  `Zip`=`ib`, `Mod`=`U`, `Sprites`=`Jc`, `Wasm`=`A`, `Reader`=`H`).
- 프로세스당 엔진 하나다. 번들이 전역에 자기를 설치하므로 두 번째 디렉터리를 열어도
  첫 번째 것이 나온다. `loadEngine()`은 그래서 메모이즈한다.

### 키트 (`artifacts/headless-sim/`, gitignore 대상)

키트는 벤치마크와 탐색용으로 그대로 둔다.

```sh
cd artifacts/headless-sim
node patch-engine.mjs  # game-v20.orig.js -> engine.js (에셋이 이미 있으면 생략 가능)
node boot.mjs          # 엔진 부팅 + 월드 생성 + 웜 2마리
node fight.mjs         # 처리량 측정
node determinism.mjs   # 시드 재현성 확인
node obs-demo.mjs      # 두 가지 관측 설계를 실제 맵에서 출력
./fetch-assets.sh      # 에셋 다시 받기
```

torch 벤치마크는 venv를 만들어서 돌린다 (`artifacts/`는 gitignore이므로 안에 둬도 된다):

```sh
cd artifacts && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python torch numpy
.venv/bin/python headless-sim/infer-torch.py
.venv/bin/python headless-sim/train-cost.py
```

알아둘 것:

- 에셋은 **버전 경로**에서 받는다: `https://www.webliero.com/v/20/` 아래 `res.dat`,
  `vendor/wasm-flate.wasm`, `vendor/json5.min.js`, `game-min.js`.
  사이트 루트(`/res.dat`)는 404다. json5는 `var JSON5`로 선언되므로 평가할 때
  `globalThis.JSON5=JSON5`를 덧붙여야 한다.
- 클래스 지도: `Pa`=World(무인자 생성자, `update()`, `Of(other)` 전체 복사, `reset(seed)`),
  `V`=Worm, `Z`=Level(`read(name, arrayBuffer)`로 176,400바이트 .lev), `U.dj`=mod json5 파서,
  `ib.read`=zip, `A.Pc(url)`=wasm 로더, `Jc.read`=스프라이트.
- 설정은 `U.dj(mods/liero133/mod.json5)` + 스프라이트 → `.normalize()` → `world.s`에 대입.
  liero133 = 스톡 Liero 1.33, 무기 40종.
- 웜 생성: `world.ox(color, ownerId, [무기 5개 id])`.
- **리스폰은 `worm.nx()`만으로는 안 된다.** 죽은 웜은 그 틱에 `world.za`에서 빠지고
  `nx()`는 다시 넣어주지 않는다. `u=true` → `nx(world, loadout)` → `za.push(worm)`까지
  해야 그 웜이 계속 시뮬레이션된다. `src/env/engine.js`의 `respawnWorm()`이 그것이다.
- 방 설정은 월드 필드다: `qd`=bonusDrops(기본 2), `Pe`=bonusSpawnFrequency(1800틱),
  `mf`=weaponChangeDelay, `Te`=damageMultiplier, `le`=loadingTimes(0.4).
  학습에서는 보급품이 떨어진 자리가 승부를 가르므로 `bonusDrops: 0`을 기본으로 둔다.
- `world.reset(seed)`는 웜·투사체·틱을 비우지만 레벨은 그대로 둔다. 파인 지형을
  되돌리려면 `world.level.Of(원본)`까지 해야 한다.
- `world.Ga`(이벤트/사운드), `world.$p`(사망)는 null이어도 되고 모든 호출부가 null 검사를 한다.

## 3. 관측 — 구현된 것

권장대로 섞었다. 상태는 벡터, 지형과 투사체는 작은 conv 입력. 둘 다
`src/env/observation.js`에 있고 `observe(view, into, kinds)`로 원하는 쪽만 만든다.

### 벡터 76개 (`VECTOR_LAYOUT` / `VECTOR_OFFSETS`로 내보낸다)

| 구간 | 개수 | 내용 |
| --- | --- | --- |
| `rays` | 16 | 오른쪽부터 시계방향 16방향, 첫 고체까지 거리 / 120 |
| `health` `velocity` `aim` `facing` | 6 | 체력, 속도 px/틱, 조준 cos·sin, 좌우 |
| `contacts` `stepping` | 5 | 엔진 자신의 상하좌우 접촉 수, 자동 턱오름 |
| `walkLeft` `walkRight` | 8 | clear/step/dirt/rock 원핫 |
| `weapons` | 15 | 슬롯 5개 × (탄 비율, 지금 쏠 수 있나, 선택됨) |
| `rope` | 5 | 나감·걸림·상대 위치·길이 |
| `foe` | 9 | 살아있나·상대 위치·거리·방향·체력·속도 |
| `projectiles` | 12 | 가까운 3발의 상대 위치와 속도 |

조준은 각도 대신 cos·sin으로 넣는다(π에서 값이 끊기지 않게). 거리는 300px로 나눠
자르되 방향은 단위벡터로 따로 넣어서, 잘려도 어느 쪽인지는 남는다.
**투사체는 자기 것도 넣는다** — 자기 폭발이 받는 피해의 절반 가까이이므로 숨길 이유가 없다.

### 지형 패치 4 × 32 × 32 = 4,096

바위 / 흙 / 빈공간 / 투사체. 셀 하나가 2×2 실제 픽셀을 맡고 **그 중 가장 단단한 것**을
답한다. 1픽셀짜리 벽이 샘플 사이로 빠져 "빈 공간"으로 보이면 안 되기 때문이다.
맵 밖은 바위로 읽는다(실제로 그렇게 행동한다). 앞의 세 채널은 배타적이라 셀마다 정확히
하나가 1이고, 그래서 "막혔다"와 "측정 안 됨"이 구분된다.

### 어느 쪽이 얼마나 드는가 (측정)

| | 1회 | 초당 |
| --- | --- | --- |
| 벡터만 | 0.0029 ms | 34만 |
| 벡터 + 패치 | 0.017 ms | 5.7만 |

**패치가 벡터의 약 10배다.** 그래서 `observations: ["vector"]`를 고를 수 있게 해 두었다.
걸어가기 같은 과제는 벡터만으로 충분하고, 속도는 그대로 돌려받는다.

### 실게임과 같은 벡터를 만드는 장치

관측 인코더는 엔진도 어댑터도 직접 읽지 않는다. 둘 다 `src/env/view.js`의 **뷰 하나**로
바꾼 다음 그것만 읽는다.

- `viewFromWorld(world, self, foes)` — 헤드리스 월드에서
- `viewFromSnapshot(state, terrain)` — 실게임 `/state` + `/map`에서

필드 이름은 어댑터 것을 그대로 쓴다. `test/env-observation.test.js`는 전 과정을
**실게임 쪽 경로로** 돌린다(픽스처 → `snapshotV20` → 뷰 → 벡터). 어댑터가 주는 값과
인코더가 바라는 값이 어긋나면 첫 실전이 아니라 테스트가 먼저 깨진다.

접촉 수·걷기 판정은 어댑터에도 같은 계산이 인라인돼 있다(페이지 안에서 평가되므로
모듈을 못 쓴다). 사본이 둘인 건 의도이고, 같은 픽스처로 둘을 대조하는 테스트가 있다.

## 4. 제안 구조

- **환경**: `src/env/env.js`의 `WormEnv`. 만들어졌다 — 아래 4-1절.
- **보상**: 상대 체력 감소 − 내 체력 감소, 킬/데스. 만들어졌다(`src/env/reward.js`).
  자기 폭발 피해가 그대로 마이너스로 들어간다 (앞 저장소 측정에서 전체 피해의 41~56%가
  자기 탓이었고, 그걸 규칙으로 막을 필요가 없어진다). 죽음과 리스폰은 뺄셈으로 처리하면
  틀린다: 죽은 웜은 남아 있던 체력만큼 잃은 것이고, 리스폰한 100은 얻은 게 아니다.
  구급상자로 오른 체력은 음수 피해가 아니라 0으로 센다.
- **학습**: PyTorch PPO + self-play(과거 스냅샷 리그). **아직 없다.**
- **부트스트랩**: 앞 저장소에 녹화된 사람 시연이 있다
  (`/Users/dongho/projects/my-first-ai-wormy/artifacts/`, 로프 기술 포함). 모방학습(BC)으로
  먼저 흉내 내게 하면 로프처럼 랜덤 탐색으로는 안 나오는 기술을 건너뛴다.
- **배치**: CPU 6~8코어 = 환경, torch 1~2스레드 + MPS = 학습기.
- **실게임 이식**: 물리는 같은 코드라 차이 0. 남는 차이는 네트워크 지연과 키 샘플링뿐이므로
  학습 때 입력 지연 0~3틱을 랜덤으로 섞는다 — `inputLatencyTicks: [0, 3]`으로 이미 된다.
  배포는 (a) 소켓으로 Python 정책, (b) ONNX로 Node 드라이버 안,
  (c) 가중치를 페이지에 넣고 거기서 — 셋 다 가능.

### 4-1. 환경 쓰는 법

```js
import { loadEngine } from "./src/env/engine.js";
import { WormEnv } from "./src/env/env.js";
import { KEYS, ROPE } from "./src/env/actions.js";

const engine = await loadEngine();              // 프로세스당 한 번
const env = new WormEnv(engine, {
  agents: 2,
  frameskip: 4,                                 // 15Hz로 판단
  episodeTicks: 3600,                           // 게임시간 1분
  inputLatencyTicks: [0, 3],                    // 실게임 이식용
  observations: ["vector"],                     // 패치가 필요해지면 "patch" 추가
  rules: { bonusDrops: 0 },
  level: (engine, seed) => pool[seed % pool.length],   // 생략하면 에피소드마다 새 맵
});

const { observations } = env.reset({ seed: 11 });
const { rewards, done, info } = env.step([
  KEYS.right | KEYS.fire,                       // 비트마스크만 줘도 되고
  { keys: KEYS.left, rope: ROPE.throw, weapon: 1 },   // 전부 줘도 된다
]);
```

- `observations[i]`는 `{ vector, patch }`이고 **버퍼를 재사용한다.** 스텝 사이에 값을
  간직하려면 복사해야 한다.
- `info.events[i]`가 그 스텝의 `{ damageDealt, damageTaken, killed, died }`,
  `info.totals[i]`가 에피소드 누계다.
- 보상은 **리스폰 전에** 읽는다. 아니면 죽음이 "체력이 다시 찼다"로 보인다.
- 시드 하나가 맵·월드 RNG·입력 지연을 전부 정한다. 같은 시드 + 같은 액션 = 같은 에피소드이고,
  `test/env-engine.test.js`가 소수점까지 대조한다.

### 4-2. 진행 상황 보기

학습은 몇 시간을 돈다. 되고 있는지 보려면 페이지가 필요하다.

```sh
npm run rollout -- --episodes 200 --level-pool 8   # 에피소드마다 한 줄씩 기록
npm run monitor                                    # http://127.0.0.1:8768
```

- 기록은 `artifacts/runs/<id>/`에 `run.json` + `metrics.jsonl`로 쌓인다. 덧붙이기만 하는
  텍스트라 중간에 죽어도 그때까지가 남고, 모니터는 **파일이 커진 만큼만** 읽는다.
  학습기는 누가 보고 있는지 알 필요가 없다.
- 기록의 **숫자 필드는 전부 자동으로 차트가 된다.** 학습기가 `policyLoss`를 남기기
  시작하면 페이지는 고치지 않아도 그게 뜬다. 아는 이름에만 한글 라벨과 "어느 쪽이
  좋은 방향인지"가 붙어서, 오르면 초록/빨강이 갈린다.
- 게임 대시보드(8766)와는 **완전히 별개 페이지·별개 포트**다. 학습에 브라우저가 필요 없고,
  대시보드에 학습이 필요 없다. 모니터는 읽기 전용이라 실행을 시작·중단할 수 없다.

## 5. 단계

1. ~~env 래퍼 + 관측/보상 + 결정론 테스트 (`npm test`에 붙인다)~~ **완료 2026-09-20.**
   `src/env/` 6개 모듈, 테스트 49개(그 중 25개가 이번 것), `npm run rollout`,
   그리고 진행 상황 페이지(4-2절).
2. **한 지점까지 걸어가기**를 RL로 — 파이프라인 검증. 몇 분~몇 시간이면 결과가 나온다.
   앞 저장소의 손으로 짠 경로탐색이 같은 과제를 몇 초에 푸는지가 비교 기준이다.
3. 1:1 전투 self-play (1천만~1억 스텝)
4. 실게임 이식 — 먼저 키 입력 경로부터 만들어야 한다(0절)

### 1단계에서 실제로 나온 처리량 (M2 Pro, 1코어, 2에이전트)

| 구성 | 에이전트 스텝/s | 배속 |
| --- | --- | --- |
| 벡터만 | 76,485 | 2,549x |
| 벡터 + 지형 패치 | 36,509 | 1,217x |

엔진만 돌릴 때(1절, 130만~200만 틱/s)와 비교하면 관측이 비용의 대부분이다.
6프로세스면 벡터 기준 초당 40만~50만 스텝이므로 **1천만 스텝이 20~30초**,
패치까지면 1~2분이다. 1억 스텝도 하룻밤이 아니라 십수 분 규모다.

## 6. 아직 안 정한 것

- 어디서 시작할지: 2단계(걸어가기)부터 / 바로 전투 self-play / BC부터
- **학습기를 어디에 둘지.** 문서는 PyTorch PPO를 전제로 썼지만, 벡터 관측이면
  JS로 짜도 된다(1절 측정: JS MLP 순전파 0.138ms). Python이면 환경을 소켓/파이프로
  내보내야 하고, JS면 프로세스 하나로 끝나는 대신 conv는 onnxruntime 쪽으로 미뤄야 한다.
- 앞 저장소의 reflex·Jev를 상대역·베이스라인으로 끌어올지, 아예 무시할지

## 7. 이 저장소에서 지킬 것

- **push·PR·CI 금지** — 그 메시지에서 명시적으로 요청하지 않는 한. 앞 저장소에서 사용자가
  분명히 한 방침이고 같은 프로젝트이므로 여기에도 적용한다. 로컬 `npm test`로 검증하고
  커밋은 로컬에만 둔다.
- 동작 변경은 **직접 측정해서 확인한다.** 사용자에게 "돌려보고 알려달라"고 넘기지 않는다.
- 이 저장소의 포트: 대시보드 **8766**, 픽스처 미리보기 **8767**, 학습 모니터 **8768**,
  Chromium CDP **9334**(앞 저장소의 8765/9333과 겹치지 않게 고른 값이다).
  Chromium 프로필은 `~/.cache/wormy-ii/chrome-profile`로 공유한다.
- CDP로 붙은 브라우저에 **`browser.close()` 금지** — 방이 죽고 다음 방 생성에 CAPTCHA가 붙는다.
  마지막 페이지를 닫아도 같다.
- 시작 전에 `.ai-memory`를 읽는다. 조사 결과는 `game.headless_engine`,
  `learning.model_cost`, 환경과 모니터는 `env.wrapper`, `training.monitor` 키에 있다.
- Node 26 (`.nvmrc`).
