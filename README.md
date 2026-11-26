# Connect 4 × ROBOTIS OMX (MuJoCo)

OMX-F 팔이 Connect 4를 두는 시뮬레이션 환경. 로봇은 빨강, 사람은 노랑.

## 실행

```bash
uv venv --python 3.12 && uv pip install -r pyproject.toml
uv pip install -r pyproject.toml --extra vlm   # VLM 보드 인식까지 쓸 때 (torch, transformers)
.venv/bin/python scripts/play.py          # 터미널에 열 번호(1~7) + Enter → 로봇이 응수
.venv/bin/python scripts/play.py --auto   # 사람 쪽도 랜덤으로 자동 진행
```

열 번호는 로봇 쪽에서 봤을 때 왼쪽이 1번(코드에서는 `col 0`, +y 방향).

## 구성

| 경로 | 내용 |
| --- | --- |
| `assets/open_manipulator_description/` | ROBOTIS-GIT/open_manipulator (commit `0f18f88`)의 OMX-F URDF, 같은 URDF에서 만든 공식 MJCF, 메시 |
| `connect4_sim/config.py` | 보드·디스크 치수, 배치(보드, 피더, 로봇 받침대 높이) |
| `connect4_sim/scene.py` | 로봇 + 보드 + 디스크 + 카메라 씬 생성 |
| `connect4_sim/env.py` | `Connect4Env`: `step`, `render`, `board_state`, `human_move`, `robot_move` |
| `connect4_sim/controller.py` | IK 기반 스크립트 pick-and-place (ACT 데모 생성용 expert) |
| `connect4_sim/game.py` | 규칙(둘 수 있는 열, 승리 판정) |
| `board_cam/camera.py` | 사람 쪽에서 보드를 보는 카메라 포즈, 이미지 속 격자 영역(`GRID_BOX`)과 크롭 |
| `board_cam/px_avg/` | 42칸 위치의 픽셀 색 평균으로 보드를 읽는 `read_board()`, 비교 실험과 결과(`test/`) |
| `board_cam/qwen4B/` | VLM(Qwen3-VL-4B)으로 보드를 읽는 `BoardReader`, 비교 실험과 결과(`test/`) |
| `board_cam/board_cam.py` | 카메라 이미지 → 보드 상태 출력 → 다음 수를 묻는 LLM 프롬프트 생성 |
| `board_cam/evaluation.py` | 두 실험이 같이 쓰는 랜덤 국면 생성과 정확도 집계 |
| `scripts/play.py` | 뷰어에서 대국. `choose_column()`이 VLM/LLM을 꽂을 자리 |

```python
from connect4_sim.env import Connect4Env, run
env = Connect4Env()
run(env.human_move(3))            # 노란 디스크를 4번째 열에 떨어뜨림
run(env.robot_move(2))            # 피더에서 집어 3번째 열에 넣음
env.board_state()                 # (6, 7) 정답 보드. 0 빈칸, 1 로봇, 2 사람, row 0이 맨 아래
env.render()                      # board_cam RGB 이미지 1280x960 (다른 카메라: env.render("overview"))
env.step(ctrl); env.joint_positions()   # 50 Hz 저수준 제어: joint1~5 + gripper 목표각(rad)
```

## 보드 인식

카메라는 사람 쪽에서 로봇을 향하므로 이미지는 로봇 시점과 좌우가 반대다(이미지 왼쪽 = 7번 열). 두 방식 모두 뒤집어서 `env.board_state()`와 같은 배치로 돌려준다.

```python
from board_cam.px_avg.reader import read_board      # 픽셀 색 평균 (기본)
board = read_board(env.render())

from board_cam.qwen4B.vlm import BoardReader        # VLM
reader = BoardReader(mode="rows")          # "full" 전체 이미지 / "crop" 보드만 / "rows" 행별 6회 질의
board, answer = reader.read(env.render())  # 형식이 깨지면 board는 None
```

비교 실험(같은 seed의 랜덤 국면 50개, 2100칸, RTX 5080):

```bash
MUJOCO_GL=egl .venv/bin/python board_cam/px_avg/test/eval_px_avg.py --n-boards 50   # 결과: board_cam/px_avg/test/results/
MUJOCO_GL=egl .venv/bin/python board_cam/qwen4B/test/eval_vlm.py --n-boards 50      # 결과: board_cam/qwen4B/test/results/
```

| 방식 | 칸 정확도 | 보드 전체 일치 | 빨강 recall | 노랑 recall | 장당 시간 |
| --- | --- | --- | --- | --- | --- |
| px_avg | 100% | 50/50 | 100% | 100% | 0.6 ms |
| VLM full | 72.1% | 1/50 | 56.3% | 57.9% | 2.0 s |
| VLM crop | 89.1% | 8/50 | 82.1% | 82.0% | 1.8 s |
| VLM rows | 96.7% | 28/49 (1개는 형식 오류) | 92.7% | 94.5% | 1.3 s |

px_avg는 카메라와 보드가 고정이라는 전제에 의존한다. 샘플링 격자를 일부러 밀었을 때 칸 정확도는 4 px까지 100%, 8 px 99.2%, 12 px 54.6%(칸 간격 약 60 px). 밝기 0.4~1.5배에서는 100%. 시뮬레이션은 조명이 균일하고 노이즈가 없으므로, 실물에서는 `GRID_BOX`와 `reader.py`의 임계값을 실제 영상으로 다시 맞춰야 한다.

## 다음 수 프롬프트

```bash
MUJOCO_GL=egl .venv/bin/python board_cam/board_cam.py --random-moves 9   # 시뮬 국면의 보드 상태와 프롬프트 출력
.venv/bin/python board_cam/board_cam.py --image frame.png                # 저장된 board_cam 이미지로
```

```python
from board_cam.board_cam import build_prompt, parse_column
prompt = build_prompt(board)             # 로봇(빨강) 차례 기준. LLM에 그대로 입력
col = parse_column(llm_answer, board)    # "column: 4" → 3 (col 0 = 로봇 왼쪽). 둘 수 없는 답이면 None
```

LLM 호출 자체는 들어 있지 않다. `scripts/play.py`의 `choose_column()`은 아직 랜덤이다.

## 실물과 맞춰야 하는 가정

- **보드 치수**는 상품 페이지(21×15×16 cm)에서 추정한 값. 디스크 지름 24 mm, 열 간격 28 mm, 슬롯 폭 7 mm로 두었으니 실측 후 `BoardConfig`를 고칠 것.
- **받침대 10 cm**: 관절 한계 안에서 그리퍼를 수직으로 내려 16 cm 보드 위 슬롯에 닿으려면 베이스를 8 cm 이상 올려야 함(`Layout.riser_height`).
- **관절 한계**: URDF에는 한계가 없어 robotis_mujoco_menagerie의 OMX 값(J2·J3 −120~90°, J4 ±100° 등)을 넣음.
- **피더**: 디스크를 세워서 한 장씩 잡을 수 있게 하는 홈. 실물에서도 같은 역할의 거치대가 필요함(눕힌 디스크는 이 그리퍼로 세워 넣을 수 없음).
- **흰 종이**: 보드의 로봇 쪽 면을 흰 종이로 막은 상태(`BoardConfig.backing_paper`). 빈 구멍이 흰색으로 보인다.
- 사람 수는 디스크를 슬롯 위로 순간이동시켜 떨어뜨리는 방식.
