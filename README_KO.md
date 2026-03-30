# ReflectAI

**귀납적 반성을 통한 신경-기호 추론** — ABL-Refl (AAAI 2026 우수논문상) 최초 오픈소스 구현

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-153%20passed-brightgreen.svg)]()

> **논문**: *"Efficient Rectification of Neuro-Symbolic Reasoning Inconsistencies by Abductive Reflection"* (AAAI 2026 우수논문상)

## ReflectAI란?

ReflectAI는 **신경망(Neural Network)**과 **논리적 추론(Logical Reasoning)**을 결합하여 제약 조건 만족 문제를 푸는 프레임워크입니다. 핵심 혁신은 **반성 헤드(Reflection Head)** — 신경망 예측 중 어떤 것이 틀릴 가능성이 높은지 감지하는 이진 오류 탐지기로, 전수 탐색 대신 **표적 수정**을 가능하게 합니다.

### 아키텍처

```
입력 (예: 스도쿠 숫자 이미지)
  |
  v
[Body Block] ── 공유 특징 추출기 (CNN/MLP)
  |         \
  v          v
[출력 헤드]   [반성 헤드]    <-- 핵심 혁신
  |               |
  v               v
예측값        오류 플래그 (이진: 1=의심, 0=신뢰)
  |               |
  +-------+-------+
          |
          v
[귀납적 추론 솔버] ── 플래그된 위치만 수정
          |
          v
   수정된 레이블 (제약 조건 만족)
```

### 왜 ReflectAI인가?

| 특성 | 기존 ABL | ReflectAI (ABL-Refl) |
|------|---------|---------------------|
| 오류 탐지 | 모든 위치 탐색 | 반성 헤드가 의심 위치 특정 |
| 속도 | 기준선 | **10,000-15,000배 빠름** |
| 데이터 효율성 | 20,000 라벨 필요 | **2,000 라벨** (10배 적음) |
| 스도쿠 정확도 | 76.5% | **97.4%** |
| 오류 탐지 재현율 | - | **99.04%** |

## 설치

```bash
# 저장소 클론
git clone https://github.com/wjddusrb03/reflectai.git
cd reflectai

# 기본 설치
pip install -e .

# Z3 SMT 솔버 포함 (복잡한 제약 조건에 권장)
pip install -e ".[z3]"

# 웹 UI 포함 (Gradio)
pip install -e ".[web]"

# 모든 선택적 백엔드 + 웹 UI 포함
pip install -e ".[all]"

# 개발용 (테스트 포함)
pip install -e ".[dev]"
```

## 빠른 시작

### 웹 UI (추천!)

```bash
pip install -e ".[web]"    # Gradio 설치
reflectai web              # http://localhost:7860 에서 열기
```

![Web UI](https://img.shields.io/badge/Web_UI-Gradio-orange.svg)

웹 UI에서 제공하는 기능:
- 색상 코딩된 스도쿠 인터랙티브 풀이
- 숫자 덧셈 태스크 시각화
- 벤치마크 실행기
- 단계별 파이프라인 설명

### CLI 데모

```bash
# 인터랙티브 데모 실행
reflectai demo --difficulty medium --verbose

# 여러 퍼즐로 벤치마크
reflectai benchmark --num-puzzles 20 --difficulty hard
```

### Python API

```python
import numpy as np
from reflectai.knowledge import build_sudoku_kb
from reflectai.tasks.sudoku import generate_sudoku, simulate_noisy_predictions
from reflectai.pipeline import solve_from_predictions

# 스도쿠 퍼즐 생성
puzzle, solution = generate_sudoku("medium", seed=42)

# 노이즈가 있는 신경망 예측 시뮬레이션 (15% 오류율)
prediction = simulate_noisy_predictions(solution, error_rate=0.15, seed=42)

# 지식 기반 구축 (27개 제약 조건: 행 + 열 + 박스)
kb = build_sudoku_kb()

# 반성 점수 시뮬레이션 (실제로는 반성 헤드가 학습)
errors = prediction.labels != solution
reflection_scores = np.where(errors, 0.8, 0.1)

# 플래그된 위치에 대해 귀납적 추론 실행
result = solve_from_predictions(
    prediction.labels,
    prediction.probabilities,
    reflection_scores,
    kb,
    threshold=0.5,
    solver_type="backtrack",
)

print(f"예측 정확도: {(prediction.labels == solution).mean():.1%}")
print(f"수정 정확도: {(result.final_labels == solution).mean():.1%}")
print(f"일관성 충족: {result.is_consistent}")
```

### ReflectAI 모델 학습

```python
import torch
from reflectai.perception import MLPBody, PerceptionModule
from reflectai.reflection import ReflectionHead
from reflectai.trainer import ReflectAIModel, Trainer
from reflectai.models import TrainConfig, KnowledgeBase

# 모델 구축
body = MLPBody(input_dim=784, hidden_dim=128)
perception = PerceptionModule(body, num_classes=10)
reflection_head = ReflectionHead(hidden_dim=128, num_classes=10)
model = ReflectAIModel(perception, reflection_head)

# 3개 손실 함수 기반 학습 설정
config = TrainConfig(
    epochs=50,
    learning_rate=1e-3,
    lambda_consistency=1.0,       # 일관성 손실 가중치
    lambda_reflection_size=0.1,   # 반성 크기 정규화 가중치
    reflection_target_rate=0.2,   # 목표 ~20% 플래깅 비율 (C=0.8)
)

# 학습 실행
kb = KnowledgeBase(num_classes=10)
trainer = Trainer(model, kb, config)
history = trainer.train(train_loader, callback=lambda s: print(s.to_dict()))
```

## 구성 요소

### 핵심 모듈

| 모듈 | 설명 |
|------|------|
| `perception.py` | 신경망 본체 (CNN/MLP) + 출력 헤드 |
| `reflection.py` | 반성 헤드 — 이진 오류 탐지기 (핵심 혁신) |
| `reasoner.py` | 귀납적 솔버 (백트래킹, Z3) |
| `knowledge.py` | 지식 기반 빌더 |
| `trainer.py` | 3개 손실 함수 학습 루프 |
| `pipeline.py` | 엔드투엔드 추론 파이프라인 |
| `cli.py` | 명령줄 인터페이스 |

### 내장 태스크

| 태스크 | 제약 조건 | 설명 |
|--------|----------|------|
| 스도쿠 (9x9) | 27 all_distinct | 행, 열, 박스 유일성 |
| 스도쿠 (4x4) | 12 all_distinct | 미니 스도쿠 (테스트용) |
| MNIST 덧셈 | 1 sum_equals | 두 숫자의 합이 목표값 |
| 방정식 | 1 in_range | 손글씨 방정식 인식 |
| N-Queens | 1 all_distinct | N x N 보드 퀸 배치 |

### 3개 손실 함수 학습

```
L_total = L_supervised + λ_c × L_consistency + λ_r × L_reflection_size
```

1. **L_supervised**: 라벨 데이터에 대한 표준 교차 엔트로피
2. **L_consistency**: 수정이 제약 조건 만족을 개선할 때 REINFORCE 보상
3. **L_reflection_size**: ~20% 플래깅 비율 유지 정규화 (사소한 해 방지)

## 핵심 개념

### 반성 벡터 (Reflection Vector)

반성 벡터 `r`은 모든 예측 위치에 대한 이진 마스크입니다:
- `r[i] = 1`: 위치 `i`는 잠재적 오류로 플래그
- `r[i] = 0`: 위치 `i`는 신뢰

플래그된 위치만 제약 조건 솔버로 전달되어 탐색 공간이 극적으로 감소합니다.

### 귀납적 추론 (Abductive Reasoning)

예측 `p`와 반성 플래그 `r`이 주어지면, 수정된 레이블 `c`를 찾습니다:
1. 신뢰하는 위치(`r[i] = 0`)에서는 `c[i] = p[i]` 유지
2. 지식 기반의 모든 제약 조건 만족
3. 플래그된 위치에서는 신경망 확률이 높은 값 우선

### 왜 "귀납적(Abductive)"인가?

귀납 = 관찰에서 최선의 설명으로 추론. 솔버는 플래그된 예측에 대한 최소한의 수정을 통해 제약 조건 위반을 "설명"합니다.

## 실용적 활용

- **퍼즐 풀기**: 스도쿠, N-Queens 등 제약 조건 만족 퍼즐
- **숫자 인식 검증**: MNIST 등 숫자 인식 결과의 논리적 일관성 검증
- **방정식 인식**: 손글씨 수학 방정식의 산술 일관성 확인
- **학습 효율성**: 적은 라벨 데이터로 높은 성능 달성 (10배 데이터 효율)

## 요구 사항

- Python >= 3.10
- PyTorch >= 2.0
- NumPy >= 1.24

선택 사항:
- z3-solver >= 4.12 (Z3 백엔드)
- python-sat >= 1.8 (SAT 백엔드)
- torchvision >= 0.15 (이미지 태스크)

## 인용

```bibtex
@inproceedings{abl-refl-2026,
  title={Efficient Rectification of Neuro-Symbolic Reasoning Inconsistencies by Abductive Reflection},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2026},
  note={Outstanding Paper Award}
}
```

## 라이선스

MIT License. [LICENSE](LICENSE) 파일을 참조하세요.
