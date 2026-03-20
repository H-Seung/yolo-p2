# Demo: YOLO Inference Visualization & Real-Time FPS Test

## 📁 구조

```
demo/
 ├── demo.py
 ├── realtime_fps_test.py
 └── README.md
```

---

# 🎯 목적

이 디렉토리는 다음 두 가지 목적을 위한 데모 코드로 구성되어 있다.

1. **영상 기반 탐지 결과 시각화 (demo.py)**
2. **실시간 환경 가정 추론 FPS 측정 (realtime_fps_test.py)**

본 데모는 **모델 간 성능을 직관적으로 비교하거나, 실시간 적용 가능성을 빠르게 확인**하는 것을 목표로 한다.

---

# 1️⃣ demo.py

## ✔ 목적

* 입력 영상에 대해 YOLO 추론을 수행하고
* bbox 및 통계 정보를 시각화하여
* **데모 영상 생성**

---

## ✔ 주요 기능

* bbox + label + confidence 표시
* bbox 크기 및 비율 정보 출력 (옵션)
* small object 자동 카운트
* 누적 detection / small 통계 표시
* FPS 표시 (전체 처리 기준)
* Scene 설명 텍스트 표시
* 결과 영상 저장

---

## ✔ 실행

```bash
python demo.py
```

---

## ✔ 입력 조건 (공정 비교 기준)

모델 간 비교 시 다음 조건을 동일하게 유지하는 것을 권장한다.

* 입력 해상도 (`imgsz`)
* confidence threshold
* 동일 입력 영상
* 동일 코드 및 처리 방식

---

## ✔ 주요 파라미터

```python
MODEL_PATH = "../models/xxx.pt"
VIDEO_PATH = "input_video.mkv"
IMG_SIZE = 960
CONF = 0.3
SMALL_THRESH = 0.0004
```

---

## ✔ Small object 기준

### 정의

```
bbox 면적 / 전체 프레임 면적 < SMALL_THRESH
```

### 특징

* COCO의 `32x32` 기준 대신 **비율 기반 기준 사용**
* 영상 해상도 및 장면 크기에 따라 유연하게 적용 가능

---

## ✔ FPS 해석

영상에 표시되는 FPS는 다음을 포함한 값이다.

* 모델 추론
* bbox 렌더링
* 텍스트 오버레이

즉, **순수 inference FPS가 아닌 데모 기준 처리 FPS**이다.

---

## ✔ 특징

* 모든 프레임을 순차적으로 처리 (offline 방식)
* 프레임 손실 없음
* 결과 영상 길이는 원본과 동일

---

## ✔ 사용 용도

* 모델 간 탐지 결과 비교
* small object 탐지 성능 비교
* 데모 영상 제작

---

# 2️⃣ realtime_fps_test.py

## ✔ 목적

* 실시간 시스템을 가정하여
* **비동기 구조에서의 추론 FPS 측정**

---

## ✔ 핵심 개념

```
Capture Thread → 최신 frame 계속 갱신
Inference Loop → 최신 frame만 추론
```

👉 중간 프레임은 처리되지 않고 drop됨

---

## ✔ 실행

```bash
python realtime_fps_test.py
```

---

## ✔ 출력

* 콘솔 출력
* `.log` 파일 저장 (append 방식)

```
[INFO] inferred_frames=120, avg_infer_fps=60.5
```

---

## ✔ 로그 특징

* 실행(run) 단위로 구분
* 이전 결과 유지 (append)

---

## ✔ FPS 의미

> 최신 프레임 기반 비동기 추론 구조에서의 평균 inference FPS


### ⚠️ 주의사항


* 실제 카메라 입력 FPS ❌
* 전체 시스템 FPS ❌
* 화면 표시 FPS ❌

---

## ✔ 특징

* 영상 저장 없음
* 화면 출력 없음
* bbox 렌더링 없음

👉 **순수 추론 성능에 가까운 측정**

---

# 3️⃣ 두 코드의 차이

| 항목     | demo.py      | realtime_fps_test.py |
| ------ | ------------ | -------------------- |
| 목적     | 영상 생성        | FPS 측정               |
| 처리 방식  | 순차 (offline) | 비동기 (latest frame)   |
| 프레임 손실 | 없음           | 있음                   |
| 출력     | 영상 파일        | 로그                   |
| FPS 의미 | 처리 FPS       | inference FPS        |

---

# 4️⃣ 권장 사용 방법

1. `demo.py` → 탐지 품질 및 시각적 비교 (“이 모델이 얼마나 잘 잡냐”)
2. `realtime_fps_test.py` → 실시간 처리 가능성 확인 (“이 모델이 실시간으로 돌아가냐”)

👉 두 결과를 함께 해석해야 모델 선택에 의미 있음

