# PatchCore 검토 및 운영 셋업

Detect Voids의 선택 방식으로 PatchCore(CNN 특징 기반 이상 탐지)를 넣었습니다. 이 문서는 검토 결과, 이 프로젝트에서의 구현, 운영 PC에 설치·빌드·설정하는 방법을 정리합니다.

## 0. 결론

- **기본은 지금의 Die-to-die를 유지합니다.** 이 데이터(정렬된 같은 칩의 반복)에서는 Die-to-die가 검출률, 오검출, 경계 정확도, 속도 모두 앞섭니다.
- PatchCore는 **선택 기능**으로 넣었습니다. 실데이터에서 Die-to-die가 잘 안 되는 경우(예: 칩마다 질감·노이즈 양상이 크게 다름)에 비교해 볼 수 있습니다.
- 가벼움을 위해 저장소에는 **MobileNetV2 모델(2.4 MB)만** 넣었습니다. PatchCore 안에서는 ResNet50(34 MB)이 더 정확하지만 2배 이상 느립니다. 필요하면 2.6절로 만들 수 있습니다.
- 모두 **CPU 전용**, onnxruntime만 추가(선택)합니다. PyTorch는 쓰지 않습니다.

## 검토 결과

합성 웨이퍼(14×14칩, 4레이어, 보이드 47개, 밝은/어두운 보이드, 복잡한 반복 layout)로 비교했습니다. "심한 변화"는 칩마다 밝기·대비·감마·칩 안 기울기를 크게 다르게 만든 데이터입니다. 경계 오차는 찾은 보이드의 반지름 오차 중앙값입니다. 시간은 4코어 CPU에서 칩·레이어 하나당 시간입니다(칩 잘라내기 포함).

| 방법 | 보통: 검출률 / 오검출 / 경계 오차 | 심한 변화: 검출률 / 오검출 / 경계 오차 | 시간 | 모델 |
|---|---|---|---|---|
| **Die-to-die (기본)** | **100% / 1 / 1.0 px** | **98% / 5 / 0.8 px** | **~11 ms** | 없음 |
| PatchCore MobileNetV2, Thr 8 | 100% / 58 / 9.9 px | 83% / 472 / 4.6 px | ~66 ms | 2.4 MB |
| PatchCore MobileNetV2, Thr 12 | 83% / 4 / 5.6 px | 45% / 114 / 4.4 px | ~66 ms | 2.4 MB |
| PatchCore ResNet50, Thr 8 | 98% / 1 / 14.9 px | 87% / 18 / 7.3 px | ~150 ms | 34 MB |
| PatchCore ResNet50, Thr 12 | 91% / 1 / 7.6 px | 51% / 0 / 5.1 px | ~150 ms | 34 MB |
| PatchCore + 밝기 맞춤 먼저(MobileNetV2, Thr 8) | 72% / 20 | 79% / 400 | ~80 ms | 2.4 MB |
| PatchCore + 밝기 맞춤 먼저(ResNet50, Thr 8) | 79% / 1 | 72% / 5 | ~165 ms | 34 MB |

- 처음 나타난 레이어에 마킹하는 동작은 모든 방법에서 거의 맞았습니다.
- **결함이 없는 영상**: Die-to-die는 0개, PatchCore는 149개를 잡았습니다. PatchCore 점수는 레이어 안에서 상대적으로 정규화되므로, 운영 전 임계값 보정이 꼭 필요합니다(2.7절).
- 원 논문 방식(위치 무관 메모리 뱅크, ResNet50)도 시험했습니다: 보통 데이터 Thr 10에서 검출 100%, 오검출 29개, 경계 오차 7.8 px로 위치 기반보다 나빴습니다.
- **841칩 × 12레이어** 기준 예상 시간(4코어 CPU): Die-to-die 약 3분, PatchCore MobileNetV2 약 11분, ResNet50 약 26분.

### 왜 이 데이터에서 Die-to-die가 유리한가

- 칩이 정렬되어 있고 구조가 같으므로, 같은 위치의 화소끼리 비교하는 것이 가장 강한 기준입니다. PatchCore는 그 정보를 1/8 해상도 특징으로만 씁니다.
- CNN 특징 격자는 입력/8(224 → 28×28)이라 경계가 수 px~15 px 어긋납니다. 보이드 타원을 그려야 하는 이 프로그램에는 불리합니다.
- ImageNet 특징은 밝기·대비·감마 변화에 민감합니다. Die-to-die는 이것을 칩마다 명시적으로 맞춥니다.
- 단, 이 결과는 **합성 데이터** 기준입니다. 실데이터에서 패턴이 질감처럼 칩마다 다르게 보인다면 결과가 달라질 수 있어, 비교할 수 있도록 PatchCore를 선택 기능으로 남겼습니다.


## 1. 구현 요약

| 항목 | 내용 |
|---|---|
| 백본 | ImageNet 사전학습 **MobileNetV2**(기본, `models/patchcore_mobilenetv2.onnx`, 2.4 MB). ResNet50(34 MB)은 비교용으로만 변환 가능 |
| 실행 | **CPU 전용**, onnxruntime. PyTorch·TensorFlow 없음 |
| 가중치 | 저장소 `models/`에 ONNX로 포함. **실행 중 다운로드 없음** |
| 특징 | 중간층 2개(stride 8, 16)를 합치고 3×3 평균 → 128차원 랜덤 투영 |
| 비교 방식 | **위치 기반 최근접 이웃**: 칩의 각 특징 칸을 다른 칩들의 같은 위치(±1칸) 특징과 비교합니다. 원 논문의 위치 무관 메모리 뱅크보다 정렬된 반복 칩에 맞고, 웨이퍼 안의 칩들로 매번 "정상"을 정하므로 별도 학습이 없습니다 |
| 점수 | 레이어마다 중앙값/MAD로 정규화, `pc_threshold`(기본 10)를 넘는 곳 |
| 후처리 | Die-to-die와 같음: 덩어리 → 타원, 레이어 0부터 **처음 나타난 레이어에 한 번만**, 점선 표시, Next Detected, Ctrl+Z 한 번 |

코드: `sat_widgets/patchcore.py`(점수), `sat_widgets/void_detect.py`(공통 흐름), `tools/build_patchcore_onnx.py`(모델 변환).

## 2. 운영 셋업

### 2.1 무엇이 필요한가

| 구성 | 위치 | 비고 |
|---|---|---|
| 모델 | `models/patchcore_mobilenetv2.onnx` | 저장소에 포함(2.4 MB) |
| 런타임 | `onnxruntime==1.20.1` (`requirements-patchcore.txt`) | CPU 전용 휠, Windows x64 / Python 3.10 |
| 프로그램 | 이 저장소 | onnxruntime이 없거나 모델 파일이 없으면 PatchCore 항목만 비활성화되고 나머지는 그대로 동작 |

### 2.2 배포용 exe 빌드 (권장: 운영 PC에는 exe만 배포)

인터넷이 되는 빌드 PC에서:

```bat
set WITH_PATCHCORE=1
build.bat
```

- `.venv-build`에 onnxruntime을 추가로 설치하고, exe에 포함합니다.
- `models\` 폴더를 `dist\TIF_Viewer\models\`로 복사합니다.
- 운영 PC는 `dist\TIF_Viewer` 폴더만 복사하면 됩니다. Python, pip, 인터넷이 필요 없습니다.
- `WITH_PATCHCORE` 없이 빌드하면 PatchCore 없는 가벼운 exe가 됩니다. 단, 이전에 같은 `.venv-build`에 onnxruntime을 설치했다면 포함될 수 있으니, 확실히 빼려면 `.venv-build` 폴더를 지우고 빌드합니다.

### 2.3 인터넷이 없는 PC에서 소스로 실행할 때

1. 인터넷 PC에서 휠을 받아 둡니다.
   ```bat
   pip download -r requirements-patchcore.txt -d wheels --only-binary=:all: --platform win_amd64 --python-version 3.10
   ```
2. `wheels` 폴더를 사내망 PC로 복사해 설치합니다.
   ```bat
   pip install --no-index --find-links wheels -r requirements-patchcore.txt
   ```
3. dev.bat 사용 시: `set WITH_PATCHCORE=1` 후 `dev.bat` (단, dev.bat의 venv 설치도 인터넷이 필요합니다. 오프라인이면 1–2번처럼 휠을 미리 준비합니다).

### 2.4 사용

1. 그리드·본딩맵을 맞추고 **Detect BBOX**로 기준 bbox를 그립니다(칩 정렬 기준).
2. Void Marking에서 마킹할 타입을 고릅니다.
3. Detect Voids 옆 방법 상자에서 **PatchCore**를 고릅니다. 비활성이면 마우스를 올려 이유(onnxruntime 없음 / 모델 없음)를 확인합니다.
4. Thr(PatchCore 기본 10)를 정하고 **Detect Voids**를 누릅니다. 방법마다 Thr 값을 따로 기억합니다.
5. 결과는 점선입니다. **Next Detected**로 확인하고, 수정(E)하면 실선이 됩니다. 전체 결과는 Ctrl+Z 한 번으로 되돌립니다.

### 2.5 설정 (`config.json` → `"void_detect"`)

```json
"void_detect": {
    "method": "patchcore",
    "pc_threshold": 10,
    "pc_model": "models/patchcore_mobilenetv2.onnx",
    "pc_input": 224,
    "pc_dims": 128,
    "pc_bank_chips": 64
}
```

| 키 | 기본값 | 의미 |
|---|---|---|
| `method` | `"die_to_die"` | 시작할 때 고를 방법 |
| `pc_threshold` | 10 | 레이어 중앙값에서 MAD 몇 배 이상을 이상으로 볼지. 낮을수록 민감 |
| `pc_model` | `models/patchcore_mobilenetv2.onnx` | 프로그램 폴더 기준 상대 경로 또는 절대 경로 |
| `pc_input` | 224 | CNN 입력 크기. 특징 격자는 입력/8(224 → 28×28). 키우면 위치가 정밀해지지만 시간은 제곱으로 늘어납니다 |
| `pc_dims` | 128 | 특징 투영 차원 |
| `pc_bank_chips` | 64 | 각 위치에서 비교할 다른 칩 수. 늘리면 정확도↑, 시간↑ |
| `pc_match` | false | CNN에 넣기 전에 Die-to-die와 같은 밝기·대비·기울기 맞춤을 할지. 합성 데이터에서는 오히려 나빠져 기본은 끔 |

공통 키(`min_size_px`, `max_area_frac`, `work_size`)는 두 방법에 모두 적용됩니다.

### 2.6 모델을 다시 만들 때 (보통 필요 없음)

인터넷 PC에서:

```bat
pip install numpy h5py onnx
curl -O https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5
python tools\build_patchcore_onnx.py mobilenet_v2 mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5 models\patchcore_mobilenetv2.onnx
```

ResNet50: `resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5`(같은 경로의 `keras-applications/resnet/`)와 `resnet50` 인자. 만든 `.onnx`를 `models/`에 커밋하면 운영 PC는 다운로드가 필요 없습니다.

### 2.7 실데이터 임계값 정하기

PatchCore 점수는 레이어 안에서 상대적으로 정규화되므로, 결함이 없는 웨이퍼에서도 분포의 꼬리를 잡을 수 있습니다(무결함 테스트 영상에서 실제로 오검출이 나왔습니다). 운영 전에:

1. 보이드가 확인된 웨이퍼 1–2장과 결함이 거의 없는 웨이퍼 1장을 준비합니다.
2. 결함 없는 웨이퍼에서 오검출이 거의 없어지는 가장 낮은 Thr를 찾습니다.
3. 그 Thr로 보이드 웨이퍼에서 놓치는 보이드가 없는지 확인합니다.
4. 정한 값을 `config.json`의 `pc_threshold`에 적어 배포합니다.

## 3. 라이선스

- MobileNetV2 가중치: Keras Applications 배포본(Google의 TensorFlow 공개 모델, Apache License 2.0로 공개).
- ResNet50 가중치: Keras Applications 배포본(원본은 Kaiming He의 ResNet 모델, MIT License).
- `models/*.onnx`는 위 가중치를 변환(배치 정규화 접기)한 것으로 원 라이선스를 따릅니다. 사내 배포 규정이 있으면 배포 전에 확인하세요.
