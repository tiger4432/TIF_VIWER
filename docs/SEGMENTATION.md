# U-Net 세그멘테이션 (Detect Voids)

칩마다 레이아웃의 대비·흐림이 달라 golden과의 차이만으로는 보이드와 레이아웃을 가를 수 없을 때 씁니다.
손 마킹으로 작은 U-Net을 학습해, 칩마다 "보이드일 확률" 지도를 만들고 그 지도를 편차 지도 대신 씁니다.
그 뒤 단계(덩어리, 크기, 레이어 규칙, 타입 규칙, Evaluate)는 그대로입니다.

- 학습: PyTorch가 있는 PC에서 `tools/train_seg.py` (CPU로 수십 분~몇 시간)
- 실행: 현장 PC는 onnxruntime만 있으면 됩니다(PatchCore와 같음, `requirements-patchcore.txt`)
- 모델: `models/void_unet.onnx` (+ `void_unet.json`), 복원 모드는 `models/void_recon.onnx`

## 1. 학습 데이터 만들기 (Seg Data 창)

Void Marking 패널의 **Seg Data**.

| 버튼 | 하는 일 |
|---|---|
| Add Open File | 지금 연 파일과 그 마킹으로 샘플을 만듭니다 |
| Add Folder... | 폴더 아래의 저장 결과(`voids.json` + `meta.json`)를 모두 처리합니다. meta.json의 TIF 경로·그리드·레이어를 쓰고, 옆에 `bonding_map.txt`가 있으면 그 칸만. TIF가 옮겨졌으면 결과 폴더 안의 같은 이름 TIF를 찾습니다. meta.json이 없는 예전 결과는 파일을 열어 Load JSON 후 Add Open File로 |
| Remove Source | 선택한 파일의 샘플 삭제 |
| Open Data Folder | 데이터 폴더 열기 (`사용자 폴더\TIF_Viewer_seg\`) |

- 샘플 = 칩 하나 × 레이어 하나. Detect Voids와 똑같이 잘라 맞춘 칩(칩 위치 맞추기, 밝기·레이아웃 맞춤 후)과 그 레이어의 golden·spread를 저장합니다.
- 마스크(정답): 이 레이어의 **손 마킹 안쪽 = 보이드**, 마킹 윤곽 주변 띠 = **모름**(마킹은 윤곽의 근사), **다른 레이어의 마킹 자리**와 **검토 안 한 자동 마킹** = 모름, 나머지 = 배경. 모름은 학습에서 빠집니다.
- 썸네일: 초록 = 보이드, 노랑 = 모름.

### 리뷰

- 왼쪽: 파일 목록. 체크를 끄면 그 파일 전체를 학습에서 뺍니다.
- 오른쪽: 칩 썸네일. **클릭하면 넣기/빼기**(뺀 칩은 빨간 X). Show로 보이드 있는 칩 / 없는 칩 / 전체를 고르고, Keep shown / Leave out shown으로 보이는 칩을 한꺼번에.
- 이런 칩은 빼세요: 마킹이 보이드와 어긋난 칩(예전 각도 버그 파일 등), 보이드인데 마킹이 빠진 칩(배경으로 학습되어 버림), 칩이 깨지거나 정렬이 크게 틀린 칩.
- 선택은 바로 저장됩니다(각 파일 폴더의 `index.json`).

## 2. 학습 (`tools/train_seg.py`)

```bat
cd TIF_VIWER
python tools\train_seg.py
python tools\train_seg.py --epochs 60 --steps 100 --out models\void_unet.onnx
python tools\train_seg.py --data D:\seg_data --threads 8
```

필요: `pip install torch onnx` (numpy, Pillow는 이미 있음). 데이터는 기본 `사용자 폴더\TIF_Viewer_seg\`. 리뷰에서 넣은 칩만 씁니다.

| 옵션 | 기본 | 뜻 |
|---|---|---|
| `--mode` | seg | `seg`: 보이드 픽셀 표시. `recon`: 보이드를 지운 칩을 다시 그림(칩별 golden) |
| `--epochs`, `--steps` | 40, 100 | 에폭 수, 에폭당 배치 수 |
| `--batch`, `--crop` | 16, 128 | 배치 크기, 학습 잘라내기 크기(작업 픽셀) |
| `--base` | 16 | U-Net 기본 채널(16 → 약 0.5M 파라미터, 24 → 약 1M) |
| `--val` | 0.15 | 검증용으로 빼는 칩 비율(같은 칩의 레이어는 같은 쪽) |
| `--synth` | 0.5 | 가짜 보이드를 붙일 확률 |

학습 방식(결함 분할 문헌: DRAEM, FC-EF 변화 검출, "simulated reference" die-to-die 검사 참고):

- 입력 3채널: 칩, golden 칩, (칩 − golden) / spread. 앱이 같은 방식으로 만듭니다(`void_seg.inputs`).
- 칩과 golden에 **서로 다른** 대비·감마·흐림·±1.5 px 이동을 줍니다. 칩마다 다른 레이아웃 모양이 "정상"이라는 것을 배웁니다.
- 흐릿한 가짜 보이드(타원·초타원, 부드러운 경계, 밝게/어둡게, 칩 잡음의 0.3~3배, 흐린 것도 자주)를 칩에 붙입니다. 실제 마킹이 적어도 학습이 됩니다.
- 손실: Focal + Dice, 모름 픽셀 제외. 회전·뒤집기 증강.
- 에폭마다 검증 칩으로 보이드 재현율·정밀도·픽셀 F1을 출력하고, 보이드 F1(재현율·정밀도의 조화평균)이 가장 좋은 에폭을 ONNX로 내보냅니다. 결과는 `.json`에 남습니다.

## 3. 사용

Void Marking의 방법 상자에서 **Segmentation U-Net**(또는 **Reconstruction U-Net**)을 고르고 Detect Voids / Evaluate.
모델 파일이 없거나 onnxruntime이 없으면 항목이 꺼져 있고, 마우스를 올리면 이유가 나옵니다.

- Segmentation: Thr = 확률 × 10 (기본 5 = 50 %). 낮추면 더 많이 잡습니다.
- Reconstruction: 모델이 그린 "보이드 없는 칩"과의 차이를 칩끼리 비교합니다(Die-to-die와 같은 Thr 의미).
- Evaluate 코드: 방법이 `sg`(짧은 코드 `s`) / `rc`(`r`).

## 4. 팁

- 처음엔 마킹 파일 여러 개를 Add Folder로 넣고, 보이드 있는 칩 위주로 훑어보며 이상한 칩만 빼면 됩니다.
- 학습에 쓰지 않은 파일로 Evaluate 해서 Die-to-die와 비교하세요.
- 제품(레이아웃)이 많이 다르면 제품별로 데이터 폴더(`--data`)와 모델(`--out`)을 나누는 편이 낫습니다.
