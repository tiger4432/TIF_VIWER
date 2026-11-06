# 릴리즈 노트

설치·업데이트 방법은 README의 "릴리즈 번들로 설치·업데이트"를 보세요. 업데이트는 새 zip을 같은 폴더에 덮어 풀면 되고, 현장에서 고친 `config.json` / `products.json` / `deploy.json` / `plugins\` 는 바뀌지 않습니다.

## 1.1.0

**업데이트 후 할 일:** Layer change 탐지 방식이 여러 군데 바뀌었으므로, 예전에 Train한 Layer change 학습 필터는 Advanced → **Reset** 후 다시 Train하세요(이전 모델은 `.bak`으로 남음).

### 탐지 (Layer change)
- 기본 탐지 방법이 Layer change가 되었습니다(제품별 `detect_method`, config `void_detect.method`로 바꿀 수 있음).
- **큰 보이드:** 큰 필터 크기를 추가하고, 증거가 강한 영역으로 크기·위치를 잡아 큰 보이드가 한 덩어리로, 실제 크기에 가깝게 잡힙니다(600 px 칩 합성: 쪼개짐 33 → 2개/77).
- **울퉁불퉁·경계 흐린 보이드:** 튀어나온 조각과 주변 약한 검출을 하나로 합치고(학습 필터 뒤에 합침), 약한 조각이 상자를 부풀리지 않게 했습니다.
- **칩 가장자리:** 가장자리 찌꺼기가 필터를 타고 안쪽으로 번지던 것을 막았습니다(레이어가 정렬된 합성 데이터 미검 11.2 → 7.5 %).
- **SNR:** 검출마다 SNR(그 레이어 신호 / 배경 잡음, 손 마킹은 100)이 붙고, Void Detection의 **SNR ≥ 슬라이더**로 숨기기·저장 제외. voids.json `snr`, voids.csv `SNR`.
- **Mark on layer = First strong:** 보이드가 위 레이어에 희미하게 비칠 때, 처음 비친 곳 대신 확실히 보이는 첫 레이어에 마킹.
- **Shapes (debug):** 검출이 본 보이드 모양을 자홍색 윤곽으로 표시(Advanced). voids.json `outline`.
- 레이어 0에서도 보이는 보이드(기포 의심)에 "0" 배지, Layer-0 버튼으로 이동.

### 학습 필터
- **Train:** 검토 안 된(점선) 검출은 학습에서 빼고 개수를 알려 줍니다(맞는 검출을 "보이드 아님"으로 배우던 문제 수정). bbox를 뺀 모든 타입이 보이드로 학습됩니다.
- **Reset** 버튼: 지금 방법·제품의 학습 필터를 처음 상태로(파일은 `.bak`으로 보관).
- Evaluate 짧은 코드: Layer change + 학습 필터는 `k`.

### 화면·조작
- **Void Detection 패널:** Detect Voids + Thr, 검토 줄(◀ / Next Detected ▶ / 남은 수 / Layer-0), SNR 슬라이더, 나머지는 접히는 **Advanced**.
- **Auto BBOX + Grid:** 한 번에 Detect BBOX → Fit Grid → Detect BBOX. **Fit Grid가 원점도 바로잡습니다**(칸 중심 = bbox 중심; 예전 동작은 `auto_bbox.keep_offset`).
- **Clear Voids:** bbox는 두고 보이드만 모두 지우기(Ctrl+Z로 되돌림).
- 보이드 선이 굵고(2 px, 어두운 테두리) 축소해도 최소 크기로 보입니다(`ui_defaults` `void_line_width`, `void_min_px`).
- 오른쪽 패널 잘림·가로 스크롤 수정, 앵글 프리뷰 좌우 여백 제거, 제목의 `&` 표시 수정.
- `ui_defaults`: `origin_x` / `origin_y`(그리드 원점 기본값), `snr_min`.

### 저장·내보내기
- mask·merged 이미지에도 layout을 그리고, merged는 칩마다 bbox 중심에 맞춰 겹칩니다.

### 자동 실행 (Auto Run)
- 제품의 `detect_method`, `auto_run.snr_min`, 원점 기본값을 따릅니다.

### 기타
- 백그라운드 Detect Voids가 끝난 뒤 작업 객체가 남던 메모리 누수 수정.
- `dev.bat`은 `DEV_PULL=1`일 때만 git pull 합니다. git 없이 쓰는 곳은 릴리즈 zip + `run.bat`.
