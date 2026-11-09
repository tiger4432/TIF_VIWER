# 릴리즈 노트

설치·업데이트 방법은 README의 "릴리즈 번들로 설치·업데이트"를 보세요.

- **git checkout(git bundle로 업데이트):** `defaults\` 폴더는 없습니다. `plugins\results_api.py`도 업데이트(merge)로 새 버전이 되고, 현장에서 고친 내용은 `--autostash`가 그 위에 다시 얹습니다(같은 줄을 고쳤으면 git이 충돌로 알림). 아래 노트의 "`defaults\`에서 옮겨 붙이세요"는 zip 설치에만 해당합니다.
- **zip 설치:** 새 zip을 같은 폴더에 덮어 풀면 되고, 현장에서 고친 `config.json` / `products.json` / `deploy.json` / `plugins\` 는 바뀌지 않습니다(새 기본값은 zip 안의 `defaults\`).

## 1.1.8

- **Detect BBOX가 전부 같은 방향으로 밀리던 문제:** 선언 격자 원점이 bbox 크기의 약 25%보다 틀리면, 칩끼리는 잘 맞았지만 절대 위치를 찾지 못해 모든 bbox가 같은 양만큼 밀렸습니다. 이제 칩끼리 맞춘 뒤 한 피치 전체에서 기준 칩의 외곽(선언 크기만큼 떨어진 경계 한 쌍)을 찾아 전체를 옮깁니다. 합성 웨이퍼에서 원점이 70~140 px(피치 300) 틀려도 bbox 오차 최대 1.9 px(예전 86~140 px).
  - 상태 표시줄에 `chip outline over a whole pitch: shifted (dx, dy) px`로 얼마나 옮겼는지 나옵니다. 외곽을 못 찾으면 경고합니다: bbox 하나를 손으로 맞추고 **Refine BBOX**.
  - 반 피치를 넘게 틀리면 이웃 칸의 칩에 붙습니다(칸 번호가 하나 밀림): 원점을 고치세요.
  - 끄려면 config.json `"auto_bbox": {"locate": false}`.

## 1.1.7

**업데이트 후 할 일:** `requests`가 requirements에 추가됐습니다. `run.bat` / `dev.bat`이 자동으로 설치하며, exe를 쓰면 `build.bat`으로 다시 빌드하세요.

- **본딩맵 표 형식:** 칸마다 한 줄인 표(`base,x,y,leg,...`, CSV / 탭 / JSON 레코드)를 자동으로 격자로 바꿉니다. 웨이퍼 중심 기준 음수 좌표도 됩니다. 위아래가 뒤집히면 `"bonding_map": {"table": {"y_up": true}}`, 한 표에 맵이 여럿이면 `"table": {"where": {"base": "..."}}` (docs/RESULTS_API.md).
- **플러그인 HTTP를 requests로:** 기본 플러그인이 `requests`로 요청합니다. `params`(dict / list 값은 JSON 문자열), `headers`, `timeout`, `verify`. 요청마다 실제 URL이 콘솔에 찍히고, 실패하면 오류 창에도 URL이 나와 필터가 어떻게 나갔는지 바로 확인할 수 있습니다.
  - zip 설치: 현장 `plugins\results_api.py`는 업데이트로 바뀌지 않습니다. 새 `defaults\plugins\results_api.py`로 바꾸거나 필요한 함수를 옮겨 붙이세요.

## 1.1.6

- **git bundle로 업데이트(압축 풀기 없음):** 릴리즈마다 `TIF_Viewer-<버전>.bundle`(전체 이력)과 `TIF_Viewer-<버전>-since-<이전 버전>.bundle`(바로 앞 릴리즈 이후 변경분, 작음)이 함께 올라갑니다. git checkout에서 받은 파일로 업데이트합니다(네트워크 pull 없음):
  - `bundle_update.bat D:\down\TIF_Viewer-1.1.6.bundle`
  - 또는 직접: `git fetch D:\down\TIF_Viewer-1.1.6.bundle main` → `git merge --ff-only --autostash FETCH_HEAD`
  - 현장에서 고친 `config.json`, `plugins\results_api.py` 등은 `--autostash`로 잠시 치웠다가 다시 얹습니다. 겹치는 줄이 있으면 git이 알려 주고 `git stash list`에 남겨 둡니다.

## 1.1.5

- **Results API JSON 필터:** `params` 값이 dict / list이면 compact JSON으로 만들어 URI 인코딩해 넣습니다(JavaScript `encodeURIComponent`와 같음). 안쪽 문자열의 `{wafer_id}` 등도 채워집니다.
  - 예: `"bonding_map": {"url": "http://서버/api/map", "params": {"filters": {"waferId": "{wafer_id}", "layer": [1, 2]}}}` → `?filters=%7B%22waferId%22%3A%22W07%22%2C%22layer%22%3A%5B1%2C2%5D%7D`
  - zip 설치: 현장 `plugins\results_api.py`는 업데이트로 바뀌지 않습니다. 새 `defaults\plugins\results_api.py`의 `_request` / `_query` / `_fill_value` / `_fill`을 옮겨 붙이세요.
- `update.bat`: 다운로드 중 네트워크 오류나 5xx(예: 504)는 2·4·8초 뒤 다시 시도합니다.

## 1.1.4

- **Results API GET 필터:** GET 요청에 data(dict)로 넣은 필터는 본문으로 가서 서버가 무시했습니다. 기본 플러그인의 `_request`가 이제 GET에서 dict를 URL 쿼리스트링(`?wafer=...&type=...`)으로 붙입니다. config에서 `"bonding_map": {"url": ..., "params": {"wafer": "{wafer_id}", "type": "csv"}}`처럼 `params`로도 줄 수 있습니다(`{wafer_id}` 등은 채워지고 URL 인코딩됨).
  - zip 설치: 현장 `plugins\results_api.py`는 업데이트로 바뀌지 않습니다. 새 `defaults\plugins\results_api.py`의 `_request` / `_fill`을 옮겨 붙이거나, 필터를 URL에 직접 넣으세요: `"url": "http://.../map?wafer={wafer_id}"`.

## 1.1.3

- **`update.bat`:** git 없이 업데이트합니다. GitHub의 최신 릴리즈 zip을 받아 sha256을 확인하고 프로그램 폴더에 덮어 풉니다. `config.json` / `products.json` / `deploy.json` / `plugins\`, 데이터, `.venv`는 그대로입니다. 뷰어를 닫고 `update.bat` → `run.bat`.
  - `update.bat --check`: 설치된 버전과 최신 버전만 표시. `update.bat 1.1.2`: 그 버전으로(되돌리기).
  - 이번 한 번은 zip을 직접 받아 덮어 풀어야 합니다(이전 버전엔 `update.bat`이 없음). 다음부터는 `update.bat`.

## 1.1.2

- **본딩맵이 CSV로 올 때:** Excel에서 저장한 한글 CSV(cp949)도 읽고(예전엔 UTF-8만), 따옴표로 묶인 칸·세미콜론 구분도 처리합니다. 현장 플러그인의 `fetch_bonding_map`이 응답 bytes를 그대로 돌려줘도 프로그램이 디코딩합니다.
  - 기본 플러그인의 `fetch_bonding_map`도 cp949를 받도록 바뀌었지만, 현장 `plugins\results_api.py`는 업데이트로 바뀌지 않습니다. 현장 파일이 `body.decode(...)`에서 막히면 `return body`(bytes 그대로)로 바꾸면 됩니다.
- GitHub Releases에 릴리즈 zip이 올라갑니다(`v<버전>`, main에 새 버전이 올라갈 때 자동).

## 1.1.1

- **Results API `can't concat str to bytes`:** 플러그인이 `urlopen`에 dict나 list를 그대로 넘기면 나는 오류입니다(urlopen은 bytes만 받음). 기본 플러그인의 `_request`가 이제 dict / list는 JSON으로, str은 UTF-8로 바꿔 보냅니다.
  - **zip 설치: 현장 `plugins\results_api.py`는 업데이트로 바뀌지 않습니다.** 새 `defaults\plugins\results_api.py`의 `_request` 함수를 현장 파일에 옮겨 붙이거나, 직접 `json.dumps(d, ensure_ascii=False).encode("utf-8")`로 bytes를 만들어 넘기세요.

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
