# 릴리즈 노트

설치·업데이트 방법은 README의 "릴리즈 번들로 설치·업데이트"를 보세요.

- **git checkout(git bundle로 업데이트):** `defaults\` 폴더는 없습니다. `plugins\results_api.py`도 업데이트(merge)로 새 버전이 되고, 현장에서 고친 내용은 `--autostash`가 그 위에 다시 얹습니다(같은 줄을 고쳤으면 git이 충돌로 알림). 아래 노트의 "`defaults\`에서 옮겨 붙이세요"는 zip 설치에만 해당합니다.
- **zip 설치:** 새 zip을 같은 폴더에 덮어 풀면 되고, 현장에서 고친 `config.json` / `products.json` / `deploy.json` / `plugins\` 는 바뀌지 않습니다(새 기본값은 zip 안의 `defaults\`).

## 1.1.34

- **Edge Stains = 칩 경계부터 어두운 띠까지 칠한 영역:** 영역을 칩 경계에서 어두운 띠(가장 먼 어두운 점)까지 채우고(끊긴 구간은 이어 채움), 화면에 반투명하게 칠해 보여 줍니다. 박스는 그 영역의 사각형.
- **칩 경계에 붙은 것만:** 영역이 칩 경계에 닿아 있어야(Touch, Contact) 오염으로 인정 → 칩 안쪽의 엉뚱한 곳을 잡지 않습니다.
- **모든 설정을 입력칸으로:** Bright, Density, Aspect, Touch, Contact, Skip, Box, Mark as(타입) 추가.

## 1.1.33

- **Edge Stains 분할 → 박스:** 어두운 점(경계)과 안쪽 밝은 부분을 분할해 Merge r 안이면 이어 붙이고, 그 영역의 사각형을 박스로(변에서 경계까지 전체). 변마다 따로 처리해 모서리 부근도 잡습니다. 합성: 찾음 97 → 99%, 과검 5 → 0, 박스가 오염 길이의 ~99%.
- **Edge Stains 별도 그룹 + 입력칸:** Length ≥, Merge r, Depth %, Thr, Dark.
- **A 키:** 영상 위에서 A → 그 칩의 감지된(점선) 마킹 일괄 리뷰 완료.

## 1.1.32

- **Edge Stains 버튼(가장자리 오염):** 칩 안쪽 가장자리에 붙어 길게 생기는 오염(경계는 점점이 어둡고 찌글찌글, 안쪽은 밝음)을 칩끼리 비교로 찾아 점선 사각형으로 마킹(선택한 타입). 합성 단일 레이어 웨이퍼에서 97% 찾음, 과검 웨이퍼당 0~5개. 칩끼리 비슷해야 합니다. 설정 config `void_detect.edge_band`(README "가장자리 오염").

## 1.1.31

- **코너 붙이기는 첫 클릭 기준:** 코너에 붙는 타입을 그릴 때, 붙을 bbox 코너를 드래그한 영역이 아니라 처음 누른 점에 가장 가까운 코너로 정합니다(드래그를 어느 쪽으로 해도 그 코너에서 놓은 점까지).

## 1.1.30

- **1.1.29는 쓰지 마세요:** 시작할 때 오류로 프로그램이 뜨지 않습니다(새 SSIM 방법이 PatchCore 모델로 검사됨). 1.1.30에서 고쳤고 내용은 1.1.29와 같습니다.

## 1.1.29

- **Detect Voids 방법 "SSIM die-to-die"(실험):** 기준 칩과 국소 구조(SSIM) 비교, 칩 정렬 + ±2 px 탐색으로 어긋남에 강건. 단일 레이어 이미지에도. 합성 데이터에서는 die-to-die보다 낫다는 결과가 없어(둘 다 Layer change에 크게 못 미침) 실제 데이터에서 Evaluate(코드 `m`)로 비교가 필요합니다.

## 1.1.28

- **Ctrl+C로 칩 이미지 복사:** 영상 위에서 Ctrl+C → 마우스 아래 칩(영상 밖이면 화면 가운데 칩)을 마킹·layout·제목줄이 들어간 overlay 이미지로 클립보드에. 붙여넣기만 하면 됩니다.

## 1.1.27

- **Measure (Scale 바로 재기):** Scale 칸 옆 Measure → 영상에서 길이를 아는 두 점 클릭 → 실제 길이(um) 입력 → Scale = 길이 / 픽셀 거리. 측정 중 선과 px 거리 표시, Esc로 취소.

## 1.1.26

- **Lock BBOX**(기본 켜짐): bbox가 마우스로 잡히지 않아, 최상위 레이어에서 칩 안 마킹의 수정·타입(bin) 변경·삭제를 할 때 bbox가 대신 선택되지 않습니다. bbox를 손으로 옮길 때만 끄세요. 버튼(Detect / Grid / Refine BBOX)은 잠가도 동작.

## 1.1.25

- **앵글 프리뷰 기본 위치를 네 사분면으로:** Center → 3사분면, Top → 2사분면, Right → 4사분면, Top-Right → 1사분면(각 0.7R). **config.json에 `angle_preview.points`가 있으면 그 값이 우선**하니, 바꾸려면 그 부분을 지우거나 README의 예로 바꾸세요.

## 1.1.24

- **Grid BBOX 버튼:** 영상 분석 없이 유효 칸마다 그리드 칸 중심에 정확히 bbox를 놓습니다. 그리드를 손으로 잘 맞췄을 때(칩 사이 간격이 좁아 Detect BBOX가 흔들릴 때). Ctrl+Z로 되돌림.

## 1.1.23

- **Angle 칸 단위 0.001도:** 화살표·휠 한 번에 0.001도(PgUp / PgDn은 0.01도). config.json `"ui_defaults": {"angle_step": 0.01}`처럼 바꿀 수 있습니다.

## 1.1.22

- **그리드를 맞춰 둔 뒤 Detect BBOX가 bbox를 밀던 문제:** 1.1.8의 칩 외곽 찾기가 그리드가 맞아도 돌아, 선언 크기만큼 떨어진 다른 경계 쌍을 잡으면 전체를 옮겼습니다. 이제 기준 칩 경계를 못 찾았을 때(원점이 많이 틀렸을 때)만 씁니다.
- **From grid**(Detect BBOX 옆): 그리드를 믿고, 칩끼리만 맞춘 채 전체 위치는 그리드 칸 중심으로. 칩 사이 간격이 아주 좁을 때나 그리드를 손으로 잘 맞췄을 때. config `"auto_bbox": {"absolute": "grid"}`(Auto Run도).

## 1.1.21

- **큰 PNG / JPG 메모리:** 처음 열 때 한 번 흑백으로 풀어 디스크 캐시(`%LOCALAPPDATA%\TIF_Viewer\image_cache`, 최근 3개)에 저장하고, 이후 메모리 맵으로 필요한 부분만 읽습니다. 12000×12000 컬러: 연 뒤 메모리 1040 → 약 90 MB, 두 번째부터는 변환 없이 바로 열림(60 MB). 25000×25000 8-bit는 캐시 파일 625 MB.

## 1.1.20

- **PNG / JPG / BMP 열기:** 단일 레이어 이미지도 열 수 있습니다(열기 창 "Images"; 8·16-bit 흑백, 컬러는 밝기로, 팔레트·알파 PNG도). 마킹·bbox·저장은 TIFF와 같고, 레이어가 하나라 레이어를 비교하는 탐지(Layer change)는 쓸 수 없습니다.

## 1.1.19

- **for_copy: 불량 다이의 모든 레이어:** 저장 창 "... all layers of a defective die"(기본 켜짐). 불량이 한 레이어에라도 있는 다이는 불러온 모든 레이어의 overlay를 `for_copy\`와 그 다이의 그룹 폴더들에 저장합니다. 끄면 예전처럼 불량이 마킹된 레이어만. config.json `"ui_defaults": {"export": {"for_copy_all_layers": false}}`.

## 1.1.18

- **불량 통계·spec out·bin (저장 시, CSV 옵션):** `defect_rate.csv`(LEG별 × 타입 / 그룹 / 전체 / bin: 발생 칩 수, spec out 칩 수, 전체 칩 수, 불량률 %), `chip_bins.csv`(칩별 bin, 타입별 0/1/2), `bin_map.txt`(bin 격자).
  - spec out 기준은 `plugins\defect_spec.py`의 `spec_out()`에 직접 작성(기본: 규칙 없음 = 발생하면 spec out). bin은 config.json `"defect_stats": {"bins": [...]}` 우선순위 또는 플러그인 `chip_bin()`. README "불량 통계" 참고.
  - zip 설치: `run.bat` / `build.bat`이 `defaults\plugins\`의 새 플러그인 파일(`defect_spec.py`)을 없을 때만 복사합니다.

## 1.1.17

- **for_copy를 불량 그룹별로도:** `for_copy\` 전체 폴더에 더해 `for_copy\<그룹>\`에도 저장합니다. 그룹은 void 타입의 `"group"`(config.json `void_types`에 `"group": "Crack"`처럼; 여러 타입을 한 그룹으로), 없으면 타입 이름. 그룹이 둘인 이미지는 두 폴더에 다 들어갑니다. 상태바에 그룹별 장수.

## 1.1.16

- **저장 시 불량 이미지만 따로 (`for_copy\`):** 불량 마킹(bbox 제외)이 그 레이어에 있는 칩의 overlay 이미지만 `for_copy` 한 폴더에 모아 추가로 저장합니다(레이어 하위 폴더 없음). 저장 창의 "Defect images only, one folder (/for_copy)", 기본 켜짐. 끝나면 상태바에 장수 표시.

## 1.1.15

- **그리드 피치를 um로:** products.json `"grid": {"pitch_um": [1000, 1000]}`(또는 숫자 하나, `pitch_x_um` / `pitch_y_um`). 피치 px = pitch_um / Scale이고 Scale을 바꾸면 따라 바뀝니다(`chip_size_um`의 bbox처럼). Auto Run도 같습니다. 원점은 px 그대로.

## 1.1.14

- **exe의 `DLL load failed while importing _ctypes`:** conda Python으로 만든 가상환경에서 빌드하면 conda 환경의 `Library\bin`에 있는 DLL(_ctypes의 libffi, _ssl의 OpenSSL 등)을 PyInstaller가 찾지 못해 exe에서 빠졌습니다. 이제 spec이 그 폴더를 직접 찾아 넣습니다. 빌드 로그에 `[spec] conda DLLs taken from ...: libffi-8.dll, ...`가 나오면 들어간 것입니다. `build.bat`으로 다시 빌드하세요.

## 1.1.13

- **콘솔 빌드(오류 확인용):** `set BUILD_CONSOLE=1` 후 `build.bat` → 같은 창에서 `dist\TIF_Viewer\TIF_Viewer.exe`를 실행하면 시작 오류 전문(어느 모듈·DLL인지)이 창에 남습니다. 빌드가 끝나면 PyInstaller가 못 찾은 모듈 목록 파일(`build\TIF_Viewer\warn-TIF_Viewer.txt`) 위치도 알려 줍니다.

## 1.1.12

- **빌드는 개발 환경에서:** `build.bat`이 별도의 `.venv-build` 대신 `dev.bat`과 같은 `.venv-dev`에 PyInstaller를 더 설치해 빌드합니다. 소스로 잘 돌던 패키지가 그대로 exe에 들어갑니다(다른 환경: `set BUILD_VENV=폴더`). PyInstaller 훅(`pyinstaller-hooks-contrib`)도 최신으로 올려, 1.1.7에 들어간 requests 쪽 컴파일 모듈이 빠져 exe에서 `DLL load failed`가 나던 경우를 막습니다. 예전 `.venv-build` 폴더는 지워도 됩니다.

## 1.1.11

- **출력에 리스트 없음:** voids.json의 `outline` / `outlineR`(Shapes 디버그용 윤곽 좌표 목록)을 기본으로 쓰지 않습니다. 이제 voids.json·voids.csv·Results API의 모든 필드가 값 하나입니다. 필요하면 config.json `"output_columns": {"json": {"outline": "outline", "outline_r": "outlineR"}}`. 윤곽이 든 예전 파일도 그대로 읽습니다(화면의 Shapes는 지금 세션의 검출 결과로 계속 보임).

## 1.1.10

- **크기 조절은 한쪽으로:** 마킹 가장자리를 끌면 반대쪽 변(점)은 그대로 있고 잡은 쪽만 움직입니다(예전: 중심 고정, 양쪽이 같이 움직임). 타원의 둥근 부분을 잡으면 반대편 점이 고정됩니다. Alt를 누른 채 잡으면 예전처럼 중심 고정.

## 1.1.9

- **Detect BBOX가 멈춘 것처럼 보이던 문제:** 진행 창이 걸음 수가 아니라 0.1초마다 갱신되고, 칩 외곽 찾기 단계도 진행 표시·Cancel을 받습니다. 콘솔에 단계별 시간이 찍힙니다: `[Detect BBOX] N cells: coarse .. s, locate .. s, fine .. s`.
- **Detect BBOX 속도:** 칩이 많으면 레이어를 칩마다 잘라 읽지 않고 한 번만 줄여 읽어 모든 칩을 거기서 자릅니다. 칩 외곽 찾기는 칩 32개 × 레이어 2장만 씁니다. 600 px 칩 100개 합성 웨이퍼: 9.6 → 4.1초(bbox 위치는 같음). 압축 TIFF·칩 수가 많을수록 차이가 큽니다.

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
