# Results API 플러그인 개발 가이드

마킹 결과를 다른 시스템으로 **보내고**(Send), 다른 시스템의 마킹을 **불러오는**(Load) 부분은 사이트마다 다르므로 직접 작성하는 플러그인 파일 하나로 분리했습니다.

- 플러그인: `plugins/results_api.py` (exe 빌드 시 `dist\TIF_Viewer\plugins\`로 복사)
- 설정: `config.json`의 `"results_api"`
- 화면: Extraction 패널의 **Results API: Preview / Send / Load**, WF Product 묶음의 **Wafer ID / Fetch Map**

버튼을 누를 때마다 플러그인 파일을 다시 읽으므로, **파일을 고치면 프로그램을 다시 시작하거나 exe를 다시 빌드하지 않아도** 바로 반영됩니다.

## 1. 흐름

| 버튼 | 호출 | 스레드 |
|---|---|---|
| Preview | `build_payload(ctx)` → 결과 JSON을 창에 표시(저장·복사 가능) | 화면 |
| Send | `build_payload(ctx)` → `send(payload, settings)` → `(ok, 메시지)`를 대화상자로 표시 | 만들기는 화면, 전송은 백그라운드 |
| Load | `fetch(ctx, settings)` → `to_voids(data, ctx)` → voids.json 형식으로 불러오기 | 요청은 백그라운드, 변환·적용은 화면 |
| Fetch Map | `fetch_bonding_map(ctx, settings)` → 본딩맵 칸에 넣고 적용(Reject Values 포함 가능) | 요청은 백그라운드 |
| 파일 열기 | (있으면) `parse_wafer_id(file_name, settings)` → Wafer ID 칸. `"bonding_map": {"auto_fetch": true}`이면 이어서 Fetch Map | 화면 |

- 전송·요청은 백그라운드에서 돌아서 화면이 멈추지 않습니다.
- 플러그인에서 예외가 나면 오류 대화상자의 **자세히**에 traceback이 나옵니다(콘솔에도 출력).
- Load는 **Ctrl+Z 한 번**으로 되돌릴 수 있습니다. `"replace": true`(기본)는 현재 마킹을 바꾸고(마킹이 있으면 확인), `false`는 현재 마킹에 더합니다.
- Export가 끝난 뒤 자동으로 보내려면 `"send": {"after_export": true}`.

## 2. 설정 (`config.json`)

```json
"results_api": {
    "plugin": "plugins/results_api.py",
    "send": {
        "url": "http://server/api/voids",
        "method": "POST",
        "timeout": 30,
        "headers": {"Authorization": "Bearer ..."},
        "after_export": false
    },
    "receive": {
        "url": "http://server/api/voids?wafer={wafer_id}&layer={current_layer}",
        "timeout": 30,
        "replace": true
    },
    "bonding_map": {
        "url": "http://mapserver/api/bondingmap/{wafer_id}",
        "timeout": 30,
        "auto_fetch": true
    }
}
```

- `plugin`: 프로그램 폴더 기준 상대 경로 또는 절대 경로. 공유 폴더의 파일(`\\server\share\results_api.py`)을 지정하면 한 곳에서 관리할 수 있습니다.
- `settings`로 이 객체 전체가 플러그인에 전달됩니다. 필요한 키는 자유롭게 추가해 플러그인에서 읽으면 됩니다.
- 기본 플러그인은 URL의 `{wafer_id}`, `{source_name}`, `{product}`, `{current_layer}`를 채웁니다(URL 인코딩).
- `"bonding_map": {"auto_fetch": true}`: 파일을 열면 Wafer ID를 채운 뒤 바로 본딩맵을 가져옵니다. 프로젝트를 열 때는 저장된 `bonding_map.txt`가 있으면 가져오지 않습니다.

## 2.5 Wafer ID

- WF Product 묶음에 **File**(연 파일 이름, 마우스를 올리면 전체 경로)과 **Wafer ID** 칸이 있습니다. Wafer ID는 직접 고칠 수 있습니다.
- 파일을 열면 이름에서 Wafer ID를 채웁니다. 순서:
  1. 플러그인에 `parse_wafer_id(file_name, settings)`가 있으면 그 결과
  2. `config.json`의 `"wafer_id": {"pattern": 정규식, "group": 1 또는 그룹 이름}` (파일 이름에서 `re.search`)
  3. 확장자를 뺀 파일 이름
- 예: `"wafer_id": {"pattern": "^(?P<lot>[A-Z]+\\d+)_W(?P<wafer>\\d+)", "group": "lot"}`. 맞지 않으면 파일 이름을 쓰고 콘솔에 알립니다.
- Export하면 `patches.json`에 `wafer_id`를 저장하고, 프로젝트를 다시 열면 그 값을 씁니다.
- `ctx["wafer_id"]`로 플러그인에 전달되고, 기본 페이로드에도 들어갑니다.

## 3. 함수

```python
def build_payload(ctx) -> object        # JSON으로 보낼 내용 (json.dumps 가능해야 함)
def send(payload, settings) -> (bool, str)   # 성공 여부, 사용자에게 보일 메시지
def fetch(ctx, settings) -> object      # 응답(파싱된 JSON 등), to_voids로 전달
def to_voids(data, ctx) -> dict | list  # voids.json 형식
def fetch_bonding_map(ctx, settings) -> str | list | dict   # ctx["wafer_id"]의 본딩맵
def parse_wafer_id(file_name, settings) -> str              # 선택: 파일 이름 -> Wafer ID
def product_for(wafer_id, file_name, settings) -> str | None   # 선택, Auto Run: 제품 이름 (settings = "auto_run")
def notify_review(title, body, recipients, ctx, settings) -> (bool, str)   # Auto Run: 검토 요청 알림
```

- `parse_wafer_id`는 `{"wafer_id": ..., "lot": ...}`도 돌려줄 수 있습니다(뷰어는 `wafer_id`만 씀, Auto Run은 Lot으로 제품을 고름).
- `notify_review`: 기본은 `"results_api": {"notify": {"url": "http://mailserver/send?to={to}&title={title}&body={body}", "recipients": [...], "recipients_sep": ",", "timeout": 30, "headers": {}}}`로 GET 요청(세 자리표시자는 URL 인코딩), `url`이 없으면 출력하고 `review_queue.jsonl`에 추가합니다. 인자·반환·기본 동작의 자세한 약속은 [AUTO_RUN.md 7절](AUTO_RUN.md#7-검토-요청-알림-notify_review-직접-바꿀-수-있는-곳)에 있습니다.

요청 설정(`send` / `receive` / `bonding_map` / `notify` 각각): `"method"`, `"params"`, `"headers"`, `"timeout"`, `"verify"`(회사 HTTPS 인증서 문제면 `false` 또는 CA 파일 경로). `params`는 `requests.get(url, params=...)`로 들어가고, 값이 dict / list이면 JSON 문자열로 바꿔 넣습니다. 문자열 안의 `{wafer_id}` 등은 채워집니다.

```json
"bonding_map": {"url": "http://서버/api/map", "params": {"filters": {"waferId": "{wafer_id}", "layer": [1, 2]}, "format": "csv"}}
```
→ `GET http://서버/api/map?filters={"waferId": "W07", "layer": [1, 2]}&format=csv`(URL 인코딩됨). 요청마다 실제 URL과 응답 코드가 콘솔에 `[results_api] GET ... -> HTTP 200`으로 찍히고, 실패하면 오류 창에도 URL이 나옵니다.

`fetch_bonding_map`이 돌려줄 수 있는 형식:

- 엑셀에서 복사한 것과 같은 텍스트(탭 구분, 행·열 번호 머리글 허용) 또는 CSV 텍스트
- 행 목록 `[["A", "A", "B"], ["A", "X", "B"], ...]`
- `{"text": ... 또는 "rows": ..., "reject_values": ["X", "NG"] 또는 "X, NG"}` (Reject Values를 주면 그 값으로 바꿈)
- **좌표 목록**(맵이 격자 전체를 덮지 않을 때): `{"cells": [{"row": 0, "col": 1, "value": "A"}, ...], "origin": 0, "reject_values": [...]}`
  - `row` = 격자 행(위에서부터), `col` = 격자 열(왼쪽부터). `"y"` / `"x"`로 써도 됩니다. `[[row, col, value], ...]`도 됩니다.
  - `origin`: 첫 칸 번호가 0이면 0(기본), 1이면 1.
  - `value`: 그 칸의 값(LEG 등). 없으면 `"1"`. 목록에 없는 칸은 본딩 안 된 칸(빈 칸)입니다.
  - 맵 크기는 가장 큰 행·열 번호까지입니다(그리드 Rows/Cols가 그 크기로 맞춰짐).
- **칸마다 한 줄인 표**(CSV / 탭 텍스트, 또는 JSON 레코드 목록): 머리글에 `x`, `y`와 값 열(`leg`, 없으면 `value` / `key` / `label` / `bin`)이 있으면 자동으로 격자로 바꿉니다. 예: `base,x,y,leg,...`. 대소문자 무관, 다른 열은 무시합니다.
  - 기본: `x` = 열(왼쪽부터), `y` = 행(위에서부터), 가장 작은 x / y가 첫 칸(`"origin": "auto"`, 웨이퍼 중심 기준 음수 좌표도 됨).
  - `config.json`의 `"bonding_map": {"table": {...}}`로 바꿉니다: `"y_up": true`(y가 위로 커지면: 가장 큰 y가 맨 윗줄), `"origin": 0` 또는 `1`(좌표 고정), `"x"` / `"y"` / `"value"`(열 이름이 다를 때), `"where": {"base": "B1"}`(한 표에 여러 맵이 있을 때 그 값의 줄만).
  - 같은 칸이 다른 값으로 두 번 나오면 오류로 알립니다(`where`로 하나를 고르세요). 위아래가 뒤집혀 보이면 `"y_up": true`.

### ctx (읽기 전용)

| 키 | 내용 |
|---|---|
| `app_version`, `timestamp`, `user`, `host` | 프로그램 버전, ISO 8601 시각, 사용자, PC 이름 |
| `product` | 선택한 WF 제품 이름 |
| `source_file`, `source_name` | 연 TIF 또는 프로젝트 폴더 경로, 확장자 없는 이름 |
| `wafer_id` | Wafer ID 칸의 값 |
| `layers`, `current_layer` | 불러온 절대 레이어 번호 목록, 현재 레이어 |
| `grid` | `start_x, start_y, pitch_x, pitch_y, rows, cols, angle` |
| `um_per_px` | Scale (0 = 미설정) |
| `bonding_map` | `{"text": 붙여넣은 맵, "reject_values": [...], "rows": [[칸별 LEG]]}` 또는 None |
| `voids_json` | "Save JSON"이 쓰는 meta.json 내용(`version, layer_numbering, um_per_px, pixel_scale, grid, types`)에 voids.json의 보이드 목록을 `voids`로 합친 dict. 각 보이드의 필드 이름은 `config.json`의 `output_columns`를 따릅니다 |
| `records` | 보이드마다 내부 이름의 dict: `layer, row, col, label(LEG), type, center_x/y, radius_x/y, p, global_x/y, bbox_rel_x/y, *_raw, *_um ...` (`sat_widgets/output_columns.py`). 픽셀 값은 그리드 X 피치 300 기준으로 정규화, `*_raw`는 실제 픽셀 (README "픽셀 값의 정규화") |
| `types` | 선언된 타입 `{이름: {color, shape, ...}}` |
| `last_export_dir` | 이번 실행에서 마지막으로 Export한 폴더 또는 None |
| `settings` | `config.json`의 `"results_api"` |

### to_voids가 돌려줄 형식

voids.json과 같습니다. 보이드마다 다음 중 하나로 위치를 줍니다.

- `globalCX`, `globalCY`: 원본 이미지(raw) 픽셀 좌표 (권장, 그리드와 무관)
- `x`(col), `y`(row), `centerX`, `centerY`: 칸 기준 상대 좌표

공통으로 `layer`(절대 번호, `"layer_numbering": "absolute"`), `type`(타입 이름), `radiusX`, `radiusY`, 선택으로 `p`. 필드 이름은 `output_columns`로 바꾼 이름과 기본 이름 둘 다 읽습니다. 위치·크기는 `globalCX_raw`, `radiusX_raw`처럼 `_raw`로 실제 픽셀을 주는 것이 가장 확실합니다. `_raw` 없이 일반 필드만 주면, 최상위에 `"pixel_scale": {"scale": 비율}`이 있을 때 그 비율로 나누고, 없으면 실제 픽셀로 읽습니다.

```python
def to_voids(data, ctx):
    return {
        "layer_numbering": "absolute",
        "voids": [{"layer": d["layer"], "type": d["defect_type"],
                   "globalCX": d["x_px"], "globalCY": d["y_px"],
                   "radiusX": d["w_px"] / 2, "radiusY": d["h_px"] / 2} for d in data["items"]],
    }
```

선언에 없는 타입 이름은 파일에 들어 있던 타입처럼 추가되어 표시됩니다(이번 실행에만).

## 4. 쓸 수 있는 모듈

기본 플러그인은 HTTP를 `requests`로 합니다(1.1.7부터 requirements와 exe에 포함). exe에는 그 밖에 플러그인이 쓸 만한 표준 라이브러리를 넣어 두었습니다: `json, urllib.request/parse/error, http.client, ssl, base64, hashlib, hmac, uuid, csv, socket, getpass, email, zipfile, gzip`.
다른 패키지가 필요하면 `requirements.txt`에 추가하고 `TIF_Viewer.spec`의 `hiddenimports`에 넣은 뒤 exe를 다시 빌드해야 합니다. 소스 실행(dev.bat)에서는 가상환경에 설치만 하면 됩니다.

## 5. 개발 팁

- **Preview**로 보낼 내용을 먼저 확인하고 **Save...**로 파일로 저장해 서버 쪽 개발에 씁니다.
- 로컬에서 시험할 때는 `python -m http.server`는 POST를 받지 못하므로, 간단한 테스트 서버를 쓰거나 `send`에서 파일로 저장하도록 잠시 바꿔 확인합니다.
- `send`가 예외 대신 `(False, "이유")`를 돌려주면 사용자에게 경고로 보입니다.
