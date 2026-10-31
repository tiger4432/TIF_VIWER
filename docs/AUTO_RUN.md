# 자동 실행 (Auto Run: 측정 파일 → 마킹 → 검토 요청)

측정된 TIF가 지정한 경로에 들어오면 **뷰어 창에서 눈에 보이게** 다음을 끝까지 하고, 사람에게 **검토만** 요청합니다.

1. Wafer ID(와 Lot) 찾기
2. 파일 열기 (Open TIF와 같음: 창에 이미지가 뜸)
3. 제품(WF Product) 정하고 적용 (그리드, Scale, bbox 크기, layout, type_rules)
4. 본딩맵 가져오기 (플러그인 `fetch_bonding_map`, Fetch Map과 같음)
5. Detect BBOX → Fit Grid(확인 창 없이 적용) → 맞춘 그리드로 Detect BBOX 다시
6. Detect Voids (기본: Layer change, 제품의 학습 필터가 있으면 사용)
7. 저장: `voids.json` + `meta.json`(뷰어에서 그대로 열림), `summary.json`, `pipeline.log`
8. (선택) Results API로 전송
9. 검토 요청: 플러그인 `notify_review` (기본: 메일 서버에 GET)

단계마다 결과가 창에 바로 나타납니다(이미지, 제품, 본딩맵, 그리드·bbox, 보이드). 무거운 계산은 별도 스레드에서 돌기 때문에 그동안에도 창을 움직이고 볼 수 있습니다. 한 번에 한 파일씩 처리하고, 그 사이에 들어온 파일은 대기열에 쌓입니다.

## 1. 사용법 (창)

Extraction 아래 **Auto Run** 묶음:

- **감시 경로**: glob 패턴. 여러 개는 `;`로 구분합니다(`**` = 모든 하위 폴더). 기본값은 `config.json`의 `"auto_run": {"watch": [...]}`이고, 칸에서 바꾼 값은 이번 실행에만 씁니다. **...** 버튼으로 폴더를 고르면 `폴더/**/*.tif; 폴더/**/*.tiff`가 들어갑니다.
- **Start Auto Run / Stop Auto Run**: 감시를 켜고 끕니다. Stop을 누르면 처리 중이던 웨이퍼를 취소하고(저장·알림 없음), 다음에 Start하면 그 파일을 다시 처리합니다.
- **상태 줄**: 감시 중인 경로, 지금 처리 중인 웨이퍼와 단계(경과 시간), 대기열 수, 마지막 결과.

레이어 범위는 `auto_run`의 `start_layer` / `end_layer`, 없으면 창의 Start / End Layer 값을 씁니다(창과 자동 실행이 같은 레이어를 보도록 Start 때 맞춤).

자동 실행 중에는 열려 있던 마킹을 묻지 않고 새 파일로 바꿉니다(결과는 이미 결과 폴더에 저장됨).

## 2. 창 없이 실행 (선택)

같은 단계를 창 없이 돌릴 수도 있습니다(서버, 스케줄러).

```bash
python tools/auto_run.py --watch                       # config.json "auto_run.watch"의 패턴
python tools/auto_run.py --watch "D:/SAT/raw/**/*.tif"  # 패턴 또는 폴더 지정
python tools/auto_run.py --once D:/SAT/raw/LOT1234_W07.tif
```

옵션: `--config`, `--products`(읽을 선언 파일; 환경 변수 `TIF_VIEWER_CONFIG`, `TIF_VIEWER_PRODUCTS`), `--output-root`, `--plugin`, `--method`, `--void-type`, `--threshold`, `--start-layer`, `--end-layer`, `--no-send`, `--stable`, `--poll`, `--state`. `--once`는 결과가 `ok`면 종료 코드 0, 아니면 1. Ctrl+C는 바로 멈추고(처리 중이던 파일은 다음에 다시), SIGTERM은 처리 중인 파일을 끝낸 뒤 멈춥니다.

## 3. 감시 동작

- `watch` 패턴에 맞는 파일을 `poll_seconds`마다 찾습니다(네트워크 공유도 됨. 찾기는 별도 스레드).
- 파일 **크기와 수정 시각이 `stable_seconds` 동안 그대로**일 때 다 써진 것으로 보고 대기열에 넣습니다(복사 중인 파일은 기다림). 크기 0인 파일은 건너뜁니다.
- 처리한 파일은 상태 파일(기본 `<output_root>/auto_run_state.json`)에 적고 다시 처리하지 않습니다. 같은 경로라도 크기나 수정 시각이 바뀌면(다시 측정) 다시 처리합니다. 강제로 다시 돌리려면 상태 파일에서 그 항목을 지우면 됩니다.
- 한 파일이 실패해도(깨진 파일, 플러그인 오류) 그 파일을 기록하고 다음 파일로 갑니다.
- Windows에서는 패턴의 대소문자를 구분하지 않고, Linux에서는 구분합니다.

## 4. 설정 (`config.json` → `"auto_run"`)

프로그램은 config.json / products.json을 **읽기만** 합니다. 모든 키는 생략할 수 있습니다.

```json
"wafer_id": {"pattern": "^(?P<lot>[A-Z]+\\d+)_W(?P<wafer>\\d+)", "group": 0, "lot_group": "lot"},
"auto_run": {
    "watch": ["D:/SAT/raw/**/*.tif", "\\\\server\\sat\\*\\*.tif"],
    "stable_seconds": 10,
    "poll_seconds": 5,
    "state_file": null,
    "output_root": "D:/SAT/auto_results",
    "output_dir": "{out}/{product}/{wafer_id}",
    "products": [
        {"lot": "^LOTA", "product": "WF_A"},
        {"lot": "^LOTB", "wafer_id": "-(0[1-9]|1[0-2])$", "product": "WF_B"},
        {"file": "_TEST_", "product": "WF_TEST"}
    ],
    "start_layer": 0, "end_layer": 11,
    "method": "layer_change",
    "threshold": 26,
    "void_type": "Void",
    "skip_layers": [0],
    "layer_rule": "first",
    "use_learned": true,
    "require_bonding_map": true,
    "fit_grid": true,
    "send_results": false,
    "plugin": "plugins/results_api.py",
    "auto_bbox": {},
    "void_detect": {},
    "qc": {
        "max_estimated_frac": 0.3,
        "max_fit_rms_px": 3.0,
        "max_angle_change": 1.0,
        "require_package_box": true,
        "max_voids_per_chip": null
    }
},
"results_api": {
    "notify": {
        "url": "http://mailserver/send?to={to}&title={title}&body={body}",
        "recipients": ["reviewer1@company.com", "reviewer2@company.com"],
        "recipients_sep": ",",
        "timeout": 30,
        "headers": {}
    }
}
```

| 키 | 기본 | 내용 |
|---|---|---|
| `watch` | 없음 | 측정 파일의 glob 패턴(문자열 하나 또는 목록, `**` 하위 폴더 전부). 창의 칸에서 바꿀 수 있음 |
| `stable_seconds` | 10 | 크기·수정 시각이 이만큼 그대로면 복사 완료로 봄 |
| `poll_seconds` | 5 | 찾는 간격 |
| `state_file` | `<output_root>/auto_run_state.json` | 처리한 파일 기록 |
| `output_root` | `auto_results` | 결과의 뿌리 폴더 (`review_queue.jsonl`도 여기) |
| `output_dir` | `{out}/{product}/{wafer_id}` | 웨이퍼별 폴더. `{out}`, `{product}`, `{wafer_id}`, `{lot}`, `{source_name}`(확장자 뺀 파일 이름), `{date}`(YYYY-MM-DD). 같은 폴더면 덮어씀 |
| `products` | 없음 | 제품 규칙 (아래) |
| `start_layer`, `end_layer` | 창의 값 (창 없이: `ui_defaults`, 없으면 전부) | 불러올 레이어 범위 |
| `method` | `layer_change` | `layer_change` / `die_to_die` / `patchcore` / `segmentation`(`"seg_model"`, 기본 `models/void_unet.onnx`) |
| `threshold` | 방법의 기본값 (`void_detect`의 `lc_threshold` 등) | 검출 Thr |
| `void_type` | `ui_defaults.void_type`, 없으면 bbox가 아닌 첫 타입 | 자동 마킹 타입 |
| `skip_layers`, `layer_rule` | `ui_defaults` → `void_detect` | 뷰어의 Skip layers / Mark on layer |
| `use_learned` | `true` | 제품의 학습 필터(예: `~/TIF_Viewer_train/products/<제품>/lc_classifier.json`)가 **있으면** 사용 |
| `require_bonding_map` | `true` | 본딩맵을 못 가져오면 멈춤. `false`면 모든 칸 검사 |
| `fit_grid` | `true` | bbox로 그리드 맞추고 bbox 다시 검출 |
| `send_results` | `false` | 끝나면 Results API `send(build_payload(ctx))`로 전송 |
| `plugin` | `results_api.plugin` | 플러그인 파일 |
| `auto_bbox`, `void_detect` | `{}` | config.json의 `auto_bbox` / `void_detect` 위에 덮어쓸 값 (자동 실행에만) |
| `qc` | 아래 | 품질 관문 |

### Wafer ID와 Lot

순서: (프로젝트 폴더) `patches.json`의 `wafer_id` → 플러그인 `parse_wafer_id(file_name, settings)` → `config.json`의 `"wafer_id": {"pattern"}` → 확장자 뺀 파일 이름.

- 플러그인은 문자열(Wafer ID) 또는 `{"wafer_id": ..., "lot": ...}`를 돌려줄 수 있습니다.
- 정규식으로는 `"group"`(Wafer ID, 기본 1 또는 전체)과 `"lot_group"`(Lot, 번호 또는 그룹 이름)으로 지정합니다. 플러그인이 Lot을 주지 않으면 Lot은 정규식에서 찾습니다.

### 제품 규칙 (`products`)

```json
"products": [
    {"lot": "정규식", "wafer_id": "정규식", "file": "정규식(선택)", "product": "제품 이름"}
]
```

- 위에서부터 보고, **적어 둔 패턴이 모두 맞는**(`re.search`) 첫 규칙의 제품을 씁니다. 패턴은 `lot`, `wafer_id`, `file`(파일 이름) 중 하나 이상이 있어야 합니다.
- 맞는 규칙이 없으면 추측하지 않고 `needs_attention` "no product for lot / wafer ID"로 멈춥니다(products.json의 default를 쓰지 않음).
- 플러그인에 `product_for(wafer_id, file_name, settings)`가 있으면 그 결과가 우선합니다(None을 돌려주면 규칙으로).
- 제품 이름이 products.json에 없으면 `needs_attention`.

### 본딩맵

플러그인 `fetch_bonding_map(ctx, settings)` (Fetch Map과 같은 함수). Reject 값은 결과의 `reject_values`, 없으면 `ui_defaults.reject_values`. 그리드 행·열은 본딩맵 크기로 정해지고, 제품 선언과 다르면 경고를 남깁니다. 프로젝트 폴더는 그 안의 `bonding_map.txt`를 씁니다.

## 5. 품질 관문 (`qc`)

관문에 걸리면 그 웨이퍼는 거기서 멈추고 상태가 `needs_attention`이 되며 이유가 적힙니다. **그때까지 만든 것(예: bbox)은 저장**하고 검토 요청도 보냅니다. 예외가 나면 `error`(traceback은 summary.json과 pipeline.log).

| 키 | 기본 | 걸리는 경우 |
|---|---|---|
| `max_estimated_frac` | 0.3 | layout과 맞지 않아 격자에서 추정한 bbox가 본딩 칸의 30 % 초과 → "grid / bonding map mismatch" |
| `max_fit_rms_px` | 3.0 | Fit Grid 잔차(px) 초과 |
| `max_angle_change` | 1.0 | 맞춘 각도가 제품 선언 각도와 1도 넘게 다름 |
| `require_package_box` | `true` | 골든 칩에서 패키지 테두리를 못 찾음: 그리드 원점이 칸의 일부만큼 어긋나 bbox가 칩 사이(스트리트)에 놓였을 가능성 |
| `max_voids_per_chip` | 없음 | 칩당 평균 보이드 수 초과(예: 5) |

그 밖에 멈추는 경우: 파일을 못 엶 / 레이어 2장 미만, 제품 없음, 본딩맵 없음·해석 불가·본딩 칸 0, `bbox` 타입 미선언, 본딩 칸 3개 미만, Fit Grid에 bbox 부족, 마킹할 보이드 타입 없음. 수치는 `summary.json`의 `qc`에 늘 기록되므로, 실제 데이터로 몇 장 돌려 보고 기준을 정하면 됩니다.

## 6. 결과

```
<output_root>/
  review_queue.jsonl          검토 요청 기록 (메일 주소가 없거나 보내기에 실패했을 때)
  auto_run_state.json         처리한 파일
  <product>/<wafer_id>/
    voids.json, meta.json     뷰어의 Save JSON과 같은 형식 (Load로 열어 검토). meta.json에 "review": "pending",
                              "auto_run": {"status", "reason"}, source_file, wafer_id, lot, product, layers
    bonding_map.txt           사용한 본딩맵
    summary.json              status / reason, 단계별 시간과 내용, 레이어·타입별 개수, qc 값, 사용한 모델 파일,
                              버전, 보낸 검토 요청(title, recipients), "review": "pending"
    pipeline.log              그 웨이퍼의 콘솔 출력
```

자동 마킹은 점선(`auto: "detected"`, bbox는 `"estimated"`)으로 표시되므로 **Next Detected / Next Estimated**로 하나씩 확인하면 됩니다. 검토를 마친 표시(`review`)를 바꾸는 기능은 아직 없습니다.

## 7. 검토 요청 알림: `notify_review` (직접 바꿀 수 있는 곳)

`plugins/results_api.py`의 함수 하나입니다. 프로그램은 제목·본문·받는 사람을 만들어 이 함수만 부릅니다. 메일 서버가 달라지면 이 함수만 고치면 됩니다(재시작 없이 다음 웨이퍼부터 반영).

```python
def notify_review(title, body, recipients, ctx, settings) -> (ok, message)
```

| 인자 | 내용 |
|---|---|
| `title` | 한 줄 제목. 정상: `[TIF Viewer] {제품} {wafer_id}: {n} voids, review`, 그 밖: `[TIF Viewer] {제품} {wafer_id} needs attention: {이유}` (또는 `error: ...`) |
| `body` | 여러 줄 본문: Product, Lot, Wafer ID, File, Status(이유), 보이드 수(레이어별, 타입별), bbox 수 / 본딩 칸, QC 값, 결과 폴더 경로, 경고 |
| `recipients` | `config.json` `"results_api": {"notify": {"recipients": [...]}}` (없으면 `[]`) |
| `ctx` | Results API의 ctx와 같은 키(`wafer_id`, `lot`, `product`, `source_file`, `grid`, `records`, `last_export_dir` = 결과 폴더 ...) + `summary`(summary.json 내용) + `review_queue`(review_queue.jsonl 경로) |
| `settings` | `config.json`의 `"results_api"` 전체 (`settings["notify"]`가 위 설정) |
| 반환 | `(True, "메시지")` 성공, `(False, "이유")` 실패 |

기본 동작:

- `notify.url`이 있으면: `{to}`(받는 사람을 `recipients_sep`, 기본 `,`로 이어 붙인 것), `{title}`, `{body}`를 URL 인코딩해 채운 주소로 **GET** 요청(`timeout`, `headers` 사용). HTTP 2xx면 성공.
- `notify.url`이 없으면: 콘솔에 출력하고 `review_queue.jsonl`에 한 줄 추가.
- 함수가 `(False, ...)`를 돌려주거나 예외가 나면 프로그램이 경고를 남기고 요청을 `review_queue.jsonl`에 적어 둡니다(잃어버리지 않음). 플러그인에 함수가 없을 때도 `review_queue.jsonl`.

예: POST 방식 메일 API로 바꾸기

```python
def notify_review(title, body, recipients, ctx, settings):
    cfg = settings.get("notify") or {}
    data = json.dumps({"to": recipients, "subject": title, "text": body}).encode("utf-8")
    status, resp = _request(cfg["url"], dict(cfg, method="POST"), data)
    return 200 <= status < 300, f"HTTP {status}"
```

## 8. 그 밖의 플러그인 함수

- 기존 함수(`build_payload`, `send`, `fetch`, `to_voids`, `fetch_bonding_map`, `parse_wafer_id`)는 뷰어와 같이 `config.json`의 `"results_api"`를 `settings`로 받습니다. 자동 실행의 `ctx`는 뷰어와 같은 키에 `lot`, `auto_run: True`가 더해집니다.
- `product_for(wafer_id, file_name, settings) -> str | None` (선택, settings = `"auto_run"`): 제품 규칙보다 우선.

## 9. 시험 (합성 웨이퍼)

```bash
SEED=204 python tools/synth/make_wafer.py /tmp/data ml_204
# 임시 config.json / products.json / 플러그인을 공유 폴더 모드로 지정 (저장소의 선언 파일은 건드리지 않음)
TIF_VIEWER_SHARED_DIR=/tmp/t/share python tif_view_GL.py      # Auto Run: /tmp/in/**/*.tif, Start
cp /tmp/data/ml_204.tif /tmp/in/
```
