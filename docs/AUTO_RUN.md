# 자동 실행 (측정 파일 → 마킹 → 검토 요청)

측정된 TIF가 폴더에 들어오면 사람 손 없이 다음을 끝까지 하고, 사람에게 **검토만** 요청합니다.

1. 파일 열기 (TIF, 또는 Export한 프로젝트 폴더 / `patches.json`)
2. Wafer ID 찾기
3. 제품(WF Product) 정하고 적용 (그리드, Scale, bbox 크기, layout, type_rules)
4. 본딩맵 가져오기 (플러그인 `fetch_bonding_map`)
5. bbox 검출 → 그리드 맞추기(Fit Grid) → 맞춘 그리드로 bbox 다시 검출
6. 보이드 검출 (기본: Layer change, 제품의 학습 필터가 있으면 사용)
7. 저장: `voids.json` + `meta.json`(뷰어에서 그대로 열림), `summary.json`, `pipeline.log`
8. (선택) Results API로 전송
9. 알림: `review_queue.jsonl`에 한 줄 추가, 플러그인 `notify`가 있으면 호출

화면(MainWindow) 없이 돌아가는 모듈 `sat_widgets/pipeline.py`와 명령 `tools/auto_run.py`로 되어 있습니다. 검출 알고리즘은 뷰어의 버튼과 같은 함수(`auto_bbox.detect_layout`, `fit_grid`, `void_detect.detect_voids`)를 그대로 부릅니다.

## 1. 실행

```bash
# 파일 하나
python tools/auto_run.py --once D:\SAT\in\LOT1234_W07.tif

# 폴더 감시 (Ctrl+C로 종료)
python tools/auto_run.py --watch D:\SAT\in
```

자주 쓰는 옵션 (config.json보다 우선):

| 옵션 | 내용 |
|---|---|
| `--config`, `--products` | 읽을 선언 파일 (기본: 뷰어와 같은 파일, 공유 폴더 모드면 그 캐시). 환경 변수 `TIF_VIEWER_CONFIG`, `TIF_VIEWER_PRODUCTS`도 됩니다 |
| `--output-root` | 결과 폴더의 뿌리 |
| `--plugin` | 플러그인 파일 (기본: `results_api.plugin`) |
| `--method` | `layer_change`(기본) / `die_to_die` / `patchcore` / `segmentation` |
| `--void-type`, `--threshold` | 자동 마킹 타입, 검출 Thr |
| `--start-layer`, `--end-layer` | 불러올 레이어 범위(절대 번호) |
| `--no-send` | Results API 전송 안 함 |
| `--pattern`, `--recursive`, `--stable`, `--poll`, `--state` | 감시 설정 (아래 `watch`) |

`--once`는 결과가 `ok`면 종료 코드 0, 아니면 1입니다(스케줄러에서 쓰기 좋게).

## 2. 감시 동작

- `patterns`에 맞는 파일 이름(대소문자 무시)을 `poll_seconds`마다 찾습니다.
- 파일 **크기와 수정 시각이 `stable_seconds` 동안 그대로**일 때 다 써진 것으로 보고 처리합니다(복사 중인 파일은 기다림). 크기 0인 파일은 건너뜁니다.
- 처리한 파일은 상태 파일(기본 `<output_root>/auto_run_state.json`)에 적고 다시 처리하지 않습니다. 같은 이름이라도 크기나 수정 시각이 바뀌면(다시 측정) 다시 처리합니다. 다시 돌리려면 상태 파일에서 그 줄을 지우면 됩니다.
- 한 파일이 실패해도(깨진 파일, 플러그인 오류) 그 파일을 `error`로 기록하고 다음 파일로 갑니다.
- Ctrl+C: 바로 멈춥니다. 처리 중이던 파일은 기록되지 않으므로 다음 실행 때 다시 처리합니다. 서비스 종료(SIGTERM)는 처리 중인 파일을 끝낸 뒤 멈춥니다.
- 프로젝트 폴더를 감시하려면 `"patterns": ["patches.json"], "recursive": true`.

## 3. 설정 (`config.json` → `"auto_run"`)

프로그램은 config.json / products.json을 **읽기만** 합니다. 모든 키는 생략할 수 있습니다.

```json
"auto_run": {
    "output_root": "D:\\SAT\\auto_results",
    "output_dir": "{out}/{product}/{wafer_id}",
    "products": [
        {"pattern": "^LOTA", "product": "WF_A"},
        {"pattern": "_B\\d+_", "product": "WF_B"}
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
    },
    "watch": {
        "patterns": ["*.tif", "*.tiff"],
        "recursive": false,
        "stable_seconds": 10,
        "poll_seconds": 5,
        "state_file": null
    },
    "notify": {}
}
```

| 키 | 기본 | 내용 |
|---|---|---|
| `output_root` | `auto_results` | 결과의 뿌리 폴더(`review_queue.jsonl`, 상태 파일도 여기) |
| `output_dir` | `{out}/{product}/{wafer_id}` | 웨이퍼별 폴더. `{out}`, `{product}`, `{wafer_id}`, `{source_name}`(확장자 뺀 파일 이름), `{date}`(YYYY-MM-DD). 같은 폴더면 덮어씀 |
| `products` | 없음 | 제품 규칙: `pattern`(정규식)이 **파일 이름**에서, 아니면 **Wafer ID**에서 맞으면 그 `product`. 위에서부터 처음 맞는 것 |
| `start_layer`, `end_layer` | `ui_defaults`의 값, 없으면 전부 | 불러올 레이어 범위 |
| `method` | `layer_change` | 보이드 검출 방법. `segmentation`이면 `"seg_model"`(기본 `models/void_unet.onnx`) |
| `threshold` | 방법의 기본값 (`void_detect`의 `lc_threshold` 등) | 검출 Thr |
| `void_type` | `ui_defaults.void_type`, 없으면 bbox가 아닌 첫 타입 | 자동 마킹 타입 |
| `skip_layers`, `layer_rule` | `ui_defaults` → `void_detect` | 뷰어의 Skip layers / Mark on layer |
| `use_learned` | `true` | 제품의 학습 필터(`void_learn.model_path`, 예: `~/TIF_Viewer_train/products/<제품>/lc_classifier.json`)가 **있으면** 사용 |
| `require_bonding_map` | `true` | 본딩맵을 못 가져오면 멈춤(`needs_attention`). `false`면 모든 칸 검사 |
| `fit_grid` | `true` | bbox로 그리드 맞추기 후 bbox 다시 검출 |
| `send_results` | `false` | 끝나면 `send(build_payload(ctx))`로 전송 |
| `plugin` | `results_api.plugin` | 플러그인 파일 |
| `auto_bbox`, `void_detect` | `{}` | config.json의 `auto_bbox` / `void_detect` 위에 덮어쓸 값 (자동 실행에만) |
| `qc` | 아래 | 품질 관문 |
| `watch` | 위 | 감시 설정 |
| `notify` 등 | | 자유 키: 플러그인 `notify` / `product_for`가 `settings`로 받음 |

제품 결정 순서: 플러그인 `product_for` → `auto_run.products` 규칙 → products.json `default`. 제품이 정해지지 않거나 products.json에 없으면 `needs_attention`.

Wafer ID 결정 순서: (프로젝트) `patches.json`의 `wafer_id` → 플러그인 `parse_wafer_id` → `config.json`의 `"wafer_id": {"pattern"}` → 확장자 뺀 파일 이름 (뷰어와 같음).

본딩맵: 플러그인 `fetch_bonding_map(ctx, settings)` (뷰어의 Fetch Map과 같은 함수). Reject 값은 결과의 `reject_values`, 없으면 `ui_defaults.reject_values`. 그리드 행·열은 본딩맵 크기로 정해지고, 제품 선언과 다르면 경고를 남깁니다. 프로젝트 폴더는 그 안의 `bonding_map.txt`를 씁니다.

## 4. 품질 관문 (`qc`)

관문에 걸리면 그 웨이퍼는 거기서 멈추고 상태가 `needs_attention`이 되며 이유가 적힙니다. **그때까지 만든 것(예: bbox)은 저장**하고 알림도 보냅니다. 예외가 나면 `error`(traceback은 summary.json과 pipeline.log).

| 키 | 기본 | 걸리는 경우 |
|---|---|---|
| `max_estimated_frac` | 0.3 | layout과 맞지 않아 격자에서 추정한 bbox가 본딩 칸의 30 % 초과 → "grid / bonding map mismatch" |
| `max_fit_rms_px` | 3.0 | Fit Grid 잔차(px) 초과 |
| `max_angle_change` | 1.0 | 맞춘 각도가 제품 선언 각도와 1도 넘게 다름 |
| `require_package_box` | `true` | 골든 칩에서 패키지 테두리를 못 찾음: 그리드 원점이 칸의 일부만큼 어긋나 bbox가 칩 사이(스트리트)에 놓였을 가능성 |
| `max_voids_per_chip` | 없음 | 칩당 평균 보이드 수 초과(예: 5) |

그 밖에 멈추는 경우: 파일을 못 엶 / 레이어 2장 미만, 제품 없음, 본딩맵 없음·해석 불가·본딩 칸 0, `bbox` 타입 미선언, 본딩 칸 3개 미만, Fit Grid에 bbox 부족, 마킹할 보이드 타입 없음. 값은 `summary.json`의 `qc`에 늘 기록되므로, 실제 데이터로 몇 장 돌려 보고 기준을 정하면 됩니다.

## 5. 결과

```
<output_root>/
  review_queue.jsonl          웨이퍼마다 한 줄: status, reason, wafer_id, product, output_dir, counts, review
  auto_run_state.json         감시 상태 (처리한 파일)
  <product>/<wafer_id>/
    voids.json, meta.json     뷰어의 Save JSON과 같은 형식 (Load로 열어 검토). meta.json에 "review": "pending",
                              "auto_run": {"status", "reason"}, source_file, wafer_id, product, layers
    bonding_map.txt           사용한 본딩맵
    summary.json              status / reason, 단계별 시간과 내용, 레이어·타입별 개수, qc 값, 사용한 모델 파일,
                              버전, "review": "pending"
    pipeline.log              콘솔 출력 전체
```

자동 마킹은 뷰어처럼 점선(`auto: "detected"`, bbox는 `"estimated"`)으로 표시되므로 **Next Detected / Next Estimated**로 하나씩 확인하면 됩니다. 검토를 마친 표시(`review`)를 바꾸는 기능은 아직 없습니다.

## 6. 플러그인 함수 (선택, `plugins/results_api.py`)

기존 함수(`build_payload`, `send`, `fetch`, `to_voids`, `fetch_bonding_map`, `parse_wafer_id`)는 뷰어와 같이 `config.json`의 `"results_api"`를 `settings`로 받습니다. `ctx`는 뷰어의 `build_context`와 같은 키를 가지고(`last_export_dir`는 결과 폴더), `ctx["auto_run"] = True`가 더해집니다.

새 함수 (settings = `config.json`의 `"auto_run"`):

```python
def product_for(wafer_id, file_name, settings) -> str | None   # 제품 이름 (None: 규칙 / default로)
def notify(summary, settings) -> None                          # 검토 요청 보내기 (메일, 메신저 ...)
```

`notify`가 없으면 콘솔 출력과 `review_queue.jsonl`만 남습니다(`review_queue.jsonl`은 `notify`가 있어도 늘 씀). 예시는 플러그인 파일의 주석에 있습니다. `notify`에서 예외가 나도 결과는 저장되어 있고 경고만 남습니다.

## 7. 시험 (합성 웨이퍼)

```bash
SEED=204 python tools/synth/make_wafer.py /tmp/data ml_204
# 임시 config.json / products.json / 플러그인을 만들고 (저장소의 선언 파일은 건드리지 않음)
python tools/auto_run.py --watch /tmp/in --config /tmp/t/config.json --products /tmp/t/products.json
cp /tmp/data/ml_204.tif /tmp/in/
```
