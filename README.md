# TIF Viewer (SAT Multi-gate)

SAT 멀티게이트 TIF를 레이어별로 보면서 디펙을 타원/사각형/초타원으로 마킹하고, 패치·마스크·메타데이터를 Export하는 프로그램입니다.

## 실행 환경

| 항목 | 버전 |
|---|---|
| OS | Windows 10/11 x64 (OpenGL 3.3 이상 지원 GPU 드라이버) |
| Python | 64-bit 3.10–3.13 (배포 exe는 3.10으로 빌드) |
| 패키지 | `requirements.txt` (numpy 2.2.6, PySide6 6.10.1, PyOpenGL 3.1.10, Pillow 11.3.0, tifffile 2025.5.10, imagecodecs 2025.3.30) |
| 빌드 | PyInstaller 6.16.0 (`requirements-build.txt`) |

`tifffile`이 없으면 PIL로 대체해서 읽지만, 이 경우 레이어마다 프레임 전체를 디코딩하므로 메모리를 많이 씁니다. 반드시 설치해서 사용하세요.

## 소스에서 실행

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python tif_view_GL.py
```

## 개발 실행 (최신 코드 + 실시간 반영)

```bat
dev.bat              REM git pull 후 소스로 실행
dev.bat watch        REM 위 + 30초마다 자동 pull (최소화된 "TIF_Viewer auto-pull" 창, 닫으면 중지)
dev.bat watch 10     REM 10초마다
```

- **동작:** `git pull` → 개발용 가상환경 `.venv-dev` 준비(`requirements.txt` + `requirements-dev.txt`) → `python -m jurigged -v tif_view_GL.py`로 실행합니다.
- **실시간 반영:** jurigged가 소스 파일을 감시합니다. 저장하거나 pull로 바뀐 함수/메서드는 실행 중인 프로그램에 바로 적용되고, 콘솔에 `Update ...`로 표시됩니다.
- **재시작이 필요한 변경:** `__init__`에서 만드는 위젯(새 버튼 등), 클래스 구조 변경이 해당합니다. `config.json`/`products.json` 변경은 "Reload Declarations"로 반영합니다.
- **Python 선택:** `build.bat`와 같은 규칙(`tools\setup_venv.bat`)을 씁니다.

## 빌드 가이드 (PyInstaller)

### 한 번에 빌드

저장소 루트에서 다음을 실행합니다.

```bat
build.bat
```

`build.bat`는 아래 순서로 동작합니다.

1. 빌드 전용 가상환경 `.venv-build`를 준비합니다(IDE가 쓰는 `.venv`와 분리). Python은 다음 순서로 찾습니다(dev.bat도 같음). **conda를 활성화하지 않아도 됩니다.**
   - `PYTHON` 환경변수 (예: `set PYTHON=C:\Python310\python.exe`)
   - 저장소 루트의 `python.local` 파일: python.exe 경로 한 줄 (git에 올라가지 않음)
   - 활성화된 conda 환경(base 제외)
   - conda 환경 `tifviewer`(다른 이름은 `set CONDA_ENV=이름`): 일반적인 설치 폴더(`%USERPROFILE%\miniconda3`, `anaconda3`, `miniforge3`, `%LOCALAPPDATA%`·`%ProgramData%` 아래, `%USERPROFILE%\.conda`)의 `envs\tifviewer\python.exe`를 직접 찾습니다
   - `py -3.10` (py launcher)
   - PATH의 `python`

   찾은 Python이 64-bit 3.10–3.13이 아니면 중단합니다. 가상환경이 다른 Python(설치 위치 또는 **버전**)으로 만들어졌으면 지우고 다시 만듭니다. 가상환경의 pip가 동작하지 않으면 그 Python에 들어 있는 pip로 고친 뒤, 그래도 안 되면 가상환경을 한 번 다시 만듭니다.
2. `requirements.txt`와 `requirements-build.txt`를 설치합니다.
3. `pyinstaller --noconfirm --clean TIF_Viewer.spec`을 실행합니다.
4. `config.json`, `products.json`, `deploy.json`을 `dist\TIF_Viewer\`(exe 옆)로 복사합니다.

conda에서 빌드하는 예시입니다.

```bat
conda create -n tifviewer -c conda-forge python=3.10
conda activate tifviewer
build.bat
```

결과물은 `dist\TIF_Viewer\TIF_Viewer.exe`와 `_internal\` 폴더입니다. 배포할 때는 `dist\TIF_Viewer` 폴더를 통째로 복사합니다.

### 수동 빌드

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -r requirements-build.txt
pyinstaller --noconfirm --clean TIF_Viewer.spec
copy /Y config.json dist\TIF_Viewer\
copy /Y products.json dist\TIF_Viewer\
```

### spec 요약 (`TIF_Viewer.spec`)

- **onedir 형식:** exe와 `_internal\`을 만듭니다. 기존 배포본과 같은 구조입니다.
- **창 모드:** `console=False`라 콘솔 창이 뜨지 않습니다. 오류는 대화상자로 표시합니다.
- **imagecodecs 하위 모듈 포함:** 압축 TIFF(LZW, JPEG 등)를 읽기 위해 넣었습니다.
- **tkinter 제외:** 쓰지 않으므로 배포 크기를 줄였습니다.

### 빌드 문제 해결

- **Python을 찾지 못함 / VS Code 터미널에서 conda가 안 불러짐:** conda 활성화 없이 됩니다. `conda create -n tifviewer ...`로 만든 환경은 자동으로 찾고, 다른 위치라면 저장소 루트에 `python.local` 파일을 만들어 python.exe 경로를 한 줄 적습니다(예: `C:\Users\me\miniconda3\envs\tifviewer\python.exe`). VS Code 터미널에서도 conda를 쓰려면 한 번 `conda init powershell`(또는 `cmd.exe`) 후 VS Code를 다시 시작합니다.
- **`AttributeError: module 'pkgutil' has no attribute 'ImpImporter'`:** 가상환경에 남아 있던 오래된 pip가 새 Python(3.12 이상)에서 실행될 때 생깁니다(같은 경로의 Python이 업그레이드된 경우 등). 이제 `dev.bat` / `build.bat`가 Python 버전을 비교해 가상환경을 다시 만들고, pip를 고칩니다. 그래도 나오면 `.venv-dev`(또는 `.venv-build`) 폴더를 지우고 다시 실행합니다.
- **`ImportError: DLL load failed while importing QtCore: 지정된 프로시저를 찾을 수 없습니다`:** Anaconda **base** Python으로 가상환경을 만든 경우입니다. Anaconda 폴더의 오래된 VC++ 런타임(`msvcp140.dll`)이 Qt 6보다 먼저 로드되어 생깁니다. 해결 방법은 둘 중 하나입니다.
  1. python.org에서 Python 3.10 x64를 설치하고, conda를 활성화하지 않은 새 콘솔에서 빌드합니다.
  2. `conda create -n tifviewer -c conda-forge python=3.10` 후 `conda activate tifviewer`로 전용 환경을 씁니다.

  그다음 `build.bat`를 다시 실행합니다. 빌드 전용 `.venv-build`가 다른 Python으로 만들어졌으면 `build.bat`가 자동으로 다시 만듭니다. PyInstaller 실행 전에 Qt가 로드되는지도 검사해서 안내합니다.
- **`.venv-build ... Permission denied` / `is in use`:** 그 가상환경의 python.exe가 실행 중이거나 에디터가 인터프리터로 잡고 있습니다. 해당 프로그램을 닫고 다시 실행합니다.
- **실행 시 OpenGL 오류:** GPU 드라이버를 업데이트합니다. 원격 데스크톱처럼 OpenGL 3.3을 지원하지 않는 환경에서는 실행되지 않습니다.
- **압축 TIF를 열지 못함:** `imagecodecs`가 설치되어 있는지 확인한 뒤 다시 빌드합니다.

### 릴리스 절차

1. `sat_widgets/version.py`의 `APP_VERSION`을 올립니다(예: `1.0.0` → `1.1.0`).
2. `build.bat`로 빌드합니다.
3. 배포 폴더에 새 버전을 올리고, 업데이트 매니페스트(아래 [업데이트 알림](#업데이트-알림))의 `version`과 `download`를 갱신합니다.

## 선언 파일과 중앙 공유 폴더

| 파일 | 내용 |
|---|---|
| `config.json` | 디펙 타입, 출력 컬럼 이름, 업데이트 확인 위치 |
| `products.json` | WF 제품군별 그리드, bbox 크기, bbox 내부 layout |
| `deploy.json` | (exe 옆) 중앙 공유 폴더 위치 |

### 중앙 공유 폴더에서 받기 (운영 배포)

exe 옆의 `deploy.json`에 공유 폴더를 적습니다. 환경변수 `TIF_VIEWER_SHARED_DIR`로도 지정할 수 있습니다.

```json
{"shared_dir": "\\\\server\\share\\TIF_Viewer"}
```

- **시작할 때:** 공유 폴더의 `config.json`, `products.json`을 사용자 캐시(`%LOCALAPPDATA%\TIF_Viewer\shared_cache`)로 받아서 로드합니다.
- **공유 폴더 접속 불가:** 5초 안에 응답이 없거나 폴더가 없으면 마지막으로 받은 캐시를 씁니다. 캐시도 없으면 exe 옆 파일을 씁니다.
- **실행 중 다시 받기:** 툴바의 "Reload Declarations"를 누르면 다시 받아서 디펙 타입, 출력 컬럼, 제품 목록에 적용합니다.
- **상태 확인:** 하단 상태바에 local / shared / cached copy 중 어느 것을 쓰는지 표시합니다.
- **업데이트 매니페스트:** `config.json`에 `update.manifest`가 없으면 공유 폴더의 `latest.json`을 사용합니다.
- **선언 파일은 프로그램이 쓰지 않습니다:** `config.json`, `products.json`은 사람이 작성합니다. "Types..." 창에서 바꾼 내용은 이번 실행에만 적용됩니다. 영구 변경은 `config.json`을 직접 수정합니다.

`shared_dir`이 비어 있으면(기본값) 로컬 모드입니다. 이때는 실행 위치(작업 폴더)와 상관없이 **exe가 있는 폴더**(소스 실행 시 저장소 루트)의 파일을 읽습니다.

### 디펙 타입 (`config.json` → `void_types`)

타입은 **이름**으로 관리하고, voids.json / CSV에도 타입 이름(`type`)을 저장합니다. 선언 순서가 숫자 키 번호(1, 2, ... 10, 11 ...)가 됩니다.

```json
"void_types": {
    "bbox":         {"color": [255,255,0], "shape": "rectangle", "def_w": 480, "def_h": 480},
    "corner":       {"color": [255,0,255], "shape": "rectangle", "snap": "bbox_corner"},
    "corner_curve": {"color": [0,255,0], "shape": "curve", "def_p": 4.0,
                     "snap": "bbox_corner", "snap_anchor": "center", "snap_offset": "SL"}
}
```

리스트 형식(`[{"name": "bbox", ...}, ...]`)과 이전 번호 키 형식(`"4": {"name": "bbox", ...}`)도 읽습니다. 이전 voids.json의 `type_id`는 파일 안의 타입 이름으로 바꿔 읽습니다.

타입 이름은 **대소문자를 구분하지 않고** 선언된 이름에 맞춥니다. 예를 들어 선언이 `bbox`, `edge`이면 voids.json / API로 불러온 `BBOX`, `Edge`나 `type_rules`, `ui_defaults`의 이름도 `bbox`, `edge`로 처리합니다. 이름이 `bbox`인 타입(대소문자 무관)이 기준 bbox입니다.

| 키 | 의미 |
|---|---|
| `shape` | `ellipse` / `rectangle` / `curve`(초타원 \|x/rx\|^p + \|y/ry\|^p = 1) |
| `def_w`, `def_h` | 0보다 크면 클릭 한 번으로 이 크기의 마킹을 만듭니다. 0이면 드래그로 크기를 정합니다. |
| `def_p` | curve의 기본 p (기본값 4) |
| `snap: "bbox_corner"` | 그리기·이동·크기 조절 중에는 마우스를 그대로 따라가고, 마우스를 놓을 때 셀 bbox의 가장 가까운 코너에 붙습니다(그린 것은 누른 점에 가까운 코너에서 놓은 점까지). |
| `snap_anchor` | `center`(기본값): 도형의 중심이 코너에 오고 bbox 안쪽 **사분면만** 그립니다. 사분면은 bbox 내 좌표로 자동 판정합니다. 그릴 때는 드래그한 영역에 가장 가까운 코너를 고르고, 놓으면 그 코너에서 드래그한 영역의 가장 먼 점까지를 크기로 씁니다(그리는 동안에도 같은 방향의 사분면으로 보임).<br>`edge`: 도형 전체의 모서리가 코너에 붙습니다. |
| `snap_offset` | 코너에서 안쪽으로 옮길 거리입니다. `"SL"`이면 선택한 제품 layout의 SL 값을 쓰고, `[x, y]`이면 픽셀 값을 씁니다. |

타입 이름이 `bbox`(대소문자 무시)인 마킹은 그 칸의 기준 bbox가 됩니다. 기준은 **가장 위(번호가 가장 낮은) 레이어**에 그린 bbox입니다. bbox는 **열어 둔 범위의 첫 레이어(레이어 0)**에서만 호버·수정·삭제·Q·타입 변경 대상이 되고, 다른 레이어에서는 점선으로만 보입니다.

### 출력 컬럼 이름 (`config.json` → `output_columns`)

voids.json의 필드와 voids.csv의 컬럼 이름을 exe를 다시 빌드하지 않고 바꿀 수 있습니다. 키 목록과 기본 이름은 `sat_widgets/output_columns.py`에 선언되어 있습니다. 값을 `null`로 주면 해당 컬럼을 출력하지 않습니다.

```json
"output_columns": {
    "csv":  {"bbox_rel_x": "ValidX", "bbox_rel_y": "ValidY", "radius_y": null},
    "json": {"bbox_rel_x": "validX"}
}
```

`bbox_rel_x`와 `bbox_rel_y`는 디펙 중심을 그 칸의 기준 bbox **좌상단**을 원점으로 한 좌표입니다(그리드 정렬 기준 픽셀). 칸에 bbox가 없으면 빈 값입니다.

### 픽셀 값의 정규화 (`config.json` → `output_scale`)

voids.json·voids.csv·결과 API의 픽셀 값(`globalCX/Y`, `centerX/Y`, `radiusX/Y`, `bboxRelX/Y`)은 **그리드 X 피치를 300으로 맞춘 비율**(값 × 300 / X 피치)로 씁니다. 해상도가 다른 스캔도 같은 기준으로 비교할 수 있습니다. 실제 픽셀값은 같은 이름에 `_raw`를 붙인 필드(`radiusX_raw`, CSV는 `RadiusX_raw` 등)에 함께 씁니다. um 값은 실제 픽셀 × Scale입니다.

```json
"output_scale": {"pitch_x": 300}
```

`null`이면 정규화하지 않습니다(두 필드 모두 실제 픽셀). meta.json에는 쓴 비율이 `"pixel_scale": {"pitch_x": 300, "pitch_x_raw": 실제 X 피치, "scale": 비율}`로 남습니다. 불러올 때는 `_raw` 필드를 먼저 쓰고, 없으면 일반 필드를 기록된 비율로 나눕니다. 비율 기록이 없는 예전 파일은 실제 픽셀로 읽습니다. `_raw` 필드 이름도 `output_columns`로 바꿀 수 있습니다(`radius_x_raw` 등).

### UI 기본값 (`config.json` → `ui_defaults`)

입력 칸의 시작 값을 선언합니다. 프로그램을 시작할 때와 Reload Declarations 때 적용하고, 프로그램은 이 값을 쓰지 않습니다(사용 중 바꾼 값은 저장되지 않음).

```json
"ui_defaults": {
    "reject_values": ["X", "NG"],
    "gpu_cache_mb": 4096,
    "start_layer": 0, "end_layer": 11,
    "show_grid": true, "grid_opacity": 40,
    "um_per_px": 2.0,
    "calib_enabled": false, "calib_robust": true, "calib_all_layers": true,
    "target_mean": 30000, "target_std": 10000,
    "void_type": "edge",
    "export": {"raw": true, "overlay": true, "mask": false, "merged": false, "json": true, "csv": true}
}
```

- 모르는 키는 무시하고 콘솔에 쓸 수 있는 키 목록을 출력합니다.
- `um_per_px`는 제품에 `um_per_px`가 있으면 제품 값이 우선합니다.
- Detect Voids의 방법·Thr 기본값은 `"void_detect"`(`method`: `die_to_die` / `layer_change` / `patchcore`, `threshold`, `lc_threshold`, `pc_threshold`)에 적습니다.

### Wafer ID

WF Product 묶음에 연 파일 이름(**File**)과 **Wafer ID** 칸이 있습니다. 파일을 열면 이름에서 Wafer ID를 자동으로 채우고, 직접 고칠 수도 있습니다. 파싱 규칙은 `config.json`에 정규식으로 선언합니다(없으면 확장자를 뺀 파일 이름).

```json
"wafer_id": {"pattern": "^(?P<lot>[A-Z]+\\d+)_W(?P<wafer>\\d+)", "group": "lot"}
```

**Fetch Map**은 Wafer ID로 다른 서버에서 본딩맵을 가져와 적용합니다(`plugins/results_api.py`의 `fetch_bonding_map`, 주소는 `"results_api": {"bonding_map": {"url": ".../{wafer_id}", "auto_fetch": true}}`). `auto_fetch`면 파일을 열 때 자동으로 가져옵니다. 플러그인에 `parse_wafer_id`를 쓰면 정규식 대신 코드로 파싱할 수 있습니다. Export한 `patches.json`에 Wafer ID가 저장되어 프로젝트를 다시 열 때 복원됩니다. 자세한 내용은 [docs/RESULTS_API.md](docs/RESULTS_API.md)를 보세요.

### 결과 전송 / 불러오기 (Results API)

마킹 결과를 다른 시스템으로 보내거나 다른 시스템에서 불러오는 부분은 직접 작성하는 플러그인 `plugins/results_api.py`로 분리했습니다(Extraction 패널의 **Results API: Preview / Send / Load**). 페이로드 만들기(`build_payload`), 전송(`send`), 요청(`fetch`), 응답 변환(`to_voids`) 네 함수를 고치면 되고, 버튼을 누를 때마다 파일을 다시 읽으므로 재시작·재빌드가 필요 없습니다. 설정은 `config.json`의 `"results_api"`입니다. 자세한 내용은 [docs/RESULTS_API.md](docs/RESULTS_API.md)를 보세요.

### 자동 실행 (Auto Run)

Extraction 아래 **Auto Run** 묶음에 측정 파일 경로(glob, 예: `D:/SAT/raw/**/*.tif`)를 넣고 **Start Auto Run**을 누르면, 새 파일이 다 써질 때마다 창에서 차례로 Wafer ID → 제품(Lot / Wafer ID 규칙) → 본딩맵 → Detect BBOX + Fit Grid → Detect Voids → 저장(`voids.json`, `meta.json`, `summary.json`)까지 하고 검토 요청 메일을 보냅니다(플러그인 `notify_review`). 단계마다 품질 관문이 있어 이상하면 `needs_attention`과 이유를 남기고 다음 파일로 넘어갑니다. 사람은 결과의 점선 마킹만 확인하면 됩니다. 창 없이 `python tools/auto_run.py --watch`로도 돌릴 수 있습니다.

설정은 `config.json`의 `"auto_run"`(감시 경로, 제품 규칙, 품질 기준)과 `"results_api": {"notify"}`(메일 서버)입니다. 자세한 내용은 [docs/AUTO_RUN.md](docs/AUTO_RUN.md)를 보세요.

### WF 제품군 (`products.json`)

```json
{
  "default": "WF_A",
  "products": {
    "WF_A": {
      "grid": {"rows": 10, "cols": 12, "pitch_x": 500, "pitch_y": 500},
      "um_per_px": 2.0,
      "chip_size_um": [960, 960],
      "layout": {
        "SL": [40, 40],
        "style": {"color": [0,200,255], "alpha": 200, "width": 1, "dash": [4, 4]},
        "boxes": [
          {"name": "die", "inset": "SL"},
          {"name": "pad", "rect": [80, 80, 160, 80], "repeat": {"cols": 4, "rows": 1, "dx": 200, "dy": 0}}
        ]
      }
    }
  }
}
```

칩 크기와 layout은 **실제 칩 치수(um)**로 선언합니다.

- **`grid`:** 적은 키만 적용합니다. 쓸 수 있는 키는 `rows`, `cols`, `pitch_x`, `pitch_y`, `start_x`, `start_y`, `angle`입니다(이미지 픽셀).
- **`um_per_px`:** 1픽셀이 몇 um인지입니다. 생략할 수 있고, 적으면 제품을 고를 때 Grid 패널의 **Scale** 칸에 들어갑니다. Scale은 화면에서 직접 바꿀 수도 있습니다.
- **`chip_size_um`:** 실제 칩(bbox) 크기(um)입니다. `bbox` 타입의 기본 크기는 `chip_size_um / Scale` 픽셀로 정해지고, Scale을 바꾸면 따라 바뀝니다. Scale이 비어 있으면 bbox는 드래그로 그리고, Detect BBOX는 검출한 크기를 씁니다.
- **`layout`:** 각 칸의 기준 bbox 안에 점선으로 그리는 세부 박스입니다.
  - 좌표는 칩 좌상단 기준 um입니다. 실제로 그린 bbox 크기에 맞춰 늘거나 줄어들므로, Scale이 조금 틀려도 bbox를 따라갑니다.
  - `SL`은 스크라이브 라인 폭(x, y, um)입니다. `snap_offset: "SL"`인 타입은 bbox 코너에서 이만큼 안쪽 지점을 기준으로 그립니다.
  - 박스는 `"inset": "SL"`(bbox를 SL만큼 안쪽으로 줄인 영역), `"rect": [x, y, w, h]`, `"repeat"`(배열 반복)로 정의합니다.
- 이전 형식(`"bbox": {"w", "h"}` 픽셀 + `layout.design_size`)도 그대로 읽습니다.
- bbox 크기는 **제품마다** 다르게 적습니다(`chip_size_um` 또는 `bbox`). 제품을 바꾸면 bbox 타입의 기본 크기(그리기, Detect BBOX)가 그 제품 값으로 바뀌고, 둘 다 없는 제품은 `config.json`의 bbox 타입 크기로 돌아갑니다.
- bbox 크기는 **항상 선택한 제품 크기**를 따릅니다. 이미 그린 bbox도 제품·Scale을 바꾸면 중심은 그대로 두고 크기만 바뀌고, voids.json·프로젝트·API에서 불러온 bbox도 저장된 크기 대신 **중심 좌표만** 쓰고 제품 크기로 그립니다. bbox는 손으로 크기를 조절할 수 없고(가장자리를 잡아도 이동), 크기가 정해지지 않은 경우(Scale 없음, 크기 선언 없음)에만 처음 그릴 때 드래그로 크기를 정합니다.
- bbox 코너에 붙은 마킹(`snap: "bbox_corner"`)은 bbox를 따라갑니다. bbox를 옮기면 모든 레이어의 그 bbox 코너 마킹이 같이 움직이고(되돌리기 한 번), bbox 크기가 바뀌거나(제품·Scale·불러오기) Detect BBOX로 다시 그려지면 가장 가까운 코너에 다시 붙습니다.

### 축척 (Scale)

Grid 패널의 **Scale (um/px)**은 1픽셀의 실제 크기입니다. 값이 있으면 voids.json / voids.csv에 um 컬럼(`RelativeX_um`, `RelativeY_um`, `RadiusX_um`, `RadiusY_um`, `BBoxRelX_um`, `BBoxRelY_um`)을 채웁니다. 이 컬럼 이름도 `output_columns`로 바꿀 수 있습니다. meta.json과 patches.json에 `um_per_px`를 함께 저장하고, 그 voids.json을 불러오면 Scale 칸에 값을 다시 넣습니다.

### 업데이트 알림

`config.json`에 최신 버전 정보를 담은 매니페스트 위치(공유 폴더 경로나 http(s) URL)를 적습니다.

```json
"update": {"manifest": "\\\\server\\share\\TIF_Viewer\\latest.json"}
```

매니페스트 예시입니다.

```json
{"version": "1.1.0", "download": "\\\\server\\share\\TIF_Viewer\\1.1.0", "notes": "변경 사항"}
```

시작할 때 백그라운드에서 버전을 비교합니다. 실행 중인 버전(`APP_VERSION`)이 낮으면 알림 창을 띄우고, "Update"를 누르면 `download` 위치(폴더, 파일, URL)를 엽니다. `manifest`가 비어 있으면 확인하지 않습니다.

## 그리드 각도 맞추기 (Angle Preview)

Grid Configuration 묶음 아래쪽의 작은 Angle Preview(항상 표시)가 웨이퍼 곳곳을 회전 보정된 패치로 2×2로 보여줍니다. 각 패치는 **그리드 선의 교차점**(칩 네 개의 모서리가 만나는 곳)을 가운데에 두고, 그 교차점을 빨간 십자로 표시합니다. 그리드 선, 기준 bbox, layout도 겹쳐 그려집니다.

- **판단 방법:** 각도가 맞으면 모든 패치에서 스트리트의 교차가 빨간 십자에 오고 노란 그리드 선이 스트리트 중앙을 지납니다. 틀리면 중심에서 먼 패치일수록 어긋납니다.
- **갱신과 이동:** 각도, 원점, 피치, 레이어, 밝기가 바뀌면 바로 다시 그립니다. 패치를 클릭하면 메인 화면이 그 칸으로 이동합니다.
- **지점 위치:** 원형 웨이퍼 기준의 극좌표(중심 = 그리드 중심, R = 그리드에 내접하는 원의 반지름)로, `config.json`의 `angle_preview`에서 바꿉니다. 기본값은 센터, 위 0.75R, 오른쪽 0.75R, 오른쪽 위(45°) 0.5R입니다. 위치는 본딩맵과 관계없이 전체 그리드 기준이며, 그 점에 가장 가까운 그리드 교차점을 씁니다.

```json
"angle_preview": {"points": [{"name": "Center", "angle": 0, "r": 0.0},
                             {"name": "Top 3/4", "angle": 90, "r": 0.75}], "view_scale": 1.4}
```

## 본딩맵

- 엑셀에서 복사한 맵을 붙여넣고 "Import Map Data"를 누릅니다.
- **Open Project:** Export 폴더에 함께 저장된 `bonding_map.txt`와 Reject Values(`bonding_map_reject.txt`)도 다시 불러옵니다. Export 폴더(patches.json + raw 패치 + voids.json)를 다시 열면 패치를 원본 이미지와 같은 좌표계(같은 크기, 같은 회전 중심)로 다시 맞춰 붙입니다. 그래서 voids.json의 마킹과 그리드가 그대로 맞습니다. patches.json에 원본 크기(`image_size`)를 저장하고, 이전 Export는 패치 좌표에서 원본 크기를 계산합니다.
- overlay 패치에는 마킹과 함께 제품 layout(점선)도 그립니다. PNG는 가벼운 압축(파일 약 15% 증가)으로 여러 스레드에서 저장해, 이전보다 약 8배 빠릅니다.
- Export 이미지(raw, overlay, mask, merged)는 **LEG(본딩맵 값)가 있는 칸만** 저장합니다. 본딩맵이 없으면 이미지 Export를 시작하지 않습니다. 붙여넣은 맵은 `bonding_map.txt`(그대로 다시 붙여넣을 수 있음), Reject Values는 `bonding_map_reject.txt`로 함께 저장합니다.
- **Reject Values:** 쉼표로 구분한 값(예: `X, NG`)과 일치하는 칸은 본딩되지 않은 칸으로 처리합니다. ROI, Export, 칸 이동, 캘리브레이션, 각도 미리보기에서 모두 빠집니다. 값을 바꾸고 Enter를 누르면 바로 다시 적용합니다.
- 화면 좌상단에 화면 중심 칸의 `X`, `Y`, `LEG`(본딩맵 값)를 노란 글씨로 표시합니다.

## 조작

오른쪽 패널 순서: WF Product → Grid → Load Configuration → Bonding Map → Auto Calibration(Detect BBOX 포함, Robust Mode·Apply to All Layers 기본 켜짐) → Coordinates → Void Marking → Extraction.

**Auto Calibration**은 본딩맵 값이 있는 칸만 계산합니다(값이 없거나 Reject된 칸은 보정 없음). 레이어마다 칸 밝기 통계를 내야 하는데, 백그라운드 레이어 로딩이 레이어를 읽을 때 5픽셀 간격 표본(8-bit 14900² 기준 레이어당 약 9 MB)을 메모리에 남겨 두므로, 그 레이어는 다시 디코딩하지 않고 계산합니다(14900 샘플 6레이어: 9.7초 → 1.9초, 값은 같음). 아직 표본이 없는 레이어만 파일에서 읽습니다.

| 동작 | 키 / 마우스 |
|---|---|
| 그리기 / 수정 / 삭제 모드 | W / E / R을 **누르고 있는 동안만**. 떼면 보기 모드. 여러 개를 누르면 마지막 키. 드래그 중에 떼면 드래그가 끝날 때 적용 |
| 보기 모드 (마킹 도구 끔) | 모드 키를 뗌 (Esc도 가능) |
| 호버 | 모든 모드에서 마우스 아래 마킹이 반투명해집니다. 겹치면 경계가 커서에 더 가까운 마킹이 선택되고, bbox는 다른 마킹이 없을 때만 선택됩니다. |
| 마우스 아래의 다른 레이어 마킹(점선)을 현재 레이어로 옮기기 | Q (텍스트 입력 중이 아니면 어느 위젯에 포커스가 있어도 동작) |
| 화면 이동 | 오른쪽 또는 휠 버튼 드래그 (모든 모드), 보기 모드에서는 왼쪽 드래그 |
| 다음 / 이전 칸 | **오른쪽 클릭**(드래그 없이): 본딩맵 값이 있는 다음 칸(왼쪽→오른쪽, 줄 끝이면 다음 줄, 마지막이면 처음으로). **Shift + 오른쪽 클릭**: 이전 칸. 본딩맵이 없으면 모든 칸 |
| 크기 조절 | 수정 모드에서 마킹 가장자리를 드래그. 중심은 고정되고 잡은 쪽이 마우스를 따라갑니다(코너에 붙는 타입은 붙은 코너 고정). 모서리 근처를 잡으면 가로세로를 함께 조절합니다. |
| 다른 레이어 마킹(점선) | 수동·자동 구분 없이 현재 레이어 마킹과 똑같이 E로 이동·크기 조절, R로 지웁니다(현재 레이어에 잡히는 마킹이 없을 때, 가까운 레이어부터). 마킹은 원래 레이어에 남고, 레이어를 옮기려면 Q. |
| 디펙 타입 | 숫자 = 타입 번호(선언 순서, 0 = 10). 0.4초 안에 이어 누르면 두 자리(1 → 2 = 12). 마킹 위에 호버 중이면 그 마킹의 타입을 바꾸고(번호 하나가 되돌리기 한 단계), 아니면 새로 그릴 타입을 고릅니다. 텍스트 입력 중에는 동작하지 않습니다. |
| 마킹 변경 되돌리기 | Ctrl+Z (추가, 이동, 크기 조절, 지우기, 레이어 이동(Q), 레이어·칩·전체 지우기(Clear All), JSON 불러오기, p 조절, Detect BBOX) |
| curve의 p 조절 | 수정 모드(E)에서 curve 위에 마우스를 올리고 휠 (한 칸에 0.02) |
| curve·코너 붙는 타입 | 마우스를 올리거나 그리는 동안 점선 박스가 보이고, 잡기·크기 조절은 그 박스 기준 (곡선은 그대로 그림) |
| 레이어 이동 | Ctrl/Shift + 휠, ↑/↓ |
| 확대·축소 | 휠 |

커서는 모드 키를 누르고 있을 때만 바뀝니다. 수정 모드의 경계 커서는 히스테리시스가 있어 경계에서 떨리지 않습니다.

| 모드 | 커서 |
|---|---|
| 보기 | 기본 화살표 (바뀌지 않음) |
| 그리기 | 선택한 타입 색의 십자선 |
| 수정 | 마킹 가운데: 이동 커서 / 가장자리: 크기 조절 방향 화살표(↔ ↕ 대각선) / 빈 곳: 기본 화살표 |
| 삭제 | 빨간 원에 X |

레이어 번호는 **절대 번호**(파일 안의 페이지 번호)로 저장합니다. `layer_numbering`이 없는 이전 voids.json은 "현재 연 범위의 시작 레이어 기준 상대 번호"로 해석합니다.

**저장 파일:** `voids.json`에는 **마킹 결과만**(보이드 목록) 저장하고, 나머지는 같은 폴더의 `meta.json`에 저장합니다. meta.json에는 형식 버전, 레이어 번호 방식, Scale(`um_per_px`), 픽셀 환산 비율(`pixel_scale`), 그리드, 타입 선언, 연 파일, Wafer ID, 제품, 불러온 레이어, 저장 시각, 프로그램 버전이 들어갑니다. Save JSON에서 다른 이름(예: `run1.json`)으로 저장하면 `run1_meta.json`이 됩니다. 불러올 때는 옆의 meta 파일을 함께 읽습니다. meta 파일이 없으면 마킹은 그대로 읽지만 파일에 있던 타입 선언과 Scale은 복원하지 않습니다. 예전 한 파일 형식(voids.json 안에 types 등)도 그대로 읽습니다.

## 자동 보이드 인식 (Detect Voids)

Void Marking 패널의 **Detect Voids**는 대략적인 보이드를 자동으로 찾아 마킹합니다. 칩 안의 배경은 layout 때문에 균일하지 않지만 칩마다 반복되므로, 평평한 배경 대신 **같은 레이어의 다른 칩들과 비교**합니다(die-to-die).

1. 본딩된 칸마다 기준 bbox(없으면 칸)에 맞춰 칩을 잘라 작업 크기로 줄입니다.
2. 레이어마다 칩들의 화소별 중앙값으로 "기준 칩"을 만들고, 칩 사이 흩어짐(MAD)을 화소별로 구합니다. layout 경계처럼 원래 흔들리는 곳은 흩어짐이 커서 잘못 잡히지 않습니다.
3. 칩마다 기준 칩에 맞춥니다. 칩마다 밝기·대비·감마가 다르고 칩 안에서도 밝기가 기울어 있을 수 있으므로:
   - 기준 칩의 구조(다이, 패드, 선, 범프...)별 밝기 단계마다 칩의 중앙값을 기준 값에 대응시키는 밝기 곡선,
   - 칩 전체에 걸친 평면형 밝기·대비 기울기(결함 화소는 제외하고 반복 맞춤)를 함께 맞춥니다.
   - 기준 칩의 3×3 최소~최대 범위 안이면 정상으로 봅니다(1 px 어긋남, 경계 모양이 조금 다른 것 허용).
   그다음 차이가 Thr(흩어짐 배수)를 넘는 곳을 찾습니다. **밝은 쪽과 어두운 쪽 모두** 잡습니다. layout 선에 잘려 둘로 나뉜 덩어리는 하나로 합칩니다.
   - **칩 맞추기(`register`, 기본 켜짐):** 비교 전에 칩마다 위치를 기준 칩에 0.1 px 단위로 맞춥니다(위상 상관). bbox가 칩마다 몇 px 어긋나 있어도 됩니다.
   - **레이아웃 맞춤(`layout_fit`, 기본 켜짐):** 칩마다 레이아웃의 대비·선명도(초점)가 다르고 칩 안에서도 달라지는 것을, golden과 흐리게 한 golden의 조합으로 맞춰 뺍니다. 보이드는 golden에 없고 국부적이라 남습니다.
   - **평균화(`smooth`, 기본 1.5 작업 픽셀):** 차이를 주변 픽셀과 평균한 뒤, 그 평균값의 칩 간 흩어짐과 비교합니다. 화소 하나하나로는 잡음에 묻히는 **크고 흐린 보이드**도 잡힙니다. `0`이면 화소 단위(윤곽이 더 정확하지만 흐린 보이드를 놓침).
4. 여러 레이어에서 잡힌 같은 자리는 **한 번만** 마킹합니다. 어느 레이어에 마킹할지는 **Mark on layer**(처음 / 가장 강한 곳 / 마지막)로 정합니다. 이미 마킹된 곳(어느 레이어든)은 건너뜁니다.

- **타입 자동 구분:** layout 위치와 모양으로 타입을 정하는 규칙을 선언할 수 있습니다(아래). 규칙이 없거나 맞는 규칙이 없으면 Void Marking에서 선택한 타입(bbox 제외)으로 마킹합니다.
- 찾은 보이드는 점선으로 그려지고, **Next Detected**로 레이어·칸 순서대로 돌아볼 수 있습니다. 수정(E)하면 실선이 됩니다. 전체 실행은 Ctrl+Z 한 번으로 되돌립니다.
- **백그라운드 실행:** 검출은 뒤에서 돌고, 그동안 이동·확대·레이어 변경은 그대로 됩니다. 찾은 보이드는 레이어 하나가 끝날 때마다 화면에 바로 늘어납니다(Die-to-die·PatchCore·U-Net, Mark on layer가 First일 때. Strongest·Last와 Layer change는 모든 레이어를 본 뒤에 정해지므로 끝날 때 한꺼번에). 진행률과 **Cancel**은 아래 상태 표시줄에 있습니다. Cancel하면 그때까지 찾은 보이드는 남고 Ctrl+Z 한 번으로 지워집니다. 실행 중에는 마킹 편집(그리기·수정·지우기, Clear, JSON 불러오기, Ctrl+Z, Detect BBOX 등)과 두 번째 실행이 막힙니다. 다른 파일을 열면 실행은 멈춥니다.
- **칸 상태 표시:** 실행 중과 끝난 뒤(다음 실행이나 파일 변경 전까지) 검사하는 칸마다 오른쪽 위에 작은 원이 그려집니다. 빈 원 = 대기, 채워지는 원 = 진행 중(처리한 레이어 비율), 초록 원에 체크 = 그 칸 완료. 그리드가 보일 때만, 칸이 화면에서 충분히 클 때만 그립니다. Void Marking 패널의 **Cell status**로 끌 수 있습니다.
- 민감도는 Thr 칸(기본 6, 낮을수록 민감)으로 조절합니다. 나머지 값은 `config.json`의 `"void_detect"`로 바꿀 수 있습니다: `threshold`, `smooth`, `skip_layers`, `min_size_px`(가장 작은 보이드 지름), `max_area_frac`, `work_size`, `template_cells`, `layer_rule`.
- 기준 bbox를 먼저 그리면(Detect BBOX) 칩 정렬이 정확해집니다.
- **Skip layers**: 검출하지 않을 레이어(절대 번호, 쉼표로 구분). 기본은 `0`(맨 위 레이어는 표면 반사로 잡음이 많음). 기본값은 `config.json`의 `"void_detect": {"skip_layers": [0]}` 또는 `"ui_defaults": {"skip_layers": [0, 1]}`.
- **Mark on layer**: 여러 레이어에서 잡힌 같은 자리를 어느 레이어에 마킹할지 고릅니다. First(위에서 처음 나온 레이어, 기본), Strongest(편차가 가장 큰 레이어), Last(가장 깊은 레이어). 기본값은 `config.json`의 `"void_detect": {"layer_rule": "last"}` 또는 `"ui_defaults": {"layer_rule": "last"}`.
- **학습 필터(Learned filter / Train)**: 보이드와 칩마다 다른 레이아웃 차이가 비슷한 세기라 임계값으로 못 가를 때 씁니다. 파일을 열고 손 마킹(또는 검토한 voids.json)을 불러온 뒤 **Train**을 누르면, 그 파일의 후보(편차 지도의 국부 최대점)를 손 마킹으로 라벨링해 학습 데이터에 더하고 전체로 다시 학습합니다(파일마다 한 묶음, 같은 파일을 다시 학습하면 교체). 결과 창에 칩 단위 교차검증 성능이 짧은 코드(`L1 f.. v.. n.. c.. h..` / `R.. P..`)로 나옵니다. **Learned filter**를 켜면 Detect Voids·Evaluate가 임계값 대신 학습한 모델로 후보를 고릅니다(Die-to-die). 학습 데이터와 모델은 `사용자 폴더\TIF_Viewer_train\`(`dataset.jsonl`, `void_classifier.json`), 공유 모델은 `config.json`의 `"void_detect": {"classifier": 경로}`. 이미 마킹된 예전 파일도 열고 불러와 Train하면 쓸 수 있습니다(좌표가 조금 어긋나도 되도록 여유를 두고 매칭). 마킹이 칩과 크게 어긋난 파일은 빼세요.
  - **Layer change에서도** 같은 방식으로 씁니다(방법 상자에서 Layer change를 고른 뒤 Train). 모델과 학습 데이터는 따로입니다(`lc_classifier.json`, `lc_dataset.jsonl`, 공유 모델은 `"void_detect": {"lc_classifier": 경로}`). 후보는 자리 점수 8(`lc_cand_threshold`)부터 뽑고, 후보마다 레이어에 걸친 증거의 모양(처음 레이어 이후 유지되는지, 바로 위 레이어에 이미 있었는지), 같은 자리의 다른 칩들과 비교한 세기, 칩 가장자리·칩 안 위치, 덩어리인지 가는 선인지 등 19개 특징으로 판정합니다. 이 필터는 **자리만** 판정하고(손 마킹이 다른 레이어에 있어도 그 자리면 보이드로 학습), 마킹 레이어는 Mark on layer 규칙으로 정합니다. 모델의 기준값은 **웨이퍼당 과검 1개**가 되도록 정합니다: 학습 파일이 3개 이상이면 파일 단위 교차검증(각 파일을 그 파일을 안 본 모델로 점수)에서 파일당 과검이 `"lc_fp_per_wafer"`(기본 1) 이하가 되는 가장 낮은 기준값. Train 결과 창의 `F..`가 그 교차검증 과검 수(파일당)입니다. 기준값을 직접 정하려면 `"lc_learned_cut"`.
  - **제품별 모델:** Layer change 학습 필터는 선택한 제품(WF Product)마다 따로 학습·적용됩니다. 제품을 바꾸면 그 제품의 모델을 씁니다(없으면 Learned filter가 꺼짐). 위치는 `사용자 폴더\TIF_Viewer_train\products\<제품명>\`(`lc_classifier.json`, `lc_dataset.jsonl`). 공유 모델은 products.json의 그 제품에 `"lc_classifier": 경로`, 또는 config.json `"void_detect": {"lc_classifier": "\\\\server\\models\\{product}\\lc_classifier.json"}`처럼 `{product}`(제품명)를 넣은 경로 하나로 씁니다. 제품을 고르지 않았으면 공용 `lc_classifier.json` 하나.
  - 모델은 **그래디언트 부스팅 트리**(numpy, `train_trees`: 깊이 4, 400회)라 특징을 조합해서 봅니다(예: 칩 가장자리이면서 한 레이어만 튐). 학습은 파일 수천 후보에 수 초.
  - 합성 8레이어 웨이퍼 13장으로 학습(웨이퍼 단위 교차검증), 같은 과검에서의 위치 미검: 웨이퍼당 과검 1개 → 52%(학습 필터 없음) / 15%(결정 그루터기) / **10%(트리)**, 3개 → 38% / 10% / **9%**. 시험 웨이퍼 ml_204(과검 1개 기준값): 위치 미검 6%, 과검 0. 레이어까지 맞추는 기준으로는 미검 22~24%(자리는 맞고 레이어가 틀린 것이 남은 약점). 학습 웨이퍼는 7장쯤부터 더 늘려도 거의 같았습니다.
- **U-Net 세그멘테이션 (Seg Data / Segmentation U-Net)**: 손 마킹으로 작은 U-Net을 학습해 보이드 확률 지도를 편차 지도 대신 씁니다. **Seg Data** 창에서 연 파일 또는 저장 결과 폴더(voids.json + meta.json)로 학습 데이터를 만들고 칩 썸네일을 보며 넣을 것/뺄 것을 고른 뒤, PyTorch가 있는 PC에서 `tools/train_seg.py`로 학습해 `models/void_unet.onnx`를 만들면 방법 상자의 **Segmentation U-Net**이 켜집니다. 자세한 내용은 [docs/SEGMENTATION.md](docs/SEGMENTATION.md).
- **Evaluate**: 검출을 마킹 없이 돌려 손 마킹과 비교합니다. 놓친 것·잘못 잡은 것·크기 차이를 원인별로 휴대폰으로 입력해 보낼 수 있는 짧은 코드로 보여 주고, 칩 이미지가 든 보고서 폴더를 만듭니다. 코드 읽는 법은 [docs/VOID_EVAL.md](docs/VOID_EVAL.md).
- **Layer change (방법 상자):** 불러온 **모든 레이어를 한 번에** 봅니다. 같은 칩은 레이어가 바뀌어도 구조가 같고(레이어마다 구조별 밝기·대비·극성만 다름), 보이드는 어느 레이어에서 생기면 그 아래 레이어에서 계속 보인다는 점을 씁니다.
  1. 칩의 레이어 k를 **같은 칩의 위 레이어들**(보이드가 처음 생기는 레이어라면 그 위는 보이드가 없음)과 **레이어별 기준 칩**으로 설명합니다. 강건 최소제곱이라 보이드가 맞춤을 끌어당기지 않고, 칩 안에서 계수가 부드럽게 변하며, 레이아웃 영역(패드·다이...)마다 계수를 따로 둡니다.
  2. 설명하고 남은 것을 칩 간 흩어짐으로 나눈 뒤 여러 크기의 원형 필터로 보고, 레이어마다 증거를 구합니다.
  3. 자리 점수는 **보이드가 생긴 레이어부터 마지막 레이어까지의 증거 합**(레이어마다 기준 수준 `lc_tlo`를 뺀 값, CUSUM)이고, 이것이 Thr(기본 26)을 넘는 자리를 찾습니다. 늦게 생긴 보이드도 일찍 생긴 것과 같은 조건으로 평가되고, 한 레이어에만 튀는 것은 한 번만 더해집니다. 마지막 레이어 아래에는 **마지막 레이어의 증거를 가우시안(σ `lc_pad_sigma`, 기본 5 작업 픽셀)으로 흐린 가상 레이어**를 붙입니다. 보이드는 덩어리라 흐려도 증거가 남고, 칩 가장자리·패드의 가는 찌꺼기는 흐려져 약해지므로, 마지막 레이어에서 처음 생긴 보이드도 한 레이어짜리 잡음과 구별됩니다(합성 데이터, 과검 0 기준: 이런 보이드 33개 중 2개 → 8개). (예전 방식인 레이어 평균 `lc_place: "rms"`는 늦게 생긴 보이드가 평균에 묻혀, 같은 미검에서 과검이 두 배였습니다.)
  4. **Mark on layer = First**면 증거가 계속 쌓이기 시작하는 레이어(CUSUM 변화점)에 마킹합니다. 크기는 그 레이어에서 잰 크기입니다.
  - 비교할 레이어를 모두 불러온 뒤 쓰세요. Skip layers에 넣은 레이어는 검출·마킹은 하지 않지만 아래 레이어를 설명하는 데는 씁니다.
  - 합성 8레이어 웨이퍼(`tools/synth`, 칩마다 레이아웃 대비·극성·흐림이 다르고 보이드가 레이어마다 크기·명암이 바뀜), Thr 26: 학습용 3장(보이드 198개)에서 위치 미검 32%, 웨이퍼당 과검 4.0개 / 시험용 ml_204(50개)에서 위치 미검 24%, 과검 2개(Die-to-die: 미검 94%, 과검 443개). 14×14칩 8레이어에 약 2분(4코어).
  - Thr과 미검·과검(학습용 3장, 위치 기준): Thr 16 → 미검 14%, 과검 16개/웨이퍼 · 20 → 25%, 9.0개 · 26 → 32%, 4.0개 · 30 → 41%, 2.3개 · 35 → 48%, 0.7개. 남은 미검의 대부분은 **마지막 레이어에서 처음 생긴 보이드**(아래 레이어가 없어 계속 보이는지 확인할 수 없음)이고, 과검은 칩 가장자리·패드 근처에서 한 레이어만 튀는 것이 많습니다.
  - **레이어 0에서 이미 보이는 보이드(기포 의심):** 레이어 0은 잡음이 많아 검출하지 않지만(Skip layers 0), 찾은 보이드마다 **레이어 0에서도 보이는지** 판정합니다. 위 레이어와 비교하지 않고, 그 자리 주변(고리 모양 영역)을 레이어별 기준 칩으로 국소 맞춤한 뒤 가운데가 얼마나 벗어나는지(가운데 평균 / 고리의 흩어짐, `lc_layer0_z` 기본 1.0 이상이면 보임)로 봅니다. 처리(`lc_layer0_check`): `"flag"`(기본: 마킹하고 리뷰용 표시, `lc_layer0_type`에 선언된 타입 이름을 적으면 그 타입으로 마킹), `"reject"`(마킹 안 함), `"off"`. 합성 웨이퍼 2장: 레이어 0부터 있던 보이드 20개 중 15개 표시, 더 깊은 레이어에서 생긴 보이드는 74개 중 3개만 잘못 표시. 레이어 0이 불러온 범위에 있어야 합니다.
  - **칩 자르기 정밀도:** 칩을 원본 해상도에서 bicubic으로 잘라 줄이고, 칩마다 실제 위치를 0.01 px까지 재서 그 자리로 다시 자릅니다. bbox가 칩마다 1 px 안쪽으로 어긋나도(실제 데이터, Detect BBOX) 과검이 늘지 않습니다(합성: 0.7 px 어긋남에서 같은 미검일 때 과검 16 → 4).
  - **작은 보이드:** 학습 필터를 작은 보이드가 있는 파일로도 Train하면 좋아집니다(합성 반지름 3~8 px: 미검 44% → 33%, 보통 보이드는 그대로).
  - 설정(`config.json`의 `"void_detect"`): `lc_threshold`, `lc_place`, `lc_pad_sigma`, `lc_layer0_check`, `lc_layer0_z`, `lc_layer0_type`, `lc_tlo`(보이드 없는 레이어의 증거 수준, 첫 레이어 판정), `lc_scales`, `lc_sources`(위 레이어 몇 장까지), `lc_goldens`, `lc_regions`, `lc_register`(레이어끼리 위치 맞춤, 기본 꺼짐: 한 스캔의 레이어는 정렬되어 있음). 실험 기록: 브랜치 `exp/C-layer-change`.
- 방법 상자에서 **PatchCore**(CNN 특징 기반, CPU, 선택 설치)를 고를 수도 있습니다. 검토 결과와 운영 셋업(오프라인 설치, 빌드, 설정, 임계값 보정)은 [docs/PATCHCORE.md](docs/PATCHCORE.md)를 보세요. 이 데이터에서는 기본 방식(Die-to-die)이 더 정확하고 빠릅니다.
- 14900×14900 샘플(841칩)에서 레이어당 약 16초, 생성기의 정답(보이드 92개, 코너 박리 51개)을 모두 찾았고 모두 처음 나타난 레이어에 마킹했습니다(추가 검출 5). 칩마다 밝기·대비·감마·기울기를 크게 다르게 만든 합성 데이터에서도 98%를 찾았습니다.

### 자동 마킹 타입 규칙 (`type_rules`)

선택한 제품의 `products.json`에 `"type_rules"`를 적으면 그 제품에서 쓰고, 없으면 `config.json`의 `"void_detect"` → `"type_rules"`를 씁니다. 위에서부터 **처음으로 조건이 모두 맞는 규칙**의 타입으로 마킹합니다. 타입 이름은 선언한 타입 이름이고, 코드에 고정된 타입은 없습니다.

```json
"type_rules": [
    {"type": "corner_delam", "near": {"box": "die", "part": "corner", "within": 15}, "polarity": "bright"},
    {"type": "pad_void", "in": "pad", "max_size_um": 40},
    {"type": "long_void", "min_aspect": 3},
    {"type": "die_void", "in": "die"}
]
```

| 조건 | 의미 |
|---|---|
| `in` / `not_in` | 덩어리 중심이 그 이름의 layout 박스 안 / 밖(이름 하나 또는 목록). `bbox`는 기준 bbox 전체 |
| `overlaps` | 덩어리 범위가 그 박스와 겹침 |
| `near` | `{"box": 이름, "part": "corner" 또는 "edge", "within": px}`: 그 박스의 코너 / 테두리에서 이 거리 안 (`within_um`도 가능) |
| `min_size` / `max_size` | 지름(범위의 긴 변), px. `_um`을 붙이면 Scale 기준 um |
| `min_aspect` / `max_aspect` | 긴 변 / 짧은 변 |
| `min_fill` / `max_fill` | 덩어리 화소 / 범위 넓이 (타원 약 0.79, 사각형 약 1, 불규칙·가는 곡선은 낮음) |
| `polarity` | `"bright"` / `"dark"`: 기준 칩보다 밝음 / 어두움 (Die-to-die만) |

- layout 박스 좌표는 점선 layout처럼 실제 그린 bbox 크기에 맞춰 계산합니다.
- 규칙의 타입이 `snap: "bbox_corner"`이면 가까운 bbox 코너(snap_offset 반영)에 붙이고, 크기는 덩어리 범위의 먼 코너까지로 맞춥니다(손으로 코너에서 그린 것과 같음).
- 선언에 없는 타입 이름의 규칙은 건너뜁니다. 상태줄에 타입별 개수가 나옵니다.
- 14900 샘플(2레이어)에서 다이 안 보이드 46개와 코너 박리 26개를 모두 맞는 타입으로 구분했고, 코너 박리의 크기 오차는 중앙값 1.1 px였습니다.

## 자동 BBOX (Detect BBOX)

Detect BBOX를 누르면 기존 bbox(모든 칸, 모든 레이어)를 모두 지우고 새로 그립니다.

**기본(레이아웃 기준, `auto_bbox.method: "layout"`)**: 패키지 경계가 흐리거나 노이즈가 많아도(한쪽이 튀어나와 커 보이는 칩 등) 맞도록, 경계 하나가 아니라 **칩 레이아웃 전체**로 맞춥니다. 모든 칩의 레이아웃이 같다는 점을 씁니다.

1. 본딩맵의 유효 칸마다 칸 중심(기존 bbox가 있으면 그 칸 안 평균 위치)에서 시작합니다. 원점·각도가 조금 틀려도 됩니다.
2. 불러온 레이어 몇 장(`refine_layers`, 기본 4)의 그래디언트 크기(대비·극성 무관)를 칩마다 더하고, 칩들의 중앙값으로 **기준 칩**을 만들어 각 칩을 상관으로 0.1 px 단위까지 맞춥니다(기준 칩은 맞춘 칩들로 3번 다시 만듦).
   - 1차(넓게, bbox 크기의 ±25% `coarse_margin`): 칩 전체(외곽 경계 포함)로 맞춰, 반복되는 레이아웃(범프 배열 등)에서 한 주기 어긋나지 않게 합니다.
   - 2차(좁게, ±8% `refine_margin`): 칩 **안쪽만**(외곽 10% `refine_inner` 제외) 써서, 튀어나온 경계가 위치를 끌지 않게 합니다.
3. **절대 위치:** 손으로 놓거나 고친 bbox가 있으면 그 위치, 없으면 기준 칩의 패키지 경계(칩 수백 개의 중앙값이라 개별 칩의 경계 노이즈가 사라짐)의 중심.
4. 맞춤이 나쁜 칩(`refine_min_ncc`)이나 다른 칩들의 격자에서 벗어난 칩은 격자에서 예측한 위치에 **점선(추정)** 으로 놓습니다. 유효 칸의 30% 이상이 안 맞으면 원점이 한 칸 밀렸을 수 있다고 경고합니다(본딩맵이 다른 칩에 붙음).
5. 합성 웨이퍼(14×14칩, 칩 40%의 한쪽 경계가 4~15 px 튀어나옴, 선언 격자 각도 0.25°/원점 12 px 어긋남): bbox 오차 최대 2.0 px(경계 방식 20.7 px), 각도 0.397°(실제 0.4°). 196칩 약 6초.

- **Refine BBOX:** 지금 bbox들을 다시 레이아웃으로 맞춥니다. **이번에 손으로 놓거나 옮긴 bbox는 그대로 두고** 나머지를 그 위치 기준으로 맞춥니다(몇 개만 손으로 고쳐 기준을 잡는 용도).
- **Fit Grid:** bbox 중심들로 격자 원점·피치·각도를 맞춰, 지금 값과 비교해 보여 주고 확인하면 적용합니다(products.json은 바꾸지 않음). bbox가 칸 안에서 놓인 평균 위치(선언된 원점 규칙)는 유지합니다. 각도가 장비마다 조금 다를 때 씁니다. 순서: Detect BBOX → Fit Grid → (필요하면) Refine BBOX.
- 예전 방식(`"method": "edge"`)은 아래 경계 검출이고, `layout_refine`(기본 켜짐)이면 이어서 레이아웃으로 맞춥니다.

**경계 방식(`"method": "edge"`)**: Auto Calibration 패널의 "Detect BBOX"를 누르면 최상단 레이어에서 본딩된 칸마다 다음을 수행합니다.

1. 칸 주변 영역을 회전 보정해 잘라냅니다.
2. 대비를 조정합니다(백분위 스트레칭).
3. 가로·세로 밝기 프로파일의 기울기로 패키지 경계를 찾습니다. 지정 크기에 가장 가까운 경계 쌍을 고르므로, SL 안쪽의 die 경계와 섞이지 않습니다.
4. 지정 크기(제품의 bbox w, h)의 BBOX를 **검출된 경계의 중심**에 그립니다.

기존 최상단 레이어 BBOX는 교체되고, 전체가 Ctrl+Z 한 번으로 되돌려집니다.

**검출이 약할 때의 대책**

- **실패 판정:** 경계가 잡음 대비 약하거나(`min_edge_ratio`), 크기가 지정 크기와 크게 다르거나(`size_tolerance`), 중심이 다른 칸들의 격자 오프셋 중앙값에서 벗어나면(`outlier_tol`, pitch 비율) 실패로 봅니다.
- **추정 배치:** 실패한 칸은 성공한 칸들의 중앙값 오프셋으로 위치를 추정해 배치하고, **점선**으로 그립니다.
- **검토:** "Next Estimated"로 추정된 칸만 차례로 이동하며 확인합니다. 수정 모드로 손보면 추정 표시가 사라집니다.
- **결과 표시:** 검출 / 추정 / 실패 개수를 상태바에 표시합니다.
- **조절:** `config.json`의 `auto_bbox`에서 기준을 바꿉니다.

```json
"auto_bbox": {"contrast_pct": [2, 98], "window_scale": 1.2, "size_tolerance": 0.3,
              "min_edge_ratio": 4.0, "outlier_tol": 0.05}
```

## 테스트용 샘플 TIF 생성

```bat
python tools\make_sample_tif.py                      REM 14900 x 14900, 12 layers, 8-bit gray, LZW (~1.7 GB)
python tools\make_sample_tif.py --compress none      REM 비압축 (~2.7 GB, 메모리 맵 경로)
python tools\make_sample_tif.py --dtype uint16       REM 16-bit
python tools\make_sample_tif.py --size 4000 --layers 3 --out small.tif
```

- **내용:** 회전된 칩 그리드에 패키지 bbox, SL 테두리 안쪽의 die, 일부 게이트에서만 보이는 보이드, SL 코너에서 시작하는 사분면 초타원 박리가 들어갑니다.
- **메모리:** 띠(strip) 단위로 생성해서 쓰므로 메모리를 적게 씁니다. 압축 모드에서는 한 레이어(8-bit 기준 약 222MB)만 메모리에 둡니다.
- **그리드 값:** 끝나면 그리드 값(origin, pitch, rows, cols, angle)과 products.json용 항목을 출력합니다. 이 값을 넣으면 칸과 bbox가 이미지와 맞습니다.

## 메모리 / GPU

- **필요한 부분만 읽기:** 비압축 TIF는 메모리 맵으로 필요한 부분만 읽습니다. 압축 TIF는 한 번에 한 프레임만 디코딩해서 캐시합니다.
- **타일 단위 업로드:** GPU에는 4096 크기 타일 단위로 하나씩 올리고, 올린 뒤 CPU 메모리를 바로 해제합니다. 8-bit 원본은 GL_R8로 올립니다.
- **GPU Cache:** "Load Configuration" 그룹의 `GPU Cache (MB)`가 상한입니다. 이를 넘으면 가장 오래 안 본 레이어를 해제하고, 다시 볼 때 디스크에서 읽습니다.
- **레이어 넘기기:** 레이어 디코딩은 백그라운드 스레드에서 합니다. 화면은 멈추지 않습니다.
  - 캐시에 없는 레이어로 가면 저해상도 미리보기(1/8)가 바로 보이고, 화면에 보이는 타일부터 원본 해상도로 바뀝니다.
  - 현재 레이어의 앞뒤 레이어(넘기는 방향 먼저)를 미리 읽어 두므로, 한 장씩 넘길 때는 거의 기다리지 않습니다. GPU 캐시에 안 들어가는 이웃 레이어는 RAM에 최대 2장 둡니다.
  - 빠르게 여러 장을 넘기면 건너뛴 레이어는 읽지 않습니다.
  - 파일을 연 뒤 나머지 레이어도 GPU 캐시 한도 안에서 백그라운드로 올립니다(이전의 "Preloading" 창 없음).
- **Angle Preview**도 패치를 백그라운드에서, 필요한 줄(strip)만 읽습니다.
