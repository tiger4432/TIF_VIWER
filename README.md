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

## 빌드 가이드 (PyInstaller)

### 한 번에 빌드

저장소 루트에서 다음을 실행합니다.

```bat
build.bat
```

`build.bat`는 아래 순서로 동작합니다.

1. `.venv`가 없으면 가상환경을 만듭니다. Python은 다음 순서로 찾습니다.
   - `PYTHON` 환경변수 (예: `set PYTHON=C:\Python310\python.exe`)
   - `py -3.10` (py launcher)
   - PATH의 `python` (예: 활성화된 conda 환경)

   찾은 Python이 64-bit 3.10–3.13이 아니면 중단합니다.
2. `requirements.txt`와 `requirements-build.txt`를 설치합니다.
3. `pyinstaller --noconfirm --clean TIF_Viewer.spec`을 실행합니다.
4. `config.json`, `products.json`, `deploy.json`을 `dist\TIF_Viewer\`(exe 옆)로 복사합니다.

conda에서 빌드하는 예시입니다.

```bat
conda create -n tifviewer python=3.10
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

- **Python을 찾지 못함:** conda 환경을 활성화하거나, python.org에서 Python 3.10 x64를 설치하거나, `PYTHON` 환경변수로 python.exe 경로를 지정합니다.
- **다른 Python으로 만든 `.venv`가 남아 있음:** `.venv` 폴더를 지운 뒤 다시 실행합니다.
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
- **타입 편집:** 프로그램의 "Types..." 창에서 바꾼 내용은 캐시에만 저장되므로, 다음 시작 때 공유 폴더 내용으로 덮어씌워집니다. 영구 변경은 공유 폴더의 `config.json`을 수정합니다.

`shared_dir`이 비어 있으면(기본값) 로컬 모드입니다. 이때는 실행 위치(작업 폴더)와 상관없이 **exe가 있는 폴더**(소스 실행 시 저장소 루트)의 파일을 읽습니다.

### 디펙 타입 (`config.json` → `void_types`)

```json
"4": {"name": "BBOX", "color": [255,255,0], "shape": "rectangle", "def_w": 480, "def_h": 480},
"5": {"name": "Corner", "color": [255,0,255], "shape": "rectangle", "snap": "bbox_corner"},
"6": {"name": "CornerCurve", "color": [0,255,0], "shape": "curve", "def_p": 4.0,
      "snap": "bbox_corner", "snap_anchor": "center", "snap_offset": "SL"}
```

| 키 | 의미 |
|---|---|
| `shape` | `ellipse` / `rectangle` / `curve`(초타원 \|x/rx\|^p + \|y/ry\|^p = 1) |
| `def_w`, `def_h` | 0보다 크면 클릭 한 번으로 이 크기의 마킹을 만듭니다. 0이면 드래그로 크기를 정합니다. |
| `def_p` | curve의 기본 p (기본값 4) |
| `snap: "bbox_corner"` | 추가·이동·크기 조절 시 셀 bbox의 가장 가까운 코너에 자동으로 붙습니다. |
| `snap_anchor` | `edge`(기본값): 도형의 모서리가 코너에 붙습니다.<br>`center`: 도형의 중심이 코너에 오고 bbox 안쪽 **사분면만** 그립니다. 사분면은 bbox 내 좌표로 자동 판정합니다. |
| `snap_offset` | 코너에서 안쪽으로 옮길 거리입니다. `"SL"`이면 선택한 제품 layout의 SL 값을 쓰고, `[x, y]`이면 픽셀 값을 씁니다. |

타입 이름이 `bbox`(대소문자 무시)인 마킹은 그 칸의 기준 bbox가 됩니다. 기준은 **가장 위(번호가 가장 낮은) 레이어**에 그린 bbox입니다.

### 출력 컬럼 이름 (`config.json` → `output_columns`)

voids.json의 필드와 voids.csv의 컬럼 이름을 exe를 다시 빌드하지 않고 바꿀 수 있습니다. 키 목록과 기본 이름은 `sat_widgets/output_columns.py`에 선언되어 있습니다. 값을 `null`로 주면 해당 컬럼을 출력하지 않습니다.

```json
"output_columns": {
    "csv":  {"bbox_rel_x": "ValidX", "bbox_rel_y": "ValidY", "radius_y": null},
    "json": {"bbox_rel_x": "validX"}
}
```

`bbox_rel_x`와 `bbox_rel_y`는 디펙 중심을 그 칸의 기준 bbox **좌상단**을 원점으로 한 좌표입니다(그리드 정렬 기준 픽셀). 칸에 bbox가 없으면 빈 값입니다.

### WF 제품군 (`products.json`)

```json
{
  "default": "WF_A",
  "products": {
    "WF_A": {
      "grid": {"rows": 10, "cols": 12, "pitch_x": 500, "pitch_y": 500},
      "bbox": {"w": 480, "h": 480},
      "layout": {
        "design_size": [480, 480],
        "SL": [20, 20],
        "style": {"color": [0,200,255], "alpha": 200, "width": 1, "dash": [4, 4]},
        "boxes": [
          {"name": "die", "inset": "SL"},
          {"name": "pad", "rect": [40, 40, 80, 40], "repeat": {"cols": 4, "rows": 1, "dx": 100, "dy": 0}}
        ]
      }
    }
  }
}
```

- **`grid`:** 적은 키만 적용합니다. 쓸 수 있는 키는 `rows`, `cols`, `pitch_x`, `pitch_y`, `start_x`, `start_y`, `angle`입니다.
- **`bbox`:** 이름이 `bbox`인 타입의 기본 크기를 정합니다.
- **`layout`:** 각 칸의 기준 bbox 안에 점선으로 그리는 세부 박스입니다.
  - 좌표는 bbox 좌상단을 기준으로 `design_size` 단위로 적습니다. 실제로 그린 bbox 크기에 맞춰 늘거나 줄어듭니다.
  - `SL`은 스크라이브 라인 폭(x, y)입니다. `snap_offset: "SL"`인 타입은 bbox 코너에서 이만큼 안쪽 지점을 기준으로 그립니다.
  - 박스는 `"inset": "SL"`(bbox를 SL만큼 안쪽으로 줄인 영역), `"rect": [x, y, w, h]`, `"repeat"`(배열 반복)로 정의합니다.

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

## 조작

| 동작 | 키 |
|---|---|
| 마킹 모드(누르고 있는 동안) | Shift |
| Draw / Edit / Erase 도구 | W / E / R |
| 디펙 타입 선택 | 1–9, 0 |
| 마킹 변경 되돌리기 | Ctrl+Z (추가, 이동, 크기 조절, 지우기, 레이어·칩 지우기, JSON 불러오기, p 조절) |
| curve의 p 조절 | Edit 도구(E)에서 curve 위에 마우스를 올리고 휠 |
| 레이어 이동 | Ctrl/Shift + 휠, ↑/↓ |
| 확대·축소 / 이동 | 휠 / 드래그 |

레이어 번호는 **절대 번호**(파일 안의 페이지 번호)로 저장합니다. `layer_numbering`이 없는 이전 voids.json은 "현재 연 범위의 시작 레이어 기준 상대 번호"로 해석합니다.

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
