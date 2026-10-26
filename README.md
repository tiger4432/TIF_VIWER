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

1. 빌드 전용 가상환경 `.venv-build`를 준비합니다(IDE가 쓰는 `.venv`와 분리). Python은 다음 순서로 찾습니다.
   - `PYTHON` 환경변수 (예: `set PYTHON=C:\Python310\python.exe`)
   - `py -3.10` (py launcher)
   - PATH의 `python` (예: 활성화된 conda 환경)

   찾은 Python이 64-bit 3.10–3.13이 아니면 중단합니다. `.venv-build`가 다른 Python으로 만들어졌으면 지우고 다시 만듭니다.
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

- **Python을 찾지 못함:** conda 환경을 활성화하거나, python.org에서 Python 3.10 x64를 설치하거나, `PYTHON` 환경변수로 python.exe 경로를 지정합니다.
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
    "BBOX":        {"color": [255,255,0], "shape": "rectangle", "def_w": 480, "def_h": 480},
    "Corner":      {"color": [255,0,255], "shape": "rectangle", "snap": "bbox_corner"},
    "CornerCurve": {"color": [0,255,0], "shape": "curve", "def_p": 4.0,
                    "snap": "bbox_corner", "snap_anchor": "center", "snap_offset": "SL"}
}
```

리스트 형식(`[{"name": "BBOX", ...}, ...]`)과 이전 번호 키 형식(`"4": {"name": "BBOX", ...}`)도 읽습니다. 이전 voids.json의 `type_id`는 파일 안의 타입 이름으로 바꿔 읽습니다.

| 키 | 의미 |
|---|---|
| `shape` | `ellipse` / `rectangle` / `curve`(초타원 \|x/rx\|^p + \|y/ry\|^p = 1) |
| `def_w`, `def_h` | 0보다 크면 클릭 한 번으로 이 크기의 마킹을 만듭니다. 0이면 드래그로 크기를 정합니다. |
| `def_p` | curve의 기본 p (기본값 4) |
| `snap: "bbox_corner"` | 추가·이동·크기 조절 시 셀 bbox의 가장 가까운 코너에 자동으로 붙습니다. |
| `snap_anchor` | `edge`(기본값): 도형의 모서리가 코너에 붙습니다.<br>`center`: 도형의 중심이 코너에 오고 bbox 안쪽 **사분면만** 그립니다. 사분면은 bbox 내 좌표로 자동 판정합니다. |
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
- **`chip_size_um`:** 실제 칩(bbox) 크기(um)입니다. `bbox` 타입의 기본 크기는 `chip_size_um / Scale` 픽셀로 정해지고, Scale을 바꾸면 따라 바뀝니다. Scale이 비어 있으면 bbox는 드래그로 그리고, Auto BBOX는 검출한 크기를 씁니다.
- **`layout`:** 각 칸의 기준 bbox 안에 점선으로 그리는 세부 박스입니다.
  - 좌표는 칩 좌상단 기준 um입니다. 실제로 그린 bbox 크기에 맞춰 늘거나 줄어들므로, Scale이 조금 틀려도 bbox를 따라갑니다.
  - `SL`은 스크라이브 라인 폭(x, y, um)입니다. `snap_offset: "SL"`인 타입은 bbox 코너에서 이만큼 안쪽 지점을 기준으로 그립니다.
  - 박스는 `"inset": "SL"`(bbox를 SL만큼 안쪽으로 줄인 영역), `"rect": [x, y, w, h]`, `"repeat"`(배열 반복)로 정의합니다.
- 이전 형식(`"bbox": {"w", "h"}` 픽셀 + `layout.design_size`)도 그대로 읽습니다.

### 축척 (Scale)

Grid 패널의 **Scale (um/px)**은 1픽셀의 실제 크기입니다. 값이 있으면 voids.json / voids.csv에 um 컬럼(`RelativeX_um`, `RelativeY_um`, `RadiusX_um`, `RadiusY_um`, `BBoxRelX_um`, `BBoxRelY_um`)을 채웁니다. 이 컬럼 이름도 `output_columns`로 바꿀 수 있습니다. voids.json과 patches.json에 `um_per_px`를 함께 저장하고, 그 voids.json을 불러오면 Scale 칸에 값을 다시 넣습니다.

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

Grid Configuration의 "Angle Preview..."를 누르면 웨이퍼 곳곳의 칸을 회전 보정된 패치로 2×2로 보여줍니다. 각 패치에는 그리드 선, 기준 bbox, layout이 겹쳐 그려집니다.

- **판단 방법:** 각도가 맞으면 모든 패치에서 노란 그리드 선이 칩 사이 스트리트 중앙을 지납니다. 틀리면 중심에서 먼 패치일수록 어긋납니다.
- **갱신과 이동:** 각도, 원점, 피치, 레이어, 밝기가 바뀌면 바로 다시 그립니다. 패치를 클릭하면 메인 화면이 그 칸으로 이동합니다.
- **지점 위치:** 원형 웨이퍼 기준의 극좌표(중심 = 그리드 중심, R = 그리드에 내접하는 원의 반지름)로, `config.json`의 `angle_preview`에서 바꿉니다. 기본값은 센터, 위 0.75R, 오른쪽 0.75R, 오른쪽 위(45°) 0.5R입니다. 본딩맵이 있으면 가장 가까운 본딩된 칸을 사용합니다.

```json
"angle_preview": {"points": [{"name": "Center", "angle": 0, "r": 0.0},
                             {"name": "Top 3/4", "angle": 90, "r": 0.75}], "view_scale": 1.4}
```

## 본딩맵

- 엑셀에서 복사한 맵을 붙여넣고 "Import Map Data"를 누릅니다.
- Export 이미지(raw, overlay, mask, merged)는 **LEG(본딩맵 값)가 있는 칸만** 저장합니다. 본딩맵이 없으면 이미지 Export를 시작하지 않습니다. 붙여넣은 맵은 `bonding_map.txt`(그대로 다시 붙여넣을 수 있음), Reject Values는 `bonding_map_reject.txt`로 함께 저장합니다.
- **Reject Values:** 쉼표로 구분한 값(예: `X, NG`)과 일치하는 칸은 본딩되지 않은 칸으로 처리합니다. ROI, Export, 칸 이동, 캘리브레이션, 각도 미리보기에서 모두 빠집니다. 값을 바꾸고 Enter를 누르면 바로 다시 적용합니다.
- 화면 좌상단에 화면 중심 칸의 `X`, `Y`, `LEG`(본딩맵 값)를 노란 글씨로 표시합니다.

## 조작

| 동작 | 키 / 마우스 |
|---|---|
| 그리기 / 수정 / 삭제 모드 | W / E / R을 **누르고 있는 동안만**. 떼면 보기 모드. 여러 개를 누르면 마지막 키. 드래그 중에 떼면 드래그가 끝날 때 적용 |
| 보기 모드 (마킹 도구 끔) | 모드 키를 뗌 (Esc도 가능) |
| 호버 | 모든 모드에서 마우스 아래 마킹이 반투명해집니다. 겹치면 경계가 커서에 더 가까운 마킹이 선택되고, bbox는 다른 마킹이 없을 때만 선택됩니다. |
| 마우스 아래의 다른 레이어 마킹(점선)을 현재 레이어로 옮기기 | Q |
| 화면 이동 | 오른쪽 또는 휠 버튼 드래그 (모든 모드), 보기 모드에서는 왼쪽 드래그 |
| 크기 조절 | 수정 모드에서 마킹 가장자리를 드래그. 잡은 쪽의 반대편은 고정됩니다. 모서리 근처를 잡으면 가로세로를 함께 조절합니다. |
| 디펙 타입 | 숫자 = 타입 번호(선언 순서, 0 = 10). 0.4초 안에 이어 누르면 두 자리(1 → 2 = 12). 마킹 위에 호버 중이면 그 마킹의 타입을 바꾸고(번호 하나가 되돌리기 한 단계), 아니면 새로 그릴 타입을 고릅니다. 텍스트 입력 중에는 동작하지 않습니다. |
| 마킹 변경 되돌리기 | Ctrl+Z (추가, 이동, 크기 조절, 지우기, 레이어 이동(Q), 레이어·칩 지우기, JSON 불러오기, p 조절, Auto BBOX) |
| curve의 p 조절 | 수정 모드(E)에서 curve 위에 마우스를 올리고 휠 |
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

## 자동 BBOX (Auto BBOX)

"Auto BBOX"를 누르면 최상단 레이어에서 본딩된 칸마다 다음을 수행합니다.

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
