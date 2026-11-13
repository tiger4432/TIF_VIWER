# Edge Stains / Edge BBOX 옵션 설명표

화면: Void Detection 아래 **Edge Stains** 그룹, Auto Calibration의 **Edge BBOX**.
시작 값은 `config.json`의 `"void_detect": {"edge_band": {...}}`(Edge Stains), `"auto_bbox": {"edges": {...}}`(Edge BBOX)에 적을 수 있습니다(키는 아래 표). 화면 입력칸이 config보다 우선합니다.

## 기준 칩(레퍼런스)이란

"정상 칩은 이렇게 생겼다"는 비교 기준입니다. 칩끼리 거의 같다는 점을 이용해, 여러 칩을 겹쳐 **픽셀마다 중앙값**을 낸 가상의 칩을 만듭니다(오염은 칩마다 자리가 달라 중앙값에서 빠짐). 각 칩을 이 기준 칩과 비교해 **더 어두운 곳 / 구조가 다른 곳**을 점으로 봅니다.
- **흩어짐(spread):** 같은 픽셀이 정상 칩끼리 얼마나 다른지(노이즈, 패드 위치 차이). 차이를 이 값으로 나눠 기준(Dark, SSIM thr)을 "흩어짐의 몇 배"로 씁니다 → 원래 들쭉날쭉한 곳(패드, 경계)은 둔감, 평평한 곳은 민감.
- **reference min / max:** 기준 칩의 주변 3×3 최솟값·최댓값. 칩이 1 px쯤 어긋나도 이 범위 안이면 차이로 보지 않습니다.
- **어떤 칩으로 만드나:** 기본은 **변(위·아래·좌·우)마다 따로**, 그 변이 가장 깨끗한 칩들(깨끗한 무리: 가장 깨끗한 10% 칩의 1.5배 안, 8개 이상, Clean %·64개 이하). 위가 오염된 칩도 아래 기준에는 쓰입니다. 오염 칩이 많아 기준 칩(Debug chip 5번)에 오염이 비치면 **Golden**(영상 위 Ctrl+G)으로 확실히 깨끗한 칩을 직접 고르세요(5개 이상 권장).

## Edge Stains 처리 순서

1. **칩 자르기:** bbox마다 영상에 놓인 그대로(회전·보간 없이) 정수 px로 자름 → 칩 외곽으로 대략, 칩 가운데로 ±1 px 미세 위치 맞춤 → 작업 해상도(`work_size`)로 블록 평균 축소 → 칩마다 밝기 정규화.
2. **기준 칩:** Golden 칩의 중앙값, 없으면 가장 깨끗한 칩들(Clean %)의 중앙값. 픽셀마다 칩끼리 흩어짐과 기준의 최솟값~최댓값 범위.
3. **점:** 기준 최솟값보다 Dark × 흩어짐만큼 어두운 곳, 또는(Compare) SSIM이 SSIM thr × 흩어짐만큼 다른 곳. Density 이상 모인 것만.
4. **띠:** 변마다 점을 변 방향으로 길게 이음(Merge r × Along ×), Link gap 이내 조각은 한 띠.
5. **영역:** 칩 경계부터 띠의 가장 먼 점까지. Length ≥, Min depth, Aspect를 넘으면 오염. 모서리를 도는 조각은 합침.

## Edge Stains 입력칸


| 입력칸 | config 키 | 기본 | 뜻 | 이럴 때 |
|---|---|---|---|---|
| Length ≥ | `min_length_px` | 100 | 변을 따라 잰 최소 띠 길이(px) | 짧은 오염을 놓치면 ↓ |
| Merge r | `merge_radius_px` | 8 | 점을 선으로 잇는 거리(px, 변을 가로질러) | 띠가 굵게 흩어져 있으면 ↑ |
| Along × | `merge_along` | 2 | 변을 따라서는 Merge r의 이 배수까지 이음 | 듬성한 띠가 끊기면 ↑ |
| Link gap | `link_gap_px` | 40 | 변을 따라 이 거리(px) 이내로 끊긴 조각은 한 띠 | 긴 띠가 여러 개/짧게 잡히면 ↑ |
| Depth % | `depth_frac` | 0.25 | 가장자리에서 안쪽으로 볼 깊이(칩 크기 대비; config는 비율) | 깊은 오염이 잘리면 ↑ |
| Dark | `dark_threshold` | 3 | 어두운 점 기준(칩끼리 흩어짐 배수) | 띠가 덜 잡히면 ↓, 엉뚱한 점이 많으면 ↑ |
| Density | `speck_density` | 0.06 | 점이 모여 있어야 하는 비율 | 띠가 듬성하면 ↓ |
| Aspect | `min_aspect` | 1.5 | 길이 ÷ 깊이 최소 | 짧고 두꺼운 오염이면 ↓ |
| Compare | `compare` | "both" | 점 기준: 어두움 + SSIM("both") / 어두움("intensity") / SSIM("ssim") | 눈으로 차이가 거의 없는 띠: both / ssim |
| SSIM thr | `ssim_threshold` | 25 | 구조가 다른 점 기준(흩어짐 배수) | 미검이면 ↓, 과검이면 ↑ |
| SSIM σ | `ssim_sigma_px` | 3 | SSIM 창(가우시안 σ, px) | 띠가 굵으면 ↑ |
| Blur σ | `blur_px` | 0 | 비교 전 가우시안 블러(px, 0=끔) | 노이즈가 심할 때 |
| Skip | `edge_skip_px` | 4 | 칩 맨 바깥에서 제외하는 폭(px) | 칩 테두리가 잡히면 ↑ |
| Min depth | `min_depth_px` | 6 | 가장 얕은 오염(px) | 얇은 줄이 잡히면 ↑ |
| Align tol | `edge_tol_px` | 1 | 칩끼리 어긋남 허용(px): 기준 칩을 ±이만큼(8방향) 옮긴 값 범위 안이면 차이 아님(Shift tol 0일 때) | 경계·패드를 따라 얇은 선이 잡히면 ↑, 약한 오염을 놓치면 ↓ |
| Texture | `texture_px` | 2 | 패드 격자 간격(px): 칩·기준을 이 크기 상자로 평균 낸 뒤 비교(0=끔) | 차이 맵(9번)이 격자 모양으로 전부 어두우면 패드 간격으로 |
| Shading | `shading` | 켬 | 칩 안의 밝기 기울기(조명) 제거: (칩 − 기준)의 평면을 빼고 비교 | 차이 맵에 한쪽이 밝고 반대쪽이 어두운 기울기가 보이면 켬 |
| Along | `edge_along_px` | 4 | 칩 경계에서 이 거리(px) 안은 기준 범위를 변 방향으로만(0=끔) | 경계 바로 옆 오염을 놓치면 ↑ |
| Shift tol | `shift_tol_px` | 0 | 칩끼리 어긋남 허용(px; 0이어도 3×3) | 패드·틈이 잡히면 ↑ |
| Edges T B L R | `edges` | "tblr" | 볼 변 | 상하단만: "tb" |
| Match | `match` | "none" | 칩 밝기 맞춤(칩 가운데로): none / linear / curve | 칩마다 전체 밝기가 다르면 linear |
| Clean % | `clean_ref` | 0.5 | 기준 칩을 만들 "가장 깨끗한 칩" 비율(최대 64개; 100 = 전체 표본) | 오염 칩이 아주 많으면 ↓ |
| Golden | `golden_cells` | 없음 | 기준 칩으로 쓸 칩 [[행, 열], ...] (Ctrl+G로 고름) | 확실히 깨끗한 칩을 알 때 |
| Mark as | `type` | 선택한 타입 | 마킹 void 타입 | |

  (그 밖에 `register_frac` 0.6: 미세 위치 맞춤에 쓰는 칩 가운데 비율, `bright_min` 0: 영역이 기준보다 이만큼 밝아야 인정(0 = 보지 않음), `work_size` 384: 작업 해상도(칩 긴 변).)

## Debug chip 이미지 읽는 법

버튼 = 화면 가운데 칩, 영상 위 **Ctrl+Shift+D** = 마우스 아래 칩. **Detect Edge Stains를 한 번 돌린 뒤에는 모든 칩의 결과가 저장되어 바로 열립니다**(돌리기 전이면 그 칩만 계산, 마킹은 하지 않음). 위쪽 글: 축소 배율, 기준 칩을 몇 개로 만들었는지(이 칩이 기준에 들어갔는지), 버려진 띠와 이유(short = Length ≥ / Link gap, thin = Min depth, not elongated = Aspect).

| 번호 | 이미지 | 볼 것 |
|---|---|---|
| 1 | raw window | 원본에서 읽은 창(여백 포함). 칩이 창 가운데 있는지 |
| 2 | cut: moved dx, dy | 위치를 맞춰 자른 칩(정수 px). 여백이 한쪽만 보이면 bbox가 많이 어긋난 것 |
| 3 | reduced, normalised | 작업 해상도 칩 |
| 4 | blurred | Blur σ > 0일 때만 |
| 5 | median chip | 기준 칩. 여기에 오염이 보이면 Golden을 고르거나 Clean %를 ↓ |
| 6 | chip-to-chip spread | 칩끼리 흩어짐. 밝을수록 그 자리는 둔감(패드·경계) |
| 7, 8 | reference min / max | 이 범위 안이면 차이 아님 |
| 9 | deviation | 파랑 = 기준보다 어두움, 빨강 = 밝음(±2×Dark에서 꽉 참). 띠가 파랗게 보여야 함 |
| 9b | SSIM change | 구조가 달라진 정도(노랑 = 2×SSIM thr) |
| 10 | dark specks | Dark / SSIM 기준을 넘은 점(가장자리 띠 안만) |
| 11 | specks kept | Density를 통과한 점. 띠의 점이 여기서 사라지면 Density ↓ |
| 12 | joined lines | 이은 띠. 끊겨 있으면 Merge r / Along × / Link gap ↑ |
| 13 | stain areas | 최종 영역(= 마킹 모양) |

**보는 순서(못 잡을 때):** 9번에 띠가 파랗게(또는 9b에 노랗게) 보이나? → 안 보이면 Dark / SSIM thr ↓, 또는 5번 기준 칩에 오염이 비치는지 확인(→ Golden). 보이면 10·11번에 점이 남나? → 11에서 사라지면 Density ↓. 12번에서 띠가 이어지나? → 끊기면 Merge r / Along × / Link gap ↑. 13번 영역이 없으면 위쪽 글의 버린 이유(short / thin / not elongated)에 맞는 입력칸을 조정.
**잘못 잡을 때:** 13번 영역이 생긴 자리의 9·9b를 보고, 정상인데 파랗거나 노라면 Dark / SSIM thr ↑ 또는 Edge tol / Shift tol ↑.

PNG로 저장(Save PNGs)해 보내 주시면 어느 단계에서 놓치는지 볼 수 있습니다.

## Edge BBOX 입력·설정

Auto Calibration의 **Edge BBOX**: bbox마다 네 변 근처에서 칩 외곽의 밝기 단차를 찾아 그 가운데로 옮깁니다(원본 영상, 0.1 px 단위). 손으로 놓은 bbox는 그대로. Ctrl+Z로 되돌림.

| 입력칸 / config 키 | 기본 | 뜻 | 이럴 때 |
|---|---|---|---|
| Edge pick / `edge_pick` | "steepest" | 변 근처 단차 중 무엇을 칩 경계로: 가장 가파른 것(steepest) / 가장 바깥(outer) / 가장 안쪽(inner) — 가장 가파른 것의 `pick_frac` 이상인 단차 중 | 경계 옆에 밝은 테두리·헤일로·스캔 꼬리가 있어 bbox가 한쪽으로 밀리면 outer 또는 inner |
| `pick_frac` | 0.5 | outer / inner에서 단차로 칠 최소 크기(가장 가파른 것 대비) | |
| `edge_search` | 0.06 | 지금 bbox 변에서 앞뒤로 찾는 범위(칩 크기 대비) | bbox가 많이 어긋나 있으면 ↑(이웃 칩 경계가 들어오지 않게) |
| `edge_band` | 0.3 | 단면을 평균 내는 칩 가운데 띠 폭(칩 크기 대비) | 가운데에 큰 구조물이 있으면 ↓ |
| `size_tol` | 0.02 | 잰 크기가 중앙값과 이만큼 넘게 다르면 한 변이 이상한 칩: 더 가파른 변 + 크기로 둠 | |

- **크기:** 선언 크기(bbox 타입 / 제품)가 잰 크기와 5% 안이면 선언 크기, 아니면 잰 크기의 중앙값(상태바에 표시).
- **단차 방향:** 변마다 길→칩이 밝아지는지 어두워지는지를 칩 다수결로 정해, 길 건너 이웃 칩의 변은 잡지 않습니다.
- **Edge debug** 버튼: 화면 가운데 칩에 대해, 바꾸지 않고 보여 줌 — 창 이미지(노랑 = 지금 bbox, 초록 = 맞춘 bbox, 빨강 = 고른 단차), 네 변 각각의 밝기 단면(흰색)과 단차 크기(주황), 찾는 범위(회색), 고른 단차(초록). 위쪽 글에 각 변의 위치(raw px)·방향·크기와 옮겨질 양. bbox가 밀리면 이 창을 보면 어느 단차를 잡았는지 바로 보입니다.
