# 구현 진행 상태

## 이번 구현

기준 계획: [PLAN.md](PLAN.md). 원본 계획은 수정 없이 보관합니다.
참고 형식: [DACON-236092-SemanticSegmentation](https://github.com/sirius-mhlee/DACON/tree/main/DACON-236092-SemanticSegmentation).

기존 코드의 `argparse`, `dataclass`, 평탄한 YAML, 기능별 모듈, 짧은 실행 절차를 따릅니다.
패키지·CLI 이름은 계획의 `vloop`를 사용하므로 구현은 `src/vloop`에 둡니다.
설정에서 알 수 없는 키를 조용히 버리던 동작은 실험 설정 오타를 발견하도록 오류로 바꿨습니다.

| 단계 | 상태 | 확인 범위 |
|---|---|---|
| 1. 환경·설정·CLI | SAM 3 추론 확인 | Python 3.12, 설정 검증, 실행 기록, CUDA 연산과 실제 SAM 3 추론 |
| 2. 이미지 등록·자동 라벨링 | 샘플 통합 검증 완료 | 중복 방지, 원본 보존, 작업별 예측, 고정 설정으로 다른 프로세스에서 재개 |
| 3. FiftyOne 검수 | 미구현 | 검수 operator·승인 해시·Annotation Schema 필요 |
| 4. COCO·DVC 릴리스 | 미구현 | 공통 RLE 변환 구현, 승인 라벨 내보내기·split·remote·복원 필요 |
| 5. 학습·MLflow | 미구현 | RF-DETR 학습·가중치 접근, run 기록·체크포인트 재개 필요 |
| 6. 모델 복원·평가 | 미구현 | 새 프로세스 복원·박스/마스크 평가 필요 |
| 7. 반복 개선 | 미구현 | 실제 데이터로 두 번의 전체 루프 검증 필요 |

## 환경 관찰 (2026-09-06)

- Python 3.12.3 가상환경 `.venv`를 생성했습니다.
- 호스트 `nvidia-smi`: NVIDIA GeForce RTX 5060 Laptop GPU, 드라이버 580.173.02, 약 8 GB VRAM.
- PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128 설치 후 CUDA 행렬 연산 결과를 CPU 기대값과
  비교해 통과했습니다. 장치 compute capability는 12.0입니다.
- 격리 환경의 `nvidia-smi` 실패와 호스트 GPU 정상 상태를 구분했습니다. 드라이버는 변경하지 않았습니다.
- FiftyOne 1.21.0과 SAM 3 0.1.0을 설치했습니다. SAM 3는 사용자가 준비한 로컬 소스를
  editable로 설치했으며 커밋은 `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`입니다.
- 사용자가 내려받은 `.vloop/models/sam3/sam3.pt`를 실제 `project.yaml`에 연결했습니다.
  체크포인트 SHA-256은 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`입니다.
- NumPy 1.26.4, OpenCV headless 4.11.0.86, SciPy 1.16.3, tifffile 2025.5.10,
  einops 0.8.2, pycocotools 2.0.11 조합을 설치하고 `autolabel` extra에 기록했습니다.
- RF-DETR·MLflow·DVC는 아직 설치하지 않았습니다. 전체 학습 파이프라인의 의존성 lock은 미확정입니다.
- 실제 데이터 폴더·클래스·프롬프트는 미입력 상태입니다. 검증에는 별도 설정과 SAM 3 공식
  샘플을 사용했으며, 실제 프로젝트에 예시 클래스를 넣지 않았습니다.

## 자동 라벨링 구현

- `vloop autolabel [--limit N]`, `vloop autolabel --resume JOB_ID`를 추가했습니다.
- 작업별 SQLite 입력·상태 목록, 최종 설정, SAM 3 커밋·체크포인트·어휘·adapter·의존성 해시를
  보관합니다. 재개에는 원래 설정을 사용하며 완료 결과를 다시 추론하지 않습니다.
- 예측 JSON을 먼저 저장한 뒤 FiftyOne에 반영합니다. DB 쓰기만 실패한 이미지는 추론 결과를
  재사용합니다. 처리 상태와 정상 빈 예측을 구분하고 실패한 이미지의 원인을 남깁니다.
- 예측 필드는 작업별로 만들며 `ground_truth`와 검수 상태를 수정하지 않습니다.
- 공통 표현에 클래스 ID, 프롬프트, 신뢰도, 픽셀 박스, 전체 이미지 크기의 COCO RLE를
  저장합니다. FiftyOne의 박스 내부 마스크와 변환할 때 크기·좌표·면적을 검사합니다.
- 8 GB 환경을 위해 이미지·프롬프트를 하나씩 처리하고 BF16을 사용합니다. 모델은 작업 중
  재사용하며 컴파일은 끕니다. FiftyOne 1.21의 내부 점수 필터도 사용자 임계값에 맞춥니다.

## 검증 결과

- 기본 테스트 58개 통과. 실제 DB·GPU 테스트 2개는 기본 실행에서 명시적으로 제외합니다.
- 별도 호스트 실행에서 FiftyOne 통합 테스트 1개 통과: 실제 등록, 다른 프로세스의 재등록,
  중복 방지, `ground_truth`와 검수 상태 보존, 예측 필드 생성 도중 중단된 상태의 복구를 확인했습니다.
- 실제 GPU 통합 테스트 1개 통과: 공식 `truck.jpg`, `groceries.jpg`를 등록하고 클래스 ID
  `7 → truck`, `42 → apple`로 처리했습니다. 첫 장 후 중단하고 현재 YAML의 클래스를 변경한
  뒤 새 프로세스로 재개해도 원래 설정으로 나머지 한 장만 처리했습니다. 첫 결과의 해시와
  사람이 수정한 정답·검수 상태, DB 저장 후 마스크 면적이 유지됐습니다.
- 비정방형 이미지와 구멍·분리 영역 마스크의 RLE 변환, 실패 이미지 재개, DB 실패 후
  결과 재사용, 설정·입력·결과 변경 감지, 스냅샷 생성 중 중단의 복구 안내를 테스트했습니다.
- `ruff check`, `ruff format --check`, `pip check` 통과.
- 샘플 설정으로 실행한 `doctor --sam3-image`는 20개 통과, 3개 실패입니다.
  실패는 RF-DETR·MLflow·DVC 미설치이며 `sam3_inference`는 통과했습니다.
  실제 `project.yaml`은 데이터·클래스가 비어 있어 필수 입력 검사도 아직 통과하지 못합니다.

### 실제 GPU 측정

| 항목 | 결과 |
|---|---|
| 입력 | 공식 `truck.jpg`, 1800×1200, 프롬프트 `truck` |
| 결과 | 트럭 1개, 신뢰도 0.8671875, 박스 xywh `[84, 282, 1625, 566]` |
| 정밀도·배치 | BF16, 이미지 1장·프롬프트 1개 |
| PyTorch 최대 할당 / 예약 메모리 | 5.2624 / 5.5098 GiB |
| 모델 로딩 포함 실행 시간 | 약 8.77초 |
| 별도 프로세스의 재개 | 완료 2장, 이번 시도 처리 1장, 최대 할당 5.2629 GiB |

이는 샘플에 대한 관찰값입니다. 전체 GPU 프로세스 메모리나 임의의 해상도·객체 수에 대한
보장값이 아니며, 실제 데이터도 작은 묶음으로 먼저 확인해야 합니다.

로컬 결과:

- 한 장 추론: `.vloop/smoke/state/runs/doctor_20260906T121142_f8085e8b/`
- 시각화: 위 폴더의 `sam3-preview.png`
- 중단·재개 통합 검증 기록: `.vloop/smoke/sam3-integration.json`

## 다음 확인 지점

실제 이미지 폴더와 클래스별 ID·이름·프롬프트를 설정한 뒤 `ingest`, `autolabel --limit 3`으로
도메인 데이터의 출력과 메모리를 확인합니다. 다음 구현은 `review` 명령, 기존 수정 내용을
보존하는 `ground_truth` 초기화, Annotation Schema, 승인·재검수 operator와 라벨 해시입니다.
검수 승인 이후 릴리스·학습·평가를 이어가며, 계획의 전체 두 차례 개선 루프는 아직 미완료입니다.
