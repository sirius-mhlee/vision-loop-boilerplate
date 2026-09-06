# 구현 진행 상태

## 이번 구현

기준 계획: [PLAN.md](PLAN.md). 원본 계획은 수정 없이 보관합니다.
참고 형식: [DACON-236092-SemanticSegmentation](https://github.com/sirius-mhlee/DACON/tree/main/DACON-236092-SemanticSegmentation).

기존 코드의 `argparse`, `dataclass`, 평탄한 YAML, 기능별 모듈, 짧은 실행 절차를 따릅니다.
패키지·CLI 이름은 계획의 `vloop`를 사용하므로 구현은 `src/vloop`에 둡니다.
설정에서 알 수 없는 키를 조용히 버리던 동작은 실험 설정 오타를 발견하도록 오류로 바꿨습니다.

| 단계 | 상태 | 확인 범위 |
|---|---|---|
| 1. 환경·설정·CLI | 진행 중 | Python 3.12, CLI, 설정 검증, 실행 기록, 호스트 CUDA 연산 |
| 2. 이미지 등록·자동 라벨링 | 등록 기반 구현 | 해시 중복 방지, EXIF 방향, 원본 보존, 파일별 실패, 재등록 |
| 3. FiftyOne 검수 | 미구현 | 검수 operator·승인 해시·Annotation Schema 필요 |
| 4. COCO·DVC 릴리스 | 미구현 | RLE 변환, split 유지, remote 저장, 별도 복원 필요 |
| 5. 학습·MLflow | 미구현 | RF-DETR 학습·가중치 접근, run 기록·체크포인트 재개 필요 |
| 6. 모델 복원·평가 | 미구현 | 새 프로세스 복원·박스/마스크 평가 필요 |
| 7. 반복 개선 | 미구현 | 실제 데이터로 두 번의 전체 루프 검증 필요 |

## 환경 관찰 (2026-09-06)

- Python 3.12.3 가상환경 `.venv`를 생성했습니다.
- 호스트 `nvidia-smi`: NVIDIA GeForce RTX 5060 Laptop GPU, 드라이버 580.173.02, 약 8 GB VRAM.
- PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128 설치 후 CUDA 행렬 연산 결과를 CPU 기대값과
  비교해 통과했습니다. 장치 compute capability는 12.0입니다.
- 격리 환경의 `nvidia-smi` 실패와 호스트 GPU 정상 상태를 구분했습니다. 드라이버는 변경하지 않았습니다.
- FiftyOne 1.21.0을 설치했습니다. RF-DETR·MLflow·DVC·SAM 3는 아직 설치하지 않았습니다.
- 실제 데이터 폴더·클래스·프롬프트·SAM 3 체크포인트는 미입력 상태입니다.

## 검증 결과

- 기본 테스트 36개 통과. 실제 DB 테스트 1개는 기본 실행에서 명시적으로 제외합니다.
- 별도 호스트 실행에서 FiftyOne 통합 테스트 1개 통과: 실제 등록, 다른 프로세스의 재등록,
  중복 방지, `ground_truth`와 검수 상태 보존을 확인했습니다.
- `ruff check`, `ruff format --check`, `pip check` 통과.
- 호스트의 실제 `vloop doctor`는 15개 검사 통과, 7개 실패를 기록했습니다.
  실패 항목은 실제 입력, SAM 3·RF-DETR·MLflow·DVC 미설치, SAM 3 체크포인트·소스 미설정입니다.
  환경 전체가 준비된 상태로 표시하지 않았습니다.
- 로컬 보고서: `.vloop/runs/doctor_20260906T024034_bc8d9c14/report.json`.

## 다음 확인 지점

실제 데이터의 작은 묶음과 클래스 프롬프트, 승인받아 내려받은 SAM 3 체크포인트를 입력합니다.
SAM 3 코드를 설치하고 커밋을 기록한 뒤 `vloop doctor --sam3-image ...`로 한 장 추론을 확인합니다.
현재 GPU의 메모리에서 실제 SAM 3 추론이 가능한지는 이 실행에서 확인해야 합니다.
현재 단계에는 이를 성공했다고 간주할 근거가 없습니다.

그다음 작업별 예측 필드·입력 목록·설정 스냅샷·이미지별 상태를 갖는 `autolabel`과 재개 기능을
구현하고, 검수용 `ground_truth` 초기화 및 승인 흐름을 연결합니다.
전체 의존성 lock과 SAM 3 커밋 고정은 실제 통합 검증을 통과한 환경을 기준으로 확정합니다.
