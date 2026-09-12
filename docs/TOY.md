# 공개 정답을 사용한 자동 통합 검증 기록

이 문서는 앞서 실행한 8장짜리 자동 검증 예제를 보관합니다. **사진만 다운로드하고 SAM 3의
예측을 직접 검수하는 실습**은 [README의 Penn-Fudan 직접 실습](../README.md#penn-fudan-직접-실습)을
따릅니다. 아래 스크립트는 공개 정답을 가져오므로, 정답이 없는 상황을 연습할 때는 실행하지 않습니다.

[Penn-Fudan](https://www.cis.upenn.edu/~jshi/ped_html/) 보행자 사진과 객체별 정답 마스크를
사용합니다. [PyTorch의 instance segmentation 튜토리얼](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)에서도
사용하는 데이터입니다. 전체 ZIP을 내려받고 그중 사진 8장과 대응하는 마스크만 추출합니다.

이 예제는 SAM 3 추론, 검수 승인, DVC 릴리스, RF-DETR 학습, MLflow 기록과 FiftyOne 평가를
실제 실행합니다. 검수 화면에서 사람이 편집하는 대신 제공된 정답 마스크를 반영합니다.
승인 기록의 검수자 이름은 `toy-reference-import`이고, 메모에도 스크립트로 가져온 정답임을
명시합니다. SAM 3 예측은 별도 필드와 작업 폴더에 남습니다.

## 실행

프로젝트 루트에서 실행합니다. 기존 설치 절차의 `autolabel`, `pipeline` 의존성과
SAM 3 소스·체크포인트 설정이 필요합니다. RF-DETR Seg Nano 초기 가중치는
`project.yaml`의 `train_checkpoint` 또는 `.vloop/models/rfdetr/rf-detr-seg-nano.pt`에
미리 준비합니다. 예제는 이미 설치한 가상환경과 GPU를 사용합니다.

```shell
source .venv/bin/activate
mkdir -p .vloop/toy-pennfudan/source
curl -fL --retry 1 \
  -o .vloop/toy-pennfudan/source/PennFudanPed.zip \
  https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
python examples/pennfudan_toy.py
```

이 컴퓨터에는 다운로드와 실행 결과가 이미 남아 있습니다. 같은 예제 명령을 다시 실행하면
`progress.json`에 완료된 단계는 건너뜁니다. 실패 원인을 해결한 뒤 다시 실행할 수도 있지만,
완료 기록을 저장하기 직전에 강제 종료했다면 생성된 작업·릴리스 상태를 먼저 확인해야 합니다.
이 스크립트의 단계 건너뛰기는 개별 `vloop --resume` 기능과 별개입니다.

원래 `project.yaml`은 변경하지 않습니다. 예제는 별도 YAML과 소스 스냅샷이 있는 작은 Git
저장소를 `.vloop/toy-pennfudan/repo`에 만들고 여기에 `dataset/v001`, `dataset/v002` 태그를
저장합니다. DVC remote도 같은 toy 폴더 안에 별도로 만듭니다. 원본 프로젝트와 예제의 Git·DB·
실험 기록이 섞이지 않으며 이 예제 Git 저장소에는 외부 Git remote를 등록하지 않습니다.

## 실행 순서

| 버전 | train | val | test | 학습 |
|---|---:|---:|---:|---|
| v001 | 4장 | 1장 | 1장 | RF-DETR Seg Nano, 2 epochs |
| v002 | 6장 | 동일한 1장 | 동일한 1장 | 같은 초기 가중치로 새 학습, 2 epochs |

1. 이미지 해시와 기본 split 정책으로 train 6장·val 1장·test 1장을 선택합니다. 선택 목록과
   이미지·정답·ZIP 해시는 `source/selection.json`에 기록합니다.
2. 첫 6장을 `ingest` → `autolabel` → `review --prepare-only`로 등록하고 예측합니다.
   공개 정답을 `ground_truth`에 반영한 뒤 실제 검수 승인 API로 승인합니다.
3. `release --version v001` → `train --dataset-version v001` →
   `evaluate --job-id TRAIN_V001 --dataset-version v001 --split val`을 실행합니다.
4. train 이미지 2장을 추가하고 등록·예측·검수 준비를 다시 실행합니다. 기존 6장의 승인된
   정답은 보존하고 새 2장에만 공개 정답을 반영해 승인합니다.
5. `v002`를 릴리스하고 새 모델을 학습한 뒤 **v001의 val**로 평가합니다. 두 평가의
   `comparison_id`가 같고 두 릴리스의 val/test 내용이 동일한지 검사합니다.
6. `restore --version v001`로 이전 확정 데이터를 다시 복원합니다. 완료된 모델·평가·검수
   DB는 모두 남기며, test 평가는 이 예제에서 실행하지 않습니다.

각 명령에는 예제 YAML 경로를 `--config`로 전달합니다. 정확한 실행은
[`examples/pennfudan_toy.py`](../examples/pennfudan_toy.py)에 있습니다.
기본 설정의 batch 1, gradient accumulation 8, `train_num_workers: 0`을 사용합니다.
작은 train은 RF-DETR 로더의 최소 학습 길이를 맞추기 위해 반복 사용될 수 있습니다.

## 결과 확인

`result.json`의 `steps.train-v001.job_id`, `steps.train-v002.job_id`는 학습 작업 ID입니다.
`steps.evaluate-v001.job_id`, `steps.evaluate-v002.job_id`는 평가 작업 ID이며 아래 `--view`에
넣습니다. 각 단계의 출력 전문은 `logs/`, 작업 보고서는 `state/runs/`에 있습니다.

```shell
# 모델과 평가 지표를 MLflow에서 확인
vloop experiments --config .vloop/toy-pennfudan/repo/project.yaml

# result.json의 평가 작업 ID로 예측·정답 및 오탐·미탐 분석 열기
vloop evaluate --config .vloop/toy-pennfudan/repo/project.yaml --view EVALUATE_JOB_ID

# 보관된 SAM 3 예측과 검수 정답 확인
vloop review --config .vloop/toy-pennfudan/repo/project.yaml --limit 8
```

`--no-browser`를 붙이면 서버만 실행하고 출력된 주소를 직접 열 수 있습니다.
FiftyOne 평가 화면은 저장한 `display_confidence` 뷰로 열리며, 전체 저장 예측은 `all` 뷰에서
확인합니다. 이 표시 필터는 이미 계산한 지표를 바꾸지 않습니다.
이번 v002 모델은 2 epochs만 학습해 기본 표시 임계값 0.5를 넘는 예측이 없었습니다.
처음 화면에서 정답만 보이면 왼쪽 위 `display_confidence`를 `all`로 바꾸면 됩니다.
저장 예측은 존재하며, 표시에서 숨겨진 결과도 낮은 평가 임계값 0.001의 지표에는 포함됩니다.

2026-09-12 이 컴퓨터에서 생성한 작업:

| 버전 | 학습 작업 ID | 평가 작업 ID |
|---|---|---|
| v001 | `train_20260911T235845_8673ec22` | `evaluate_20260911T235933_84c0e86e` |
| v002 | `train_20260912T000125_d31ae9ca` | `evaluate_20260912T000207_e97b0a36` |

작업 ID 시각은 UTC입니다. 이 실행의 v002 결과를 바로 열려면:

```shell
vloop evaluate --config .vloop/toy-pennfudan/repo/project.yaml \
  --view evaluate_20260912T000207_e97b0a36
```

박스 mAP는 **0.064922 → 0.137064**, 마스크 mAP는 **0.070934 → 0.087104**였습니다.
각각 같은 v001 val의 두 모델 결과이며, 학습 중 RF-DETR가 기록하는 validation 지표와는
계산 경로가 다릅니다. 모델 간 비교에는 같은 `evaluate` 설정의 결과를 사용합니다.

8장·2 epochs·val 1장의 실행은 연결된 기능과 기록·복원을 확인하기 위한 예제입니다.
공개 데이터가 사전학습에 포함됐을 가능성도 배제하지 않으므로 성능 벤치마크로 해석하지 않습니다.
v002 지표가 반드시 올라가야 성공하는 테스트도 아닙니다. 실제 개선 여부는 충분한 독립 val과
도메인 데이터로 별도 확인해야 합니다.

## 남기는 파일

```text
.vloop/toy-pennfudan/
├── source/         # 원본 ZIP, 선택한 사진·정답, 선택 목록과 해시
├── repo/           # 코드 스냅샷, 예제 YAML, 데이터 버전 Git 태그
├── input/          # ingest 입력 8장
├── state/          # 관리 이미지, 예측, 검수 DB, DVC cache/pool/restored, MLflow
├── dvc-remote/     # 두 확정 버전의 DVC 원격 저장소 역할
├── logs/           # 단계별 CLI 출력
├── progress.json   # 완료한 예제 단계와 각 명령 보고서
└── result.json     # 최종 두 모델 비교와 실행 기록
```

이 폴더는 메인 Git에서 제외됩니다. 모델·데이터·DB는 Git에 올리지 않습니다. 예제를 완전히
새로 실행하려면 새 `--workspace` 폴더의 `source/`에 ZIP을 준비하고 해당 경로를 지정합니다.
이번 실행 후 toy 폴더의 디스크 사용량은 약 1.9 GB였습니다. 이미지 8장 외에 두 모델의
최적 가중치·재개 체크포인트와 DB를 보관한 크기이며, 기존 공용 SAM 3 가중치는 재사용합니다.
