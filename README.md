# vision-loop-boilerplate

A boilerplate for an iterative computer vision pipeline covering auto-labeling, dataset versioning, training, and evaluation using FiftyOne, DVC, PyTorch, and MLflow

로컬 이미지의 자동 라벨링 → 검수 → 데이터 버전 저장 → 학습 → 평가를 반복하는 프로젝트입니다.
원본 [구현 계획](docs/PLAN.md)과 현재 [진행 상태](docs/PROGRESS.md)를 함께 관리합니다.

현재 실행 가능한 명령은 `doctor`, `ingest`, `autolabel`, `review`입니다. RTX 5060 Laptop
8 GB에서 SAM 3 추론·재개를 검증했고, FiftyOne 브라우저에서 마스크 수정·승인·재검수를
확인했습니다. DVC 릴리스, RF-DETR 학습, MLflow 평가는 후속 단계입니다.

## Requirement

- Linux, Python 3.12 가상환경
- GPU 검증: PyTorch 2.10.0 + CUDA 12.8, torchvision 0.25.0
- 초기 호환성 기준: FiftyOne 1.21.0, RF-DETR 1.8.2, MLflow 3.16.0

```shell
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[dev,autolabel]'
```

`autolabel` extra는 FiftyOne과 SAM 3 실행에 필요한 주변 라이브러리를 설치합니다.
SAM 3 패키지 자체와 체크포인트는 포함하지 않으므로 아래 소스 설치 절차도 필요합니다.
검증한 NumPy 1.26.4, OpenCV 4.11.0.86 등의 버전을 지정했습니다.

`pipeline` extra는 FiftyOne·RF-DETR·MLflow·DVC를 설치하며, 해당 단계 작업 시 추가합니다.
FiftyOne은 두 extra에 같은 버전으로 선언되어 있어 함께 선택해도 중복 설치되지 않습니다.

```shell
python -m pip install -e '.[pipeline]'
```

`pyproject.toml`의 버전은 초기 확인 대상입니다. 전체 파이프라인 검증 전에는 완성된 의존성
lock으로 취급하지 않습니다. 각 실행의 `dependencies.json`에는 실제 설치 버전을 저장합니다.
PyTorch wheel은 CUDA 12.8 런타임을 포함합니다. `nvidia-smi`에 표시되는 CUDA 버전과
`torch.version.cuda`는 별개의 값입니다.

## Config

```shell
cp project.example.yaml project.yaml
```

`project.yaml`에 실제 `image_dir`, `classes`의 `id`·`name`·`prompts`를 입력합니다.
예시 파일에는 실제 적용 가능한 클래스 기본값이 없습니다. `project.yaml`은 Git에서 제외합니다.

- 기존 DACON 프로젝트의 `argparse` + `dataclass Config` + 평탄한 YAML 형식을 따릅니다.
- 상대 경로는 YAML 파일이 있는 폴더를 기준으로 해석합니다.
- `--config`가 없으면 현재 폴더부터 Git 프로젝트 루트까지 `project.yaml`을 찾습니다.
- 설정 오타, 중복 클래스 ID·이름, 모호한 프롬프트, 잘못된 비율·임계값은 오류로 보고합니다.
- 프로젝트 클래스 ID와 연속 인덱스는 `Config.class_to_index`로 명시적으로 연결합니다.
- `storage_dir`와 `image_dir`는 겹칠 수 없습니다. DVC remote는 Git 저장소 밖에 둡니다.

## Doctor

```shell
vloop doctor
vloop doctor --config /absolute/path/project.yaml
```

Python·의존성 버전, 필수 입력, 저장 경로 쓰기 가능 여부, DVC remote 위치,
`nvidia-smi`, 실제 CUDA 행렬 연산, SAM 3 체크포인트 읽기와 SHA-256,
설치된 SAM 3 소스 커밋을 확인합니다. 파일을 읽을 수 있다는 것과 모델 추론 성공은 별도 검사입니다.
RF-DETR 가중치 접근 검사는 학습 adapter 구현 단계에 추가합니다.

검사에 실패하면 종료 코드는 `1`입니다. 현재 미입력 설정이나 미설치 후속 의존성으로 인해
실패할 수 있으며, 보고서에서 항목별 결과를 확인할 수 있습니다. 격리 환경의 GPU 접근 오류만으로
드라이버를 변경하지 말고 호스트 터미널의 `nvidia-smi`와 CUDA 연산 결과를 확인합니다.

### SAM 3 한 장 추론

[SAM 3 공식 설치 문서](https://github.com/facebookresearch/sam3#installation)를 따라 소스를 준비하고,
[체크포인트 접근](https://huggingface.co/facebook/sam3)을 신청한 뒤 본인 계정으로 내려받습니다.
토큰은 YAML이나 Git에 저장하지 않습니다.

```shell
git clone https://github.com/facebookresearch/sam3.git .vloop/vendor/sam3
git -C .vloop/vendor/sam3 rev-parse HEAD
python -m pip install -e .vloop/vendor/sam3
```

소스를 준비한 뒤 프로젝트와 SAM 3를 한 명령으로 설치할 수도 있습니다.
이미 같은 가상환경에 해당 SAM 3 소스를 설치했다면 반복할 필요는 없습니다.

```shell
python -m pip install -e '.[dev,autolabel]' -e .vloop/vendor/sam3
```

출력된 전체 커밋을 `sam3_commit`, 체크아웃 경로를 `sam3_source_dir`, 내려받은 가중치 경로를
실제 `project.yaml`의 `sam3_checkpoint`에 입력합니다. `project.example.yaml`만 수정해도
기존 `project.yaml`에는 자동 반영되지 않습니다. 위 SAM 3 설치는 PyTorch 설치 후 수행합니다.

이번에 검증한 SAM 3 커밋은 `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`입니다.
다른 환경에서 같은 구성을 준비하려면 SAM 3 저장소를 이 커밋으로 체크아웃한 뒤 설치합니다.

```yaml
sam3_checkpoint: .vloop/models/sam3/sam3.pt
sam3_source_dir: .vloop/vendor/sam3
sam3_commit: 660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7
sam3_precision: bfloat16
```

```shell
python -m pip check
```

```shell
vloop doctor --sam3-image /absolute/path/image.jpg
```

[FiftyOne 공식 SAM 3 모델](https://docs.voxel51.com/model_zoo/models/segment_anything_3_image_torch.html)의
concept 모드에 클래스별 프롬프트를 전달합니다. 공식 모델 wrapper의 단일 이미지 추론으로
정규화된 이미지·픽셀 좌표 박스·클래스 ID·인스턴스 마스크 PNG를 실행 폴더에 저장합니다.
마스크 PNG와 JSON의 COCO RLE는 모두 정규화된 이미지 전체 좌표계를 사용합니다.
모델은 로컬 체크포인트와 어휘 파일을 읽고, 이미지 1장·프롬프트 1개씩 BF16으로 추론합니다.
컴파일과 원격 가중치 자동 다운로드는 사용하지 않습니다. 이 결과는 자동 예측이며 검수 승인이 아닙니다.
이 명령은 자동 라벨링 작업의 중단·재개 기능을 제공하지 않습니다.

공식 `truck.jpg`(1800×1200)에서 트럭 1개를 검출했고, PyTorch 최대 할당 메모리는
5.26 GiB, 최대 예약 메모리는 5.51 GiB였습니다. 이는 샘플 이미지의 측정값이며 해상도·객체 수와
다른 GPU 프로그램의 사용량에 따라 달라집니다. CUDA 추론만 검증했으며 CPU 경로는 미검증입니다.

## Ingest

```shell
vloop ingest
```

이미지를 재귀적으로 탐색해 원본 파일의 SHA-256을 이미지 ID로 사용합니다.
원본은 그대로 두고, EXIF 방향을 적용한 RGB PNG를 관리 저장소에 저장합니다.
후속 단계는 이 PNG의 좌표계를 사용해야 합니다. 색상은 RGB로 통일하고 메타데이터는 제거합니다.
애니메이션·다중 페이지 이미지는 첫 장만 임의로 선택하지 않고 오류로 보고합니다.

이미지 ID·원본 경로·관리 파일 해시·크기를 SQLite 목록에 기록한 뒤 영속 FiftyOne 데이터셋에
동기화합니다. 동일 내용의 파일은 이미지 하나와 여러 원본 경로로 기록합니다.
재실행은 기존 `ground_truth`와 `review_status`를 보존합니다.
손상·미지원 파일은 `files.jsonl`에 원인을 남기고 정상 파일 처리는 계속합니다.

FiftyOne을 설치하기 전 이미지 등록만 확인하려면 다음 명령을 사용합니다.

```shell
vloop ingest --local-only
```

`--local-only` 결과에는 `FiftyOne sync: not_requested`가 표시됩니다.
이후 `vloop ingest`를 실행하면 등록된 목록을 FiftyOne에 동기화합니다.

## Autolabel

실제 이미지 폴더와 클래스 프롬프트를 입력한 뒤 작은 묶음부터 실행합니다.

```shell
vloop ingest
vloop autolabel --limit 3
vloop autolabel --resume <JOB_ID>
```

`--limit`는 새 작업의 입력을 이미지 ID 순으로 제한합니다. 생략하면 현재 등록된 전체 이미지를
대상으로 합니다. 재개할 때는 추가 이미지나 현재 YAML의 바뀐 프롬프트를 반영하지 않습니다.
새 이미지·설정으로 실험하려면 `--resume` 없이 새 작업을 시작합니다.

- 시작 시 입력 목록, 설정, 임계값, SAM 3 커밋·가중치 해시·adapter 해시·의존성을 고정합니다.
- 예측은 FiftyOne의 `pred_autolabel_<작업 ID의 날짜·접미사>` 필드와 이미지별 JSON에 저장합니다.
  `ground_truth`와 검수 상태를 자동으로 변경하지 않습니다.
- 완료된 이미지는 추론을 건너뜁니다. 추론 결과 저장 후 DB 반영만 실패했다면 저장된 결과를
  재사용합니다. 입력·결과 변조나 모델 환경 변경이 확인되면 해당 결과를 덮어쓰지 않고 오류를 냅니다.
- 객체가 없는 정상 결과는 빈 `Detections`로 저장합니다. 실패·미완료와 별도로 집계합니다.
- 이미지별 오류는 기록하고 나머지 처리를 계속합니다. GPU 메모리 부족이나 모델 로딩 실패는
  작업을 멈추고 재개할 ID를 남깁니다. 실행 전 입력 스냅샷 생성에 실패하면 새 작업으로 시작합니다.
- 같은 클래스에 프롬프트를 여러 개 지정하면 중복 객체가 나올 수 있습니다. 프롬프트별 결과를
  보존하며, 중복 정리는 후속 검수에서 수행합니다.

FiftyOne에는 박스 내부 마스크를 저장하고, JSON에는 이미지 전체 크기의 COCO RLE를 저장합니다.
변환 시 박스 위치·마스크 크기·면적을 검사하며 구멍이나 분리 영역을 단순 다각형으로 바꾸지 않습니다.

## Review

```shell
vloop review
vloop review --job-id <AUTO_LABEL_JOB_ID>
vloop review --prepare-only
vloop review --no-browser
```

등록된 데이터셋에 검수용 필드·Annotation Schema·operator·상태별 저장 뷰를 준비하고
`http://127.0.0.1:5151`에서 FiftyOne을 엽니다. 포트는 `fiftyone_port`로 설정합니다.
`--prepare-only`는 준비 후 종료하고, `--no-browser`는 브라우저 자동 실행만 생략합니다.
검수 서버는 터미널에서 계속 실행되며 `Ctrl+C`로 종료합니다.

`--job-id`를 생략하면 가장 최근에 등록된 자동 라벨링 작업을 선택하고 ID를 출력합니다.
선택한 작업에서 성공한 예측만 `ground_truth`로 복사합니다. 기존 정답이나 검수 이력이
있으면 그대로 보존하며, 한 번 초기화한 정답을 비우거나 삭제해도 재실행으로 덮어쓰지 않습니다.
원래 `pred_<job-id>`는 별도 필드로 남기고 Annotation Schema에서 읽기 전용으로 설정합니다.
실패·미완료 예측은 빈 정답으로 바꾸지 않습니다.

### 화면에서 검수하기

1. 이미지를 열고 `ground_truth`를 표시합니다. 비교가 필요하지 않으면 `pred_...` 표시는 끕니다.
2. `Annotate` 탭에서 클래스를 선택하고 박스·인스턴스 마스크를 수정합니다. 마스크는 Brush로
   추가·제거할 수 있습니다. 새 인스턴스에도 마스크가 필요하며 박스만 있는 객체는 승인되지 않습니다.
3. 자동 저장이 끝난 것을 확인하고 `Explore` 탭으로 이동합니다. 백틱 키(일반적인 키보드의
   Esc 아래)를 눌러 operator 검색창을 열고 `VLoop`를 검색합니다.
4. `VLoop: 검수 시작 / 재검수`, `VLoop: 검수 완료`, `VLoop: 검수 제외` 중 하나를 실행합니다.
   검수 완료에는 검수자 이름과 확인 체크가 필요합니다. 객체가 없는 이미지에는 빈 정답 확인도
   체크합니다. 예측이 없는 이미지의 빈 정답은 먼저 검수 시작으로 명시적으로 생성해야 합니다.

이미지를 확대한 상태에서는 그 이미지 한 장, 목록에서는 직접 선택한 이미지만 처리합니다.
아무 이미지도 선택하지 않으면 실행할 수 없습니다. 결과 창의 성공·실패 건수와 원인을 확인합니다.
FiftyOne 1.21에서 펼쳐 둔 이미지의 상태 표시가 이전 값을 유지할 수 있으므로, 이 경우 브라우저를
새로고침합니다. DB에 기록된 승인 여부는 operator 결과와 아래 승인 검증을 따릅니다.

| 저장 뷰 | 상태 |
|---|---|
| `vloop-unreviewed` | 미검수 |
| `vloop-in_progress` | 수정 중 |
| `vloop-completed` | 완료 |
| `vloop-excluded` | 제외 |

### 승인 기록과 재검수

완료 시 이미지·클래스 매핑·박스·마스크의 해시, 검수자, 시간, 예측 출처와 정답 스냅샷을
`reviews/<image-id>/<record-id>.json`에 저장합니다. DB에는 승인 기록 ID·해시와 `review_history`를
남깁니다. 재검수·제외는 현재 승인을 해제하고 이전 기록은 보존합니다. 클래스 ID는 수정한
클래스 이름에서 다시 찾으며, 예측에서 복사된 ID나 신뢰도를 정답 판단에 사용하지 않습니다.

FiftyOne의 브러시는 화면 크기에 맞춘 마스크와 소수 좌표 박스를 저장할 수 있습니다.
승인 스냅샷에서는 박스 경계를 원본 픽셀에 반올림하고 nearest-neighbor로 마스크를 변환해
전체 이미지 좌표의 COCO RLE를 만듭니다. 편집기가 저장한 원래 박스와 마스크 RLE도 함께
보존합니다. 이 변환은 자동 예측 원본을 변경하지 않습니다.

서버는 2초 간격으로 완료된 이미지의 라벨·관리 이미지·승인 기록을 검사합니다. 변경이 발견되면
`in_progress`로 되돌려 재승인을 요구합니다. 서버가 꺼져 있는 동안의 변경은 다음 실행에서
검사합니다. 현재는 완료 이미지 전체를 검사하므로 많은 이미지에서는 검사 시간이 늘어납니다.
후속 릴리스 구현은 새로 읽은 샘플에 `approved_annotation()`을 호출해 승인 내용과의 일치를
다시 확인해야 합니다. 현재 단계에는 릴리스 명령이 없습니다.

editable 설치 상태에서는 코드 수정이 반영되지만, 실행 중인 검수 서버에는 모듈이 이미 로드되어
있으므로 서버를 다시 시작해야 합니다.

## Results / Recovery

```text
.vloop/
├── catalog.sqlite3           # 이미지 ID와 원본 경로 목록
├── images/<hash-prefix>/    # 방향이 확정된 관리 이미지
├── fiftyone/db/             # 영속 검수 DB
├── fiftyone/plugins/        # review 실행 시 설치하는 프로젝트 검수 operator
├── reviews/<image-id>/      # 승인·재검수·제외 기록과 승인한 정답 스냅샷
└── runs/<job-id>/
    ├── config.json          # 절대 경로를 포함한 최종 설정
    ├── dependencies.json    # 실제 설치된 의존성 버전
    ├── report.json          # 상태, 코드 커밋, 건수, 결과 위치
    ├── files.jsonl          # ingest 파일별 결과
    ├── manifest.json       # autolabel 설정·모델·입력 해시
    ├── samples.sqlite3     # 고정 입력과 이미지별 처리 상태
    ├── predictions/        # 이미지별 박스·클래스·전체 마스크 RLE
    ├── errors/             # 이미지별 실패 원인
    └── attempts/           # 재개 이전 보고서
```

- 성공 `0`, 처리·검사 실패 `1`, 설정·호출 오류 `2`, 사용자 중단 `130`의 종료 코드를 사용합니다.
- 등록 도중 중단되면 같은 `vloop ingest` 명령을 다시 실행합니다. 기존 등록은 중복 처리됩니다.
- FiftyOne 연결에 실패해도 등록 목록은 남습니다. 원인을 해결하고 같은 명령을 실행합니다.
- 원본이 있고 관리 이미지가 사라진 경우 재등록으로 복구합니다. 관리 파일이 변경된 경우는 오류로
  보고하므로 해당 원인을 확인해야 합니다.
- 강제 종료로 `running` 보고서가 남을 수 있습니다. 등록은 `vloop ingest`를 다시 실행하고,
  자동 라벨링은 `vloop autolabel --resume JOB_ID`로 원래 작업을 이어갑니다.
- 데이터 변경 작업이 겹치면 잠금 오류를 반환합니다. 검수 서버는 검사할 때만 잠금을 사용합니다.

FiftyOne DB와 모델 저장소는 프로젝트의 `storage_dir` 아래에 둡니다. 앱 연결 주소는
`127.0.0.1`로 설정합니다. MLflow 서버를 여는 명령은 후속 단계입니다.
현재 릴리스 생성·복원 명령은 구현하지 않았으며, 절차는 해당 단계의 실제 복원 검증과 함께 추가합니다.
이미지·모델·DB를 Git에 직접 추가하지 않습니다.

## Test

```shell
python -m pytest -q
ruff check src tests
ruff format --check src tests
```

실제 FiftyOne DB를 사용하는 검증은 호스트에서 별도로 실행합니다. 임시 이미지·DB를 만들고,
다른 프로세스의 재등록 후 수정 내용 보존, 검수 승인·무효화·재승인, 빈 정답,
동시 수정 충돌과 operator 등록을 확인합니다.

```shell
VLOOP_TEST_FIFTYONE=1 python -m pytest tests/test_fiftyone_integration.py tests/test_review_integration.py -q
```

실제 SAM 3 GPU 검증은 체크포인트와 소스가 설정된 YAML 경로를 지정합니다. 프로젝트의 실제
이미지·클래스 대신 공식 샘플 2장과 별도 임시 DB를 사용합니다. 한 장 처리 후 중단하고 새
프로세스에서 재개하며, 설정 변경과 사람의 수정 내용 보존도 확인합니다.

```shell
VLOOP_TEST_SAM3_CONFIG=/absolute/path/project.yaml python -m pytest tests/test_sam3_integration.py -q -s
```

## Structure

```text
project.example.yaml
pyproject.toml
src/vloop/
├── cli.py          # argparse 진입점
├── config.py       # dataclass 설정과 검증
├── doctor.py       # 환경 검사
├── ingest.py       # 이미지 등록
├── fiftyone.py     # DB 설정과 등록 목록 동기화
├── autolabel.py    # 작업 스냅샷, 이미지별 상태, 재개
├── sam3.py         # FiftyOne SAM 3 adapter, 한 장 추론 확인
├── labels.py       # 박스·마스크 좌표 변환과 COCO RLE
├── review.py       # 검수 준비·상태 변경·승인 후 변경 감지
├── approval.py     # 정답 스냅샷·해시와 승인 일치 검사
├── review_operators.py # FiftyOne 검수 operator
├── review_plugin/  # 프로젝트에 설치할 플러그인 등록 파일
└── runtime.py      # 해시, 실행 기록, 잠금
tests/
docs/
```
