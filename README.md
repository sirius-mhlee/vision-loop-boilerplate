# vision-loop-boilerplate

A boilerplate for an iterative computer vision pipeline covering auto-labeling, dataset versioning, training, and evaluation using FiftyOne, DVC, PyTorch, and MLflow

로컬 이미지의 자동 라벨링 → 검수 → 데이터 버전 저장 → 학습 → 평가를 반복하는 프로젝트입니다.
원본 [구현 계획](docs/PLAN.md)과 현재 [진행 상태](docs/PROGRESS.md)를 함께 관리합니다.

현재 실행 가능한 명령은 `doctor`, `ingest`입니다. SAM 3 한 장 추론 확인 경로도 포함했으며,
실제 체크포인트를 사용한 검증은 아직 필요합니다. 자동 라벨링 작업·재개, 검수 operator,
DVC 릴리스, RF-DETR 학습, MLflow 평가 명령은 후속 단계입니다.

## Requirement

- Linux, Python 3.12 가상환경
- GPU 검증: PyTorch 2.10.0 + CUDA 12.8, torchvision 0.25.0
- 초기 호환성 기준: FiftyOne 1.21.0, RF-DETR 1.8.2, MLflow 3.16.0

```shell
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,vision]'
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
```

RF-DETR·MLflow·DVC는 해당 단계 작업 시 설치합니다.

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

출력된 전체 커밋을 `sam3_commit`, 체크아웃 경로를 `sam3_source_dir`, 내려받은 가중치 경로를
`sam3_checkpoint`에 입력합니다. 이 시점의 커밋은 후보이며 한 장 추론과 통합 검증 후 고정합니다.
위 SAM 3 설치는 PyTorch 설치 후 수행합니다. 의존성 충돌이 보고되면 먼저 해결합니다.

```shell
vloop doctor --sam3-image /absolute/path/image.jpg
```

[FiftyOne 공식 SAM 3 모델](https://docs.voxel51.com/model_zoo/models/segment_anything_3_image_torch.html)의
concept 모드에 클래스별 프롬프트를 전달합니다. 임시 FiftyOne 데이터셋에서 배치 1로 실행하며,
정규화된 이미지·픽셀 좌표 박스·클래스 ID·인스턴스 마스크 PNG를 실행 폴더에 저장합니다.
마스크 PNG의 좌표계는 해당 박스 내부입니다. 이 결과는 자동 예측이며 검수 승인이 아닙니다.
이 명령은 자동 라벨링 작업의 중단·재개 기능을 제공하지 않습니다.

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

## Results / Recovery

```text
.vloop/
├── catalog.sqlite3           # 이미지 ID와 원본 경로 목록
├── images/<hash-prefix>/    # 방향이 확정된 관리 이미지
├── fiftyone/db/             # 영속 검수 DB
└── runs/<job-id>/
    ├── config.json          # 절대 경로를 포함한 최종 설정
    ├── dependencies.json    # 실제 설치된 의존성 버전
    ├── report.json          # 상태, 코드 커밋, 건수, 결과 위치
    └── files.jsonl          # ingest 파일별 결과
```

- 성공 `0`, 처리·검사 실패 `1`, 설정·호출 오류 `2`, 사용자 중단 `130`의 종료 코드를 사용합니다.
- 등록 도중 중단되면 같은 `vloop ingest` 명령을 다시 실행합니다. 기존 등록은 중복 처리됩니다.
- FiftyOne 연결에 실패해도 등록 목록은 남습니다. 원인을 해결하고 같은 명령을 실행합니다.
- 원본이 있고 관리 이미지가 사라진 경우 재등록으로 복구합니다. 관리 파일이 변경된 경우는 오류로
  보고하므로 해당 원인을 확인해야 합니다.
- 강제 종료로 `running` 보고서가 남을 수 있습니다. 재실행은 새 작업 ID를 만들고 이미 등록된
  이미지부터 확인합니다. 이는 앞으로 구현할 `autolabel --resume JOB_ID`와 별개입니다.
- 한 프로젝트에서 실행을 겹치면 잠금 오류를 반환합니다.

FiftyOne DB와 모델 저장소는 프로젝트의 `storage_dir` 아래에 둡니다. 앱 연결 주소는
`127.0.0.1`로 설정합니다. 검수 화면과 MLflow 서버를 여는 명령은 후속 단계입니다.
현재 릴리스 생성·복원 명령은 구현하지 않았으며, 절차는 해당 단계의 실제 복원 검증과 함께 추가합니다.
이미지·모델·DB를 Git에 직접 추가하지 않습니다.

## Test

```shell
python -m pytest -q
ruff check src tests
ruff format --check src tests
```

실제 FiftyOne DB를 사용하는 검증은 호스트에서 별도로 실행합니다. 임시 이미지·DB를 만들고,
다른 프로세스의 재등록 후에도 사람이 수정한 라벨과 상태가 남는지 확인합니다.

```shell
VLOOP_TEST_FIFTYONE=1 python -m pytest tests/test_fiftyone_integration.py -q
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
├── sam3.py         # 한 장 추론 확인
└── runtime.py      # 해시, 실행 기록, 잠금
tests/
docs/
```
