# vision-loop-boilerplate

A boilerplate for an iterative computer vision pipeline covering auto-labeling, dataset versioning, training, and evaluation using FiftyOne, DVC, PyTorch, and MLflow

로컬 이미지의 자동 라벨링 → 검수 → 데이터 버전 저장 → 학습 → 평가를 반복하는 프로젝트입니다.
원본 [구현 계획](docs/PLAN.md)과 현재 [진행 상태](docs/PROGRESS.md)를 함께 관리합니다.

현재 실행 가능한 명령은 `doctor`, `ingest`, `autolabel`, `review`, `review-batch`,
`review-audit`, `release`, `restore`, `train`, `evaluate`, `experiments`입니다. RTX 5060 Laptop
8 GB에서 SAM 3 추론·재개를 검증했고, FiftyOne 브라우저에서 마스크 수정·승인·재검수를
확인했습니다. 승인된 COCO 데이터를 DVC에 저장하고 Git 태그로 복원하는 경로도 구현했습니다.
RF-DETR 학습·MLflow 기록·체크포인트 재개와 저장 모델의 새 프로세스 복원을 검증했습니다.
평가 CLI와 박스·마스크 지표, FiftyOne 분석 화면을 연결했습니다.
공개 Penn-Fudan 사진 8장으로 `v001`·`v002` 학습과 동일 val 비교까지 실행했습니다.
이전 [자동 통합 검증](docs/TOY.md)은 제공된 정답 마스크를 사용했습니다. 직접 사진만 받아
SAM 3 예측을 사람이 검수하는 절차는 아래 [Penn-Fudan 직접 실습](#penn-fudan-직접-실습)에 있습니다.
1~7단계 핵심 기능은 구현됐으며, Linux/Python 3.12/CUDA 12.8 의존성을 고정했습니다.
다른 컴퓨터의 GPU·검수 화면 검증, 대규모 실측과 실제 도메인 성능 검증은 남아 있습니다.

## Requirement

- Linux, Python 3.12 가상환경, Git·curl·unzip
- CUDA를 사용할 NVIDIA GPU와 호환 드라이버. 호스트에서 `nvidia-smi`가 동작해야 합니다.
- GPU 검증: PyTorch 2.10.0 + CUDA 12.8, torchvision 0.25.0
- 초기 호환성 기준: FiftyOne 1.21.0, RF-DETR 1.8.2, MLflow 3.16.0

```shell
git clone https://github.com/sirius-mhlee/vision-loop-boilerplate.git
cd vision-loop-boilerplate
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/bootstrap.lock
python -m pip install --no-build-isolation --no-deps -r requirements/linux-py312-cu128.lock
python -m pip install --no-build-isolation --no-deps -e '.[dev,autolabel,pipeline]'
```

`autolabel` extra는 FiftyOne과 SAM 3 실행에 필요한 주변 라이브러리를 설치합니다.
SAM 3 패키지 자체와 체크포인트는 포함하지 않으므로 아래 소스 설치 절차도 필요합니다.
검증한 NumPy 1.26.4, OpenCV 4.11.0.86 등의 버전을 지정했습니다. 위 lock에는 SAM 3의
주변 의존성도 포함하며, PyTorch `2.10.0+cu128`과 torchvision `0.25.0+cu128`도 함께 설치합니다.

`pipeline` extra는 FiftyOne·RF-DETR·MLflow·DVC를 설치합니다. 위 명령은 전체 실습용 의존성을
함께 설치합니다. 일부 extra만 선택해 설치하려면 아래처럼 lock을 `-c`로 적용합니다.
FiftyOne은 두 extra에 같은 버전으로 선언되어 있어 함께 선택해도 중복 설치되지 않습니다.
FiftyOne 하위 패키지 ETA 0.17과 MLflow를 같은 프로세스에서 사용하기 위해
`importlib-metadata==7.2.1`도 고정합니다. 8 이상에서는 누락된 메타데이터 키 조회가
`KeyError`로 바뀌어 현재 ETA의 `author` 조회가 실패합니다.
[importlib-metadata 변경 기록](https://importlib-metadata.readthedocs.io/en/latest/history.html#v8-0-0)

이 고정은 현재 조합의 실행 오류를 막는 호환성 조치입니다. ETA 자체의 메타데이터 조회가
수정된 것은 아닙니다. 이 핀을 올리려면 ETA의 수정 여부를 확인하고 FiftyOne·MLflow를 같은
프로세스에서 사용하는 평가 통합 테스트를 통과해야 합니다.

```shell
# pipeline을 아직 설치하지 않은 환경에서만 추가
python -m pip install --no-build-isolation -c requirements/linux-py312-cu128.lock -e '.[pipeline]'
```

`pyproject.toml`은 패키지와 extra의 의존성을 선언하고, `requirements/linux-py312-cu128.lock`은
검증 환경의 외부 패키지 285개를 정확한 버전으로 고정합니다. `bootstrap.lock`은 pip·setuptools·wheel을
고정합니다. 전체 의존성을 먼저 설치했으므로 소스 설치의 `--no-deps`는 의존성을 다시 선택하지 않으며,
`--no-build-isolation`은 미리 설치한 빌드 도구를 사용합니다. `-e`의 코드 수정 반영 동작은 그대로입니다.
설치 범위·갱신 방법은 [의존성 고정 안내](requirements/README.md)에 있습니다.
각 실행의 `dependencies.json`에도 실제 설치 버전을 계속 저장합니다.
PyTorch wheel은 CUDA 12.8 런타임을 포함합니다. `nvidia-smi`에 표시되는 CUDA 버전과
`torch.version.cuda`는 별개의 값입니다.

## Config

```shell
cp project.example.yaml project.yaml
```

`project.example.yaml`에는 Penn-Fudan 사진 경로 `data/PennFudanPed/PNGImages`, 데이터셋 이름
`penn-fudan`, 클래스 ID `1`·이름 `person`·프롬프트 `[person]`을 채웠습니다. 아래 다운로드·압축
해제 후 사용할 수 있습니다. 직접 수집한 데이터에는 `image_dir`, `dataset_name`, `classes`를
맞게 바꿉니다. 기존 DB의 클래스 매핑을 바꾸는 절차는 아니므로 다른 프로젝트는 별도 저장소를
사용합니다. `project.yaml`과 `data/`는 Git에서 제외합니다.

- 기존 DACON 프로젝트의 `argparse` + `dataclass Config` + 평탄한 YAML 형식을 따릅니다.
- 상대 경로는 YAML 파일이 있는 폴더를 기준으로 해석합니다.
- `--config`가 없으면 현재 폴더부터 Git 프로젝트 루트까지 `project.yaml`을 찾습니다.
- 설정 오타, 중복 클래스 ID·이름, 모호한 프롬프트, 잘못된 비율·임계값은 오류로 보고합니다.
- 프로젝트 클래스 ID와 연속 인덱스는 `Config.class_to_index`로 연결합니다. YAML에 적은 순서와
  관계없이 클래스 ID 오름차순으로 0부터 배정하며, COCO 릴리스와 RF-DETR에서 같은 매핑을 씁니다.
- `storage_dir`와 `image_dir`는 겹칠 수 없습니다. DVC remote는 Git 저장소 밖에 둡니다.

## Doctor

```shell
vloop doctor
vloop doctor --config /absolute/path/project.yaml
```

Python·의존성 버전, 필수 입력, 저장 경로 쓰기 가능 여부, DVC remote 위치,
`nvidia-smi`, 실제 CUDA 행렬 연산, SAM 3 체크포인트 읽기와 SHA-256,
설치된 SAM 3 소스 커밋과 로컬 RF-DETR 가중치의 읽기·SHA-256을 확인합니다.
파일을 읽을 수 있다는 것과 모델 추론 성공은 별도 검사입니다. RF-DETR 가중치가 없으면
`rfdetr_checkpoint` 검사는 실패하며, `train_checkpoint: null`인 첫 학습에서 공식 가중치를 받습니다.

검사에 실패하면 종료 코드는 `1`입니다. 현재 미입력 설정이나 미설치 후속 의존성으로 인해
실패할 수 있으며, 보고서에서 항목별 결과를 확인할 수 있습니다. 격리 환경의 GPU 접근 오류만으로
드라이버를 변경하지 말고 호스트 터미널의 `nvidia-smi`와 CUDA 연산 결과를 확인합니다.

### SAM 3 한 장 추론

[SAM 3 공식 설치 문서](https://github.com/facebookresearch/sam3#installation)를 따라 소스를 준비하고,
[체크포인트 접근](https://huggingface.co/facebook/sam3)을 신청한 뒤 본인 계정으로 내려받습니다.
토큰은 YAML이나 Git에 저장하지 않습니다.

```shell
git clone https://github.com/facebookresearch/sam3.git .vloop/vendor/sam3
git -C .vloop/vendor/sam3 checkout 660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7
python -m pip install --no-build-isolation --no-deps -e .vloop/vendor/sam3
```

소스를 준비한 뒤 프로젝트와 SAM 3를 한 명령으로 설치할 수도 있습니다.
이미 같은 가상환경에 해당 SAM 3 소스를 설치했다면 반복할 필요는 없습니다.

```shell
python -m pip install --no-build-isolation -c requirements/linux-py312-cu128.lock \
  -e '.[dev,autolabel]' -e .vloop/vendor/sam3
```

예제 YAML의 `sam3_commit`, `sam3_source_dir`, `sam3_checkpoint`는 위 소스와 아래 가중치 경로에
맞춰져 있습니다. 다른 경로를 쓰면 실제 `project.yaml`도 수정합니다. `project.example.yaml`만
수정해도 기존 `project.yaml`에는 자동 반영되지 않습니다. 위 SAM 3 설치는 PyTorch 설치 후 수행합니다.

[SAM 3 체크포인트 페이지](https://huggingface.co/facebook/sam3)에서 접근 권한을 받은 계정으로
로그인한 뒤 가중치를 받습니다. 다른 컴퓨터에서도 이 인증 또는 이미 받은 파일의 복사가 필요합니다.

```shell
hf auth login
hf download facebook/sam3 sam3.pt --local-dir .vloop/models/sam3
```

이미 받은 `sam3.pt`가 있으면 다운로드 대신 `.vloop/models/sam3/sam3.pt`에 복사합니다.
인증 토큰이나 가중치를 Git에 넣지 않습니다.

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
# 위의 전체 환경과 두 소스 패키지를 설치한 경우
python requirements/check.py
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

## Penn-Fudan 직접 실습

**공개 정답을 읽지 않고 사진만 받아 `ingest`부터 직접 실행하는 절차**입니다.
위 Requirement → Config → SAM 3 설치·체크포인트 준비를 마친 뒤, clone한 프로젝트 루트에서
실행합니다. `project.example.yaml`을 `project.yaml`로 복사하면 아래 데이터 경로와 `person`
클래스가 이미 설정돼 있습니다. 기존 8장짜리 자동 통합 검증 스크립트는 이 실습에 사용하지 않습니다.

### 1. 사진 다운로드와 압축 해제

[Penn-Fudan 공식 데이터](https://www.cis.upenn.edu/~jshi/ped_html/)의 ZIP은 약 51 MB이며 사진
170장이 들어 있습니다. ZIP에는 정답도 있지만 **`PNGImages`의 사진만 선택해서 압축을 풉니다.**
`PedMasks`와 `Annotation`은 추출하지 않으며 파이프라인에서 읽지 않습니다.

```shell
mkdir -p data
curl -fL --retry 3 \
  -o data/PennFudanPed.zip \
  https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
unzip -n data/PennFudanPed.zip 'PennFudanPed/PNGImages/*.png' -d data
```

`-n`은 같은 파일이 있으면 덮어쓰지 않습니다. 설정과 실제 경로는 다음과 같습니다.

```yaml
image_dir: data/PennFudanPed/PNGImages
dataset_name: penn-fudan
classes:
  - id: 1
    name: person
    prompts: [person]
```

```text
vision-loop-boilerplate/
├── project.yaml
├── data/
│   ├── PennFudanPed.zip
│   └── PennFudanPed/PNGImages/    # 입력 사진 170장
└── .vloop/                      # ingest 이후 관리 이미지·예측·DB·모델·DVC cache
```

`data/`와 `.vloop/`는 이미 Git ignore 대상입니다. 데이터를 Git에 추가하거나 별도 예제 Git
저장소를 만들 필요가 없습니다. 이 실습의 릴리스 태그는 지금 clone한 Git 저장소에 생성됩니다.

```shell
vloop doctor
```

처음에는 RF-DETR 초기 가중치가 없어 `rfdetr_checkpoint`만 실패할 수 있습니다.
`train_checkpoint: null`이면 첫 `train`에서 공식 가중치를 내려받습니다. 다른 항목의 실패는
해당 설정·설치를 해결한 뒤 진행합니다. 짧게 실행하려면 `project.yaml`의 `epochs: 30`을
`epochs: 2`로 바꿉니다. 2 epochs는 기능 확인용이며 충분한 학습을 의미하지 않습니다.

### 2. 등록과 자동 라벨링

```shell
vloop ingest
vloop autolabel
```

170장 전체를 처리합니다. `autolabel` 출력의 성공·실패·빈 예측 건수를 확인합니다.
실패·중단이 있으면 출력된 작업 ID로 `vloop autolabel --resume AUTO_LABEL_JOB_ID`를 사용합니다.
다음 블록의 `autolabel_...`에는 실제 출력된 `Job ID`를 넣습니다.

```shell
AUTO_LABEL_JOB_ID=autolabel_...
vloop review-batch --job-id "$AUTO_LABEL_JOB_ID" \
  --min-confidence 0.5 --sample-rate 0.2 --actor "$USER"
```

`sample-rate`는 `review-batch`의 옵션입니다. 실습에서는 `autolabel_confidence: 0.5`와 같은
채택 기준을 사용합니다. 성공 예측이 저장된 미검수 후보 전체에서 신뢰도와 무관하게 약 20%를
먼저 `sample`로 뽑습니다. 빈 예측·저신뢰도 예측도 뽑히면 `sample`에 들어갑니다.
뽑히지 않은 이미지는 빈 예측이면 `empty`, 기준 미달이면 `low_confidence`, 나머지는 `accept`로
나눕니다. 정확히 34장을 뽑는 옵션은 아닙니다. 예측에 없는 사람은 신뢰도만으로 알 수 없으므로
표본에서는 누락도 확인합니다. 추론 실패와 이미 승인·편집한 데이터는 새 표본 후보가 아닙니다.

출력의 `decisions`를 확인한 뒤 적용합니다. 아래 `review_batch_...`는 방금 만든 **미리보기의
작업 ID**로 바꿉니다. 자동 라벨링 작업 ID와 구별합니다.

```shell
REVIEW_BATCH_JOB_ID=review_batch_...
vloop review-batch --resume "$REVIEW_BATCH_JOB_ID" --apply
```

적용하면 `accept` 대상은 `auto_accepted`가 되고, `sample`은 아직 승인되지 않은 검수 대상으로
남습니다. 미리보기만 만들고 적용하지 않으면 이 상태 변경과 검수 큐 등록이 일어나지 않습니다.

### 3. 표본을 사람이 검수

```shell
vloop review --job-id "$AUTO_LABEL_JOB_ID" --queue sample --limit 170
```

`--limit 170`은 이번 작은 실습에서 검수용 정답을 준비할 최대 장수입니다. 화면에는 `sample`
큐만 열립니다. 예측은 `pred_...`에 보관되고 편집할 라벨은 `ground_truth`입니다.

1. 이미지를 열고 백틱 키로 operator 검색창을 열어 `VLoop: 검수 시작 / 재검수`를 실행합니다.
2. `Annotate`에서 `ground_truth`의 사람별 마스크·박스·클래스를 확인하고 누락·오탐을 수정합니다.
   새 사람을 추가할 때도 마스크가 필요합니다.
3. 저장된 결과를 확인한 뒤 `Explore`에서 `VLoop: 검수 완료`를 실행합니다. 검수자와 확인
   체크를 입력합니다. 승인되면 `completed`가 되고 sample 큐에서 빠집니다.
4. 남은 표본을 같은 방식으로 검수하고, 터미널에서 `Ctrl+C`로 검수 서버를 닫습니다.

표본 전체를 확인한 뒤 선택해서 승인할 수는 있지만, 단순히 전체 선택으로 확인 과정을 대체하지
않습니다. 한 번의 수동 operator는 최대 100장입니다. 완료 목록은 `vloop-completed` 저장 뷰에서
다시 볼 수 있습니다.

이번 실습에서 `sample`만 처리하면 `low_confidence`·`empty` 등 미승인 이미지는 릴리스에서
빠집니다. 포함하고 싶을 때만 해당 큐도 열어 검수합니다. `empty`는 실제 사람이 없는지 확인하고,
누락된 사람이 있으면 정답을 추가해야 합니다. `preserve`는 기존 승인·편집 내용을 보존한 경우도
있으므로 사유를 확인하고, `error`나 적용 실패는 수정 전까지 정상 처리로 간주하지 않습니다.

```shell
# 해당 이미지까지 포함하고 싶은 경우
vloop review --queue low_confidence --limit 170
vloop review --queue empty --limit 170
```

### 4. 분할 확인과 릴리스

이 실습은 첫 릴리스에 **`--manual-to-val-test`**를 지정합니다. 자동 채택은 train에 넣고,
신규 수동 승인 이미지는 `split_ratios`의 val:test 비율로만 나눕니다. 예제의 `[0.8, 0.1, 0.1]`은
수동 승인분에 대해 1:1입니다. 후보 1,000장 중 표본 200장을 수동 승인하고 나머지 800장이
모두 자동 채택 기준을 통과했다면 예상 구성은 다음과 같습니다.

| 승인 종류 | train | val | test |
|---|---:|---:|---:|
| 자동 채택 약 800장 | 800 | 0 | 0 |
| 수동 승인 약 200장 | 0 | 약 100 | 약 100 |
| 합계 | 약 800 | 약 100 | 약 100 |

실제 비율은 표본 수·미승인 이미지·그룹 크기에 따라 달라집니다. 표본 외 `low_confidence`·`empty`를
추가로 수동 승인하면 그 신규 이미지도 val/test 후보가 되므로 평가 데이터 비율이 커질 수 있습니다.
작은 데이터에서는 val/test가 비어 있을 수도 있습니다. `--include-auto-accepted`만 사용하고
`--manual-to-val-test`를 생략하면 수동 승인분도 80/10/10으로 나뉘어 위 예시는 약 96/2/2가 됩니다.
두 옵션을 모두 생략하면 수동 승인 약 200장만 포함해 약 160/20/20장으로 나눕니다.

자동 채택도 train에 포함하려면 `--include-auto-accepted`가 필요합니다.
먼저 분할·빈 정답·클래스별 객체 수를 확인합니다.
데이터 태그를 만들려면 이 컴퓨터의 Git 이름·이메일이 설정돼 있어야 합니다.

```shell
vloop release --version v001 --include-auto-accepted --manual-to-val-test --prepare-only
```

train/val 모두에 이미지와 정답 객체가 있는지 확인합니다. `test`는 최종 평가용이므로 실제로
test를 평가하려면 해당 이미지도 있어야 합니다. 부족하면 `vloop review --limit 170`의
`vloop-auto_accepted` 등에서 추가로 직접 검수하거나 데이터를 보충합니다.
**준비 이후 검수를 바꿨다면 같은 `--version v001 ... --prepare-only` 명령으로 새 준비 작업을
만들어 다시 확인합니다.** 이전 준비 작업의 스냅샷은 바뀌지 않으므로 이전 ID로 재개하지 않습니다.
아직 버전 태그를 생성하지 않은 상태에서만 같은 버전의 새 준비 작업을 만들 수 있습니다.

구성이 확인된 **최신 준비 작업 ID**로 저장·업로드·태그 생성을 완료합니다.

```shell
RELEASE_JOB_ID=release_...
vloop release --resume "$RELEASE_JOB_ID"
```

이제 `dataset/v001` 태그와 DVC remote `../vision-loop-dvc-remote`에 확정 데이터가 남습니다.
`--include-auto-accepted`를 빼면 수동 승인한 표본만 릴리스됩니다. 미검수 이미지가 자동으로
승인되지는 않습니다. 이 옵션 조합에서 첫 릴리스의 train이 비면 학습할 수 없습니다.
`review-audit`는 이 절차의 필수 선행 명령이 아닙니다.

### 5. 학습과 평가 화면

```shell
vloop train --dataset-version v001
```

clone한 커밋 그대로라면 학습할 수 있습니다. 소스나 README, `project.example.yaml` 등 Git 추적
파일을 수정했다면 먼저 자신의 변경을 커밋해야 합니다. Git에서 제외된 `project.yaml`, 데이터,
실행 결과는 커밋하지 않아도 되며 최종 설정은 작업에 기록됩니다.

```shell
TRAIN_JOB_ID=train_...
vloop evaluate --job-id "$TRAIN_JOB_ID" --dataset-version v001 --split val

EVALUATE_JOB_ID=evaluate_...
vloop evaluate --view "$EVALUATE_JOB_ID"
```

각 변수는 바로 앞 명령이 출력한 실제 작업 ID로 바꿉니다. `--view`는 저장 결과를 열며 다시
추론하지 않습니다. 예측이 기본 `display_confidence` 뷰에서 안 보이면 `all` 뷰로 바꿔 봅니다.
짧은 학습은 예측 신뢰도가 낮을 수 있습니다. `Ctrl+C`로 화면 서버를 닫은 뒤 MLflow도 열 수 있습니다.

```shell
vloop experiments
# 서버만 띄우고 주소를 직접 열려면 위 화면 명령에 --no-browser 추가
```

### 6. v002로 반복하기

val 오류를 참고해 **train 데이터의** 누락·오탐을 재검수하거나 새 학습 사진을 추가합니다.
정답은 여전히 SAM 3 예측과 사람의 수정으로 만들며 Penn-Fudan의 제공 정답을 읽지 않습니다.
새 사진이 있다면 `ingest` → `autolabel` → 새 `review-batch` 미리보기·적용 → `review`를
반복합니다. 기존 승인·편집은 보존됩니다.

`--manual-to-val-test`는 릴리스마다 선택합니다. v002에서도 신규 수동 승인 그룹을 val/test에
넣으려면 아래처럼 다시 지정합니다. 사람이 새로 라벨링한 데이터도 학습에 배분하려면 이 옵션을
생략해 신규 수동 그룹을 train/val/test로 나눕니다. 기존 train은 사람이 수정·재승인해도 train에
남습니다. 평가 데이터가 늘어도 모델끼리 비교할 때는 아래처럼 동일한 v001 val을 명시합니다.

```shell
vloop release --version v002 --include-auto-accepted --manual-to-val-test
vloop train --dataset-version v002

TRAIN_V002_JOB_ID=train_...
vloop evaluate --job-id "$TRAIN_V002_JOB_ID" --dataset-version v001 --split val
```

두 모델을 같은 v001 val로 평가해 `comparison_id`와 지표를 비교합니다. 기존 데이터의 split은
유지되고, 새 그룹만 이번 릴리스의 정책으로 분할합니다. test 결과는 반복 개선에 사용하지 않고
최종 모델을 정한 뒤 명시적으로 실행합니다.

```shell
vloop evaluate --job-id "$TRAIN_V002_JOB_ID" --dataset-version v001 --split test
# 확정 데이터 자체를 별도로 다시 복원할 때
vloop restore --version v001
```

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
vloop autolabel --resume JOB_ID
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

이 문서의 `AUTO_LABEL_JOB_ID`, `REVIEW_BATCH_JOB_ID`, `TRAIN_JOB_ID` 같은 표기는 각 명령이
출력한 실제 작업 ID로 바꿔 입력합니다. `$AUTO_LABEL_JOB_ID`처럼 `$`가 붙은 예제는 앞에서
해당 셸 변수에 작업 ID를 저장한 경우입니다.

```shell
vloop review
vloop review --job-id AUTO_LABEL_JOB_ID
vloop review --prepare-only
vloop review --no-browser
vloop review --queue sample --limit 100
```

등록된 데이터셋에 검수용 필드·Annotation Schema·operator·상태별 저장 뷰를 준비하고
`http://127.0.0.1:5151`에서 FiftyOne을 엽니다. 포트는 `fiftyone_port`로 설정합니다.

| 명령 | 검수용 데이터 준비 | 검수 웹서버 실행 | 브라우저 자동 열기 |
|---|---|---|---|
| `vloop review` | 수행 | 실행 유지 | 열기 |
| `vloop review --no-browser` | 수행 | 실행 유지 | 생략 |
| `vloop review --prepare-only` | 수행 후 명령 종료 | 생략 | 생략 |

일반적인 수동 검수에는 `review` 또는 `review --no-browser`를 사용하면 됩니다.
두 명령 모두 준비 작업을 포함하므로 `--prepare-only`를 먼저 실행할 필요는 없습니다.
`--no-browser`에서도 웹서버와 주기적인 승인 확인은 계속 실행됩니다. 이미 열린 브라우저나
직접 입력한 URL로 접속할 때 사용하며, 터미널에서 `Ctrl+C`를 누르면 검수 서버가 종료됩니다.

`--prepare-only`는 서버를 켜두지 않고 검수 필드·설정·초기 정답을 DB에 반영한 뒤 끝내야 하는
자동화 스크립트나 테스트용 보조 옵션입니다. 조회나 미리보기만 하는 옵션은 아니며,
일반적인 검수 절차의 필수 단계도 아닙니다.

`--job-id`를 생략하면 가장 최근에 등록된 자동 라벨링 작업을 선택하고 ID를 출력합니다.
선택한 작업에서 성공한 예측 중 아직 초기화하지 않은 이미지 **최대 100장**을 준비합니다.
범위는 `--limit` 또는 YAML의 `review_prepare_limit`로 바꿉니다. 출력 건수는 이번에 살펴본
범위의 결과입니다. 준비되지 않은 이미지는 `검수 시작` operator를 실행할 때 해당 예측을
복사합니다. 기존 정답이나 검수 이력이
있으면 그대로 보존하며, 한 번 초기화한 정답을 비우거나 삭제해도 재실행으로 덮어쓰지 않습니다.
원래 `pred_<job-id>`는 별도 필드로 남기고 Annotation Schema에서 읽기 전용으로 설정합니다.
실패·미완료 예측은 빈 정답으로 바꾸지 않습니다.

`--job-id`는 아직 준비하지 않은 정답에 복사할 예측의 기본 출처입니다. 화면을 그 작업의
이미지만으로 제한하거나, 이미 준비한 정답을 다른 모델의 예측으로 교체하는 옵션은 아닙니다.
`review-batch`가 만든 queue에는 이미지별 출처 작업이 저장되어 있으며, 이 출처가 `--job-id`보다
우선합니다. 따라서 미리보기를 적용한 뒤 `review --queue low_confidence`처럼 열 때는 작업 ID를
생략해도 해당 queue를 만든 예측을 사용합니다.

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
수동 operator는 한 번에 최대 100장까지 처리합니다. 아무 이미지도 선택하지 않으면 실행할 수
없습니다. 대량 예측 채택에는 아래 CLI를 사용합니다. 결과 창의 성공·실패 건수와 원인을 확인합니다.
FiftyOne 1.21에서 펼쳐 둔 이미지의 상태 표시가 이전 값을 유지할 수 있으므로, 이 경우 브라우저를
새로고침합니다. DB에 기록된 승인 여부는 operator 결과와 아래 승인 검증을 따릅니다.

| 저장 뷰 | 상태 |
|---|---|
| `vloop-unreviewed` | 미검수 |
| `vloop-in_progress` | 수정 중 |
| `vloop-completed` | 사람이 검수 완료 |
| `vloop-auto_accepted` | 자동 예측 일괄 채택 |
| `vloop-excluded` | 제외 |
| `vloop-queue-sample` | 전체 미검수 후보에서 신뢰도와 무관하게 뽑은 표본; 빈 예측도 포함 |
| `vloop-queue-low_confidence` | 표본 외 이미지 중 기준 미만 객체가 있거나 신뢰도 정보가 없는 이미지 |
| `vloop-queue-empty` | 표본 외 이미지 중 객체를 예측하지 못한 이미지 |

### 대량 이미지의 일괄 채택

브라우저에서 전체 선택할 필요 없이 자동 라벨링 작업 ID로 처리합니다. 우선 **미리보기**를
만들어 예상 건수를 확인합니다. 아래 임계값과 표본 비율은 실행 방법을 보여주는 예시입니다.

```shell
vloop review-batch --job-id AUTO_LABEL_JOB_ID --min-confidence 0.9 --sample-rate 0.001 --actor mhlee
```

- 성공 예측이 있는 미검수 후보 전체에서 `sample-rate` 비율을 먼저 `sample`로 뽑습니다.
  `0.001`은 전체 후보의 약 0.1%이며 정확한 고정 장수는 아닙니다. 선택은 신뢰도·빈 예측 여부와
  무관합니다. `0`은 표본 없음, `1`은 유효한 후보 전부를 표본으로 보냅니다.
  이미지 ID와 `seed`를 사용하므로 순서·배치 크기가 바뀌어도 선택 결과는 같습니다.
- `release_group_field`를 설정하면 이미지 ID 대신 장면 그룹과 `seed`로 뽑습니다. 같은 그룹의
  후보는 함께 뽑히며, 비율은 그룹 선택 확률입니다. 큰 그룹이 있으면 이미지 비율은 크게 달라질 수
  있습니다. 그룹 필드는 첫 `review-batch` 전에 채웁니다. 없는 값은 `error`로 남겨 채택하지 않습니다.
- 표본에 뽑힌 빈 예측·저신뢰도 이미지도 `sample` 큐에 들어갑니다. 표본 외에는 빈 예측을 `empty`,
  기준 미달을 `low_confidence`로 보내고, 객체가 있으며 모든 객체가 기준 이상인 경우에만
  `accept`로 채택합니다. 세 검수 큐는 중복되지 않습니다. 빈 예측은 자동으로 정답 처리하지 않습니다.
  높은 신뢰도만으로 누락된 객체를 찾아낼 수 없으므로, 표본은 직접 살펴봐야 합니다.
- 기존 수동 정답·검수 이력·수정된 자동 정답은 보존합니다. 미리보기 이후 수정된 이미지도
  적용할 때 건너뜁니다. 실패한 추론은 성공 예측이 없으므로 대상에 포함되지 않습니다.
- 미리보기는 정답을 복사하거나 승인하지 않습니다. `decisions`의 `accept`, `sample`,
  `low_confidence`, `empty`, `preserve`, `error` 건수를 확인합니다. 미리보기에 오류가 있어도
  나머지 유효한 대상은 처리할 수 있으며 오류 행은 적용하지 않습니다.
- 실제 적용은 미리보기 결과의 `review_batch_...` ID를 지정합니다. 자동 라벨링 ID와 다릅니다.

```shell
vloop review-batch --resume REVIEW_BATCH_JOB_ID --apply
```

채택한 정답은 `ground_truth`에 저장하고 상태를 **`auto_accepted`**, 종류를 `automatic`으로
기록합니다. 직접 검수한 `completed`와 별개의 상태입니다. 정책을 선택한 사람, 원본 예측 해시,
정답 해시, 작업 ID와 기준을 `reviews/records.sqlite3`에 기록합니다. 이후 직접 검수 완료를
실행하면 별도의 수동 승인 기록이 생깁니다.

적용 후 필요한 검수 목록만 엽니다. 표본과 낮은 신뢰도·빈 예측의 마스크는 화면을 준비하거나
검수를 시작할 때 복사하므로 미리 전부 복제하지 않습니다.

```shell
vloop review --queue sample
vloop review --queue low_confidence
vloop review --queue empty
```

입력 ID·예측 해시·수정 시점·기준과 진행 상태를 작업별 SQLite에 저장합니다. 메모리에는
`--batch-size`개 ID(기본 100, 최대 1000)와 처리 중인 이미지의 라벨을 유지합니다.
미리보기를 중단했으면 `--resume ID`, 적용을 중단했으면 `--resume ID --apply`로 이어갑니다.
재개 시 원본 작업·기준·검수자·대상을 바꿀 수 없습니다. 판정·승인 코드의 해시도 고정하므로
해당 코드를 수정했다면 새 미리보기를 만듭니다. 다른 기준에도 새 미리보기를 사용합니다.
DB 반영 직후 중단돼도 재개 시 같은 이미지를 중복 채택하지 않습니다.

`--limit N`은 새 미리보기의 대상 수를 제한합니다. 스냅샷 생성 자체가 끝나기 전에 중단되면
새 미리보기를 만듭니다. 적용 실패 행은 `outcome=failed`로 남으며, 원인을 해결한 뒤 새
미리보기를 만들어 처리합니다. 대상별 사유는 작업의 `samples.sqlite3`에서 조회할 수 있습니다.

```sql
SELECT image_id, decision, outcome, detail
FROM inputs
WHERE decision IN ('preserve', 'error') OR outcome IN ('preserved', 'failed');
```

### 승인 기록과 재검수

완료 시 이미지·클래스 매핑·박스·마스크의 해시, 검수자, 시간, 예측 출처와 정답 스냅샷을
`reviews/<image-id>/<record-id>.json`에 저장합니다. DB에는 승인 기록 ID·해시와 `review_history`를
남깁니다. 재검수·제외는 현재 승인을 해제하고 이전 기록은 보존합니다. 클래스 ID는 수정한
클래스 이름에서 다시 찾으며, 예측에서 복사된 ID나 신뢰도를 정답 판단에 사용하지 않습니다.

FiftyOne의 브러시는 화면 크기에 맞춘 마스크와 소수 좌표 박스를 저장할 수 있습니다.
승인 스냅샷에서는 박스 경계를 원본 픽셀에 반올림하고 nearest-neighbor로 마스크를 변환해
전체 이미지 좌표의 COCO RLE를 만듭니다. 편집기가 저장한 원래 박스와 마스크 RLE도 함께
보존합니다. 이 변환은 자동 예측 원본을 변경하지 않습니다.

서버는 기본 5초 간격으로 수정 시점 인덱스에서 최대 10,000건의 메타데이터를 스트리밍으로 읽고,
실제 변경된 승인은 최대 100건 검사합니다.
새로 승인한 그대로의 이미지는 파일·마스크를 다시 읽지 않고, 이후 DB 수정이 있는 승인 데이터만
확인합니다. 내용이 달라지면 `in_progress`로 되돌립니다. 조회 위치를 저장하므로 서버가 꺼져
있던 동안의 DB 수정도 이어서 검사합니다. 쌓인 변경이 많으면 순서대로 처리하므로 즉시는 아닙니다.
간격과 건수는 `review_poll_seconds`, `review_audit_scan_size`, `review_audit_batch_size`로 조절합니다.

DB 수정 시점을 바꾸지 않는 외부 파일 변경이나 직접 DB 조작은 이 증분 검사만으로 발견할 수
없습니다. 이미지·승인 기록까지 전부 확인할 때는 중단·재개 가능한 전체 검사를 실행합니다.

```shell
vloop review-audit
vloop review-audit --resume REVIEW_AUDIT_JOB_ID
```

`review-audit`는 승인된 이미지 파일·마스크·승인 기록을 읽어 확인하는 전체 검사입니다.
이미지가 많으면 오래 걸리므로 `Ctrl+C`로 멈추거나, 프로세스 종료·재부팅·DB 연결 오류로
중단된 뒤 같은 작업을 이어갈 수 있도록 `--resume`을 제공합니다. 저장한 검사 위치부터
진행하며, 강제 종료 직전 저장하지 못한 묶음은 일부 다시 검사할 수 있습니다.
`--resume` 없이 실행하면 새 전체 검사를 시작합니다. 개별 이미지의 승인 불일치는 보통 해당
이미지를 재검수 상태로 바꾸고 계속 검사하며, 그 자체가 전체 작업의 중단을 뜻하지는 않습니다.

전체 검사도 ID 범위를 이어가며 정해진 건수만 메모리에 읽습니다. 인덱스를 이용하는 범위 조회는
[MongoDB의 범위 기반 페이지 조회](https://www.mongodb.com/docs/manual/reference/method/cursor.skip/)
방식을 따릅니다. 릴리스는 공통 승인 검증 함수를 통해 현재 라벨과 승인 기록의 일치를 다시
확인합니다. 기본적으로 수동 승인만 허용하며, `--include-auto-accepted`를 명시해야 자동 채택을
학습 split에 포함합니다. 검증·테스트에는 수동 승인만 허용합니다.

editable 설치 상태에서는 코드 수정이 반영되지만, 실행 중인 검수 서버에는 모듈이 이미 로드되어
있으므로 서버를 다시 시작해야 합니다.

## Release / DVC

`release`는 **승인된 이미지와 정답을 학습·검증·테스트에 사용할 고정 버전으로 확정하는 단계**입니다.
개별 이미지의 검수 승인은 `review`에서 수행하고, `release`는 승인 기록과 현재 데이터가
일치하는지 확인한 뒤 해당 버전을 저장합니다. 기본 대상은 수동 승인 데이터이며,
`--include-auto-accepted`를 지정한 경우에만 자동 채택 데이터도 train에 포함합니다.

공유하는 범위는 **확정된 학습 데이터셋**입니다. 진행 중인 검수 작업이나 FiftyOne DB 전체를
공유하는 단계가 아닙니다. 받는 사람은 `restore`로 같은 이미지·정답·split을 복원해서 사용하며,
이를 위해 `ingest`, `autolabel`, `review`를 다시 실행할 필요가 없습니다. 원본 검수 환경에서
이후 라벨을 수정해도 완성된 버전은 유지되고, 다시 승인한 결과는 다음 버전으로 릴리스합니다.

이미지·라벨은 DVC remote에 저장하고, Git에는 최대 260개의 폴더 추적 파일과 작은 릴리스
설명만 기록합니다. 이미지 한 장마다 `.dvc` 파일을 만들지 않습니다. DVC 3.67.1과
RF-DETR 1.8.2의 실제 COCO 로더로 저장·복원·마스크 보존을 검증했습니다.

### 생성과 복원

위의 전체 lock 설치 절차에는 릴리스 의존성도 포함됩니다. 자동 라벨링용으로만 준비한 환경에는
릴리스에 필요한 의존성만 추가할 수 있습니다.

```shell
python -m pip install --no-build-isolation -c requirements/linux-py312-cu128.lock \
  'dvc>=3,<4' 'ijson>=3.4,<4'
```

실제 이미지와 클래스를 설정하고 검수를 마친 뒤 실행합니다. 작은 데이터는 첫 명령으로
곧바로 저장할 수 있습니다. 대량 데이터는 선택적으로 준비 결과와 용량 추정을 먼저 봅니다.

```shell
# 방법 1: 수동 승인 데이터를 바로 저장
vloop release --version v001

# 방법 2: 같은 수동 승인 대상을 준비만 하기 (이미지 복사·업로드·태그 생성 없음)
vloop release --version v001 --prepare-only
# 위 준비 작업이 출력한 작업 ID로 실제 저장
vloop release --resume RELEASE_JOB_ID

# 과거 버전 복원: 현재 Git checkout과 FiftyOne 데이터는 유지
vloop restore --version v001
```

위 예제에서 `v001`을 바로 저장하는 방법과 준비 후 저장하는 방법은 같은 버전을 만드는 두 가지
실행 방식입니다. 둘 중 하나를 선택합니다. `review-audit`를 먼저 실행할 필요는 없습니다.
자동 채택도 train에 포함하려면 어느 방식이든 새 작업을 만드는 `release --version ...` 명령에
`--include-auto-accepted`를 추가합니다.

새 릴리스의 버전 번호는 **사용자가 직접 지정하고 올립니다**. 자동 증가 기능은 없습니다.
새 작업에는 `--version v001`처럼 버전을 지정해야 하며, 버전을 생략하면 오류입니다.
이미 완성된 `v001`을 다시 지정해도 `v002`로 바뀌지 않고 오류가 발생합니다. 다음 버전은
사용자가 `--version v002`처럼 기존 버전보다 큰 번호로 실행합니다.

여기서 **재개는 `release --resume RELEASE_JOB_ID`로 같은 릴리스 생성 작업을 이어가는 것**입니다.
`--prepare-only`로 준비해 둔 작업이나 저장 도중 중단된 작업에 사용합니다. 작업에 기록된
대상 버전과 시작 당시 `project.yaml` 설정, 자동 채택 포함 여부, 수동 승인 분할 정책을 다시
사용하므로 `--version`, `--include-auto-accepted`, `--manual-to-val-test`를 재지정하지 않습니다.
예를 들어 `v001` 준비 시 자동 채택을 포함했다면 그 작업을 재개할 때도 자동 채택을 train에
포함합니다. 두 정책 옵션 모두 새 `v002` 작업에 자동 계승되지 않습니다. 새 버전에서도 자동
채택을 포함하거나 신규 수동 그룹을 val/test에만 넣으려면 각각의 옵션을 다시 지정합니다.

`restore --version v001`은 **이미 완성된 버전의 이미지·라벨·메타데이터를 복원하는 명령**입니다.
`storage_dir/releases/cache`를 사용하고, 실제 사용 경로는
`storage_dir/releases/restored/v001/dataset/`입니다. 캐시에 이미 있는 내용은 재사용하며,
릴리스 생성 작업을 이어가거나 새 버전 번호를 만들지 않습니다. 현재 Git checkout과
FiftyOne DB를 유지하고, 복원된 라벨을 FiftyOne에 등록하지 않습니다.

준비 결과는 작업 폴더의 `report.json`과 `dvc-work/dataset/metadata/summary.json`에 있습니다.
`snapshot.sqlite3`의 `records`에 포함 이미지·정답·승인 기록·검수 이력·출처·split이 있고,
`held_reason`이 있는 행은 보류한 이미지입니다. `summary.held_reasons`에 사유별 건수를
기록합니다. `new_image_in_heldout_group`은 기존 검증/테스트 그룹과 겹친 신규 이미지,
`mixed_approval_group`은 새 정책에서 자동·수동 승인이 섞인 신규 그룹입니다.
모든 후보가 보류돼도 미리보기는 만들지만 포함 이미지가 0장인 버전은 발행하지 않습니다.
검수 상태만 `completed`로 변경하거나 승인 후 라벨/파일을 수정한 경우 릴리스를 실패 처리합니다.
**객체가 없음을 확인하고 수동 승인한 빈 정답 이미지는 포함합니다.** 미검수라서 라벨이 없는
이미지와는 다릅니다. 미검수·수정 중·제외 상태는 포함하지 않습니다.

### 학습 데이터 구성

기본 정책은 매 릴리스의 신규 수동 그룹을 `seed: 42`, 80/10/10 비율의 해시 구간으로 배정합니다. 실제 개수는
근사 비율이며 작은 데이터나 큰 그룹은 차이가 클 수 있습니다. `release_group_field`에
FiftyOne의 촬영/장면 문자열 필드를 지정하면 같은 그룹은 같은 split에 들어갑니다.
기본 `null`은 이미지별 그룹입니다. 원본 경로로 장면을 자동 추정하지 않으므로 그룹이 필요하면
표본을 뽑는 첫 `review-batch`와 릴리스 전에 필드를 채웁니다. 필드를 지정했는데 값이 없으면
해당 이미지를 자동 채택하거나 릴리스하지 않습니다.

이후 릴리스는 기존 이미지의 split과 COCO ID를 유지하며, 한 버전에서 빠진 이미지의 분할도
기록해 둡니다. 같은 train 그룹의 신규 이미지는 train에 추가하고, 기존 val/test 그룹과 겹치는
신규 이미지는 보류합니다. 기본 정책에서 자동 채택이 포함된 신규 그룹은 통째로 train에
배정합니다. 기존 검증/테스트 그룹을 자동 채택 때문에 train으로 옮기지 않습니다.
그룹 필드·기존 이미지의 그룹·클래스 매핑 변경은 자동 처리하지 않습니다.

두 옵션은 **새 릴리스마다 독립적으로 선택**합니다. 기본 비율 `[0.8, 0.1, 0.1]`일 때의 동작은
다음과 같습니다. 배분은 처음 split을 정하는 이미지·그룹에 적용하며 기존 split은 보존합니다.

| 이번 릴리스의 옵션 | 자동 채택 | 신규 수동 승인 그룹 |
|---|---|---|
| `--include-auto-accepted --manual-to-val-test` | train에 포함 | val:test = 1:1 |
| `--include-auto-accepted` | train에 포함 | train:val:test = 8:1:1 |
| 없음 | 제외 | train:val:test = 8:1:1 |
| `--manual-to-val-test` | 제외 | val:test = 1:1 |

수동 검수분을 평가에 집중하려는 릴리스에는 다음처럼 지정합니다.

```shell
vloop release --version v001 --include-auto-accepted --manual-to-val-test --prepare-only
```

- 처음 배정하는 자동 승인 그룹은 train, 수동 승인 그룹은 val/test로 보냅니다. 수동 승인 그룹은
  `split_ratios`의 val:test 비율만 사용하므로 `[0.8, 0.1, 0.1]`이면 1:1입니다. 표본 추출과 분할은
  서로 다른 해시를 사용합니다. 전체 80/10/10을 맞추기 위해 자동 정답을 평가에 넣지는 않습니다.
- 기존 이미지의 split·COCO ID는 항상 유지합니다. 기존 train 이미지의 라벨을 사람이 수정해도
  train에 남습니다. 같은 train 그룹의 새 이미지도 train입니다. 기존 val/test 그룹의 새 이미지는
  계속 보류합니다. 이 옵션을 지정하면 v002 이후의 새 수동 그룹도 val/test로 배정합니다.
- 자동·수동 승인이 섞인 신규 그룹은 전체를 `mixed_approval_group`으로 보류합니다. 자동 승인
  멤버까지 사람이 검수해 수동 그룹으로 만든 뒤 새 준비 작업을 만들 수 있습니다. 단순히 제외한
  멤버는 릴리스 대상이 아니며, 이후 같은 그룹으로 다시 들어오면 기존 그룹 분할 규칙을 따릅니다.
- `--manual-to-val-test`를 생략하면 신규 수동 그룹은 `split_ratios` 전체 비율로 배분합니다.
  자동 채택을 포함한 신규 혼합 그룹은 그룹을 나누지 않고 통째로 train에 배정합니다.
  위 표의 비율은 수동 승인만으로 구성된 신규 그룹에 적용합니다.
- 이전 버전에서 사용한 옵션을 다음 버전에서 빼거나, 나중에 추가할 수 있습니다. `seed`,
  `split_ratios` 변경도 새 그룹 배정에만 적용합니다. 이미 배정된 val/test를 다시 8:1:1로
  섞거나, 수동 재승인한 기존 train을 평가로 이동하지 않습니다. 준비 후 선택을 바꾸려면
  같은 미발행 버전으로 새 준비 작업을 만들고, 같은 작업의 `--resume`은 저장한 정책을 사용합니다.
- val/test에 새 이미지가 들어가거나 정답이 수정되면 평가 기준도 바뀝니다. 모델 비교에는 같은
  `--dataset-version`과 `--split`을 사용합니다. 자동 채택을 포함하지 않아 train이 없는 릴리스는
  평가 데이터로는 보관할 수 있지만 학습에는 사용할 수 없습니다.

보류 그룹은 준비 작업의 SQLite에서 확인할 수 있습니다.

```sql
SELECT image_id, group_key, automatic, held_reason
FROM records
WHERE held_reason IS NOT NULL
ORDER BY group_key, image_id;
```

복원 데이터는 다음 구조입니다. 이미지 내용은 하나의 공유 폴더에서 재사용하고 COCO의
`file_name`은 `../images/<prefix>/<sha256>.png`를 가리킵니다.

```text
.vloop/releases/restored/v001/dataset/
├── images/<SHA-256 앞 2자리>/<SHA-256>.png
├── train/_annotations.coco.json
├── valid/_annotations.coco.json   # CLI의 val
├── test/_annotations.coco.json
└── metadata/
    ├── snapshot.sqlite3
    ├── snapshot.seal.json
    └── summary.json
```

`snapshot.sqlite3`는 릴리스의 이미지 ID·정답·승인 이력·split 등을 보존하는 기록이며,
FiftyOne DB 백업이 아닙니다. 이 파일을 복원해도 검수 DB가 동기화되지는 않습니다.

클래스 ID·이름을 COCO에 그대로 보관합니다. RF-DETR 1.8.2의 사용자 데이터 로더는 ID를
오름차순으로 정렬해 연속 인덱스로 변환하므로, 릴리스 설명의 `classes[].model_index`가
학습용 매핑입니다. RLE는 구멍과 분리 영역을 보존합니다. split별 클래스 이미지/객체 수와
빈 정답 수를 기록하고 객체가 없는 클래스는 평가 가용성을 `N/A`로 표시합니다. val/test의
이미지·정답이 바뀌면 `data_id`가 바뀌며 train만 바뀌면 그 식별자는 유지됩니다.

100만 장을 RF-DETR로 학습하는 메모리 검증은 별도 과제입니다. RF-DETR의 COCO 로더는
라벨 인덱스를 메모리에 올리므로 내보내기가 스트리밍이라고 학습 로더까지 같은 메모리로
동작한다고 볼 수는 없습니다.

### DVC 캐시와 remote, 저장 용량

아래 경로의 `.vloop`는 기본 `storage_dir`입니다. **캐시는 디스크에 실제 파일 내용을 저장하는
DVC 로컬 저장소**입니다. 메모리 캐시나 해시 목록만 있는 폴더가 아닙니다. 이 프로젝트는 DVC의
캐시 위치를 `.vloop/releases/cache`로 지정해 여러 릴리스와 복원 작업에서 함께 사용합니다.

| 위치 | 저장 내용과 역할 |
|---|---|
| `image_dir` | 사용자가 준비한 원본 이미지 |
| `.vloop/images/` | `ingest`가 방향·색상을 정규화한 RGB PNG. 검수와 자동 라벨링에서 사용 |
| `.vloop/releases/cache/` | DVC가 파일 내용의 해시로 보관하는 릴리스 이미지·라벨·메타데이터 |
| `.vloop/releases/pool/`, `.vloop/releases/restored/<version>/dataset/` | 캐시와 파일 내용을 공유하는 이미지 재사용 경로 및 복원 데이터 경로 |
| `dvc_remote` | 다른 환경에서도 버전을 복원할 수 있도록 데이터를 저장하는 별도 폴더 또는 마운트한 공유 저장소 |

DVC의 저장·전송은 다음 역할로 나뉩니다.

1. `add`: 릴리스 파일을 로컬 캐시에 저장하고 해당 내용을 가리키는 `.dvc` 추적 파일을 만듭니다.
   [DVC add 문서](https://doc.dvc.org/command-reference/add)
2. `push`: 지정한 추적 파일이 참조하는 데이터 중 remote에 없는 내용을 캐시에서 복사합니다.
   업로드 후에도 로컬 캐시는 남습니다. 매번 모든 과거 버전의 캐시를 보내는 동작은 아닙니다.
   [DVC push 문서](https://doc.dvc.org/command-reference/push)
3. `pull`: 해당 버전에 필요한 내용 중 로컬 캐시에 없는 데이터를 remote에서 받고,
   캐시의 내용으로 복원 경로를 구성합니다. 이 프로젝트는 reflink/hardlink를 사용합니다.
   [DVC pull 문서](https://doc.dvc.org/command-reference/pull)

일반 DVC 작업에서는 `add`를 여러 번 수행하고 나중에 `push`할 수 있습니다. 현재 `vloop release`는
격리된 작업 폴더에서 추적 폴더별 `add`와 `push`를 수행하고, 원격 데이터 검증과 Git 태그 생성까지
묶어서 처리합니다. 원격 저장까지 성공해야 완성된 릴리스로 취급하기 때문입니다.
`--prepare-only`는 대상·COCO·용량 추정을 준비하는 옵션이며 `add`까지만 실행하는 옵션은 아닙니다.
`vloop restore`는 태그의 추적 정보를 읽어 내부적으로 `pull`을 수행합니다.

#### pool이 생기고 재사용되는 순서: v001 → v002

아래는 이미지 내용이 서로 다른 A·B를 `v001`로 릴리스한 뒤, 새 이미지 C를 더해
A·B·C를 `v002`로 릴리스하는 예입니다. 실제 파일명에는 이미지 해시를 사용합니다.
릴리스 작업 폴더는 `.vloop/runs/RELEASE_JOB_ID/dvc-work/dataset/`입니다.

| 순서 | v001: A·B를 처음 릴리스 |
|---|---|
| 1 | `ingest`가 원본 A·B를 관리용 PNG로 저장하고 검수를 마칩니다. |
| 2 | 승인 스냅샷·COCO를 만들고 **이미지 복사 전에 pool에서 A·B를 찾습니다.** 처음이므로 없습니다. |
| 3 | 관리 PNG A·B를 이번 릴리스 작업 폴더에 실제 복사합니다. |
| 4 | `dvc add`가 내용을 cache에 저장하고 작업 파일을 cache와 reflink/hardlink로 연결합니다. |
| 5 | `dvc push`가 remote에 없는 내용을 전송합니다. |
| 6 | 전송한 이미지의 **작업 파일에서 pool 경로로 hardlink를 생성**합니다. 이미지 내용은 복사하지 않습니다. |
| 7 | 원격 데이터 검증·Git 태그 생성이 성공하면 **dvc-work 작업 폴더 전체를 삭제**합니다. cache·pool·remote는 남습니다. |

4~6번은 이미지 해시 앞자리로 나눈 추적 폴더마다 실행합니다. hardlink는 한 경로를 따라가는
바로가기가 아니라 같은 파일에 붙인 다른 경로이므로, 작업 폴더를 삭제해도 pool의 파일은 남습니다.
DVC가 reflink를 선택하면 cache와 작업 파일은 디스크 블록을 공유하고, pool은 그 작업 파일과
hardlink를 공유합니다. 현재 설정은 symlink나 일반 복사 방식으로의 fallback을 사용하지 않습니다.

| 순서 | v002: 기존 A·B에 새 C를 추가 |
|---|---|
| 1 | C를 ingest·검수하고, v001의 메타데이터를 기준으로 기존 split을 유지한 새 스냅샷·COCO를 만듭니다. |
| 2 | 새로운 릴리스 작업 폴더를 만들고 pool을 조회합니다. A·B는 있고 C는 없습니다. |
| 3 | **A·B는 pool의 파일에서 새 작업 폴더로 hardlink를 생성**합니다. ingest에서 다시 복사하지 않습니다. |
| 4 | **C만** ingest 관리 영역에서 작업 폴더로 실제 복사합니다. |
| 5 | `dvc add`에서 기존 A·B의 캐시 내용을 재사용하고 C를 새로 저장합니다. 새 라벨·메타데이터도 추적합니다. |
| 6 | `push`로 remote에 없는 내용을 전송하고, 작업 파일을 기준으로 pool의 링크를 갱신합니다. pool에는 A·B·C가 있습니다. |
| 7 | 검증·태그 생성 성공 후 v002의 작업 폴더를 삭제합니다. |

pool은 **다음 릴리스의 작업 폴더를 준비할 때 기존 이미지의 복사를 피하기 위한 공유 경로**입니다.
DVC의 최종 중복 제거 자체에 필수인 폴더는 아닙니다. pool만 제거하면 현재 준비 로직은 기존
이미지도 작업 폴더에 복사한 뒤 DVC에서 중복을 확인하게 됩니다. 복사 전 캐시 조회·재사용을
다른 방식으로 구현하면 pool을 대체할 수 있습니다. 기존 이미지도 무결성 검사를 위해 읽습니다.

`restore v001`/`restore v002`는 pool을 거치지 않고 해당 버전의 DVC 추적 정보를 사용해
cache와 연결된 `restored/<version>/dataset/`을 구성합니다. pool에는 여러 릴리스의 이미지가
모여 있고, restored에는 선택한 버전의 이미지·라벨·분할이 구성됩니다.

원본을 보관한 채 최초 릴리스까지 만들면 **원본, 관리 PNG, DVC 로컬 캐시, DVC remote**의
네 저장 계층이 생깁니다. 원본 JPEG가 PNG로 변환되면 크기가 커질 수 있고, 릴리스에는 승인된
대상만 들어가므로 원본 용량의 정확히 네 배라는 뜻은 아닙니다. 용량과 시간은 다음처럼 구분합니다.

- 원본과 `ingest`가 만든 관리 이미지는 기존 공간을 사용합니다. 릴리스는 수정 가능한 관리
  이미지와 분리된 최초의 고정 사본을 만듭니다. 이 비용을 없애려고 관리 파일을 DVC 캐시에
  직접 하드링크하지 않습니다.
- `.vloop/releases/cache`와 `pool`은 reflink/읽기 전용 hardlink를 이용해 이미지 내용을
  공유합니다. 새 릴리스에서도 동일 이미지는 이 공간을 재사용합니다. 완료한 작업의 별도
  이미지 트리는 제거하고, 복원한 버전만 별도 디렉터리에 링크합니다. 링크가 지원되지 않으면
  실패하며 대량의 일반 복사로 자동 전환하지 않습니다. 이 디렉터리의 파일은 직접 수정하지 않습니다.
- Git 밖의 `dvc_remote`에도 데이터 공간이 필요합니다. 예를 들어 고정 이미지가 100 GB라면
  일반 복사를 기준으로 고정 캐시 약 100 GB와 원격 약 100 GB가 추가될 수 있습니다.
  reflink 지원 여부에 따라 실제 물리 사용량은 줄어듭니다. 준비 결과에는 링크 절감 전의
  임시 저장까지 포함한 이미지 공간 상한 추정과 각 파일시스템의 여유 공간을 표시합니다.
  라벨·SQLite·디렉터리 메타데이터와 다른 프로세스의 사용량은 이 추정에 추가됩니다.
- DVC는 파일 내용 단위로 중복을 제거합니다. 라벨 하나만 수정해도 변경된 COCO JSON과
  검수 이력 SQLite는 파일 전체가 새 객체로 저장됩니다. 따라서 라벨·이력 공간은 버전 수에
  따라 늘며, 이미지 재사용이 전체 버전의 저장 비용을 0으로 만들지는 않습니다.
- 이미지는 해시 앞 2자리의 최대 256개 폴더로 나누고 라벨 폴더 4개를 더해 추적합니다.
  전체 ID나 COCO를 Python 목록으로 만들지 않고 SQLite·스트리밍 JSON을 사용합니다.
  DVC도 폴더 하나씩 처리하지만 전체 파일 탐색·해시 계산·원격 검증은 이미지 수/바이트 수에
  비례합니다. Git이 가벼워지는 것과 DVC 작업이 즉시 끝나는 것은 별개입니다.

초기 remote가 같은 디스크의 다른 폴더이면 디스크 고장에 대비한 사본이 되지는 않습니다.
필요하면 `dvc_remote`를 외장 디스크나 로컬에 마운트한 NAS 경로로 설정합니다. 로컬 캐시와
remote에는 과거 버전도 들어 있으므로 특정 버전만 기준으로 `dvc gc`를 실행하지 않습니다.
[DVC 공식 대용량 데이터 지침](https://doc.dvc.org/user-guide/data-management/large-dataset-optimization)

스냅샷 수집은 샘플별 승인 검증이며 전체 MongoDB의 단일 시점 트랜잭션은 아닙니다. 준비 중에는
대상 검수를 마친 상태로 두는 것이 좋습니다. 수집이 끝나기 전에 중단되면 재개 시 다시 수집하고,
완료된 스냅샷은 재개해도 변경하지 않습니다. 이후 파일이 바뀌면 저장을 거부합니다.
COCO를 다시 읽어 검증하고 로컬 원격 객체의 실제 해시까지 확인한 뒤에만 태그를 생성합니다.
업로드 실패/중단 후에는 출력된 작업 ID로 재개하며, 이미 올라간 객체를 재사용합니다.

### Git 태그로 데이터 버전 공유하기

소스코드와 데이터 버전 태그는 **같은 Git 저장소**에 있으며 같은 Git remote(`origin`)으로
공유합니다. `dvc_remote`는 이미지·라벨의 실제 내용을 저장하는 별도 저장소입니다.

| 공유 경로 | 받는 내용 |
|---|---|
| Git remote (`origin`) | 소스코드, `dataset/v001` 태그, 해당 버전의 `.dvc` 추적 파일과 릴리스 설명 |
| DVC remote (`dvc_remote`) | 추적 파일이 가리키는 실제 이미지·COCO 라벨·릴리스 기록 |

Git 태그는 `dataset/v001`이며 릴리스 시작 시점의 HEAD에 데이터 메타데이터만 추가한 별도 커밋을
가리킵니다. 태그는 특정 커밋에 붙이는 이름이며, `main` 브랜치와 별개로 관리됩니다.
예를 들어 `B`에서 릴리스한 뒤 코드 작업을 계속하면 다음 구조입니다.

```text
A ── B ── C    ← main: 소스코드 변경
     └── R     ← dataset/v001: B에 데이터 메타데이터를 추가한 커밋
```

릴리스는 현재 브랜치·Git index·코드 수정을 그대로 유지합니다. 메타데이터는 태그 안의
`vloop-dataset/`에 있으며, 작업 디렉터리에 이 폴더를 펼치지 않아 `git status`가 이미지
트리를 탐색하지 않습니다. 아래 명령으로 내용을 확인할 수 있습니다.

```shell
git show dataset/v001:vloop-dataset/release.json
```

`git push origin main`만 실행하면 현재 구조의 릴리스 커밋과 태그는 올라가지 않습니다.
브랜치와 함께 공유할 태그를 명시해야 합니다. `--follow-tags`도 현재 태그에는 적용되지 않습니다.
이 옵션은 전송하는 브랜치 이력에 연결된 annotated tag를 대상으로 하고, 현재 릴리스는 별도
커밋을 가리키는 lightweight tag를 만들기 때문입니다. [Git push 문서](https://git-scm.com/docs/git-push)

`git fetch origin`은 기본적으로 가져오는 브랜치 이력에 연결된 태그를 함께 받습니다.
위의 `R`처럼 브랜치 이력 밖의 커밋을 가리키는 태그까지 받으려면 `--tags`를 지정합니다.
`git merge origin/main`은 받은 코드 변경을 합치는 작업으로, 원격 태그를 추가로 가져오지 않습니다.
[Git fetch 문서](https://git-scm.com/docs/git-fetch)

다른 환경에서 복원하려면 **Git 태그와 DVC remote 접근이 모두 필요합니다.** 예를 들어
공유폴더를 마운트했다면 각자의 `project.yaml`에서 다음 항목을 설정합니다.
각 컴퓨터의 마운트 경로가 달라도 같은 실제 공유폴더를 가리키면 됩니다.

```yaml
storage_dir: .vloop
dvc_remote: /mnt/team-share/vision-loop-dvc
```

`storage_dir`는 각자 로컬에 두며, 위 remote 경로는 Git 저장소와 `storage_dir` 밖에 있어야 합니다.
현재 설정은 파일시스템 경로를 사용하므로 NAS는 먼저 로컬에 마운트합니다. `s3://`나 `smb://` URL을
직접 지정하는 방식은 지원하지 않습니다. 게시자는 공유폴더 읽기·쓰기 권한, 받는 사람은 읽기 권한이
필요합니다. 받는 사람에게도 내려받은 데이터를 보관할 로컬 캐시 공간이 필요합니다.

공유하는 사람은 검수를 마친 뒤 릴리스를 만들고 태그를 게시합니다. Git remote 이름은 `origin`,
코드 브랜치는 `main`으로 가정합니다.

```shell
vloop release --version v001

# 코드와 해당 데이터 태그를 함께 게시
git push origin main refs/tags/dataset/v001
```

`release`가 DVC 데이터 업로드까지 수행하므로 별도 `dvc push`는 필요하지 않습니다.
Git 태그의 원격 전송은 자동 수행하지 않습니다. 코드가 이미 공유되어 있다면
`git push origin refs/tags/dataset/v001`로 태그와 그 커밋만 게시할 수 있습니다.

받는 사람은 프로젝트를 clone하고 위 릴리스 의존성과 `project.yaml`을 준비한 환경에서 실행합니다.
아래 코드 갱신 예제는 현재 `main` 브랜치에 있다고 가정합니다.

```shell
# 최신 코드와 데이터 태그 받기
git fetch origin --tags
git merge origin/main

# 지정한 버전의 실제 이미지·라벨 복원
vloop restore --version v001
```

데이터 복원만 필요하고 코드가 준비되어 있다면 `merge`는 필요하지 않습니다. 태그를 받아도
데이터가 자동 다운로드되지는 않으며, `restore`가 지정한 버전을 복원합니다. 이후 `v002`가
공유되어도 자동으로 전환하지 않습니다. 태그를 다시 받은 뒤 `vloop restore --version v002`로
선택합니다. 원본 `image_dir`나 게시자의 FiftyOne DB·SAM 3 체크포인트 없이 복원할 수 있습니다.
복원한 버전들은 같은 로컬 캐시를 재사용하며, 재복원도 체크섬 검증에는 시간이 듭니다.

## Train / MLflow

**지정한 릴리스를 복원해 RF-DETR Seg Nano를 학습하고, 실행별 데이터·설정·모델을 MLflow에
기록합니다.** 실행 중인 FiftyOne의 라벨 변경은 이미 확정된 학습 입력에 반영되지 않습니다.
학습 전에 위 `pipeline` extra를 설치하고 사용할 릴리스와 DVC remote를 준비합니다.

```shell
vloop train --dataset-version v001 --notes "첫 기준 모델"
vloop experiments
# 브라우저를 자동으로 열지 않고 같은 서버 실행
vloop experiments --no-browser
```

`train`이 내부에서 복원하므로 `restore`를 먼저 실행할 필요는 없습니다. 다른 컴퓨터에서 받은
릴리스라면 앞 절의 Git 태그 수신과 DVC remote 설정이 필요합니다. 원본 이미지 폴더나
SAM 3 체크포인트는 학습에 필요하지 않습니다. 클래스 매핑은 릴리스에 저장된 값을 사용합니다.
train과 val에 각각 이미지와 하나 이상의 정답 객체가 있어야 합니다. 승인된 빈 정답 이미지도
함께 학습하며, 전체 val이 빈 정답이면 최적 모델을 고를 mask mAP를 계산할 수 없어 시작을 거부합니다.

학습 코드를 추적하기 위해 **현재 코드 저장소를 커밋한 상태에서 실행**합니다. 미커밋 변경이나
추적되지 않은 파일이 있으면 실패 처리합니다. Git에서 제외한 `project.yaml`은 실험마다 바꿀 수
있고, 실제 사용한 설정은 별도로 기록합니다. `train`은 Git 커밋이나 push를 자동 수행하지 않습니다.

### 학습 설정과 기록

```yaml
train_model: RFDETRSegNano
epochs: 30
batch_size: 1
grad_accum_steps: 8
learning_rate: 0.0001
train_checkpoint: null
train_num_workers: 0
train_gradient_checkpointing: false
```

- `train_checkpoint: null`이면 첫 학습에서 공식 Seg Nano 가중치를
  `storage_dir/models/rfdetr/rf-detr-seg-nano.pt`에 받고 검증한 뒤 재사용합니다.
  직접 준비한 초기 가중치는 로컬 경로로 지정할 수 있습니다. **학습 재개에는 아래 작업 ID 옵션을
  사용합니다.** 초기 가중치 지정만으로 이전 옵티마이저·에포크 상태를 이어가지는 않습니다.
- 입력 해상도는 모델 기본값 312이며, RF-DETR의 기본 다중 해상도 증강을 사용하므로 실제 학습
  입력 크기는 달라질 수 있습니다. 전체 모델·학습 기본값까지 `training.json`에 저장합니다.
  GPU 정밀도는 공식 trainer가 선택하고 실제 값을 기록합니다. 검증한 RTX 5060에서는 BF16입니다.
- `train_num_workers: 0`은 큰 COCO 인덱스를 여러 작업 프로세스로 복제하는 비용을 줄이기 위한
  초기 설정입니다. `train_gradient_checkpointing`은 중간 활성값을 재계산해 GPU 메모리를
  줄이는 선택 사항이며, 속도와 메모리를 확인하고 새 실험에서 변경합니다.
- 학습은 train으로 수행하고 에포크별 검증에는 val(`valid/`)을 사용합니다. 최종 test는 자동
  실행하지 않습니다. 최적 모델은 regular/EMA 중 가장 높은 **validation mask mAP**로 선택합니다.
- MLflow에는 데이터 태그의 실제 Git 커밋, DVC 추적 정보, 이미지·라벨 식별자, 학습 코드 커밋,
  클래스 매핑, 설정, 의존성 버전, 초기 가중치 해시, 에포크별 손실·검증 지표를 남깁니다.
  `--notes`는 실험 목적·관찰 메모이며 MLflow 화면에서도 확인할 수 있습니다.

MLflow 메타데이터는 `storage_dir/mlflow/mlflow.db`, 모델 파일은
`storage_dir/mlflow/artifacts/<RUN_ID>/artifacts/`에 보관합니다. 모델·체크포인트는 이 로컬
아티팩트 경로에 원자적으로 저장하며 MLflow 화면에서도 조회할 수 있습니다.

```text
<RUN_ID>/artifacts/
├── training.json          # 고정 데이터·코드·설정·매핑·초기 가중치 해시
├── config.json            # 실제 적용한 프로젝트 설정
├── dependencies.json      # 실행 환경의 설치 버전
├── report.json            # 완료·실패·중단 상태와 결과 위치
├── model/
│   ├── best.pt            # 최적 추론용 가중치와 모델 구성
│   └── model.json         # 클래스 매핑·전처리·후처리·가중치 해시
└── resume/
    ├── epoch-<번호>.ckpt  # 마지막 완료 에포크의 전체 학습 상태
    └── latest.json        # 재개 체크포인트 위치·해시·에포크·step
```

재개용 체크포인트에는 옵티마이저·스케줄러·에포크·step·EMA·난수 상태와 이전 최적 모델을
포함하므로 `best.pt`보다 큽니다. 같은 run에서는 새 체크포인트를 저장하고 참조 정보를 갱신한
뒤 이전 재개 파일을 정리합니다. 이전 run의 결과는 보존하며 여러 run 사이의 모델 중복 제거는
하지 않습니다. 이 모델 아티팩트는 데이터 릴리스의 DVC remote에 자동 업로드되지 않습니다.

### 중단한 학습 재개

```shell
vloop train --resume TRAIN_JOB_ID
```

사용자는 학습 시작 시 출력하는 **`train_...` 형태의 vloop 작업 ID만 사용**합니다. 보고서의
`run_id`에는 MLflow가 생성한 내부 ID를 저장하고, 재개 시 그 값을 읽어 원래 MLflow run의
체크포인트를 찾습니다. 이름이나 시각으로 추정하지 않으며 MLflow의 `vloop.job_id` 태그와
보고서의 대응도 확인합니다. MLflow 화면의 실행 이름은 vloop 작업 ID입니다.

재개하면 **새 vloop 작업 ID·폴더·보고서와 새 MLflow run을 생성**합니다. 예를 들어
`train_A`를 재개하면 `train_B`에서 기록을 시작하고 `resume_from_job_id = train_A`를 남깁니다.
MLflow에는 `vloop.resume_from_job_id`와 내부 연결용 `vloop.resume_from_run_id`를 자동 기록합니다.
원래 폴더와 중단 상태는 보존하며, 새 지표를 원래 run에 이어 쓰지 않습니다.

MLflow 초기화가 실패해도 vloop 작업 폴더의 설정·오류·보고서는 남습니다. 초기화 전에 실패한
작업에는 MLflow 체크포인트가 없으므로 원인을 해결한 뒤 `train --dataset-version ...`로 새로
시작합니다. 작업 ID가 있다는 것과 재개 가능한 학습 체크포인트가 있다는 것은 별개입니다.

원래 run의 데이터 버전·학습 설정을 가져오므로 현재 YAML의 `epochs`, 학습률 등을 바꿔도
재개에는 적용하지 않습니다. 이 로컬 구현은 원래 작업의 `runs/TRAIN_JOB_ID/report.json`과
MLflow 저장소에 접근할 수 있어야 합니다.
원래 코드 커밋과 학습 의존성 버전이 같아야 하며, 태그·고정 설정·체크포인트 해시가 달라지면
재개를 거부합니다. 로컬 저장소와 DVC remote 경로는 현재 설정으로 연결합니다.

체크포인트는 **에포크 완료 시점**에 저장합니다. 에포크 도중 중단되면 마지막으로 완료한
에포크 다음부터 다시 실행하며, 중단된 에포크의 일부 배치는 다시 처리할 수 있습니다.
첫 에포크 저장 전 중단되었다면 새 학습을 시작합니다. 원래 지정한 총 에포크 수까지 이미
완료했으면 `--resume`로 학습 횟수를 늘리지 않습니다. 설정 변경은 새 실험으로 실행합니다.
재개 시 이전 최적 모델도 이어받으므로 이후 성능이 나빠져도 이전 최적 가중치를 잃지 않습니다.

일반적인 중단은 MLflow `KILLED`, 오류는 `FAILED`, 완료는 `FINISHED`로 기록합니다.
프로세스 강제 종료나 전원 차단 때는 상태가 `RUNNING`으로 남을 수 있지만, 이미 저장된
체크포인트로 재개할 수 있습니다. 전체 실행의 비트 단위 동일성까지 보장하지는 않습니다.

### 실험 화면과 검증 범위

`vloop experiments`는 같은 SQLite와 아티팩트 저장소를 사용해
`http://127.0.0.1:5000`에서 MLflow를 실행합니다. 포트는 `mlflow_port`로 변경합니다.
별도 터미널에서 학습 중에도 화면을 볼 수 있고, 화면 서버를 꺼도 학습 기록은 계속 저장됩니다.
`Ctrl+C`로 서버를 종료합니다. 화면을 열었다고 학습이 시작되지는 않습니다.

RF-DETR 1.8.2와 Lightning 2.6.5에서 확인한 두 호환성 처리는 프로젝트 adapter에 있습니다.
누적 배치 손실의 중복 나눗셈을 보정하고, 프로젝트의 N개 클래스 밖 추가 출력 채널을 top-k
선택 전에 제외합니다. 검증과 저장 모델의 후처리에 같은 규칙을 적용하며, 설치된 라이브러리
소스는 수정하지 않습니다. 저장 모델은 `vloop.trained_model.load_model(cfg, job_id)`에
같은 vloop 학습 작업 ID를 넘겨 복원합니다.
현재 YAML의 클래스·임계값 대신 아티팩트의 매핑·전후처리 설정을 기준으로 후속 평가를 구성합니다.
[RF-DETR 공식 학습 인터페이스](https://rfdetr.roboflow.com/1.8.2/learn/train/)

생성한 소형 이미지 4장으로 실제 DVC 릴리스 → GPU 학습 → 첫 에포크 후 중단 → 새 프로세스
재개 → 모델 복원·마스크 추론을 검증했습니다. 최대 PyTorch GPU 할당 약 1.25 GiB,
예약 약 1.30 GiB, 프로세스 최대 RSS 약 4.10 GiB였습니다. 이는 실행 경로 검증이며 실제
도메인 데이터의 정확도·메모리 보장값은 아닙니다. 검증 기록은 `.vloop/train-integration.json`에
있고, 검증에 사용한 임시 데이터·모델·DB는 정리됩니다.

별도 COCO 인덱스 측정에서는 100만 이미지·100만 단순 RLE 라벨의 JSON 약 275 MiB를
약 6.19초에 읽었고 프로세스 최대 RSS는 약 2.27 GiB였습니다. 모델·이미지 픽셀 로딩·증강·
옵티마이저·검증 예측 누적은 포함하지 않습니다. 실제 마스크의 복잡도와 객체 수에 따라
메모리가 증가하며, 전체 학습은 작은 도메인 데이터로 먼저 확인합니다.
측정 스크립트는 `tests/train_loader_scale_runner.py`, 결과는 `.vloop/train-loader-scale.json`입니다.

## Evaluate

학습이 출력한 **vloop 학습 작업 ID**로 저장된 최적 모델을 평가합니다. 평가도 별도의
`evaluate_...` 작업 ID를 만들며, MLflow 내부 ID를 직접 입력하지 않습니다.

```shell
# 기본값: 학습에 사용한 데이터 버전의 val 전체
vloop evaluate --job-id TRAIN_JOB_ID

# 처음에는 작은 범위에서 속도·메모리 확인
vloop evaluate --job-id TRAIN_JOB_ID --dataset-version v001 --limit 100

# 완료된 평가 화면 열기: 위 명령에서 출력한 evaluate_... ID 사용
vloop evaluate --view EVALUATE_JOB_ID
vloop evaluate --view EVALUATE_JOB_ID --no-browser

# 최종 test 평가는 명시적으로 선택
vloop evaluate --job-id TRAIN_JOB_ID --dataset-version v001 --split test
```

계산 명령은 지표를 저장한 후 종료합니다. `--view`는 저장된 결과를 열고, `--no-browser`는
브라우저 자동 실행 없이 localhost 서버를 유지합니다. `Ctrl+C`로 닫습니다.
`--view`에는 데이터 버전·임계값 등 계산 옵션을 함께 넣을 수 없습니다.
평가 기본값은 `val`입니다. YAML의 `eval_split`을 `test`로 바꿔도 자동으로 test를 평가하지
않으며 명령에서 `--split test`를 요구합니다.

평가 이미지는 DVC 릴리스에서 복원합니다. 현재 YAML의 클래스가 아니라 **저장 모델과 릴리스의
클래스 ID·이름·모델 인덱스**를 대조합니다. 검수 중인 `ground_truth`를 읽거나 변경하지 않으며,
평가용 FiftyOne 데이터셋 `vloop-eval-EVALUATE_JOB_ID`를 따로 만듭니다.
릴리스 GT와 예측 필드는 화면에서 읽기 전용으로 설정합니다.

| 결과 | 계산 기준 |
|---|---|
| 박스 mAP / AP50 | 모델의 원본 이미지 좌표 박스 IoU |
| 마스크 mAP / AP50 | 원본 크기 이진 마스크의 실제 픽셀 IoU |
| 클래스별 AP / AP50 | 정답 객체가 없는 클래스는 `null` / N/A, 평균에서 제외 |
| TP / FP / FN | 추론 임계값을 통과한 예측, 같은 클래스끼리 IoU 0.5에서 매칭 |

FiftyOne의 COCO 평가를 사용합니다. mAP는 IoU 0.50부터 0.95까지 0.05 간격과
101개 recall 지점으로 계산합니다. 마스크가 예측 박스 밖에 있어도 잘라내지 않도록
박스용·마스크용 예측 필드를 분리합니다. 빈 예측과 사람이 승인한 빈 정답 이미지도 포함하고,
면적이 0인 예측 마스크는 제거하지 않아 마스크 FP로 계산합니다.
[FiftyOne 박스·마스크 평가](https://docs.voxel51.com/user_guide/evaluation/detections.html)

기본 추론 임계값 `0.001`, 표시 임계값 `0.5`, 이미지당 최대 검출 수 `100`은 **학습할 때
모델에 저장한 설정**에서 읽습니다. 현재 YAML 변경은 기존 모델의 평가에 자동 반영되지 않습니다.
다른 기준을 시험하려면 새 평가 명령에서 `--confidence`, `--display-confidence`,
`--max-detections`로 명시적으로 변경합니다. 표시 임계값은 추론 임계값 이상이어야 하고,
최대 검출 수는 저장 모델의 top-k 한도를 넘을 수 없습니다.

화면은 `display_confidence` 저장 뷰로 열립니다. `all`은 낮은 추론 임계값을 통과한 모든 예측,
`boxes_fp`, `boxes_fn`, `masks_fp`, `masks_fn`은 해당 오류가 있는 이미지,
`empty_predictions`는 예측이 하나도 없는 이미지입니다. 평가 키에서 evaluation patches로
매칭 객체도 확인할 수 있습니다. 화면에서 표시 임계값을 바꿔도 이미 저장한 mAP·FP·FN은
다시 계산되지 않습니다. 오류 뷰에는 낮은 점수의 FP도 포함될 수 있습니다.

평가 아티팩트는 MLflow 로컬 아티팩트 폴더에 저장합니다.

```text
evaluation.json     # 학습 출처·모델 정보, 데이터 태그, 선택 입력, 평가 설정·비교 해시
samples.sqlite3     # 선택한 이미지·고정 GT·박스/마스크 예측 RLE·처리 오류
metrics.json        # 박스/마스크 지표, 클래스별 AP, TP/FP/FN
config.json         # 실행한 프로젝트 설정
dependencies.json   # 실제 설치 의존성
report.json         # 완료/실패, 처리 건수, 결과 체크섬, 소요 시간·프로세스 최대 RSS
```

이미지 자체를 아티팩트에 다시 복사하지 않습니다. `--view`는 결과 체크섬과 데이터 태그를
검사하고, 평가용 FiftyOne 데이터셋이 없어졌다면 아티팩트에서 다시 구성합니다. 모델 추론은
다시 하지 않습니다. 로컬 작업 보고서·MLflow 아티팩트, 원래 데이터 Git 태그와 DVC remote가
필요합니다. 실패·중단된 평가는 전체 지표로 인정하지 않고 오류를 남깁니다. 현재 평가는
재개 옵션 없이 새 명령으로 다시 계산하며, 화면 재구성은 완료된 평가에만 허용합니다.

`comparison_id`는 실제 평가 이미지·정답·클래스 매핑·split·계산 설정이 같은지 확인하는
기준입니다. 학습 작업 ID나 학습 데이터 버전, 로컬 저장 경로, 표시 임계값은 제외합니다.
따라서 `v001`과 `v002` 모델을 동일한 `v001`의 val로 평가하면 비교할 수 있습니다.
`--limit`은 이미지 ID 순서로 앞 N장을 선택하며, 선택 범위가 달라지면 비교 해시도 달라집니다.
작은 일부 이미지의 점수를 val 전체 결과로 해석하지 않습니다.

`vloop experiments`에서 `vloop.kind=evaluate`와 같은 `vloop.comparison_id` 태그를 가진
평가 run들을 골라 지표를 비교합니다. `comparison_id`가 같아도 두 평가는 별도 작업으로
보존됩니다. 이미지별 결과는 각 평가의 `vloop evaluate --view EVALUATE_JOB_ID`로 확인하며,
화면을 다시 여는 것만으로 새 평가 작업이나 비교 ID를 만들지 않습니다.

입력·예측은 SQLite로 저장하고 추론은 한 장씩, FiftyOne 적재는 20장씩 처리합니다. 다만
**FiftyOne의 mAP 계산은 전체 객체 매칭 결과를 RAM에 모읍니다. 100만 장 전체 평가의 메모리
상한은 보장하지 않습니다.** 평가 DB도 이미지당 GT·예측 라벨 공간을 사용합니다.
`--limit`은 추론·평가 대상을 제한하지만 현재 DVC 복원은 해당 버전의 모든 split을 복원·검사합니다.
처음에는 작은 val 범위로 `report.json`의 `peak_rss_mib`와 시간을 확인하고 범위를 늘립니다.
이 RSS는 모델 로딩을 포함한 Python 프로세스 최대값이며 별도 MongoDB 서버 메모리는 제외합니다.
생성 이미지 4장의 GPU 통합 검증에서는 val/test 각각 1장을 평가했고 프로세스 최대 RSS는
약 2.19 GiB였습니다. 평가 데이터셋을 지운 뒤 재구성·HTTP 응답·서버 종료까지 확인했으며,
기록은 `.vloop/evaluate-integration.json`에 있습니다. 실제 데이터의 성능 보장값은 아닙니다.

## Results / Recovery

```text
.vloop/
├── catalog.sqlite3           # 이미지 ID와 원본 경로 목록
├── images/<hash-prefix>/    # 방향이 확정된 관리 이미지
├── fiftyone/db/             # 영속 검수 DB
├── fiftyone/plugins/        # review 실행 시 설치하는 프로젝트 검수 operator
├── reviews/<image-id>/      # 승인·재검수·제외 기록과 승인한 정답 스냅샷
├── reviews/records.sqlite3  # 자동 채택 기록을 모은 DB (WAL 포함)
├── reviews/audit-cursor.json # 증분 검사 조회 위치
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
`127.0.0.1`로 설정합니다. MLflow 서버는 `vloop experiments`로 엽니다.
FiftyOne 1.21은 같은 사용자의 기존 MongoDB 프로세스를 재사용할 수 있으므로 서로 다른
프로젝트의 FiftyOne을 동시에 실행하지 않습니다. 실제 DB 경로가 설정과 다르면 vloop는
라벨을 읽거나 쓰기 전에 오류를 반환합니다. 이 경우 다른 프로젝트의 FiftyOne 프로세스를
종료하고 다시 실행합니다. 같은 프로젝트의 검수 화면과 CLI는 같은 DB를 사용할 수 있습니다.
완성된 학습 데이터 버전은 위의 `release` / `restore` 절차로 저장·복원합니다.
이미지·모델·DB를 Git에 직접 추가하지 않습니다.

## Test

코드 변경 후 기본 검사는 다음과 같습니다.

```shell
python -m pytest -q
ruff check src tests examples requirements
ruff format --check src tests examples requirements
python -m pip check
python requirements/check.py
```

기본 pytest 실행은 GPU·로컬 서버를 사용하는 선택 테스트를 건너뜁니다. 아래 환경 변수를
모두 지정하면 전체 테스트를 실행합니다. 전체 lock 환경, 사용 가능한 NVIDIA GPU, SAM 3
소스·체크포인트가 연결된 설정 파일과 기본 관리 경로의 RF-DETR Seg Nano 가중치가 필요합니다.
테스트는 임시 DB·데이터·Git 저장소를 사용하며, 기본 프로젝트의 검수·릴리스는 변경하지 않습니다.
`requirements/check.py`는 전체 lock 설치 환경에서 실행합니다.

```shell
VLOOP_TEST_FIFTYONE=1 VLOOP_TEST_RELEASE=1 VLOOP_TEST_MLFLOW=1 \
VLOOP_TEST_TRAIN=1 VLOOP_TEST_ITERATION=1 \
VLOOP_TEST_SAM3_CONFIG="$PWD/project.yaml" \
NO_ALBUMENTATIONS_UPDATE=1 python -m pytest -q
```

서로 다른 프로젝트의 FiftyOne 화면이나 Python 세션은 먼저 종료합니다. 테스트 중에도 같은
사용자로 다른 프로젝트의 FiftyOne을 시작하면 DB 경로 검증에서 실패할 수 있습니다.

실제 FiftyOne DB를 사용하는 검증은 호스트에서 별도로 실행합니다. 임시 이미지·DB를 만들고,
다른 프로세스의 재등록 후 수정 내용 보존, 검수 승인·무효화·재승인, 빈 정답,
동시 수정 충돌과 operator 등록을 확인합니다.

```shell
VLOOP_TEST_FIFTYONE=1 python -m pytest tests/test_fiftyone_integration.py tests/test_review_integration.py tests/test_review_batch_integration.py -q
```

DVC·FiftyOne·MLflow 로컬 서버까지 포함한 통합 검증:

```shell
VLOOP_TEST_FIFTYONE=1 VLOOP_TEST_RELEASE=1 VLOOP_TEST_MLFLOW=1 python -m pytest -q
# 평가 지표의 정답·오답·빈 예측과 아티팩트 화면 복원만 검증
VLOOP_TEST_FIFTYONE=1 python -m pytest tests/test_evaluate_integration.py -q
```

RF-DETR GPU 검증은 별도로 실행합니다. 기본 관리 경로의 공식 Seg Nano 가중치가 필요하며,
임시 Git 저장소·DVC 릴리스·MLflow DB와 생성 이미지 4장을 사용합니다. 첫 에포크 후 중단,
별도 프로세스에서 설정을 보존한 재개, 새 프로세스에서 저장 모델의 추론, val/test 평가,
평가 데이터셋 삭제 후 아티팩트 복원과 FiftyOne HTTP 응답까지 확인합니다.

```shell
VLOOP_TEST_TRAIN=1 python -m pytest tests/test_train_integration.py -q
# 100만 건 COCO 인덱스 로딩만 측정: 실제 픽셀 읽기·학습은 포함하지 않음
python tests/train_loader_scale_runner.py --count 1000000
```

두 버전의 데이터·모델·평가 이력 보존 검증은 별도 GPU 테스트로 실행합니다. 생성 이미지에
학습 라벨 수정과 새 이미지를 반영하고, 고정 val의 비교 ID 일치, 기존 split·COCO ID,
이전 아티팩트·태그 보존, 과거 데이터 복원 후 현재 검수 정답 보존을 확인합니다.
실제 사진과 SAM 3까지 자동으로 연결하는 검증에는 [공개 정답 사용 예제](docs/TOY.md)를 사용합니다.
정답을 가져오지 않고 직접 검수할 때는 [Penn-Fudan 직접 실습](#penn-fudan-직접-실습)을 따릅니다.

```shell
VLOOP_TEST_ITERATION=1 python -m pytest tests/test_iteration_integration.py -q
```

100만 건의 합성 메타데이터를 SQLite 작업 목록으로 만드는 메모리 검증:

```shell
python tests/review_scale_runner.py --count 1000000
```

로컬 측정은 약 4초, 프로세스 최대 RSS 44.7 MiB, 작업 중 RSS 증가 2.5 MiB였습니다.
실제 이미지 100만 장의 추론·MongoDB·마스크 처리 성능을 측정한 결과는 아닙니다.
실제 처리 시간과 디스크 사용량은 이미지·객체 수와 저장 장치에 따라 달라집니다. 기존 ingest와
autolabel의 전체 해시 검사·이미지별 파일 저장 등은 별도의 대규모 검증이 필요합니다.

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
├── review_batch.py # 대량 채택 미리보기·선별·중단 및 재개
├── review_store.py # 자동 채택 기록 SQLite 저장
├── review_audit.py # 중단·재개 가능한 전체 무결성 검사
├── approval.py     # 정답 스냅샷·해시와 승인 일치 검사
├── review_operators.py # FiftyOne 검수 operator
├── review_plugin/  # 프로젝트에 설치할 플러그인 등록 파일
├── release.py      # 승인 데이터 스냅샷과 릴리스 흐름
├── release_data.py # 승인 스냅샷·분할·COCO 스트리밍 생성·검증
├── release_dvc.py  # DVC 저장·태그·격리 복원
├── train.py        # 릴리스 복원, 고정 학습 설정과 재개
├── training_engine.py # RF-DETR Lightning 학습·최적 모델·전체 체크포인트
├── tracking.py     # MLflow 실행·아티팩트·localhost 화면
├── trained_model.py # 저장된 추론 모델·클래스·후처리 복원
├── evaluate.py     # 평가 실행·MLflow 기록·아티팩트 검증과 화면 복원
├── evaluation_data.py # 고정 평가 입력·예측 SQLite, 좌표 검증·비교 해시
├── evaluation_view.py # FiftyOne 박스/마스크 COCO 평가와 오류 뷰
└── runtime.py      # 해시, 실행 기록, 잠금
tests/
docs/
```
