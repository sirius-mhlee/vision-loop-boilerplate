# 의존성 고정

이 고정 파일은 Linux x86_64, CPython 3.12, CUDA 12.8용 전체 실행·개발 환경입니다.
기존에 검증한 패키지를 유지하며, `.[dev,autolabel,pipeline]`와 SAM 3 이미지 추론에 필요한
전이 의존성까지 포함합니다. Windows·macOS·ARM·다른 Python/CUDA 조합용 파일은 아닙니다.
Python 인터프리터, OS 라이브러리, NVIDIA 드라이버, 모델 체크포인트는 pip 파일로 설치하지 않습니다.

| 파일 | 역할 |
|---|---|
| `bootstrap.lock` | pip 24.0, setuptools 78.1.0, wheel 0.45.1 |
| `linux-py312-cu128.lock` | 외부 패키지 285개의 정확한 버전과 PyPI·PyTorch CUDA 인덱스 |
| `check.py` | 설치 버전, 프로젝트 의존성 및 중첩 extra의 의존성 누락·충돌 검사 |

두 lock에 공통으로 있는 setuptools는 동일 버전입니다. `vloop`와 `sam3`는 로컬 소스에서
설치하므로 lock에 절대 경로나 editable 항목을 넣지 않습니다. SAM 3는 소스 커밋
`660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`을 사용하고, `vloop doctor`가 YAML과 실제 커밋을
대조합니다. 다른 컴퓨터의 프로젝트 코드는 같은 Git 커밋을 사용합니다.

## 설치

프로젝트 루트에서 새 가상환경을 만들고 다음 순서로 실행합니다.

```shell
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/bootstrap.lock
python -m pip install --no-build-isolation --no-deps -r requirements/linux-py312-cu128.lock
python -m pip install --no-build-isolation --no-deps -e '.[dev,autolabel,pipeline]'

git clone https://github.com/facebookresearch/sam3.git .vloop/vendor/sam3
git -C .vloop/vendor/sam3 checkout 660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7
python -m pip install --no-build-isolation --no-deps -e .vloop/vendor/sam3

python -m pip check
python requirements/check.py
```

이미 SAM 3 소스가 있다면 해당 커밋과 변경 상태를 확인하고 clone 단계는 생략합니다.
체크포인트 준비와 설정은 [README](../README.md#sam-3-한-장-추론)를 따릅니다.

`--no-deps`는 lock에 없는 패키지를 자동으로 추가하지 않습니다. `--no-build-isolation`은
소스 빌드 때 bootstrap에서 설치한 도구를 사용합니다. 필요한 의존성을 먼저 전부 설치하는 순서이므로
두 검사가 끝나기 전에는 준비된 환경으로 취급하지 않습니다. `pip check`에 더해 `check.py`는
`rfdetr[train]` → `torchmetrics[detection]`처럼 extra로 활성화되는 의존성도 검사합니다.

일부 기능만 필요한 별도 환경에서는 bootstrap 설치 후 다음처럼 `-c`를 사용할 수 있습니다.

```shell
python -m pip install --no-build-isolation -c requirements/linux-py312-cu128.lock \
  -e '.[dev,autolabel]' -e .vloop/vendor/sam3
```

`-c`는 필요한 패키지의 버전만 제한합니다. 전체 lock을 설치하는 `-r`과 달리 모든 패키지를
설치하지 않습니다. `check.py`는 전체 환경용이므로 이런 부분 설치에는 사용하지 않습니다.
새 의존성을 추가하면 lock 갱신도 필요합니다. 고정 파일만으로 미등록 의존성 추가를 금지하는
부분 설치 방식은 아니므로 재현성 검증은 위 전체 설치 절차를 기준으로 합니다.

## 갱신

lock은 현재 환경을 자동으로 업그레이드하는 파일이 아닙니다. 버전을 바꿀 때는 별도 검증
가상환경에서 의존성을 조정하고 `pip check`, 회귀 검사, 해당 GPU·로컬 서비스 검증을 실행합니다.
검증 환경의 외부 패키지 목록은 다음 명령으로 추출할 수 있습니다.

```shell
mkdir -p .vloop
python -m pip freeze --exclude-editable --exclude pip --exclude wheel > .vloop/dependencies.new.txt
```

새 파일에서 개인 경로·추가 도구·사용하지 않는 패키지가 없는지 확인한 뒤, 기존 runtime lock의
헤더와 두 인덱스 옵션을 유지하고 버전 목록을 교체합니다. pip와 wheel 변경은 `bootstrap.lock`에,
setuptools 변경은 두 파일에 같은 버전으로 반영합니다. `pyproject.toml`에 명시한 범위도 함께
맞춥니다. `python requirements/check.py`로 확인하고 빈 가상환경에서 다시 설치해 검증합니다.
실행별 `dependencies.json` 기록은 남겨 이전 모델의 실제 환경과 비교할 수 있습니다.

이 파일은 패키지 **버전 고정**을 제공합니다. 배포 파일 전체의 해시 고정이나 오프라인 설치
번들은 포함하지 않습니다. 다운로드 출처의 가용성과 OS·드라이버까지 동일하게 만드는 기능은
별도입니다. 이 범위는 [pip의 반복 가능한 설치 안내](https://pip.pypa.io/en/stable/topics/repeatable-installs/)의
버전 고정 방식에 해당합니다.
