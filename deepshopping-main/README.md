
---

# UV 사용 가이드

본 프로젝트는 Python 패키지 관리 및 가상 환경 관리를 위해 UV를 사용합니다.

## 설치 방법

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

또는,
```python
pip install uv
```

## 기본 사용법

### 가상 환경 동기화

```bash
# 가상 환경 생성 및 패키지 동기화
uv sync

# 가상 환경 활성화 방법 1: activate 사용
# macOS / Linux
source .venv/bin/activate

# 가상 환경 활성화 방법 2: uv run 사용
# 가상 환경을 활성화하지 않고 직접 명령어 실행
uv run python your_script.py
uv run pytest
```

### .env 환경설정
```yaml
GOOGLE_API_KEY=
GOOGLE_MODEL_NAME=
SERPER_API_KEY=
```