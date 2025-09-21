# 1. 베이스 이미지 선택
# 'slim' 버전은 이미지 크기가 더 작아 효율적입니다.
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
# 컨테이너 내에서 명령어가 실행될 기본 경로를 지정합니다.
WORKDIR /app

# 3. 프로젝트 파일 복사
# 현재 폴더(호스트)의 모든 파일을 컨테이너의 /app 폴더로 복사합니다.
COPY . .

# 4. 의존성 및 프로젝트 설치
# conda의 'pip install -e .'와 동일한 역할을 수행합니다.
# --no-cache-dir 옵션은 불필요한 캐시를 남기지 않아 이미지 크기를 줄여줍니다.
RUN pip install --no-cache-dir -e .
