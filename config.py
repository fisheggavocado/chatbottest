# 프로젝트 전역 설정 파일
# PDF 경로, 모델명, 청킹 파라미터 등을 한 곳에서 관리한다.

import os

from dotenv import load_dotenv

# .env 파일에 저장된 환경변수(OPENAI_API_KEY, HF_TOKEN 등)를 불러온다.
load_dotenv()

# gpt-5-mini 비전 API 호출에 사용할 OpenAI API 키
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 커스텀 엔드포인트(프록시, 게이트웨이 등)를 통해 API 토큰을 쓰는 경우의 base URL.
# 표준 OpenAI 엔드포인트를 쓰면 비워두면 된다 (None이면 openai 패키지 기본값 사용).
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# Hugging Face Hub 접근 토큰 (Dataset repo 업로드/다운로드에 사용, private repo는 필수)
HF_TOKEN = os.getenv("HF_TOKEN")

# TODO: 추후 실제 PDF 폴더 경로로 교체
PDF_DIR = r"D:\누리\이어드림 AI 교육\수업 자료\5강"  # 예: r"D:\docs\pdfs"  (임베딩할 PDF들이 들어있는 폴더)
OUTPUT_DIR = r"D:\누리\이어드림 AI 교육\project\first_project\output"  # 임베딩 결과(인덱스)를 저장할 폴더

# Hugging Face Dataset repo id (예: "your-username/pdf-dataset") — .env의 HF_REPO_ID에 한 번만 설정
HF_REPO_ID = os.getenv("HF_REPO_ID")
HF_LOCAL_SYNC_DIR = r"./hf_pdfs"  # HF Dataset repo에서 내려받은 PDF를 저장할 로컬 캐시 폴더

# 모델명은 비밀값은 아니라 코드에 둬도 되지만, .env에서 값을 주면 코드 수정 없이 바꿀 수 있다 (안 쓰면 기본값 사용).
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-5-mini")        # PDF 페이지 시각적 해석 (OpenAI API)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")  # 로컬 오픈소스 임베딩 모델

CHUNK_SIZE = 700       # 임베딩 청크 최대 토큰 수
CHUNK_OVERLAP = 70     # 청크 간 중복 토큰 수 (문맥 단절 방지용)
PDF_IMAGE_DPI = 200    # PDF -> 이미지 변환 해상도 (높을수록 선명하지만 처리 비용 증가)

# 이 길이(글자 수) 미만으로 텍스트 레이어가 추출되면 스캔 이미지 페이지로 간주해 비전 API로 처리한다.
MIN_EXTRACTABLE_TEXT_LENGTH = 30

DEVICE = "cuda"  # BGE-M3 실행 장치, GPU가 없으면 "cpu"로 변경

SECONDS_PER_PAGE_ESTIMATE = 6  # 실제 처리 전 대략적인 예상 시간 계산에 쓰는 가정치(초/페이지)
PERSIST_EVERY_N_PAGES = 1      # 몇 페이지마다 인덱스를 디스크에 저장할지 (1 = 매 페이지 즉시 저장)
