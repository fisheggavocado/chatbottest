# chatbot — 이론 PDF RAG 파이프라인

챗봇 모델을 설계할 때 이론을 전부 암기하지 않고, 색인된 PDF에서 검색해 근거로 참조하기 위한 RAG 인덱싱 파이프라인.
"AI 챗봇 구축 방법론" PDF(약 3,000장, 30개 파일 예상)를 대상으로 하며, 챗봇 자체는 파인튜닝 없이 **LLM API 호출 + RAG** 방식으로 만든다.

## 파이프라인

| 단계 | 내용 |
|---|---|
| 1. 원본 PDF 확보 | 로컬 폴더(`PDF_DIR`) 또는 Hugging Face Dataset(`HF_REPO_ID`)에서 동기화 |
| 2. 텍스트 추출 | PyMuPDF로 텍스트 레이어 우선 추출, 부족한 페이지(스캔 이미지)만 `gpt-5-mini` 비전 API로 폴백 |
| 3. 청킹 | `SentenceSplitter` (`CHUNK_SIZE=700`, `CHUNK_OVERLAP=70`) |
| 4. 임베딩 | `BAAI/bge-m3` (한국어/다국어 지원, 로컬 실행) |
| 5. 후보 검색 | BM25(kiwipiepy) + 벡터(BGE-M3, bi-encoder) 검색을 RRF(Reciprocal Rank Fusion)로 융합해 상위 `CANDIDATE_TOP_K`(기본 30)개 추출 |
| 6. 재정렬 | `BAAI/bge-reranker-v2-m3`(cross-encoder)로 후보를 재정렬해 최종 `RERANK_TOP_K`(기본 10)개만 반환 |
| 7. 저장·재개 | 페이지 단위 스트리밍 저장 + `checkpoint.json` 기반 중단/재개, 완료 시 HF Dataset에 자동 백업 |
| 8. 챗봇 응답 생성 | (예정) 검색 결과를 프롬프트에 넣어 LLM API 호출 |

## 설치

```bash
pip install -r requirements.txt
```

`.env.example`을 `.env`로 복사한 뒤 값을 채운다.

```bash
cp .env.example .env
```

| 변수 | 설명 |
|---|---|
| `OPENAI_API_KEY` | `gpt-5-mini` 비전 API 호출용 |
| `OPENAI_BASE_URL` | 프록시/게이트웨이 사용 시에만 설정 (표준 OpenAI 엔드포인트면 비움) |
| `HF_TOKEN` | Hugging Face Hub 토큰 (private dataset repo 사용 시 필수) |
| `HF_REPO_ID` | PDF/임베딩 결과를 보관할 HF Dataset repo id |

PDF 경로, 청킹 파라미터 등 나머지 설정은 [config.py](config.py)에서 관리한다.

## 사용법

### 1. 인덱스 생성 (`main.py`)

```bash
python main.py                 # PDF_DIR 전체 처리
python main.py --limit 1       # 시험 삼아 PDF 1개만 처리
python main.py --use-hf        # PDF_DIR 대신 HF_REPO_ID에서 PDF를 동기화해 처리
```

중간에 중단돼도 `checkpoint.json`을 보고 이어서 처리한다 (완료된 PDF/페이지는 건너뜀).

### 2. 검색 검증 (`verify_embeddings.py`)

```bash
python verify_embeddings.py "질문 내용"                          # 하이브리드(BM25+벡터 RRF) top 30 -> 리랭크 top 10
python verify_embeddings.py "질문 내용" --mode vector             # 벡터(BGE-M3)만 + 리랭크
python verify_embeddings.py "질문 내용" --mode bm25               # BM25(키워드)만 + 리랭크
python verify_embeddings.py "질문 내용" --candidates 20 --top_k 5 # 후보/최종 개수 직접 조절
python verify_embeddings.py "질문 내용" --no-rerank               # 리랭크 전/후 비교용
```

## 구성 파일

- `config.py` — 경로/모델명/청킹 파라미터 등 전역 설정
- `main.py` — 인덱싱 실행 진입점
- `llama_pdf_reader.py` — 하이브리드 OCR(PyMuPDF + 비전 API) PDF 리더
- `llama_embedding.py` — BGE-M3 임베딩 래퍼
- `llama_bm25_retriever.py` — 한국어 형태소 분석 기반 BM25 리트리버
- `checkpoint.py` — 인덱싱 진행 상황 저장/재개
- `hf_storage.py` — Hugging Face Dataset 동기화/백업
- `reranker.py` — `bge-reranker-v2-m3` 기반 cross-encoder 재정렬
- `verify_embeddings.py` — 저장된 인덱스 검색 검증 스크립트 (후보 검색 + 리랭크)

## 현재 상태

테스트 단계 — 그리드서치로 청킹/검색 파라미터는 확정했으나, 실제 코퍼스(30개 PDF) 대상 본 실행은 아직 진행 전. 자세한 진행 상황은 `chat_log/project_status.md` 참고 (로컬 전용, `.gitignore`로 제외됨).
