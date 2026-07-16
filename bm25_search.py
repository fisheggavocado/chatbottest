# BM25(키워드 기반) 검색을 위한 인덱싱/토크나이징 모듈
# 한국어는 띄어쓰기만으로 단어를 나누면 BM25 품질이 떨어지므로, 형태소 분석기(kiwipiepy)로 토큰화한다.

from typing import List

from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

_kiwi = Kiwi()


def tokenize(text: str) -> List[str]:
    """한국어 형태소 분석 기반 토크나이저 (BM25 인덱싱/질의에 동일하게 사용)."""
    return [token.form for token in _kiwi.tokenize(text)]


def build_bm25_index(records: List[dict]) -> BM25Okapi:
    """저장된 청크 텍스트 전체로 BM25 인덱스를 생성한다."""
    tokenized_corpus = [tokenize(record["text"]) for record in records]
    return BM25Okapi(tokenized_corpus)
