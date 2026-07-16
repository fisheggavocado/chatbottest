# BAAI/bge-reranker-v2-m3(cross-encoder)로 검색 후보를 재정렬하는 모듈
# bi-encoder(BGE-M3)+BM25는 질의와 청크를 각각 따로 인코딩해 빠르지만 정밀도가 낮다.
# cross-encoder는 질의-청크 쌍을 함께 인코딩해 느리지만 훨씬 정확하므로,
# 후보군(CANDIDATE_TOP_K)에 한해서만 적용해 정밀도와 속도를 절충한다.

from typing import List

from FlagEmbedding import FlagReranker
from llama_index.core.schema import NodeWithScore

from config import DEVICE, RERANK_MODEL

_reranker = FlagReranker(RERANK_MODEL, use_fp16=True, device=DEVICE)


def rerank(query: str, candidates: List[NodeWithScore], top_k: int) -> List[NodeWithScore]:
    """bi-encoder/BM25로 뽑은 후보를 cross-encoder 점수로 재정렬해 top_k개만 반환한다."""
    if not candidates:
        return []

    pairs = [[query, candidate.node.get_content()] for candidate in candidates]
    scores = _reranker.compute_score(pairs, normalize=True)
    if not isinstance(scores, list):
        scores = [scores]

    # 기존 bi-encoder/RRF 점수를 cross-encoder 점수로 덮어써서 최종 순위 기준으로 삼는다.
    for candidate, score in zip(candidates, scores):
        candidate.score = float(score)

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_k]
