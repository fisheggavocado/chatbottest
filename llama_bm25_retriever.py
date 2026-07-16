# BM25(키워드) 검색을 LlamaIndex 리트리버 인터페이스(BaseRetriever)에 연결하는 어댑터
# 한국어 형태소 토큰화(kiwipiepy) + BM25 인덱싱은 bm25_search.py의 것을 그대로 재사용한다.

from typing import List

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle

from bm25_search import build_bm25_index, tokenize


class KiwiBM25Retriever(BaseRetriever):
    """노드 텍스트로 BM25 인덱스를 만들고, 질의에 대해 점수순으로 노드를 반환한다."""

    def __init__(self, nodes: List[BaseNode], similarity_top_k: int = 10):
        self._nodes = nodes
        self._similarity_top_k = similarity_top_k
        # bm25_search.build_bm25_index()는 {"text": ...} 형태의 레코드 리스트를 받는다.
        records = [{"text": node.get_content()} for node in nodes]
        self._bm25_index = build_bm25_index(records)
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        scores = self._bm25_index.get_scores(tokenize(query_bundle.query_str))
        scored = sorted(zip(scores, self._nodes), key=lambda x: x[0], reverse=True)
        return [
            NodeWithScore(node=node, score=float(score))
            for score, node in scored[: self._similarity_top_k]
        ]
