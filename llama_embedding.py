# embedder.py의 BGE-M3 임베딩을 LlamaIndex의 임베딩 인터페이스(BaseEmbedding)에 연결하는 어댑터
# LlamaIndex의 기본 임베딩(OpenAI)이 아닌, 이미 구축된 로컬 BGE-M3 모델을 그대로 쓰기 위함이다.

from typing import List

from llama_index.core.embeddings import BaseEmbedding

from embedder import embed_texts


class BGEM3Embedding(BaseEmbedding):
    """embedder.py의 BGE-M3 dense 임베딩을 LlamaIndex 인덱스/리트리버에 연결하는 래퍼."""

    def _get_query_embedding(self, query: str) -> List[float]:
        return embed_texts([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return embed_texts([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        # 여러 청크를 한 번에 임베딩해 반복 호출보다 빠르게 처리한다.
        return embed_texts(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
