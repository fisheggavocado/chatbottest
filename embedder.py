# BGE-M3(로컬 오픈소스 모델)로 텍스트를 임베딩하는 모듈
# 청킹은 main.py에서 LlamaIndex의 SentenceSplitter가 담당하며, encode_tokens()는 그 청킹에 쓸 토큰 수 계산 함수다.

from typing import List

from FlagEmbedding import BGEM3FlagModel

from config import EMBEDDING_MODEL, DEVICE

# 모델과 토크나이저는 모듈 로드 시 한 번만 초기화해 재사용한다 (매 호출마다 로드하면 매우 느림).
_model = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=True, device=DEVICE)
_tokenizer = _model.tokenizer


def encode_tokens(text: str) -> List[int]:
    """BGE-M3 토크나이저로 텍스트를 토큰 id 리스트로 변환한다."""
    return _tokenizer.encode(text, add_special_tokens=False)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """여러 텍스트를 한 번에 임베딩한다 (dense vector만 사용).

    BGE-M3는 dense/sparse/colbert 세 종류 벡터를 낼 수 있으나, 여기서는 dense만 사용한다.
    """
    output = _model.encode(
        texts, return_dense=True, return_sparse=False, return_colbert_vecs=False
    )
    return output["dense_vecs"].tolist()
