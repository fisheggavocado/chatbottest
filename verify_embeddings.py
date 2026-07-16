# 임베딩 결과 검증/검색용 스크립트 (LlamaIndex 기반)
# 저장된 LlamaIndex 인덱스를 불러와 벡터 검색 + BM25 검색을 QueryFusionRetriever(RRF)로 융합한다.
# 사용법:
#   python verify_embeddings.py "질문 내용" --top_k 5                # 기본값: 하이브리드
#   python verify_embeddings.py "질문 내용" --mode vector             # 벡터(BGE-M3)만
#   python verify_embeddings.py "질문 내용" --mode bm25               # BM25(키워드)만

import argparse
from pathlib import Path

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.retrievers import QueryFusionRetriever

from config import OUTPUT_DIR
from llama_bm25_retriever import KiwiBM25Retriever
from llama_embedding import BGEM3Embedding

# 인덱스를 저장할 때와 동일한 임베딩 모델을 지정해야 질의 임베딩 차원이 맞는다.
Settings.embed_model = BGEM3Embedding()


def load_index():
    storage_context = StorageContext.from_defaults(persist_dir=str(OUTPUT_DIR))
    return load_index_from_storage(storage_context)


def build_retriever(index, mode: str, top_k: int):
    """mode에 따라 벡터 / BM25 / 하이브리드(RRF) 리트리버를 구성한다."""
    all_nodes = list(index.docstore.docs.values())

    if mode == "vector":
        return index.as_retriever(similarity_top_k=top_k)

    if mode == "bm25":
        return KiwiBM25Retriever(nodes=all_nodes, similarity_top_k=top_k)

    # hybrid: 벡터 리트리버 + BM25 리트리버를 QueryFusionRetriever(RRF)로 융합한다.
    vector_retriever = index.as_retriever(similarity_top_k=top_k)
    bm25_retriever = KiwiBM25Retriever(nodes=all_nodes, similarity_top_k=top_k)
    return QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=top_k,
        num_queries=1,  # 1로 설정하면 LLM으로 질의를 재생성하지 않고 원 질의만 사용한다.
        mode="reciprocal_rerank",  # RRF 융합
        use_async=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description="LlamaIndex 기반 BM25(키워드) + 벡터(의미) RRF 하이브리드 검색"
    )
    parser.add_argument("query", type=str, help="검색할 질문/질의문")
    parser.add_argument("--top_k", type=int, default=5, help="반환할 상위 결과 개수 (기본값: 5)")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "vector", "bm25"],
        default="hybrid",
        help="검색 방식 선택: hybrid(기본값) / vector(BGE-M3만) / bm25(키워드만)",
    )
    args = parser.parse_args()

    if not Path(OUTPUT_DIR).exists():
        print(f"'{OUTPUT_DIR}'에서 인덱스를 찾지 못했습니다. main.py를 먼저 실행하세요.")
        return

    index = load_index()
    retriever = build_retriever(index, args.mode, args.top_k)
    results = retriever.retrieve(args.query)

    print(f'\n질의: "{args.query}" (mode={args.mode})\n')
    for rank, node_with_score in enumerate(results, start=1):
        node = node_with_score.node
        preview = node.get_content()[:150].replace("\n", " ")
        source = node.metadata.get("source", "unknown")
        page = node.metadata.get("page", "?")
        print(f"[{rank}] 점수: {node_with_score.score:.4f} | 출처: {source} (page {page})")
        print(f"    미리보기: {preview}...\n")


if __name__ == "__main__":
    main()
