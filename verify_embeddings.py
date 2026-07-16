# 임베딩 결과 검증/검색용 스크립트 (LlamaIndex 기반)
# 1) 벡터(BGE-M3, bi-encoder) + BM25 검색을 QueryFusionRetriever(RRF)로 융합해 후보 CANDIDATE_TOP_K개를 뽑고
# 2) bge-reranker-v2-m3(cross-encoder)로 그 후보들을 재정렬해 최종 RERANK_TOP_K개만 반환한다.
#
# 사용법:
#   python verify_embeddings.py "질문 내용"                          # 하이브리드 검색 + 리랭크 (기본값)
#   python verify_embeddings.py "질문 내용" --mode vector             # 벡터(BGE-M3)만 + 리랭크
#   python verify_embeddings.py "질문 내용" --mode bm25               # BM25(키워드)만 + 리랭크
#   python verify_embeddings.py "질문 내용" --no-rerank               # 리랭크 없이 후보 순위 그대로 비교

import argparse
from pathlib import Path

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.retrievers import QueryFusionRetriever

from config import CANDIDATE_TOP_K, OUTPUT_DIR, RERANK_TOP_K
from llama_bm25_retriever import KiwiBM25Retriever
from llama_embedding import BGEM3Embedding

# 인덱스를 저장할 때와 동일한 임베딩 모델을 지정해야 질의 임베딩 차원이 맞는다.
Settings.embed_model = BGEM3Embedding()


def load_index():
    storage_context = StorageContext.from_defaults(persist_dir=str(OUTPUT_DIR))
    return load_index_from_storage(storage_context)


def build_retriever(index, mode: str, candidates_k: int):
    """mode에 따라 벡터 / BM25 / 하이브리드(RRF) 리트리버를 구성한다. candidates_k는 리랭킹 전 후보 개수."""
    all_nodes = list(index.docstore.docs.values())

    if mode == "vector":
        return index.as_retriever(similarity_top_k=candidates_k)

    if mode == "bm25":
        return KiwiBM25Retriever(nodes=all_nodes, similarity_top_k=candidates_k)

    # hybrid: 벡터 리트리버(bi-encoder) + BM25 리트리버를 QueryFusionRetriever(RRF)로 융합한다.
    vector_retriever = index.as_retriever(similarity_top_k=candidates_k)
    bm25_retriever = KiwiBM25Retriever(nodes=all_nodes, similarity_top_k=candidates_k)
    return QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=candidates_k,
        num_queries=1,  # 1로 설정하면 LLM으로 질의를 재생성하지 않고 원 질의만 사용한다.
        mode="reciprocal_rerank",  # RRF 융합
        use_async=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description="LlamaIndex 기반 BM25+벡터(RRF) 후보 검색 -> bge-reranker-v2-m3 재정렬"
    )
    parser.add_argument("query", type=str, help="검색할 질문/질의문")
    parser.add_argument(
        "--candidates", type=int, default=CANDIDATE_TOP_K, help=f"리랭킹 전 후보 개수 (기본값: {CANDIDATE_TOP_K})"
    )
    parser.add_argument(
        "--top_k", type=int, default=RERANK_TOP_K, help=f"최종 반환 개수 (기본값: {RERANK_TOP_K})"
    )
    parser.add_argument(
        "--mode",
        choices=["hybrid", "vector", "bm25"],
        default="hybrid",
        help="후보 검색 방식 선택: hybrid(기본값) / vector(BGE-M3만) / bm25(키워드만)",
    )
    parser.add_argument(
        "--no-rerank", action="store_true", help="크로스 인코더 리랭킹 없이 후보 순위 그대로 출력 (비교용)"
    )
    args = parser.parse_args()

    if not Path(OUTPUT_DIR).exists():
        print(f"'{OUTPUT_DIR}'에서 인덱스를 찾지 못했습니다. main.py를 먼저 실행하세요.")
        return

    index = load_index()
    retriever = build_retriever(index, args.mode, args.candidates)
    candidates = retriever.retrieve(args.query)

    if args.no_rerank:
        results = candidates[: args.top_k]
        stage_label = f"mode={args.mode}, rerank 없음"
    else:
        from reranker import rerank

        results = rerank(args.query, candidates, args.top_k)
        stage_label = f"mode={args.mode}, candidates={len(candidates)} -> rerank(bge-reranker-v2-m3) top {args.top_k}"

    print(f'\n질의: "{args.query}" ({stage_label})\n')
    for rank, node_with_score in enumerate(results, start=1):
        node = node_with_score.node
        preview = node.get_content()[:150].replace("\n", " ")
        source = node.metadata.get("source", "unknown")
        page = node.metadata.get("page", "?")
        print(f"[{rank}] 점수: {node_with_score.score:.4f} | 출처: {source} (page {page})")
        print(f"    미리보기: {preview}...\n")


if __name__ == "__main__":
    main()
